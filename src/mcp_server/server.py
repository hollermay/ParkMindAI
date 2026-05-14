"""
SmartPark MCP Server — FastAPI implementation of the Model Context Protocol (MCP)
over HTTP using JSON-RPC 2.0.

Exposes one tool:
  write_confirmed_reservation
    • Receives approved reservation data from the admin agent.
    • Writes one entry per approval to ``data/confirmed_reservations.txt``.
    • Entry format: Name | Car Number | Reservation Period | Approval Time

Security:
  • Bearer-token / X-API-Key authentication on every request.
  • Pydantic input validation with pipe / newline injection prevention.
  • Sliding-window rate limiter per client IP (30 req / 60 s by default).
  • CORS restricted to localhost origins.
  • Swagger / ReDoc UI disabled to minimise attack surface.

Reliability:
  • Thread-safe file writes protected by a ``threading.Lock``.
  • Atomic append via standard Python open() with explicit flush + fsync.
  • Automatic creation of parent directories.
  • Per-request structured logging for full audit trail.

Endpoints:
  POST /mcp      — JSON-RPC 2.0 MCP messages (auth required)
  GET  /health   — Liveness probe (no auth required)

Server lifecycle:
  Call ``start_mcp_server(host, port)`` to start uvicorn in a daemon thread.
  Safe to call multiple times — only starts once.
"""
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

import src.config as cfg

logger = logging.getLogger(__name__)

# ─── Server lifecycle state ───────────────────────────────────────────────────

_server_started = False
_server_lock = threading.Lock()

# ─── File-write lock (shared across all request-handler threads) ──────────────

_file_lock = threading.Lock()

# ─── In-memory rate limiter (client_ip → (count, window_start)) ──────────────

_rate_limits: Dict[str, tuple] = {}
_rate_lock = threading.Lock()

RATE_LIMIT_MAX = 30       # maximum requests allowed per window
RATE_LIMIT_WINDOW = 60    # seconds per window


# ─── FastAPI application ──────────────────────────────────────────────────────

app = FastAPI(
    title="SmartPark MCP Server",
    description=(
        "Model Context Protocol server that processes confirmed parking "
        "reservations and persists them to a text log."
    ),
    version="1.0.0",
    # Disable API docs to reduce attack surface
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        f"http://localhost:{cfg.ADMIN_API_PORT}",
        f"http://127.0.0.1:{cfg.ADMIN_API_PORT}",
        f"http://localhost:{cfg.MCP_SERVER_PORT}",
        f"http://127.0.0.1:{cfg.MCP_SERVER_PORT}",
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ─── Security helpers ─────────────────────────────────────────────────────────

def _verify_api_key(
    authorization: Optional[str],
    x_api_key: Optional[str],
) -> None:
    """
    Verify that the request carries a valid API key.
    Accepts either:
      Authorization: Bearer <key>
      X-API-Key: <key>
    Raises HTTP 401 if the key is missing or incorrect.
    """
    token: Optional[str] = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    if token is None and x_api_key:
        token = x_api_key

    # Constant-time comparison to resist timing attacks
    import hmac
    expected = cfg.MCP_API_KEY.encode()
    provided = (token or "").encode()
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API key.")


def _check_rate_limit(client_ip: str) -> None:
    """
    Sliding-window rate limiter.
    Raises HTTP 429 if the client has exceeded RATE_LIMIT_MAX requests
    within the current RATE_LIMIT_WINDOW.
    """
    now = time.monotonic()
    with _rate_lock:
        count, window_start = _rate_limits.get(client_ip, (0, now))
        if now - window_start > RATE_LIMIT_WINDOW:
            # Start a fresh window
            _rate_limits[client_ip] = (1, now)
        else:
            if count >= RATE_LIMIT_MAX:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit exceeded: max {RATE_LIMIT_MAX} requests "
                        f"per {RATE_LIMIT_WINDOW}s."
                    ),
                )
            _rate_limits[client_ip] = (count + 1, window_start)


# ─── MCP Protocol helpers ─────────────────────────────────────────────────────

def _jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


# ─── Tool schema ──────────────────────────────────────────────────────────────

_TOOLS: List[dict] = [
    {
        "name": "write_confirmed_reservation",
        "description": (
            "Write an approved parking reservation to the persistent reservations log. "
            "Called after administrator approval to record the confirmed booking. "
            "Entry format: Name | Car Number | Reservation Period | Approval Time."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["full_name", "car_number", "reservation_period", "approval_time"],
            "properties": {
                "full_name": {
                    "type": "string",
                    "description": "Full name of the guest (e.g. 'John Doe').",
                },
                "car_number": {
                    "type": "string",
                    "description": "Vehicle registration plate number (e.g. 'ABC-1234').",
                },
                "reservation_period": {
                    "type": "string",
                    "description": "Human-readable period (e.g. '2026-06-01 to 2026-06-07').",
                },
                "approval_time": {
                    "type": "string",
                    "description": "ISO timestamp of admin approval (YYYY-MM-DD HH:MM:SS).",
                },
            },
        },
    }
]


# ─── Input model ──────────────────────────────────────────────────────────────

class WriteReservationInput(BaseModel):
    """Validated input for write_confirmed_reservation."""

    full_name: str
    car_number: str
    reservation_period: str
    approval_time: str

    @field_validator("full_name", "car_number", "reservation_period", "approval_time")
    @classmethod
    def sanitise(cls, v: str) -> str:
        """
        Prevent log/format injection:
          - strip pipe characters (would break column format)
          - strip newline / carriage-return (would break line-per-entry format)
          - collapse leading/trailing whitespace
        """
        return v.replace("|", "-").replace("\n", " ").replace("\r", "").strip()


# ─── File writer ──────────────────────────────────────────────────────────────

def _write_reservation_to_file(data: WriteReservationInput) -> str:
    """
    Append one reservation line to the confirmed reservations log.

    The file is created (with header) if it does not yet exist.
    All writes are serialised through ``_file_lock`` to prevent interleaving
    in multi-threaded scenarios, and ``flush + fsync`` ensures durability.
    """
    output_path = Path(cfg.RESERVATIONS_FILE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entry = (
        f"{data.full_name} | {data.car_number} | "
        f"{data.reservation_period} | {data.approval_time}\n"
    )

    with _file_lock:
        write_header = not output_path.exists() or output_path.stat().st_size == 0
        with open(output_path, "a", encoding="utf-8") as fh:
            if write_header:
                fh.write("Name | Car Number | Reservation Period | Approval Time\n")
                fh.write("-" * 70 + "\n")
            fh.write(entry)
            fh.flush()
            import os
            os.fsync(fh.fileno())

    logger.info(
        "[MCPServer] Reservation logged: %s | %s | %s | %s",
        data.full_name,
        data.car_number,
        data.reservation_period,
        data.approval_time,
    )
    return (
        f"Reservation confirmed for {data.full_name} ({data.car_number}) — "
        f"written to {output_path.name}."
    )


# ─── MCP request dispatcher ───────────────────────────────────────────────────

def _handle_mcp_message(body: dict) -> Optional[dict]:
    """
    Dispatch a single JSON-RPC 2.0 MCP message and return the response dict
    (or None for notification messages that require no response).
    """
    if body.get("jsonrpc") != "2.0":
        return _jsonrpc_error(None, -32600, "Invalid Request: jsonrpc must be '2.0'.")

    req_id = body.get("id")       # None for notifications
    method = body.get("method", "")
    params = body.get("params") or {}

    # ── initialize ────────────────────────────────────────────────────────────
    if method == "initialize":
        return _jsonrpc_result(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "SmartPark MCP Server", "version": "1.0.0"},
        })

    # ── notifications/initialized — no response required ─────────────────────
    if method == "notifications/initialized":
        return None

    # ── tools/list ────────────────────────────────────────────────────────────
    if method == "tools/list":
        return _jsonrpc_result(req_id, {"tools": _TOOLS})

    # ── tools/call ────────────────────────────────────────────────────────────
    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}

        if tool_name == "write_confirmed_reservation":
            try:
                data = WriteReservationInput(**arguments)
                message = _write_reservation_to_file(data)
                return _jsonrpc_result(req_id, {
                    "content": [{"type": "text", "text": message}],
                    "isError": False,
                })
            except Exception as exc:
                logger.error("[MCPServer] write_confirmed_reservation failed: %s", exc)
                return _jsonrpc_result(req_id, {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                })

        return _jsonrpc_error(req_id, -32601, f"Unknown tool: '{tool_name}'.")

    # ── Unknown method ────────────────────────────────────────────────────────
    return _jsonrpc_error(req_id, -32601, f"Method not found: '{method}'.")


# ─── FastAPI route handlers ───────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness probe — no authentication required."""
    return {
        "status": "ok",
        "server": "SmartPark MCP Server",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/mcp")
async def mcp_endpoint(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    """
    JSON-RPC 2.0 MCP endpoint.

    Accepts:
      • A single request object  → returns a single response object.
      • A batch array of objects → returns an array of response objects
        (notifications with no response are silently dropped from the batch).

    Authentication: Bearer token in ``Authorization`` header
                    or plain key in ``X-API-Key`` header.
    """
    client_ip = (request.client.host if request.client else "unknown")

    # ── Security checks ───────────────────────────────────────────────────────
    _verify_api_key(authorization, x_api_key)
    _check_rate_limit(client_ip)

    # ── Parse body ────────────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            # Per JSON-RPC spec, parse errors still return HTTP 200
            content=_jsonrpc_error(None, -32700, "Parse error: request body is not valid JSON."),
            status_code=200,
        )

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if isinstance(body, list):
        responses = [_handle_mcp_message(item) for item in body]
        # Filter out None (notifications that expect no response)
        responses = [r for r in responses if r is not None]
        return JSONResponse(content=responses)

    result = _handle_mcp_message(body)
    if result is None:
        # Notification — respond with 204 No Content
        return JSONResponse(content=None, status_code=204)
    return JSONResponse(content=result)


# ─── Server lifecycle ─────────────────────────────────────────────────────────

def start_mcp_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> bool:
    """
    Start the FastAPI MCP server in a background daemon thread using uvicorn.

    Safe to call multiple times — only starts once.

    Args:
        host: Bind address (default: cfg.MCP_SERVER_HOST).
        port: TCP port    (default: cfg.MCP_SERVER_PORT).

    Returns:
        True on success, False if the server could not be started.
    """
    global _server_started

    bind_host = host or cfg.MCP_SERVER_HOST
    bind_port = port or cfg.MCP_SERVER_PORT

    with _server_lock:
        if _server_started:
            return True

        try:
            import uvicorn

            config = uvicorn.Config(
                app,
                host=bind_host,
                port=bind_port,
                log_level="error",
                # Disable access log to avoid flooding with polling requests
                access_log=False,
            )
            server = uvicorn.Server(config)
            # Disable signal-handler installation so uvicorn can run in a non-main thread
            server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

            t = threading.Thread(
                target=server.run,
                daemon=True,
                name="mcp-server",
            )
            t.start()

            # ── Wait until the socket is accepting connections (max 10 s) ──────
            # Always poll on 127.0.0.1 to avoid Windows IPv6 "localhost" ambiguity.
            import socket as _socket
            poll_host = "127.0.0.1"
            deadline = time.monotonic() + 10.0
            ready = False
            while time.monotonic() < deadline:
                try:
                    with _socket.create_connection((poll_host, bind_port), timeout=0.5):
                        ready = True
                        break
                except OSError:
                    time.sleep(0.1)

            if not ready:
                logger.warning(
                    "[MCPServer] Server did not become ready within 8s on %s:%d — "
                    "direct-write fallback will be used.",
                    bind_host, bind_port,
                )
            else:
                logger.info(
                    "[MCPServer] FastAPI MCP server ready: http://%s:%d/mcp",
                    bind_host,
                    bind_port,
                )

            _server_started = True
            return True

        except Exception as exc:
            logger.error("[MCPServer] Failed to start: %s", exc, exc_info=True)
            return False
