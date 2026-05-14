"""
MCP client + LangChain tool wrapper for the SmartPark MCP server.

Provides two layers of access:
  1. ``MCPClient``                    — low-level JSON-RPC 2.0 HTTP client.
  2. ``call_write_confirmed_reservation`` — high-level helper used by graph nodes.
  3. ``write_confirmed_reservation_tool`` — @tool decorator for LangChain agents.

Fallback strategy:
  If the MCP server is unreachable (e.g. not yet started, port conflict),
  ``call_write_confirmed_reservation`` falls back to a direct file write so
  the reservation is never lost.
"""
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.tools import tool

import src.config as cfg

logger = logging.getLogger(__name__)

# ─── Thread-safe fallback file lock ───────────────────────────────────────────
_fallback_lock = threading.Lock()


# ─── Low-level MCP HTTP client ─────────────────────────────────────────────────

class MCPClient:
    """
    Minimal JSON-RPC 2.0 HTTP client for the SmartPark MCP server.

    Sends requests to ``POST /mcp`` with a Bearer token in the
    ``Authorization`` header.  Raises ``httpx.HTTPError`` on failures.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        # Resolve the base URL, replacing "localhost" with "127.0.0.1" on all
        # platforms to avoid Windows IPv6 ambiguity (localhost → ::1 vs 127.0.0.1).
        host = cfg.MCP_SERVER_HOST.replace("localhost", "127.0.0.1")
        self.base_url = base_url or f"http://{host}:{cfg.MCP_SERVER_PORT}"
        self.api_key = api_key or cfg.MCP_API_KEY
        self.timeout = timeout
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call a tool on the MCP server and return the full JSON-RPC response dict.

        Args:
            tool_name: Name of the MCP tool to invoke.
            arguments: Input arguments for the tool.

        Returns:
            Parsed JSON response dict from the server.

        Raises:
            httpx.HTTPError: On HTTP-level failures.
            ValueError: If the server returns a JSON-RPC error.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            response.raise_for_status()
            return response.json()

    def health_check(self) -> bool:
        """Return True if the MCP server is reachable and healthy."""
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False


# ─── High-level helper ─────────────────────────────────────────────────────────

def call_write_confirmed_reservation(
    full_name: str,
    car_number: str,
    start_date: str,
    end_date: str,
    approval_time: Optional[str] = None,
) -> str:
    """
    Write a confirmed reservation to the log via the MCP server.

    Attempts to reach the MCP server first; if unavailable, falls back to a
    direct, thread-safe file write so the entry is never silently dropped.

    Args:
        full_name:     Full name of the guest (first + last).
        car_number:    Vehicle registration plate.
        start_date:    Reservation start date (YYYY-MM-DD).
        end_date:      Reservation end date   (YYYY-MM-DD).
        approval_time: ISO-formatted timestamp of approval.
                       Defaults to the current UTC time.

    Returns:
        Confirmation message string.
    """
    if not approval_time:
        approval_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    reservation_period = f"{start_date} to {end_date}"

    arguments = {
        "full_name": full_name,
        "car_number": car_number,
        "reservation_period": reservation_period,
        "approval_time": approval_time,
    }

    # ── Primary path: MCP server (up to 3 attempts) ──────────────────────────
    import time as _time
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            client = MCPClient()
            response = client.call_tool("write_confirmed_reservation", arguments)
            result = response.get("result", {})
            content = result.get("content") or [{}]
            message = content[0].get("text", "OK") if content else "OK"
            if result.get("isError"):
                logger.warning("[MCPClient] Tool reported error: %s", message)
            else:
                logger.info("[MCPClient] Reservation written via MCP: %s", message)
            return message
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                _time.sleep(0.5)

    logger.warning(
        "[MCPClient] MCP server unreachable after 3 attempts (%s); using direct write fallback.",
        last_exc,
    )

    # ── Fallback: direct file write ───────────────────────────────────────────
    return _direct_write(
        full_name=full_name,
        car_number=car_number,
        reservation_period=reservation_period,
        approval_time=approval_time,
    )


def _direct_write(
    full_name: str,
    car_number: str,
    reservation_period: str,
    approval_time: str,
) -> str:
    """
    Thread-safe direct file write — used when the MCP server is unavailable.

    Sanitises all fields to prevent pipe / newline injection before writing.
    """
    def _sanitise(value: str) -> str:
        return value.replace("|", "-").replace("\n", " ").replace("\r", "").strip()

    full_name         = _sanitise(full_name)
    car_number        = _sanitise(car_number)
    reservation_period = _sanitise(reservation_period)
    approval_time     = _sanitise(approval_time)

    entry = (
        f"{full_name} | {car_number} | "
        f"{reservation_period} | {approval_time}\n"
    )

    output_path = Path(cfg.RESERVATIONS_FILE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _fallback_lock:
        write_header = not output_path.exists() or output_path.stat().st_size == 0
        with open(output_path, "a", encoding="utf-8") as fh:
            if write_header:
                fh.write("Name | Car Number | Reservation Period | Approval Time\n")
                fh.write("-" * 70 + "\n")
            fh.write(entry)
            fh.flush()
            os.fsync(fh.fileno())

    logger.info(
        "[DirectWriter] Reservation logged: %s | %s", full_name, car_number
    )
    return (
        f"Reservation logged (direct write) for {full_name} ({car_number})."
    )


# ─── LangChain @tool ──────────────────────────────────────────────────────────

@tool
def write_confirmed_reservation_tool(
    full_name: str,
    car_number: str,
    reservation_period: str,
    approval_time: str,
) -> str:
    """
    Write an approved parking reservation to the persistent log via the MCP server.

    Use this tool after the administrator has approved a reservation. It contacts
    the SmartPark MCP server which securely appends the confirmed booking to the
    reservations log file.

    Args:
        full_name:           Full name of the guest (e.g. 'Jane Smith').
        car_number:          Vehicle registration plate (e.g. 'XY-9876').
        reservation_period:  Human-readable period (e.g. '2026-06-01 to 2026-06-07').
        approval_time:       Timestamp of admin approval (YYYY-MM-DD HH:MM:SS UTC).

    Returns:
        Confirmation message from the MCP server (or fallback writer).
    """
    try:
        client = MCPClient()
        response = client.call_tool(
            "write_confirmed_reservation",
            {
                "full_name": full_name,
                "car_number": car_number,
                "reservation_period": reservation_period,
                "approval_time": approval_time,
            },
        )
        result = response.get("result", {})
        content = result.get("content") or [{}]
        return content[0].get("text", "Reservation recorded.") if content else "Reservation recorded."
    except Exception as exc:
        logger.warning("[MCPTool] MCP server unreachable (%s); falling back.", exc)
        return _direct_write(
            full_name=full_name,
            car_number=car_number,
            reservation_period=reservation_period,
            approval_time=approval_time,
        )
