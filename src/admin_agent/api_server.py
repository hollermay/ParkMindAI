"""
FastAPI REST API — Admin Decision Server.

Provides the human administrator with a web interface and JSON API to:
  - View all pending reservation approval requests
  - Approve or reject individual requests
  - Review reservation details

Endpoints:
  GET  /admin                                   HTML dashboard (auto-refreshes)
  GET  /admin/pending                           JSON list of pending requests
  GET  /admin/reservation/{code}                JSON details for one request
  POST /admin/decide           {code, approved, notes}   JSON response
  POST /admin/decide           form fields: code, decision, notes
  GET  /admin/decide?code=X&decision=approve    one-click from email links

Runs as a daemon thread (uvicorn) — shuts down automatically when the main
process exits.  Same lifecycle pattern as the MCP server.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

_server_started = False
_server_lock = threading.Lock()
_server_host = "localhost"
_server_port = 5001

# ─── HTML templates ───────────────────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="10">
  <title>SmartPark Admin Dashboard</title>
  <style>
    *    { box-sizing: border-box; }
    body { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto;
           padding: 0 20px; background: #f7fafc; color: #333; }
    h1   { color: #2c5282; }
    .badge { background: #e53e3e; color: white; border-radius: 50%;
             padding: 2px 8px; font-size: 13px; margin-left: 8px; }
    .card { background: white; border: 1px solid #e2e8f0; border-radius: 10px;
            padding: 24px; margin: 20px 0; box-shadow: 0 2px 8px #0001; }
    table { width: 100%; border-collapse: collapse; }
    td, th { padding: 10px 14px; border: 1px solid #eee; text-align: left; }
    th { background: #ebf8ff; font-weight: bold; }
    .code { font-family: monospace; font-size: 16px; font-weight: bold;
            color: #2b6cb0; }
    .actions { margin-top: 16px; }
    .btn  { padding: 10px 24px; border: none; border-radius: 6px; color: white;
            cursor: pointer; font-size: 14px; font-weight: bold; margin: 4px; }
    .approve { background: #38a169; }
    .approve:hover { background: #276749; }
    .reject  { background: #e53e3e; }
    .reject:hover  { background: #9b2c2c; }
    input[type=text] { padding: 8px 12px; border: 1px solid #ccc;
                       border-radius: 4px; font-size: 13px; width: 300px; }
    form { display: inline; }
    .empty { color: #888; font-style: italic; text-align: center;
             padding: 40px 0; }
    .hint { font-size: 12px; color: #999; margin-top: 8px; }
    .refresh-note { font-size: 12px; color: #aaa; }
    .history th { background: #f0fff4; }
    .status-approved { color: #276749; font-weight: bold; }
    .status-rejected { color: #9b2c2c; font-weight: bold; }
    .status-pending  { color: #744210; font-weight: bold; }
    .status-cancelled { color: #718096; font-weight: bold; }
    h2 { color: #2c5282; margin: 32px 0 8px; font-size: 20px; }
    .source-tag { font-size: 11px; background: #ebf8ff; color: #2b6cb0;
                  border-radius: 4px; padding: 2px 6px; margin-left: 6px; }
    .source-tag.web { background: #f0fff4; color: #276749; }
  </style>
</head>
<body>
  <h1>🅿 SmartPark — Admin Dashboard
    {% set total_pending = (pending|length) + (db_pending|length) %}
    {% if total_pending %}
      <span class="badge">{{ total_pending }}</span>
    {% endif %}
  </h1>
  <p class="refresh-note">Page auto-refreshes every 10 seconds.</p>

  {% if pending %}
  <h2>🔔 Chatbot Reservation Requests</h2>
    {% for code, rd in pending.items() %}
    <div class="card">
      <p class="code">Request Code: {{ code }} <span class="source-tag">chatbot</span></p>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Name</td><td>{{ rd.full_name }}</td></tr>
        <tr><td>Email</td><td>{{ rd.email or '—' }}</td></tr>
        <tr><td>Vehicle Plate</td><td>{{ rd.car_number }}</td></tr>
        <tr><td>Zone</td><td>Zone {{ rd.zone }}</td></tr>
        <tr><td>Start Date</td><td>{{ rd.start_date }}</td></tr>
        <tr><td>End Date</td><td>{{ rd.end_date }}</td></tr>
        <tr><td>Card</td><td>{{ rd.card_masked or (rd.card_number if rd.card_number else '—') }}</td></tr>
      </table>
      <div class="actions">
        <form method="post" action="/admin/decide">
          <input type="hidden" name="code" value="{{ code }}">
          <input type="hidden" name="decision" value="approve">
          <button class="btn approve" type="submit">✅ Approve</button>
        </form>
        <form method="post" action="/admin/decide">
          <input type="hidden" name="code" value="{{ code }}">
          <input type="hidden" name="decision" value="reject">
          <input type="text" name="notes" placeholder="Rejection reason (optional)">
          <button class="btn reject" type="submit">❌ Reject</button>
        </form>
        <p class="hint">
          API: POST /admin/decide  { "code": "{{ code }}", "approved": true }
        </p>
      </div>
    </div>
    {% endfor %}
  {% endif %}

  {% if db_pending %}
  <h2>🌐 Web Form Reservation Requests</h2>
    {% for r in db_pending %}
    <div class="card">
      <p class="code">Reservation Code: {{ r.code }} <span class="source-tag web">web form</span></p>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Name</td><td>{{ r.name }}</td></tr>
        <tr><td>Email</td><td>{{ r.email or '—' }}</td></tr>
        <tr><td>Vehicle Plate</td><td>{{ r.vehicle }}</td></tr>
        <tr><td>Zone</td><td>Zone {{ r.zone }}</td></tr>
        <tr><td>Start Date</td><td>{{ r.start }}</td></tr>
        <tr><td>End Date</td><td>{{ r.end }}</td></tr>
        <tr><td>Card</td><td>{{ r.card or '—' }}</td></tr>
        <tr><td>Submitted</td><td>{{ r.created }}</td></tr>
      </table>
      <div class="actions">
        <form method="post" action="/admin/decide_db">
          <input type="hidden" name="code" value="{{ r.code }}">
          <input type="hidden" name="decision" value="approve">
          <button class="btn approve" type="submit">✅ Approve</button>
        </form>
        <form method="post" action="/admin/decide_db">
          <input type="hidden" name="code" value="{{ r.code }}">
          <input type="hidden" name="decision" value="reject">
          <input type="text" name="notes" placeholder="Rejection reason (optional)">
          <button class="btn reject" type="submit">❌ Reject</button>
        </form>
      </div>
    </div>
    {% endfor %}
  {% endif %}

  {% if not pending and not db_pending %}
    <div class="card">
      <p class="empty">✅ No pending reservation requests.</p>
    </div>
  {% endif %}

  <h2>📋 Reservation History</h2>
  {% if history %}
  <div class="card">
    <table class="history">
      <tr>
        <th>Code</th><th>Name</th><th>Vehicle</th><th>Zone</th>
        <th>Start</th><th>End</th><th>Status</th><th>Notes</th>
      </tr>
      {% for r in history %}
      <tr>
        <td><strong>{{ r.code }}</strong></td>
        <td>{{ r.name }}</td>
        <td>{{ r.vehicle }}</td>
        <td>Zone {{ r.zone }}</td>
        <td>{{ r.start }}</td>
        <td>{{ r.end }}</td>
        <td class="status-{{ r.status }}">{{ r.status.upper() }}</td>
        <td>{{ r.notes }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% else %}
  <div class="card"><p class="empty">No reservation history yet.</p></div>
  {% endif %}

</body>
</html>"""

_RESULT_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Decision Recorded</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 80px auto;
           text-align: center; }}
    h1 {{ color: {color}; }}
    a  {{ color: #2b6cb0; }}
  </style>
</head>
<body>
  <h1>{icon} {message}</h1>
  <p>Request <strong>{code}</strong> has been <strong>{action}</strong>.</p>
  <p><a href="/admin">← Back to Dashboard</a></p>
</body>
</html>"""


# ─── Pydantic model for JSON decide requests ─────────────────────────────────

from pydantic import BaseModel


class DecideRequest(BaseModel):
    code: str
    approved: bool
    notes: str = ""


# ─── FastAPI app factory ──────────────────────────────────────────────────────

def create_app():
    from fastapi import FastAPI, Form, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from jinja2 import Template

    from src.admin_agent.decision_store import (
        get_reservation,
        list_pending,
        submit_decision,
    )

    app = FastAPI(
        title="SmartPark Admin API",
        # Disable docs UI to reduce attack surface (mirrors MCP server pattern)
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # ── GET /admin — HTML dashboard ───────────────────────────────────────────

    @app.get("/admin", response_class=HTMLResponse)
    async def dashboard():
        pending = list_pending()
        import src.config as cfg
        from src.database.models import init_db
        from src.database.operations import get_all_reservations_full
        try:
            init_db(cfg.SQLITE_DB_PATH)
            all_res = get_all_reservations_full(cfg.SQLITE_DB_PATH)
            history = [
                {
                    "code": r.reservation_code,
                    "name": r.full_name,
                    "vehicle": r.car_number,
                    "zone": r.zone,
                    "start": str(r.start_datetime.date()) if r.start_datetime else "",
                    "end": str(r.end_datetime.date()) if r.end_datetime else "",
                    "status": r.status,
                    "notes": r.admin_notes or "",
                }
                for r in all_res
                if r.status != "pending"
            ]
        except Exception:
            history = []

        try:
            from src.database.operations import get_db_pending_reservations
            raw_pending = get_db_pending_reservations(cfg.SQLITE_DB_PATH)
            db_pending = [
                {
                    "code": r.reservation_code,
                    "name": r.full_name,
                    "email": r.email or "",
                    "vehicle": r.car_number,
                    "zone": r.zone,
                    "start": str(r.start_datetime.date()) if r.start_datetime else "",
                    "end": str(r.end_datetime.date()) if r.end_datetime else "",
                    "card": r.card_number or "",
                    "created": str(r.created_at)[:16] if r.created_at else "",
                }
                for r in raw_pending
            ]
        except Exception:
            db_pending = []

        html = Template(_DASHBOARD_HTML).render(
            pending=pending, history=history, db_pending=db_pending
        )
        return HTMLResponse(content=html)

    # ── GET /admin/pending — JSON list ────────────────────────────────────────

    @app.get("/admin/pending")
    async def api_pending():
        return list_pending()

    # ── GET /admin/reservation/{code} — JSON detail ───────────────────────────

    @app.get("/admin/reservation/{code}")
    async def api_reservation(code: str):
        rd = get_reservation(code)
        if rd is None:
            raise HTTPException(status_code=404, detail="Not found")
        return rd

    # ── POST /admin/decide — JSON body OR form submission ─────────────────────

    @app.post("/admin/decide")
    async def api_decide_post(request: Request):
        """
        Accepts two content types on the same endpoint:
          • application/json  → used by the admin agent / REST clients
          • application/x-www-form-urlencoded → used by the dashboard HTML forms
        """
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
            code = data.get("code", "")
            approved = bool(data.get("approved", False))
            notes = data.get("notes", "")
            is_json = True
        else:
            form = await request.form()
            code = str(form.get("code", ""))
            decision = str(form.get("decision", "reject"))
            approved = (decision == "approve")
            notes = str(form.get("notes", ""))
            is_json = False

        if not submit_decision(code, approved, notes):
            if is_json:
                raise HTTPException(
                    status_code=404, detail="Code not found or already processed"
                )
            return HTMLResponse(
                content=_RESULT_HTML.format(
                    color="#e53e3e", icon="❌", message="Error",
                    code=code, action="not found — it may have already been processed",
                ),
                status_code=404,
            )

        action = "approved" if approved else "rejected"
        logger.info("[AdminAPI] Reservation %s %s (notes: %s)", code, action, notes)
        if is_json:
            return JSONResponse({"status": "ok", "code": code, "approved": approved})
        return HTMLResponse(
            content=_RESULT_HTML.format(
                color="#38a169" if approved else "#e53e3e",
                icon="✅" if approved else "❌",
                message="Approved!" if approved else "Rejected.",
                code=code, action=action,
            )
        )

    # ── GET /admin/decide — one-click from email links ────────────────────────

    @app.get("/admin/decide", response_class=HTMLResponse)
    async def api_decide_get(
        code: str = Query(default=""),
        decision: str = Query(default="reject"),
        notes: str = Query(default=""),
    ):
        """One-click approve/reject links embedded in notification emails."""
        approved = (decision == "approve")
        if not submit_decision(code, approved, notes):
            return HTMLResponse(
                content=_RESULT_HTML.format(
                    color="#e53e3e", icon="❌", message="Error",
                    code=code, action="not found or already processed",
                ),
                status_code=404,
            )
        action = "approved" if approved else "rejected"
        logger.info("[AdminAPI] Reservation %s %s via email link.", code, action)
        return HTMLResponse(
            content=_RESULT_HTML.format(
                color="#38a169" if approved else "#e53e3e",
                icon="✅" if approved else "❌",
                message="Approved!" if approved else "Rejected.",
                code=code, action=action,
            )
        )

    # ── POST /admin/decide_db — web-form reservations ─────────────────────────

    @app.post("/admin/decide_db", response_class=HTMLResponse)
    async def api_decide_db(
        code: str = Form(default=""),
        decision: str = Form(default="reject"),
        notes: str = Form(default=""),
    ):
        """Approve or reject a DB-pending reservation submitted via the web form."""
        import src.config as cfg
        from src.database.operations import (
            approve_pending_reservation,
            reject_pending_reservation,
        )
        approved = (decision == "approve")
        if approved:
            ok = approve_pending_reservation(cfg.SQLITE_DB_PATH, code, notes)
        else:
            ok = reject_pending_reservation(cfg.SQLITE_DB_PATH, code, notes)

        if not ok:
            return HTMLResponse(
                content=_RESULT_HTML.format(
                    color="#e53e3e", icon="❌", message="Error",
                    code=code, action="not found — it may have already been processed",
                ),
                status_code=404,
            )
        action = "approved" if approved else "rejected"
        logger.info("[AdminAPI] Web-form reservation %s %s (notes: %s)", code, action, notes)
        return HTMLResponse(
            content=_RESULT_HTML.format(
                color="#38a169" if approved else "#e53e3e",
                icon="✅" if approved else "❌",
                message="Approved!" if approved else "Rejected.",
                code=code, action=action,
            )
        )

    return app


# ─── Server lifecycle ─────────────────────────────────────────────────────────

def get_admin_url() -> str:
    return f"http://{_server_host}:{_server_port}"


def start_api_server(host: str = "localhost", port: int = 5001) -> bool:
    """
    Start the FastAPI admin server in a background daemon thread via uvicorn.
    Safe to call multiple times — only starts once.
    Returns True on success.
    """
    global _server_started, _server_host, _server_port

    with _server_lock:
        if _server_started:
            return True

        _server_host = host
        _server_port = port

        try:
            import socket as _socket

            import uvicorn

            app = create_app()
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="error",
                access_log=False,
            )
            server = uvicorn.Server(config)
            # Signal handlers can only be registered on the main thread;
            # disable so uvicorn runs safely in a daemon thread.
            server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

            t = threading.Thread(
                target=server.run,
                daemon=True,
                name="admin-api-server",
            )
            t.start()

            # Poll until the socket is accepting connections (max 10 s).
            # Always use 127.0.0.1 to avoid Windows IPv6 "localhost" ambiguity.
            poll_host = "127.0.0.1"
            deadline = time.monotonic() + 10.0
            ready = False
            while time.monotonic() < deadline:
                try:
                    with _socket.create_connection((poll_host, port), timeout=0.5):
                        ready = True
                        break
                except OSError:
                    time.sleep(0.1)

            if not ready:
                logger.warning(
                    "[AdminAPI] Server did not become ready within 10s on %s:%d.", host, port
                )
            else:
                logger.info(
                    "[AdminAgent] FastAPI admin server ready: http://%s:%d/admin", host, port
                )

            _server_started = True
            return True

        except Exception as exc:
            logger.error("[AdminAgent] Failed to start REST API: %s", exc)
            return False
