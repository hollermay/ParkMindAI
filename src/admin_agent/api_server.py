"""
Flask REST API — Admin Decision Server.

Provides the human administrator with a web interface and JSON API to:
  - View all pending reservation approval requests
  - Approve or reject individual requests
  - Review reservation details

Endpoints:
  GET  /admin                                   HTML dashboard (auto-refreshes)
  GET  /admin/pending                           JSON list of pending requests
  GET  /admin/reservation/<code>                JSON details for one request
  POST /admin/decide           {code, approved, notes}   JSON response
  POST /admin/decide           form fields: code, decision, notes
  GET  /admin/decide?code=X&decision=approve    one-click from email links

Runs as a daemon thread — shuts down automatically when the main process exits.
"""
import logging
import threading

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
        <tr><td>Name</td><td>{{ rd.first_name }} {{ rd.last_name }}</td></tr>
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


# ─── Flask app factory ────────────────────────────────────────────────────────

def create_app():
    from flask import Flask, jsonify, render_template_string, request

    from src.admin_agent.decision_store import (
        get_reservation,
        list_pending,
        submit_decision,
    )

    app = Flask(__name__)

    # Silence Flask's request logger
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/admin")
    def dashboard():
        pending = list_pending()
        from src.database.models import init_db
        from src.database.operations import get_all_reservations_full
        import src.config as cfg
        try:
            init_db(cfg.SQLITE_DB_PATH)
            all_res = get_all_reservations_full(cfg.SQLITE_DB_PATH)
            history = [
                {
                    "code": r.reservation_code,
                    "name": f"{r.first_name} {r.last_name}",
                    "vehicle": r.car_number,
                    "zone": r.zone,
                    "start": str(r.start_datetime.date()) if r.start_datetime else "",
                    "end": str(r.end_datetime.date()) if r.end_datetime else "",
                    "status": r.status,
                    "notes": r.admin_notes or "",
                }
                for r in all_res
                if r.status != "pending"  # pending shown separately above
            ]
        except Exception:
            history = []

        # DB-pending reservations from web form
        try:
            from src.database.operations import get_db_pending_reservations
            raw_pending = get_db_pending_reservations(cfg.SQLITE_DB_PATH)
            db_pending = [
                {
                    "code": r.reservation_code,
                    "name": f"{r.first_name} {r.last_name}",
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

        return render_template_string(_DASHBOARD_HTML, pending=pending, history=history, db_pending=db_pending)

    @app.route("/admin/pending", methods=["GET"])
    def api_pending():
        return jsonify(list_pending())

    @app.route("/admin/reservation/<code>", methods=["GET"])
    def api_reservation(code: str):
        rd = get_reservation(code)
        if rd is None:
            return jsonify({"error": "Not found"}), 404
        return jsonify(rd)

    @app.route("/admin/decide", methods=["POST"])
    def api_decide_post():
        """Handle JSON body (API) and HTML form submissions."""
        if request.is_json:
            data = request.get_json(force=True) or {}
            code = data.get("code", "")
            approved = bool(data.get("approved", False))
            notes = data.get("notes", "")
        else:
            code = request.form.get("code", "")
            decision = request.form.get("decision", "reject")
            approved = (decision == "approve")
            notes = request.form.get("notes", "")

        if not submit_decision(code, approved, notes):
            if request.is_json:
                return jsonify({"error": "Code not found or already processed"}), 404
            return _RESULT_HTML.format(
                color="#e53e3e", icon="❌", message="Error",
                code=code, action="not found — it may have already been processed",
            ), 404

        action = "approved" if approved else "rejected"
        logger.info("[AdminAPI] Reservation %s %s (notes: %s)", code, action, notes)
        if request.is_json:
            return jsonify({"status": "ok", "code": code, "approved": approved})
        return _RESULT_HTML.format(
            color="#38a169" if approved else "#e53e3e",
            icon="✅" if approved else "❌",
            message="Approved!" if approved else "Rejected.",
            code=code, action=action,
        )

    @app.route("/admin/decide", methods=["GET"])
    def api_decide_get():
        """One-click approve/reject from email links."""
        code = request.args.get("code", "")
        decision = request.args.get("decision", "reject")
        approved = (decision == "approve")
        notes = request.args.get("notes", "")

        if not submit_decision(code, approved, notes):
            return _RESULT_HTML.format(
                color="#e53e3e", icon="❌", message="Error",
                code=code, action="not found or already processed",
            ), 404

        action = "approved" if approved else "rejected"
        logger.info("[AdminAPI] Reservation %s %s via email link.", code, action)
        return _RESULT_HTML.format(
            color="#38a169" if approved else "#e53e3e",
            icon="✅" if approved else "❌",
            message="Approved!" if approved else "Rejected.",
            code=code, action=action,
        )

    @app.route("/admin/decide_db", methods=["POST"])
    def api_decide_db():
        """Approve or reject a DB-pending reservation submitted via the web form."""
        import src.config as cfg
        from src.database.operations import approve_pending_reservation, reject_pending_reservation
        code = request.form.get("code", "")
        decision = request.form.get("decision", "reject")
        notes = request.form.get("notes", "")
        approved = (decision == "approve")

        if approved:
            ok = approve_pending_reservation(cfg.SQLITE_DB_PATH, code, notes)
        else:
            ok = reject_pending_reservation(cfg.SQLITE_DB_PATH, code, notes)

        if not ok:
            return _RESULT_HTML.format(
                color="#e53e3e", icon="❌", message="Error",
                code=code, action="not found — it may have already been processed",
            ), 404

        action = "approved" if approved else "rejected"
        logger.info("[AdminAPI] Web-form reservation %s %s (notes: %s)", code, action, notes)
        return _RESULT_HTML.format(
            color="#38a169" if approved else "#e53e3e",
            icon="✅" if approved else "❌",
            message="Approved!" if approved else "Rejected.",
            code=code, action=action,
        )

    return app


# ─── Server lifecycle ─────────────────────────────────────────────────────────

def get_admin_url() -> str:
    return f"http://{_server_host}:{_server_port}"


def start_api_server(host: str = "localhost", port: int = 5001) -> bool:
    """
    Start the Flask admin API in a daemon thread.
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
            app = create_app()

            t = threading.Thread(
                target=lambda: app.run(
                    host=host,
                    port=port,
                    use_reloader=False,
                    debug=False,
                ),
                daemon=True,
                name="admin-api-server",
            )
            t.start()
            _server_started = True
            logger.info(
                "[AdminAgent] REST API server started: http://%s:%d/admin", host, port
            )
            return True
        except Exception as exc:
            logger.error("[AdminAgent] Failed to start REST API: %s", exc)
            return False
