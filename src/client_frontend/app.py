"""
SmartPark City Center — Integrated Client Portal

Provides a unified client-facing web application with:
  - Login / Register
  - Dashboard: stats + reservation cards (approved / pending / rejected / cancelled)
    with inline cancel and embedded AI chat panel
  - AI Chatbot for reservations and Q&A (inline, no redirect to port 5000)
  - Cancel an approved reservation inline
  - Admin Dashboard link → http://localhost:5001/admin

Routes:
  GET  /               redirect to /dashboard or /login
  GET  /login          login page
  POST /login          authenticate
  GET  /register       register page
  POST /register       create account
  GET  /dashboard      integrated dashboard + chat (requires login)
  POST /cancel_booking cancel an approved reservation (JSON)
  GET  /logout         clear session
  POST /chat           AI chat endpoint (JSON)
  GET  /chat_status    poll for admin-approval result (JSON)
  GET  /chat_reset     start a new chat conversation
"""

import logging
import sys
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import src.config as cfg
from src.database.models import init_db
from src.guardrails.filters import filter_input

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = cfg.__dict__.get("FLASK_CLIENT_SECRET", "smartpark-client-portal-secret-2024")

# ─── Chatbot session store ────────────────────────────────────────────────────
# keyed by chatbot session-id (separate from the Flask login session)
_chat_sessions: dict = {}
_chat_lock = threading.Lock()


def _get_or_create_chat(sid: str) -> dict:
    with _chat_lock:
        if sid not in _chat_sessions:
            from src.chatbot.graph import get_graph
            g = get_graph()
            _chat_sessions[sid] = {
                "graph": g,
                "config": {"configurable": {"thread_id": sid}},
                "pending_code": None,
                "final_response": None,
                "approval_thread": None,
            }
        return _chat_sessions[sid]


def _last_ai_message(graph_state) -> str:
    messages = graph_state.values.get("messages", [])
    for msg in reversed(messages):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content and msg.__class__.__name__ == "AIMessage":
            return content
    return ""


def _run_approval_background(chat_sess: dict, code: str) -> None:
    """Poll decision_store, resume graph when admin decides."""
    from langgraph.types import Command

    from src.admin_agent import decision_store

    # Notification was already sent by notify_admin inside the human_approval node.
    # Do NOT call send_notification again — that would create a duplicate admin alert.

    timeout = cfg.ADMIN_DECISION_TIMEOUT
    elapsed = 0
    poll_interval = 2

    while elapsed < timeout:
        decision = decision_store.get_decision(code)
        if decision is not None:
            decision_store.clear_decision(code)
            approved = decision["approved"]
            notes = decision.get("notes", "")
            try:
                chat_sess["graph"].invoke(
                    Command(resume={"approved": approved, "notes": notes}),
                    config=chat_sess["config"],
                )
                final_state = chat_sess["graph"].get_state(chat_sess["config"])
                chat_sess["final_response"] = _last_ai_message(final_state)
            except Exception as exc:
                logger.error("Graph resume error: %s", exc)
                chat_sess["final_response"] = (
                    "There was an error processing the admin decision. "
                    "Please contact support at +1 (555) 123-4567."
                )
            chat_sess["pending_code"] = None
            return
        time.sleep(poll_interval)
        elapsed += poll_interval

    # Timeout — auto-reject
    try:
        chat_sess["graph"].invoke(
            Command(resume={"approved": False, "notes": "Timed out waiting for admin."}),
            config=chat_sess["config"],
        )
        final_state = chat_sess["graph"].get_state(chat_sess["config"])
        chat_sess["final_response"] = _last_ai_message(final_state)
    except Exception:
        chat_sess["final_response"] = (
            "Sorry, the admin approval timed out. "
            "Please try again or call +1 (555) 123-4567."
        )
    chat_sess["pending_code"] = None


# ─── Auth helper ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─── Shared CSS ───────────────────────────────────────────────────────────────

_BASE_STYLE = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #2b6cb0 100%);
    min-height: 100vh;
    color: #2d3748;
  }
  a { color: inherit; text-decoration: none; }

  .navbar {
    background: rgba(26,54,93,0.97);
    backdrop-filter: blur(8px);
    color: white;
    padding: 0 24px;
    height: 58px;
    display: flex;
    align-items: center;
    gap: 14px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    flex-wrap: nowrap;
  }
  .navbar .logo { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; white-space: nowrap; }
  .navbar .logo span { color: #63b3ed; }
  .navbar .spacer { flex: 1; }
  .navbar .nav-user { font-size: 12px; opacity: 0.8; white-space: nowrap; }

  .btn {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 8px 16px; border: none; border-radius: 8px;
    font-size: 12.5px; font-weight: 600; cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
    text-decoration: none; white-space: nowrap;
  }
  .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(0,0,0,0.2); }
  .btn:active { transform: none; }
  .btn-primary { background: linear-gradient(135deg,#3182ce,#2b6cb0); color: white; }
  .btn-success { background: linear-gradient(135deg,#38a169,#276749); color: white; }
  .btn-danger  { background: #e53e3e; color: white; }
  .btn-ghost   { background: rgba(255,255,255,0.12); color: white; border: 1px solid rgba(255,255,255,0.25); }
  .btn-ghost:hover { background: rgba(255,255,255,0.22); }
  .btn-outline { background: white; color: #2c5282; border: 1.5px solid #bee3f8; }
  .btn-outline:hover { background: #ebf8ff; }
  .btn-sm { padding: 5px 12px; font-size: 12px; }

  .card { background: white; border-radius: 14px; box-shadow: 0 4px 24px rgba(0,0,0,0.1); padding: 24px; }

  .form-group { margin-bottom: 16px; }
  .form-group label { display: block; font-size: 13px; font-weight: 600; color: #4a5568; margin-bottom: 5px; }
  .form-input {
    width: 100%; padding: 10px 13px;
    border: 1.5px solid #e2e8f0; border-radius: 8px;
    font-size: 13.5px; outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .form-input:focus { border-color: #3182ce; box-shadow: 0 0 0 3px rgba(49,130,206,0.15); }

  .alert { padding: 11px 15px; border-radius: 8px; font-size: 13px; margin-bottom: 14px; }
  .alert-error   { background: #fff5f5; border: 1px solid #fed7d7; color: #9b2c2c; }
  .alert-success { background: #f0fff4; border: 1px solid #9ae6b4; color: #276749; }

  .badge { display: inline-block; padding: 3px 9px; border-radius: 20px; font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
  .badge-approved  { background: #c6f6d5; color: #22543d; }
  .badge-pending   { background: #fefcbf; color: #744210; }
  .badge-rejected  { background: #fed7d7; color: #742a2a; }
  .badge-cancelled { background: #e2e8f0; color: #4a5568; }
"""

# ─── Auth page ────────────────────────────────────────────────────────────────

_AUTH_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartPark — Client Portal</title>
<style>
""" + _BASE_STYLE + """
  body { display: flex; align-items: center; justify-content: center; padding: 24px; }
  .auth-wrapper {
    display: flex; width: 100%; max-width: 880px; min-height: 540px;
    border-radius: 20px; overflow: hidden; box-shadow: 0 24px 80px rgba(0,0,0,0.4);
  }
  .auth-brand {
    flex: 0 0 340px; background: linear-gradient(160deg,#1a365d 0%,#2b6cb0 100%);
    color: white; padding: 48px 32px; display: flex; flex-direction: column; justify-content: center;
  }
  .auth-brand .big-icon { font-size: 60px; margin-bottom: 18px; }
  .auth-brand h1 { font-size: 26px; font-weight: 800; line-height: 1.2; margin-bottom: 10px; }
  .auth-brand p  { font-size: 13.5px; opacity: 0.8; line-height: 1.7; margin-bottom: 24px; }
  .auth-brand .feature { display: flex; align-items: center; gap: 10px; font-size: 13px; margin-bottom: 10px; opacity: 0.9; }
  .auth-brand .feature .fi { font-size: 17px; width: 26px; text-align: center; }
  .auth-form-panel { flex: 1; background: white; padding: 44px 36px; display: flex; flex-direction: column; justify-content: center; }
  .tab-bar { display: flex; border-bottom: 2px solid #e2e8f0; margin-bottom: 24px; }
  .tab {
    padding: 9px 22px; font-size: 13.5px; font-weight: 600; color: #718096; cursor: pointer;
    border-bottom: 3px solid transparent; margin-bottom: -2px; transition: color 0.2s, border-color 0.2s;
    background: none; border-top: none; border-left: none; border-right: none;
  }
  .tab.active { color: #2b6cb0; border-bottom-color: #2b6cb0; }
  .tab:hover:not(.active) { color: #4a5568; }
  .form-panel { display: none; }
  .form-panel.active { display: block; }
  .form-panel h2 { font-size: 19px; font-weight: 700; color: #1a365d; margin-bottom: 5px; }
  .form-panel .subtitle { font-size: 12.5px; color: #718096; margin-bottom: 20px; }
  .form-row { display: flex; gap: 12px; }
  .form-row .form-group { flex: 1; }
  .submit-btn { width: 100%; padding: 12px; font-size: 14px; border-radius: 10px; margin-top: 4px; }
  .alt-link { text-align: center; margin-top: 14px; font-size: 12.5px; color: #718096; }
  .alt-link button { background: none; border: none; color: #3182ce; font-weight: 600; cursor: pointer; font-size: 12.5px; }
  .alt-link button:hover { text-decoration: underline; }
  @media (max-width: 600px) { .auth-brand { display: none; } .auth-form-panel { padding: 32px 20px; } }
</style>
</head>
<body>
<div class="auth-wrapper">
  <div class="auth-brand">
    <div class="big-icon">&#x1F17F;</div>
    <h1>SmartPark<br>City Center</h1>
    <p>Your smart parking solution — reserve, track, and manage bookings in one place.</p>
    <div class="feature"><span class="fi">&#x1F4C5;</span> Easy online reservations</div>
    <div class="feature"><span class="fi">&#x2705;</span> Real-time booking status</div>
    <div class="feature"><span class="fi">&#x274C;</span> Instant cancellations</div>
    <div class="feature"><span class="fi">&#x1F4AC;</span> AI-powered parking assistant</div>
  </div>
  <div class="auth-form-panel">
    {%- if error %}<div class="alert alert-error">{{ error }}</div>{%- endif %}
    {%- if success %}<div class="alert alert-success">{{ success }}</div>{%- endif %}
    <div class="tab-bar">
      <button class="tab {% if active_tab == 'login' %}active{% endif %}" onclick="showTab('login')">Sign In</button>
      <button class="tab {% if active_tab == 'register' %}active{% endif %}" onclick="showTab('register')">Create Account</button>
    </div>
    <div id="panel-login" class="form-panel {% if active_tab == 'login' %}active{% endif %}">
      <h2>Welcome back</h2>
      <p class="subtitle">Sign in to manage your parking reservations</p>
      <form method="post" action="/login">
        <div class="form-group">
          <label>Email Address</label>
          <input class="form-input" name="email" type="email" placeholder="you@example.com" required autocomplete="email" value="{{ prefill_email or '' }}">
        </div>
        <div class="form-group">
          <label>Password</label>
          <input class="form-input" name="password" type="password" placeholder="Your password" required autocomplete="current-password">
        </div>
        <button type="submit" class="btn btn-primary submit-btn">Sign In &rarr;</button>
      </form>
      <div class="alt-link">Don't have an account? <button onclick="showTab('register')">Create one</button></div>
    </div>
    <div id="panel-register" class="form-panel {% if active_tab == 'register' %}active{% endif %}">
      <h2>Create your account</h2>
      <p class="subtitle">Start managing your parking reservations today</p>
      <form method="post" action="/register">
        <div class="form-row">
                  <div class="form-group">
            <label>Full Name</label>
            <input class="form-input" name="full_name" type="text" placeholder="John Doe" required autocomplete="name">
          </div>
        </div>
        <div class="form-group">
          <label>Email Address</label>
          <input class="form-input" name="email" type="email" placeholder="you@example.com" required autocomplete="email">
        </div>
        <div class="form-group">
          <label>Password</label>
          <input class="form-input" name="password" type="password" placeholder="At least 8 characters" minlength="8" required autocomplete="new-password">
        </div>
        <div class="form-group">
          <label>Confirm Password</label>
          <input class="form-input" name="password2" type="password" placeholder="Repeat your password" minlength="8" required>
        </div>
        <button type="submit" class="btn btn-success submit-btn">Create Account &rarr;</button>
      </form>
      <div class="alt-link">Already have an account? <button onclick="showTab('login')">Sign in</button></div>
    </div>
  </div>
</div>
<script>
  function showTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.form-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('panel-' + name).classList.add('active');
    document.querySelectorAll('.tab')[name === 'login' ? 0 : 1].classList.add('active');
  }
</script>
</body></html>"""

# ─── Dashboard + Embedded Chat page ──────────────────────────────────────────

_DASHBOARD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartPark — Dashboard</title>
<style>
""" + _BASE_STYLE + """

  /* ── Layout ── */
  .app-shell {
    display: flex;
    height: calc(100vh - 58px);
    overflow: hidden;
  }

  /* ── Left panel: dashboard ── */
  .panel-left {
    flex: 1 1 0;
    min-width: 0;
    overflow-y: auto;
    padding: 24px 22px;
  }
  .panel-left::-webkit-scrollbar { width: 5px; }
  .panel-left::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.3); border-radius: 3px; }

  /* ── Divider handle ── */
  .panel-divider {
    width: 6px;
    background: rgba(255,255,255,0.12);
    cursor: col-resize;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s;
  }
  .panel-divider:hover { background: rgba(255,255,255,0.25); }
  .panel-divider::after {
    content: '';
    width: 2px; height: 40px;
    background: rgba(255,255,255,0.4);
    border-radius: 2px;
  }

  /* ── Right panel: chat ── */
  .panel-right {
    flex: 0 0 420px;
    min-width: 300px;
    max-width: 600px;
    display: flex;
    flex-direction: column;
    background: white;
    box-shadow: -4px 0 20px rgba(0,0,0,0.15);
  }

  /* ── Stats ── */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }
  .stat-card {
    background: white; border-radius: 10px;
    padding: 14px 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
  }
  .stat-card .slabel { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #718096; margin-bottom: 4px; }
  .stat-card .svalue { font-size: 28px; font-weight: 800; line-height: 1; }
  .stat-card.total   .svalue { color: #2b6cb0; }
  .stat-card.approved .svalue { color: #276749; }
  .stat-card.pending  .svalue { color: #b7791f; }
  .stat-card.other    .svalue { color: #c53030; }

  /* ── Section header ── */
  .section-hdr { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
  .section-hdr h2 { font-size: 16px; font-weight: 700; color: white; }

  /* ── Filter tabs ── */
  .filter-tabs { display: flex; gap: 5px; flex-wrap: wrap; }
  .filter-tab {
    padding: 5px 13px; border-radius: 20px; font-size: 11.5px; font-weight: 600;
    border: 1.5px solid rgba(255,255,255,0.35); color: rgba(255,255,255,0.75);
    background: rgba(255,255,255,0.1); cursor: pointer; transition: background 0.15s, color 0.15s;
  }
  .filter-tab:hover { background: rgba(255,255,255,0.2); color: white; }
  .filter-tab.active { background: white; color: #2b6cb0; border-color: white; }

  /* ── Reservation cards ── */
  .res-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }

  .res-card {
    background: white; border-radius: 10px;
    padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    border-left: 4px solid #e2e8f0;
    transition: transform 0.15s, box-shadow 0.15s;
    display: none;
  }
  .res-card:hover { transform: translateY(-1px); box-shadow: 0 5px 16px rgba(0,0,0,0.11); }
  .res-card.approved  { border-left-color: #38a169; }
  .res-card.pending   { border-left-color: #d69e2e; }
  .res-card.rejected  { border-left-color: #e53e3e; }
  .res-card.cancelled { border-left-color: #a0aec0; }

  .res-card-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
  .res-code { font-family: monospace; font-size: 14px; font-weight: 700; color: #2b6cb0; }
  .res-meta { font-size: 11px; color: #718096; margin-top: 2px; }
  .res-details {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 6px 10px; font-size: 12px; margin-bottom: 10px;
  }
  .res-details .k { color: #718096; font-size: 10.5px; font-weight: 600; text-transform: uppercase; }
  .res-details .v { color: #2d3748; font-weight: 500; }
  .res-notes {
    font-size: 11.5px; color: #718096; background: #f7fafc;
    border-radius: 5px; padding: 6px 8px; margin-bottom: 8px; border-left: 3px solid #e2e8f0;
  }
  .res-actions { display: flex; gap: 6px; flex-wrap: wrap; }

  /* ── Empty states ── */
  .empty-state {
    text-align: center; padding: 44px 16px;
    background: white; border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
  }
  .empty-state .icon { font-size: 40px; margin-bottom: 12px; }
  .empty-state h3 { font-size: 15px; color: #2d3748; margin-bottom: 6px; }
  .empty-state p  { font-size: 12.5px; color: #718096; }

  /* ── Cancel modal ── */
  .modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.55); z-index: 200;
    align-items: center; justify-content: center;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: white; border-radius: 14px;
    padding: 28px; width: 360px; max-width: 95vw;
    box-shadow: 0 12px 48px rgba(0,0,0,0.35);
  }
  .modal h3 { font-size: 16px; color: #2c5282; margin-bottom: 14px; }
  .modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 18px; }

  /* ── Toast ── */
  #toast {
    position: fixed; bottom: 24px; right: 24px; z-index: 400;
    background: #2d3748; color: white;
    padding: 11px 18px; border-radius: 9px; font-size: 13px; font-weight: 500;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    opacity: 0; transform: translateY(10px);
    transition: opacity 0.3s, transform 0.3s; pointer-events: none;
  }
  #toast.show { opacity: 1; transform: translateY(0); }

  /* ─── Chat panel ─── */
  .chat-header {
    background: linear-gradient(90deg, #2c5282, #2b6cb0);
    color: white; padding: 14px 18px;
    display: flex; align-items: center; gap: 12px; flex-shrink: 0;
  }
  .chat-header .ch-icon { font-size: 22px; }
  .chat-header .ch-title { font-size: 14px; font-weight: 700; }
  .chat-header .ch-sub   { font-size: 11px; opacity: 0.75; }
  .chat-header .ch-reset {
    margin-left: auto; background: rgba(255,255,255,0.15); border: none; color: white;
    border-radius: 7px; padding: 5px 10px; cursor: pointer; font-size: 12px;
    transition: background 0.2s;
  }
  .chat-header .ch-reset:hover { background: rgba(255,255,255,0.28); }

  .chat-msgs {
    flex: 1; overflow-y: auto; padding: 14px 14px 8px;
    display: flex; flex-direction: column; gap: 10px;
    background: #f7fafc;
  }
  .chat-msgs::-webkit-scrollbar { width: 4px; }
  .chat-msgs::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 2px; }

  .cmsg {
    max-width: 82%; padding: 10px 13px; border-radius: 14px;
    font-size: 13.5px; line-height: 1.58; word-wrap: break-word;
  }
  .cmsg.bot {
    background: white; color: #2d3748;
    border: 1px solid #e2e8f0; border-bottom-left-radius: 3px;
    align-self: flex-start; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .cmsg.user {
    background: linear-gradient(135deg,#3182ce,#2b6cb0);
    color: white; border-bottom-right-radius: 3px; align-self: flex-end;
    box-shadow: 0 2px 8px rgba(49,130,206,0.28);
  }
  .cmsg.system {
    background: #fffbeb; border: 1px solid #f6e05e; color: #744210;
    border-radius: 9px; align-self: center; max-width: 92%;
    font-size: 12.5px; text-align: center;
  }
  .cmsg-lbl { font-size: 10px; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 3px; opacity: 0.6; text-transform: uppercase; }
  .cmsg.user .cmsg-lbl { text-align: right; }

  .typing-ind {
    display: none; align-self: flex-start;
    background: white; border: 1px solid #e2e8f0;
    border-radius: 14px; border-bottom-left-radius: 3px;
    padding: 10px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .typing-ind.active { display: block; }
  .typing-dots span {
    display: inline-block; width: 7px; height: 7px;
    background: #90cdf4; border-radius: 50%; margin: 0 2px;
    animation: tdots 1.2s infinite;
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes tdots {
    0%,80%,100% { transform: translateY(0); }
    40% { transform: translateY(-7px); }
  }

  .chat-status-bar {
    text-align: center; font-size: 11.5px; color: #718096;
    padding: 4px 14px 8px; background: white; flex-shrink: 0;
  }
  .chat-status-bar.waiting { color: #d69e2e; font-weight: 600; }

  .chat-input-area {
    padding: 10px 12px; background: white;
    border-top: 1px solid #e2e8f0;
    display: flex; gap: 8px; align-items: flex-end; flex-shrink: 0;
  }
  .chat-input-area textarea {
    flex: 1; border: 1.5px solid #e2e8f0; border-radius: 10px;
    padding: 9px 12px; font-size: 13.5px; font-family: inherit;
    resize: none; outline: none; transition: border-color 0.2s; max-height: 100px; line-height: 1.5;
  }
  .chat-input-area textarea:focus { border-color: #3182ce; box-shadow: 0 0 0 3px rgba(49,130,206,0.1); }
  .chat-send {
    background: linear-gradient(135deg,#3182ce,#2b6cb0);
    color: white; border: none; border-radius: 10px;
    width: 42px; height: 42px; cursor: pointer; font-size: 18px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    transition: transform 0.15s, opacity 0.2s;
  }
  .chat-send:hover { transform: scale(1.06); }
  .chat-send:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  /* Bot message markdown */
  .cmsg.bot strong { font-weight: 700; }
  .cmsg.bot em { font-style: italic; }
  .cmsg.bot code { background:#f0f4f8; border-radius:3px; padding:1px 4px; font-family:monospace; font-size:12px; }
  .cmsg.bot a { color:#3182ce; text-decoration:underline; }
  .cmsg.bot ul, .cmsg.bot ol { padding-left:18px; margin:5px 0; }
  .cmsg.bot li { margin:2px 0; }
  .cmsg.bot h3 { font-size:13.5px; margin-bottom:3px; color:#2c5282; }

  @media (max-width: 700px) {
    .app-shell { flex-direction: column; height: auto; overflow: auto; }
    .panel-right { flex: none; height: 480px; min-width: 0; max-width: 100%; }
    .panel-divider { width: 100%; height: 6px; cursor: row-resize; }
    .panel-divider::after { width: 40px; height: 2px; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .res-details { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar">
  <div class="logo">&#x1F17F; Smart<span>Park</span></div>
  <div class="spacer"></div>
  <span class="nav-user">&#x1F464; {{ user_name }}</span>
  <a class="btn btn-ghost btn-sm" href="http://localhost:5001/admin" target="_blank">&#x2699;&#xFE0F; Admin</a>
  <a class="btn btn-danger btn-sm" href="/logout">Logout</a>
</nav>

<div class="app-shell">

  <!-- ── LEFT: reservations dashboard ── -->
  <div class="panel-left">

    <div class="stats-grid">
      <div class="stat-card total">
        <div class="slabel">Total</div>
        <div class="svalue">{{ stats.total }}</div>
      </div>
      <div class="stat-card approved">
        <div class="slabel">Approved</div>
        <div class="svalue">{{ stats.approved }}</div>
      </div>
      <div class="stat-card pending">
        <div class="slabel">Pending</div>
        <div class="svalue">{{ stats.pending }}</div>
      </div>
      <div class="stat-card other">
        <div class="slabel">Rejected/Cancelled</div>
        <div class="svalue">{{ stats.rejected + stats.cancelled }}</div>
      </div>
    </div>

    <div class="section-hdr">
      <h2>&#x1F4CB; My Reservations</h2>
      <div class="filter-tabs">
        <button class="filter-tab active" data-filter="all">All</button>
        <button class="filter-tab" data-filter="approved">&#x2705; Approved</button>
        <button class="filter-tab" data-filter="pending">&#x23F3; Pending</button>
        <button class="filter-tab" data-filter="rejected">&#x274C; Rejected</button>
        <button class="filter-tab" data-filter="cancelled">&#x1F6AB; Cancelled</button>
      </div>
    </div>

    {% if reservations %}
    <div class="res-grid" id="resGrid">
      {% for r in reservations %}
      <div class="res-card {{ r.status }}" data-status="{{ r.status }}">
        <div class="res-card-top">
          <div>
            <div class="res-code">{{ r.code }}</div>
            <div class="res-meta">Submitted {{ r.created }}</div>
          </div>
          <span class="badge badge-{{ r.status }}">{{ r.status }}</span>
        </div>
        <div class="res-details">
          <div class="k">Zone</div>    <div class="v">Zone {{ r.zone }}</div>
          <div class="k">Vehicle</div> <div class="v">{{ r.vehicle }}</div>
          <div class="k">Start</div>   <div class="v">{{ r.start }}</div>
          <div class="k">End</div>     <div class="v">{{ r.end }}</div>
        </div>
        {% if r.admin_notes %}
        <div class="res-notes">&#x1F4DD; {{ r.admin_notes }}</div>
        {% endif %}
        {% if r.status == 'approved' %}
        <div class="res-actions">
          <button class="btn btn-danger btn-sm" onclick="openCancel('{{ r.code }}')">&#x274C; Cancel</button>
        </div>
        {% elif r.status == 'pending' %}
        <div style="font-size:11.5px;color:#b7791f;">&#x23F3; Awaiting admin review</div>
        {% endif %}
      </div>
      {% endfor %}
    </div>
    {% else %}
    <div class="empty-state" id="emptyAll">
      <div class="icon">&#x1F17F;</div>
      <h3>No reservations yet</h3>
      <p>Use the chat panel on the right to reserve your first parking space.<br>Your bookings will appear here automatically.</p>
    </div>
    {% endif %}
    <div class="empty-state" id="emptyFilter" style="display:none;">
      <div class="icon">&#x1F50D;</div>
      <h3>No matches</h3>
      <p>No reservations match this filter. Try a different status.</p>
    </div>

  </div><!-- /panel-left -->

  <div class="panel-divider" id="divider"></div>

  <!-- ── RIGHT: embedded AI chatbot ── -->
  <div class="panel-right" id="panelRight">

    <div class="chat-header">
      <span class="ch-icon">&#x1F4AC;</span>
      <div>
        <div class="ch-title">ParkBot — AI Assistant</div>
        <div class="ch-sub">Ask anything or reserve a space</div>
      </div>
      <button class="ch-reset" onclick="chatReset()">&#x1F504; New chat</button>
    </div>

    <div class="chat-msgs" id="chatMsgs">
      <div class="cmsg bot">
        <div class="cmsg-lbl">ParkBot</div>
        <div class="cmsg-content">&#x1F44B; <strong>Welcome to SmartPark!</strong><br><br>
        I can help you:<br>
        &bull; &#x1F17F; <strong>Reserve a parking space</strong> — just say &ldquo;I want to reserve a spot&rdquo;<br>
        &bull; &#x2139;&#xFE0F; <strong>Check availability, pricing, hours</strong><br>
        &bull; &#x2753; <strong>Answer any parking questions</strong><br><br>
        How can I help you today?</div>
      </div>
    </div>

    <div class="typing-ind" id="typingInd">
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>

    <div class="chat-status-bar" id="chatStatusBar">Connected</div>

    <div class="chat-input-area">
      <textarea id="chatInput" rows="1"
        placeholder="Type a message… (Enter to send)"
        onkeydown="chatKey(event)" oninput="chatResize(this)"></textarea>
      <button class="chat-send" id="chatSendBtn" onclick="chatSend()">&#x27A4;</button>
    </div>

  </div><!-- /panel-right -->
</div><!-- /app-shell -->

<!-- Cancel modal -->
<div class="modal-overlay" id="cancelOverlay">
  <div class="modal">
    <h3>&#x274C; Cancel Reservation</h3>
    <p style="font-size:13px;color:#4a5568;margin-bottom:14px;">
      Cancel <strong id="cancelCodeDisp"></strong>? This cannot be undone.
    </p>
    <div class="form-group">
      <label>Reason (optional)</label>
      <input class="form-input" id="cancelReason" type="text" placeholder="e.g. Change of plans">
    </div>
    <div id="cancelAlert" class="alert alert-error" style="display:none;"></div>
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="closeCancel()">Back</button>
      <button class="btn btn-danger" id="confirmCancelBtn" onclick="submitCancel()">Confirm Cancel</button>
    </div>
  </div>
</div>

<div id="toast"></div>

""" + r"""<script>
/* ══════════════════════════════════════════════════════
   Reservation panel — filter + cancel
══════════════════════════════════════════════════════ */
const allCards = Array.from(document.querySelectorAll('.res-card'));
const emptyFilter = document.getElementById('emptyFilter');
allCards.forEach(c => c.style.display = 'block'); // show all on load

document.querySelectorAll('.filter-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const f = tab.dataset.filter;
    let vis = 0;
    allCards.forEach(c => {
      const show = f === 'all' || c.dataset.status === f;
      c.style.display = show ? 'block' : 'none';
      if (show) vis++;
    });
    emptyFilter.style.display = (vis === 0 && allCards.length > 0) ? 'block' : 'none';
  });
});

let cancelCode = null;
function openCancel(code) {
  cancelCode = code;
  document.getElementById('cancelCodeDisp').textContent = code;
  document.getElementById('cancelReason').value = '';
  document.getElementById('cancelAlert').style.display = 'none';
  document.getElementById('cancelOverlay').classList.add('open');
}
function closeCancel() { document.getElementById('cancelOverlay').classList.remove('open'); cancelCode = null; }
async function submitCancel() {
  if (!cancelCode) return;
  const btn = document.getElementById('confirmCancelBtn');
  const alertEl = document.getElementById('cancelAlert');
  btn.disabled = true; btn.textContent = 'Cancelling\u2026';
  try {
    const res = await fetch('/cancel_booking', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({code: cancelCode, reason: document.getElementById('cancelReason').value.trim()}),
    });
    const data = await res.json();
    if (data.ok) {
      closeCancel();
      showToast('\u2705 Reservation ' + cancelCode + ' cancelled.');
      setTimeout(() => location.reload(), 1500);
    } else {
      alertEl.textContent = data.message || 'Could not cancel.';
      alertEl.style.display = 'block';
      btn.disabled = false; btn.textContent = 'Confirm Cancel';
    }
  } catch { alertEl.textContent = 'Network error.'; alertEl.style.display = 'block'; btn.disabled = false; btn.textContent = 'Confirm Cancel'; }
}
document.getElementById('cancelOverlay').addEventListener('click', e => { if (e.target === e.currentTarget) closeCancel(); });

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3500);
}

/* ══════════════════════════════════════════════════════
   Chatbot
══════════════════════════════════════════════════════ */
const msgsEl    = document.getElementById('chatMsgs');
const inputEl   = document.getElementById('chatInput');
const sendBtn   = document.getElementById('chatSendBtn');
const typingEl  = document.getElementById('typingInd');
const statusBar = document.getElementById('chatStatusBar');

let awaitingAdmin = false;
let pollTimer     = null;

function renderMarkdown(text) {
  let s = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  s = s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  s = s.replace(/__(.+?)__/g,'<strong>$1</strong>');
  s = s.replace(/\*([^*\n]+?)\*/g,'<em>$1</em>');
  s = s.replace(/`([^`]+)`/g,'<code>$1</code>');
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,'<a href="$2" target="_blank">$1</a>');
  s = s.replace(/^[-*] (.+)$/gm,'<li>$1</li>');
  s = s.replace(/(<li>[\s\S]*?<\/li>)/g,'<ul>$1</ul>');
  s = s.replace(/<\/ul><ul>/g,'');
  s = s.replace(/\n/g,'<br>');
  s = s.replace(/### (.+?)(<br>|$)/g,'<h3>$1</h3>');
  return s;
}

function addMsg(role, text) {
  const wrap = document.createElement('div');
  wrap.className = 'cmsg ' + role;
  if (role === 'user' || role === 'bot') {
    const lbl = document.createElement('div');
    lbl.className = 'cmsg-lbl';
    lbl.textContent = role === 'user' ? 'You' : 'ParkBot';
    wrap.appendChild(lbl);
  }
  const content = document.createElement('div');
  content.className = 'cmsg-content';
  content.innerHTML = role === 'user' ? text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : renderMarkdown(text);
  wrap.appendChild(content);
  msgsEl.appendChild(wrap);
  msgsEl.scrollTop = msgsEl.scrollHeight;
  return wrap;
}

function setLoading(on) {
  sendBtn.disabled = on; inputEl.disabled = on;
  typingEl.classList.toggle('active', on);
  if (on) msgsEl.scrollTop = msgsEl.scrollHeight;
}

function setStatus(txt, waiting) {
  statusBar.textContent = txt;
  statusBar.className = 'chat-status-bar' + (waiting ? ' waiting' : '');
}

function startPolling() {
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch('/chat_status');
      const d = await r.json();
      if (d.status === 'ok') {
        clearInterval(pollTimer); pollTimer = null;
        awaitingAdmin = false;
        addMsg('bot', d.message);
        setStatus('Connected', false);
        setTimeout(() => location.reload(), 2500);
      } else if (d.status !== 'awaiting_admin') {
        clearInterval(pollTimer); pollTimer = null;
        awaitingAdmin = false;
        setStatus('Connected', false);
      }
    } catch {}
  }, 2500);
}

async function chatSend() {
  const text = inputEl.value.trim();
  if (!text || awaitingAdmin) return;
  addMsg('user', text);
  inputEl.value = ''; chatResize(inputEl);
  setLoading(true); setStatus('ParkBot is thinking\u2026', false);
  try {
    const res = await fetch('/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const data = await res.json();
    setLoading(false);
    if (data.status === 'awaiting_admin') {
      awaitingAdmin = true;
      addMsg('bot', data.message);
      setStatus('\u23f3 Waiting for admin approval\u2026', true);
      startPolling();
    } else if (data.status === 'ok') {
      addMsg('bot', data.message);
      setStatus('Connected', false);
      if (data.refresh) setTimeout(() => location.reload(), 1800);
    } else {
      addMsg('system', data.message || 'Something went wrong. Please try again.');
      setStatus('Connected', false);
    }
  } catch {
    setLoading(false);
    addMsg('system', 'Network error. Please try again.');
    setStatus('Connected', false);
  }
}

function chatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); chatSend(); }
}
function chatResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}
async function chatReset() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  awaitingAdmin = false;
  await fetch('/chat_reset');
  msgsEl.innerHTML = '';
  addMsg('bot', '\ud83d\udd04 Chat cleared. How can I help you?');
  setStatus('Connected', false);
}

/* ── Resizable divider ── */
const divider = document.getElementById('divider');
const panelRight = document.getElementById('panelRight');
let dragging = false;
divider.addEventListener('mousedown', e => { dragging = true; e.preventDefault(); });
document.addEventListener('mousemove', e => {
  if (!dragging) return;
  const shell = divider.parentElement;
  const shellRect = shell.getBoundingClientRect();
  const newRight = shellRect.right - e.clientX;
  panelRight.style.flex = 'none';
  panelRight.style.width = Math.max(280, Math.min(700, newRight)) + 'px';
});
document.addEventListener('mouseup', () => { dragging = false; });
</script>
</body></html>"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    error = None
    prefill_email = ""
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        prefill_email = email
        from src.database.operations import authenticate_user
        try:
            init_db(cfg.SQLITE_DB_PATH)
            user = authenticate_user(cfg.SQLITE_DB_PATH, email, password)
        except Exception as exc:
            logger.error("Login DB error: %s", exc)
            user = None
        if user:
            session["user_id"] = user.id
            session["user_email"] = user.email
            session["user_name"] = user.full_name
            return redirect(url_for("dashboard"))
        error = "Invalid email or password. Please try again."
    return render_template_string(_AUTH_PAGE, active_tab="login", error=error, success=None, prefill_email=prefill_email)


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    error = None
    success = None
    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        password2 = request.form.get("password2") or ""
        if password != password2:
            error = "Passwords do not match."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        else:
            from src.database.operations import register_user
            try:
                init_db(cfg.SQLITE_DB_PATH)
                ok, msg = register_user(cfg.SQLITE_DB_PATH, full_name, email, password)
            except Exception as exc:
                logger.error("Register DB error: %s", exc)
                ok, msg = False, "A server error occurred. Please try again."
            if ok:
                return render_template_string(_AUTH_PAGE, active_tab="login", error=None,
                    success="Account created! You can now sign in.", prefill_email=email)
            error = msg
    return render_template_string(_AUTH_PAGE, active_tab="register", error=error, success=success, prefill_email="")


@app.route("/dashboard")
@login_required
def dashboard():
    from src.database.operations import get_reservations_for_user
    email = session.get("user_email", "")
    try:
        init_db(cfg.SQLITE_DB_PATH)
        raw = get_reservations_for_user(cfg.SQLITE_DB_PATH, email)
    except Exception as exc:
        logger.error("Dashboard DB error: %s", exc)
        raw = []

    reservations = [{
        "code": r.reservation_code,
        "status": r.status,
        "zone": r.zone,
        "vehicle": r.car_number,
        "start": str(r.start_datetime.date()) if r.start_datetime else "—",
        "end": str(r.end_datetime.date()) if r.end_datetime else "—",
        "admin_notes": r.admin_notes or "",
        "created": str(r.created_at)[:16] if r.created_at else "—",
    } for r in raw]

    stats = {
        "total":     len(reservations),
        "approved":  sum(1 for r in reservations if r["status"] == "approved"),
        "pending":   sum(1 for r in reservations if r["status"] == "pending"),
        "rejected":  sum(1 for r in reservations if r["status"] == "rejected"),
        "cancelled": sum(1 for r in reservations if r["status"] == "cancelled"),
    }
    return render_template_string(_DASHBOARD_PAGE, user_name=session.get("user_name", "User"),
                                  reservations=reservations, stats=stats)


@app.route("/cancel_booking", methods=["POST"])
@login_required
def cancel_booking():
    from src.database.operations import (
        cancel_reservation,
        get_cancellation_policy,
        get_reservation_by_code,
    )
    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip().upper()
    reason = (data.get("reason") or "").strip()
    if not code:
        return jsonify({"ok": False, "message": "No reservation code provided."}), 400
    try:
        init_db(cfg.SQLITE_DB_PATH)
        reservation = get_reservation_by_code(cfg.SQLITE_DB_PATH, code)
    except Exception as exc:
        logger.error("Cancel lookup error: %s", exc)
        return jsonify({"ok": False, "message": "A server error occurred."}), 500
    if not reservation:
        return jsonify({"ok": False, "message": f"Reservation {code} not found."})
    user_email = session.get("user_email", "").strip().lower()
    if (reservation.email or "").strip().lower() != user_email:
        return jsonify({"ok": False, "message": "You are not authorised to cancel this reservation."}), 403
    policy = get_cancellation_policy(reservation)
    if not policy["eligible"]:
        return jsonify({"ok": False, "message": policy["reason"]})
    ok = cancel_reservation(cfg.SQLITE_DB_PATH, code, reason)
    if not ok:
        return jsonify({"ok": False, "message": "Could not cancel — reservation may already be cancelled."})
    refund = "Full refund" if policy["free"] else f"{policy['refund_pct']}% refund"
    return jsonify({"ok": True, "message": f"Reservation {code} cancelled. {refund} within 3–5 business days."})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─── Chat API routes ──────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    from langchain_core.messages import HumanMessage

    # Each logged-in user gets their own chatbot session, keyed to their user_id
    sid = session.get("chat_sid")
    if not sid:
        sid = f"portal-{session['user_id']}-{uuid.uuid4().hex[:8]}"
        session["chat_sid"] = sid

    data = request.get_json(force=True, silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    chat_sess = _get_or_create_chat(sid)

    if chat_sess["pending_code"] is not None:
        return jsonify({
            "status": "awaiting_admin",
            "message": "\u23f3 Your reservation is awaiting admin approval. Please wait\u2026",
        })

    filter_result = filter_input(user_text)
    if not filter_result.is_safe:
        return jsonify({"status": "ok", "message": filter_result.blocked_reason})

    try:
        chat_sess["graph"].invoke(
            {"messages": [HumanMessage(content=filter_result.sanitised_input)]},
            config=chat_sess["config"],
        )
    except Exception as exc:
        logger.error("Graph invocation error: %s", exc, exc_info=True)
        return jsonify({"status": "ok", "message": "I encountered an unexpected error. Please try again."})

    graph_state = chat_sess["graph"].get_state(chat_sess["config"])

    if any(t.interrupts for t in (graph_state.tasks or ())):
        from src.admin_agent import decision_store
        from src.reservation.handler import mask_card, mask_email

        rd = dict(graph_state.values.get("reservation_data") or {})

        # Automatically populate email from the logged-in user's account
        user_email = session.get("user_email", "")
        if not rd.get("email") and user_email:
            rd["email"] = user_email
            # Persist the email back into the graph state so finalize_reservation
            # can save it to the database when the reservation is approved.
            chat_sess["graph"].update_state(
                chat_sess["config"],
                {"reservation_data": rd},
            )

        # Reuse the request code already registered by the human_approval node
        # (it is embedded in the interrupt payload).  Generating a second code
        # would create a duplicate entry on the admin dashboard.
        code = None
        for _task in (graph_state.tasks or ()):
            for _intr in (_task.interrupts or ()):
                _val = getattr(_intr, "value", None)
                if isinstance(_val, dict):
                    code = _val.get("request_code")
                if code:
                    break
            if code:
                break

        if not code:
            # Fallback: human_approval failed to register — create a new entry.
            code = decision_store.generate_request_code()

        # Update the pending entry with full data (adds email + masked card/email
        # that the basic notify_admin call did not include).
        rd_for_store = {
            **rd,
            "card_masked": mask_card(rd.get("card_number", "")),
            "email_masked": mask_email(rd.get("email", "")),
        }
        decision_store.add_pending(code, rd_for_store)
        chat_sess["pending_code"] = code

        t = threading.Thread(
            target=_run_approval_background, args=(chat_sess, code),
            daemon=True, name=f"approval-{code}",
        )
        chat_sess["approval_thread"] = t
        t.start()

        from src.admin_agent.api_server import get_admin_url
        admin_url = get_admin_url() + "/admin"
        return jsonify({
            "status": "awaiting_admin",
            "code": code,
            "message": (
                f"\u2705 Reservation details collected!\n\n"
                f"**Request Code: {code}**\n\n"
                f"Pending admin approval. You will be notified here once decided.\n\n"
                f"*(Admin Dashboard: [{admin_url}]({admin_url}))*"
            ),
        })

    reply = _last_ai_message(graph_state)
    if not reply:
        reply = "I'm here to help! How can I assist you with parking?"

    # Signal the front-end to refresh the reservations panel if a booking was just made
    refresh = "confirmed" in reply.lower() or "reservation code" in reply.lower() or "approved" in reply.lower()
    return jsonify({"status": "ok", "message": reply, "refresh": refresh})


@app.route("/chat_status")
@login_required
def chat_status():
    sid = session.get("chat_sid")
    if not sid:
        return jsonify({"status": "idle"})
    with _chat_lock:
        chat_sess = _chat_sessions.get(sid)
    if chat_sess is None:
        return jsonify({"status": "idle"})
    if chat_sess["final_response"] is not None:
        resp = chat_sess["final_response"]
        chat_sess["final_response"] = None
        chat_sess["pending_code"] = None
        return jsonify({"status": "ok", "message": resp})
    if chat_sess["pending_code"] is not None:
        return jsonify({"status": "awaiting_admin"})
    return jsonify({"status": "idle"})


@app.route("/chat_reset")
@login_required
def chat_reset():
    old_sid = session.pop("chat_sid", None)
    if old_sid:
        with _chat_lock:
            _chat_sessions.pop(old_sid, None)
    return ("", 204)
