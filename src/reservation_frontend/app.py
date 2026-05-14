"""
SmartPark City Center — Web Chat Frontend

Serves a browser-based chat UI that talks to the LangGraph chatbot.
The chatbot handles reservations, Q&A, and admin-approval flow.

Routes:
  GET  /          — chat UI
  POST /chat      — send a user message, returns bot reply (JSON)
  GET  /status    — poll for pending-approval / final result (JSON)
  GET  /reset     — start a new conversation
"""
import logging
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import src.config as cfg
from src.database.models import init_db
from src.guardrails.filters import filter_input

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = cfg.__dict__.get("FLASK_SECRET", "smartpark-chat-secret-key-2024")

# Per-session state:
# session_id -> {"graph": graph, "config": dict, "pending_code": str|None,
#                "final_response": str|None, "approval_thread": Thread|None}
_sessions: dict = {}
_sessions_lock = threading.Lock()


def _get_or_create_session(sid: str) -> dict:
    with _sessions_lock:
        if sid not in _sessions:
            from src.chatbot.graph import get_graph
            g = get_graph()
            _sessions[sid] = {
                "graph": g,
                "config": {"configurable": {"thread_id": sid}},
                "pending_code": None,
                "final_response": None,
                "approval_thread": None,
            }
        return _sessions[sid]


def _last_ai_message(graph_state) -> str:
    messages = graph_state.values.get("messages", [])
    for msg in reversed(messages):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content and msg.__class__.__name__ == "AIMessage":
            return content
    return ""


def _run_approval_background(sess: dict, code: str) -> None:
    """Poll decision_store, resume the graph when admin decides."""
    from src.admin_agent import decision_store
    from src.admin_agent.notification import send_notification
    from src.admin_agent.api_server import get_admin_url
    from langgraph.types import Command

    rd = sess["graph"].get_state(sess["config"]).values.get("reservation_data", {})
    send_notification(code, rd, get_admin_url())

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
                sess["graph"].invoke(
                    Command(resume={"approved": approved, "notes": notes}),
                    config=sess["config"],
                )
                final_state = sess["graph"].get_state(sess["config"])
                sess["final_response"] = _last_ai_message(final_state)
            except Exception as exc:
                logger.error("Graph resume error: %s", exc)
                sess["final_response"] = (
                    "There was an error processing the admin decision. "
                    "Please contact support at +1 (555) 123-4567."
                )
            sess["pending_code"] = None
            return
        time.sleep(poll_interval)
        elapsed += poll_interval

    # Timeout — auto-reject
    try:
        sess["graph"].invoke(
            Command(resume={"approved": False, "notes": "Timed out waiting for admin."}),
            config=sess["config"],
        )
        final_state = sess["graph"].get_state(sess["config"])
        sess["final_response"] = _last_ai_message(final_state)
    except Exception:
        sess["final_response"] = (
            "Sorry, the admin approval timed out. "
            "Please try again or call +1 (555) 123-4567."
        )
    sess["pending_code"] = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return _CHAT_PAGE_HTML


@app.route("/reset")
def reset():
    old_sid = session.pop("sid", None)
    if old_sid:
        with _sessions_lock:
            _sessions.pop(old_sid, None)
    session["sid"] = str(uuid.uuid4())
    return ("", 204)


@app.route("/chat", methods=["POST"])
def chat():
    from langchain_core.messages import HumanMessage

    sid = session.get("sid")
    if not sid:
        session["sid"] = str(uuid.uuid4())
        sid = session["sid"]

    data = request.get_json(force=True, silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    sess = _get_or_create_session(sid)

    if sess["pending_code"] is not None:
        return jsonify({
            "status": "awaiting_admin",
            "message": "\u23f3 Your reservation is awaiting admin approval. Please wait\u2026",
        })

    filter_result = filter_input(user_text)
    if not filter_result.is_safe:
        return jsonify({"status": "ok", "message": filter_result.blocked_reason})

    try:
        sess["graph"].invoke(
            {"messages": [HumanMessage(content=filter_result.sanitised_input)]},
            config=sess["config"],
        )
    except Exception as exc:
        logger.error("Graph invocation error: %s", exc, exc_info=True)
        return jsonify({
            "status": "ok",
            "message": "I encountered an unexpected error. Please try again.",
        })

    graph_state = sess["graph"].get_state(sess["config"])

    if any(t.interrupts for t in (graph_state.tasks or ())):
        from src.admin_agent import decision_store
        from src.reservation.handler import mask_card, mask_email

        rd = graph_state.values.get("reservation_data") or {}
        code = decision_store.generate_request_code()
        rd_for_store = {
            **rd,
            "card_masked": mask_card(rd.get("card_number", "")),
            "email_masked": mask_email(rd.get("email", "")),
        }
        decision_store.add_pending(code, rd_for_store)
        sess["pending_code"] = code

        t = threading.Thread(
            target=_run_approval_background,
            args=(sess, code),
            daemon=True,
            name=f"approval-{code}",
        )
        sess["approval_thread"] = t
        t.start()

        from src.admin_agent.api_server import get_admin_url
        admin_url = get_admin_url() + "/admin"
        return jsonify({
            "status": "awaiting_admin",
            "code": code,
            "message": (
                f"\u2705 Your reservation details have been collected!\n\n"
                f"**Request Code: {code}**\n\n"
                f"Your reservation is now pending admin approval. "
                f"You will be notified here once a decision is made.\n\n"
                f"*(Admin Dashboard: [{admin_url}]({admin_url}))*"
            ),
        })

    reply = _last_ai_message(graph_state)
    if not reply:
        reply = "I'm here to help! How can I assist you with parking?"

    return jsonify({"status": "ok", "message": reply})


@app.route("/cancel", methods=["POST"])
def cancel():
    """Immediately cancel an approved reservation by code."""
    from src.database.operations import (
        get_reservation_by_code,
        get_cancellation_policy,
        cancel_reservation,
    )

    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip().upper()
    reason = (data.get("reason") or "").strip()

    if not code:
        return jsonify({"ok": False, "message": "Please provide your reservation code."}), 400

    reservation = get_reservation_by_code(cfg.SQLITE_DB_PATH, code)
    if not reservation:
        return jsonify({
            "ok": False,
            "message": f"No reservation found with code **{code}**. Please check and try again.",
        })

    policy = get_cancellation_policy(reservation)
    if not policy["eligible"]:
        return jsonify({
            "ok": False,
            "message": (
                f"\u274C Cannot cancel reservation **{code}**.\n\n"
                f"{policy['reason']}"
            ),
        })

    ok = cancel_reservation(cfg.SQLITE_DB_PATH, code, reason)
    if not ok:
        return jsonify({
            "ok": False,
            "message": "Could not cancel. The reservation may already be cancelled or not in an approved state.",
        })

    refund_note = (
        "**Full refund** will be processed within 3–5 business days." if policy["free"]
        else f"**{policy['refund_pct']}% refund** will be processed within 3–5 business days per our cancellation policy."
    )
    return jsonify({
        "ok": True,
        "message": (
            f"\u2705 Reservation **{code}** has been **cancelled immediately**.\n\n"
            f"{refund_note}\n\n"
            f"The parking space has been released back to the pool."
        ),
    })


@app.route("/status")
def status():
    sid = session.get("sid")
    if not sid:
        return jsonify({"status": "idle"})

    with _sessions_lock:
        sess = _sessions.get(sid)

    if sess is None:
        return jsonify({"status": "idle"})

    if sess["final_response"] is not None:
        resp = sess["final_response"]
        sess["final_response"] = None
        sess["pending_code"] = None
        return jsonify({"status": "ok", "message": resp})

    if sess["pending_code"] is not None:
        return jsonify({"status": "awaiting_admin"})

    return jsonify({"status": "idle"})


# ── Inline HTML page ───────────────────────────────────────────────────────────

_CHAT_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartPark City Center</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #2b6cb0 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
  .chat-container {
    background: white;
    border-radius: 18px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    width: 100%;
    max-width: 760px;
    display: flex;
    flex-direction: column;
    height: 88vh;
    max-height: 820px;
    overflow: hidden;
  }
  .chat-header {
    background: linear-gradient(90deg, #2c5282, #2b6cb0);
    color: white;
    padding: 18px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
  }
  .chat-header .icon { font-size: 32px; line-height: 1; }
  .chat-header h1 { font-size: 19px; font-weight: 700; letter-spacing: 0.3px; }
  .chat-header p { font-size: 12px; opacity: 0.8; margin-top: 2px; }
  .header-actions { margin-left: auto; display: flex; gap: 8px; }
  .btn-icon {
    background: rgba(255,255,255,0.15);
    border: none; color: white; border-radius: 8px;
    padding: 7px 12px; cursor: pointer; font-size: 13px;
    transition: background 0.2s; text-decoration: none;
    display: inline-block;
  }
  .btn-icon:hover { background: rgba(255,255,255,0.3); }

  .chat-messages {
    flex: 1; overflow-y: auto; padding: 20px;
    display: flex; flex-direction: column; gap: 14px;
    background: #f7fafc;
  }
  .chat-messages::-webkit-scrollbar { width: 5px; }
  .chat-messages::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 3px; }

  .msg {
    max-width: 78%; padding: 12px 16px; border-radius: 16px;
    font-size: 14.5px; line-height: 1.6; word-wrap: break-word;
  }
  .msg.bot {
    background: white; color: #2d3748;
    border: 1px solid #e2e8f0;
    border-bottom-left-radius: 4px; align-self: flex-start;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .msg.user {
    background: linear-gradient(135deg, #3182ce, #2b6cb0);
    color: white; border-bottom-right-radius: 4px; align-self: flex-end;
    box-shadow: 0 2px 8px rgba(49,130,206,0.3);
  }
  .msg.system {
    background: #fffbeb; border: 1px solid #f6e05e; color: #744210;
    border-radius: 10px; align-self: center;
    max-width: 90%; font-size: 13.5px; text-align: center;
  }
  .msg.error {
    background: #fff5f5; border: 1px solid #fed7d7; color: #9b2c2c;
    border-radius: 10px; align-self: center;
    max-width: 90%; font-size: 13.5px; text-align: center;
  }
  .msg-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
    margin-bottom: 4px; opacity: 0.6; text-transform: uppercase;
  }
  .msg.user .msg-label { text-align: right; }

  .typing-indicator {
    display: none; align-self: flex-start;
    background: white; border: 1px solid #e2e8f0;
    border-radius: 16px; border-bottom-left-radius: 4px;
    padding: 12px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .typing-dots span {
    display: inline-block; width: 8px; height: 8px;
    background: #90cdf4; border-radius: 50%; margin: 0 2px;
    animation: bounce 1.2s infinite;
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-8px); }
  }

  .chat-input-area {
    padding: 16px 20px; background: white;
    border-top: 1px solid #e2e8f0;
    display: flex; gap: 10px; align-items: flex-end; flex-shrink: 0;
  }
  .chat-input-area textarea {
    flex: 1; border: 1.5px solid #e2e8f0; border-radius: 12px;
    padding: 11px 14px; font-size: 14.5px; font-family: inherit;
    resize: none; outline: none; transition: border-color 0.2s;
    max-height: 120px; line-height: 1.5;
  }
  .chat-input-area textarea:focus {
    border-color: #3182ce;
    box-shadow: 0 0 0 3px rgba(49,130,206,0.1);
  }
  .send-btn {
    background: linear-gradient(135deg, #3182ce, #2b6cb0);
    color: white; border: none; border-radius: 12px;
    width: 48px; height: 48px; cursor: pointer; font-size: 20px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: transform 0.15s, opacity 0.2s;
  }
  .send-btn:hover { transform: scale(1.06); }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  .status-bar {
    text-align: center; font-size: 12px; color: #718096;
    padding: 4px 20px 10px; background: white; flex-shrink: 0;
  }
  .status-bar.waiting { color: #d69e2e; font-weight: 600; }

  /* Markdown inside bot messages */
  .msg.bot strong { font-weight: 700; }
  .msg.bot em { font-style: italic; }
  .msg.bot code { background: #f0f4f8; border-radius: 4px; padding: 1px 5px; font-family: monospace; font-size: 13px; }
  .msg.bot a { color: #3182ce; text-decoration: underline; }
  .msg.bot ul, .msg.bot ol { padding-left: 20px; margin: 6px 0; }
  .msg.bot li { margin: 3px 0; }
  .msg.bot h3 { font-size: 15px; margin-bottom: 4px; color: #2c5282; }
</style>
</head>
<body>
<div class="chat-container">
  <div class="chat-header">
    <div class="icon">&#x1F17F;</div>
    <div>
      <h1>SmartPark City Center</h1>
      <p>AI Parking Assistant &mdash; ask anything or make a reservation</p>
    </div>
    <div class="header-actions">
      <button class="btn-icon" onclick="newChat()">&#x1F504; New Chat</button>
      <button class="btn-icon" onclick="openCancel()">&#x274C; Cancel Booking</button>
      <a class="btn-icon" href="http://localhost:5001/admin" target="_blank">&#x2699; Admin</a>
    </div>
  </div>

  <!-- Cancel modal -->
  <div id="cancelModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:100;align-items:center;justify-content:center;">
    <div style="background:white;border-radius:14px;padding:28px 32px;width:360px;max-width:95vw;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
      <h2 style="color:#2c5282;margin-bottom:16px;font-size:17px;">&#x274C; Cancel a Reservation</h2>
      <label style="font-size:13px;font-weight:600;color:#4a5568;">Reservation Code</label>
      <input id="cancelCode" type="text" placeholder="e.g. REQ-A1B2C3"
        style="width:100%;margin:6px 0 14px;padding:10px 12px;border:1.5px solid #e2e8f0;border-radius:8px;font-size:14px;outline:none;"
        oninput="this.value=this.value.toUpperCase()">
      <label style="font-size:13px;font-weight:600;color:#4a5568;">Reason (optional)</label>
      <input id="cancelReason" type="text" placeholder="e.g. Change of plans"
        style="width:100%;margin:6px 0 18px;padding:10px 12px;border:1.5px solid #e2e8f0;border-radius:8px;font-size:14px;outline:none;">
      <div style="display:flex;gap:10px;justify-content:flex-end;">
        <button onclick="closeCancel()"
          style="padding:9px 20px;border:1.5px solid #e2e8f0;border-radius:8px;background:white;cursor:pointer;font-size:13px;color:#4a5568;">Back</button>
        <button onclick="submitCancel()"
          style="padding:9px 22px;border:none;border-radius:8px;background:#e53e3e;color:white;font-weight:700;cursor:pointer;font-size:13px;">Cancel Booking</button>
      </div>
      <p id="cancelError" style="color:#e53e3e;font-size:12px;margin-top:10px;display:none;"></p>
    </div>
  </div>

  <div class="chat-messages" id="messages">
    <div class="msg bot">
      <div class="msg-label">ParkBot</div>
      <div class="msg-content">&#x1F44B; <strong>Welcome to SmartPark City Center!</strong><br><br>
      I can help you with:<br>
      &bull; &#x1F17F; <strong>Reserving a parking space</strong> &mdash; say &ldquo;I want to reserve a spot&rdquo;<br>
      &bull; &#x2139;&#xFE0F; <strong>Parking info</strong> &mdash; zones, prices, hours, availability<br>
      &bull; &#x2753; <strong>General questions</strong> about our facilities<br><br>
      How can I help you today?</div>
    </div>
  </div>

  <div class="typing-indicator" id="typing">
    <div class="typing-dots"><span></span><span></span><span></span></div>
  </div>

  <div class="status-bar" id="statusBar">Connected to SmartPark AI</div>

  <div class="chat-input-area">
    <textarea id="userInput" rows="1"
      placeholder="Type a message&hellip; (Enter to send, Shift+Enter for new line)"
      onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button class="send-btn" id="sendBtn" onclick="sendMessage()">&#x27A4;</button>
  </div>
</div>

<script>
  const messagesEl = document.getElementById('messages');
  const inputEl    = document.getElementById('userInput');
  const sendBtn    = document.getElementById('sendBtn');
  const typingEl   = document.getElementById('typing');
  const statusBar  = document.getElementById('statusBar');

  let awaitingAdmin = false;
  let pollInterval  = null;

  function renderMarkdown(text) {
    let s = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    s = s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
    s = s.replace(/__(.+?)__/g,'<strong>$1</strong>');
    s = s.replace(/\*([^*\n]+?)\*/g,'<em>$1</em>');
    s = s.replace(/`([^`]+)`/g,'<code>$1</code>');
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,'<a href="$2" target="_blank">$1</a>');
    s = s.replace(/^[*\-] (.+)$/gm,'<li>$1</li>');
    s = s.replace(/(<li>[\s\S]*?<\/li>)/g,'<ul>$1</ul>');
    s = s.replace(/<\/ul><ul>/g,'');
    s = s.replace(/\n/g,'<br>');
    s = s.replace(/###\s(.+?)(<br>|$)/g,'<h3>$1</h3>');
    return s;
  }

  function addMessage(role, text) {
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    if (role === 'user' || role === 'bot') {
      const lbl = document.createElement('div');
      lbl.className = 'msg-label';
      lbl.textContent = role === 'user' ? 'You' : 'ParkBot';
      wrap.appendChild(lbl);
    }
    const content = document.createElement('div');
    content.className = 'msg-content';
    if (role === 'user') {
      content.textContent = text;
    } else {
      content.innerHTML = renderMarkdown(text);
    }
    wrap.appendChild(content);
    messagesEl.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

  function setLoading(on) {
    sendBtn.disabled = on;
    inputEl.disabled = on;
    typingEl.style.display = on ? 'block' : 'none';
    if (on) scrollToBottom();
  }

  function setStatus(text, waiting) {
    statusBar.textContent = text;
    statusBar.className = 'status-bar' + (waiting ? ' waiting' : '');
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || awaitingAdmin) return;
    addMessage('user', text);
    inputEl.value = '';
    autoResize(inputEl);
    setLoading(true);
    setStatus('ParkBot is thinking\u2026', false);
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: text}),
      });
      const data = await res.json();
      setLoading(false);
      if (data.status === 'awaiting_admin') {
        awaitingAdmin = true;
        addMessage('bot', data.message);
        setStatus('\u23F3 Waiting for admin approval\u2026', true);
        startPolling();
      } else if (data.status === 'ok') {
        addMessage('bot', data.message);
        setStatus('Connected to SmartPark AI', false);
      } else {
        addMessage('error', data.message || 'Something went wrong. Please try again.');
        setStatus('Connected to SmartPark AI', false);
      }
    } catch(err) {
      setLoading(false);
      addMessage('error', '\u26A0 Connection error. Please refresh and try again.');
      setStatus('Connection error', false);
    }
  }

  function startPolling() {
    if (pollInterval) return;
    pollInterval = setInterval(async () => {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        if (data.status === 'ok') {
          stopPolling();
          awaitingAdmin = false;
          addMessage('bot', data.message);
          setStatus('Connected to SmartPark AI', false);
          inputEl.disabled = false;
          sendBtn.disabled = false;
        } else if (data.status === 'idle') {
          stopPolling();
          awaitingAdmin = false;
          setStatus('Connected to SmartPark AI', false);
          inputEl.disabled = false;
          sendBtn.disabled = false;
        }
      } catch(e) {}
    }, 3000);
  }

  function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }

  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  function newChat() {
    if (!confirm('Start a new conversation? The current chat will be cleared.')) return;
    stopPolling();
    awaitingAdmin = false;
    fetch('/reset').then(() => {
      messagesEl.innerHTML = '';
      addMessage('bot', '\uD83D\uDC4B **New conversation started.**\n\nHow can I help you with parking today?');
      setStatus('Connected to SmartPark AI', false);
      inputEl.disabled = false;
      sendBtn.disabled = false;
    });
  }

  // ── Cancel modal ───────────────────────────────────────────────────────────
  function openCancel() {
    document.getElementById('cancelModal').style.display = 'flex';
    document.getElementById('cancelCode').focus();
    document.getElementById('cancelError').style.display = 'none';
  }
  function closeCancel() {
    document.getElementById('cancelModal').style.display = 'none';
    document.getElementById('cancelCode').value = '';
    document.getElementById('cancelReason').value = '';
  }
  async function submitCancel() {
    const code   = document.getElementById('cancelCode').value.trim();
    const reason = document.getElementById('cancelReason').value.trim();
    const errEl  = document.getElementById('cancelError');
    if (!code) { errEl.textContent = 'Please enter a reservation code.'; errEl.style.display='block'; return; }
    errEl.style.display = 'none';
    try {
      const res  = await fetch('/cancel', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({code, reason}),
      });
      const data = await res.json();
      closeCancel();
      addMessage('bot', data.message);
    } catch(e) {
      errEl.textContent = 'Connection error. Please try again.';
      errEl.style.display = 'block';
    }
  }
  // Close modal on outside click
  document.getElementById('cancelModal').addEventListener('click', function(e) {
    if (e.target === this) closeCancel();
  });

  inputEl.focus();
  scrollToBottom();
</script>
</body>
</html>"""


def create_app() -> Flask:
    """Factory — initialises DB then returns app."""
    init_db(cfg.SQLITE_DB_PATH)
    return app
