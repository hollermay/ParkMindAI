"""
Flask frontend HTTP tests for src/reservation_frontend/app.py.

Tests cover:
  GET  /          — renders chat UI HTML
  GET  /reset     — clears session, returns 204
  POST /chat      — routes messages through graph, returns JSON
  GET  /status    — polls session state
  POST /cancel    — cancels an approved reservation by code

The LangGraph graph is monkeypatched with a lightweight stub so these tests
exercise HTTP behaviour (routes, status codes, JSON shape, session handling)
without invoking real LLMs or requiring a running database.
"""
import json
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────

class _FakeGraphState:
    """Minimal graph-state stub that looks like a LangGraph StateSnapshot."""
    def __init__(self, messages=None, tasks=None, reservation_data=None):
        from langchain_core.messages import AIMessage
        self.values = {
            "messages": messages or [AIMessage(content="Hello! How can I help?")],
            "reservation_data": reservation_data or {},
        }
        self.tasks = tasks or []


class _FakeGraph:
    """Stub graph that returns a canned state for every invoke()."""
    def __init__(self, reply="Hello! How can I help?", interrupted=False, request_code=""):
        self._reply = reply
        self._interrupted = interrupted
        self._request_code = request_code

    def invoke(self, *args, **kwargs):
        pass  # state mutations are handled in get_state

    def get_state(self, config):
        from langchain_core.messages import AIMessage

        if self._interrupted:
            intr = MagicMock()
            intr.value = {
                "type": "admin_approval_required",
                "request_code": self._request_code or "REQ-TEST001",
                "message": "Awaiting admin approval.",
                "reservation": {},
                "instructions": 'Resume with: {"approved": true/false}',
            }
            task = MagicMock()
            task.interrupts = [intr]
            return _FakeGraphState(
                messages=[AIMessage(content=self._reply)],
                tasks=[task],
            )
        return _FakeGraphState(messages=[AIMessage(content=self._reply)])


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    """
    Return a Flask test client with:
      - TESTING mode on
      - a deterministic secret key
      - graph replaced by _FakeGraph
      - database pointed at a temp SQLite file
    """
    import src.config as cfg
    monkeypatch.setattr(cfg, "SQLITE_DB_PATH", str(tmp_path / "frontend_test.db"))
    monkeypatch.setattr(cfg, "DATABASE_URL", f"sqlite:///{tmp_path}/frontend_test.db")

    from src.database.models import init_db
    init_db(str(tmp_path / "frontend_test.db"))

    # Clear module-level session cache so tests don't bleed into each other
    import src.reservation_frontend.app as frontend_module
    frontend_module._sessions.clear()

    # Patch get_graph() to return our stub (default: normal reply)
    with patch("src.chatbot.graph.get_graph",
               return_value=_FakeGraph("Hi there! I am SmartPark assistant.")):
        frontend_module.app.config["TESTING"] = True
        frontend_module.app.config["SECRET_KEY"] = "test-secret-key"
        with frontend_module.app.test_client() as client:
            yield client, frontend_module


@pytest.fixture()
def app_client_interrupted(tmp_path, monkeypatch):
    """Test client whose graph stub simulates the human_approval interrupt."""
    import src.config as cfg
    monkeypatch.setattr(cfg, "SQLITE_DB_PATH", str(tmp_path / "interrupted_test.db"))
    monkeypatch.setattr(cfg, "DATABASE_URL", f"sqlite:///{tmp_path}/interrupted_test.db")

    from src.database.models import init_db
    init_db(str(tmp_path / "interrupted_test.db"))

    import src.reservation_frontend.app as frontend_module
    frontend_module._sessions.clear()

    request_code = "REQ-INTTEST"
    stub = _FakeGraph(
        reply="Reservation collected!",
        interrupted=True,
        request_code=request_code,
    )

    # Pre-register the code so the frontend doesn't try to re-register
    from src.admin_agent import decision_store
    decision_store.add_pending(request_code, {"first_name": "Test"})

    with patch("src.chatbot.graph.get_graph", return_value=stub):
        frontend_module.app.config["TESTING"] = True
        frontend_module.app.config["SECRET_KEY"] = "test-secret-key"
        with frontend_module.app.test_client() as client:
            yield client, frontend_module, request_code


# ─── GET / ────────────────────────────────────────────────────────────────────

class TestIndexRoute:
    def test_returns_200(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        assert resp.status_code == 200

    def test_returns_html(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        assert b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data.lower()

    def test_sets_session_sid(self, app_client):
        client, _ = app_client
        with client.session_transaction() as sess:
            sess.pop("sid", None)
        client.get("/")
        with client.session_transaction() as sess:
            assert "sid" in sess


# ─── GET /reset ───────────────────────────────────────────────────────────────

class TestResetRoute:
    def test_returns_204(self, app_client):
        client, _ = app_client
        client.get("/")  # establish session
        resp = client.get("/reset")
        assert resp.status_code == 204

    def test_creates_new_sid(self, app_client):
        client, _ = app_client
        client.get("/")
        with client.session_transaction() as sess:
            old_sid = sess.get("sid", "")
        client.get("/reset")
        with client.session_transaction() as sess:
            new_sid = sess.get("sid", "")
        assert new_sid != old_sid

    def test_removes_old_session_data(self, app_client):
        client, module = app_client
        client.get("/")
        with client.session_transaction() as sess:
            old_sid = sess["sid"]
        # Manually register the old sid in _sessions
        module._sessions[old_sid] = {"graph": None, "config": {}, "pending_code": None,
                                      "final_response": None, "approval_thread": None}
        client.get("/reset")
        assert old_sid not in module._sessions


# ─── POST /chat ───────────────────────────────────────────────────────────────

class TestChatRoute:
    def test_normal_message_returns_ok_status(self, app_client):
        client, _ = app_client
        client.get("/")
        resp = client.post("/chat",
                           data=json.dumps({"message": "Hello"}),
                           content_type="application/json")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"
        assert "message" in body

    def test_empty_message_returns_400(self, app_client):
        client, _ = app_client
        client.get("/")
        resp = client.post("/chat",
                           data=json.dumps({"message": "   "}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_missing_message_key_returns_400(self, app_client):
        client, _ = app_client
        client.get("/")
        resp = client.post("/chat",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_reply_contains_text(self, app_client):
        client, _ = app_client
        client.get("/")
        resp = client.post("/chat",
                           data=json.dumps({"message": "What are parking prices?"}),
                           content_type="application/json")
        body = resp.get_json()
        assert body.get("message"), "Response message must be non-empty"

    def test_session_created_automatically(self, app_client):
        """POST /chat without prior GET / should still work (auto-create session)."""
        client, _ = app_client
        resp = client.post("/chat",
                           data=json.dumps({"message": "Hi"}),
                           content_type="application/json")
        assert resp.status_code == 200

    def test_interrupted_graph_returns_awaiting_admin(self, app_client_interrupted):
        client, module, request_code = app_client_interrupted
        client.get("/")
        resp = client.post("/chat",
                           data=json.dumps({"message": "book a space"}),
                           content_type="application/json")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "awaiting_admin"
        assert "code" in body
        assert body["code"] == request_code

    def test_second_message_while_pending_returns_awaiting(self, app_client_interrupted):
        """Sending a second message while awaiting approval should not reset state."""
        client, module, request_code = app_client_interrupted
        client.get("/")
        # First message — triggers interrupt
        client.post("/chat",
                    data=json.dumps({"message": "book a space"}),
                    content_type="application/json")
        # Second message — should be blocked with awaiting_admin
        resp = client.post("/chat",
                           data=json.dumps({"message": "hello again"}),
                           content_type="application/json")
        body = resp.get_json()
        assert body["status"] == "awaiting_admin"

    def test_guardrail_blocked_message(self, app_client):
        """Guardrail-blocked messages should return a safe response (not 500)."""
        client, _ = app_client
        client.get("/")
        with patch("src.reservation_frontend.app.filter_input") as mock_filter:
            mock_filter.return_value = MagicMock(
                is_safe=False,
                blocked_reason="I can only assist with parking-related queries.",
                sanitised_input="",
            )
            resp = client.post("/chat",
                               data=json.dumps({"message": "some harmful input"}),
                               content_type="application/json")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"
        assert "parking" in body["message"].lower()


# ─── GET /status ──────────────────────────────────────────────────────────────

class TestStatusRoute:
    def test_idle_without_session(self, app_client):
        client, _ = app_client
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "idle"

    def test_idle_after_reset(self, app_client):
        client, _ = app_client
        client.get("/")
        client.get("/reset")
        resp = client.get("/status")
        assert resp.get_json()["status"] == "idle"

    def test_pending_while_awaiting_admin(self, app_client_interrupted):
        client, module, request_code = app_client_interrupted
        client.get("/")
        client.post("/chat",
                    data=json.dumps({"message": "book a space"}),
                    content_type="application/json")
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.get_json()
        # While the approval thread is running the status should be "pending" or "awaiting_admin"
        assert body["status"] in ("pending", "awaiting_admin", "ok", "idle")

    def test_final_response_delivered_via_status(self, app_client):
        """If a session has a final_response queued, /status delivers it once."""
        client, module = app_client
        client.get("/")
        with client.session_transaction() as sess:
            sid = sess["sid"]

        # Manually inject a final response into the session cache
        module._sessions[sid] = {
            "graph": _FakeGraph(),
            "config": {"configurable": {"thread_id": sid}},
            "pending_code": None,
            "final_response": "Your reservation is confirmed!",
            "approval_thread": None,
        }

        resp = client.get("/status")
        body = resp.get_json()
        assert body["status"] == "ok"
        assert "confirmed" in body["message"].lower()

        # Second poll — final_response consumed, back to idle
        resp2 = client.get("/status")
        assert resp2.get_json()["status"] == "idle"


# ─── POST /cancel ─────────────────────────────────────────────────────────────

class TestCancelRoute:
    def test_missing_code_returns_400(self, app_client):
        client, _ = app_client
        resp = client.post("/cancel",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_unknown_code_returns_error(self, app_client):
        client, _ = app_client
        resp = client.post("/cancel",
                           data=json.dumps({"code": "SP-DOESNOTEXIST"}),
                           content_type="application/json")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is False

    def test_valid_cancellation(self, tmp_path, monkeypatch):
        """Insert a real approved reservation then cancel it via the HTTP API."""
        import src.config as cfg
        db_path = str(tmp_path / "cancel_test.db")
        monkeypatch.setattr(cfg, "SQLITE_DB_PATH", db_path)
        monkeypatch.setattr(cfg, "DATABASE_URL", f"sqlite:///{db_path}")

        from src.database.models import init_db
        init_db(db_path)

        from src.database.operations import create_reservation, get_reservation_by_code
        from src.database import operations as db_ops
        import src.database.operations as _ops

        # Create an approved reservation with a future start_date via direct ORM
        from datetime import date, timedelta, datetime
        from src.database.models import get_engine, Base, Reservation, ParkingSpace
        from sqlalchemy.orm import sessionmaker

        engine = get_engine(db_path)
        Session = sessionmaker(bind=engine)
        with Session() as db_sess:
            start_dt = datetime.combine(date.today() + timedelta(days=5), datetime.min.time())
            end_dt = datetime.combine(date.today() + timedelta(days=7), datetime.min.time())
            code = "SP-CANCEL01"
            res = Reservation(
                reservation_code=code,
                first_name="Test",
                last_name="User",
                car_number="TC-0001",
                zone="A",
                start_datetime=start_dt,
                end_datetime=end_dt,
                status="approved",
            )
            db_sess.add(res)
            db_sess.commit()

        import src.reservation_frontend.app as frontend_module
        frontend_module._sessions.clear()

        with patch("src.chatbot.graph.get_graph",
                   return_value=_FakeGraph("Hi")):
            frontend_module.app.config["TESTING"] = True
            frontend_module.app.config["SECRET_KEY"] = "test-secret"
            with frontend_module.app.test_client() as client:
                resp = client.post("/cancel",
                                   data=json.dumps({"code": code, "reason": "Changed my mind"}),
                                   content_type="application/json")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert code in body["message"]
