"""
Tests for the SmartPark MCP Server and MCP client.

Coverage:
  TestMCPHealth           — GET /health liveness probe
  TestMCPAuthentication   — missing / wrong / valid API key
  TestMCPRateLimiter      — sliding-window rate limiter
  TestMCPProtocol         — JSON-RPC 2.0 protocol compliance
  TestMCPToolsList        — tools/list method
  TestMCPToolCall         — tools/call → write_confirmed_reservation
  TestMCPFileWrite        — actual file content after successful tool call
  TestMCPInjectionDefence — pipe / newline injection prevention
  TestMCPBatchRequest     — JSON-RPC 2.0 batch requests
  TestInputModel          — Pydantic model sanitisation
  TestDirectWriter        — _direct_write fallback helper
  TestCallHelper          — call_write_confirmed_reservation with/without server
  TestMCPClientClass      — MCPClient health_check and base_url resolution
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── FastAPI test client (no live server needed) ──────────────────────────────

from fastapi.testclient import TestClient

import src.config as cfg
from src.mcp_server.server import (
    WriteReservationInput,
    _rate_limits,
    app,
)

_GOOD_KEY = "test-secret-key"
_AUTH_HEADER = {"Authorization": f"Bearer {_GOOD_KEY}"}


@pytest.fixture(autouse=True)
def patch_api_key(monkeypatch):
    """Override MCP_API_KEY for every test so we control credentials."""
    monkeypatch.setattr(cfg, "MCP_API_KEY", _GOOD_KEY)


@pytest.fixture(autouse=True)
def clear_rate_limits():
    """Reset the in-memory rate-limit dict before every test."""
    _rate_limits.clear()
    yield
    _rate_limits.clear()


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def tmp_reservations_file(tmp_path, monkeypatch):
    """Point RESERVATIONS_FILE_PATH at a temp file for each test."""
    f = tmp_path / "reservations.txt"
    monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(f))
    return f


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _tool_call_body(full_name="Alice Smith", car_number="ABC-1234",
                    reservation_period="2026-06-01 to 2026-06-07",
                    approval_time="2026-05-12 10:00:00 UTC"):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "write_confirmed_reservation",
            "arguments": {
                "full_name": full_name,
                "car_number": car_number,
                "reservation_period": reservation_period,
                "approval_time": approval_time,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPHealth
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_body_has_server_name(self, client):
        r = client.get("/health")
        assert "SmartPark MCP Server" in r.json()["server"]

    def test_health_body_has_timestamp(self, client):
        r = client.get("/health")
        assert "timestamp" in r.json()

    def test_health_requires_no_auth(self, client):
        """Health endpoint must be accessible without any credentials."""
        r = client.get("/health")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPAuthentication
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPAuthentication:
    def test_missing_auth_returns_401(self, client):
        r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        assert r.status_code == 401

    def test_wrong_bearer_returns_401(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 401

    def test_correct_bearer_accepted(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        assert r.status_code == 200

    def test_x_api_key_header_accepted(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"X-API-Key": _GOOD_KEY},
        )
        assert r.status_code == 200

    def test_wrong_x_api_key_returns_401(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"X-API-Key": "bad-key"},
        )
        assert r.status_code == 401

    def test_empty_bearer_returns_401(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Authorization": "Bearer "},
        )
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPRateLimiter
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPRateLimiter:
    def test_within_limit_accepted(self, client):
        """First request should always be accepted."""
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        assert r.status_code == 200

    def test_exceeding_limit_returns_429(self, client, monkeypatch):
        """Simulate the rate limiter being hit by injecting a pre-filled window."""
        import time as _time

        from src.mcp_server import server as srv

        # Pre-fill the limiter at the max count for the test IP (testclient uses "testclient")
        test_ip = "testclient"
        _rate_limits[test_ip] = (srv.RATE_LIMIT_MAX, _time.monotonic())

        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        assert r.status_code == 429

    def test_window_reset_allows_new_requests(self, client, monkeypatch):
        """An expired window should reset the counter."""
        import time as _time

        from src.mcp_server import server as srv

        test_ip = "testclient"
        # Put a full window that started 120 seconds ago (well past the 60s window)
        _rate_limits[test_ip] = (srv.RATE_LIMIT_MAX, _time.monotonic() - 120)

        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPProtocol
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPProtocol:
    def test_invalid_json_returns_parse_error(self, client):
        r = client.post("/mcp", content=b"not-json", headers={**_AUTH_HEADER, "Content-Type": "application/json"})
        body = r.json()
        assert body["error"]["code"] == -32700

    def test_wrong_jsonrpc_version_returns_invalid_request(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "1.0", "id": 1, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        body = r.json()
        assert body["error"]["code"] == -32600

    def test_unknown_method_returns_method_not_found(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "nonexistent/method", "params": {}},
            headers=_AUTH_HEADER,
        )
        body = r.json()
        assert body["error"]["code"] == -32601

    def test_notification_returns_204(self, client):
        """notifications/initialized has no id and expects no response (204)."""
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=_AUTH_HEADER,
        )
        assert r.status_code == 204

    def test_initialize_returns_protocol_version(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers=_AUTH_HEADER,
        )
        result = r.json()["result"]
        assert "protocolVersion" in result
        assert result["serverInfo"]["name"] == "SmartPark MCP Server"

    def test_response_includes_id(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 42, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        assert r.json()["id"] == 42


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPToolsList
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPToolsList:
    def test_tools_list_returns_list(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        assert isinstance(r.json()["result"]["tools"], list)

    def test_tools_list_contains_write_confirmed_reservation(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        names = [t["name"] for t in r.json()["result"]["tools"]]
        assert "write_confirmed_reservation" in names

    def test_tool_schema_has_required_fields(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        tool = r.json()["result"]["tools"][0]
        required = tool["inputSchema"]["required"]
        for field in ["full_name", "car_number", "reservation_period", "approval_time"]:
            assert field in required

    def test_tool_has_description(self, client):
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=_AUTH_HEADER,
        )
        tool = r.json()["result"]["tools"][0]
        assert len(tool["description"]) > 10


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPToolCall
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPToolCall:
    def test_successful_tool_call_returns_200(self, client, tmp_reservations_file):
        r = client.post("/mcp", json=_tool_call_body(), headers=_AUTH_HEADER)
        assert r.status_code == 200

    def test_successful_tool_call_is_not_error(self, client, tmp_reservations_file):
        r = client.post("/mcp", json=_tool_call_body(), headers=_AUTH_HEADER)
        result = r.json()["result"]
        assert result["isError"] is False

    def test_successful_tool_call_returns_text_content(self, client, tmp_reservations_file):
        r = client.post("/mcp", json=_tool_call_body(), headers=_AUTH_HEADER)
        content = r.json()["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "Alice Smith" in content[0]["text"]

    def test_unknown_tool_name_returns_error(self, client):
        body = {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
        r = client.post("/mcp", json=body, headers=_AUTH_HEADER)
        assert r.json()["error"]["code"] == -32601

    def test_tool_call_confirmation_message_contains_car_number(self, client, tmp_reservations_file):
        r = client.post("/mcp", json=_tool_call_body(car_number="XY-9876"), headers=_AUTH_HEADER)
        assert "XY-9876" in r.json()["result"]["content"][0]["text"]


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPFileWrite
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPFileWrite:
    def test_file_created_after_tool_call(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(), headers=_AUTH_HEADER)
        assert tmp_reservations_file.exists()

    def test_file_has_header_row(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(), headers=_AUTH_HEADER)
        content = tmp_reservations_file.read_text(encoding="utf-8")
        assert "Name | Car Number | Reservation Period | Approval Time" in content

    def test_file_has_separator_line(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(), headers=_AUTH_HEADER)
        content = tmp_reservations_file.read_text(encoding="utf-8")
        assert "---" in content

    def test_file_entry_contains_all_fields(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(
            full_name="Bob Jones",
            car_number="ZZ-0001",
            reservation_period="2026-07-01 to 2026-07-05",
            approval_time="2026-05-12 12:00:00 UTC",
        ), headers=_AUTH_HEADER)
        content = tmp_reservations_file.read_text(encoding="utf-8")
        assert "Bob Jones" in content
        assert "ZZ-0001" in content
        assert "2026-07-01 to 2026-07-05" in content
        assert "2026-05-12 12:00:00 UTC" in content

    def test_multiple_calls_append_entries(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(full_name="Guest One", car_number="AA-0001"), headers=_AUTH_HEADER)
        client.post("/mcp", json=_tool_call_body(full_name="Guest Two", car_number="BB-0002"), headers=_AUTH_HEADER)
        content = tmp_reservations_file.read_text(encoding="utf-8")
        assert "Guest One" in content
        assert "Guest Two" in content

    def test_header_written_only_once_for_multiple_calls(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(), headers=_AUTH_HEADER)
        client.post("/mcp", json=_tool_call_body(), headers=_AUTH_HEADER)
        content = tmp_reservations_file.read_text(encoding="utf-8")
        assert content.count("Name | Car Number") == 1

    def test_entry_uses_pipe_delimiter(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(
            full_name="Pipe Test",
            car_number="PT-1234",
        ), headers=_AUTH_HEADER)
        lines = [line for line in tmp_reservations_file.read_text(encoding="utf-8").splitlines()
                 if "Pipe Test" in line]
        assert len(lines) == 1
        parts = lines[0].split(" | ")
        assert len(parts) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPInjectionDefence
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPInjectionDefence:
    def test_pipe_in_full_name_is_stripped(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(full_name="Evil | Name"), headers=_AUTH_HEADER)
        content = tmp_reservations_file.read_text(encoding="utf-8")
        # The pipe in the name should be replaced with '-'
        data_lines = [line for line in content.splitlines() if "Evil" in line]
        assert len(data_lines) == 1
        # Only 3 real delimiter pipes expected (4 fields)
        assert data_lines[0].count(" | ") == 3

    def test_newline_in_field_is_stripped(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(
            full_name="Alice\nInjected",
        ), headers=_AUTH_HEADER)
        content = tmp_reservations_file.read_text(encoding="utf-8")
        # The injected newline should be collapsed — no blank line in the entry
        data_lines = [line for line in content.splitlines() if "Alice" in line]
        assert len(data_lines) == 1

    def test_carriage_return_stripped(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(
            full_name="Test\rUser",
        ), headers=_AUTH_HEADER)
        content = tmp_reservations_file.read_text(encoding="utf-8")
        assert "\r" not in content

    def test_pipe_in_car_number_is_replaced(self, client, tmp_reservations_file):
        client.post("/mcp", json=_tool_call_body(car_number="AB|1234"), headers=_AUTH_HEADER)
        content = tmp_reservations_file.read_text(encoding="utf-8")
        data_lines = [line for line in content.splitlines() if "AB" in line and "1234" in line]
        assert len(data_lines) == 1
        assert "AB|1234" not in data_lines[0]


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPBatchRequest
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPBatchRequest:
    def test_batch_returns_list(self, client):
        batch = [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}},
        ]
        r = client.post("/mcp", json=batch, headers=_AUTH_HEADER)
        assert isinstance(r.json(), list)

    def test_batch_returns_correct_count(self, client):
        batch = [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}},
        ]
        r = client.post("/mcp", json=batch, headers=_AUTH_HEADER)
        assert len(r.json()) == 2

    def test_batch_notifications_excluded_from_response(self, client):
        """Notifications (no id) must not appear in the batch response."""
        batch = [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},  # notification
        ]
        r = client.post("/mcp", json=batch, headers=_AUTH_HEADER)
        assert len(r.json()) == 1
        assert r.json()[0]["id"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestInputModel
# ═══════════════════════════════════════════════════════════════════════════════

class TestInputModel:
    def test_valid_input_passes(self):
        m = WriteReservationInput(
            full_name="Alice Smith",
            car_number="ABC-1234",
            reservation_period="2026-06-01 to 2026-06-07",
            approval_time="2026-05-12 10:00:00 UTC",
        )
        assert m.full_name == "Alice Smith"

    def test_pipe_sanitised_in_full_name(self):
        m = WriteReservationInput(
            full_name="Evil | Hacker",
            car_number="ABC-1234",
            reservation_period="2026-06-01 to 2026-06-07",
            approval_time="2026-05-12 10:00:00 UTC",
        )
        assert "|" not in m.full_name

    def test_newline_sanitised(self):
        m = WriteReservationInput(
            full_name="Line\nBreak",
            car_number="ABC-1234",
            reservation_period="2026-06-01 to 2026-06-07",
            approval_time="2026-05-12 10:00:00 UTC",
        )
        assert "\n" not in m.full_name

    def test_carriage_return_sanitised(self):
        m = WriteReservationInput(
            full_name="CR\rTest",
            car_number="ABC-1234",
            reservation_period="per",
            approval_time="time",
        )
        assert "\r" not in m.full_name

    def test_whitespace_stripped(self):
        m = WriteReservationInput(
            full_name="  Alice  ",
            car_number="  ABC-1234  ",
            reservation_period="  period  ",
            approval_time="  time  ",
        )
        assert m.full_name == "Alice"
        assert m.car_number == "ABC-1234"

    def test_pipe_replaced_with_dash_not_removed(self):
        m = WriteReservationInput(
            full_name="A|B",
            car_number="C",
            reservation_period="P",
            approval_time="T",
        )
        assert "A-B" == m.full_name


# ═══════════════════════════════════════════════════════════════════════════════
# TestDirectWriter
# ═══════════════════════════════════════════════════════════════════════════════

class TestDirectWriter:
    def test_creates_file(self, tmp_path, monkeypatch):
        f = tmp_path / "res.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(f))
        from src.mcp_server.client import _direct_write
        _direct_write("Alice", "ABC-1", "2026-06-01 to 2026-06-07", "2026-05-12 10:00 UTC")
        assert f.exists()

    def test_writes_header_on_first_call(self, tmp_path, monkeypatch):
        f = tmp_path / "res.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(f))
        from src.mcp_server.client import _direct_write
        _direct_write("A", "B", "P", "T")
        content = f.read_text(encoding="utf-8")
        assert "Name | Car Number" in content

    def test_appends_second_entry_without_second_header(self, tmp_path, monkeypatch):
        f = tmp_path / "res.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(f))
        from src.mcp_server.client import _direct_write
        _direct_write("Guest One", "AA-1", "P1", "T1")
        _direct_write("Guest Two", "BB-2", "P2", "T2")
        content = f.read_text(encoding="utf-8")
        assert content.count("Name | Car Number") == 1
        assert "Guest One" in content
        assert "Guest Two" in content

    def test_return_message_contains_name(self, tmp_path, monkeypatch):
        f = tmp_path / "res.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(f))
        from src.mcp_server.client import _direct_write
        msg = _direct_write("Charlie", "CC-3", "P", "T")
        assert "Charlie" in msg

    def test_injection_sanitised_in_direct_write(self, tmp_path, monkeypatch):
        f = tmp_path / "res.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(f))
        from src.mcp_server.client import _direct_write
        _direct_write("Evil|User", "AB\n1234", "P", "T")
        content = f.read_text(encoding="utf-8")
        assert "Evil|User" not in content
        assert "\n\n" not in content  # no blank line from injected newline

    def test_creates_parent_dirs(self, tmp_path, monkeypatch):
        nested = tmp_path / "deep" / "dir" / "res.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(nested))
        from src.mcp_server.client import _direct_write
        _direct_write("A", "B", "P", "T")
        assert nested.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# TestCallHelper
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallHelper:
    def test_falls_back_to_direct_write_when_server_unreachable(self, tmp_path, monkeypatch):
        """When MCPClient raises, _direct_write must be called and write the file."""
        f = tmp_path / "res.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(f))
        monkeypatch.setattr(cfg, "MCP_SERVER_PORT", 19999)  # nothing listening here

        from src.mcp_server.client import call_write_confirmed_reservation
        # Patch sleep so retries don't slow the test down
        with patch("time.sleep"):
            msg = call_write_confirmed_reservation(
                full_name="Fallback User",
                car_number="FB-1234",
                start_date="2026-08-01",
                end_date="2026-08-05",
                approval_time="2026-05-12 11:00:00 UTC",
            )
        assert f.exists()
        assert "Fallback User" in f.read_text(encoding="utf-8")
        assert "Fallback User" in msg or "direct write" in msg.lower()

    def test_approval_time_defaults_to_utc_now(self, tmp_path, monkeypatch):
        """When approval_time is omitted, a UTC timestamp is generated."""
        f = tmp_path / "res.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(f))
        monkeypatch.setattr(cfg, "MCP_SERVER_PORT", 19999)

        from src.mcp_server.client import call_write_confirmed_reservation
        with patch("time.sleep"):
            call_write_confirmed_reservation(
                full_name="NoTime User",
                car_number="NT-0001",
                start_date="2026-09-01",
                end_date="2026-09-03",
            )
        content = f.read_text(encoding="utf-8")
        assert "UTC" in content

    def test_reservation_period_formatted_correctly(self, tmp_path, monkeypatch):
        f = tmp_path / "res.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(f))
        monkeypatch.setattr(cfg, "MCP_SERVER_PORT", 19999)

        from src.mcp_server.client import call_write_confirmed_reservation
        with patch("time.sleep"):
            call_write_confirmed_reservation(
                full_name="Period Test",
                car_number="PR-1111",
                start_date="2026-10-01",
                end_date="2026-10-10",
                approval_time="2026-05-12 12:00:00 UTC",
            )
        content = f.read_text(encoding="utf-8")
        assert "2026-10-01 to 2026-10-10" in content


# ═══════════════════════════════════════════════════════════════════════════════
# TestMCPClientClass
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPClientClass:
    def test_default_base_url_uses_127_0_0_1(self, monkeypatch):
        monkeypatch.setattr(cfg, "MCP_SERVER_HOST", "localhost")
        monkeypatch.setattr(cfg, "MCP_SERVER_PORT", 5002)
        from importlib import reload

        import src.mcp_server.client as client_mod
        reload(client_mod)
        c = client_mod.MCPClient()
        assert "127.0.0.1" in c.base_url
        assert "localhost" not in c.base_url

    def test_custom_base_url_preserved(self, monkeypatch):
        monkeypatch.setattr(cfg, "MCP_SERVER_HOST", "127.0.0.1")
        monkeypatch.setattr(cfg, "MCP_SERVER_PORT", 5002)
        from src.mcp_server.client import MCPClient
        c = MCPClient(base_url="http://custom-host:9999")
        assert c.base_url == "http://custom-host:9999"

    def test_request_id_increments(self, monkeypatch):
        monkeypatch.setattr(cfg, "MCP_SERVER_HOST", "127.0.0.1")
        monkeypatch.setattr(cfg, "MCP_SERVER_PORT", 5002)
        from src.mcp_server.client import MCPClient
        c = MCPClient()
        assert c._next_id() == 1
        assert c._next_id() == 2
        assert c._next_id() == 3

    def test_health_check_returns_false_when_server_down(self, monkeypatch):
        monkeypatch.setattr(cfg, "MCP_SERVER_PORT", 19998)
        from src.mcp_server.client import MCPClient
        c = MCPClient(base_url="http://127.0.0.1:19998")
        assert c.health_check() is False

    def test_api_key_defaults_to_config(self, monkeypatch):
        monkeypatch.setattr(cfg, "MCP_API_KEY", "my-config-key")
        monkeypatch.setattr(cfg, "MCP_SERVER_HOST", "127.0.0.1")
        monkeypatch.setattr(cfg, "MCP_SERVER_PORT", 5002)
        from src.mcp_server.client import MCPClient
        c = MCPClient()
        assert c.api_key == "my-config-key"
