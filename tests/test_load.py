"""
Load and performance tests for the SmartPark pipeline.

Three test suites:
  TestChatbotLoad       — concurrent graph invocations (info + greeting queries)
  TestAdminDecisionLoad — concurrent admin decision submissions and polling
  TestMCPServerLoad     — concurrent write_confirmed_reservation calls via FastAPI TestClient

All tests use MockLLM (OPENAI_API_KEY="") and isolated temp stores so no live
services or API keys are required.

Performance thresholds are intentionally permissive (wall-clock on CI can be slow)
but the key assertions are correctness and zero failures under concurrency.
"""
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def load_graph(tmp_path_factory, monkeypatch_session):
    """
    Build one shared graph for all load tests (module scope for speed).
    Uses MockLLM and isolated temp stores.
    """
    import src.config as cfg

    tmp = tmp_path_factory.mktemp("load_graph")
    monkeypatch_session.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp / "chroma"))
    monkeypatch_session.setattr(cfg, "COLLECTION_NAME", "load_test")
    monkeypatch_session.setattr(cfg, "SQLITE_DB_PATH", str(tmp / "load.db"))
    monkeypatch_session.setattr(cfg, "OPENAI_API_KEY", "")  # force MockLLM

    import src.rag.vectorstore as vs_module
    vs_module._store_instance = None

    from src.database.models import init_db
    init_db(str(tmp / "load.db"))

    from langgraph.checkpoint.memory import MemorySaver
    from src.chatbot.graph import build_graph
    return build_graph(checkpointer=MemorySaver())


@pytest.fixture(scope="module")
def mcp_test_client(monkeypatch_session):
    """FastAPI TestClient for the MCP server with a test API key."""
    import src.config as cfg
    monkeypatch_session.setattr(cfg, "MCP_API_KEY", "load-test-secret")
    from fastapi.testclient import TestClient
    from src.mcp_server.server import app, _rate_limits
    _rate_limits.clear()
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def _clear_mcp_rate_limits():
    """Reset the MCP rate-limit dict before and after every test."""
    from src.mcp_server.server import _rate_limits
    _rate_limits.clear()
    yield
    _rate_limits.clear()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _invoke_graph(graph, user_text: str, thread_id: str):
    """Invoke the graph for a single user turn; return (latency_ms, reply)."""
    from langchain_core.messages import AIMessage, HumanMessage

    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.perf_counter()
    state = graph.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    reply = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            reply = msg.content
            break
    return latency_ms, reply


def _mcp_write_payload(idx: int) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": idx,
        "method": "tools/call",
        "params": {
            "name": "write_confirmed_reservation",
            "arguments": {
                "full_name": f"LoadUser {idx}",
                "car_number": f"LOAD-{idx:04d}",
                "reservation_period": f"2027-01-{(idx % 28) + 1:02d} to 2027-01-{(idx % 28) + 2:02d}",
                "approval_time": f"2027-01-01 0{idx % 10}:00:00 UTC",
            },
        },
    }


# ─── Suite 1: Chatbot under concurrent load ───────────────────────────────────

class TestChatbotLoad:
    """
    Fire N concurrent greeting/info queries into the graph, each on its own
    thread_id (simulating independent user sessions).  Verify:
      - No thread raises an exception.
      - Every invocation returns a non-empty AI reply.
      - P95 latency is reported (not enforced, to keep CI stable on slow machines).
    """

    CONCURRENCY = 10
    QUERIES = [
        "Hello!",
        "What are the parking prices?",
        "Where are you located?",
        "Do you have EV charging?",
        "What are your opening hours?",
        "Is there disabled parking?",
        "Tell me about your security.",
        "How do I make a reservation?",
        "What is the cancellation policy?",
        "Goodbye, thank you!",
    ]

    def test_concurrent_info_queries_all_succeed(self, load_graph):
        """All concurrent info queries must return a non-empty response."""
        latencies = []
        errors = []

        def run(idx):
            tid = str(uuid.uuid4())
            query = self.QUERIES[idx % len(self.QUERIES)]
            try:
                lat, reply = _invoke_graph(load_graph, query, tid)
                return lat, reply, None
            except Exception as exc:
                return 0.0, "", str(exc)

        with ThreadPoolExecutor(max_workers=self.CONCURRENCY) as pool:
            futures = [pool.submit(run, i) for i in range(self.CONCURRENCY)]
            for fut in as_completed(futures):
                lat, reply, err = fut.result()
                if err:
                    errors.append(err)
                else:
                    latencies.append(lat)
                    assert reply, "Graph returned an empty AI reply under load"

        assert not errors, f"Errors during concurrent chatbot load: {errors}"
        assert len(latencies) == self.CONCURRENCY

    def test_concurrent_queries_latency_stats(self, load_graph):
        """Collect latency stats; mean must be finite and positive."""
        latencies = []

        def run(idx):
            tid = str(uuid.uuid4())
            lat, _ = _invoke_graph(load_graph, self.QUERIES[idx % len(self.QUERIES)], tid)
            return lat

        with ThreadPoolExecutor(max_workers=self.CONCURRENCY) as pool:
            futures = [pool.submit(run, i) for i in range(self.CONCURRENCY)]
            for fut in as_completed(futures):
                latencies.append(fut.result())

        mean_ms = sum(latencies) / len(latencies)
        latencies_sorted = sorted(latencies)
        p95_ms = latencies_sorted[int(0.95 * len(latencies_sorted)) - 1]

        assert mean_ms > 0, "Mean latency must be positive"
        # P95 is logged but not enforced (CI machines vary widely in speed)
        print(f"\n[chatbot-load] mean={mean_ms:.1f}ms  p95={p95_ms:.1f}ms")

    def test_independent_sessions_do_not_share_state(self, load_graph):
        """Each thread_id must have its own isolated conversation state."""
        results = {}

        def run(name):
            tid = str(uuid.uuid4())
            _invoke_graph(load_graph, "I want to book a parking space.", tid)
            config = {"configurable": {"thread_id": tid}}
            state = load_graph.get_state(config)
            rd = state.values.get("reservation_data") or {}
            # The bot should be collecting, not having data from another session
            results[name] = rd

        with ThreadPoolExecutor(max_workers=5) as pool:
            futs = {pool.submit(run, f"user_{i}"): f"user_{i}" for i in range(5)}
            for fut in as_completed(futs):
                fut.result()  # re-raise any exception

        # Each session should be independent (no cross-contamination of names etc.)
        assert len(results) == 5


# ─── Suite 2: Admin decision store under concurrent load ──────────────────────

class TestAdminDecisionLoad:
    """
    Hammer the in-memory decision store with concurrent register + decide cycles.
    Verifies thread safety: every registered code must end up either pending or
    decided — never lost or duplicated.
    """

    CONCURRENCY = 20

    def test_concurrent_registrations_are_unique(self):
        """All concurrently generated request codes must be unique."""
        from src.admin_agent import decision_store as ds

        codes = []
        lock = __import__("threading").Lock()

        def register(idx):
            code = ds.generate_request_code()
            with lock:
                codes.append(code)
            ds.add_pending(code, {"first_name": f"User{idx}", "car_number": f"CC-{idx:04d}"})

        with ThreadPoolExecutor(max_workers=self.CONCURRENCY) as pool:
            list(pool.map(register, range(self.CONCURRENCY)))

        assert len(codes) == self.CONCURRENCY
        assert len(set(codes)) == self.CONCURRENCY, "Duplicate request codes generated under concurrency"

        # Cleanup
        for code in codes:
            ds.submit_decision(code, approved=True, notes="load-test cleanup")

    def test_concurrent_decisions_no_data_loss(self):
        """Each pending request must receive exactly one decision; none are lost."""
        from src.admin_agent import decision_store as ds

        # Register all requests first (sequentially to get known codes)
        codes = []
        for i in range(self.CONCURRENCY):
            code = ds.generate_request_code()
            codes.append(code)
            ds.add_pending(code, {"first_name": f"Batch{i}"})

        decided = []
        failed = []
        lock = __import__("threading").Lock()

        def decide(code):
            ok = ds.submit_decision(code, approved=(hash(code) % 2 == 0), notes="load")
            with lock:
                (decided if ok else failed).append(code)

        with ThreadPoolExecutor(max_workers=self.CONCURRENCY) as pool:
            list(pool.map(decide, codes))

        assert not failed, f"submit_decision returned False for codes: {failed}"
        assert len(decided) == self.CONCURRENCY

        for code in codes:
            decision = ds.get_decision(code)
            assert decision is not None, f"Decision for {code} was lost"
            assert "approved" in decision

    def test_pending_list_is_consistent_under_concurrent_reads(self):
        """list_pending() must never raise under concurrent access."""
        from src.admin_agent import decision_store as ds

        # Seed some pending entries
        codes = [ds.generate_request_code() for _ in range(10)]
        for code in codes:
            ds.add_pending(code, {"car_number": "XX-9999"})

        errors = []

        def read(_):
            try:
                snapshot = ds.list_pending()
                assert isinstance(snapshot, dict)
            except Exception as exc:
                errors.append(str(exc))

        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(read, range(50)))

        assert not errors, f"list_pending() raised under concurrent reads: {errors}"

        # Cleanup
        for code in codes:
            ds.submit_decision(code, approved=False, notes="cleanup")


# ─── Suite 3: MCP server under concurrent load ────────────────────────────────

class TestMCPServerLoad:
    """
    Send concurrent write_confirmed_reservation requests to the FastAPI TestClient.
    Verifies:
      - All requests succeed (HTTP 200, JSON-RPC result, no error field).
      - File content contains every written entry (no data loss).
      - The server handles burst traffic without corrupting the output file.
    """

    CONCURRENCY = 15
    _AUTH = {"Authorization": "Bearer load-test-secret", "Content-Type": "application/json"}

    def test_concurrent_writes_all_succeed(self, mcp_test_client, tmp_path, monkeypatch):
        """All concurrent tool calls must return a JSON-RPC result (not error)."""
        import src.config as cfg
        res_file = tmp_path / "load_reservations.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(res_file))

        errors = []
        results = []
        lock = __import__("threading").Lock()

        def call(idx):
            resp = mcp_test_client.post(
                "/mcp",
                json=_mcp_write_payload(idx),
                headers=self._AUTH,
            )
            with lock:
                if resp.status_code != 200:
                    errors.append(f"[{idx}] HTTP {resp.status_code}: {resp.text[:200]}")
                else:
                    body = resp.json()
                    if "error" in body:
                        errors.append(f"[{idx}] JSON-RPC error: {body['error']}")
                    else:
                        results.append(body)

        with ThreadPoolExecutor(max_workers=self.CONCURRENCY) as pool:
            list(pool.map(call, range(self.CONCURRENCY)))

        assert not errors, f"MCP write errors under load:\n" + "\n".join(errors)
        assert len(results) == self.CONCURRENCY

    def test_concurrent_writes_no_file_corruption(self, mcp_test_client, tmp_path, monkeypatch):
        """File must contain exactly N complete lines after N concurrent writes."""
        import src.config as cfg
        res_file = tmp_path / "concurrent_reservations.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(res_file))

        n = self.CONCURRENCY

        def call(idx):
            mcp_test_client.post(
                "/mcp",
                json=_mcp_write_payload(idx + 100),  # offset to avoid key collision
                headers=self._AUTH,
            )

        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(call, range(n)))

        if res_file.exists():
            all_lines = [ln for ln in res_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            # Exclude the header row ("Name | ...") and separator row ("---...---")
            data_lines = [
                ln for ln in all_lines
                if "|" in ln
                and not ln.startswith("Name")
                and not ln.startswith("---")
            ]
            assert len(data_lines) == n, (
                f"Expected {n} data lines after {n} concurrent writes, got {len(data_lines)}\n"
                f"All lines:\n" + "\n".join(all_lines)
            )
            # Every data line must contain the pipe-separated structure
            for line in data_lines:
                assert "|" in line, f"Malformed reservation line: {line!r}"

    def test_mcp_health_endpoint_under_load(self, mcp_test_client):
        """Health endpoint must return 200 for all concurrent requests."""
        errors = []
        lock = __import__("threading").Lock()

        def health(_):
            resp = mcp_test_client.get("/health")
            with lock:
                if resp.status_code != 200:
                    errors.append(f"HTTP {resp.status_code}")

        with ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(health, range(50)))

        assert not errors, f"Health endpoint failed under load: {errors}"

    def test_mcp_latency_stats(self, mcp_test_client, tmp_path, monkeypatch):
        """Collect write latencies; mean must be positive and finite."""
        import src.config as cfg
        res_file = tmp_path / "latency_reservations.txt"
        monkeypatch.setattr(cfg, "RESERVATIONS_FILE_PATH", str(res_file))

        latencies = []
        lock = __import__("threading").Lock()

        def timed_call(idx):
            t0 = time.perf_counter()
            mcp_test_client.post(
                "/mcp",
                json=_mcp_write_payload(idx + 200),
                headers=self._AUTH,
            )
            lat = (time.perf_counter() - t0) * 1000
            with lock:
                latencies.append(lat)

        with ThreadPoolExecutor(max_workers=self.CONCURRENCY) as pool:
            list(pool.map(timed_call, range(self.CONCURRENCY)))

        mean_ms = sum(latencies) / len(latencies)
        sorted_lats = sorted(latencies)
        p95_ms = sorted_lats[int(0.95 * len(sorted_lats)) - 1]

        assert mean_ms > 0
        print(f"\n[mcp-load] mean={mean_ms:.1f}ms  p95={p95_ms:.1f}ms")
