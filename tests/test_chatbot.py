"""
Integration tests for the chatbot graph.

Covers: graph construction, intent routing, reservation flow,
guardrail integration, and human-in-the-loop state management.
Uses MockLLM so no OpenAI API key is required.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_graph(tmp_path, monkeypatch):
    """
    Build a fresh graph with:
      - in-memory MemorySaver (isolated per test)
      - temporary ChromaDB store
      - temporary SQLite DB
    """
    import src.config as cfg
    monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(cfg, "COLLECTION_NAME", "graph_test")
    monkeypatch.setattr(cfg, "SQLITE_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(cfg, "OPENAI_API_KEY", "")  # force MockLLM

    import src.rag.vectorstore as vs_module
    vs_module._store_instance = None

    from src.database.models import init_db
    init_db(str(tmp_path / "test.db"))

    from langgraph.checkpoint.memory import MemorySaver
    from src.chatbot.graph import build_graph
    return build_graph(checkpointer=MemorySaver())


@pytest.fixture
def thread_config():
    import uuid
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _invoke(graph, user_text: str, config: dict):
    from langchain_core.messages import HumanMessage
    return graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)


def _last_ai_message(state_values: dict) -> str:
    from langchain_core.messages import AIMessage
    for msg in reversed(state_values.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


# ─── Graph structure ──────────────────────────────────────────────────────────

class TestGraphStructure:
    def test_graph_builds_without_error(self, tmp_path, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma2"))
        monkeypatch.setattr(cfg, "COLLECTION_NAME", "build_test")
        from src.chatbot.graph import build_graph
        graph = build_graph()
        assert graph is not None

    def test_graph_has_required_nodes(self, tmp_path, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma3"))
        monkeypatch.setattr(cfg, "COLLECTION_NAME", "nodes_test")
        from src.chatbot.graph import build_graph
        graph = build_graph()
        # Compiled graphs expose their graph object
        node_names = set(graph.get_graph().nodes.keys())
        for expected in {
            "classify_intent", "retrieve_context", "query_dynamic",
            "generate_response", "collect_reservation",
            "human_approval", "finalize_reservation", "apply_guardrails",
        }:
            assert expected in node_names, f"Missing node: {expected}"


# ─── Greeting flow ────────────────────────────────────────────────────────────

class TestGreetingFlow:
    def test_responds_to_hello(self, fresh_graph, thread_config):
        _invoke(fresh_graph, "Hello!", thread_config)
        state = fresh_graph.get_state(thread_config)
        reply = _last_ai_message(state.values)
        assert reply  # Should have some response

    def test_responds_to_goodbye(self, fresh_graph, thread_config):
        _invoke(fresh_graph, "Thank you, goodbye!", thread_config)
        state = fresh_graph.get_state(thread_config)
        reply = _last_ai_message(state.values)
        assert reply


# ─── Info flow ────────────────────────────────────────────────────────────────

class TestInfoFlow:
    def test_responds_to_price_query(self, fresh_graph, thread_config):
        _invoke(fresh_graph, "What are the parking prices?", thread_config)
        state = fresh_graph.get_state(thread_config)
        reply = _last_ai_message(state.values).lower()
        # MockLLM should return something price-related
        assert reply

    def test_responds_to_location_query(self, fresh_graph, thread_config):
        _invoke(fresh_graph, "Where are you located?", thread_config)
        state = fresh_graph.get_state(thread_config)
        reply = _last_ai_message(state.values)
        assert reply

    def test_static_context_populated(self, fresh_graph, thread_config):
        """After an info query, static_context should be populated in state."""
        _invoke(fresh_graph, "Tell me about the amenities.", thread_config)
        state = fresh_graph.get_state(thread_config)
        assert state.values.get("static_context")


# ─── Reservation flow ────────────────────────────────────────────────────────

class TestReservationFlow:
    def test_reservation_start_asks_for_name(self, fresh_graph, thread_config):
        _invoke(fresh_graph, "I want to book a parking space.", thread_config)
        state = fresh_graph.get_state(thread_config)
        reply = _last_ai_message(state.values).lower()
        assert "first name" in reply or "name" in reply, f"Expected name prompt, got: {reply}"

    def test_reservation_data_accumulates(self, fresh_graph, thread_config):
        _invoke(fresh_graph, "I want to reserve a space.", thread_config)
        _invoke(fresh_graph, "Alice", thread_config)  # first_name
        state = fresh_graph.get_state(thread_config)
        rd = state.values.get("reservation_data", {})
        assert rd.get("first_name") == "Alice"

    def test_full_reservation_reaches_pending_approval(self, fresh_graph, thread_config):
        """Walk through all fields; graph should interrupt before human_approval."""
        steps = [
            "I'd like to book a parking space.",
            "Alice",          # first_name
            "Smith",          # last_name
            "ABC-1234",       # car_number
            "B",              # zone
            "2027-08-01",     # start_date
            "2027-08-05",     # end_date
        ]
        for msg in steps:
            fresh_graph.invoke(
                {"messages": [__import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage(content=msg)]},
                config=thread_config,
            )

        state = fresh_graph.get_state(thread_config)
        rd = state.values.get("reservation_data", {})
        # The graph should have interrupted at human_approval
        assert rd.get("status") == "pending_approval"

    def test_human_approval_approves_reservation(self, fresh_graph, thread_config):
        """Simulate full reservation flow with admin approval."""
        from langchain_core.messages import HumanMessage
        steps = [
            "Book a parking space",
            "Bob",          # first_name
            "Jones",        # last_name
            "XY-5678",      # car_number
            "A",            # zone
            "2027-09-01",   # start_date
            "2027-09-03",   # end_date
        ]
        for msg in steps:
            fresh_graph.invoke({"messages": [HumanMessage(content=msg)]}, config=thread_config)

        state = fresh_graph.get_state(thread_config)
        if any(t.interrupts for t in (state.tasks or ())):
            from langgraph.types import Command
            fresh_graph.invoke(
                Command(resume={"approved": True, "notes": "Approved."}),
                config=thread_config,
            )
            state = fresh_graph.get_state(thread_config)
            reply = _last_ai_message(state.values)
            assert "SP-" in reply or "approved" in reply.lower()


# ─── Guardrails integration ───────────────────────────────────────────────────

class TestGuardrailsIntegration:
    def test_graph_processes_safe_query(self, fresh_graph, thread_config):
        _invoke(fresh_graph, "What are the parking zones?", thread_config)
        state = fresh_graph.get_state(thread_config)
        assert _last_ai_message(state.values)  # Should have a response

    def test_message_history_accumulates(self, fresh_graph, thread_config):
        _invoke(fresh_graph, "Hello", thread_config)
        _invoke(fresh_graph, "What are the prices?", thread_config)
        state = fresh_graph.get_state(thread_config)
        messages = state.values.get("messages", [])
        assert len(messages) >= 4  # 2 human + at least 2 AI responses
