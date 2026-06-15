"""
LangGraph StateGraph definition for the SmartPark Chatbot.

Flow overview:
  START
    │
    ▼
  classify_intent
    │
    ├─── "info"        ──► retrieve_context ──► query_dynamic ──► generate_response ──► apply_guardrails ──► END
    │
    ├─── "greeting"    ──────────────────────────────────────────► generate_response ──► apply_guardrails ──► END
    │
    ├─── "blocked"   ──────────────────────────────────────────► apply_guardrails ──► END
    │
    └─── "reservation" ──► collect_reservation
                                │
                                ├─ still_collecting ──────────────────────────────────────────────────────► apply_guardrails ──► END
                                │
                                └─ pending_approval ──► [INTERRUPT] human_approval ──► finalize_reservation ──► apply_guardrails ──► END

Human-in-the-loop:
  The graph is compiled with interrupt_before=["human_approval"].
  When the graph reaches that node, execution pauses.
  The administrator reviews the reservation data, then calls:
      graph.update_state(config, {"human_approved": True/False, "admin_notes": "..."})
      graph.invoke(None, config=config)
  to resume and finalize.
"""
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.chatbot.nodes import (
    apply_guardrails,
    classify_intent,
    collect_reservation,
    finalize_reservation,
    generate_response,
    human_approval,
    query_dynamic,
    retrieve_context,
)
from src.chatbot.state import ChatInputSchema, ChatState

logger = logging.getLogger(__name__)


# ─── Edge routing functions ────────────────────────────────────────────────────

def _route_intent(state: ChatState) -> str:
    """Choose the next node based on the classified intent."""
    intent = state.get("intent", "info")
    if intent == "info":
        return "retrieve_context"
    if intent == "reservation":
        return "collect_reservation"
    if intent == "blocked":
        return "apply_guardrails"
    # greeting, off_topic, or anything else → direct LLM response
    return "generate_response"


def _route_after_collection(state: ChatState) -> str:
    """
    After collect_reservation runs:
      - If reservation is pending admin approval → go to human_approval (INTERRUPT)
      - Otherwise → apply guardrails and end the turn (wait for next user message)
    """
    rd = state.get("reservation_data") or {}
    if rd.get("status") == "pending_approval":
        return "human_approval"
    return "apply_guardrails"


# ─── Graph builder ────────────────────────────────────────────────────────────

_UNSET = object()  # sentinel: "caller did not supply a checkpointer"


def build_graph(checkpointer=_UNSET):
    """
    Build and compile the LangGraph. Returns the compiled graph.

    Args:
        checkpointer: A LangGraph checkpointer for state persistence.
                      Pass ``None`` to compile without any checkpointer
                      (required by LangGraph Platform / Studio, which
                      manages persistence itself).
                      Defaults to MemorySaver (in-memory) for local CLI use.
    """
    if checkpointer is _UNSET:
        checkpointer = MemorySaver()

    workflow = StateGraph(ChatState, input=ChatInputSchema)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("classify_intent",    classify_intent)
    workflow.add_node("retrieve_context",   retrieve_context)
    workflow.add_node("query_dynamic",      query_dynamic)
    workflow.add_node("generate_response",  generate_response)
    workflow.add_node("collect_reservation", collect_reservation)
    workflow.add_node("human_approval",     human_approval)
    workflow.add_node("finalize_reservation", finalize_reservation)
    workflow.add_node("apply_guardrails",   apply_guardrails)

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.add_edge(START, "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "retrieve_context":    "retrieve_context",
            "collect_reservation": "collect_reservation",
            "generate_response":   "generate_response",
            "apply_guardrails":    "apply_guardrails",
        },
    )

    # Info path
    workflow.add_edge("retrieve_context",  "query_dynamic")
    workflow.add_edge("query_dynamic",     "generate_response")
    workflow.add_edge("generate_response", "apply_guardrails")
    workflow.add_edge("apply_guardrails",  END)

    # Reservation path
    workflow.add_conditional_edges(
        "collect_reservation",
        _route_after_collection,
        {
            "human_approval":  "human_approval",
            "apply_guardrails": "apply_guardrails",
        },
    )
    workflow.add_edge("human_approval",      "finalize_reservation")
    workflow.add_edge("finalize_reservation", "apply_guardrails")

    # ── Compile ───────────────────────────────────────────────────────────────
    # human_approval uses langgraph.types.interrupt() internally, so no
    # interrupt_before is needed — the node handles its own pause.
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    graph = workflow.compile(**compile_kwargs)
    logger.info("LangGraph compiled successfully.")
    return graph


# ─── Singleton ────────────────────────────────────────────────────────────────

_graph_instance = None


def get_graph(force_rebuild: bool = False):
    """Return a cached compiled graph."""
    global _graph_instance
    if _graph_instance is None or force_rebuild:
        _graph_instance = build_graph()
    return _graph_instance
