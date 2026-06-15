"""
LangGraph conversation state definition.

The TypedDict is the single source of truth for all state that flows
through the graph nodes. Using Annotated[list, add_messages] for the
`messages` field ensures LangGraph appends rather than replaces messages.
"""
from typing import Annotated, Any, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ReservationData(TypedDict, total=False):
    """Incrementally populated during the reservation flow."""
    full_name: str
    car_number: str       # vehicle registration plate
    zone: str             # preferred parking zone
    start_date: str       # ISO date string  YYYY-MM-DD
    end_date: str         # ISO date string  YYYY-MM-DD
    status: str           # collecting | validating | pending_approval | approved | rejected


class ChatInputSchema(TypedDict):
    """Minimal input schema exposed to LangGraph Studio — enables the chat UI."""
    messages: Annotated[List[Any], add_messages]


class ChatState(TypedDict, total=False):
    # ── Core conversation ────────────────────────────────────────────────────
    messages: Annotated[List[Any], add_messages]

    # ── Routing ──────────────────────────────────────────────────────────────
    intent: str           # info | reservation | greeting | off_topic

    # ── RAG context ──────────────────────────────────────────────────────────
    static_context: str   # retrieved from vector store
    dynamic_context: str  # retrieved from SQL (prices / hours / availability)

    # ── Reservation ──────────────────────────────────────────────────────────
    reservation_data: ReservationData
    current_field: str    # which field we're currently asking for

    # ── Human-in-the-loop ────────────────────────────────────────────────────
    human_approved: Optional[bool]   # None = not yet decided
    admin_notes: str

    # ── Guardrails ───────────────────────────────────────────────────────────
    guardrail_blocked: bool
    guardrail_reason: str

    # ── Evaluation metadata ───────────────────────────────────────────────────
    retrieval_latency_ms: float
    generation_latency_ms: float
