"""
Admin Notification Agent — the second LangChain agent.

This agent acts as the bridge between the user-facing chatbot (first agent)
and the human administrator. It:

  1. Receives a reservation request from the first agent
  2. Uses the notify_admin tool to register it and alert the administrator
     (email + REST API dashboard)
  3. Uses the poll_for_decision tool to wait for the administrator's response
  4. Returns the decision (approved / rejected + notes) to the first agent

Built using LangChain 1.x's `create_agent` (LangGraph-based tool-calling agent),
using the same LLM factory as the first agent (Groq / Gemini / OpenAI / Mock).

Public interface:
  request_admin_approval(reservation_data: dict) -> tuple[bool | None, str]
    approved = True  → admin approved
    approved = False → admin rejected (notes contains reason)
    approved = None  → timeout or error (caller should fall back to terminal)
"""
import json
import logging
from typing import Optional, Tuple

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

import src.config as cfg
from src.admin_agent.tools import get_pending_requests, notify_admin, poll_for_decision
from src.chatbot.llm import get_llm

logger = logging.getLogger(__name__)

# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the SmartPark Admin Notification Agent.

Your sole responsibility is to coordinate the approval of parking reservation
requests between the user-facing chatbot and the human administrator.

Follow these steps in order — do not skip any step:

Step 1 — Notify:
  Call the notify_admin tool with the reservation data as a valid JSON string.
  The tool registers the request and sends an email to the administrator.
  It returns the unique request code (looks like REQ-XXXXXX) and the dashboard URL.

Step 2 — Poll:
  Extract the request code from the notify_admin result.
  Call the poll_for_decision tool with that exact code.
  This tool blocks until the admin approves or rejects — do not call anything
  else while waiting.

Step 3 — Report:
  Return your final answer as EXACTLY one of these lines (nothing else):
    DECISION: approved
    DECISION: rejected | Notes: <reason>
    DECISION: timeout
"""

_TOOLS = [notify_admin, poll_for_decision, get_pending_requests]


# ─── Agent builder ─────────────────────────────────────────────────────────────

def _build_agent():
    """Build a LangChain 1.x create_agent (LangGraph-based tool-calling agent)."""
    llm = get_llm(temperature=0.0)
    return create_agent(
        model=llm,
        tools=_TOOLS,
        system_prompt=_SYSTEM_PROMPT,
    )


# ─── Public entry point ────────────────────────────────────────────────────────

def request_admin_approval(
    reservation_data: dict,
) -> Tuple[Optional[bool], str]:
    """
    Invoke the admin agent to notify the administrator and wait for their decision.

    Args:
        reservation_data: dict containing first_name, last_name, car_number,
                          zone, start_date, end_date (collected by the first agent).

    Returns:
        (approved, notes) where:
          approved = True   → admin approved the reservation
          approved = False  → admin rejected (notes contains the reason)
          approved = None   → timeout or agent error (caller should fall back)
    """
    rd_clean = {
        k: reservation_data.get(k, "")
        for k in ("first_name", "last_name", "car_number", "zone", "start_date", "end_date")
    }
    rd_json = json.dumps(rd_clean)

    input_text = (
        f"Process this parking reservation approval request:\n\n"
        f"{rd_json}\n\n"
        f"Register it, notify the administrator, wait for their decision, "
        f"then report the result."
    )

    try:
        agent = _build_agent()
        result = agent.invoke({"messages": [HumanMessage(content=input_text)]})

        # The agent returns a dict with "messages" — the last AI message is the answer
        messages = result.get("messages", [])
        output = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                output = msg.content.strip()
                break

    except Exception as exc:
        logger.error("[AdminAgent] Agent execution failed: %s", exc, exc_info=True)
        return None, "agent_error"

    # ── Parse the DECISION line ───────────────────────────────────────────────
    logger.debug("[AdminAgent] Raw output: %r", output)
    lower = output.lower()

    if "decision: approved" in lower:
        return True, ""

    if "decision: rejected" in lower:
        notes = "Rejected by administrator."
        if "notes:" in lower:
            try:
                notes = output.split("Notes:", 1)[1].strip().split("\n")[0].strip()
            except IndexError:
                pass
        return False, notes

    if "decision: timeout" in lower or "timeout" in lower:
        logger.warning("[AdminAgent] Decision timed out.")
        return None, "timeout"

    # Soft fallback: infer from keywords
    if "approved" in lower and "rejected" not in lower:
        return True, ""
    if "rejected" in lower:
        return False, "Rejected by administrator."

    logger.warning("[AdminAgent] Could not parse decision from: %r", output)
    return None, "parse_error"

