"""
LangGraph node implementations.

Each function is a pure node: it receives the current ChatState, performs
one well-defined piece of work, and returns a partial state update dict.

Node responsibilities:
  classify_intent   — route user message to the correct handler
  retrieve_context  — RAG lookup in the static vector store
  query_dynamic     — pull live prices / hours / availability from SQLite
  generate_response — call LLM to produce a response
  collect_reservation — gather reservation fields turn-by-turn
  apply_guardrails  — filter LLM output before showing to the user
"""
import logging
import time
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

import src.config as cfg
from src.chatbot.llm import get_llm
from src.chatbot.prompts import (
    GENERAL_RESPONSE_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    OFF_TOPIC_RESPONSE,
    RAG_ANSWER_PROMPT,
    RESERVATION_APPROVED_TEMPLATE,
    RESERVATION_FIELD_ORDER,
    RESERVATION_FIELD_PROMPTS,
    RESERVATION_REJECTED_TEMPLATE,
    RESERVATION_SUMMARY_TEMPLATE,
)
from src.chatbot.state import ChatState
from src.database.models import init_db
from src.database.operations import (
    create_reservation,
    format_hours_context,
    format_pricing_context,
    get_availability_summary,
    reject_reservation_record,
)
from src.guardrails.filters import filter_output
from src.rag.retriever import retrieve
from src.reservation.handler import (
    extract_field_value,
    get_next_field,
    is_complete,
    mask_card,
    mask_email,
    to_datetime,
    validate_date_range,
)

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _last_human_message(state: ChatState) -> str:
    """Return the content of the most recent HumanMessage in the state."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _ensure_db() -> None:
    init_db(cfg.SQLITE_DB_PATH)


# ─── Node 1: classify_intent ──────────────────────────────────────────────────

def classify_intent(state: ChatState) -> Dict[str, Any]:
    """
    Classify the user's intent as one of: info | reservation | greeting | off_topic.
    Skips classification if we're mid-reservation (preserves current flow).
    """
    # If we're mid-reservation (either current_field is set, or reservation_data has
    # partial data), preserve the reservation intent without calling the LLM.
    # Note: empty dict {} is falsy, so we check current_field as the primary signal.
    current_field = state.get("current_field")
    rd = state.get("reservation_data") or {}
    if current_field or (rd and not is_complete(rd)):
        return {"intent": "reservation"}

    user_msg = _last_human_message(state)
    if not user_msg:
        return {"intent": "greeting"}

    # Grab last bot message for context-aware classification
    last_bot = ""
    for msg in reversed(state.get("messages", [])):
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            last_bot = msg.content[:300]  # trim to avoid huge prompts
            break

    # Short affirmative after a reservation-offering bot message → reservation
    _aff = {"yes", "sure", "yep", "yeah", "ok", "okay", "please", "ok please",
            "yes please", "go ahead", "do it", "i do", "make a reservation",
            "book", "reserve", "i want to", "let's do it", "let's go"}
    _reservation_trigger_words = ["reserv", "book", "want to make", "pre-book",
                                   "guarantee", "would you like to make"]
    if user_msg.strip().lower() in _aff and any(
        w in last_bot.lower() for w in _reservation_trigger_words
    ):
        return {"intent": "reservation"}

    llm = get_llm(temperature=0.0)
    chain = INTENT_CLASSIFICATION_PROMPT | llm | StrOutputParser()

    try:
        intent = chain.invoke({
            "user_message": user_msg,
            "last_bot_message": last_bot,
        }).strip().lower()
    except Exception as exc:
        logger.warning("Intent classification failed (%s). Defaulting to 'info'.", exc)
        intent = "info"

    # Normalise
    if intent not in {"info", "reservation", "greeting", "off_topic"}:
        # Keyword fallback
        low = user_msg.lower()
        if any(w in low for w in ["reserve", "book", "reservation", "booking", "park a car"]):
            intent = "reservation"
        elif any(w in low for w in ["hi", "hello", "hey", "thanks", "bye", "goodbye"]):
            intent = "greeting"
        else:
            intent = "info"

    logger.debug("Intent classified as: %s", intent)
    return {"intent": intent}


# ─── Node 2: retrieve_context ──────────────────────────────────────────────────

def retrieve_context(state: ChatState) -> Dict[str, Any]:
    """Retrieve relevant documents from the static knowledge vector store."""
    user_msg = _last_human_message(state)
    t0 = time.perf_counter()
    result = retrieve(user_msg, k=cfg.TOP_K_DOCUMENTS)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "static_context": result.formatted_context,
        "retrieval_latency_ms": latency_ms,
    }


# ─── Node 3: query_dynamic ────────────────────────────────────────────────────

def query_dynamic(state: ChatState) -> Dict[str, Any]:
    """Pull live prices, working hours, and availability from SQLite."""
    _ensure_db()
    try:
        pricing_text = format_pricing_context(cfg.SQLITE_DB_PATH)
        hours_text = format_hours_context(cfg.SQLITE_DB_PATH)
        availability = get_availability_summary(cfg.SQLITE_DB_PATH)

        avail_lines = ["Current Availability:"]
        zone_labels = {
            "A": "Zone A (Premium)", "B": "Zone B (Standard)",
            "C": "Zone C (Economy)", "D": "Zone D (EV)",
            "E": "Zone E (Disabled)",
        }
        for zone, data in sorted(availability.items()):
            label = zone_labels.get(zone, f"Zone {zone}")
            avail_lines.append(
                f"  {label}: {data['available']} of {data['total']} spaces available"
            )

        dynamic_ctx = "\n".join([pricing_text, "\n", hours_text, "\n", "\n".join(avail_lines)])
    except Exception as exc:
        logger.warning("Dynamic DB query failed: %s", exc)
        dynamic_ctx = "Live operational data is temporarily unavailable."

    return {"dynamic_context": dynamic_ctx}


# ─── Node 4: generate_response ────────────────────────────────────────────────

def generate_response(state: ChatState) -> Dict[str, Any]:
    """Call the LLM to produce the final user-facing response."""
    intent = state.get("intent", "info")
    user_msg = _last_human_message(state)
    llm = get_llm()
    t0 = time.perf_counter()

    try:
        if intent == "info":
            chain = RAG_ANSWER_PROMPT | llm | StrOutputParser()
            response = chain.invoke({
                "user_message": user_msg,
                "static_context": state.get("static_context", "No context retrieved."),
                "dynamic_context": state.get("dynamic_context", "No live data."),
            })
        elif intent == "off_topic":
            response = OFF_TOPIC_RESPONSE
        else:
            chain = GENERAL_RESPONSE_PROMPT | llm | StrOutputParser()
            response = chain.invoke({"user_message": user_msg})

    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        response = (
            "I'm sorry, I'm having trouble generating a response right now. "
            "Please try again or call +1 (555) 123-4567 for assistance."
        )

    gen_latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "messages": [AIMessage(content=response)],
        "generation_latency_ms": gen_latency_ms,
    }


# ─── Helpers for collect_reservation ─────────────────────────────────────────

_ZONE_DETAILS = {
    "A": ("Premium, ground floor", "$6/hr"),
    "B": ("Standard, levels 2–3", "$3/hr"),
    "C": ("Economy, level 4", "$2.50/hr"),
    "D": ("EV charging, rooftop", "$3/hr + charging"),
    "E": ("Disabled/accessible", "free with badge"),
}


def _build_zone_availability_prompt(all_avail: dict, start_date: str, end_date: str) -> str:
    """Build the zone-selection prompt with live availability for the given date range."""
    lines = [
        f"Which **parking zone** would you prefer for **{start_date} → {end_date}**?\n"
    ]
    for z, (desc, price) in _ZONE_DETAILS.items():
        n = all_avail.get(z, 0)
        avail_str = f"✅ {n} spot(s) available" if n > 0 else "❌ Fully booked"
        lines.append(f"  • **Zone {z}** — {desc} ({price}) — {avail_str}")
    lines.append("\nType the zone letter (A, B, C, D, or E):")
    return "\n".join(lines)


# ─── Node 5: collect_reservation ──────────────────────────────────────────────

def collect_reservation(state: ChatState) -> Dict[str, Any]:
    """
    Multi-turn reservation data collector.

    On each invocation:
      1. Determine which field we were collecting (current_field).
      2. Extract that field's value from the latest human message.
      3. Update reservation_data.
      4. Determine the next missing field.
      5. Ask the user for the next field (or show summary if complete).
    """
    user_msg = _last_human_message(state)
    rd: dict = dict(state.get("reservation_data") or {})
    current_field = state.get("current_field")

    # ── Extract value for the field we were asking about ──────────────────────
    # Only extract if the bot has already asked for this field (i.e., at least one
    # AIMessage exists in history). Without this guard, the trigger phrase that
    # started the reservation flow ("I want to book a space") would be mistakenly
    # extracted as the first_name value.
    has_ai_messages = any(isinstance(m, AIMessage) for m in state.get("messages", []))
    extraction_error: str | None = None
    if current_field and user_msg and has_ai_messages:
        value, error = extract_field_value(current_field, user_msg)
        if error:
            extraction_error = error
        else:
            rd[current_field] = value

            # Extra validation: after end_date is set, check date range
            if current_field == "end_date":
                valid, date_error = validate_date_range(rd)
                if not valid:
                    del rd["end_date"]
                    extraction_error = date_error
                elif "zone" in rd and "start_date" in rd:
                    # Check zone capacity for the chosen date range
                    from src.database.operations import (
                        count_available_spaces_for_dates,
                        get_all_zones_availability_for_dates,
                    )
                    s_dt = to_datetime(rd["start_date"])
                    e_dt = to_datetime(rd["end_date"])
                    avail = count_available_spaces_for_dates(
                        cfg.SQLITE_DB_PATH, rd["zone"], s_dt, e_dt
                    )
                    if avail == 0:
                        failed_zone = rd.pop("zone")
                        all_avail = get_all_zones_availability_for_dates(
                            cfg.SQLITE_DB_PATH, s_dt, e_dt
                        )
                        zone_lines = "\n".join(
                            f"  \u2022 Zone {z}: {'\u2705 ' + str(n) + ' spot(s) available' if n > 0 else '\u274c Fully booked'}"
                            for z, n in sorted(all_avail.items())
                        )
                        extraction_error = (
                            f"\u274c **Zone {failed_zone} is fully booked** for "
                            f"{rd['start_date']} \u2192 {rd['end_date']}.\n\n"
                            f"**Spots available for those dates:**\n{zone_lines}\n\n"
                            "Please choose an available zone."
                        )
                elif "zone" in rd and "start_date" in rd:
                    # Check zone capacity for the chosen date range
                    from src.database.operations import (
                        count_available_spaces_for_dates,
                        get_all_zones_availability_for_dates,
                    )
                    s_dt = to_datetime(rd["start_date"])
                    e_dt = to_datetime(rd["end_date"])
                    avail = count_available_spaces_for_dates(
                        cfg.SQLITE_DB_PATH, rd["zone"], s_dt, e_dt
                    )
                    if avail == 0:
                        failed_zone = rd.pop("zone")
                        all_avail = get_all_zones_availability_for_dates(
                            cfg.SQLITE_DB_PATH, s_dt, e_dt
                        )
                        zone_lines = "\n".join(
                            f"  • Zone {z}: {'\u2705 ' + str(n) + ' spot(s) available' if n > 0 else '\u274c Fully booked'}"
                            for z, n in sorted(all_avail.items())
                        )
                        extraction_error = (
                            f"\u274c **Zone {failed_zone} is fully booked** for "
                            f"{rd['start_date']} \u2192 {rd['end_date']}.\n\n"
                            f"**Spots available for those dates:**\n{zone_lines}\n\n"
                            "Please choose an available zone."
                        )

    # ── Determine next field ──────────────────────────────────────────────────
    next_field = get_next_field(rd)

    # ── Compose response ──────────────────────────────────────────────────────
    if extraction_error:
        # When zone was cleared due to full booking, re-ask zone with live availability
        zone_was_cleared = (
            current_field == "end_date"
            and "zone" not in rd
            and rd.get("start_date")
            and rd.get("end_date")
        )
        if zone_was_cleared:
            from src.database.operations import get_all_zones_availability_for_dates
            all_avail = get_all_zones_availability_for_dates(
                cfg.SQLITE_DB_PATH,
                to_datetime(rd["start_date"]),
                to_datetime(rd["end_date"]),
            )
            resp = extraction_error + "\n\n" + _build_zone_availability_prompt(
                all_avail, rd["start_date"], rd["end_date"]
            )
            return {
                "reservation_data": rd,
                "current_field": "zone",
                "messages": [AIMessage(content=resp)],
            }
        # Normal re-ask for the same field
        resp = f"\u26a0\ufe0f {extraction_error}\n\n{RESERVATION_FIELD_PROMPTS.get(current_field, '')}"
        try:
            resp = resp.format(**rd)
        except KeyError:
            pass
        return {
            "reservation_data": rd,
            "current_field": current_field,
            "messages": [AIMessage(content=resp)],
        }

    if next_field:
        # When re-asking zone and dates already known, show live availability
        if next_field == "zone" and rd.get("start_date") and rd.get("end_date"):
            from src.database.operations import get_all_zones_availability_for_dates
            all_avail = get_all_zones_availability_for_dates(
                cfg.SQLITE_DB_PATH,
                to_datetime(rd["start_date"]),
                to_datetime(rd["end_date"]),
            )
            prompt_text = _build_zone_availability_prompt(
                all_avail, rd["start_date"], rd["end_date"]
            )
        else:
            prompt_template = RESERVATION_FIELD_PROMPTS.get(next_field, f"Please provide your {next_field}:")
            try:
                prompt_text = prompt_template.format(**rd)
            except KeyError:
                prompt_text = prompt_template
        return {
            "reservation_data": rd,
            "current_field": next_field,
            "messages": [AIMessage(content=prompt_text)],
        }

    # ── All fields collected — show summary ───────────────────────────────────
    # Calculate duration and estimated cost
    try:
        from datetime import datetime as _dt
        _start = _dt.strptime(rd["start_date"], "%Y-%m-%d").date()
        _end   = _dt.strptime(rd["end_date"],   "%Y-%m-%d").date()
        _days  = max(1, (_end - _start).days)
    except Exception:
        _days = 1
    try:
        from src.database.operations import get_pricing_for_zone
        _pricing = get_pricing_for_zone(cfg.SQLITE_DB_PATH, rd.get("zone", ""))
        if _pricing and _pricing.daily_max and _pricing.daily_max > 0:
            _cost = _days * _pricing.daily_max
            _cost_str = f"${_cost:,.2f} (${_pricing.daily_max:.2f}/day × {_days} day(s))"
        else:
            _cost_str = "Free (disability badge zone)"
    except Exception:
        _cost_str = "N/A"
    rd_display = {
        **rd,
        "email_masked": mask_email(rd.get("email", "")),
        "card_masked": mask_card(rd.get("card_number", "")),
        "days": _days,
        "total_cost": _cost_str,
    }
    summary = RESERVATION_SUMMARY_TEMPLATE.format(**rd_display)
    rd["status"] = "pending_approval"
    return {
        "reservation_data": rd,
        "current_field": None,
        "messages": [AIMessage(content=summary)],
    }


# ─── Node 6: human_approval (interrupt node) ──────────────────────────────────

def human_approval(state: ChatState) -> Dict[str, Any]:
    """
    Agent 2 integration + HITL pause.

    Step 1 — Agent 2 (notify_admin tool):
      Registers the reservation in the shared decision store and sends an
      email alert to the administrator.  This runs *inside* the graph node
      so it happens regardless of which frontend (CLI, web, Studio) is used.

    Step 2 — interrupt():
      Pauses graph execution.  The request_code generated in Step 1 is
      included in the interrupt payload so callers can poll without
      re-registering the request.

    Step 3 — resume (caller side):
      graph.invoke(Command(resume={"approved": bool, "notes": str}), config=…)
    """
    import json
    from langgraph.types import interrupt
    from src.admin_agent.tools import notify_admin

    rd = state.get("reservation_data") or {}

    # ── Step 1: Agent 2 — register & notify ──────────────────────────────────
    rd_for_notify = {
        k: rd.get(k, "")
        for k in ("first_name", "last_name", "car_number", "zone", "start_date", "end_date")
    }
    request_code = ""
    try:
        notify_result = notify_admin.invoke({"reservation_json": json.dumps(rd_for_notify)})
        # notify_admin returns "Request registered with code: REQ-XXXXXX\n..."
        for line in notify_result.splitlines():
            if line.startswith("Request registered with code:"):
                request_code = line.split(":", 1)[1].strip()
                break
        logger.info("[HumanApproval] Agent 2 registered request %s", request_code)
    except Exception as exc:
        # Graceful fallback: register directly so the pipeline never stalls
        logger.warning("[HumanApproval] Agent 2 notify_admin failed (%s) — falling back.", exc)
        from src.admin_agent import decision_store as _ds
        request_code = _ds.generate_request_code()
        _ds.add_pending(request_code, rd_for_notify)

    # ── Step 2: interrupt — pause until admin resumes ─────────────────────────
    decision = interrupt({
        "type": "admin_approval_required",
        "request_code": request_code,
        "message": (
            f"Please approve or reject this parking reservation:\n"
            f"  Name:    {rd.get('first_name', '')} {rd.get('last_name', '')}\n"
            f"  Vehicle: {rd.get('car_number', '')}\n"
            f"  Zone:    Zone {rd.get('zone', '')}\n"
            f"  From:    {rd.get('start_date', '')}  →  {rd.get('end_date', '')}"
        ),
        "reservation": rd,
        "instructions": 'Resume with: {"approved": true/false, "notes": "reason"}',
    })

    # decision = {"approved": True/False, "notes": "..."} from Command(resume=...)
    return {
        "human_approved": bool(decision.get("approved", False)),
        "admin_notes": decision.get("notes", ""),
    }


# ─── Node 7: finalize_reservation ────────────────────────────────────────────

def finalize_reservation(state: ChatState) -> Dict[str, Any]:
    """
    Create or reject the reservation in SQLite based on admin decision,
    then produce a final message for the user.
    """
    _ensure_db()
    rd = state.get("reservation_data") or {}
    approved = state.get("human_approved")
    admin_notes = state.get("admin_notes", "")

    if approved:
        try:
            reservation = create_reservation(
                db_path=cfg.SQLITE_DB_PATH,
                first_name=rd.get("first_name", ""),
                last_name=rd.get("last_name", ""),
                car_number=rd.get("car_number", ""),
                zone=rd.get("zone", "B"),
                start_datetime=to_datetime(rd["start_date"]),
                end_datetime=to_datetime(rd["end_date"]),
                email=rd.get("email", ""),
                card_number=mask_card(rd.get("card_number", "")),
                admin_notes=admin_notes,
            )
            try:
                from datetime import datetime as _dt2
                _s2 = _dt2.strptime(rd["start_date"], "%Y-%m-%d").date()
                _e2 = _dt2.strptime(rd["end_date"],   "%Y-%m-%d").date()
                _days2 = max(1, (_e2 - _s2).days)
            except Exception:
                _days2 = 1
            try:
                from src.database.operations import get_pricing_for_zone
                _p2 = get_pricing_for_zone(cfg.SQLITE_DB_PATH, rd.get("zone", ""))
                if _p2 and _p2.daily_max and _p2.daily_max > 0:
                    _cost2_str = f"${_days2 * _p2.daily_max:,.2f} (${_p2.daily_max:.2f}/day × {_days2} day(s))"
                else:
                    _cost2_str = "Free (disability badge zone)"
            except Exception:
                _cost2_str = "N/A"
            resp = RESERVATION_APPROVED_TEMPLATE.format(
                code=reservation.reservation_code,
                first_name=rd.get("first_name", ""),
                last_name=rd.get("last_name", ""),
                car_number=rd.get("car_number", ""),
                zone=rd.get("zone", ""),
                start_date=rd.get("start_date", ""),
                end_date=rd.get("end_date", ""),
                days=_days2,
                total_cost=_cost2_str,
                card_masked=mask_card(rd.get("card_number", "")),
            )
            rd["status"] = "approved"

            # ── Persist to MCP server text log ────────────────────────────────
            try:
                from datetime import datetime, timezone
                from src.mcp_server.client import call_write_confirmed_reservation
                call_write_confirmed_reservation(
                    full_name=f"{rd.get('first_name', '')} {rd.get('last_name', '')}".strip(),
                    car_number=rd.get("car_number", ""),
                    start_date=rd.get("start_date", ""),
                    end_date=rd.get("end_date", ""),
                    approval_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                )
            except Exception as mcp_exc:
                # Non-fatal — reservation is already saved in the DB
                logger.warning("[Nodes] MCP reservation log write failed: %s", mcp_exc)

        except Exception as exc:
            logger.error("Failed to create reservation in DB: %s", exc)
            resp = (
                "✅ Your reservation has been approved by our administrator! "
                "However, we encountered a technical issue saving it. "
                "Please contact us at +1 (555) 123-4567 to confirm. We apologise for the inconvenience."
            )
    else:
        reason = admin_notes or "The requested time slot or zone is unavailable."
        try:
            reject_reservation_record(cfg.SQLITE_DB_PATH, rd, admin_notes=reason)
        except Exception:
            pass
        resp = RESERVATION_REJECTED_TEMPLATE.format(reason=reason)
        rd["status"] = "rejected"

    return {
        "reservation_data": rd,
        "human_approved": None,   # Reset for next reservation
        "messages": [AIMessage(content=resp)],
    }


# ─── Node 8: apply_guardrails ─────────────────────────────────────────────────

def apply_guardrails(state: ChatState) -> Dict[str, Any]:
    """
    Scan the most recent AI message for PII or sensitive data before delivery.
    If the output is unsafe, replace it with a safe fallback.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return {}

    result = filter_output(last_msg.content)

    if not result.is_safe or result.filtered_output != last_msg.content:
        # Replace the message with the filtered version (LangGraph append semantics)
        # We rebuild the message list up to and excluding the last AIMessage,
        # then append the cleaned version.
        safe_msg = AIMessage(content=result.filtered_output)
        return {
            "messages": [safe_msg],   # add_messages will append this
            "guardrail_blocked": not result.is_safe,
            "guardrail_reason": "; ".join(result.warnings),
        }

    return {"guardrail_blocked": False, "guardrail_reason": ""}
