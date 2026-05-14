"""
LangChain tools for the Admin Notification Agent.

Three tools expose the admin workflow to the LangChain agent:
  notify_admin         — register request, send email, return code + dashboard URL
  poll_for_decision    — poll the decision store until admin responds or timeout
  get_pending_requests — list all currently pending requests (for inspection)
"""
import json
import logging
import time

from langchain_core.tools import tool

import src.config as cfg
from src.admin_agent import decision_store
from src.admin_agent.notification import send_notification

logger = logging.getLogger(__name__)


@tool
def notify_admin(reservation_json: str) -> str:
    """
    Register a parking reservation approval request and notify the administrator.

    Sends an email notification (if SMTP is configured) and registers the request
    in the REST API dashboard so the admin can approve or reject it.

    Args:
        reservation_json: JSON string containing reservation fields:
            first_name, last_name, car_number, zone, start_date, end_date

    Returns:
        A string with the unique request code and the admin dashboard URL.
    """
    try:
        rd = json.loads(reservation_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"ERROR: Invalid reservation JSON — {exc}"

    code = decision_store.generate_request_code()
    decision_store.add_pending(code, rd)

    api_url = f"http://{cfg.ADMIN_API_HOST}:{cfg.ADMIN_API_PORT}"
    dashboard_url = f"{api_url}/admin"

    email_sent = send_notification(code, rd, api_url)

    lines = [
        f"Request registered with code: {code}",
        f"Admin dashboard: {dashboard_url}",
        f"Email notification: {'sent to ' + cfg.ADMIN_EMAIL if email_sent else 'skipped (SMTP not configured)'}",
        f"Waiting for admin to decide at: {dashboard_url}",
    ]
    return "\n".join(lines)


@tool
def poll_for_decision(request_code: str) -> str:
    """
    Poll the decision store until the administrator approves or rejects the request.

    Checks every 5 seconds until a decision is recorded or the timeout is reached.

    Args:
        request_code: The unique request code returned by notify_admin.

    Returns:
        One of:
          "approved"                    — admin approved
          "rejected | Notes: <reason>"  — admin rejected with optional reason
          "timeout | <message>"         — no decision within the configured timeout
    """
    timeout_seconds = cfg.ADMIN_DECISION_TIMEOUT
    poll_interval = 5
    elapsed = 0

    dashboard_url = f"http://{cfg.ADMIN_API_HOST}:{cfg.ADMIN_API_PORT}/admin"
    logger.info(
        "[AdminAgent] Polling for decision on %s (timeout=%ds, dashboard=%s)",
        request_code, timeout_seconds, dashboard_url,
    )

    while elapsed < timeout_seconds:
        decision = decision_store.get_decision(request_code)
        if decision is not None:
            decision_store.clear_decision(request_code)
            if decision["approved"]:
                return "approved"
            notes = decision.get("notes", "") or "No reason provided."
            return f"rejected | Notes: {notes}"

        time.sleep(poll_interval)
        elapsed += poll_interval

    return (
        f"timeout | No admin decision received within {timeout_seconds} seconds. "
        f"Admin dashboard: {dashboard_url}"
    )


@tool
def get_pending_requests() -> str:
    """
    Return all currently pending reservation approval requests as a JSON string.
    Useful for the admin to see what is waiting for their review.
    """
    pending = decision_store.list_pending()
    if not pending:
        return "No pending reservation requests at this time."
    return json.dumps(pending, indent=2, default=str)
