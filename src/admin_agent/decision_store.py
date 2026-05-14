"""
Thread-safe in-memory store for reservation approval requests.

Two separate dicts:
  _pending   : code → reservation_data dict   (awaiting admin decision)
  _decisions : code → {approved, notes}        (decisions submitted by admin)

The Flask API and the polling tool both access this module directly,
so all mutations are protected by a single threading.Lock.
"""
import random
import string
import threading
from typing import Optional

_lock = threading.Lock()
_pending: dict[str, dict] = {}    # request_code → reservation data
_decisions: dict[str, dict] = {}  # request_code → {"approved": bool, "notes": str}


def generate_request_code() -> str:
    """Generate a unique request reference like REQ-A1B2C3."""
    return "REQ-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def add_pending(code: str, reservation_data: dict) -> None:
    """Register a new reservation request as pending admin decision."""
    with _lock:
        _pending[code] = dict(reservation_data)


def list_pending() -> dict:
    """Return a snapshot of all pending requests."""
    with _lock:
        return dict(_pending)


def get_reservation(code: str) -> Optional[dict]:
    """Return details of a single pending request, or None if not found."""
    with _lock:
        return dict(_pending[code]) if code in _pending else None


def submit_decision(code: str, approved: bool, notes: str = "") -> bool:
    """
    Record an admin decision and remove the request from the pending queue.
    Returns True if successful, False if the code was not in the pending queue.
    """
    with _lock:
        if code not in _pending:
            return False
        _decisions[code] = {"approved": approved, "notes": notes}
        del _pending[code]
        return True


def get_decision(code: str) -> Optional[dict]:
    """Return the decision dict {approved, notes} or None if not yet decided."""
    with _lock:
        return dict(_decisions[code]) if code in _decisions else None


def clear_decision(code: str) -> None:
    """Remove a decision from the store after it has been consumed."""
    with _lock:
        _decisions.pop(code, None)
