"""
Reservation business logic: field validation, extraction, and state helpers.

Keeps all reservation-domain rules outside the graph nodes to keep
nodes thin and this module independently testable.
"""
import re
from datetime import datetime, date
from typing import Optional, Tuple

from src.chatbot.prompts import RESERVATION_FIELD_ORDER


# ─── Valid zones ──────────────────────────────────────────────────────────────
VALID_ZONES = {"A", "B", "C", "D", "E"}

# ─── Car number regex (permissive — covers most international plates) ─────────
CAR_NUMBER_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-\s]{1,14}[A-Z0-9]$", re.IGNORECASE)

DATE_FORMAT = "%Y-%m-%d"


# ─── Field extraction ─────────────────────────────────────────────────────────

def extract_field_value(field: str, user_message: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to extract the value for `field` from the user's raw message.
    Returns (value, error_message).  error_message is None on success.
    """
    text = user_message.strip()

    if field in ("first_name", "last_name"):
        # Accept only letters, hyphens, apostrophes, spaces
        clean = re.sub(r"[^a-zA-Z\-' ]", "", text).strip()
        if not clean:
            return None, "That doesn't look like a valid name. Please enter letters only."
        return clean.title(), None

    if field == "car_number":
        clean = text.upper().replace(" ", "").replace(".", "")
        if not re.match(r"^[A-Z0-9\-]{2,20}$", clean):
            return None, (
                "That registration number doesn't seem valid. "
                "Please enter your plate number (e.g., ABC-1234 or XY12ABC)."
            )
        return clean, None

    if field == "zone":
        letter = text.upper().strip().replace("ZONE", "").strip()
        if letter not in VALID_ZONES:
            return None, (
                f"Zone '{text}' is not recognised. "
                "Please enter one of: A, B, C, D, or E."
            )
        return letter, None

    if field == "email":
        if not re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", text):
            return None, (
                "That doesn't look like a valid email address. "
                "Please enter a valid email, e.g. yourname@example.com."
            )
        return text.lower(), None

    if field == "card_number":
        digits_only = re.sub(r"[\s\-]", "", text)
        if not re.match(r"^\d{13,19}$", digits_only):
            return None, (
                "That doesn't look like a valid card number. "
                "Please enter 13–19 digits (you can include spaces or dashes)."
            )
        return digits_only, None

    if field in ("start_date", "end_date"):
        return _parse_date(text)

    return text, None


def mask_email(email: str) -> str:
    """Return a masked email, e.g. j***@gmail.com"""
    if not email or "@" not in email:
        return email
    local, domain = email.rsplit("@", 1)
    return local[0] + "***@" + domain


def mask_card(card_number: str) -> str:
    """Return a masked card number showing only the last 4 digits."""
    digits = re.sub(r"[\s\-]", "", card_number)
    if len(digits) < 4:
        return "****"
    return "**** **** **** " + digits[-4:]


def _parse_date(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a date string; try several common formats including natural language."""
    clean = text.strip()

    # Structured formats first
    formats = [
        "%Y-%m-%d",   # 2026-06-15
        "%d/%m/%Y",   # 15/06/2026
        "%m/%d/%Y",   # 06/15/2026
        "%d-%m-%Y",   # 15-06-2026
        "%d.%m.%Y",   # 15.06.2026
        "%B %d %Y",   # June 15 2026
        "%B %d, %Y",  # June 15, 2026
        "%d %B %Y",   # 15 June 2026
        "%d %B, %Y",  # 15 June, 2026
        "%b %d %Y",   # Jun 15 2026
        "%b %d, %Y",  # Jun 15, 2026
        "%d %b %Y",   # 15 Jun 2026
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(clean, fmt).date()
            if parsed < date.today():
                return None, "The date you entered is in the past. Please enter a future date."
            return parsed.strftime(DATE_FORMAT), None
        except ValueError:
            continue

    return None, (
        f"I couldn't understand the date \"{clean}\". "
        "You can use formats like: 2026-06-15, 15/06/2026, or June 15 2026."
    )


# ─── Completion check ─────────────────────────────────────────────────────────

def get_next_field(reservation_data: dict) -> Optional[str]:
    """Return the next field that still needs to be collected, or None if done."""
    for field in RESERVATION_FIELD_ORDER:
        if not reservation_data.get(field):
            return field
    return None


def is_complete(reservation_data: dict) -> bool:
    return get_next_field(reservation_data) is None


# ─── Date range validation ────────────────────────────────────────────────────

def validate_date_range(reservation_data: dict) -> Tuple[bool, str]:
    """Validate that start < end and duration ≤ 30 days."""
    start_str = reservation_data.get("start_date")
    end_str = reservation_data.get("end_date")
    if not start_str or not end_str:
        return True, ""  # Not enough data to validate yet

    try:
        start = datetime.strptime(start_str, DATE_FORMAT).date()
        end = datetime.strptime(end_str, DATE_FORMAT).date()
    except ValueError:
        return False, "Invalid date format in reservation data."

    if end <= start:
        return False, "End date must be after the start date."

    duration = (end - start).days
    if duration > 30:
        return False, f"Maximum reservation period is 30 days (you requested {duration} days)."

    return True, ""


# ─── Datetime conversion ──────────────────────────────────────────────────────

def to_datetime(date_str: str) -> datetime:
    """Convert a YYYY-MM-DD string to a midnight datetime."""
    return datetime.strptime(date_str, DATE_FORMAT)
