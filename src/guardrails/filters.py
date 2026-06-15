"""
Guardrails: input and output safety filters.

Stage 4 requirement — prevents:
  1. Prompt injection / jailbreak attempts (input filter)
  2. Exposure of other users' PII in responses (output filter)
  3. Leakage of internal system data (output filter)
  4. Off-topic / harmful content (input filter)
"""
import logging
import re
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


# ─── Input filter ─────────────────────────────────────────────────────────────

# Patterns that indicate prompt injection / jailbreak attempts
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"forget\s+(everything|all)\s+(above|before|prior)", re.I),
    re.compile(r"you\s+are\s+now\s+(?!parkbot)", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?(?!parkbot)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(?!parkbot)", re.I),
    re.compile(r"(system|admin)\s+prompt", re.I),
    re.compile(r"override\s+(system|safety|rules?)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\s+mode", re.I),
]

# Patterns that suggest a user is trying to extract the DB / other users' data
_DATA_EXFILTRATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(show|list|dump|print|display|give)(\s+\w+)?\s+(all\s+)?(users?|customers?|reservations?|bookings?|records?|database|table)", re.I),
    re.compile(r"(select|insert|update|delete|drop)\s+.{0,30}\s+from", re.I),  # SQL injection
    re.compile(r"other\s+(users?|customers?|people)('s)?\s+(data|info|details|reservations?)", re.I),
    re.compile(r"admin\s+(password|credentials?|login|access)", re.I),
    re.compile(r"(internal|secret|hidden)\s+(api|system)\s*(keys?|secrets?|tokens?)", re.I),
    re.compile(r"api\s+(secrets?|keys?|tokens?)", re.I),
]


@dataclass
class InputFilterResult:
    is_safe: bool
    blocked_reason: str = ""
    sanitised_input: str = ""


def filter_input(user_message: str) -> InputFilterResult:
    """
    Validate the user's raw input before it reaches the LLM or graph nodes.
    Returns InputFilterResult with is_safe=False and a reason if blocked.
    """
    if not user_message or not user_message.strip():
        return InputFilterResult(is_safe=True, sanitised_input="")

    # Check length — reject absurdly long messages (DoS / context stuffing)
    if len(user_message) > 2000:
        return InputFilterResult(
            is_safe=False,
            blocked_reason="Message is too long (maximum 2000 characters).",
        )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(user_message):
            logger.warning("Prompt injection attempt detected: %r", user_message[:120])
            return InputFilterResult(
                is_safe=False,
                blocked_reason=(
                    "I detected an attempt to manipulate my behaviour. "
                    "I can only assist with SmartPark City Center parking services."
                ),
            )

    for pattern in _DATA_EXFILTRATION_PATTERNS:
        if pattern.search(user_message):
            logger.warning("Data exfiltration attempt detected: %r", user_message[:120])
            return InputFilterResult(
                is_safe=False,
                blocked_reason=(
                    "I'm not able to provide access to the database or other users' data. "
                    "If you need help with your own reservation, please ask!"
                ),
            )

    # Light sanitisation — strip leading/trailing whitespace
    sanitised = user_message.strip()
    return InputFilterResult(is_safe=True, sanitised_input=sanitised)


# ─── Output filter ─────────────────────────────────────────────────────────────

# Regex to detect PII that should not appear in AI responses
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
# Phone regex — requires at least 7 digit characters in the match, and explicitly
# excludes ISO date strings (YYYY-MM-DD, DD/MM/YYYY, etc.) to avoid false positives.
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_DATE_LIKE_RE = re.compile(
    r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"          # YYYY-MM-DD
    r"|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b"   # DD/MM/YYYY or D.M.YYYY
)
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ \-]?){13,16}\b")

# Phrases that should never appear in a public-facing response
_SENSITIVE_PHRASES: List[re.Pattern] = [
    re.compile(r"admin\s+password", re.I),
    re.compile(r"internal\s+(api|secret|key|token)", re.I),
    re.compile(r"database\s+(password|credentials?)", re.I),
    re.compile(r"connection\s+string", re.I),
]


@dataclass
class OutputFilterResult:
    is_safe: bool
    filtered_output: str
    warnings: List[str]


def filter_output(ai_response: str) -> OutputFilterResult:
    """
    Scan the AI-generated response for PII or sensitive internal data.
    Replaces matches with safe placeholders and returns warnings.
    """
    warnings: List[str] = []
    output = ai_response

    # Redact email addresses that are not the official SmartPark email
    official_emails = {"support@smartpark-citycenter.example.com"}
    for match in _EMAIL_RE.finditer(output):
        email = match.group()
        if email.lower() not in official_emails:
            output = output.replace(email, "[EMAIL REDACTED]")
            warnings.append(f"Redacted non-official email: {email}")

    # Redact phone numbers that are not the official numbers.
    # First, temporarily mask date-like strings so they don't trigger the phone regex.
    date_placeholders = {}
    masked_output = output
    for i, m in enumerate(_DATE_LIKE_RE.finditer(output)):
        placeholder = f"__DATE_{i}__"
        date_placeholders[placeholder] = m.group()
        masked_output = masked_output.replace(m.group(), placeholder, 1)

    for match in _PHONE_RE.finditer(masked_output):
        phone = match.group().strip()
        # Skip if this is one of our date placeholders (shouldn't happen but defensive)
        if any(ph in phone for ph in date_placeholders):
            continue
        # Count actual digit characters — real phone numbers have ≥ 7 digits
        digit_count = sum(1 for c in phone if c.isdigit())
        if digit_count < 7:
            continue
        normalised = re.sub(r"[\s\-().+]", "", phone)
        if normalised not in {"15551234567", "15559117275"}:
            masked_output = masked_output.replace(phone, "[PHONE REDACTED]")
            warnings.append(f"Redacted non-official phone: {phone}")

    # Restore date placeholders
    for placeholder, original in date_placeholders.items():
        masked_output = masked_output.replace(placeholder, original)
    output = masked_output

    # Redact credit card patterns
    for match in _CREDIT_CARD_RE.finditer(output):
        card = match.group()
        output = output.replace(card, "[CARD NUMBER REDACTED]")
        warnings.append("Redacted potential card number.")

    # Block responses containing internal sensitive phrases
    for pattern in _SENSITIVE_PHRASES:
        if pattern.search(output):
            logger.error("Sensitive internal phrase detected in LLM output — blocking response.")
            return OutputFilterResult(
                is_safe=False,
                filtered_output=(
                    "I'm sorry, I encountered an issue generating a response. "
                    "Please contact support at +1 (555) 123-4567."
                ),
                warnings=["Response blocked: contained sensitive internal phrase."],
            )

    if warnings:
        logger.warning("Output filter applied %d redaction(s).", len(warnings))

    return OutputFilterResult(is_safe=True, filtered_output=output, warnings=warnings)
