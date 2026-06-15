"""
Prompt templates used by all graph nodes.
Keeping prompts separate from node logic makes them easy to iterate on.
"""
from langchain_core.prompts import ChatPromptTemplate

# ─── System identity ──────────────────────────────────────────────────────────
SYSTEM_PERSONA = (
    "You are ParkBot, the friendly and knowledgeable assistant for SmartPark City Center. "
    "Your role is to help visitors with parking information, answer questions about zones, "
    "prices, working hours, location, amenities, and policies, and to guide them through "
    "the parking reservation process. "
    "Be concise, accurate, and polite. "
    "Only provide information about SmartPark City Center — do not make up facts. "
    "If you don't know something, say so and suggest the customer call +1 (555) 123-4567."
)

# ─── Intent classification ─────────────────────────────────────────────────────
INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the user's message into exactly one of these intents:\n"
     "  - info        : asking about parking (location, prices, hours, zones, amenities, policies, FAQ)\n"
     "  - reservation : wants to book / reserve a parking space\n"
     "  - greeting    : hello, hi, thanks, bye, etc.\n"
     "  - off_topic   : anything unrelated to parking\n\n"
     "IMPORTANT: If the previous assistant message offered to make a reservation or asked "
     "whether the user wants to book, and the user replies with a short affirmative "
     "(yes, sure, okay, yep, please, go ahead, make a reservation, I do, etc.) — "
     "classify as 'reservation'.\n\n"
     "Previous assistant message (may be empty): {last_bot_message}\n\n"
     "Reply with ONLY the intent label, nothing else."),
    ("human", "{user_message}"),
])

# ─── RAG answer generation ─────────────────────────────────────────────────────
RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     SYSTEM_PERSONA + "\n\n"
     "Use the following retrieved context to answer the user's question. "
     "Prioritise the DYNAMIC CONTEXT (live data) for prices and availability. "
     "If the context contains the answer, use it directly — do NOT say you don't have the information.\n\n"
     "--- STATIC CONTEXT (knowledge base) ---\n{static_context}\n\n"
     "--- DYNAMIC CONTEXT (live data) ---\n{dynamic_context}"),
    ("human", "{user_message}"),
])

# ─── General / greeting response ──────────────────────────────────────────────
GENERAL_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PERSONA),
    ("human", "{user_message}"),
])

# ─── Reservation: field collection ────────────────────────────────────────────
RESERVATION_FIELD_PROMPTS = {
    "full_name": (
        "I'd be happy to help you reserve a parking space at SmartPark City Center! "
        "Let's start — what is your **full_name**?"
    ),
    "car_number": (
        "Great, {full_name}! "
        "Please provide your **vehicle registration plate number** (e.g., ABC-1234)."
    ),
    "zone": (
        "Which **parking zone** would you prefer?\n"
        "Each zone has **20 spaces** (tracked daily — zone is blocked when full).\n\n"
        "  • **Zone A** — Premium, ground floor ($6/hr)\n"
        "  • **Zone B** — Standard, levels 2–3 ($3/hr)\n"
        "  • **Zone C** — Economy, level 4 ($2.50/hr)\n"
        "  • **Zone D** — EV charging, rooftop ($3/hr + charging)\n"
        "  • **Zone E** — Disabled/accessible (free with badge)\n\n"
        "Type the zone letter (A, B, C, D, or E):"
    ),
    "start_date": (
        "What **date** would you like to **start** your reservation?\n"
        "You can type it in any common format, for example:\n"
        "  • 2026-06-15   (YYYY-MM-DD)\n"
        "  • 15/06/2026   (DD/MM/YYYY)\n"
        "  • June 15 2026"
    ),
    "end_date": (
        "And what **date** should the reservation **end**?\n"
        "Same formats accepted, e.g. 2026-06-20 or 20/06/2026.\n"
        "Maximum 30 days; minimum 1 day."
    ),
    "email": (
        "Almost done! Please provide your **email address** so we can send "
        "booking confirmation and any important updates."
    ),
    "card_number": (
        "Finally, please enter your **payment card number** (digits only, no spaces or dashes). "
        "This will be used to process your parking fee."
    ),
}

RESERVATION_FIELD_ORDER = ["full_name", "car_number", "zone",
                           "start_date", "end_date"]

RESERVATION_SUMMARY_TEMPLATE = """Here is a summary of your reservation request:

```
  Name      : {full_name}
  Email     : (your registered account email)
  Vehicle   : {car_number}
  Zone      : Zone {zone}
  Start     : {start_date}
  End       : {end_date}
  Duration  : {days} day(s)
  Est. Cost : {total_cost}
  Card      : {card_masked}
```

Your request has been sent for **administrator approval**.
You will receive confirmation shortly.
"""

RESERVATION_APPROVED_TEMPLATE = """✅ **Your reservation has been APPROVED!**

```
  Reservation Code : {code}
  Name             : {full_name}
  Vehicle          : {car_number}
  Zone             : Zone {zone}
  Start Date       : {start_date}
  End Date         : {end_date}
  Duration         : {days} day(s)
  Total Cost       : {total_cost}
  Card             : {card_masked}
```

A confirmation has been sent to your registered email address.
Please **save your reservation code** — you'll need it at the barrier.
Thank you for choosing SmartPark City Center!
"""

RESERVATION_REJECTED_TEMPLATE = """❌ Unfortunately, your reservation request has been **declined**.

**Reason:** {reason}

Please contact our customer service for assistance:

```
  Phone : +1 (555) 123-4567
  Email : support@smartpark-citycenter.example.com
```

We apologise for the inconvenience.
"""

# ─── Admin approval prompt (printed to console) ────────────────────────────────
ADMIN_APPROVAL_BANNER = """
╔══════════════════════════════════════════════════════╗
║        ⚠  ADMIN APPROVAL REQUIRED  ⚠                ║
╠══════════════════════════════════════════════════════╣
║  A new reservation request awaits your review.       ║
╚══════════════════════════════════════════════════════╝

Reservation Details:
  Name    : {full_name}
  Email   : {email}
  Vehicle : {car_number}
  Zone    : {zone}
  Start   : {start_date}
  End     : {end_date}
  Card    : {card_masked}
"""

OFF_TOPIC_RESPONSE = (
    "I'm ParkBot, specialised in SmartPark City Center parking services. "
    "I can help you with parking information, reservations, prices, and more. "
    "Is there anything parking-related I can assist you with today?"
)
