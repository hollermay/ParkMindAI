"""
LLM factory — returns a LangChain chat model for the configured provider.
Supported providers: groq, gemini, openai, mock (set LLM_PROVIDER in .env).
"""
import logging

import src.config as cfg

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.2):
    """
    Return a LangChain chat model based on LLM_PROVIDER in .env.

    Priority:
      1. LLM_PROVIDER=groq    → ChatGroq (requires GROQ_API_KEY)
      2. LLM_PROVIDER=gemini  → ChatGoogleGenerativeAI (requires GEMINI_API_KEY)
      3. LLM_PROVIDER=openai  → ChatOpenAI (requires OPENAI_API_KEY)
      4. Anything else / no key → MockLLM (offline demo)
    """
    provider = cfg.LLM_PROVIDER.lower()

    if provider == "groq":
        if not cfg.GROQ_API_KEY:
            logger.warning("LLM_PROVIDER=groq but GROQ_API_KEY is not set — falling back to MockLLM.")
            return _MockLLM()
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError(
                "langchain-groq is required for Groq. "
                "Run: pip install langchain-groq"
            ) from exc
        logger.info("Using Groq model: %s", cfg.GROQ_MODEL)
        return ChatGroq(
            model=cfg.GROQ_MODEL,
            groq_api_key=cfg.GROQ_API_KEY,
            temperature=temperature,
        )

    if provider == "gemini":
        if not cfg.GEMINI_API_KEY:
            logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set — falling back to MockLLM.")
            return _MockLLM()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required for Gemini. "
                "Run: pip install langchain-google-genai"
            ) from exc
        logger.info("Using Gemini model: %s", cfg.GEMINI_MODEL)
        return ChatGoogleGenerativeAI(
            model=cfg.GEMINI_MODEL,
            google_api_key=cfg.GEMINI_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    if provider == "openai":
        if not cfg.OPENAI_API_KEY:
            logger.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is not set — falling back to MockLLM.")
            return _MockLLM()
        from langchain_openai import ChatOpenAI
        logger.info("Using OpenAI model: %s", cfg.OPENAI_MODEL)
        return ChatOpenAI(
            model=cfg.OPENAI_MODEL,
            temperature=temperature,
            openai_api_key=cfg.OPENAI_API_KEY,
        )

    logger.warning("No valid LLM provider configured — using MockLLM.")
    return _MockLLM()


class _MockLLM:
    """
    Minimal mock LLM for offline testing / demo without an OpenAI API key.
    Returns template-based responses rather than real LLM completions.
    """

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        # Extract text from the last human message
        if isinstance(messages, list):
            last_human = next(
                (m for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
                None,
            )
            content = last_human.content if last_human else ""
        elif hasattr(messages, "messages"):
            content = str(messages)
        else:
            content = str(messages)

        content_lower = content.lower()

        if any(w in content_lower for w in ["price", "cost", "rate", "fee", "charge"]):
            reply = (
                "SmartPark pricing (mock mode):\n"
                "  Zone A (Premium): $6/hr | $35 daily max\n"
                "  Zone B/C (Standard): $3/hr | $20 daily max\n"
                "  Zone D (EV): $3/hr + $0.30/kWh\n"
                "  Zone E (Disabled): Free with badge"
            )
        elif any(w in content_lower for w in ["hour", "open", "close", "time"]):
            reply = (
                "SmartPark is open 24/7 for automated access.\n"
                "Staffed booth: Mon–Fri 07:00–22:00 | Sat 08:00–20:00 | Sun 09:00–18:00."
            )
        elif any(w in content_lower for w in ["location", "address", "where", "direction"]):
            reply = (
                "SmartPark City Center is at 123 Innovation Boulevard, City Center. "
                "Opposite City Hall, 200m east of Central Station."
            )
        elif any(w in content_lower for w in ["classify", "intent"]):
            # Intent node uses this; default to 'info'
            reply = "info"
        elif any(w in content_lower for w in ["hello", "hi", "hey", "greet"]):
            reply = "Hello! Welcome to SmartPark City Center. How can I help you today?"
        else:
            reply = (
                "[Mock mode — no OpenAI key] I can help with parking info and reservations. "
                "Please set OPENAI_API_KEY in your .env for full AI-powered responses."
            )
        return AIMessage(content=reply)

    def __or__(self, other):
        """Support `llm | parser` chaining."""
        from langchain_core.runnables import RunnableSequence
        return RunnableSequence(first=self, last=other)
