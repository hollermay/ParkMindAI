"""
LLM factory — returns a LangChain chat model for the configured provider.
Supported providers: groq, mock (set LLM_PROVIDER in .env).
"""
import logging
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import src.config as cfg

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.2):
    """
    Return a LangChain chat model based on LLM_PROVIDER in .env.

    Priority:
      1. LLM_PROVIDER=groq → ChatGroq (requires GROQ_API_KEY)
      2. Anything else / no key → MockLLM (offline demo)
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

    logger.warning("No valid LLM provider configured — using MockLLM.")
    return _MockLLM()


class _MockLLM(BaseChatModel):
    """
    Minimal mock LLM for offline testing / demo without an API key.
    Inherits BaseChatModel so it is a proper LangChain Runnable and works
    in chains like ``prompt | llm | StrOutputParser()``.
    """

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_human = next(
            (m for m in reversed(messages) if getattr(m, "type", None) == "human"),
            None,
        )
        content = last_human.content if last_human else ""
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
            reply = "info"
        elif any(w in content_lower for w in ["hello", "hi", "hey", "greet"]):
            reply = "Hello! Welcome to SmartPark City Center. How can I help you today?"
        else:
            reply = (
                "[Mock mode — no API key] I can help with parking info and reservations. "
                "Please set LLM_PROVIDER and the corresponding API key in your .env for "
                "full AI-powered responses."
            )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
