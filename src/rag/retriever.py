"""
RAG retrieval logic: queries the vector store and formats the results
into a single context string that the LLM can consume.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple

from langchain_core.documents import Document

import src.config as cfg
from src.rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    documents: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    query: str = ""
    latency_ms: float = 0.0

    @property
    def formatted_context(self) -> str:
        if not self.documents:
            return "No relevant parking information found for this query."
        parts = []
        for i, doc in enumerate(self.documents, 1):
            section = doc.metadata.get("section", "info")
            parts.append(f"[Source {i} — {section}]\n{doc.page_content}")
        return "\n\n".join(parts)


def retrieve(query: str, k: int = cfg.TOP_K_DOCUMENTS) -> RetrievalResult:
    """
    Perform a similarity search in the vector store and return the top-k
    most relevant documents along with their similarity scores.
    """
    store = get_vector_store()
    t0 = time.perf_counter()

    docs_with_scores: List[Tuple[Document, float]] = store.similarity_search_with_score(query, k=k)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.debug("RAG retrieval for %r: %.1f ms, %d docs", query, latency_ms, len(docs_with_scores))

    if docs_with_scores:
        docs, scores = zip(*docs_with_scores)
    else:
        docs, scores = [], []

    return RetrievalResult(
        documents=list(docs),
        scores=list(scores),
        query=query,
        latency_ms=latency_ms,
    )


def retrieve_context_string(query: str, k: int = cfg.TOP_K_DOCUMENTS) -> str:
    """Convenience wrapper — returns just the formatted context string."""
    result = retrieve(query, k=k)
    return result.formatted_context
