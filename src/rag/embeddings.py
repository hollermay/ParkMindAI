"""
Embedding model factory — returns a LangChain-compatible Embeddings object.

Uses sentence-transformers (runs locally, no API key needed).
"""
import logging

import src.config as cfg

logger = logging.getLogger(__name__)

_embeddings_instance = None


def get_embeddings():
    """Return a cached embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info("Using local HuggingFace embeddings model: %s", cfg.EMBEDDING_MODEL)
    _embeddings_instance = HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return _embeddings_instance
