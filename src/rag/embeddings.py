"""
Embedding model factory — returns a LangChain-compatible Embeddings object.

By default we use sentence-transformers (runs locally, no API key needed).
Set USE_OPENAI_EMBEDDINGS=true in .env to use OpenAI text-embedding-ada-002.
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

    if cfg.USE_OPENAI_EMBEDDINGS:
        if not cfg.OPENAI_API_KEY:
            raise ValueError(
                "USE_OPENAI_EMBEDDINGS=true but OPENAI_API_KEY is not set. "
                "Either provide the key or set USE_OPENAI_EMBEDDINGS=false."
            )
        from langchain_openai import OpenAIEmbeddings
        logger.info("Using OpenAI embeddings model: text-embedding-ada-002")
        _embeddings_instance = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=cfg.OPENAI_API_KEY,
        )
    else:
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
