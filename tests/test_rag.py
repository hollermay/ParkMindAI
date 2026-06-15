"""
Tests for the RAG vector store and retriever.

Covers: document loading, indexing, similarity search, metadata,
and retrieval result formatting.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Document loading ─────────────────────────────────────────────────────────

class TestDocumentLoading:
    def test_loads_nonzero_chunks(self):
        """parking_info.json should produce at least 10 chunks."""
        import src.config as cfg
        from src.rag.vectorstore import load_parking_documents
        docs = load_parking_documents(cfg.STATIC_DATA_PATH)
        assert len(docs) >= 10, f"Expected ≥10 chunks, got {len(docs)}"

    def test_chunks_have_content(self):
        """Every chunk must have non-empty page_content."""
        import src.config as cfg
        from src.rag.vectorstore import load_parking_documents
        docs = load_parking_documents(cfg.STATIC_DATA_PATH)
        for doc in docs:
            assert doc.page_content.strip(), "Found a chunk with empty page_content"

    def test_chunks_have_section_metadata(self):
        """Every chunk should carry a 'section' metadata key."""
        import src.config as cfg
        from src.rag.vectorstore import load_parking_documents
        docs = load_parking_documents(cfg.STATIC_DATA_PATH)
        for doc in docs:
            assert "section" in doc.metadata, f"Missing 'section' metadata in: {doc.page_content[:60]}"

    def test_chunk_size_respected(self):
        """All chunks should be within the configured CHUNK_SIZE (+overlap tolerance)."""
        import src.config as cfg
        from src.rag.vectorstore import load_parking_documents
        docs = load_parking_documents(cfg.STATIC_DATA_PATH)
        # Allow 2× tolerance because RecursiveCharacterTextSplitter can exceed by overlap
        max_allowed = cfg.CHUNK_SIZE * 2
        for doc in docs:
            assert len(doc.page_content) <= max_allowed, (
                f"Chunk too large ({len(doc.page_content)} chars): {doc.page_content[:80]}"
            )


# ─── Vector store building ────────────────────────────────────────────────────

class TestVectorStoreBuild:
    def test_build_returns_nonempty_store(self, tmp_chroma_dir, monkeypatch):
        """build_vector_store should return a store with documents."""
        import src.config as cfg
        monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", tmp_chroma_dir)
        monkeypatch.setattr(cfg, "COLLECTION_NAME", "test_build")

        from src.rag.vectorstore import build_vector_store
        store = build_vector_store()
        count = store._collection.count()
        assert count > 0, "Vector store is empty after build"

    def test_build_idempotent(self, tmp_chroma_dir, monkeypatch):
        """Calling build_vector_store twice (no force_rebuild) should not double-index."""
        import src.config as cfg
        monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", tmp_chroma_dir)
        monkeypatch.setattr(cfg, "COLLECTION_NAME", "test_idempotent")

        from src.rag.vectorstore import build_vector_store
        store1 = build_vector_store()
        count1 = store1._collection.count()

        store2 = build_vector_store()
        count2 = store2._collection.count()

        assert count1 == count2, "Document count changed on second build (double-indexed?)"


# ─── Retrieval ────────────────────────────────────────────────────────────────

class TestRetrieval:
    @pytest.fixture(autouse=True)
    def _patch_store(self, tmp_chroma_dir, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", tmp_chroma_dir)
        monkeypatch.setattr(cfg, "COLLECTION_NAME", "test_retrieval")
        # Reset singleton
        import src.rag.vectorstore as vs_module
        vs_module._store_instance = None
        from src.rag.vectorstore import build_vector_store
        build_vector_store()

    def test_retrieve_returns_k_docs(self):
        """retrieve() should return exactly k documents."""
        from src.rag.retriever import retrieve
        result = retrieve("parking prices", k=4)
        assert len(result.documents) == 4

    def test_retrieve_pricing_query(self):
        """A pricing query should return docs mentioning pricing."""
        from src.rag.retriever import retrieve
        result = retrieve("How much does parking cost?", k=4)
        combined = " ".join(d.page_content.lower() for d in result.documents)
        assert any(keyword in combined for keyword in ["price", "rate", "cost", "zone", "$"]), (
            "Pricing query did not retrieve pricing-related content"
        )

    def test_retrieve_location_query(self):
        """A location query should return docs about the address."""
        from src.rag.retriever import retrieve
        result = retrieve("Where is the parking located?", k=4)
        combined = " ".join(d.page_content.lower() for d in result.documents)
        assert any(kw in combined for kw in ["address", "boulevard", "location", "city center", "directions"]), (
            "Location query did not retrieve location content"
        )

    def test_retrieve_has_scores(self):
        """Each retrieved document should have a similarity score."""
        from src.rag.retriever import retrieve
        result = retrieve("EV charging", k=3)
        assert len(result.scores) == len(result.documents)
        for score in result.scores:
            assert isinstance(score, float)

    def test_retrieve_latency_recorded(self):
        """Retrieval latency should be a positive float."""
        from src.rag.retriever import retrieve
        result = retrieve("working hours", k=2)
        assert result.latency_ms > 0

    def test_formatted_context_nonempty(self):
        """formatted_context should be a non-empty string."""
        from src.rag.retriever import retrieve
        result = retrieve("disabled parking", k=2)
        assert result.formatted_context.strip()

    def test_retrieve_context_string(self):
        """retrieve_context_string() convenience wrapper should return a string."""
        from src.rag.retriever import retrieve_context_string
        ctx = retrieve_context_string("security features", k=3)
        assert isinstance(ctx, str) and len(ctx) > 10
