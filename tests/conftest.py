"""
Shared pytest fixtures for the SmartPark chatbot test suite.
"""
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Use a temporary SQLite DB for all tests ─────────────────────────────────

@pytest.fixture(scope="session")
def tmp_db_path(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "test_parking.db"
    return str(db_file)


@pytest.fixture(scope="session")
def seeded_db(tmp_db_path):
    """Initialise a temporary database with seed data once per session."""
    from src.database.models import init_db
    init_db(tmp_db_path)
    return tmp_db_path


# ─── Temporary vector store ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_chroma_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("chroma"))


@pytest.fixture(scope="session")
def vector_store(tmp_chroma_dir, monkeypatch_session):
    """Build a test vector store in a temp directory."""
    import src.config as cfg
    monkeypatch_session.setattr(cfg, "CHROMA_PERSIST_DIR", tmp_chroma_dir)
    monkeypatch_session.setattr(cfg, "COLLECTION_NAME", "test_collection")

    from src.rag.vectorstore import build_vector_store
    store = build_vector_store()
    return store


@pytest.fixture(scope="session")
def monkeypatch_session(request):
    """Session-scoped monkeypatch (pytest's built-in is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()
