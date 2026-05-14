"""
Vector store setup: loads parking JSON, chunks it into documents,
embeds them, and persists to Weaviate (cloud/local) or ChromaDB (local fallback).

Static data (general info, location, zones, policies, FAQ …) lives here.
Dynamic data (live prices, availability) is kept in the SQL database and injected
separately at query time so the vector index never goes stale.
"""
import json
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import src.config as cfg
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


# ─── Document loading ─────────────────────────────────────────────────────────

def _flatten_json_to_documents(data: dict, prefix: str = "") -> List[Document]:
    """
    Recursively convert a nested JSON dict into a list of LangChain Documents.
    Each top-level key becomes a separate document so retrieval is granular.
    """
    docs: List[Document] = []
    for key, value in data.items():
        section_title = f"{prefix}{key}".replace("_", " ").title()

        if isinstance(value, dict):
            # Render nested dicts as YAML-style text
            content = _dict_to_text(value, indent=0)
            docs.append(Document(page_content=f"{section_title}:\n{content}", metadata={"section": key}))
        elif isinstance(value, list):
            if all(isinstance(item, dict) for item in value):
                # List of objects (e.g. FAQ)
                for i, item in enumerate(value):
                    item_text = _dict_to_text(item, indent=0)
                    docs.append(Document(
                        page_content=f"{section_title} #{i + 1}:\n{item_text}",
                        metadata={"section": key, "index": i},
                    ))
            else:
                content = "\n".join(f"- {item}" for item in value)
                docs.append(Document(page_content=f"{section_title}:\n{content}", metadata={"section": key}))
        else:
            docs.append(Document(page_content=f"{section_title}: {value}", metadata={"section": key}))
    return docs


def _dict_to_text(d: dict, indent: int = 0) -> str:
    lines = []
    pad = "  " * indent
    for k, v in d.items():
        nice_key = str(k).replace("_", " ").title()
        if isinstance(v, dict):
            lines.append(f"{pad}{nice_key}:")
            lines.append(_dict_to_text(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{pad}{nice_key}:")
            for item in v:
                if isinstance(item, dict):
                    lines.append(_dict_to_text(item, indent + 1))
                else:
                    lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{nice_key}: {v}")
    return "\n".join(lines)


def load_parking_documents(json_path: str) -> List[Document]:
    """Load and chunk static parking JSON into LangChain Documents."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_docs = _flatten_json_to_documents(data)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("Loaded %d chunks from %s", len(chunks), json_path)
    return chunks


# ─── Vector store factory ────────────────────────────────────────────────────

def build_vector_store(documents: List[Document] | None = None, force_rebuild: bool = False):
    """
    Build or load the vector store.
    - If VECTOR_STORE_TYPE == "weaviate": use Weaviate (cloud or local Docker).
    - If VECTOR_STORE_TYPE == "chroma":   use local ChromaDB (fallback).
    """
    store_type = cfg.VECTOR_STORE_TYPE.lower()

    if store_type == "chroma":
        return _get_chroma(documents, force_rebuild)
    elif store_type == "weaviate":
        return _get_weaviate(documents, force_rebuild)
    else:
        raise ValueError(f"Unsupported VECTOR_STORE_TYPE: {store_type!r}. Choose 'weaviate' or 'chroma'.")


def _get_chroma(documents: List[Document] | None, force_rebuild: bool) -> Chroma:
    embeddings = get_embeddings()
    persist_dir = cfg.CHROMA_PERSIST_DIR
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    # Load existing collection if it already has data
    existing = Chroma(
        collection_name=cfg.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    if not force_rebuild and existing._collection.count() > 0:
        logger.info(
            "Loaded existing ChromaDB collection '%s' (%d documents).",
            cfg.COLLECTION_NAME,
            existing._collection.count(),
        )
        return existing

    # Build from documents
    if documents is None:
        documents = load_parking_documents(cfg.STATIC_DATA_PATH)

    if force_rebuild and existing._collection.count() > 0:
        existing.delete_collection()
        logger.info("Dropped existing ChromaDB collection for rebuild.")

    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=cfg.COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    logger.info("Built ChromaDB collection with %d chunks.", len(documents))
    return store


def _get_weaviate(documents: List[Document] | None, force_rebuild: bool = False):
    """
    Weaviate vector store.

    Works with:
    - Weaviate Cloud (WCD): set WEAVIATE_URL and WEAVIATE_API_KEY in .env
    - Local Docker:         set WEAVIATE_URL=http://localhost:8080, leave WEAVIATE_API_KEY blank

    Requires: pip install weaviate-client
    """
    try:
        import weaviate
        from weaviate.auth import AuthApiKey
        from weaviate.classes.init import AdditionalConfig, Timeout
    except ImportError as exc:
        raise ImportError(
            "weaviate-client is required. Run: pip install weaviate-client"
        ) from exc

    if not cfg.WEAVIATE_URL:
        raise ValueError(
            "WEAVIATE_URL is not set. "
            "Set it to your Weaviate Cloud URL (https://xxxx.weaviate.network) "
            "or local Docker URL (http://localhost:8080)."
        )

    embeddings = get_embeddings()

    # ─── connect ─────────────────────────────────────────────────────────
    auth = AuthApiKey(cfg.WEAVIATE_API_KEY) if cfg.WEAVIATE_API_KEY else None
    client = weaviate.connect_to_custom(
        http_host=cfg.WEAVIATE_URL.split("://")[-1].split(":")[0],
        http_port=int(cfg.WEAVIATE_URL.split(":")[-1]) if ":" in cfg.WEAVIATE_URL.split("://")[-1] else (443 if cfg.WEAVIATE_URL.startswith("https") else 80),
        http_secure=cfg.WEAVIATE_URL.startswith("https"),
        grpc_host=cfg.WEAVIATE_URL.split("://")[-1].split(":")[0],
        grpc_port=50051,
        grpc_secure=cfg.WEAVIATE_URL.startswith("https"),
        auth_credentials=auth,
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60)),
    )

    class_name = cfg.COLLECTION_NAME  # e.g. "ParkingKnowledge"

    # ─── LangChain wrapper ──────────────────────────────────────────────
    # langchain-weaviate may not satisfy Python 3.14 deps either, so we use
    # a thin adapter that implements similarity_search_with_score directly
    # via the weaviate-client v4 native API.
    store = _WeaviateLCAdapter(client, class_name, embeddings, cfg.TOP_K_DOCUMENTS)

    # ─── upsert documents if collection is empty or force_rebuild ───────────
    if documents is None:
        documents = load_parking_documents(cfg.STATIC_DATA_PATH)

    schema_exists = client.collections.exists(class_name)

    if force_rebuild and schema_exists:
        client.collections.delete(class_name)
        schema_exists = False
        logger.info("Dropped Weaviate collection '%s' for rebuild.", class_name)

    if not schema_exists:
        from weaviate.classes.config import Configure, Property, DataType
        client.collections.create(
            name=class_name,
            vectorizer_config=Configure.Vectorizer.none(),   # we supply our own vectors
            properties=[
                Property(name="text",    data_type=DataType.TEXT),
                Property(name="section", data_type=DataType.TEXT),
            ],
        )
        logger.info("Created Weaviate collection '%s'.", class_name)

    collection = client.collections.get(class_name)
    count = collection.aggregate.over_all(total_count=True).total_count

    if count == 0:
        _weaviate_upsert(collection, documents, embeddings)
        logger.info("Indexed %d documents into Weaviate '%s'.", len(documents), class_name)
    else:
        logger.info(
            "Weaviate collection '%s' already has %d objects — skipping upsert."
            " Pass force_rebuild=True to re-index.",
            class_name, count,
        )

    return store


def _weaviate_upsert(collection, documents: List[Document], embeddings) -> None:
    """Batch-insert documents with pre-computed embeddings into a Weaviate collection."""
    texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(texts)
    with collection.batch.dynamic() as batch:
        for doc, vector in zip(documents, vectors):
            batch.add_object(
                properties={
                    "text":    doc.page_content,
                    "section": doc.metadata.get("section", ""),
                },
                vector=vector,
            )


class _WeaviateLCAdapter:
    """
    Minimal LangChain-compatible wrapper around the Weaviate v4 native client.
    Exposes similarity_search_with_score() so the existing retriever.py works
    without any changes.
    """

    def __init__(self, client, class_name: str, embeddings, default_k: int = 4):
        self._client = client
        self._class_name = class_name
        self._embeddings = embeddings
        self._default_k = default_k

    def similarity_search_with_score(
        self, query: str, k: int | None = None
    ) -> List[tuple]:
        from weaviate.classes.query import MetadataQuery

        k = k or self._default_k
        query_vector = self._embeddings.embed_query(query)
        collection = self._client.collections.get(self._class_name)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            doc = Document(
                page_content=obj.properties.get("text", ""),
                metadata={"section": obj.properties.get("section", "")},
            )
            # Weaviate returns distance (0=identical); convert to similarity score
            score = 1.0 - (obj.metadata.distance or 0.0)
            results.append((doc, score))
        return results

    def add_documents(self, documents: List[Document]) -> None:
        collection = self._client.collections.get(self._class_name)
        _weaviate_upsert(collection, documents, self._embeddings)


# ─── Convenience singleton ────────────────────────────────────────────────────

_store_instance = None


def get_vector_store(force_rebuild: bool = False):
    """Return a cached vector store instance (built on first call)."""
    global _store_instance
    if _store_instance is None or force_rebuild:
        _store_instance = build_vector_store(force_rebuild=force_rebuild)
    return _store_instance
