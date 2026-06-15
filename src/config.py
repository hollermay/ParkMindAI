"""
Application-wide configuration loaded from environment variables.
Copy .env.example to .env and populate with your values.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DATA_PATH = str(DATA_DIR / "static" / "parking_info.json")

# ─── LLM ─────────────────────────────────────────────────────────────────────
# Provider: "groq" | "mock"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

# Groq
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Vector Store ─────────────────────────────────────────────────────────────
# Options: "weaviate" (cloud/local) | "chroma" (local fallback)
VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "weaviate")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db"))
COLLECTION_NAME: str = "ParkingKnowledge"

# ─── Weaviate ─────────────────────────────────────────────────────────────────
# Weaviate Cloud: set WEAVIATE_URL + WEAVIATE_API_KEY
# Local Docker:  set WEAVIATE_URL=http://localhost:8080, leave WEAVIATE_API_KEY blank
WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY: str = os.getenv("WEAVIATE_API_KEY", "")

# ─── SQL Database (PostgreSQL) ────────────────────────────────────────────────
# Full SQLAlchemy connection URL.  Examples:
#   postgresql://user:password@localhost:5432/smartpark
#   postgresql+psycopg2://user:pass@host/db
# Falls back to a local SQLite file when not set (development only).
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    os.getenv("SQLITE_DB_PATH", f"sqlite:///{DATA_DIR / 'parking.db'}"),
)
# Backward-compatibility alias — all existing call sites use cfg.SQLITE_DB_PATH
SQLITE_DB_PATH: str = DATABASE_URL

# ─── RAG Settings ─────────────────────────────────────────────────────────────
TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "4"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ─── LangSmith / LangGraph Studio ────────────────────────────────────────────
# Set LANGCHAIN_TRACING_V2=true and supply LANGCHAIN_API_KEY to enable tracing.
# These are read directly by the LangChain/LangGraph SDK — no extra code needed.
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "smartpark-chatbot")
LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ─── Application ──────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")

# ─── Admin Agent — REST API ────────────────────────────────────────────────────
# The second agent exposes a web dashboard for the administrator to approve/reject
# reservations. Set ADMIN_API_ONLY=true to disable the terminal fallback.
ADMIN_API_HOST: str = os.getenv("ADMIN_API_HOST", "localhost")
ADMIN_API_PORT: int = int(os.getenv("ADMIN_API_PORT", "5001"))
ADMIN_API_ONLY: bool = os.getenv("ADMIN_API_ONLY", "false").lower() == "true"
# Seconds to wait for admin decision via REST API before falling back to terminal
ADMIN_DECISION_TIMEOUT: int = int(os.getenv("ADMIN_DECISION_TIMEOUT", "120"))

# ─── Admin Agent — Email (SMTP) ────────────────────────────────────────────────
# Leave SMTP_HOST empty to disable email notifications (REST API still works).
ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "")       # recipient (admin's email)
SMTP_HOST: str = os.getenv("SMTP_HOST", "")           # e.g. smtp.gmail.com
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USER: str = os.getenv("SMTP_USER", "")           # login username
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")   # login password / app password
SMTP_FROM: str = os.getenv("SMTP_FROM", "parkbot@smartpark.com")

# ─── MCP Server ───────────────────────────────────────────────────────────────
# FastAPI Model Context Protocol server that processes confirmed reservations.
# Set MCP_API_KEY to a strong random secret in production.
MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "5002"))
MCP_API_KEY: str = os.getenv("MCP_API_KEY", "smartpark-mcp-secret-key")
# Path to the confirmed-reservations text log written by the MCP server
RESERVATIONS_FILE_PATH: str = os.getenv(
    "RESERVATIONS_FILE_PATH",
    str(DATA_DIR / "confirmed_reservations.txt"),
)
