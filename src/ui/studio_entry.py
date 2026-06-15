"""
LangGraph Studio / LangSmith UI — entry point for SmartPark Chatbot.

This module is discovered by `langgraph dev` via langgraph.json.
It exports a compiled graph object so that LangGraph Studio can:
  • Render the visual graph diagram
  • Stream interactive chat sessions
  • Display per-node state updates in real time
  • Support the human-in-the-loop (admin approval) interrupt

Usage:
    langgraph dev          # starts Studio at http://localhost:2024
"""
import sys
from pathlib import Path

# ── Ensure repo root is importable when launched from any cwd ────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.config as cfg  # noqa: E402
from src.database.models import init_db  # noqa: E402

# ── One-time initialisation ───────────────────────────────────────────────────
# The database must exist before the first graph invocation.
init_db(cfg.SQLITE_DB_PATH)

# ── Build graph ───────────────────────────────────────────────────────────────
from src.chatbot.graph import build_graph  # noqa: E402

# LangGraph Platform (Studio / Cloud) handles persistence itself.
# Passing checkpointer=None tells build_graph to compile without one.
graph = build_graph(checkpointer=None)
