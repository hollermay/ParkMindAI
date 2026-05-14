"""
SmartPark City Center — Web Chat Launcher

Starts three servers:
  - Port 5000  Customer chat UI            (http://localhost:5000)
  - Port 5001  Admin approval dashboard    (http://localhost:5001/admin)
  - Port 5002  MCP server                  (http://localhost:5002/health)

Usage:
    python run_frontend.py
"""

import sys
import threading
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import src.config as cfg
from src.database.models import init_db

# ── Initialise the database if it doesn't already exist ──────────────────────
print("  Initialising database …")
init_db(cfg.SQLITE_DB_PATH)

# ── Start the MCP server (daemon thread) ──────────────────────────────────────
from src.mcp_server.server import start_mcp_server

start_mcp_server(host=cfg.MCP_SERVER_HOST, port=cfg.MCP_SERVER_PORT)
print(f"  MCP Server       →  http://{cfg.MCP_SERVER_HOST}:{cfg.MCP_SERVER_PORT}/health")

# ── Start the admin approval server on port 5001 (daemon thread) ─────────────
from src.admin_agent.api_server import start_api_server

start_api_server(host=cfg.ADMIN_API_HOST, port=cfg.ADMIN_API_PORT)
print(f"  Admin Dashboard  →  http://{cfg.ADMIN_API_HOST}:{cfg.ADMIN_API_PORT}/admin")

# ── Start the web chat server on port 5000 ────────────────────────────────────
import os
from src.reservation_frontend.app import app as chat_app

chat_port = int(os.getenv("CHAT_PORT", "5000"))
print(f"  Chat UI          →  http://localhost:{chat_port}")
print()
print("  Press Ctrl+C to stop both servers.")
print()

chat_app.run(host="0.0.0.0", port=chat_port, debug=False, use_reloader=False)
