"""
SmartPark MCP Server package.

Exposes the FastAPI-based Model Context Protocol server that processes
confirmed reservations and writes them to persistent storage.

Public API:
  start_mcp_server(host, port) → bool   — start server in a daemon thread
"""
from src.mcp_server.server import start_mcp_server

__all__ = ["start_mcp_server"]
