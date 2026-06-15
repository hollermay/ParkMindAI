"""
SmartPark City Center — Chatbot CLI Entry Point

Usage:
    python src/main.py              # Start the chatbot (chat mode)
    python src/main.py --evaluate   # Run RAG evaluation and print the report
    python src/main.py --init-db    # (Re)initialise the SQLite database
    python src/main.py --rebuild    # Rebuild the vector store from scratch
"""
import argparse
import logging
import sys
import threading
import time
import uuid
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

# ─── Project bootstrap ────────────────────────────────────────────────────────
# Ensure the repo root is on the path when running as `python src/main.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config as cfg
from src.admin_agent.api_server import start_api_server
from src.database.models import init_db
from src.guardrails.filters import filter_input

console = Console()
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL, logging.WARNING))
# Silence noisy third-party loggers
for _noisy in ("httpx", "httpcore", "google_genai", "google.auth", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


# ─── Background cleanup scheduler ───────────────────────────────────────────────────

CLEANUP_INTERVAL_SECONDS = 60  # Check every 60 seconds


def _run_cleanup_loop(db_path: str, stop_event: threading.Event) -> None:
    """Background thread: delete expired reservations every CLEANUP_INTERVAL_SECONDS."""
    from src.database.operations import cleanup_expired_reservations
    while not stop_event.is_set():
        try:
            n = cleanup_expired_reservations(db_path)
            if n:
                logger.info("[Scheduler] Removed %d expired reservation(s).", n)
        except Exception as exc:
            logger.warning("[Scheduler] Cleanup error: %s", exc)
        stop_event.wait(CLEANUP_INTERVAL_SECONDS)


def start_cleanup_scheduler(db_path: str) -> threading.Event:
    """Start the background cleanup thread and return its stop event."""
    stop_event = threading.Event()
    t = threading.Thread(
        target=_run_cleanup_loop,
        args=(db_path, stop_event),
        daemon=True,   # dies automatically when main process exits
        name="reservation-cleanup",
    )
    t.start()
    return stop_event


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _print_bot(message: str) -> None:
    console.print(Panel(Markdown(message), title="[bold cyan]ParkBot[/]", border_style="cyan"))


def _print_user(message: str) -> None:
    console.print(f"[bold green]You:[/] {message}\n")


def _print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/] {message}")


def _admin_approval_prompt(state_values: dict, request_code: str = "") -> tuple[bool, str]:
    """
    Admin approval flow — dashboard first, terminal fallback.

    The human_approval node (Agent 2) registers the request and sends the
    email notification *before* the interrupt.  This function receives the
    pre-registered ``request_code`` so it can poll immediately without
    double-registering.  If no code is supplied (e.g. fallback path), it
    registers the request itself.
    """
    from src.admin_agent import decision_store
    from src.admin_agent.api_server import get_admin_url
    from src.admin_agent.notification import send_notification
    from src.reservation.handler import mask_card, mask_email

    rd = state_values.get("reservation_data") or {}
    dashboard_url = get_admin_url() + "/admin"
    timeout = cfg.ADMIN_DECISION_TIMEOUT

    # ── Register & notify (fallback only — normally done by the node) ─────────
    if not request_code:
        code = decision_store.generate_request_code()
        rd_for_store = {
            **rd,
            "card_masked": mask_card(rd.get("card_number", "")),
            "email_masked": mask_email(rd.get("email", "")),
        }
        decision_store.add_pending(code, rd_for_store)
        send_notification(code, rd_for_store, get_admin_url())
    else:
        # human_approval node (Agent 2) already registered; just use the code.
        code = request_code

    # ── Show panel ────────────────────────────────────────────────────────────
    console.print(Rule("[bold yellow]⚠  ADMIN APPROVAL REQUIRED  ⚠[/]"))
    console.print(Panel(
        f"[bold]Request Code:[/] [cyan]{code}[/]\n\n"
        f"[bold]Name:[/]    {rd.get('full_name', '')}\n"
        f"[bold]Email:[/]   {mask_email(rd.get('email', ''))}\n"
        f"[bold]Vehicle:[/] {rd.get('car_number', '')}\n"
        f"[bold]Zone:[/]    Zone {rd.get('zone', '')}\n"
        f"[bold]Start:[/]   {rd.get('start_date', '')}\n"
        f"[bold]End:[/]     {rd.get('end_date', '')}\n"
        f"[bold]Card:[/]    {mask_card(rd.get('card_number', ''))}\n\n"
        f"[cyan]🌐 Open the admin dashboard to approve/reject:[/]\n"
        f"[bold cyan]{dashboard_url}[/]",
        title="[yellow]Awaiting Admin Decision — Please use the Dashboard[/]",
        border_style="yellow",
    ))
    console.print(
        f"[dim]Waiting up to {timeout}s for admin decision "
        f"(request {code})...[/]\n"
    )

    # ── Poll the decision store ───────────────────────────────────────────────
    poll_interval = 2
    elapsed = 0
    while elapsed < timeout:
        decision = decision_store.get_decision(code)
        if decision is not None:
            decision_store.clear_decision(code)
            action = "✅ APPROVED" if decision["approved"] else "❌ REJECTED"
            colour = "green" if decision["approved"] else "red"
            console.print(f"[bold {colour}]Admin decision received: {action}[/]\n")
            return decision["approved"], decision.get("notes", "")
        time.sleep(poll_interval)
        elapsed += poll_interval

    # ── Timeout fallback: terminal prompt ─────────────────────────────────────
    console.print(
        f"[yellow]No dashboard response within {timeout}s. "
        "Falling back to terminal prompt.[/]\n"
    )
    decision_store.submit_decision(code, False, "")   # remove from pending
    terminal_decision = Prompt.ask(
        "[yellow]Admin decision[/]",
        choices=["approve", "reject", "a", "r"],
        default="approve",
    ).lower()
    approved = terminal_decision in ("approve", "a")
    notes = ""
    if not approved:
        notes = Prompt.ask(
            "[yellow]Reason for rejection[/]",
            default="Requested slot unavailable.",
        )
    return approved, notes


# ─── Chat session ─────────────────────────────────────────────────────────────

def run_chat() -> None:
    """Start an interactive chat session."""
    from langchain_core.messages import HumanMessage

    from src.chatbot.graph import get_graph

    # Initialise DB and vector store on first run
    console.print(Panel(
        "[bold cyan]SmartPark City Center — Parking Chatbot[/]\n"
        "[dim]Type your message and press Enter. Type 'quit' or 'exit' to stop.[/]\n"
        "[dim]Type [bold]/reservations[/bold] to view all reservations.[/]",
        border_style="cyan",
    ))

    try:
        console.print("[dim]Initialising systems...[/]", end=" ")
        init_db(cfg.SQLITE_DB_PATH)
        # Run an immediate cleanup of any leftover expired reservations
        from src.database.operations import cleanup_expired_reservations
        cleanup_expired_reservations(cfg.SQLITE_DB_PATH)
        # Start background scheduler
        _stop_cleanup = start_cleanup_scheduler(cfg.SQLITE_DB_PATH)
        # Start admin agent REST API server
        start_api_server(host=cfg.ADMIN_API_HOST, port=cfg.ADMIN_API_PORT)
        # Start MCP server (processes confirmed reservations → text log)
        from src.mcp_server.server import start_mcp_server
        start_mcp_server(host=cfg.MCP_SERVER_HOST, port=cfg.MCP_SERVER_PORT)
        graph = get_graph()
        # Warm up the vector store (triggers embedding model download on first run)
        from src.rag.vectorstore import get_vector_store
        get_vector_store()
        console.print("[green]Ready![/]\n")
        console.print(
            f"[cyan]🌐 Admin Dashboard → "
            f"http://{cfg.ADMIN_API_HOST}:{cfg.ADMIN_API_PORT}/admin[/]\n"
            f"[cyan]🔗 MCP Server     → "
            f"http://{cfg.MCP_SERVER_HOST}:{cfg.MCP_SERVER_PORT}/health[/]\n"
            "[dim](Open Admin Dashboard in browser to approve/reject reservations)[/]\n"
        )
    except Exception as exc:
        _print_error(f"Startup failed: {exc}")
        raise

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if user_input.lower().strip() in ("quit", "exit", "q", "bye"):
            console.print("[dim]Thank you for using SmartPark. Goodbye![/]")
            break

        if user_input.strip().lower() in ("/reservations", "/history", "/bookings"):
            show_reservations()
            continue

        if not user_input.strip():
            continue

        # ── Input guardrail ───────────────────────────────────────────────────
        filter_result = filter_input(user_input)
        if not filter_result.is_safe:
            _print_bot(filter_result.blocked_reason)
            continue

        _print_user(filter_result.sanitised_input)

        # ── Invoke graph ──────────────────────────────────────────────────────
        try:
            graph.invoke(
                {"messages": [HumanMessage(content=filter_result.sanitised_input)]},
                config=config,
            )
        except Exception as exc:
            logger.error("Graph invocation error: %s", exc, exc_info=True)
            _print_bot(
                "I encountered an unexpected error. Please try again or "
                "call our support line at +1 (555) 123-4567."
            )
            continue

        # ── Check for human-in-the-loop interrupt ─────────────────────────────
        from langgraph.types import Command
        graph_state = graph.get_state(config)

        if any(t.interrupts for t in (graph_state.tasks or ())):
            # Extract request_code set by the human_approval node (Agent 2)
            _interrupt_val: dict = {}
            for _task in (graph_state.tasks or ()):
                for _intr in (_task.interrupts or ()):
                    if isinstance(_intr.value, dict):
                        _interrupt_val = _intr.value
                    break
                break
            # Admin review — poll REST API / email / terminal
            approved, notes = _admin_approval_prompt(
                graph_state.values,
                request_code=_interrupt_val.get("request_code", ""),
            )

            # Resume the graph with the admin decision
            try:
                graph.invoke(
                    Command(resume={"approved": approved, "notes": notes}),
                    config=config,
                )
                graph_state = graph.get_state(config)
            except Exception as exc:
                logger.error("Graph resume error: %s", exc, exc_info=True)

        # ── Print the last AI message ─────────────────────────────────────────
        messages = graph_state.values.get("messages", [])
        if messages:
            last_msg = messages[-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            _print_bot(content)

# ─── Reservations view ───────────────────────────────────────────────────────

def _status_colour(status: str) -> str:
    return {"approved": "green", "rejected": "red", "pending": "yellow"}.get(status, "white")


def show_reservations() -> None:
    """Print a Rich table of all reservations in the database."""
    from src.database.operations import get_all_reservations_full

    init_db(cfg.SQLITE_DB_PATH)
    rows = get_all_reservations_full(cfg.SQLITE_DB_PATH)

    if not rows:
        console.print(Panel("[dim]No reservations found in the database.[/]", border_style="cyan"))
        return

    # Group counts
    counts: dict = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = "  ".join(
        f"[{_status_colour(s)}]{s.capitalize()}: {n}[/]" for s, n in sorted(counts.items())
    )

    table = Table(
        title="SmartPark — All Reservations",
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Code", style="bold")
    table.add_column("Name")
    table.add_column("Vehicle")
    table.add_column("Zone", justify="center")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Status", justify="center")
    table.add_column("Notes")

    for r in rows:
        colour = _status_colour(r.status)
        table.add_row(
            r.reservation_code,
            r.full_name,
            r.car_number,
            r.zone,
            str(r.start_datetime.date()) if r.start_datetime else "",
            str(r.end_datetime.date()) if r.end_datetime else "",
            f"[{colour}]{r.status.upper()}[/]",
            r.admin_notes or "",
        )

    console.print()
    console.print(table)
    console.print(f"  Total: {len(rows)}   {summary}\n")


# ─── Evaluation mode ──────────────────────────────────────────────────────────

def run_evaluation(full_pipeline: bool = False) -> None:
    """Run the RAG evaluation suite and print the report."""
    from src.evaluation.metrics import (
        evaluate_full_pipeline,
        evaluate_retrieval,
        save_report,
    )

    console.print(Panel("[bold]Running RAG Evaluation Suite...[/]", border_style="blue"))
    try:
        if full_pipeline:
            console.print("[dim]Running full pipeline evaluation (includes LLM generation)...[/]")
            report = evaluate_full_pipeline(k=cfg.TOP_K_DOCUMENTS)
        else:
            console.print("[dim]Running retrieval-only evaluation...[/]")
            report = evaluate_retrieval(k=cfg.TOP_K_DOCUMENTS)
    except Exception as exc:
        _print_error(f"Evaluation failed: {exc}")
        raise

    console.print(report.summary())
    save_report(report, "data/evaluation_report.json")
    console.print("[green]Report saved to data/evaluation_report.json[/]")


# ─── CLI entry ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmartPark City Center Parking Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run the RAG evaluation suite (retrieval-only).",
    )
    parser.add_argument(
        "--evaluate-full", action="store_true",
        help="Run the full pipeline evaluation (requires LLM).",
    )
    parser.add_argument(
        "--init-db", action="store_true",
        help="(Re)initialise the SQLite database with seed data.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild the vector store from scratch.",
    )
    parser.add_argument(
        "--reservations", action="store_true",
        help="Show all reservations (approved, rejected, pending) from the database.",
    )
    args = parser.parse_args()

    if args.init_db:
        console.print("[dim]Initialising database...[/]")
        init_db(cfg.SQLITE_DB_PATH)
        console.print("[green]Database initialised.[/]")
        return

    if args.rebuild:
        console.print("[dim]Rebuilding vector store...[/]")
        from src.rag.vectorstore import get_vector_store
        get_vector_store(force_rebuild=True)
        console.print("[green]Vector store rebuilt.[/]")
        return

    if args.evaluate or args.evaluate_full:
        init_db(cfg.SQLITE_DB_PATH)
        from src.rag.vectorstore import get_vector_store
        get_vector_store()
        run_evaluation(full_pipeline=args.evaluate_full)
        return

    if args.reservations:
        show_reservations()
        return

    run_chat()


if __name__ == "__main__":
    main()
