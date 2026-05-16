# SmartPark City Center — Intelligent Parking Chatbot

A production-grade intelligent parking chatbot built with **LangChain**, **LangGraph**, and a **Retrieval-Augmented Generation (RAG)** architecture. The system is composed of **two cooperating LangChain agents**: a user-facing chatbot and a dedicated admin notification agent. Together they handle parking information queries, multi-turn reservations, human-in-the-loop admin approval (via browser dashboard + email), immediate reservation cancellations with automatic refund policy enforcement, live zone availability tracking, and full data-protection guardrails — all accessible via a **browser-based chat UI**.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Usage](#usage)
5. [Features by Stage](#features-by-stage)
6. [Admin Agent — Second Agent](#admin-agent--second-agent)
7. [MCP Server — Reservation Persistence](#mcp-server--reservation-persistence)
8. [Configuration Reference](#configuration-reference)
9. [Running Tests](#running-tests)
10. [Evaluation](#evaluation)
11. [CI/CD](#cicd)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                  User (Browser Chat UI)                      │
│              http://localhost:5000                           │
└───────────────────────────┬──────────────────────────────────┘
                            │ user message (POST /chat)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   Input Guardrails                           │
│  (prompt injection detection, length check, SQL-i block)     │
└───────────────────────────┬──────────────────────────────────┘
                            │ sanitised input
                            ▼
┌──────────────────────────────────────────────────────────────┐
│               LangGraph Conversation Graph                   │
│                                                              │
│  classify_intent  (context-aware — uses last bot message)    │
│       │                                                      │
│       ├─ info ──→ retrieve_context (ChromaDB / Weaviate)     │
│       │                │                                     │
│       │           query_dynamic (PostgreSQL)                 │
│       │                │                                     │
│       │           generate_response (LLM)                    │
│       │                                                      │
│       ├─ reservation ──→ collect_reservation (multi-turn, 8 fields)    │
│       │               (live zone availability check)         │
│       │                       │                              │
│       │              [INTERRUPT] human_approval              │
│       │                       │                              │
│       │              finalize_reservation                    │
│       │               (cost calculated & shown)              │
│       │                  │                                   │
│       │              [MCP] write_confirmed_reservation       │
│       │                  │                                   │
│       │          data/confirmed_reservations.txt             │
│       │                                                      │
│       └─ greeting / off_topic ──→ generate_response          │
│                                                              │
│  apply_guardrails (output filter)                            │
└───────────────────────────┬──────────────────────────────────┘
                            │ filtered response (JSON)
                            ▼
                  Browser Chat UI (streaming bubbles)
```

### Three-server architecture (started by `run_frontend.py`)

```
┌──────────────────────────┐      ┌──────────────────────────┐      ┌─────────────────────────────┐
│  Web Chat UI + Graph     │ ───► │  Admin Dashboard          │      │  MCP Server (FastAPI)        │
│  Flask  :5000            │      │  Flask  :5001             │      │  JSON-RPC 2.0  :5002         │
│  /  — chat bubbles       │      │  /admin — approve/reject  │      │  POST /mcp                  │
│  /chat — POST JSON       │      │                           │      │  GET  /health               │
│  /cancel — instant cancel│      │                           │      │                             │
│  /status — poll approval │      │                           │      │  write_confirmed_           │
│  (LangGraph runs here)   │ ─────────────────────────────────────►  │  reservation tool           │
└──────────────────────────┘  on approval, calls MCP tool            │        │                    │
                                                                      │        ▼                    │
                                                                      │  confirmed_reservations.txt │
                                                                      │  Name | Car | Period | Time │
                                                                      └─────────────────────────────┘
```

### Data Architecture

| Data Type | Storage | Examples |
|-----------|---------|----------|
| **Static** | ChromaDB vector store | General info, location, zones, policies, FAQ |
| **Dynamic** | PostgreSQL (via SQLAlchemy) | Prices, availability, working hours, reservations (incl. email, masked card) |

---

## Project Structure

```
task_AI/
├── src/
│   ├── config.py               # Env-based configuration
│   ├── main.py                 # CLI entry point, cleanup scheduler
│   ├── chatbot/                # Agent 1 — User-facing chatbot
│   │   ├── graph.py            # LangGraph StateGraph definition
│   │   ├── nodes.py            # Graph node implementations (8 nodes, context-aware intent)
│   │   ├── state.py            # ChatState TypedDict
│   │   ├── prompts.py          # Prompt templates (8-field order, cost in summaries)
│   │   └── llm.py              # LLM factory (Groq / Gemini / OpenAI / Mock)
│   ├── admin_agent/            # Agent 2 — Admin Notification Agent
│   │   ├── agent.py            # LangChain create_agent (tool-calling)
│   │   ├── tools.py            # notify_admin, poll_for_decision, get_pending_requests
│   │   ├── api_server.py       # Flask REST API + HTML dashboard (port 5001)
│   │   ├── decision_store.py   # Thread-safe shared decision bus between agents
│   │   └── notification.py     # SMTP email sender (HTML email with Approve/Reject buttons)
│   ├── mcp_server/             # MCP Server — confirmed reservation persistence
│   │   ├── server.py           # FastAPI JSON-RPC 2.0 server (port 5002)
│   │   └── client.py           # MCPClient + call_write_confirmed_reservation helper
│   ├── rag/
│   │   ├── vectorstore.py      # ChromaDB / Weaviate setup
│   │   ├── retriever.py        # Similarity search + context formatting
│   │   └── embeddings.py       # Embedding model factory
│   ├── database/
│   │   ├── models.py           # SQLAlchemy ORM models (zone-specific space counts, 500 total)
│   │   └── operations.py       # CRUD, expiry cleanup, cancellation, date-range availability
│   ├── guardrails/
│   │   └── filters.py          # Input/output safety filters
│   ├── reservation/
│   │   └── handler.py          # Field validation & reservation logic
│   ├── reservation_frontend/
│   │   └── app.py              # Browser chat UI Flask app (port 5000)
│   └── evaluation/
│       └── metrics.py          # Precision@K, Recall@K, MRR, faithfulness
├── data/
│   ├── static/
│   │   └── parking_info.json           # Static knowledge base
│   ├── chroma_db/                      # ChromaDB vector store (auto-created)
│   ├── parking.db                      # SQLite fallback (dev only; set DATABASE_URL for PostgreSQL)
│   └── confirmed_reservations.txt      # MCP server text log (auto-created on first approval)
├── tests/
│   ├── conftest.py
│   ├── test_rag.py
│   ├── test_guardrails.py
│   ├── test_database.py
│   ├── test_reservation.py
│   ├── test_evaluation.py
│   ├── test_chatbot.py
│   ├── test_mcp_server.py
│   ├── test_frontend.py
│   └── test_load.py
├── run_frontend.py             # Primary launcher (Chat UI + Admin + MCP — 3 servers)
├── .github/workflows/ci.yml    # GitHub Actions CI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- A free API key from one of the supported LLM providers (Groq recommended — fastest, free tier available)
- Git

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd task_AI
```

### 2. Create & activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **First run note:** The local embedding model (`all-MiniLM-L6-v2`, ~80 MB) is downloaded automatically on first use.

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API key and preferred provider
```

### 5. Start the web chat (recommended)

The database, vector store, and all three servers are started automatically.

```bash
python run_frontend.py
```

This starts:
- **Chat UI** → http://localhost:5000
- **Admin Dashboard** → http://localhost:5001/admin
- **MCP Server** → http://localhost:5002/health

### 5b. Start the CLI chatbot (alternative)

```bash
python src/main.py
```

---

## Usage

### Start the web chat

```bash
python run_frontend.py
```

Open **http://localhost:5000** in your browser. You will see a chat window.

- Admin dashboard: **http://localhost:5001/admin**
- MCP server health: **http://localhost:5002/health**

### Chat interactions

| What you type | What happens |
|---------------|--------------|
| Any natural language question | ParkBot answers using RAG + live DB data |
| `I want to reserve a spot` / `yes` (after offer) | Starts the 8-field reservation flow |
| Any short affirmative after a booking offer | Context-aware routing → reservation |
| Click **❌ Cancel Booking** in header | Opens the instant cancellation modal |
| Click **🔄 New Chat** | Starts a fresh conversation |
| Click **⚙ Admin** | Opens admin dashboard in new tab |

### Example session

```
You: What are the parking prices?
ParkBot: Zone A (Premium): $6.00/hr, $35 daily max
         Zone B (Standard): $3.00/hr, $20 daily max ...

You: I want to reserve a parking space
ParkBot: I'd be happy to help! What is your first name?

You: Alice
ParkBot: Thank you, Alice! What is your last name?

You: Smith
ParkBot: Please provide your vehicle registration plate number.

You: ABC-1234
ParkBot: Which parking zone would you prefer for 2026-06-01 → 2026-06-05?
  • Zone A — Premium, ground floor ($6/hr)     — ✅ 20 spot(s) available
  • Zone B — Standard, levels 2–3 ($3/hr)      — ✅ 18 spot(s) available
  • Zone C — Economy, level 4 ($2.50/hr)       — ❌ Fully booked
  ...

You: B
ParkBot: What date would you like to start your reservation?

You: 2026-06-01
ParkBot: And what date should the reservation end?

You: 2026-06-05
ParkBot: Almost done! Please provide your email address.

You: alice@example.com
ParkBot: Finally, please enter your payment card number.

You: 4111111111111111
ParkBot: Here is a summary of your reservation request:
  Name      : Alice Smith
  Email     : a***@example.com
  Vehicle   : ABC-1234
  Zone      : Zone B
  Start     : 2026-06-01
  End       : 2026-06-05
  Duration  : 4 day(s)
  Est. Cost : $80.00 ($20.00/day × 4 day(s))
  Card      : **** **** **** 1111

[Browser shows ⏳ Waiting for admin approval...]
[Admin opens http://localhost:5001/admin and clicks Approve]

ParkBot: ✅ Your reservation has been APPROVED!
  Reservation Code : SP-N5JCKO
  ...
  Duration         : 4 day(s)
  Total Cost       : $80.00 ($20.00/day × 4 day(s))
  Card             : **** **** **** 1111

[To cancel — click ❌ Cancel Booking in the header, enter SP-N5JCKO]
ParkBot: ✅ Reservation SP-N5JCKO has been cancelled immediately.
         Full refund will be processed within 3–5 business days.
         The parking space has been released back to the pool.
```

### Other CLI commands (alternative to web UI)

| Command | Purpose |
|---------|--------|
| `python src/main.py` | Start CLI chatbot (terminal mode) |
| `python src/main.py --reservations` | Print all reservations table |
| `python src/main.py --init-db` | Seed the PostgreSQL database only |
| `python src/main.py --rebuild` | Rebuild the ChromaDB vector store from scratch |
| `python src/main.py --evaluate` | Retrieval-only RAG evaluation (fast, no LLM) |
| `python src/main.py --evaluate-full` | Full pipeline evaluation including LLM generation |

### Screenshots

**Initial chatbot session & first reservation approval**

![Chatbot startup and first reservation flow](screenshots/ss1.png)

![First reservation approved on admin dashboard](screenshots/ss2.png)

---

**Subsequent reservation approvals — different reservation IDs**

![New reservation approval — unique ID (ss3)](screenshots/ss3.png)

![New reservation approval — unique ID (ss4)](screenshots/ss4.png)

![New reservation approval — unique ID (ss5)](screenshots/ss5.png)

![New reservation approval — unique ID (ss6)](screenshots/ss6.png)

![New reservation approval — unique ID (ss7)](screenshots/ss7.png)

![New reservation approval — unique ID (ss8)](screenshots/ss8.png)

![New reservation approval — unique ID (ss9)](screenshots/ss9.png)

**Reservation UI Created**
![alt text](screenshots/ss11.png)

![alt text](screenshots/ss12.png)

![alt text](screenshots/ss13.png)

**Reservation UI with dashboard**:
![alt text](screenshots/ss14.png)
---

**MCP Server — confirmed reservations text log**

![confirmed_reservations.txt written by the MCP server after admin approval](screenshots/confirmed_reservations_screenshot.png)

---

**Stage 4 — Load & Frontend Tests**

![Load test results — concurrent chatbot, admin, and MCP server threads all passing](screenshots/ss15.png)

![Frontend test results — 21 Flask HTTP tests passing across all chat UI routes](screenshots/ss16.png)

---


## Features by Stage

### Stage 1 — RAG Architecture
- Static knowledge base loaded from `parking_info.json` into ChromaDB
- Embedding via `sentence-transformers/all-MiniLM-L6-v2` (local) or OpenAI
- LangChain `similarity_search_with_score` for top-K retrieval
- LangGraph `StateGraph` with `MemorySaver` checkpointing for multi-turn memory

### Stage 2 — Vector + SQL Database Integration
- **Static** data (general info, location, zones, policies, FAQ) → ChromaDB vector store
- **Dynamic** data (prices, availability, working hours, reservations) → PostgreSQL via SQLAlchemy
- **20 parking spaces per zone** (100 total across zones A–E); all pricing and hours pre-seeded
- Live prices and availability injected into every LLM prompt alongside RAG context
- **Date-range availability tracking:** availability counts reflect active approved reservations for any given date range — not just the raw `is_available` flag
- **Reservation expiry:** Approved reservations are automatically deleted from PostgreSQL when their `end_date` passes. A daemon background thread checks every 60 seconds and immediately frees up the assigned parking bay upon expiry.

### Stage 3 — Interactive Features
- **Context-aware intent classification:** the classifier receives the last bot message as context, so short affirmatives like `"yes"` or `"sure"` correctly route to `reservation` when the bot just offered to book a space
- Intent classification: `info | reservation | greeting | off_topic`
- **Browser-based chat UI** at `http://localhost:5000` — replaces terminal interaction; chat bubbles, typing indicator, markdown rendering, admin shortcut, and cancel modal built in
- Multi-turn reservation form: collects first name, last name, plate number, zone, start date, end date, **email address**, and **payment card number** (8 fields total)
- **Live zone availability during booking:** after the user provides dates, the bot checks how many of the 20 spaces in the chosen zone are free for that period. If the zone is fully booked it shows a live availability table for all zones and asks the user to pick another
- **Estimated cost shown upfront:** both the pre-approval summary and the approved confirmation include duration (days) and total estimated cost e.g. `$80.00 ($20.00/day × 4 day(s))`
- **Flexible date input:** accepts `2026-06-15`, `15/06/2026`, `June 15 2026`, `15 Jun 2026`, and more
- Field-level validation with clear, actionable re-prompt messages
- Date range validation: must be future, end > start, maximum 30 days
- **Human-in-the-loop:** Graph interrupts at `human_approval` node; Agent 2 takes over, notifies admin, and resumes Agent 1 with the decision automatically; the browser polls `/status` every 3 s and delivers the result in the chat window
- **Instant reservation cancellation:** click **❌ Cancel Booking** in the chat header, enter the reservation code — the space is freed immediately and status set to `cancelled` (no admin step needed)
- **Cancellation refund policy** (enforced server-side, cannot be bypassed):
  - Within 24h of booking → ✅ Free cancellation (100% refund)
  - After 24h → ⚠️ 50% refund
- **Reservation history:** `--reservations` CLI flag shows a full table (approved, rejected, cancelled, pending)

### Stage 4 — Guardrails, Pipeline Integration & Production Hardening

**Input filters:**
- Prompt injection / jailbreak patterns (`ignore previous instructions`, `act as`, etc.)
- SQL injection attempts
- Data exfiltration requests (show all users, dump database)
- Message length cap (2000 chars)

**Output filters (before response is shown to user):**
- Non-official email addresses redacted → `[EMAIL REDACTED]`
- Credit card number patterns redacted → `[CARD NUMBER REDACTED]`
- Phone numbers redacted unless they are the official SmartPark numbers — **date strings like `2026-06-01` are explicitly excluded** from phone detection to prevent false positives
- Sensitive internal phrases (admin password, API keys) → entire response blocked with a safe fallback
- Official contact details always preserved

**Agent 2 pipeline integration:**
- `human_approval` node now invokes Agent 2's `notify_admin` tool directly before calling `interrupt()` — no duplicate notifications from the frontend
- Interrupt payload carries the pre-registered `request_code` so both CLI and web paths use the same code
- Web frontend (`/chat`) extracts `request_code` from the interrupt payload instead of generating a new one

**PostgreSQL production engine:**
- `get_engine()` configures connection pooling for PostgreSQL: `pool_pre_ping=True`, `pool_recycle=300`, `pool_size=5`, `max_overflow=10`
- SQLite fallback retains `check_same_thread=False` for dev/test environments
- `datetime.utcnow()` replaced with `datetime.now(timezone.utc)` throughout to eliminate deprecation warnings

**Realistic parking inventory:**
- Database seed updated from 20 spaces per zone (100 total) to zone-specific counts:

| Zone | Type | Spaces |
|------|------|--------|
| A | Premium | 80 |
| B | Standard | 200 |
| C | Economy | 150 |
| D | Compact | 50 |
| E | Accessible | 20 |
| **Total** | | **500** |

**Test coverage expansion:**
- `test_frontend.py` — 21 Flask HTTP tests for the chat UI (index, reset, chat, status, cancel routes)
- `test_load.py` — 10 concurrent load tests for chatbot (10 threads), admin decision store (20 threads), and MCP server (15 threads)
- End-to-end pipeline tests in `test_chatbot.py` for the full approval/rejection flow (skip gracefully in mock LLM env)

### Stage 5 — Evaluation
Metrics implemented in `src/evaluation/metrics.py`:

| Metric | Formula |
|--------|---------|
| **Precision@K** | `|relevant ∩ retrieved[:K]| / K` |
| **Recall@K** | `|relevant ∩ retrieved[:K]| / |relevant|` |
| **MRR** | Mean Reciprocal Rank across queries |
| **Faithfulness** | Lexical overlap between answer and retrieved context |
| **Latency** | Retrieval + generation time in milliseconds |

10 ground-truth queries are included in the evaluation dataset.

---

## MCP Server — Reservation Persistence

Once the administrator approves a reservation, the `finalize_reservation` graph node calls the **SmartPark MCP Server** — a FastAPI service that implements the [Model Context Protocol](https://modelcontextprotocol.io/) over HTTP using JSON-RPC 2.0.

### How it works

1. `finalize_reservation` saves the record to PostgreSQL (primary store)
2. It then calls `call_write_confirmed_reservation()` from `src/mcp_server/client.py`
3. The client POSTs a `tools/call` JSON-RPC request to `POST http://127.0.0.1:5002/mcp`
4. The MCP server validates the API key, sanitises the input, and appends one line to `data/confirmed_reservations.txt`
5. If the MCP server is unreachable (e.g. port conflict), the client automatically falls back to a direct thread-safe file write — the entry is **never silently dropped**

### Output file format

```
Name | Car Number | Reservation Period | Approval Time
----------------------------------------------------------------------
Alice Smith | ABC-1234 | 2026-06-01 to 2026-06-07 | 2026-05-12 06:21:32 UTC
John Doe   | XY-9876  | 2026-07-10 to 2026-07-15 | 2026-05-12 09:44:11 UTC
```
![alt text](screenshots/ss10.png)

### MCP Server endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET`  | `/health` | None | Liveness probe — confirms server is up |
| `POST` | `/mcp`    | Bearer token / `X-API-Key` | JSON-RPC 2.0 MCP messages |

### MCP tool schema

```json
{
  "name": "write_confirmed_reservation",
  "inputSchema": {
    "required": ["full_name", "car_number", "reservation_period", "approval_time"]
  }
}
```

### Security

| Control | Detail |
|---------|--------|
| **Authentication** | Bearer token or `X-API-Key` header; constant-time HMAC comparison |
| **Rate limiting** | 30 requests / 60 s per client IP (sliding window) |
| **Input sanitisation** | Pipe and newline characters stripped to prevent log injection |
| **CORS** | Restricted to `127.0.0.1` origins only |
| **API docs disabled** | No Swagger / ReDoc UI exposed |

### Configuration

```ini
MCP_SERVER_HOST=127.0.0.1    # Bind address
MCP_SERVER_PORT=5002          # TCP port
MCP_API_KEY=your-secret-key   # Change this in production!
RESERVATIONS_FILE_PATH=data/confirmed_reservations.txt
```

---

## Admin Agent — Second Agent

The system includes a **dedicated second LangChain agent** (`src/admin_agent/`) that handles all administrator communication independently from the user-facing chatbot.

### Architecture

| Component | File | Role |
|-----------|------|------|
| **Agent** | `agent.py` | LangChain `create_agent` with tool-calling loop |
| **Tools** | `tools.py` | `notify_admin`, `poll_for_decision`, `get_pending_requests` |
| **Dashboard** | `api_server.py` | Flask REST API + auto-refreshing HTML dashboard |
| **Message Bus** | `decision_store.py` | Thread-safe dict shared between Agent 1 and Agent 2 |
| **Email** | `notification.py` | SMTP HTML email with one-click Approve/Reject links |

### Admin Dashboard

Open `http://localhost:5001/admin` in any browser while the chatbot is running:

- **Pending Reservation Requests** — approve or reject with one click
- **Reservation History** — full table of all reservations with status colour-coding (approved, rejected, cancelled)

> **Note:** Cancellations are **immediate** — no admin action is required. When a user cancels via the chat UI the space is freed instantly and the reservation is marked `cancelled`.

The page auto-refreshes every 10 seconds.

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin` | HTML dashboard |
| `GET` | `/admin/pending` | JSON list of pending approval requests |
| `POST` | `/admin/decide` | `{code, approved, notes}` — decide a reservation |
| `GET` | `/admin/decide?code=X&decision=approve` | One-click from email link |


### Email Notifications (optional)

Set the following in `.env` to enable email:

```ini
ADMIN_EMAIL=admin@yourcompany.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your_app_password
```

Leave `SMTP_HOST` blank to disable email — the REST API dashboard works without it.

---

## Configuration Reference

Copy `.env.example` to `.env` and fill in your values.

### LLM Provider — choose one

**Groq (recommended — fast & free)**
```ini
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
> Free key at [console.groq.com/keys](https://console.groq.com/keys)

| Groq Model | Speed | Notes |
|------------|-------|-------|
| `llama-3.3-70b-versatile` | Fast | Best quality |
| `llama-3.1-8b-instant` | Fastest | Good for testing |
| `mixtral-8x7b-32768` | Fast | Long context |

**Google Gemini**
```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```
> Free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

**OpenAI**
```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

**Offline / Mock (no API key)**
```ini
LLM_PROVIDER=mock
```

---

### All configuration variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | Active LLM: `groq`, `gemini`, `openai`, `mock` |
| `GROQ_API_KEY` | _(empty)_ | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GEMINI_API_KEY` | _(empty)_ | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `OPENAI_API_KEY` | _(empty)_ | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `USE_OPENAI_EMBEDDINGS` | `false` | `true` = OpenAI embeddings; `false` = local HuggingFace |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `VECTOR_STORE_TYPE` | `chroma` | `chroma` (local) or `weaviate` (cloud) |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage directory |
| `DATABASE_URL` | `postgresql://user:pass@localhost:5432/smartpark` | PostgreSQL connection URL (falls back to SQLite when unset) |
| `TOP_K_DOCUMENTS` | `4` | Number of documents to retrieve |
| `CHUNK_SIZE` | `500` | Text chunk size for indexing |
| `CHUNK_OVERLAP` | `50` | Chunk overlap for indexing |
| `LOG_LEVEL` | `WARNING` | Python log level |
| `ADMIN_PASSWORD` | `admin123` | Admin console password |
| `ADMIN_API_HOST` | `localhost` | Admin dashboard host |
| `ADMIN_API_PORT` | `5001` | Admin dashboard port |
| `ADMIN_DECISION_TIMEOUT` | `120` | Seconds to wait for admin decision before terminal fallback |
| `ADMIN_EMAIL` | _(empty)_ | Admin email address for notifications |
| `SMTP_HOST` | _(empty)_ | SMTP server host (leave blank to disable email) |
| `SMTP_PORT` | `587` | SMTP port |
| `SMTP_USE_TLS` | `true` | Enable STARTTLS |
| `SMTP_USER` | _(empty)_ | SMTP login username |
| `SMTP_PASSWORD` | _(empty)_ | SMTP login password / app password |
| `SMTP_FROM` | `parkbot@smartpark.com` | From address for notification emails |
| `MCP_SERVER_HOST` | `127.0.0.1` | MCP server bind address |
| `MCP_SERVER_PORT` | `5002` | MCP server TCP port |
| `MCP_API_KEY` | `smartpark-mcp-secret-key` | Bearer token for MCP API (change in production) |
| `RESERVATIONS_FILE_PATH` | `data/confirmed_reservations.txt` | Path to confirmed reservations text log |

### Weaviate (optional cloud vector store)

```bash
pip install Weaviate-client langchain-Weaviate
```

```ini
VECTOR_STORE_TYPE=Weaviate
Weaviate_API_KEY=your_key
Weaviate_INDEX_NAME=parking-chatbot
Weaviate_ENVIRONMENT=us-east-1-aws
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# By module
pytest tests/test_guardrails.py -v      # Fast unit tests (no external deps)
pytest tests/test_reservation.py -v
pytest tests/test_database.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_rag.py -v             # Downloads embedding model on first run
pytest tests/test_chatbot.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

**Minimum 2 tests per module — current counts:**

| Module | Tests |
|--------|-------|
| `test_guardrails.py` | 30 |
| `test_reservation.py` | 40 |
| `test_database.py` | 20 |
| `test_rag.py` | 13 |
| `test_evaluation.py` | 28 |
| `test_chatbot.py` | 16 |
| `test_mcp_server.py` | 63 |
| `test_frontend.py` | 21 |
| `test_load.py` | 10 |
| **Total** | **241** |

---

## Evaluation

Run the retrieval evaluation suite:

```bash
python src/main.py --evaluate
```

Sample output:

```
============================================================
   SmartPark RAG Evaluation Report
============================================================
  Queries evaluated : 10
  K (top-K)         : 4
  Precision@4       : 0.675
  Recall@4          : 0.850
  MRR               : 0.820
  Faithfulness      : 0.000   ← 0 without LLM generation
  Avg retrieval lat : 42.3 ms
  Avg generation lat: 0.0 ms
============================================================
Report saved to data/evaluation_report.json
```

---

## CI/CD

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push/PR to `main`:

1. **Lint** — Ruff static analysis
2. **Tests** — Matrix: Python 3.10, 3.11, 3.12
3. **Build check** — All modules import cleanly
4. **Coverage** — Uploaded to Codecov

---

## Security Notes

- All SQL operations use SQLAlchemy ORM — no raw string interpolation, zero SQL injection risk
- Input guardrails block prompt injection, SQL injection, and data exfiltration **before** the LLM is invoked
- Output guardrails redact PII and block sensitive internal data before any response reaches the user
- Date strings (e.g. `2026-06-01`) are explicitly excluded from phone-number redaction to prevent false positives
- The admin password is only used in the local CLI session — never transmitted or stored in plaintext in code
- Expired reservations are automatically deleted so the database does not accumulate stale PII
- Cancellation refund policy is enforced server-side — cannot be bypassed from the client
- **MCP Server** uses Bearer-token authentication with constant-time HMAC comparison, a sliding-window rate limiter, and input sanitisation to prevent log-injection attacks; API docs (Swagger/ReDoc) are disabled
- Payment card numbers are **never stored in plain text** — only the masked form (`**** **** **** 1234`) is persisted in the database and shown in any user-facing output
- Guest email addresses are masked (`a***@example.com`) in user-facing confirmations; the full address is accessible only on the admin dashboard for operational use
