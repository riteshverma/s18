# S18Share

**Agentic AI** – A FastAPI backend for AI agents with memory, RAG, MCP servers, scheduled jobs, and a skills system.

- **Python:** 3.11+
- **Version:** 0.2.0

---

## Features

- **Agent loop** – Multi-step planning and execution with retries and circuit breakers
- **REMME (Remember Me)** – User memory and preferences: extraction, staging, normalizer, belief updates, and hubs (Preferences, Operating Context, Soft Identity). See [remme/ARCHITECTURE.md](remme/ARCHITECTURE.md).
- **RAG** – Document indexing and search (FAISS + optional BM25), chunking, and ingestion
- **MCP servers** – RAG, browser, sandbox, and configurable external servers
- **Scheduler** – Cron-style jobs with skill routing (e.g. Market Analyst, System Monitor, Web Clipper) and inbox integration
- **Skills** – Pluggable skills with intent matching and run/success hooks
- **Streaming** – SSE endpoint for real-time events from the event bus
- **Config** – Centralized settings in `config/` (Ollama, models, RAG, agent, REMME)

---

## Quick start

### 1. Install dependencies

Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### 2. Environment variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (used for agents, apps, and some MCP tools when configured) |

Optional:

- **Ollama** – Default config points to `http://127.0.0.1:11434`. Run [Ollama](https://ollama.ai) locally for embedding, semantic chunking, and optional agent overrides.
- **Git** – Required for GitHub explorer features; the API will warn at startup if Git is not found.

### 3. Run the API

```bash
uv run python api.py
```

Or:

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

- **API:** http://localhost:8000  
- **Health:** http://localhost:8000/health  
- **Docs:** http://localhost:8000/docs  

The app expects a frontend at `http://localhost:5173` (CORS is configured for it).

---

## Project structure

| Path | Description |
|------|-------------|
| `api.py` | FastAPI app, lifespan, CORS, router includes |
| `core/` | Agent loop, scheduler, event bus, circuit breaker, persistence, model manager, skills |
| `remme/` | Memory and preferences pipeline (extractor, store, hubs, normalizer) |
| `routers/` | API routes: RAG, remme, agent, chat, runs, stream, cron, skills, inbox, etc. |
| `mcp_servers/` | MCP server implementations (RAG, browser, sandbox, multi_mcp) |
| `config/` | Settings loader, `settings.json`, `settings.defaults.json`, agent config |
| `data/` | Inbox DB, system jobs/snapshot, RAG documents |
| `memory/` | Execution context, remme index, debug logs |
| `agents/` | Agent runner and config-driven agents |
| `scripts/` | Utility and test scripts |
| `tests/` | Verification and integration-style tests |

---

## Configuration

- **Main settings:** `config/settings.json` (created from `config/settings.defaults.json` if missing).
- **Agent prompts and MCP:** `config/agent_config.yaml`.
- **REMME extraction prompt and options:** under `remme` in settings.

---

## License

See repository or project metadata for license information.
