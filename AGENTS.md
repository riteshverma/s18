# AGENTS.md

## Cursor Cloud specific instructions

### Overview

S18Share is a FastAPI backend for agentic AI with memory (REMME), RAG, MCP servers, scheduled jobs, and a skills system. Single service: `api.py` on port 8000 (dev) or 8001 (Docker).

### Running the API (dev mode)

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Or equivalently: `uv run python api.py` (runs on port 8001 with reload).

### Key endpoints

- Health: `GET /health`
- Swagger docs: `GET /docs`
- Metrics: `GET /metrics/prometheus`, `GET /metrics/dashboard`

### Environment

- Copy `.env.example` to `.env` before first run. Auth is disabled by default (`AUTH_ENABLED=false`).
- `GEMINI_API_KEY` is needed for agent runs to complete (LLM-powered). Without it the API starts fine but agent runs will fail.
- Ollama is optional; the default config points to `http://ollama:11434` (Docker). For local dev without Docker, change to `http://127.0.0.1:11434` in `config/settings.json` or via `OLLAMA_BASE_URL` env var. The API starts without Ollama but embedding/chunking features will be degraded.

### Tests

Tests live in `tests/` and use `unittest` (no pytest). Run them with `PYTHONPATH` set:

```bash
PYTHONPATH=/workspace uv run python tests/audit_features.py -v
PYTHONPATH=/workspace uv run python tests/test_supabase_auth.py -v
```

Note: `tests/audit_features.py` requires the API to be running on port 8000. The `test_supabase_auth.py` suite has a known flaky test (`test_missing_bearer_token_returns_401`).

### Lint / build

No dedicated linter is configured. Use `python -m compileall -q . -x '.venv'` for a compile-check across the codebase.

### Gotchas

- The Yahoo Finance MCP server (`yfinance_mcp_server.py`) fails to start at boot; this is a known issue (script not found) and does not affect other MCP servers (browser, rag, sandbox, mockehr all connect fine).
- `config/settings.json` is auto-created from `config/settings.defaults.json` on first run if missing.
- The `data/`, `memory/`, and `config/` directories hold persistent state; they are created automatically.
