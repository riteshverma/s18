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
| --- | --- |
| `GEMINI_API_KEY` | Google Gemini API key (used for agents, apps, and some MCP tools when configured) |
| `AUTH_ENABLED` | Enable backend bearer-token verification (`true`/`false`) |
| `S18_AUTH_ENABLED` | Docker-only override mapped to `AUTH_ENABLED` for this service (prevents cross-repo env collisions) |
| `SUPABASE_URL` | Supabase project URL (used for auth verify and optional logging) |
| `SUPABASE_ANON_KEY` | Supabase anon key (optional for frontend/public client flows) |
| `SUPABASE_JWT_AUDIENCE` | Expected access-token `aud` claim for backend verification (default `authenticated`) |
| `SUPABASE_LOGGING_ENABLED` | Enable request/result persistence to Supabase tables (`true`/`false`) |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key for backend writes to Supabase tables |

Optional:

- **Ollama** – Default config points to `http://127.0.0.1:11434`. Run [Ollama](https://ollama.ai) locally for embedding, semantic chunking, and optional agent overrides.
- **Git** – Required for GitHub explorer features; the API will warn at startup if Git is not found.
- **WISE_MOCKEHR_BASE_URL** – Base URL of wise-ai Mock EHR API. When set, the EHRDataMinerAgent's mockehr MCP fetches `/patients/{id}` and `/patients/{id}/labs` from wise-ai for end-to-end integration. Example: `http://backend:8000` (Docker) or `http://localhost:8000` (host).

### Supabase integration contract (S18)

- Frontend/S18 performs login with Supabase Auth and sends `Authorization: Bearer <access_token>`.
- Backend verifies the JWT on protected endpoints using Supabase JWKS (`/auth/v1/.well-known/jwks.json`) with issuer/audience checks (no backend-managed Supabase session).
- If S18 is called through another backend/proxy, it also accepts `X-Forwarded-Authorization: Bearer <access_token>`.
- Optional persistence can write to two Supabase tables:
  - `ehr_request_log` (inbound request/audit trail)
  - `ehr_clinical_result` (normalized RAC/CBC/ABDM/FHIR-aligned outcome)
- Reference SQL schema: `docs/supabase_ehr_schema.sql`
- Quick environment/table readiness check:

```bash
python scripts/check_supabase_integration.py
```

### 3. Run the API

```bash
uv run python api.py
```

Or:

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

- **API:** [http://localhost:8000](http://localhost:8000)  
- **Health:** [http://localhost:8000/health](http://localhost:8000/health)  
- **Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)  

The app expects a frontend at `http://localhost:5173` (CORS is configured for it).

---

## Docker

### 1. Prepare environment file

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env`.

### 2. Run API only (host Ollama)

Set in `.env`:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Then:

```bash
docker compose up --build -d api
```

### 3. Run API + Ollama in Docker

Keep in `.env`:

```bash
OLLAMA_BASE_URL=http://ollama:11434
```

Then:

```bash
docker compose up --build -d
```

### 4. Verify (Docker mapping)

- API: [http://localhost:8001](http://localhost:8001)
- Health: [http://localhost:8001/health](http://localhost:8001/health)
- Docs: [http://localhost:8001/docs](http://localhost:8001/docs)
- Prometheus scrape: [http://localhost:8001/metrics/prometheus](http://localhost:8001/metrics/prometheus)

Persistent state is stored on host-mounted folders:

- `data/`
- `memory/`
- `config/`
- `mcp_servers/faiss_index/`

---

## Monitoring (Dev + Staging Baseline)

Monitoring assets are in `monitoring/` and run as an additive stack:

- Prometheus config/rules: `monitoring/prometheus/`
- Alertmanager config: `monitoring/alertmanager/`
- Grafana provisioning/dashboard: `monitoring/grafana/`

### Start API + Monitoring

```bash
docker compose up --build -d api
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

If you want local Ollama in Docker too:

```bash
docker compose up --build -d
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

### Validate Monitoring

- Prometheus target page: [http://localhost:9090/targets](http://localhost:9090/targets)
- Alertmanager: [http://localhost:9093](http://localhost:9093)
- Grafana: [http://localhost:3000](http://localhost:3000) (`admin` / `admin`)

Expected key metric families:

- `wiseai_api_requests_total`
- `wiseai_api_requests_success_total`
- `wiseai_api_request_latency_ms`
- `wiseai_orchestrator_runs_total`
- `wiseai_orchestrator_run_latency_ms`
- `wiseai_rag_requests_total`
- `wiseai_mcp_tool_calls_total`
- `wiseai_memory_operations_total`

### Port Overrides

If local ports conflict, override host mappings in `monitoring/docker-compose.monitoring.yml`:

- Prometheus: `9090`
- Alertmanager: `9093`
- Grafana: `3000`

---

### CI Docker target

This repo now includes a dedicated Docker build target for CI:

```bash
docker build --target ci -t s18share-ci .
docker run --rm s18share-ci
```

The CI target uses pinned dependencies from `requirements-ci.txt` (exported from `uv.lock`) and runs a quick compile sanity check.

---

## Project structure

| Path | Description |
| ---- | ----------- |
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
