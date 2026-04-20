# 5-minute quickstart: clone → API → agent (with UI)

Single path from zero to a **running agent** you can drive from a browser. No Docker required.

## What you need

- **Git**, **Python 3.11+**, and **[uv](https://github.com/astral-sh/uv)** installed
- A **Google Gemini** API key ([AI Studio](https://aistudio.google.com/apikey))

## The five minutes

| Step | Action |
| --- | --- |
| **0 — Clone** | `git clone <this-repo-url> && cd S18Share` |
| **1 — Deps** | `uv sync` |
| **2 — Env** | Copy `.env.example` to `.env` and set `GEMINI_API_KEY=...`. Leave `AUTH_ENABLED=false` for local tryouts. |
| **3 — Config** | Ensure `config/settings.json` exists (copy from `config/settings.defaults.json` if the loader has not created it yet). |
| **4 — Run** | `uv run python api.py` |
| **5 — UI** | Open **[http://localhost:8000/docs](http://localhost:8000/docs)** (Swagger UI). |

With **`AUTH_ENABLED=false`**, protected routes use a built-in dev user so you do not need Supabase for your first run.

## Trigger the agent from the UI

1. In Swagger, open **`POST /runs`** → **Try it out**.
2. Use a minimal body, for example:

```json
{
  "query": "Summarize what you can do in one paragraph, then suggest a next step."
}
```

3. **Execute**. Copy the returned `id` if you want to poll **`GET /runs/{run_id}`** or use other run endpoints listed under the same tag.

That is the full loop: **HTTP API → canonical run → agent loop → tools (including MCP)**.

## Optional: companion web app on port 5173

The API allows CORS for `http://localhost:5173`. If you run a separate Vite (or Wise-AI) frontend that targets this backend, point it at `http://localhost:8000` and send `Authorization: Bearer <supabase_access_token>` when **`AUTH_ENABLED=true`**.

## Optional: Supabase (auth + logging)

When you are ready for real users:

- Set `AUTH_ENABLED=true`, `SUPABASE_URL`, and related keys as in the main [README](../README.md#quick-start).
- Optionally enable `SUPABASE_LOGGING_ENABLED` and the service role key for audit/clinical tables.

## If something fails

- **`/health`** should return OK: [http://localhost:8000/health](http://localhost:8000/health)
- **Ollama**: default configs expect embeddings/RAG tooling; if you do not run Ollama yet, some RAG-heavy paths may warn—core `/runs` with Gemini still works for a first agent turn.
- **Windows**: the app sets `WindowsProactorEventLoopPolicy` for MCP subprocesses; use PowerShell or cmd from the repo root.

For architecture context, see the diagram in the root [README](../README.md#architecture-at-a-glance).
