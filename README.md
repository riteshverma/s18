# S18Share

S18Share is an open-source AI agent runtime and orchestration API. Use `POST /runs` to execute workflows with MCP tools, memory, and RAG. Quick start: `uv sync`, set `GEMINI_API_KEY`, run `uv run python api.py`, then open `/docs`. For full architecture, integration, deployment, and operations details, see `docs/` and `docs/QUICKSTART_5_MIN.md`.

## Fast decision routing (Jev / System One)

Query routing, skill matching, and planner fast-paths can use TypeSafe AI's Jev ("System One") model for typed decisions with calibrated confidence — falling back to the built-in regex guards whenever Jev is unkeyed, unreachable, or low-confidence. Set `JEV_API_KEY` (or `TYPESAFE_API_KEY`) to enable; `S18_DECISIONS_ENABLED=0` disables it at runtime. Tune the `decisions` block in `config/settings.json`, and validate your key with `python scripts/jev_probe.py "state"`.
