# S18Share

S18Share is an open-source AI agent runtime and orchestration API. Use `POST /runs` to execute workflows with MCP tools, memory, and RAG. Quick start: `uv sync`, set `GEMINI_API_KEY`, run `uv run python api.py`, then open `/docs`. For full architecture, integration, deployment, and operations details, see `docs/` and `docs/QUICKSTART_5_MIN.md`.

## Fast decision routing (Jev / System One)

Query routing, skill matching, and planner fast-paths can use TypeSafe AI's Jev ("System One") model for typed decisions with calibrated confidence — falling back to the built-in regex guards whenever Jev is unkeyed, unreachable, or low-confidence. Set `JEV_API_KEY` (or `TYPESAFE_API_KEY`) to enable; `S18_DECISIONS_ENABLED=0` disables it at runtime. Tune the `decisions` block in `config/settings.json`, and validate your key with `python scripts/jev_probe.py "state"`.

## Native computer use (Cua Driver)

Planner agents can drive the desktop on Windows/macOS/Linux through `mcp_servers/server_computer.py`, a safety proxy over the [Cua Driver](https://github.com/trycua/cua). It discovers the driver's tools at runtime and re-exposes them as `computer_<name>` (`computer_click`, `computer_type_text`, and so on) behind a default denylist, an optional allowlist, and a JSONL audit log at `data/system/computer_audit.jsonl`. Install the driver first (Windows: `irm https://cua.ai/driver/install.ps1 | iex`); without it the server still starts and exposes only `computer_driver_status` and `computer_refresh_tools`. Computer use is a vision task, so route these runs to the Gemini arm or another strong vision model rather than the small Ollama default. Tuning knobs (`S18_COMPUTER_TOOL_DENYLIST`, `S18_COMPUTER_PERMISSION_MODE`, and the rest) are documented in `mcp_servers/README_computer.md`.
