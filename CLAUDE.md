# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # install deps (Python >=3.11)
uv run python api.py             # run the API on :8000, Swagger at /docs
```

Testing and lint (CI runs exactly these — see `.github/workflows/docker-ci.yml`):

```bash
pytest -m "not integration and not slow" --timeout=60 --timeout-method=thread   # fast lane
pytest -m "integration or slow" --timeout=120 --timeout-method=thread           # gating lane
ruff check .                     # pinned to ruff==0.16.8 in CI

pytest tests/test_decision_router.py               # single file
pytest tests/test_decision_router.py::test_name    # single test
pytest -p no:randomly                              # disable test-order randomization
```

`PYTHONPATH=.` is set in CI; run pytest from the repo root so the top-level packages import.

Docker parity with CI: `docker build --target ci -t s18share-ci . && docker run --rm s18share-ci`.

## Architecture

**Request path:** `POST /runs` → `routers/runs.py` → `core/run_service.py::process_run` → `core/loop.py::AgentLoop4` → agents → MCP tools. `core/run_service.py::process_resume` restarts a run from its session file.

**`AgentLoop4` (`core/loop.py`, ~1.6k lines) is the orchestrator.** It builds a **NetworkX DAG** (the "plan graph") held in `memory/context.py`, then `_execute_dag` walks it and `_execute_step` runs each node. Understanding a run means reading these two files together — the graph carries `globals_schema`, `memory_context`, and per-node timing as graph/node attributes, so state lives on the graph rather than in the loop.

Replanning, budget downgrades, and failure handling are all methods on this class (`_should_replan`, `_apply_budget_downgrade_if_needed`, `_handle_failures`). Several domain guards (CBC blood-panel, mental-health) are deliberately inlined here as `_is_*_query` / `_enforce_*` hooks that delegate to the decisions layer.

**Agents** are prompt-driven, not subclasses: `agents/base_agent.py::AgentRunner.run_agent(agent_type, ...)` pairs an agent type with a Markdown prompt in `prompts/` (`planner.md`, `retriever.md`, `coder.md`, …). Adding an agent usually means adding a prompt plus config, not a new class.

**MCP layer** — `mcp_servers/multi_mcp.py::MultiMCP` starts every server listed in `mcp_servers/mcp_config.json` and injects their tools into planner agents. Two modes: `strict` (required servers must come up, gates readiness) and legacy (prefers cached metadata in `config/mcp_cache.json`). Per-tool disables live in `config/disabled_tools.json`.

**Config layering** — `config/settings_loader.py` merges `config/settings.json` over `config/settings.defaults.json`; always read settings through that loader's accessors rather than loading the JSON directly. `config/models.json` defines model backends (gemini, ollama, azure-openai, llama-cpp) and `config/profiles/*.json` are named routing arms (`railway-gemini.json`, `local-laptop-gemma.json`, …). `benchmarks/clawbench/` compares arms.

**Other subsystems:** `harness/` (drivers, runtime, store) is a separate execution harness from `core/loop.py`; `memory/` holds run context, Remme extraction, and session summaries; `integrations/` carries tenancy, adapters (wiseai, powerapps), and ingest; `core/decisions/` is the Jev typed-decision router that always falls back to regex guards.

## Conventions that aren't obvious from the code

**Writing an MCP server.** Follow `mcp_servers/server_computer.py` (Cua Driver) and `mcp_servers/server_browserskill.py` (BrowserSkill CLI) — both proxy an external binary and share one pattern:

- Call `configure_mcp_stdio_logging()` from `mcp_servers/stdio_safety.py` at import. Stray log lines on stdout corrupt JSON-RPC frames; this is why it exists.
- Enforce allow/deny policy at a **single chokepoint** reached by every tool path, including any passthrough. Filtering only at tool-discovery leaves the tool reachable by name.
- Append every call to a JSONL audit log under `data/system/`, truncating or redacting argument values.
- **Start successfully when the external binary is absent** and expose a status tool that reports it, so planners can detect the missing capability instead of failing mid-run.
- Register the server in `mcp_servers/mcp_config.json`, then document it in `mcp_servers/README_<name>.md` and add a README.md section.

**Windows/asyncio.** Any entry point that spawns a subprocess must set `WindowsProactorEventLoopPolicy` on `win32` before the loop starts — see the top of `api.py`, `multi_mcp.py`, and the MCP servers. Without it, `asyncio` subprocesses fail on Windows.

**Test markers are auto-assigned by filename.** `tests/conftest.py` holds `_INTEGRATION_STEMS`, `_SLOW_STEMS`, and `_CONTRACT_STEMS` frozensets keyed on file stem. A new slow or integration test file must be added to the right frozenset or it lands in the fast CI lane. Markers are `--strict-markers`, declared in `pytest.ini`.

Some suites (`agent_automated_test.py`, `audit_features.py`, `real_verification_suite.py`) deliberately do **not** match the `test_*.py` glob, so they are never collected by CI — they hit live services. Keep it that way.

**Test doubles speak real protocols.** `tests/fixtures/fake_cua_driver.py` is an actual FastMCP stdio server and `tests/fixtures/fake_bsk.py` is a real CLI stub. Prefer a working stub over mocking the transport — it catches framing and argv bugs that mocks hide.

**Ruff is a narrow bug gate, not a formatter.** `[tool.ruff.lint]` selects only `E722` (bare except) and `F` (pyflakes). Line-length and style rules are deliberately off; this is a legacy codebase and reformatting is out of scope. Don't broaden the rule set casually.

**`data/` is runtime state, not source.** Ingest jobs, workspaces, audit logs, and indexes are written there at runtime and are gitignored. Don't commit them, and don't read them as fixtures.

## Known drift

`pyproject.toml` still declares `name = "S15A"` (the repo is S18Share). Renaming is a tracked but unfinished task — verify no import drift before changing it.
