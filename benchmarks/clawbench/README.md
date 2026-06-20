# ClawBench benchmark for S18

Run [ClawBench](https://github.com/openclaw/clawbench) Core v1 tasks against the [S18](https://github.com/riteshverma/s18) agent runtime. Task definitions, workspace fixtures, and trace-based scoring come from ClawBench; agent execution uses S18's `AgentLoop4` and MCP stack.

## Prerequisites

1. Clone ClawBench next to this repo (or anywhere):

```bash
git clone https://github.com/openclaw/clawbench.git
```

2. Install ClawBench into the same Python environment as S18:

```bash
pip install -e "C:/path/to/clawbench"
```

3. Configure S18 model credentials (for example `GEMINI_API_KEY` in `.env`, or a running Ollama instance matching `config/settings.json`).

4. Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CLAWBENCH_ROOT` | `~/Downloads/clawbench` | Path to ClawBench checkout |
| `CLAWBENCH_S18_MAX_STEPS` | `12` | Planner steps per S18 run |
| `CLAWBENCH_S18_MAX_LIFELINES` | `4` | Retries per plan step |
| `CLAWBENCH_KEEP_WORKSPACES` | unset | Set to `1` to keep task workspaces for debugging |
| `CLAWBENCH_USE_OLLAMA` | auto when `S18_PROFILE=local-laptop-gemma` | Force Ollama instead of Gemini when `GEMINI_API_KEY` is set |
| `CLAWBENCH_OLLAMA_WAIT_SECONDS` | `120` | Seconds to wait for `127.0.0.1:11434` after auto-starting Docker |

### Host-side Ollama (`s18share-ollama`)

When running from the host (not inside Compose), rewrite Docker DNS to localhost — the runner does this automatically when `CLAWBENCH_USE_OLLAMA=1` or `S18_PROFILE=local-laptop-gemma`.

```powershell
docker start s18share-ollama
curl http://127.0.0.1:11434/api/tags

$env:CLAWBENCH_USE_OLLAMA="1"
python benchmarks/clawbench/runner.py -t t1-fs-quick-note --runs 1 -o benchmarks/clawbench/results/smoke_t1_ollama.json
```

## Quick smoke test

From the S18 repo root:

```bash
python benchmarks/clawbench/runner.py -t t1-fs-quick-note --runs 1
```

This runs the tier-1 “quick note” task once and writes JSON under `benchmarks/clawbench/results/`.

## Core v1 (19 tasks)

```bash
python benchmarks/clawbench/runner.py --core-v1 --runs 3 -o benchmarks/clawbench/results/s18_core_v1.json
```

Task IDs match [tasks-public/MANIFEST.yaml](https://github.com/openclaw/clawbench/blob/main/tasks-public/MANIFEST.yaml).

## Scoring axes

ClawBench scores each run on four axes (see ClawBench README):

- **Completion (40%)** — deterministic verifiers (`pytest`, file checks, scripts in task assets)
- **Trajectory (30%)** — tool-call trace quality (read-before-write, self-verification)
- **Behavior (20%)** — planning, progress, safety patterns in the transcript
- **Judge (advisory)** — optional LLM judge; disabled by default for S18 runs (no OpenClaw gateway required)

## Limitations

- S18 maps sandbox `read_workspace_file` / `write_workspace_file` tools to ClawBench trajectory names (`read_file`, `apply_patch`). Post-run materialization writes plan outputs when agents skip tool calls.
- Tasks that require OpenClaw memory/session/cron gateway checks are scored on filesystem/execution checks only; memory/session assertions may be skipped when the stub gateway is used.
- Browser-heavy tier-2+ tasks need S18's browser MCP server available and are slower on first run.

## Compare with OpenClaw baseline

To compare against OpenClaw on the same tasks, run ClawBench's native harness (requires OpenClaw gateway):

```bash
cd clawbench
clawbench run --model anthropic/claude-opus-4-6 -t t1-fs-quick-note --runs 3 -o results/opus_smoke.json
```

Use the same task IDs and run counts for a fair comparison on completion scores; treat adapter differences as expected when interpreting trajectory/behavior gaps.
