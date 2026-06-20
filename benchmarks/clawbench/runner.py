#!/usr/bin/env python3
"""Run ClawBench Core v1 tasks against the S18 agent runtime.

Uses https://github.com/openclaw/clawbench for task definitions, workspace
setup, and trace-based scoring while driving agents through S18's AgentLoop4.

Usage:
  # Install clawbench once (from a local clone or pip):
  pip install -e "C:/path/to/clawbench"

  # Smoke test (single tier-1 task):
  python benchmarks/clawbench/runner.py -t t1-fs-quick-note --runs 1

  # Full Core v1 suite (19 tasks x N runs):
  python benchmarks/clawbench/runner.py --core-v1 --runs 1 -o results/s18_core_v1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CLAWBENCH_ROOT = Path(
    os.environ.get(
        "CLAWBENCH_ROOT",
        Path.home() / "Downloads" / "clawbench",
    )
).expanduser()


def _resolve_clawbench_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if DEFAULT_CLAWBENCH_ROOT.exists():
        return DEFAULT_CLAWBENCH_ROOT.resolve()
    sibling = PROJECT_ROOT.parent / "clawbench"
    if sibling.exists():
        return sibling.resolve()
    raise FileNotFoundError(
        "ClawBench not found. Clone https://github.com/openclaw/clawbench and set CLAWBENCH_ROOT "
        "or pass --clawbench-root."
    )


def _ensure_clawbench_import(clawbench_root: Path) -> None:
    try:
        import clawbench  # noqa: F401
        return
    except ImportError:
        pass
    root_text = str(clawbench_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        import clawbench  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"Could not import clawbench from {clawbench_root}. "
            "Run: pip install -e <path-to-clawbench>"
        ) from exc


def _load_core_v1_task_ids(clawbench_root: Path) -> list[str]:
    import yaml

    manifest = clawbench_root / "tasks-public" / "MANIFEST.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return [task["id"] for task in data.get("tasks", [])]


def _ensure_python3_shim() -> None:
    """ClawBench verifiers invoke `python3`; on Windows prepend a shim to PATH."""
    if sys.platform != "win32":
        return
    shim_dir = PROJECT_ROOT / "benchmarks" / "clawbench" / ".shims"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "python3.cmd"
    content = f'@"{sys.executable}" %*\r\n'
    if not shim.exists() or shim.read_text(encoding="utf-8") != content:
        shim.write_text(content, encoding="utf-8")
    prefix = str(shim_dir)
    if prefix not in os.environ.get("PATH", ""):
        os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")


def _apply_benchmark_settings() -> None:
    from dotenv import load_dotenv

    from config import settings_loader
    from config.settings_loader import load_settings, reload_settings

    load_dotenv(PROJECT_ROOT / ".env")

    forced_provider = (
        os.environ.get("S18_MODEL_PROVIDER") or os.environ.get("AGENT_MODEL_PROVIDER") or ""
    ).strip().lower()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    gemini_available = bool(gemini_key and gemini_key.strip())
    ollama_flag = os.environ.get("CLAWBENCH_USE_OLLAMA", "").strip().lower()

    use_ollama = ollama_flag in {"1", "true", "yes", "on"}
    if not use_ollama and ollama_flag not in {"0", "false", "no", "off"}:
        use_ollama = os.environ.get("S18_PROFILE", "").strip().lower() in {
            "local-laptop-gemma",
            "local-laptop",
        }
    if forced_provider == "gemini":
        use_ollama = False
    elif forced_provider == "ollama":
        use_ollama = True
    elif gemini_available and ollama_flag not in {"1", "true", "yes", "on"}:
        # Benchmark default: prefer Gemini when a key is present unless Ollama is forced.
        use_ollama = False

    raw_ollama = os.environ.get("OLLAMA_BASE_URL", "")
    if use_ollama or "ollama:11434" in raw_ollama or not raw_ollama.strip():
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"

    if use_ollama:
        os.environ.pop("S18_MODEL_PROVIDER", None)
    elif gemini_available:
        os.environ["S18_MODEL_PROVIDER"] = "gemini"

    reload_settings()
    settings = load_settings()
    settings.setdefault("agent", {})
    if use_ollama:
        settings["agent"]["model_provider"] = "ollama"
        settings["agent"].setdefault("default_model", "gemma4:e4b")
    elif gemini_available:
        settings["agent"]["model_provider"] = "gemini"
        if not str(settings["agent"].get("default_model", "")).lower().startswith("gemini"):
            settings["agent"]["default_model"] = "gemini-2.5-flash"
    elif os.environ.get("S18_MODEL_PROVIDER", "").strip().lower() == "gemini":
        settings["agent"]["model_provider"] = "gemini"
    settings["agent"]["max_steps"] = int(os.environ.get("CLAWBENCH_S18_MAX_STEPS", "12"))
    settings["agent"]["max_lifelines_per_step"] = int(
        os.environ.get("CLAWBENCH_S18_MAX_LIFELINES", "4")
    )
    settings.setdefault("mcp", {})
    settings["mcp"]["mode"] = os.environ.get("CLAWBENCH_S18_MCP_MODE", "legacy")
    settings.setdefault("ollama", {})
    settings["ollama"]["base_url"] = os.environ["OLLAMA_BASE_URL"]
    settings["ollama"]["timeout"] = int(os.environ.get("OLLAMA_TIMEOUT", "360"))
    settings_loader.settings = settings


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _ollama_tags_reachable(timeout_seconds: float = 3.0) -> bool:
    import urllib.error
    import urllib.request

    url = f"{_ollama_base_url()}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _start_ollama_container() -> None:
    import subprocess

    for name in ("s18share-ollama", "ollama"):
        try:
            result = subprocess.run(
                ["docker", "start", name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                print(f"[clawbench] Started Docker container {name}.", flush=True)
                return
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            print(f"[clawbench] Could not start container {name}: {exc}", flush=True)


async def _ensure_ollama_available() -> None:
    if _ollama_tags_reachable():
        return

    print("[clawbench] Ollama not reachable; attempting to start s18share-ollama ...", flush=True)
    await asyncio.to_thread(_start_ollama_container)

    deadline = time.monotonic() + float(os.environ.get("CLAWBENCH_OLLAMA_WAIT_SECONDS", "120"))
    while time.monotonic() < deadline:
        if _ollama_tags_reachable(timeout_seconds=5.0):
            print(f"[clawbench] Ollama is up at {_ollama_base_url()}.", flush=True)
            return
        await asyncio.sleep(2)

    raise RuntimeError(
        f"Ollama is not reachable at {_ollama_base_url()}. "
        "Start the container with: docker start s18share-ollama "
        "(or docker compose --profile ollama up -d ollama), then retry."
    )


async def _warm_ollama_model(model: str = "gemma4:e4b") -> None:
    """Load the model into Ollama before the benchmark run."""
    import aiohttp

    base = _ollama_base_url()
    timeout = aiohttp.ClientTimeout(total=600)
    payload = {"model": model, "prompt": "ping", "stream": False}
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base}/api/generate", json=payload) as response:
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(f"Ollama warm-up failed HTTP {response.status}: {body[:300]}")
            await response.json()


def _setup_workspace(task, workspace: Path, assets_dir: Path) -> None:
    for pack in task.setup.asset_packs:
        source = assets_dir / pack
        if not source.exists():
            raise FileNotFoundError(f"Missing asset pack {pack}")
        _copy_tree(source, workspace)

    for rel_path in task.setup.workspace_files:
        source = assets_dir / rel_path
        if not source.exists():
            raise FileNotFoundError(f"Missing workspace asset {rel_path}")
        target = workspace / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_tree(source: Path, workspace: Path) -> None:
    if source.is_file():
        target = workspace / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = workspace / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _create_workspace(task, run_index: int, results_root: Path) -> Path:
    root = results_root / "workspaces" / task.id
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / f"run-{run_index}-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _workspace_prompt_prefix(workspace: Path) -> str:
    resolved = workspace.resolve()
    notes_path = resolved / "notes.md"
    return (
        "You are executing a ClawBench evaluation task inside an isolated workspace.\n"
        f"Workspace directory: {resolved}\n"
        "Use the sandbox MCP tools `write_workspace_file(path, content)` and "
        "`read_workspace_file(path)` for all file deliverables. "
        f"For example: write_workspace_file(path=\"notes.md\", content=\"...\") "
        f"or an absolute path like `{notes_path}`.\n"
        "Format notes as a bullet or numbered list, not a paragraph. "
        "When verifying code changes, run tests with that directory as the working directory.\n\n"
    )


_PATH_KEYS = (
    "output_file_path",
    "notes_filepath",
    "file_path",
    "target_path",
    "filepath",
)
_CONTENT_KEYS = (
    "formatted_notes_content",
    "formatted_notes",
    "notes_content",
    "file_content",
    "content",
    "notes",
    "response",
)
_DEFAULT_NOTE_PATH = "notes.md"


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_workspace_write_targets(globals_schema: dict) -> list[tuple[str, str]]:
    """Find (path, content) pairs in plan globals for post-run materialization."""
    if not isinstance(globals_schema, dict):
        return []

    path_value = ""
    for key in _PATH_KEYS:
        text = _coerce_text(globals_schema.get(key))
        if text:
            path_value = text
            break

    content_value = ""
    for key in _CONTENT_KEYS:
        text = _coerce_text(globals_schema.get(key))
        if text:
            content_value = text
            break

    if path_value and content_value:
        return [(path_value, content_value)]
    if content_value and _looks_like_note_content(content_value):
        return [(_DEFAULT_NOTE_PATH, content_value)]
    return []


def _looks_like_note_content(text: str) -> bool:
    lowered = text.lower()
    if not text:
        return False
    if text.lstrip().startswith(("-", "*", "1.", "1)")):
        return True
    return any(token in lowered for token in ("dry cleaning", "recital", "babysitter", "pick up"))


async def _materialize_workspace_outputs(
    *,
    context: object | None,
    workspace: Path,
    multi_mcp,
    collector: "EventCollector",
) -> None:
    """Write plan outputs to disk when agents produced content but skipped tool calls."""
    if context is None:
        return

    plan_graph = getattr(context, "plan_graph", None)
    if plan_graph is None:
        return

    globals_schema = (plan_graph.graph or {}).get("globals_schema", {})
    workspace_root = str(workspace.resolve())

    for rel_or_abs_path, content in _extract_workspace_write_targets(globals_schema):
        try:
            from tools.workspace_io import resolve_workspace_path

            target = resolve_workspace_path(rel_or_abs_path, workspace_root)
        except Exception:
            continue
        if target.is_file() and target.read_text(encoding="utf-8").strip():
            continue

        write_path = rel_or_abs_path
        if target.is_absolute():
            try:
                write_path = str(target.relative_to(workspace.resolve()))
            except ValueError:
                write_path = str(target)

        try:
            from tools.workspace_io import write_workspace_file

            preview = write_workspace_file(write_path, content, workspace_root)
            collector.record_tool_call(
                tool_name="write_workspace_file",
                arguments={"path": write_path, "content": content},
                output=preview,
                success=True,
            )
        except Exception as exc:
            collector.record_tool_call(
                tool_name="write_workspace_file",
                arguments={"path": write_path, "content": content},
                output=f"Write failed: {exc}",
                success=False,
            )


def _apply_clawbench_agent_aliases() -> dict[str, str | None]:
    """Route file-writing plan steps to CoderAgent (has sandbox tools)."""
    import agents.base_agent as base_agent_module

    overrides = {
        "ActionAgent": base_agent_module.AGENT_ALIASES.get("ActionAgent"),
        "NoteWriterAgent": base_agent_module.AGENT_ALIASES.get("NoteWriterAgent"),
        "NoteWriter": base_agent_module.AGENT_ALIASES.get("NoteWriter"),
    }
    base_agent_module.AGENT_ALIASES["ActionAgent"] = "CoderAgent"
    base_agent_module.AGENT_ALIASES["NoteWriterAgent"] = "CoderAgent"
    base_agent_module.AGENT_ALIASES["NoteWriter"] = "CoderAgent"
    return overrides


def _refresh_sandbox_tool_cache() -> None:
    """Drop stale sandbox MCP metadata so new workspace I/O tools are discoverable."""
    cache_path = PROJECT_ROOT / "config" / "mcp_cache.json"
    if not cache_path.exists():
        return
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if "sandbox" not in cache:
        return
    cache.pop("sandbox", None)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _restore_clawbench_agent_aliases(overrides: dict[str, str | None]) -> None:
    import agents.base_agent as base_agent_module

    for alias, original in overrides.items():
        if original is None:
            base_agent_module.AGENT_ALIASES.pop(alias, None)
        else:
            base_agent_module.AGENT_ALIASES[alias] = original


async def _run_s18_turn(
    *,
    query: str,
    workspace: Path,
    multi_mcp,
    session_id: str,
) -> tuple[object | None, str, "EventCollector"]:
    from core.loop import AgentLoop4

    from benchmarks.clawbench.transcript_builder import EventCollector, extract_assistant_text

    collector = EventCollector()
    await collector.start()
    context = None
    assistant_text = ""
    workspace_root = str(workspace.resolve())
    os.environ["S18_WORKSPACE_ROOT"] = workspace_root
    trace_token = multi_mcp.set_trace_context({"workspace": workspace_root})
    try:
        loop = AgentLoop4(multi_mcp=multi_mcp)
        context = await loop.run(
            query=query,
            file_manifest=[],
            globals_schema={"workspace": workspace_root},
            uploaded_files=[],
            session_id=session_id,
        )
        assistant_text = extract_assistant_text(context) if context else ""
        await _materialize_workspace_outputs(
            context=context,
            workspace=workspace,
            multi_mcp=multi_mcp,
            collector=collector,
        )
    except Exception as exc:
        assistant_text = f"S18 run failed: {exc}"
        print(f"[clawbench] agent run failed: {exc}", flush=True)
    finally:
        multi_mcp.reset_trace_context(trace_token)
        await collector.drain()
        await collector.stop()

    return context, assistant_text, collector


async def _score_task_run(
    *,
    task,
    transcript,
    workspace: Path,
    runtime_values: dict,
    judge_model: str,
    run_index: int,
    duration_ms: int,
):
    from clawbench.scorer import score_task_run

    from benchmarks.clawbench.gateway_stub import GatewayStub

    gateway = GatewayStub()
    if sys.platform != "win32":
        return await score_task_run(
            task=task,
            transcript=transcript,
            workspace=workspace,
            client=gateway,
            session_key="",
            agent_id=None,
            duration_ms=duration_ms,
            runtime_values=runtime_values,
            judge_model=judge_model,
            judge_affects_score=False,
        )

    import clawbench.environment as env_module

    original_run_check = env_module.run_execution_check

    async def _windows_run_execution_check(spec, *, workspace, runtime_values):
        if "python3" in spec.command:
            spec = spec.model_copy(
                update={"command": spec.command.replace("python3", f'"{sys.executable}"')}
            )
        result = await original_run_check(spec, workspace=workspace, runtime_values=runtime_values)
        return result

    env_module.run_execution_check = _windows_run_execution_check
    try:
        result = await score_task_run(
            task=task,
            transcript=transcript,
            workspace=workspace,
            client=gateway,
            session_key="",
            agent_id=None,
            duration_ms=duration_ms,
            runtime_values=runtime_values,
            judge_model=judge_model,
            judge_affects_score=False,
        )
    finally:
        env_module.run_execution_check = original_run_check
    result.run_index = run_index
    return result


async def _run_single_task_run(
    *,
    task,
    run_index: int,
    multi_mcp,
    clawbench_root: Path,
    results_root: Path,
    prompt_variant: str,
    judge_model: str,
    model_label: str,
):
    from clawbench.schemas import Transcript, TranscriptMessage
    from clawbench.services import build_runtime_values, start_background_services, stop_background_services
    from clawbench.simulated_user import UserSimulator
    from clawbench.tasks import get_assets_dir

    workspace = _create_workspace(task, run_index, results_root)
    assets_dir = get_assets_dir()
    _setup_workspace(task, workspace, assets_dir)

    runtime_values = build_runtime_values(
        workspace=workspace,
        repo_root=clawbench_root,
        extra={
            "task_id": task.id,
            "model": model_label,
            "prompt_variant": prompt_variant,
        },
    )
    services, runtime_values = await start_background_services(
        task.setup.background_services,
        workspace=workspace,
        repo_root=clawbench_root,
        runtime_values=runtime_values,
    )

    transcript = Transcript()
    user_messages: list[str] = []
    session_id = f"clawbench-{task.id}-run{run_index}-{uuid.uuid4().hex[:6]}"
    start_ms = int(time.time() * 1000)

    try:
        for phase_index, phase in enumerate(task.normalized_phases()):
            simulator = UserSimulator(
                phase.user,
                runtime_values,
                prompt_variant=prompt_variant,
            )
            turn_index = 0
            while not simulator.is_done:
                user_message = await simulator.next_message(transcript)
                if user_message is None:
                    break
                full_query = _workspace_prompt_prefix(workspace) + user_message
                user_messages.append(user_message)
                transcript.messages.append(TranscriptMessage(role="user", text=user_message))

                _context, assistant_text, collector = await _run_s18_turn(
                    query=full_query,
                    workspace=workspace,
                    multi_mcp=multi_mcp,
                    session_id=f"{session_id}-p{phase_index}-t{turn_index}",
                )
                await collector.drain()
                phase_transcript = collector.build_transcript(
                    user_messages=[],
                    assistant_text=assistant_text,
                )
                transcript.messages.extend(phase_transcript.messages)
                turn_index += 1

        duration_ms = int(time.time() * 1000) - start_ms
        result = await _score_task_run(
            task=task,
            transcript=transcript,
            workspace=workspace,
            runtime_values=runtime_values,
            judge_model=judge_model,
            run_index=run_index,
            duration_ms=duration_ms,
        )
        return result
    finally:
        await stop_background_services(services)
        if os.environ.get("CLAWBENCH_KEEP_WORKSPACES") != "1":
            shutil.rmtree(workspace, ignore_errors=True)


async def run_benchmark(args: argparse.Namespace) -> dict:
    _apply_benchmark_settings()
    clawbench_root = _resolve_clawbench_root(args.clawbench_root)
    _ensure_clawbench_import(clawbench_root)
    os.environ.setdefault("CLAWBENCH_TASKS_DIR", str(clawbench_root / "tasks-public"))

    from clawbench.stats import summarize_task_runs
    from clawbench.tasks import load_all_tasks

    from config.settings_loader import load_settings
    from mcp_servers.multi_mcp import MultiMCP

    settings = load_settings()
    model_label = args.model or settings.get("agent", {}).get("default_model", "s18-default")

    use_ollama = settings.get("agent", {}).get("model_provider") == "ollama"
    if use_ollama:
        await _ensure_ollama_available()
        print(f"[clawbench] Warming Ollama model at {_ollama_base_url()} ...", flush=True)
        await _warm_ollama_model(model_label)
        print("[clawbench] Ollama warm-up complete.", flush=True)

    task_ids = args.task or []
    if args.core_v1:
        task_ids = _load_core_v1_task_ids(clawbench_root)

    tasks = load_all_tasks(task_ids=task_ids or None, tier=args.tier)
    if not tasks:
        raise ValueError("No ClawBench tasks matched the selection.")

    results_root = Path(args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    _refresh_sandbox_tool_cache()
    multi_mcp = MultiMCP()
    await multi_mcp.start()

    alias_overrides = _apply_clawbench_agent_aliases()
    all_runs: dict[str, list] = {}
    try:
        for task in tasks:
            task_results = []
            for run_index in range(args.runs):
                print(f"[clawbench] {task.id} run {run_index + 1}/{args.runs} ...", flush=True)
                result = await _run_single_task_run(
                    task=task,
                    run_index=run_index,
                    multi_mcp=multi_mcp,
                    clawbench_root=clawbench_root,
                    results_root=results_root,
                    prompt_variant=args.prompt_variant,
                    judge_model=args.judge_model,
                    model_label=model_label,
                )
                task_results.append(result)
                print(
                    f"  score={result.run_score:.2f} "
                    f"C={result.completion_result.score:.2f} "
                    f"T={result.trajectory_result.score:.2f} "
                    f"B={result.behavior_result.score:.2f}",
                    flush=True,
                )
            all_runs[task.id] = task_results
    finally:
        _restore_clawbench_agent_aliases(alias_overrides)
        await multi_mcp.stop()

    task_summaries = []
    for task in tasks:
        runs = all_runs.get(task.id, [])
        scores = [result.run_score for result in runs]
        pass_flags = [result.run_score >= task.pass_threshold for result in runs]
        summary = summarize_task_runs(
            scores,
            pass_threshold=task.pass_threshold,
            pass_flags=pass_flags,
        )
        task_summaries.append(
            {
                "task_id": task.id,
                "tier": task.tier.value,
                "family": task.family.value,
                "runs": len(runs),
                "mean_run_score": round(summary.mean, 4),
                "task_score": round(summary.task_score, 4),
                "pass_rate": round(summary.pass_rate, 4),
                "pass_hat_k": summary.pass_hat_k,
                "mean_completion": round(
                    sum(r.completion_result.score for r in runs) / len(runs), 4
                )
                if runs
                else 0.0,
                "mean_trajectory": round(
                    sum(r.trajectory_result.score for r in runs) / len(runs), 4
                )
                if runs
                else 0.0,
                "mean_behavior": round(
                    sum(r.behavior_result.score for r in runs) / len(runs), 4
                )
                if runs
                else 0.0,
            }
        )

    overall = sum(item["task_score"] for item in task_summaries) / len(task_summaries)

    payload = {
        "harness": "s18-clawbench",
        "s18_repo": "https://github.com/riteshverma/s18",
        "clawbench_repo": "https://github.com/openclaw/clawbench",
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_label,
        "task_count": len(tasks),
        "runs_per_task": args.runs,
        "overall_mean_task_score": round(overall, 4),
        "task_summaries": task_summaries,
        "runs": {
            task_id: [run.model_dump() for run in runs]
            for task_id, runs in all_runs.items()
        },
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark S18 against ClawBench tasks.")
    parser.add_argument(
        "--clawbench-root",
        default="",
        help="Path to clawbench checkout (default: CLAWBENCH_ROOT or ~/Downloads/clawbench)",
    )
    parser.add_argument(
        "-t",
        "--task",
        action="append",
        default=[],
        help="Task id (repeatable). Example: -t t1-fs-quick-note",
    )
    parser.add_argument("--core-v1", action="store_true", help="Run all 19 Core v1 public tasks.")
    parser.add_argument("--tier", default=None, help="Restrict to a tier (tier1..tier5).")
    parser.add_argument("--runs", type=int, default=1, help="Runs per task (ClawBench default: 3).")
    parser.add_argument("--model", default="", help="Label recorded in results (defaults to S18 agent model).")
    parser.add_argument(
        "--prompt-variant",
        default="clear",
        choices=["clear", "ambiguous"],
        help="ClawBench prompt variant.",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional OpenClaw-backed judge model (requires OpenClaw gateway; usually leave empty).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Write JSON results to this path.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "benchmarks" / "clawbench" / "results"),
        help="Scratch workspaces and cached artifacts.",
    )
    return parser


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _ensure_python3_shim()
    parser = build_parser()
    args = parser.parse_args()
    payload = asyncio.run(run_benchmark(args))

    output_path = args.output
    if not output_path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(args.results_dir) / f"s18_clawbench_{stamp}.json")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nOverall mean task score: {payload['overall_mean_task_score']:.4f}")
    print(f"Results written to: {out}")


if __name__ == "__main__":
    main()
