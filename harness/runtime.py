from __future__ import annotations

import asyncio
import os
import subprocess
from datetime import UTC, datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Optional

from config.settings_loader import load_settings
from core.event_bus import event_bus
from harness.drivers import HarnessDriverRegistry
from harness.models import HarnessJobRequest, HarnessJobState, HarnessJobStatus
from harness.store import HarnessJobStore


class HarnessRuntime:
    """Process manager for trusted CLI harness jobs."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._store = HarnessJobStore(project_root=project_root)
        self._active_processes: Dict[str, subprocess.Popen] = {}
        self._state_lock = asyncio.Lock()
        self._output_limit = self._default_output_limit()

    async def create_job(self, request: HarnessJobRequest) -> HarnessJobState:
        timeout_seconds = request.timeout_seconds or self._default_timeout()
        job_id = str(int(datetime.now().timestamp() * 1000))
        request_id = f"hreq_{job_id}"
        cwd = self._resolve_cwd(request.cwd)
        drivers = HarnessDriverRegistry(settings=load_settings())
        execution_plan = drivers.build_plan(request)
        state = HarnessJobState(
            id=job_id,
            request_id=request_id,
            provider=request.provider,
            prompt=request.prompt,
            cwd=str(cwd),
            additional_args=request.additional_args,
            timeout_seconds=timeout_seconds,
            env=request.env,
            metadata=request.metadata,
            status=HarnessJobStatus.ACCEPTED,
            command=execution_plan.command,
        )
        await asyncio.to_thread(self._store.save, state)
        await self._publish("job.accepted", state.id, {"provider": state.provider.value})
        return state

    async def run_job(self, job_id: str) -> None:
        state = await asyncio.to_thread(self._store.load, job_id)
        if not state:
            return
        await self._transition(state, HarnessJobStatus.STARTING)

        try:
            drivers = HarnessDriverRegistry(settings=load_settings())
            request = HarnessJobRequest(
                provider=state.provider,
                prompt=state.prompt,
                cwd=state.cwd,
                additional_args=state.additional_args,
                timeout_seconds=state.timeout_seconds,
                env=state.env,
                metadata=state.metadata,
            )
            plan = drivers.build_plan(request)
            state.command = plan.command
            await asyncio.to_thread(self._store.save, state)

            process = subprocess.Popen(
                plan.command,
                cwd=state.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._build_env(state.env),
            )
            async with self._state_lock:
                self._active_processes[state.id] = process
            state.pid = process.pid
            state.started_at = datetime.now(UTC).isoformat()
            await self._transition(state, HarnessJobStatus.RUNNING)

            stdout_task = asyncio.create_task(
                self._pump_stream(state.id, process.stdout, "stdout")
            )
            stderr_task = asyncio.create_task(
                self._pump_stream(state.id, process.stderr, "stderr")
            )

            if plan.stdin_payload is not None and process.stdin:
                process.stdin.write(plan.stdin_payload)
                if not plan.stdin_payload.endswith("\n"):
                    process.stdin.write("\n")
                process.stdin.flush()
                process.stdin.close()
            elif process.stdin:
                process.stdin.close()

            timed_out = False
            try:
                await asyncio.to_thread(process.wait, timeout=state.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                state.timed_out = True
                process.kill()
                await asyncio.to_thread(process.wait)

            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            await self._drain_remaining_output(state, process.stdout, "stdout")
            await self._drain_remaining_output(state, process.stderr, "stderr")
            state.return_code = process.returncode
            state.completed_at = datetime.now(UTC).isoformat()

            if state.cancelled:
                await self._transition(state, HarnessJobStatus.CANCELLED)
            elif timed_out:
                await self._transition(state, HarnessJobStatus.TIMEOUT)
            elif process.returncode == 0:
                await self._transition(state, HarnessJobStatus.COMPLETED)
            else:
                stderr_preview = self._tail_for_stream(state, "stderr")
                stdout_preview = self._tail_for_stream(state, "stdout")
                preview = stderr_preview or stdout_preview
                if preview:
                    state.error = (
                        f"Process exited with code {process.returncode}: {preview}"
                    )
                else:
                    state.error = f"Process exited with code {process.returncode}"
                await self._transition(state, HarnessJobStatus.FAILED)

        except FileNotFoundError as exc:
            state.error = f"{type(exc).__name__}: {exc}"
            state.completed_at = datetime.now(UTC).isoformat()
            await self._transition(state, HarnessJobStatus.FAILED)
        except Exception as exc:
            state.error = f"{type(exc).__name__}: {exc!r}"
            state.completed_at = datetime.now(UTC).isoformat()
            await self._transition(state, HarnessJobStatus.FAILED)
        finally:
            async with self._state_lock:
                self._active_processes.pop(state.id, None)
            await asyncio.to_thread(self._store.save, state)
            await self._publish(
                "job.status",
                state.id,
                {
                    "status": state.status.value,
                    "return_code": state.return_code,
                    "timed_out": state.timed_out,
                    "cancelled": state.cancelled,
                },
            )

    async def stop_job(self, job_id: str) -> HarnessJobState:
        state = await asyncio.to_thread(self._store.load, job_id)
        if not state:
            raise FileNotFoundError(f"Harness job {job_id} not found")
        async with self._state_lock:
            process = self._active_processes.get(job_id)
        if process and process.returncode is None:
            state.cancelled = True
            process.terminate()
            await self._publish("job.stop_requested", job_id, {"pid": state.pid})
        await asyncio.to_thread(self._store.save, state)
        return state

    async def resume_job(self, job_id: str) -> HarnessJobState:
        state = await asyncio.to_thread(self._store.load, job_id)
        if not state:
            raise FileNotFoundError(f"Harness job {job_id} not found")
        await self._publish("job.resume", job_id, {"status": state.status.value})
        return state

    async def get_job(self, job_id: str) -> Optional[HarnessJobState]:
        return await asyncio.to_thread(self._store.load, job_id)

    async def list_jobs(self, limit: int = 100) -> list[HarnessJobState]:
        return await asyncio.to_thread(self._store.list_jobs, limit)

    async def _pump_stream(
        self, job_id: str, stream: Optional[TextIOWrapper], stream_name: str
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = await asyncio.to_thread(stream.readline)
            if chunk == "":
                break
            text = chunk.rstrip("\r\n")
            state = await asyncio.to_thread(self._store.load, job_id)
            if not state:
                continue
            state.append_output(stream_name, text, self._output_limit)
            await asyncio.to_thread(self._store.save, state)
            await self._publish(
                f"job.{stream_name}",
                job_id,
                {"line": text, "stream": stream_name, "status": state.status.value},
            )

    async def _transition(self, state: HarnessJobState, status: HarnessJobStatus) -> None:
        state.status = status
        if status in {
            HarnessJobStatus.COMPLETED,
            HarnessJobStatus.FAILED,
            HarnessJobStatus.CANCELLED,
            HarnessJobStatus.TIMEOUT,
        } and not state.completed_at:
            state.completed_at = datetime.now(UTC).isoformat()
        await asyncio.to_thread(self._store.save, state)
        await self._publish("job.status", state.id, {"status": status.value})

    async def _drain_remaining_output(
        self, state: HarnessJobState, stream: Optional[TextIOWrapper], stream_name: str
    ) -> None:
        if stream is None:
            return
        remainder = await asyncio.to_thread(stream.read)
        if not remainder:
            return
        for raw_line in remainder.splitlines():
            text = raw_line.rstrip("\r\n")
            if not text:
                continue
            state.append_output(stream_name, text, self._output_limit)
            await self._publish(
                f"job.{stream_name}",
                state.id,
                {"line": text, "stream": stream_name, "status": state.status.value},
            )

    async def _publish(self, event_type: str, job_id: str, payload: dict) -> None:
        data = {"job_id": job_id, **payload}
        await event_bus.publish(event_type, "harness", data)

    @staticmethod
    def _tail_for_stream(state: HarnessJobState, stream_name: str) -> str:
        for chunk in reversed(state.output_tail):
            if chunk.stream == stream_name and chunk.text.strip():
                return chunk.text.strip()
        return ""

    def _resolve_cwd(self, requested: Optional[str]) -> Path:
        candidate = Path(requested) if requested else self.project_root
        if not candidate.is_absolute():
            candidate = (self.project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        allowed_roots = self._allowed_roots()
        if not any(candidate.is_relative_to(root) for root in allowed_roots):
            raise ValueError("cwd must be under an allowed harness root")
        if not candidate.exists() or not candidate.is_dir():
            raise ValueError("cwd must point to an existing directory")
        return candidate

    def _allowed_roots(self) -> list[Path]:
        settings = load_settings().get("harness", {})
        roots = settings.get("allowed_roots", []) if isinstance(settings, dict) else []
        parsed = []
        for raw in roots:
            text = str(raw).strip()
            if not text:
                continue
            root = Path(text)
            if not root.is_absolute():
                root = (self.project_root / root).resolve()
            else:
                root = root.resolve()
            parsed.append(root)
        if not parsed:
            parsed.append(self.project_root.resolve())
        return parsed

    @staticmethod
    def _default_timeout() -> int:
        harness = load_settings().get("harness", {})
        if isinstance(harness, dict):
            timeout = harness.get("default_timeout_seconds")
            if isinstance(timeout, int) and timeout > 0:
                return timeout
        return 900

    @staticmethod
    def _default_output_limit() -> int:
        harness = load_settings().get("harness", {})
        if isinstance(harness, dict):
            limit = harness.get("output_tail_lines")
            if isinstance(limit, int) and 10 <= limit <= 2000:
                return limit
        return 250

    @staticmethod
    def _build_env(job_env: Dict[str, str]) -> Dict[str, str]:
        base = dict(os.environ)
        allowed_env = {
            "PATH",
            "HOME",
            "HOMEDRIVE",
            "HOMEPATH",
            "USERPROFILE",
            "USERNAME",
            "APPDATA",
            "LOCALAPPDATA",
            "PROGRAMDATA",
            "TMP",
            "TEMP",
            "SYSTEMROOT",
            "COMSPEC",
            "PATHEXT",
            "TERM",
            "LANG",
            "LC_ALL",
            "PYTHONUTF8",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        }
        env = {k: v for k, v in base.items() if k in allowed_env}
        for key, value in base.items():
            upper = key.upper()
            if upper.startswith(("CLAUDE_", "ANTHROPIC_", "GOOGLE_", "GEMINI_", "OPENAI_")):
                env[key] = value
        for key, value in job_env.items():
            if key in allowed_env:
                env[key] = value
        return env

