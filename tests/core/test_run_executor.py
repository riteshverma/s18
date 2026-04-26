import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core import run_store as run_store_module
from core.run_executor import execute_resume, execute_run
from core.run_store import RunStore
from integrations.contracts import CanonicalRunRequest


def test_execute_run_defaults_to_in_process(monkeypatch):
    request = CanonicalRunRequest(query="demo")
    process_run = AsyncMock(return_value={"status": "completed"})
    monkeypatch.delenv("S18_RUN_EXECUTOR", raising=False)
    monkeypatch.setattr("routers.runs.process_run", process_run)

    result = asyncio.run(execute_run("run-1", request, {"request_id": "req-1"}, None))

    assert result == {"status": "completed"}
    process_run.assert_awaited_once()


def test_execute_run_enqueues_celery_task(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(run_id="run-2", status="accepted", query="demo")
    monkeypatch.setattr(run_store_module, "_run_store", store)
    monkeypatch.setenv("S18_RUN_EXECUTOR", "celery")

    captured = {}

    def fake_delay(*args):
        captured["args"] = args
        return SimpleNamespace(id="task-123")

    from workers.agent_tasks import run_agent_task

    monkeypatch.setattr(run_agent_task, "delay", fake_delay)

    request = CanonicalRunRequest(query="queued demo", integration_id="default")
    result = asyncio.run(execute_run("run-2", request, None, {"tenant_id": "default"}))

    assert result == {"run_id": "run-2", "status": "accepted", "task_id": "task-123"}
    assert captured["args"][0] == "run-2"
    assert captured["args"][1]["query"] == "queued demo"
    assert store.get_run("run-2")["metadata"]["celery_task_id"] == "task-123"


def test_execute_resume_enqueues_celery_task(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(run_id="run-3", status="interrupted", query="resume demo")
    monkeypatch.setattr(run_store_module, "_run_store", store)
    monkeypatch.setenv("S18_RUN_EXECUTOR", "celery")

    def fake_delay(*args):
        return SimpleNamespace(id="task-456")

    from workers.agent_tasks import resume_agent_task

    monkeypatch.setattr(resume_agent_task, "delay", fake_delay)

    result = asyncio.run(execute_resume("run-3", {"request_id": "resume-run-3"}))

    assert result == {"run_id": "run-3", "status": "accepted", "task_id": "task-456"}
    assert store.get_run("run-3")["metadata"]["execution_backend"] == "celery"
