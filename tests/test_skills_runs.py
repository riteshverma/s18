import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.skills.lifecycle import resolve_skill, run_skill_success
from core.skills.manager import skill_manager


def test_explicit_skill_id_system_monitor_produces_artifact():
    skill_manager.initialize()
    skill, effective_query, resolved_skill_id = asyncio.run(
        resolve_skill(
            query="Check system cpu usage",
            run_id="test_run_system_monitor",
            agent_id="runs",
            explicit_skill_id="system_monitor",
            integration_id="default",
            workflow_id="generic",
        )
    )

    assert resolved_skill_id == "system_monitor"
    assert skill is not None
    assert effective_query == "Check system cpu usage"

    fake_path = Path("data/Notes/System/Health_test_run_system_monitor.md")
    skill.on_run_success = AsyncMock(
        return_value={
            "file_path": str(fake_path),
            "type": "system_health",
            "summary": "CPU: 10%, RAM: 20%, Disk: 30%",
        }
    )

    artifact = asyncio.run(
        run_skill_success(
            skill,
            {"status": "completed", "summary": "done", "output": "ok"},
        )
    )

    assert artifact is not None
    assert artifact["type"] == "system_health"
    assert artifact["file_path"].endswith(".md")
    assert artifact["file_path"] == str(fake_path)


def test_auto_match_for_default_generic_run():
    skill_manager.initialize()
    skill, _effective_query, resolved_skill_id = asyncio.run(
        resolve_skill(
            query="Please check cpu usage right now",
            run_id="test_run_auto_match",
            agent_id="runs",
            integration_id="default",
            workflow_id="generic",
        )
    )

    assert resolved_skill_id == "system_monitor"
    assert skill is not None


def test_no_auto_match_for_wiseai_runs():
    skill_manager.initialize()
    skill, effective_query, resolved_skill_id = asyncio.run(
        resolve_skill(
            query="market signal and cbc values",
            run_id="test_run_wiseai_guard",
            agent_id="cdss",
            integration_id="wiseai",
            workflow_id="cdss",
        )
    )

    assert skill is None
    assert resolved_skill_id is None
    assert effective_query == "market signal and cbc values"


def test_invalid_skill_id_falls_back_to_normal_query():
    skill_manager.initialize()
    skill, effective_query, resolved_skill_id = asyncio.run(
        resolve_skill(
            query="normal query",
            run_id="test_run_invalid_skill",
            agent_id="runs",
            explicit_skill_id="does_not_exist",
            integration_id="default",
            workflow_id="generic",
        )
    )

    assert skill is None
    assert resolved_skill_id is None
    assert effective_query == "normal query"


def test_scheduler_delegates_skill_lifecycle_to_process_run(monkeypatch):
    from core.scheduler import JobDefinition, SchedulerService

    captured = {}

    async def fake_process_run(run_id, canonical_request):
        captured["run_id"] = run_id
        captured["request"] = canonical_request
        return {
            "status": "completed",
            "summary": "skill summary",
            "skill": {"summary": "skill summary", "file_path": "data/Notes/out.md"},
        }

    fake_runs = types.ModuleType("routers.runs")
    fake_runs.process_run = fake_process_run
    monkeypatch.setitem(sys.modules, "routers.runs", fake_runs)

    fake_inbox = types.ModuleType("routers.inbox")
    fake_inbox.send_to_inbox = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "routers.inbox", fake_inbox)

    fake_event_bus = types.ModuleType("core.event_bus")
    fake_event_bus.event_bus = SimpleNamespace(publish=AsyncMock())
    monkeypatch.setitem(sys.modules, "core.event_bus", fake_event_bus)

    async def forbidden_lifecycle(*_args, **_kwargs):
        raise AssertionError("scheduler must not run skill lifecycle hooks")

    fake_lifecycle = types.ModuleType("core.skills.lifecycle")
    fake_lifecycle.resolve_skill = forbidden_lifecycle
    fake_lifecycle.run_skill_success = forbidden_lifecycle
    fake_lifecycle.run_skill_failure = forbidden_lifecycle
    monkeypatch.setitem(sys.modules, "core.skills.lifecycle", fake_lifecycle)

    class FakeScheduler:
        def add_job(self, func, *_args, **_kwargs):
            self.func = func

        def get_job(self, _job_id):
            return None

    service = SchedulerService()
    fake_scheduler = FakeScheduler()
    monkeypatch.setattr(service, "scheduler", fake_scheduler)
    monkeypatch.setattr(service, "save_jobs_async", AsyncMock())

    job = JobDefinition(
        id="job1",
        name="Market job",
        cron_expression="* * * * *",
        agent_type="research",
        query="market update",
        skill_id="market_analyst",
    )
    service._schedule_job(job)
    asyncio.run(fake_scheduler.func())

    assert captured["request"].query == "market update"
    assert captured["request"].skill_id == "market_analyst"
    assert captured["request"].integration_id == "default"
    assert captured["request"].workflow_id == "generic"
    assert job.last_output == "skill summary"
