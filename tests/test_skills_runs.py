import asyncio
from pathlib import Path
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
