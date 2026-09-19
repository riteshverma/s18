"""Tests for the computer-use safety proxy (mcp_servers/server_computer.py).

Unit tests cover allow/deny filtering and wrapper-signature generation.
The end-to-end test drives the proxy against tests/fixtures/fake_cua_driver.py
(a real FastMCP stdio server), so discovery, filtering, calls, and audit all
run over actual MCP stdio without needing the cua-driver binary. The final
test performs a live handshake with the installed driver and is skipped when
the binary is missing. Async entry points use asyncio.run(), matching the
DecisionRouter test style.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import mcp_servers.server_computer as cps

FIXTURE_DRIVER = Path(__file__).parent / "fixtures" / "fake_cua_driver.py"


def _fake_tool(name, properties=None, required=None, description="stub"):
    schema = {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
    }
    return SimpleNamespace(name=name, description=description, inputSchema=schema)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_default_denylist_removes_dangerous_tools(monkeypatch):
    monkeypatch.delenv("S18_COMPUTER_TOOL_ALLOWLIST", raising=False)
    monkeypatch.delenv("S18_COMPUTER_TOOL_DENYLIST", raising=False)
    tools = [_fake_tool(n) for n in ("list_apps", "click", "kill_app", "set_config", "type_text")]
    kept = {t.name for t in cps._filter_tools(tools)}
    assert kept == {"list_apps", "click", "type_text"}


def test_custom_allowlist_narrows_exposure(monkeypatch):
    monkeypatch.setenv("S18_COMPUTER_TOOL_ALLOWLIST", "click, list_apps")
    monkeypatch.delenv("S18_COMPUTER_TOOL_DENYLIST", raising=False)
    tools = [_fake_tool(n) for n in ("list_apps", "click", "type_text")]
    kept = {t.name for t in cps._filter_tools(tools)}
    assert kept == {"list_apps", "click"}


def test_denylist_wins_over_allowlist(monkeypatch):
    monkeypatch.setenv("S18_COMPUTER_TOOL_ALLOWLIST", "*")
    monkeypatch.setenv("S18_COMPUTER_TOOL_DENYLIST", "click")
    tools = [_fake_tool(n) for n in ("list_apps", "click")]
    kept = {t.name for t in cps._filter_tools(tools)}
    assert kept == {"list_apps"}


# ---------------------------------------------------------------------------
# Wrapper generation
# ---------------------------------------------------------------------------


def test_wrapper_signature_mirrors_driver_schema():
    tool = _fake_tool(
        "click",
        properties={
            "pid": {"type": "integer"},
            "element_index": {"type": "integer"},
            "comment": {"type": "string"},
        },
        required=["pid"],
        description="Click an element.",
    )
    wrapper = cps._make_wrapper(tool)
    assert wrapper.__name__ == "computer_click"
    assert "Click an element." in (wrapper.__doc__ or "")
    sig = wrapper.__signature__
    assert sig.parameters["pid"].annotation is int
    assert sig.parameters["pid"].default is sig.parameters["pid"].empty  # required
    assert sig.parameters["element_index"].default is None  # optional
    assert sig.parameters["comment"].annotation is str


def test_wrapper_skips_non_identifier_params():
    tool = _fake_tool("weird", properties={"bad-name": {"type": "string"}}, required=["bad-name"])
    wrapper = cps._make_wrapper(tool)
    assert "bad-name" not in wrapper.__signature__.parameters


# ---------------------------------------------------------------------------
# End-to-end against the fake driver
# ---------------------------------------------------------------------------


def test_end_to_end_fake_driver_discovery_call_and_audit(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "S18_COMPUTER_DRIVER_CMD",
        json.dumps([sys.executable, str(FIXTURE_DRIVER)]),
    )
    monkeypatch.delenv("S18_COMPUTER_TOOL_ALLOWLIST", raising=False)
    monkeypatch.delenv("S18_COMPUTER_TOOL_DENYLIST", raising=False)
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("S18_COMPUTER_AUDIT_PATH", str(audit_path))

    async def scenario():
        try:
            summary = await cps.discover_and_register()
            assert summary["driver_tools"] == 3  # list_apps, click, kill_app
            assert summary["exposed"] == 2  # kill_app denied by default
            assert "computer_list_apps" in cps._registered
            assert "computer_click" in cps._registered
            assert "computer_kill_app" not in cps._registered

            result = await cps._invoke("click", {"pid": 42, "element_index": 3})
            assert "clicked pid=42 element_index=3" in result

            missing = await cps._invoke("kill_app", {"pid": 1})
            assert missing.startswith("ERROR:")  # denylist enforced on the call path too
        finally:
            await cps._teardown()

    asyncio.run(scenario())

    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    clicks = [e for e in entries if e["tool"] == "click"]
    assert clicks and clicks[0]["ok"] is True
    assert clicks[0]["arguments"]["pid"] == 42


# ---------------------------------------------------------------------------
# Live handshake (integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("cua-driver") is None, reason="cua-driver not installed")
def test_integration_real_driver_handshake():
    """Live discovery against the installed driver; validates install + protocol."""

    async def scenario():
        await cps._teardown()  # isolate from any session left by earlier tests
        try:
            session = await cps._ensure_session()
            result = await session.list_tools()
            names = {t.name for t in result.tools}
        finally:
            await cps._teardown()
        return names

    names = asyncio.run(scenario())
    assert len(names) > 10
    assert "list_apps" in names
