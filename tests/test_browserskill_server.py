"""Tests for the BrowserSkill CLI proxy (mcp_servers/server_browserskill.py).

Unit tests cover allow/deny policy, flag rendering, ref validation, output-path
confinement and audit redaction. The end-to-end tests drive the proxy against
tests/fixtures/fake_bsk.py, a stub CLI that echoes its argv, so argv
construction and env plumbing run through a real subprocess without the Rust
binary. The final test performs a live handshake with an installed bsk and is
skipped when the binary is missing. Async entry points use asyncio.run(),
matching the computer-proxy test style.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

import mcp_servers.server_browserskill as bsp

FIXTURE_BSK = Path(__file__).parent / "fixtures" / "fake_bsk.py"

POLICY_ENV = (
    "S18_BSK_COMMAND_ALLOWLIST",
    "S18_BSK_COMMAND_DENYLIST",
    "S18_BSK_BIN",
    "S18_BSK_HOME",
    "S18_BSK_AUTO_START",
    "S18_BSK_AUDIT_PATH",
    "S18_BSK_AUDIT_VALUES",
    "S18_BSK_ARTIFACT_DIR",
    "S18_BSK_TIMEOUT",
    "S18_BSK_MAX_TIMEOUT",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Default policy, audit and artifacts isolated to tmp_path."""
    for name in POLICY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("S18_BSK_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("S18_BSK_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    return tmp_path


@pytest.fixture
def stub_cli(clean_env, monkeypatch):
    """Point the proxy at the echoing stub CLI."""
    monkeypatch.setenv("S18_BSK_BIN", json.dumps([sys.executable, str(FIXTURE_BSK)]))
    return clean_env


def _audit_entries(tmp_path) -> list:
    path = tmp_path / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["evaluate", "record", "upload", "update", "config"])
def test_default_denylist_blocks_dangerous_commands(clean_env, command):
    assert bsp._is_denied(command) is True


def test_denylist_matches_subcommands_by_head(clean_env):
    """'daemon' in the denylist must also cover 'daemon start'."""
    assert bsp._is_denied("daemon start") is True
    assert bsp._is_denied("record start") is True


@pytest.mark.parametrize("command", ["observe", "click", "session start", "tab borrow"])
def test_ordinary_commands_are_allowed_by_default(clean_env, command):
    assert bsp._is_denied(command) is False


def test_custom_allowlist_narrows_exposure(clean_env, monkeypatch):
    monkeypatch.setenv("S18_BSK_COMMAND_ALLOWLIST", "observe, navigate")
    assert bsp._is_denied("observe") is False
    assert bsp._is_denied("click") is True


def test_denylist_wins_over_allowlist(clean_env, monkeypatch):
    monkeypatch.setenv("S18_BSK_COMMAND_ALLOWLIST", "*")
    monkeypatch.setenv("S18_BSK_COMMAND_DENYLIST", "click")
    assert bsp._is_denied("click") is True


def test_denied_command_is_not_executed_and_is_audited(stub_cli):
    """The policy gate runs before the subprocess, for wrappers and bsk_run alike."""
    result = asyncio.run(bsp._run("evaluate", ["document.cookie"]))
    assert result.startswith("ERROR:")
    assert "blocked" in result
    entries = _audit_entries(stub_cli)
    assert entries[-1]["ok"] is False
    assert entries[-1]["command"] == "evaluate"


def test_bsk_run_passthrough_cannot_bypass_the_denylist(stub_cli):
    result = asyncio.run(bsp.bsk_run("daemon start"))
    assert "blocked" in result


def test_malformed_command_is_rejected(stub_cli):
    """Shell-looking command strings never reach argv."""
    result = asyncio.run(bsp._run("status && evaluate"))
    assert result.startswith("ERROR: malformed")


# ---------------------------------------------------------------------------
# Argument construction
# ---------------------------------------------------------------------------


def test_flags_render_underscores_as_hyphens_and_drop_empties():
    assert bsp._flags(tab_id="t1", session="s1") == ["--tab-id", "t1", "--session", "s1"]
    assert bsp._flags(full_page=True, scope=None, probe_hover=False) == ["--full-page"]


@pytest.mark.parametrize("ref", ["--session", "-x", "", "e3", "@e3; rm"])
def test_invalid_refs_are_rejected(ref):
    with pytest.raises(ValueError):
        bsp._check_ref(ref)


def test_valid_ref_passes():
    assert bsp._check_ref("@e12") == "@e12"


def test_flag_shaped_ref_is_refused_before_execution(stub_cli):
    result = asyncio.run(bsp.bsk_click("--session", session="s1"))
    assert result.startswith("ERROR: invalid element ref")


def test_screenshot_rejects_ref_with_full_page(stub_cli):
    result = asyncio.run(bsp.bsk_screenshot(session="s1", ref="@e3", full_page=True))
    assert "cannot be combined" in result


def test_wheel_requires_a_nonzero_delta(stub_cli):
    result = asyncio.run(bsp.bsk_wheel(session="s1"))
    assert "nonzero" in result


# ---------------------------------------------------------------------------
# Output-path confinement
# ---------------------------------------------------------------------------


def test_relative_output_resolves_under_the_artifact_dir(clean_env):
    out = Path(bsp._confined_out("shot.png"))
    assert out.parent == (clean_env / "artifacts").resolve()


@pytest.mark.parametrize("bad", ["../escape.png", "a/../../escape.png"])
def test_traversal_outside_the_artifact_dir_is_refused(clean_env, bad):
    with pytest.raises(ValueError):
        bsp._confined_out(bad)


def test_absolute_path_outside_the_artifact_dir_is_refused(clean_env, tmp_path):
    with pytest.raises(ValueError):
        bsp._confined_out(str(tmp_path / "elsewhere.png"))


def test_download_refuses_to_write_outside_the_artifact_dir(stub_cli):
    result = asyncio.run(bsp.bsk_download("@e3", "../loot.bin", session="s1"))
    assert result.startswith("ERROR:")
    assert "must stay under" in result


# ---------------------------------------------------------------------------
# Audit redaction
# ---------------------------------------------------------------------------


def test_fill_values_are_redacted_by_default(stub_cli):
    asyncio.run(bsp.bsk_fill("@e3", "hunter2-secret", session="s1"))
    argv = _audit_entries(stub_cli)[-1]["argv"]
    assert "hunter2-secret" not in argv
    assert "[redacted len=14]" in argv


def test_audit_values_opt_in_records_the_value(stub_cli, monkeypatch):
    monkeypatch.setenv("S18_BSK_AUDIT_VALUES", "1")
    asyncio.run(bsp.bsk_fill("@e3", "plain", session="s1"))
    assert "plain" in _audit_entries(stub_cli)[-1]["argv"]


def test_audit_can_be_disabled(stub_cli, monkeypatch):
    monkeypatch.setenv("S18_BSK_AUDIT_PATH", "")
    assert bsp._audit_path() is None
    asyncio.run(bsp.bsk_observe(session="s1"))
    assert _audit_entries(stub_cli) == []


# ---------------------------------------------------------------------------
# End-to-end against the stub CLI
# ---------------------------------------------------------------------------


def test_subcommand_and_flags_reach_the_cli(stub_cli):
    raw = asyncio.run(bsp.bsk_tab_borrow("tab-9", session="s1", timeout="120s"))
    argv = json.loads(raw)["argv"]
    assert argv == ["tab", "borrow", "tab-9", "--session", "s1", "--timeout", "120s"]


def test_json_capable_commands_get_the_json_flag(stub_cli):
    argv = json.loads(asyncio.run(bsp.bsk_session_start()))["argv"]
    assert argv == ["session", "start", "--json"]


def test_non_json_commands_do_not_get_the_json_flag(stub_cli):
    argv = json.loads(asyncio.run(bsp.bsk_observe(session="s1")))["argv"]
    assert argv == ["observe", "--session", "s1"]
    assert "--json" not in argv


def test_home_and_auto_start_are_passed_to_the_child(stub_cli, monkeypatch):
    monkeypatch.setenv("S18_BSK_HOME", "/tmp/bskhome")
    monkeypatch.setenv("S18_BSK_AUTO_START", "0")
    payload = json.loads(asyncio.run(bsp.bsk_session_list()))
    assert payload["bsk_home"] == "/tmp/bskhome"
    assert payload["bsk_auto_start"] == "0"


def test_nonzero_exit_is_reported_as_an_error_and_audited(stub_cli):
    result = asyncio.run(bsp._run("fail"))
    assert result.startswith("ERROR: bsk fail exited 3")
    assert "stub failure" in result
    assert _audit_entries(stub_cli)[-1]["ok"] is False


def test_successful_call_is_audited(stub_cli):
    asyncio.run(bsp.bsk_navigate("https://example.com", session="s1"))
    entry = _audit_entries(stub_cli)[-1]
    assert entry["command"] == "navigate"
    assert entry["ok"] is True
    assert "https://example.com" in entry["argv"]


@pytest.mark.slow
def test_timeout_kills_the_child(stub_cli, monkeypatch):
    monkeypatch.setenv("S18_BSK_TIMEOUT", "1")
    result = asyncio.run(bsp._run("hang"))
    assert "timed out" in result


def test_missing_binary_reports_install_hint(clean_env, monkeypatch):
    monkeypatch.setenv("S18_BSK_BIN", "definitely-not-installed-bsk")
    result = asyncio.run(bsp._run("status"))
    assert "not found" in result
    assert "BrowserSkill" in result


def test_status_tool_works_without_the_cli_installed(clean_env, monkeypatch):
    """Planners must be able to detect the capability's absence."""
    monkeypatch.setenv("S18_BSK_BIN", "definitely-not-installed-bsk")
    status = json.loads(asyncio.run(bsp.bsk_cli_status()))
    assert status["installed"] is False
    assert status["available"] is False
    assert "evaluate" in status["denylist"]


# ---------------------------------------------------------------------------
# Live handshake
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("bsk") is None, reason="bsk CLI not installed")
def test_live_bsk_status_handshake(clean_env):
    status = json.loads(asyncio.run(bsp.bsk_cli_status()))
    assert status["installed"] is True
    assert "daemon" in status
