"""Unit tests for the hardened sandbox execution path.

The default run_user_code path (multi_mcp=None) must execute code in an
isolated subprocess with a configurable timeout, while keeping the return
contract (status/result/raw/logs/execution_time/total_time) identical for
callers. The in-process path (multi_mcp set) is exercised via mocks where
needed.
"""

import asyncio
import time

import pytest

import tools.sandbox as sandbox
from config.settings_loader import get_sandbox_timeout_seconds
from tools.sandbox import check_code_safety, run_user_code


@pytest.fixture(autouse=True)
def _isolated_session_vars(monkeypatch):
    """Keep session state out of the repo's action/sandbox_state dir."""
    monkeypatch.setattr(sandbox, "load_session_vars", lambda session_id: {})
    monkeypatch.setattr(sandbox, "save_session_vars", lambda session_id, value: None)


def _run(code: str) -> dict:
    return asyncio.run(run_user_code(code, multi_mcp=None, session_id="test_sandbox_subprocess"))


def test_subprocess_success_print_and_result():
    result = _run('print("hello from sandbox")\nresult = 1 + 2\n')

    assert result["status"] == "success"
    assert result["result"] == {"result": 3}
    assert result["raw"] == result["result"]
    assert "hello from sandbox" in result["logs"]
    assert "execution_time" in result
    assert "total_time" in result


def test_return_dict_result_contract():
    result = _run("return {'x': 5}")

    assert result["status"] == "success"
    assert result["result"] == {"x": 5}


def test_code_exception_returns_error_dict():
    result = _run("x = 1 / 0")

    assert result["status"] == "error"
    assert result["error"].startswith("ZeroDivisionError")
    assert "traceback" in result


def test_blocked_code_never_executes(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("blocked code must not reach execution")

    monkeypatch.setattr(sandbox, "_run_user_code_in_subprocess", _fail)

    result = _run('import os\nos.system("echo hacked")')

    assert result["status"] == "blocked"
    assert "Security violation" in result["error"]


def test_timeout_returns_error_shape_quickly(monkeypatch):
    monkeypatch.setattr(sandbox, "_get_sandbox_timeout_seconds", lambda: 1.0)

    start = time.perf_counter()
    result = _run("import time\nfor _ in range(1000):\n    time.sleep(5)\n")
    elapsed = time.perf_counter() - start

    assert result["status"] == "error"
    assert "timed out" in result["error"]
    assert elapsed < 8


def test_session_vars_are_injected(monkeypatch):
    monkeypatch.setattr(
        sandbox, "load_session_vars", lambda session_id: {"injected_var": 21}
    )

    result = _run("result = injected_var * 2")

    assert result["status"] == "success"
    assert result["result"] == {"result": 42}


def test_allowed_modules_are_available():
    result = _run("import math\nresult = math.floor(3.7)")

    assert result["status"] == "success"
    assert result["result"] == {"result": 3}


def test_check_code_safety_flags_dangerous_patterns():
    is_safe, violations = check_code_safety("eval('1+1')")

    assert not is_safe
    assert violations[0]["description"] == "Eval execution"

    is_safe, violations = check_code_safety("x = 1 + 2")

    assert is_safe
    assert violations == []


def test_settings_default_timeout_is_configured():
    assert get_sandbox_timeout_seconds() >= 1
