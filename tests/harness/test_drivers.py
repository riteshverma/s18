from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.drivers import HarnessDriverRegistry
from harness.models import HarnessJobRequest


def test_build_plan_for_claude_uses_prompt_flag():
    req = HarnessJobRequest(provider="claude", prompt="hello world")
    with patch("harness.drivers.shutil.which", return_value="/usr/bin/claude"):
        plan = HarnessDriverRegistry(settings={}).build_plan(req)
    assert plan.command == ["/usr/bin/claude", "-p", "hello world"]
    assert plan.stdin_payload is None


def test_build_plan_for_codex_uses_stdin_default():
    req = HarnessJobRequest(provider="codex", prompt="hello codex")
    with patch("harness.drivers.shutil.which", return_value="/usr/bin/codex"):
        plan = HarnessDriverRegistry(settings={}).build_plan(req)
    assert plan.command == ["/usr/bin/codex"]
    assert plan.stdin_payload == "hello codex"


def test_build_plan_raises_when_binary_missing():
    req = HarnessJobRequest(provider="gemini", prompt="hello")
    with patch("harness.drivers.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError):
            HarnessDriverRegistry(settings={}).build_plan(req)

