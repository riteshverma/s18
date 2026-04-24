from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.models import HarnessJobRequest, HarnessJobStatus
from harness.runtime import HarnessRuntime


def test_create_job_persists_state(tmp_path: Path):
    runtime = HarnessRuntime(project_root=tmp_path)
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    request = HarnessJobRequest(
        provider="claude",
        prompt="run task",
        cwd=str(tmp_path / "workspace"),
    )
    with patch("harness.runtime.load_settings", return_value={}):
        with patch("harness.drivers.shutil.which", return_value="/usr/bin/claude"):
            state = asyncio.run(runtime.create_job(request))
            loaded = asyncio.run(runtime.get_job(state.id))

    assert loaded is not None
    assert loaded.status == HarnessJobStatus.ACCEPTED
    assert loaded.command == ["/usr/bin/claude", "-p", "run task"]


def test_create_job_rejects_disallowed_cwd(tmp_path: Path):
    runtime = HarnessRuntime(project_root=tmp_path)
    outside = tmp_path.parent
    request = HarnessJobRequest(
        provider="codex",
        prompt="run task",
        cwd=str(outside),
    )
    with patch("harness.runtime.load_settings", return_value={}):
        with pytest.raises(ValueError, match="relative path"):
            asyncio.run(runtime.create_job(request))


def test_create_job_rejects_path_traversal_segments(tmp_path: Path):
    runtime = HarnessRuntime(project_root=tmp_path)
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    request = HarnessJobRequest(
        provider="codex",
        prompt="run task",
        cwd="../",
    )
    with patch("harness.runtime.load_settings", return_value={}):
        with pytest.raises(ValueError, match="path traversal"):
            asyncio.run(runtime.create_job(request))

