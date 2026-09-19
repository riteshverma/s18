import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core.logging_setup import configure_logging

_REPO_ROOT = str(Path(__file__).resolve().parents[2])


@pytest.fixture()
def clean_root_logger():
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers = []
    try:
        yield root
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        import core.logging_setup as logging_setup
        logging_setup._CONFIGURED = False


def _run_in_clean_process(body: str) -> subprocess.CompletedProcess:
    """Run configure_logging scenarios in a fresh interpreter.

    pytest's logging plugin attaches a capture handler to the root logger
    during the call phase, which makes the cold-start branch unreachable
    in-process; a subprocess sees the real behavior.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=60,
    )


def test_configure_logging_attaches_handler_and_is_idempotent():
    proc = _run_in_clean_process(
        """
        import logging
        from core.logging_setup import LOG_FORMAT, configure_logging
        configure_logging()
        configure_logging()
        root = logging.getLogger()
        ours = sum(
            1 for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and getattr(getattr(h, "formatter", None), "_fmt", None) == LOG_FORMAT
        )
        print("HANDLERS", ours)
        print("LEVEL", logging.getLevelName(root.level))
        root.handlers[0].stream.flush()
        logging.getLogger("x").info("hello")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "HANDLERS 1" in proc.stdout
    assert "LEVEL INFO" in proc.stdout
    assert "INFO    x | hello" in proc.stderr


def test_configure_logging_respects_s18_log_level(clean_root_logger, monkeypatch):
    monkeypatch.setenv("S18_LOG_LEVEL", "DEBUG")
    configure_logging()
    assert clean_root_logger.level == logging.DEBUG


def test_configure_logging_keeps_existing_handlers(clean_root_logger, monkeypatch):
    existing = logging.NullHandler()
    clean_root_logger.addHandler(existing)

    monkeypatch.setenv("S18_LOG_LEVEL", "WARNING")
    configure_logging()

    assert clean_root_logger.level == logging.WARNING
    assert existing in clean_root_logger.handlers


def test_log_step_emits_payload_via_logging(caplog):
    from core.utils import log_step

    with caplog.at_level(logging.INFO, logger="core.utils"):
        log_step("Step finished", payload={"k": "v"}, symbol="X")

    assert "X Step finished" in caplog.text
    assert '"k": "v"' in caplog.text


def test_log_error_includes_exception_detail(caplog):
    from core.utils import log_error

    with caplog.at_level(logging.ERROR, logger="core.utils"):
        log_error("boom", err=ValueError("bad input"))

    assert "boom" in caplog.text
    assert "bad input" in caplog.text
