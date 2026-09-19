"""Process-wide logging configuration for service entry points.

Runtime modules emit through the stdlib `logging` module; this module is the
single place that decides where those records go. Library modules must not
call it — only entry points (API lifespan, Celery task bodies) do.
"""

import logging
import os
import sys

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging() -> None:
    """Configure root logging once per process.

    Level comes from S18_LOG_LEVEL (default INFO). If a host process (e.g.
    Celery's worker bootstrap) already installed root handlers, keep them and
    only apply an explicit S18_LOG_LEVEL; otherwise attach a stderr handler.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # Logs are frequently piped to files or collectors on Windows where the
    # console codepage cannot encode emoji/symbol output; keep that from
    # raising UnicodeEncodeError mid-write.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

    root = logging.getLogger()
    env_level = os.getenv("S18_LOG_LEVEL", "").strip().upper()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(env_level or logging.INFO)
    elif env_level:
        root.setLevel(env_level)
