import os
from types import SimpleNamespace
from typing import Any, Callable

try:
    from celery import Celery
except ModuleNotFoundError:  # pragma: no cover - covered indirectly in tests.
    Celery = None


def _broker_url() -> str:
    return os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")


def _result_backend_url() -> str:
    return os.getenv("CELERY_RESULT_BACKEND", _broker_url())

RUNS_QUEUE = os.getenv("S18_CELERY_RUNS_QUEUE", "celery").strip() or "celery"
INGEST_QUEUE = os.getenv("S18_CELERY_INGEST_QUEUE", "ingest").strip() or "ingest"


if Celery is None:
    class _FallbackTask:
        def __init__(self, fn: Callable[..., Any]):
            self._fn = fn
            self.request = SimpleNamespace(id="missing-celery")

        def __call__(self, *args: Any, **kwargs: Any):
            return self._fn(*args, **kwargs)

        def delay(self, *_args: Any, **_kwargs: Any):
            raise RuntimeError("Celery is not installed. Install the 'celery' package to enqueue tasks.")

    class _FallbackCelery:
        def task(self, *_args: Any, **_kwargs: Any):
            def decorator(fn: Callable[..., Any]):
                return _FallbackTask(fn)

            return decorator

    celery_app = _FallbackCelery()
else:
    celery_app = Celery(
        "s18share",
        broker=_broker_url(),
        backend=_result_backend_url(),
        include=["workers.agent_tasks", "workers.ingest_tasks"],
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone=os.getenv("S18_CELERY_TIMEZONE", "UTC"),
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        task_default_queue=RUNS_QUEUE,
        task_routes={
            "s18share.run_agent": {"queue": RUNS_QUEUE},
            "s18share.resume_agent": {"queue": RUNS_QUEUE},
            "s18share.ingest.materialize": {"queue": INGEST_QUEUE},
            "s18share.ingest.parse_and_chunk": {"queue": INGEST_QUEUE},
            "s18share.ingest.embed_and_index": {"queue": INGEST_QUEUE},
        },
        worker_prefetch_multiplier=int(os.getenv("S18_CELERY_PREFETCH_MULTIPLIER", "1")),
        broker_connection_retry_on_startup=True,
    )
