import os

from celery import Celery


def _broker_url() -> str:
    return os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")


def _result_backend_url() -> str:
    return os.getenv("CELERY_RESULT_BACKEND", _broker_url())


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
    worker_prefetch_multiplier=int(os.getenv("S18_CELERY_PREFETCH_MULTIPLIER", "1")),
    broker_connection_retry_on_startup=True,
)
