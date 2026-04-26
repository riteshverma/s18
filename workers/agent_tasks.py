import asyncio
from typing import Any, Dict, Optional

from core.celery_app import celery_app
from core.run_store import get_run_store
from integrations.contracts import CanonicalRunRequest


def _run_async(coro):
    return asyncio.run(coro)


def _merge_metadata(run_id: str, updates: Dict[str, Any]) -> None:
    store = get_run_store()
    existing = store.get_run(run_id) or {}
    metadata = dict(existing.get("metadata") or {})
    metadata.update(updates)
    store.update_run(run_id, metadata=metadata)


@celery_app.task(name="s18share.run_agent", bind=True)
def run_agent_task(
    self,
    run_id: str,
    canonical_request_payload: Dict[str, Any],
    audit_context: Optional[Dict[str, Any]] = None,
    tenant_context: Optional[Dict[str, str]] = None,
):
    """Celery task wrapper for the existing AgentLoop4 run path."""
    _merge_metadata(
        run_id,
        {"celery_task_id": self.request.id, "execution_backend": "celery"},
    )
    canonical_request = CanonicalRunRequest(**canonical_request_payload)

    from routers.runs import process_run

    return _run_async(
        process_run(
            run_id=run_id,
            canonical_request=canonical_request,
            audit_context=audit_context,
            tenant_context=tenant_context,
        )
    )


@celery_app.task(name="s18share.resume_agent", bind=True)
def resume_agent_task(self, run_id: str, audit_context: Optional[Dict[str, Any]] = None):
    """Celery task wrapper for resuming a saved AgentLoop4 run."""
    _merge_metadata(
        run_id,
        {"celery_task_id": self.request.id, "execution_backend": "celery"},
    )

    from routers.runs import process_resume

    return _run_async(process_resume(run_id=run_id, audit_context=audit_context))
