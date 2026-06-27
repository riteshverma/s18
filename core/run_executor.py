import os
from typing import Any, Dict, Optional

from integrations.contracts import CanonicalRunRequest


def execution_backend() -> str:
    return os.getenv("S18_RUN_EXECUTOR", "in_process").strip().lower()


def is_celery_enabled() -> bool:
    return execution_backend() == "celery"


def _merge_run_metadata(run_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    from core.run_store import get_run_store

    store = get_run_store()
    existing = store.get_run(run_id) or {}
    metadata = dict(existing.get("metadata") or {})
    metadata.update(updates)
    return store.update_run(run_id, metadata=metadata)


async def _mark_enqueue_failed(
    run_id: str,
    audit_context: Optional[Dict[str, Any]],
    error: str,
) -> None:
    from core.run_store import get_run_store

    store = get_run_store()
    store.update_status(run_id, "failed", error=error)
    _merge_run_metadata(
        run_id,
        {"execution_backend": "celery", "enqueue_error": error},
    )
    if audit_context and audit_context.get("idempotency_key"):
        try:
            from core.supabase_logging import update_request_status

            await update_request_status(
                idempotency_key=audit_context["idempotency_key"],
                run_id=run_id,
                status="failed",
                error_code=error,
            )
        except Exception:
            pass


async def execute_run(
    run_id: str,
    canonical_request: CanonicalRunRequest,
    audit_context: Optional[Dict[str, Any]] = None,
    tenant_context: Optional[Dict[str, str]] = None,
):
    """Execution boundary for run processing."""
    if is_celery_enabled():
        try:
            from workers.agent_tasks import run_agent_task

            task = run_agent_task.delay(
                run_id,
                canonical_request.model_dump(),
                audit_context,
                tenant_context,
            )
        except Exception as exc:
            await _mark_enqueue_failed(run_id, audit_context, str(exc))
            raise
        _merge_run_metadata(
            run_id,
            {"celery_task_id": task.id, "execution_backend": "celery"},
        )
        return {"run_id": run_id, "status": "accepted", "task_id": task.id}

    from routers.runs import process_run

    return await process_run(
        run_id=run_id,
        canonical_request=canonical_request,
        audit_context=audit_context,
        tenant_context=tenant_context,
    )


async def execute_resume(run_id: str, audit_context: Optional[Dict[str, Any]] = None):
    """Execution boundary for resume processing."""
    if is_celery_enabled():
        try:
            from workers.agent_tasks import resume_agent_task

            task = resume_agent_task.delay(run_id, audit_context)
        except Exception as exc:
            await _mark_enqueue_failed(run_id, audit_context, str(exc))
            raise
        _merge_run_metadata(
            run_id,
            {"celery_task_id": task.id, "execution_backend": "celery"},
        )
        return {"run_id": run_id, "status": "accepted", "task_id": task.id}

    from routers.runs import process_resume

    return await process_resume(run_id=run_id, audit_context=audit_context)
