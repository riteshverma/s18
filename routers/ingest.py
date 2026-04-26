"""HTTP endpoints for the Power Apps -> Cloud RAG ingest pipeline.

These endpoints intentionally accept the same canonical envelope used
elsewhere in S18 (`tenant_id`, `tenant_tier`, `data_region`, `integration_id`)
so a Power Automate flow, a custom connector, or any other client can talk to
us identically. The actual cloud routing happens in
:mod:`integrations.storage` and :mod:`integrations.vectors`.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from config.settings_loader import settings
from core.run_executor import is_celery_enabled
from core.supabase_auth import require_supabase_user
from integrations.ingest import get_job_store
from integrations.registry import get_integration_adapter
from integrations.storage import get_object_store
from integrations.tenancy import resolve_tenant_context
from integrations.vectors import get_vector_store


router = APIRouter(prefix="/ingest", tags=["Ingest"])


def _resolve_tenant(payload: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, str]:
    return resolve_tenant_context(
        request_payload=payload,
        user=user,
        tenancy_settings=settings.get("tenancy", {}),
    )


def _adapt_canonical(payload: Dict[str, Any], tenant_context: Dict[str, str]):
    payload = dict(payload)
    payload.update(
        {
            "tenant_id": tenant_context["tenant_id"],
            "tenant_tier": tenant_context["tenant_tier"],
            "data_region": tenant_context["data_region"],
        }
    )
    if not payload.get("integration_id"):
        payload["integration_id"] = "powerapps"
    if not payload.get("source_system"):
        payload["source_system"] = "powerapps"
    adapter = get_integration_adapter(
        integration_id=payload.get("integration_id"),
        source_system=payload.get("source_system"),
        tenant_context=tenant_context,
    )
    try:
        canonical = adapter.to_canonical(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return canonical


@router.get("/health")
async def ingest_health(user: Dict[str, Any] = Depends(require_supabase_user)) -> Dict[str, Any]:
    """Backend availability probe.

    Returns the resolved providers for the caller's tenant so a Power Apps
    admin can verify that the right cloud will be used before sending data.
    """
    payload: Dict[str, Any] = {}
    tenant_context = _resolve_tenant(payload, user)
    object_store = get_object_store(tenant_context)
    vector_store = get_vector_store(tenant_context)
    return {
        "status": "ok",
        "tenant_id": tenant_context["tenant_id"],
        "tenant_tier": tenant_context["tenant_tier"],
        "data_region": tenant_context["data_region"],
        "object_store": {"provider": object_store.provider},
        "vector_store": {
            "provider": vector_store.provider,
            "stats": vector_store.stats(),
        },
        "celery_enabled": is_celery_enabled(),
    }


def _enqueue_or_run(
    *,
    job_id: str,
    canonical_payload: Dict[str, Any],
    raw_payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    if is_celery_enabled():
        from workers.ingest_tasks import enqueue_ingest

        task_id = enqueue_ingest(
            job_id=job_id,
            canonical_payload=canonical_payload,
            raw_payload=raw_payload,
        )
        get_job_store().update(job_id, metadata={"celery_task_id": task_id})
        return {"job_id": job_id, "status": "accepted", "task_id": task_id, "execution_backend": "celery"}

    from workers.ingest_tasks import run_ingest_inline

    def _runner() -> None:
        try:
            run_ingest_inline(
                job_id=job_id,
                canonical_payload=canonical_payload,
                raw_payload=raw_payload,
            )
        except Exception as exc:
            get_job_store().append_error(job_id, {"stage": "runner", "reason": str(exc)})
            get_job_store().update(job_id, status="failed")

    background_tasks.add_task(_runner)
    return {"job_id": job_id, "status": "accepted", "execution_backend": "in_process"}


@router.post("/powerapps")
async def ingest_powerapps(
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_supabase_user),
) -> Dict[str, Any]:
    """Primary ingest endpoint for Power Automate / Power Apps connectors.

    Body shape (any field optional unless noted)::

        {
          "tenant_id": "acme",
          "tenant_tier": "enterprise-health",
          "data_region": "in",
          "integration_id": "powerapps",
          "workflow_id": "claims",
          "trigger": {...},
          "record": {...} | "records": [...],
          "files": [
            {"name": "form.pdf", "contentType": "application/pdf",
             "contentBytes": "<base64>"}
          ],
          "consent_ref": "...",
          "idempotency_key": "..."
        }
    """
    tenant_context = _resolve_tenant(payload, user)
    canonical = _adapt_canonical(payload, tenant_context)

    job = get_job_store().create(
        tenant_id=tenant_context["tenant_id"],
        integration_id=canonical.integration_id,
        workflow_id=canonical.workflow_id,
        metadata={
            "auth_sub": user.get("sub"),
            "consent_ref": canonical.consent_ref,
            "idempotency_key": canonical.idempotency_key,
            "source_system": canonical.source_system,
        },
    )
    canonical_payload = canonical.model_dump()
    raw_payload = canonical.raw_payload or payload
    return _enqueue_or_run(
        job_id=job.job_id,
        canonical_payload=canonical_payload,
        raw_payload=raw_payload,
        background_tasks=background_tasks,
    )


@router.post("/powerapps/files")
async def ingest_powerapps_files(
    background_tasks: BackgroundTasks,
    envelope: str = Form(..., description="JSON-encoded canonical envelope (records, trigger, etc.)"),
    files: List[UploadFile] = File(default_factory=list),
    user: Dict[str, Any] = Depends(require_supabase_user),
) -> Dict[str, Any]:
    """Multipart variant for large attachments.

    Power Automate's HTTP action serializes binaries as base64; that is fine
    up to a few MB. For bigger files (CT scans, scanned PDFs) use this
    endpoint with a multipart upload, which Power Apps custom connectors
    natively support via the `multipart/form-data` action.
    """
    try:
        payload = json.loads(envelope) if envelope else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"envelope must be JSON: {exc}")

    payload.setdefault("files", [])
    for upload in files:
        contents = await upload.read()
        payload["files"].append(
            {
                "name": upload.filename,
                "contentType": upload.content_type,
                "contentBytes": base64.b64encode(contents).decode("ascii"),
                "size": len(contents),
            }
        )

    tenant_context = _resolve_tenant(payload, user)
    canonical = _adapt_canonical(payload, tenant_context)
    job = get_job_store().create(
        tenant_id=tenant_context["tenant_id"],
        integration_id=canonical.integration_id,
        workflow_id=canonical.workflow_id,
        metadata={
            "auth_sub": user.get("sub"),
            "consent_ref": canonical.consent_ref,
            "idempotency_key": canonical.idempotency_key,
            "source_system": canonical.source_system,
            "transport": "multipart",
        },
    )
    canonical_payload = canonical.model_dump()
    raw_payload = canonical.raw_payload or payload
    return _enqueue_or_run(
        job_id=job.job_id,
        canonical_payload=canonical_payload,
        raw_payload=raw_payload,
        background_tasks=background_tasks,
    )


@router.get("/jobs/{job_id}")
async def get_ingest_job(
    job_id: str,
    user: Dict[str, Any] = Depends(require_supabase_user),
) -> Dict[str, Any]:
    job = get_job_store().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()
