from typing import Any, Dict, Optional

from integrations.contracts import CanonicalRunRequest


async def execute_run(
    run_id: str,
    canonical_request: CanonicalRunRequest,
    audit_context: Optional[Dict[str, Any]] = None,
    tenant_context: Optional[Dict[str, str]] = None,
):
    """Execution boundary for run processing (in-process today, worker-ready later)."""
    from routers.runs import process_run

    return await process_run(
        run_id=run_id,
        canonical_request=canonical_request,
        audit_context=audit_context,
        tenant_context=tenant_context,
    )


async def execute_resume(run_id: str, audit_context: Optional[Dict[str, Any]] = None):
    """Execution boundary for resume processing (in-process today)."""
    from routers.runs import process_resume

    return await process_resume(run_id=run_id, audit_context=audit_context)
