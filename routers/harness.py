from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from core.supabase_auth import require_supabase_user
from harness.models import HarnessJobRequest
from shared.state import get_harness_runtime

router = APIRouter(prefix="/harness", tags=["Harness"])
runtime = get_harness_runtime()


@router.post("/jobs")
async def create_harness_job(
    request: HarnessJobRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    _ = user
    try:
        state = await runtime.create_job(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    background_tasks.add_task(runtime.run_job, state.id)
    return state.model_dump(mode="json")


@router.get("/jobs")
async def list_harness_jobs(
    limit: int = Query(default=100, ge=1, le=500),
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    _ = user
    jobs = await runtime.list_jobs(limit=limit)
    return [job.model_dump(mode="json") for job in jobs]


@router.get("/jobs/{job_id}")
async def get_harness_job(
    job_id: str,
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    _ = user
    state = await runtime.get_job(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Harness job not found")
    return state.model_dump(mode="json")


@router.post("/jobs/{job_id}/stop")
async def stop_harness_job(
    job_id: str,
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    _ = user
    try:
        state = await runtime.stop_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Harness job not found")
    return state.model_dump(mode="json")


@router.post("/jobs/{job_id}/resume")
async def resume_harness_job(
    job_id: str,
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    _ = user
    try:
        state = await runtime.resume_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Harness job not found")
    return state.model_dump(mode="json")


@router.get("/jobs/{job_id}/events")
async def stream_harness_job_events(
    job_id: str,
    request: Request,
    include_history: bool = Query(default=True),
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    _ = user
    state = await runtime.get_job(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Harness job not found")

    from core.event_bus import event_bus

    queue = await event_bus.subscribe()

    async def _event_generator():
        if include_history:
            for chunk in state.output_tail:
                payload = {"job_id": job_id, "stream": chunk.stream, "line": chunk.text}
                yield {"event": f"job.{chunk.stream}", "data": json.dumps(payload)}
        try:
            while True:
                if await request.is_disconnected():
                    break
                event = await queue.get()
                data = event.get("data", {})
                if data.get("job_id") != job_id:
                    continue
                yield {"event": event.get("type", "message"), "data": json.dumps(data)}
        finally:
            event_bus.unsubscribe(queue)

    return EventSourceResponse(_event_generator())

