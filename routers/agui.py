import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from core.event_bus import event_bus
from core.loop import AgentLoop4
from shared.state import active_loops, get_multi_mcp

router = APIRouter(prefix="/agui", tags=["AG-UI"])


class AGUIMessage(BaseModel):
    role: str
    content: str
    id: Optional[str] = None


class RunAgentInput(BaseModel):
    thread_id: str
    run_id: Optional[str] = None
    messages: List[AGUIMessage]
    state: Optional[dict] = None
    context: Optional[dict] = None
    single_pass: bool = False


def _agui(event_type: str, **fields: Any) -> Dict[str, Any]:
    return {"type": event_type, "timestamp_ms": int(time.time() * 1000), **fields}


def _render_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("response", "summary", "final_answer", "result"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(output, default=str)[:4000]
    if isinstance(output, list):
        return json.dumps(output, default=str)[:4000]
    return str(output or "")


def _translate(bus_event: Dict[str, Any]) -> List[Dict[str, Any]]:
    event_type = bus_event.get("type")
    data = bus_event.get("data", {})
    session_id = data.get("session_id")

    if event_type == "run_started":
        return [_agui("RUN_STARTED", thread_id=session_id, run_id=session_id)]

    if event_type == "plan_ready":
        return [_agui("STATE_SNAPSHOT", snapshot={"plan_graph": data.get("plan_graph", {})})]

    if event_type == "step_start":
        return [
            _agui(
                "STATE_DELTA",
                delta=[
                    {
                        "op": "replace",
                        "path": f"/plan_graph/nodes/{data.get('step_id')}/status",
                        "value": "running",
                    }
                ],
            )
        ]

    if event_type == "step_update":
        step_id = data.get("step_id")
        status = data.get("status")
        output = data.get("output")
        events = [
            _agui(
                "STATE_DELTA",
                delta=[
                    {
                        "op": "replace",
                        "path": f"/plan_graph/nodes/{step_id}/status",
                        "value": status,
                    }
                ],
            )
        ]
        if status == "completed" and output:
            message_id = str(uuid.uuid4())
            text = _render_output(output)
            events.extend(
                [
                    _agui("TEXT_MESSAGE_START", message_id=message_id, role="assistant", step_id=step_id),
                    _agui("TEXT_MESSAGE_CONTENT", message_id=message_id, delta=text),
                    _agui("TEXT_MESSAGE_END", message_id=message_id),
                ]
            )
        if status == "waiting_input":
            events.append(
                _agui(
                    "CUSTOM",
                    name="input_required",
                    value={"step_id": step_id, "output": output},
                )
            )
        if status == "failed":
            events.append(
                _agui(
                    "CUSTOM",
                    name="step_failed",
                    value={"step_id": step_id, "error": data.get("error")},
                )
            )
        return events

    if event_type == "tool_call":
        tool_call_id = f"{data.get('step_id')}:{data.get('tool_name')}"
        if data.get("phase") == "start":
            return [
                _agui("TOOL_CALL_START", tool_call_id=tool_call_id, tool_call_name=data.get("tool_name")),
                _agui("TOOL_CALL_ARGS", tool_call_id=tool_call_id, delta=json.dumps(data.get("arguments", {}))),
            ]
        return [
            _agui(
                "TOOL_CALL_END",
                tool_call_id=tool_call_id,
                result_preview=data.get("result_preview"),
            )
        ]

    if event_type == "run_finished":
        status = data.get("status")
        final_event = "RUN_FINISHED" if status == "success" else "RUN_ERROR"
        return [
            _agui(
                final_event,
                thread_id=session_id,
                run_id=session_id,
                status=status,
                error=data.get("error"),
            )
        ]

    return []


@router.post("/run")
async def run_agent(input_data: RunAgentInput, request: Request):
    session_id = input_data.thread_id
    query = next((m.content for m in reversed(input_data.messages) if m.role == "user"), "")
    if not query:
        raise HTTPException(status_code=400, detail="No user message in input.messages")

    queue = await event_bus.subscribe(max_queue_size=500)
    loop = AgentLoop4(get_multi_mcp())
    active_loops[session_id] = loop
    runner = asyncio.create_task(
        loop.run(
            query=query,
            file_manifest=[],
            globals_schema={},
            uploaded_files=[],
            session_id=session_id,
        )
    )

    async def event_generator():
        final_emitted = False
        try:
            while True:
                if await request.is_disconnected():
                    loop.stop()
                    break

                try:
                    bus_event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if runner.done():
                        if not final_emitted:
                            error = runner.exception()
                            event = _agui(
                                "RUN_ERROR" if error else "RUN_FINISHED",
                                thread_id=session_id,
                                run_id=session_id,
                                status="failed" if error else "success",
                                error=str(error) if error else None,
                            )
                            yield {"event": event["type"], "data": json.dumps(event)}
                            final_emitted = True
                        break
                    continue

                if bus_event.get("data", {}).get("session_id") != session_id:
                    continue

                for agui_event in _translate(bus_event):
                    if agui_event["type"] in {"RUN_FINISHED", "RUN_ERROR"}:
                        final_emitted = True
                    yield {"event": agui_event["type"], "data": json.dumps(agui_event)}

                # Deterministic smoke-path mode: stop after first terminal step state and emit RUN_FINISHED.
                if (
                    input_data.single_pass
                    and bus_event.get("type") == "step_update"
                    and bus_event.get("data", {}).get("status") in {"completed", "waiting_input", "failed"}
                ):
                    loop.stop()
                    event = _agui(
                        "RUN_FINISHED",
                        thread_id=session_id,
                        run_id=session_id,
                        status="single_pass",
                    )
                    yield {"event": event["type"], "data": json.dumps(event)}
                    final_emitted = True
                    break

                if bus_event.get("type") == "run_finished":
                    break
        finally:
            event_bus.unsubscribe(queue)
            active_loops.pop(session_id, None)
            if not runner.done():
                runner.cancel()

    return EventSourceResponse(event_generator())
