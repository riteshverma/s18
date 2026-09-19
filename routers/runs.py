# Runs Router - Handles agent run execution, listing, and management
import asyncio
import json
import logging
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Body, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from shared.state import (
    active_loops,
    PROJECT_ROOT,
)
from core.graph_adapter import nx_to_reactflow
from core.supabase_auth import require_supabase_user
from core.supabase_logging import (
    build_idempotency_key,
    compute_payload_hash,
    log_inbound_request,
)
from core.run_store import generate_run_id, get_run_store
from core.run_executor import execute_resume, execute_run, is_celery_enabled
# Run execution lives in core.run_service; re-exported here so the endpoints
# and existing tests keep their names.
from core.run_service import (
    _build_memory_context,  # noqa: F401 -- re-export; tests call runs._build_memory_context
    _find_session_file,
    _load_runs_index,
    _normalize_run_status,
    multi_mcp,
)
from config.settings_loader import settings, get_run_poll_timeout
from integrations.contracts import CanonicalRunRequest
from integrations.tenancy import (
    can_route_to_growth,
    resolve_tenant_context,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Runs"])
run_store = get_run_store()


def _has_idempotency_signal(canonical_request: CanonicalRunRequest) -> bool:
    """True when the request carries a caller-supplied idempotency signal.

    Only a client idempotency_key or an external_event_id marks a request as a
    retry candidate; plain repeated queries hash to the same payload-hash key
    but must keep starting fresh runs.
    """
    return bool(canonical_request.idempotency_key or canonical_request.external_event_id)


def _deduped_run_response(
    adapter,
    existing_run: Dict[str, Any],
    canonical_request: CanonicalRunRequest,
) -> Dict[str, Any]:
    """Build the create-run response for a retry that matched an existing run."""
    return adapter.from_canonical(
        {
            "id": existing_run["id"],
            "request_id": existing_run.get("request_id"),
            "status": _normalize_run_status(existing_run.get("status")),
            "created_at": existing_run.get("created_at") or datetime.now().isoformat(),
            "query": canonical_request.query,
            "idempotency_key": existing_run.get("idempotency_key"),
            "tenant_id": existing_run.get("tenant_id") or canonical_request.tenant_id,
            "tenant_tier": existing_run.get("tenant_tier") or canonical_request.tenant_tier,
            "data_region": existing_run.get("data_region") or canonical_request.data_region,
            "poll_timeout_seconds": get_run_poll_timeout(),
            # Lets callers and ops tell a deduped retry from a fresh accept.
            "deduplicated": True,
        },
        canonical_request,
    )


# === Pydantic Models ===

class RunRequest(BaseModel):
    query: str = Field(min_length=1)
    model: str = None  # Will use settings default if not provided
    contract_version: Optional[str] = "v1"
    integration_id: Optional[str] = None
    workflow_id: Optional[str] = None
    source_system: Optional[str] = "s18"
    tenant_id: Optional[str] = None
    tenant_tier: Optional[str] = None
    data_region: Optional[str] = None
    external_event_id: Optional[str] = None
    consent_ref: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    skill_id: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.model is None:
            self.model = settings.get("agent", {}).get("default_model", "gemini-2.5-flash")


class RunResponse(BaseModel):
    id: str
    status: str
    created_at: str
    query: str


class UserInputRequest(BaseModel):
    node_id: str
    response: str


# === Endpoints ===

@router.post("/runs")
async def create_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    # Epoch-ms prefix keeps ids roughly time-ordered; the random suffix makes
    # concurrent creations unique where a bare timestamp would collide.
    run_id = generate_run_id()
    request_payload = request.model_dump()
    tenant_context = resolve_tenant_context(
        request_payload=request_payload,
        user=user,
        tenancy_settings=settings.get("tenancy", {}),
    )
    request_payload.update(
        {
            "tenant_id": tenant_context["tenant_id"],
            "tenant_tier": tenant_context["tenant_tier"],
            "data_region": tenant_context["data_region"],
        }
    )
    from integrations.registry import get_integration_adapter

    adapter = get_integration_adapter(
        integration_id=request.integration_id,
        source_system=request.source_system,
        tenant_context=tenant_context,
    )
    try:
        canonical_request = adapter.to_canonical(request_payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    payload_hash = compute_payload_hash(canonical_request.query, canonical_request.raw_payload)
    source_system = (canonical_request.source_system or "s18").strip().lower()
    idempotency_key = canonical_request.idempotency_key or build_idempotency_key(
        source_system,
        canonical_request.external_event_id or "",
        payload_hash,
    )
    request_id = f"req_{run_id}"

    # Retry dedupe: only requests carrying a real idempotency signal (a
    # client-supplied key or an external event id) may collapse onto an
    # existing run. The lookup is tenant-scoped so identical payloads from
    # different tenants never collide.
    dedupe_signal = _has_idempotency_signal(canonical_request)
    if dedupe_signal:
        existing_run = await asyncio.to_thread(
            run_store.get_run_by_idempotency_key,
            idempotency_key,
            canonical_request.tenant_id,
        )
        if existing_run:
            logger.info(
                "Deduplicated retry for tenant=%s key=%s: returning existing run %s",
                canonical_request.tenant_id,
                idempotency_key,
                existing_run["id"],
            )
            return _deduped_run_response(adapter, existing_run, canonical_request)

    audit_context = {
        "request_id": request_id,
        "idempotency_key": idempotency_key,
    }
    if can_route_to_growth(tenant_context, settings.get("tenancy", {})):
        logger.info(
            "[%s] Growth routing hook active for tenant=%s tier=%s region=%s",
            run_id,
            tenant_context["tenant_id"],
            tenant_context["tenant_tier"],
            tenant_context["data_region"],
        )

    try:
        await log_inbound_request(
            {
                "run_id": run_id,
                "request_id": request_id,
                "source_system": source_system,
                "external_event_id": canonical_request.external_event_id,
                "idempotency_key": idempotency_key,
                "payload_hash": payload_hash,
                "query": canonical_request.query,
                "raw_payload": canonical_request.raw_payload,
                "integration_id": canonical_request.integration_id,
                "workflow_id": canonical_request.workflow_id,
                "contract_version": canonical_request.contract_version,
                "auth_sub": user.get("sub"),
                "auth_email": user.get("email"),
                "consent_ref": canonical_request.consent_ref,
                "tenant_id": canonical_request.tenant_id,
                "tenant_tier": canonical_request.tenant_tier,
                "data_region": canonical_request.data_region,
                "status": "accepted",
            }
        )
    except Exception as e:
        logger.warning("Supabase inbound logging failed for run %s: %s", run_id, e)

    await asyncio.to_thread(
        run_store.upsert_run,
        run_id=run_id,
        status="accepted",
        query=canonical_request.query,
        created_at=datetime.now().isoformat(),
        request_id=request_id,
        # Only signal-carrying requests persist a dedupe key: the unique
        # (idempotency_key, tenant_id) index backstops concurrent creates,
        # while plain repeated queries must stay free to re-run.
        idempotency_key=idempotency_key if dedupe_signal else None,
        integration_id=canonical_request.integration_id,
        workflow_id=canonical_request.workflow_id,
        tenant_id=canonical_request.tenant_id,
        tenant_tier=canonical_request.tenant_tier,
        data_region=canonical_request.data_region,
        metadata={
            "source_system": source_system,
            "contract_version": canonical_request.contract_version,
            "skill_id": canonical_request.skill_id,
        },
    )

    # Celery mode enqueues immediately; local mode keeps the existing in-process background path.
    if is_celery_enabled():
        await execute_run(run_id, canonical_request, audit_context, tenant_context)
    else:
        background_tasks.add_task(execute_run, run_id, canonical_request, audit_context, tenant_context)
    
    return adapter.from_canonical(
        {
        "id": run_id,
        "request_id": request_id,
        "status": "starting",
        "created_at": datetime.now().isoformat(),
        "query": canonical_request.query,
        "idempotency_key": idempotency_key,
        "tenant_id": canonical_request.tenant_id,
        "tenant_tier": canonical_request.tenant_tier,
        "data_region": canonical_request.data_region,
        "poll_timeout_seconds": get_run_poll_timeout(),
        },
        canonical_request,
    )


@router.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    if run_id in active_loops:
        return {
            "id": run_id,
            "status": "running",
            "poll_timeout_seconds": get_run_poll_timeout(),
        }

    summaries_dir = PROJECT_ROOT / "memory" / "session_summaries_index"
    found_file = _find_session_file(run_id, summaries_dir)
    if not found_file:
        raise HTTPException(status_code=404, detail="Run not found")

    request_id = f"resume_{run_id}"
    idempotency_key = f"resume:{run_id}"
    audit_context = {"request_id": request_id, "idempotency_key": idempotency_key}
    try:
        await log_inbound_request(
            {
                "run_id": run_id,
                "request_id": request_id,
                "source_system": "s18",
                "idempotency_key": idempotency_key,
                "payload_hash": f"resume:{run_id}",
                "query": "resume_run",
                "raw_payload": {"event": "resume", "run_id": run_id},
                "auth_sub": user.get("sub"),
                "auth_email": user.get("email"),
                "status": "accepted",
            }
        )
    except Exception as e:
        logger.warning("Supabase inbound resume logging failed for run %s: %s", run_id, e)

    await asyncio.to_thread(
        run_store.update_status,
        run_id,
        "starting",
        metadata={"resume": True},
    )
    if is_celery_enabled():
        await execute_resume(run_id, audit_context)
    else:
        background_tasks.add_task(execute_resume, run_id, audit_context)
    return {
        "id": run_id,
        "status": "resuming",
        "created_at": datetime.now().isoformat(),
        "poll_timeout_seconds": get_run_poll_timeout(),
    }


@router.get("/runs")
async def list_runs(user: Dict[str, Any] = Depends(require_supabase_user)):
    """List runs from durable registry, with disk fallback for legacy sessions."""
    stored_runs = await asyncio.to_thread(run_store.list_runs, 300)
    if stored_runs:
        return stored_runs
    runs = await _load_runs_index()
    return [{k: v for k, v in row.items() if k != "path"} for row in runs]


@router.get("/runs/{run_id}")
async def get_run(run_id: str, user: Dict[str, Any] = Depends(require_supabase_user)):
    """Get graph state for a run"""
    stored = await asyncio.to_thread(run_store.get_run, run_id)

    # Check memory first (if running), then disk fallback.
    if run_id in active_loops:
        loop = active_loops[run_id]
        context = getattr(loop, "context", None)
        plan_graph = getattr(context, "plan_graph", None) if context else None

        if plan_graph is not None:
            react_flow = nx_to_reactflow(plan_graph)
            node_statuses = [
                data.get("status", "pending")
                for node_id, data in plan_graph.nodes(data=True)
                if node_id != "ROOT"
            ]
            if any(s == "waiting_input" for s in node_statuses):
                status = "waiting_input"
            elif any(s == "running" for s in node_statuses):
                status = "running"
            elif any(s == "failed" for s in node_statuses):
                status = "failed"
            elif node_statuses and all(s == "completed" for s in node_statuses):
                status = "completed"
            else:
                status = plan_graph.graph.get("status", "running")
            await asyncio.to_thread(run_store.update_status, run_id, _normalize_run_status(status))

            return {
                "id": run_id,
                "status": status,
                "graph": react_flow,
                "poll_timeout_seconds": get_run_poll_timeout(),
            }

        # Run exists but graph has not been initialized yet; avoid transient 404.
        return {
            "id": run_id,
            "status": "running",
            "graph": {"nodes": [], "edges": []},
            "poll_timeout_seconds": get_run_poll_timeout(),
        }
    
    # Search disk
    summaries_dir = PROJECT_ROOT / "memory" / "session_summaries_index"
    found_file = _find_session_file(run_id, summaries_dir)
        
    if found_file:
        data = await asyncio.to_thread(lambda: json.loads(found_file.read_text(encoding="utf-8", errors="ignore")))
        # Reconstruct Graph to use adapter
        import networkx as nx
        if "edges" in data:
            G = nx.node_link_graph(data, edges="edges")
        elif "links" in data:
            G = nx.node_link_graph(data, edges="links")
        elif "link" in data:
            # networkx default uses 'link' (singular)
            G = nx.node_link_graph(data, edges="link")
        else:
            data["edges"] = []
            G = nx.node_link_graph(data, edges="edges")
        react_flow = nx_to_reactflow(G)
        
        # Determine status: Running if in memory, else use file status
        status = "running" if run_id in active_loops else data.get("graph", {}).get("status", "completed")
        session_file = str(found_file)
        await asyncio.to_thread(
            run_store.update_status,
            run_id,
            _normalize_run_status(status),
            session_file=session_file,
        )

        return {
            "id": run_id,
            "status": status,
            "graph": react_flow,
            "poll_timeout_seconds": get_run_poll_timeout(),
        }

    # Bridge short startup races before background task registers active loop/session file.
    if run_id.isdigit():
        try:
            ts_value = int(run_id)
            created_ms = ts_value if len(run_id) >= 13 else ts_value * 1000
            age_ms = int(datetime.now().timestamp() * 1000) - created_ms
            if 0 <= age_ms <= 30000:
                return {
                    "id": run_id,
                    "status": "starting",
                    "graph": {"nodes": [], "edges": []},
                    "poll_timeout_seconds": get_run_poll_timeout(),
                }
        except (OverflowError, ValueError):
            pass

    if stored:
        return {
            "id": run_id,
            "status": _normalize_run_status(stored.get("status")),
            "graph": {"nodes": [], "edges": []},
            "poll_timeout_seconds": get_run_poll_timeout(),
            "summary": stored.get("summary"),
            "error": stored.get("error"),
            "created_at": stored.get("created_at"),
            "updated_at": stored.get("updated_at"),
        }

    raise HTTPException(status_code=404, detail="Run not found")


@router.post("/runs/{run_id}/input")
async def provide_input(
    run_id: str,
    request: UserInputRequest,
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    """Provide user input to a running agent (e.g., ClarificationAgent response)"""
    if run_id in active_loops:
        loop = active_loops[run_id]
        if loop.context:
            try:
                def _is_clarification_node(node_id: str, node_data: Dict[str, Any]) -> bool:
                    status = node_data.get("status")
                    if status not in ["running", "waiting_input"]:
                        return False
                    tags = " ".join(
                        str(node_data.get(k, "")).lower()
                        for k in ["agent", "type", "label", "description"]
                    )
                    if "clarification" in tags or "clarify" in str(node_id).lower():
                        return True
                    writes = node_data.get("writes") or []
                    return any("user_clarification" in str(w).lower() for w in writes)

                selected_node_id: Optional[str] = None
                selected_node_data: Optional[Dict[str, Any]] = None

                # Prefer explicit node id from client, then fall back to first active clarification node.
                if request.node_id in loop.context.plan_graph.nodes:
                    candidate = loop.context.plan_graph.nodes[request.node_id]
                    if _is_clarification_node(request.node_id, candidate):
                        selected_node_id = request.node_id
                        selected_node_data = candidate

                if selected_node_id is None:
                    for node_id, node_data in loop.context.plan_graph.nodes(data=True):
                        if _is_clarification_node(node_id, node_data):
                            selected_node_id = node_id
                            selected_node_data = node_data
                            break

                if selected_node_id is None or selected_node_data is None:
                    raise HTTPException(status_code=400, detail="No clarification node is currently waiting for input")

                # Get the writes key for this clarification
                writes = selected_node_data.get("writes") or []
                write_key = writes[0] if writes else f"user_clarification_{selected_node_id}"

                # Store user input in globals_schema
                loop.context.plan_graph.graph.setdefault("globals_schema", {})[write_key] = request.response

                # Mark the clarification step as completed with user's response as output
                loop.context.plan_graph.nodes[selected_node_id]["output"] = {
                    "clarificationMessage": "User provided clarification",
                    "node_id": selected_node_id,
                    write_key: request.response,
                }
                loop.context.plan_graph.nodes[selected_node_id]["status"] = "completed"

                # Save the session
                await asyncio.to_thread(loop.context._save_session)
                await asyncio.to_thread(run_store.update_status, run_id, "running")

                return {
                    "id": run_id,
                    "status": "input_received",
                    "node_id": selected_node_id,
                    "stored_as": write_key,
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Context not initialized")
    
    raise HTTPException(status_code=404, detail="Active run not found")


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str, user: Dict[str, Any] = Depends(require_supabase_user)):
    """Stop a running agent execution"""
    if run_id in active_loops:
        loop = active_loops[run_id]
        loop.stop()
        await asyncio.to_thread(run_store.update_status, run_id, "stopped")
        return {"id": run_id, "status": "stopping"}

    stored = await asyncio.to_thread(run_store.get_run, run_id)
    if stored:
        await asyncio.to_thread(run_store.update_status, run_id, "stopped")
        return {"id": run_id, "status": "stopped"}

    raise HTTPException(status_code=404, detail="Active run not found")


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, user: Dict[str, Any] = Depends(require_supabase_user)):
    """Delete a run from disk and memory"""
    # 1. Stop if running
    if run_id in active_loops:
        loop = active_loops[run_id]
        loop.stop()
        del active_loops[run_id]
        
    # 2. Delete file
    summaries_dir = PROJECT_ROOT / "memory" / "session_summaries_index"
    deleted = False
    found_file = _find_session_file(run_id, summaries_dir)
    if found_file:
        try:
            found_file.unlink()
            deleted = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
            
    if not deleted and run_id not in active_loops: # If wasn't running and file not found
        # Might be okay if it was just in memory? But we are memory-less persistence mostly
        # Let's return success if we stopped it at least, or warn
        pass

    return {"id": run_id, "status": "deleted"}


# === AGENT TESTING ENDPOINTS ===

class AgentTestRequest(BaseModel):
    input: Optional[str] = None

@router.post("/runs/{run_id}/agent/{node_id}/test")
async def test_agent(
    run_id: str,
    node_id: str,
    request: AgentTestRequest = Body(None),
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    """
    Re-run a single agent in TEST MODE (sandbox).
    - Loads the session
    - Extracts the node's inputs from globals_schema
    - Runs the agent with those inputs
    - Returns the NEW output WITHOUT saving to session
    """
    try:
        # 1. Find the session file
        summaries_dir = PROJECT_ROOT / "memory" / "session_summaries_index"
        found_file = _find_session_file(run_id, summaries_dir)
        
        if not found_file:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 2. Load session data
        import networkx as nx
        session_data = json.loads(found_file.read_text(encoding="utf-8", errors="ignore"))
        if "edges" in session_data:
            G = nx.node_link_graph(session_data, edges="edges")
        elif "links" in session_data:
            G = nx.node_link_graph(session_data, edges="links")
        elif "link" in session_data:
            G = nx.node_link_graph(session_data, edges="link")
        else:
            session_data["edges"] = []
            G = nx.node_link_graph(session_data, edges="edges")
        
        # 3. Find the node
        if node_id not in G.nodes:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found in session")
        
        node_data = G.nodes[node_id]
        agent_type = node_data.get("agent")
        
        if not agent_type:
            raise HTTPException(status_code=400, detail="Node has no agent type")
        
        # 4. Collect inputs from globals_schema based on 'reads'
        globals_schema = G.graph.get("globals_schema", {})
        reads = node_data.get("reads", [])
        inputs = {key: globals_schema.get(key) for key in reads if key in globals_schema}
        
        # 5. Build the input payload helper
        def build_agent_input(instruction=None, previous_output=None, iteration_context=None):
            # Determine base values
            prompt_to_use = instruction or node_data.get("agent_prompt", node_data.get("description", ""))
            query_to_use = G.graph.get("original_query", "")

            # Apply overrides from request if present
            if request and request.input:
                if agent_type == "PlannerAgent":
                    query_to_use = request.input
                else:
                    # For downstream agents, the input overrides the instruction/goal
                    prompt_to_use = request.input

            payload = {
                "step_id": node_id,
                "agent_prompt": prompt_to_use,
                "reads": reads,
                "writes": node_data.get("writes", []),
                "inputs": inputs,
                "original_query": query_to_use,
                "session_context": {
                    "session_id": run_id,
                    "created_at": G.graph.get("created_at", ""),
                    "file_manifest": G.graph.get("file_manifest", []),
                },
                **({"previous_output": previous_output} if previous_output else {}),
                **({"iteration_context": iteration_context} if iteration_context else {})
            }
             # Formatter-specific additions
            if agent_type == "FormatterAgent":
                global_data = G.graph.get('globals_schema', {}).copy()
                logger.debug("DEBUG FORMATTER: run_id=%s", run_id)
                logger.debug("DEBUG FORMATTER: file=%s", found_file)
                logger.debug("DEBUG FORMATTER: globals_keys=%s", list(global_data.keys()))
                if 'formatted_report_T010' in global_data:
                    logger.debug("DEBUG FORMATTER: FOUND STALE KEY 'formatted_report_T010'!")
                payload["all_globals_schema"] = global_data
            return payload

        # 6. Execute with ReAct Loop (Max 15 turns)
        from agents.base_agent import AgentRunner
        from memory.context import ExecutionContextManager
        
        agent_runner = AgentRunner(multi_mcp)
        temp_context = ExecutionContextManager.__new__(ExecutionContextManager)
        temp_context.plan_graph = G
        temp_context.multi_mcp = multi_mcp

        max_turns = 15
        current_input = build_agent_input()
        iterations_data = []
        final_output = {}
        final_execution_result = None

        for turn in range(1, max_turns + 1):
            logger.info("Test Mode: %s Iteration %d/%d", agent_type, turn, max_turns)
            
            # Run Agent
            result = await agent_runner.run_agent(agent_type, current_input)
            
            if not result["success"]:
                return {
                    "status": "error",
                    "error": result.get("error", "Agent execution failed"),
                    "node_id": node_id,
                    "agent_type": agent_type
                }
            
            output = result["output"]
            final_output = output # Update final output
            iterations_data.append({"iteration": turn, "output": output})
            
            # 1. Check for 'call_tool' (ReAct)
            if output.get("call_tool"):
                tool_call = output["call_tool"]
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
                
                logger.info("Test Mode: Executing Tool: %s", tool_name)
                
                try:
                    # Execute tool via MultiMCP
                    tool_result = await multi_mcp.route_tool_call(tool_name, tool_args)
                    
                    # Serialize result content
                    if isinstance(tool_result.content, list):
                        result_str = "\n".join([str(item.text) for item in tool_result.content if hasattr(item, "text")])
                    else:
                        result_str = str(tool_result.content)

                    # Save result to history
                    iterations_data[-1]["tool_result"] = result_str
                    
                    # Prepare input for next iteration
                    instruction = output.get("thought", "Use the tool result to generate the final output.")
                    if turn == max_turns - 1:
                         instruction += " \n\n⚠️ WARNING: This is your FINAL turn. You MUST provide the final 'output' now. Do not call any more tools. Summarize what you have."

                    current_input = build_agent_input(
                        instruction=instruction,
                        previous_output=output,
                        iteration_context={"tool_result": result_str}
                    )
                    continue # Loop to next turn

                except Exception as e:
                    logger.warning("Test Mode: Tool Execution Failed: %s", e)
                    current_input = build_agent_input(
                        instruction="The tool execution failed. Try a different approach or tool.",
                        previous_output=output,
                        iteration_context={"tool_result": f"Error: {str(e)}"}
                    )
                    continue

            # 2. Check for call_self (Legacy/Advanced recursion)
            elif output.get("call_self"):
                # Handle code execution if needed
                if temp_context._has_executable_code(output):
                     # Pass 'inputs' as overrides so variables from prev iterations (like ipl_urls_1A) are available
                    execution_result = await temp_context._auto_execute_code(node_id, output, input_overrides=inputs)
                    final_execution_result = execution_result
                    
                    # Save result to history
                    iterations_data[-1]["execution_result"] = execution_result

                    if execution_result.get("status") == "success":
                        execution_data = execution_result.get("result", {})
                        inputs = {**inputs, **execution_data}  # Update inputs for next iteration
                
                # Prepare input for next iteration
                current_input = build_agent_input(
                    instruction=output.get("next_instruction", "Continue the task"),
                    previous_output=output,
                    iteration_context=output.get("iteration_context", {})
                )
                continue

            # 3. Success (No tool call, just output)
            else:
                 # Execute code if present (Final Iteration)
                if temp_context._has_executable_code(output):
                     # Pass 'inputs' as overrides here too
                    final_execution_result = await temp_context._auto_execute_code(node_id, output, input_overrides=inputs)
                    iterations_data[-1]["execution_result"] = final_execution_result
                    if final_execution_result:
                         final_output = temp_context._merge_execution_results(output, final_execution_result)
                break # Exit loop
        
        # 8. Get the original output for comparison
        original_output = node_data.get("output", {})
        
        # Ensure final_execution_result is passed even if loop broke early
        if not final_execution_result and iterations_data:
             final_execution_result = iterations_data[-1].get("execution_result")

        return {
            "status": "success",
            "node_id": node_id,
            "agent_type": agent_type,
            "original_output": original_output,
            "test_output": final_output,
            "execution_result": final_execution_result,
            "inputs_used": inputs,
            "iterations": iterations_data # Optional: Pass full iterations if needed by UI
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent test execution failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/{run_id}/agent/{node_id}/save")
async def save_agent_test(
    run_id: str,
    node_id: str,
    request: Request,
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    """
    Save test results back to the session file.
    - Updates the node's output
    - Updates globals_schema with new writes
    """
    try:
        body = await request.json()
        new_output = body.get("output")
        
        if not new_output:
            raise HTTPException(status_code=400, detail="Missing 'output' in request body")
        
        # 1. Find the session file
        summaries_dir = PROJECT_ROOT / "memory" / "session_summaries_index"
        found_file = _find_session_file(run_id, summaries_dir)
        
        if not found_file:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 2. Load and update session
        import networkx as nx
        session_data = json.loads(found_file.read_text(encoding="utf-8", errors="ignore"))
        if "edges" in session_data:
            G = nx.node_link_graph(session_data, edges="edges")
        elif "links" in session_data:
            G = nx.node_link_graph(session_data, edges="links")
        elif "link" in session_data:
            G = nx.node_link_graph(session_data, edges="link")
        else:
            session_data["edges"] = []
            G = nx.node_link_graph(session_data, edges="edges")
        
        if node_id not in G.nodes:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        # 2.5 SPECIAL HANDLING: PlannerAgent Graph Update
        # If this is a PlannerAgent (has plan_graph in output), we must REBUILD the graph structure.
        if "plan_graph" in new_output:
            logger.info("Planner Update Detected for %s. Rebuilding graph...", node_id)
            plan_graph = new_output["plan_graph"]
            
            # 1. Keep crucial nodes (ROOT and the Planner/Query node itself)
            # We assume node_id is the Planner node. 
            nodes_to_keep = ["ROOT", node_id] 
            
            # 2. Identify nodes to remove (all existing nodes except kept ones)
            nodes_to_remove = [n for n in G.nodes if n not in nodes_to_keep]
            for n in nodes_to_remove:
                G.remove_node(n)
                
            # 3. Add NEW nodes from plan
            # plan_graph['nodes'] is a list of dicts
            new_nodes = plan_graph.get("nodes", [])
            for n_data in new_nodes:
                nid = n_data["id"] 
                # Ensure we don't overwrite the planner if it's in the list for some reason (unlikely but safe)
                if nid not in G.nodes:
                    G.add_node(nid, **n_data)
                    # Initialize status for new nodes
                    G.nodes[nid]["status"] = "idle"
            
            # 4. Add NEW edges from plan
            # plan_graph['edges'] or 'links'
            new_edges = plan_graph.get("edges", plan_graph.get("links", []))
            
            # clear existing edges? We already removed nodes, so connected edges are gone.
            # But we need to ensure ROOT -> Planner connection exists if not implicitly handled.
            # In our schema, ROOT->Query (Planner). 
            # The plan_graph usually defines edges from "ROOT" to the first new node.
            # We must REMAP "ROOT" in the plan to be "node_id" (The Planner Node) 
            # so that the flow is ROOT -> Planner -> FirstNode
            
            for edge in new_edges:
                src = edge["source"]
                tgt = edge["target"]
                
                # REMAP ROOT -> Current Planner Node
                if src == "ROOT":
                    src = node_id
                    
                G.add_edge(src, tgt)
                
            # Ensure ROOT is connected to Planner (node_id)
            if not G.has_edge("ROOT", node_id):
                G.add_edge("ROOT", node_id)
                
            logger.info("Graph Rebuilt. Nodes: %d, Edges: %d", len(G.nodes), len(G.edges))

        node_data = G.nodes[node_id]
        writes = node_data.get("writes", [])
        
        # 3. Update node output
        node_data["output"] = new_output
        node_data["last_tested"] = datetime.now().isoformat()
        node_data["status"] = "completed"
        
        # 4. Update globals_schema with execution results if available
        # 4. Update globals_schema
        # CRITICAL FIX: Prioritize the 'merged' output (new_output) which contains the actual results
        # Execution result is less reliable as it might be raw or unmerged
        
        globals_schema = G.graph.get("globals_schema", {})
        
        # 1. Try extracting from new_output (which is test_output from frontend = merged result)
        if isinstance(new_output, dict):
             for key in writes:
                if key in new_output:
                     # Validate it's not just an empty placeholder if possible, but trust the save
                     val = new_output[key]
                     # If it's a list and not empty, or dict and not empty, update
                     if val or val == 0 or val is False: 
                         globals_schema[key] = val
                         
        # 2. Fallback to execution_result only if new_output didn't have it
        exec_result = body.get("execution_result")
        if exec_result and isinstance(exec_result, dict):
             result_data = exec_result.get("result", exec_result) # Handle {status:..., result:...} or direct
             
             for key in writes:
                 if key not in globals_schema or not globals_schema[key]: # Only if missing/empty
                     if isinstance(result_data, dict) and key in result_data:
                         globals_schema[key] = result_data[key]
        
        G.graph["globals_schema"] = globals_schema
        
        # 5. Update iterations array with execution_result if provided
        execution_result = body.get("execution_result")
        if execution_result:
            # Check if iterations exist, if not create a default one
            if not node_data.get("iterations"):
                 node_data["iterations"] = []
            
            iterations = node_data["iterations"]
            
            if iterations:
                # Update the last iteration with the new execution result
                iterations[-1]["execution_result"] = execution_result
            else:
                # Create a pseudo-iteration if none exist (e.g. single-shot agents)
                iterations.append({
                    "iteration": 1, 
                    "output": new_output,
                    "execution_result": execution_result
                })

        # 5.5. Cascading Invalidation: Mark downstream nodes as 'stale'
        # This gives visual feedback (muted opacity) in the frontend that these nodes need re-running
        try:
             descendants = nx.descendants(G, node_id)
             for desc_id in descendants:
                 if desc_id in G.nodes:
                     # Only mark as stale if they were previously completed or failed
                     # If they are 'pending', they stay pending.
                     current_status = G.nodes[desc_id].get('status')
                     if current_status in ['completed', 'failed', 'running']:
                        G.nodes[desc_id]['status'] = 'stale'
        except Exception as e:
             logger.warning("Failed to invalidate downstream nodes: %s", e)
        
        # 6. Save back to file
        # Use edges="edges" to match our expected format (not default "link")
        graph_data = nx.node_link_data(G, edges="edges")
        with open(found_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, default=str, ensure_ascii=False)
        
        # 7. AUTO-SAVE TO NOTES (for FormatterAgent "Run Again" saves)
        agent_type = G.nodes[node_id].get("agent", "")
        if agent_type == "FormatterAgent" and isinstance(new_output, dict):
            try:
                import re
                notes_dir = PROJECT_ROOT / "data" / "Notes" / "Arcturus"
                notes_dir.mkdir(parents=True, exist_ok=True)
                
                def sanitize_filename(title):
                    title = re.sub(r'[\\/*?:"<>|#]', "", title)
                    title = title.replace("\n", " ").strip()
                    return title[:60].strip()

                def extract_title(content):
                    match = re.search(r'^#+\s+(.+)$', content, re.MULTILINE)
                    if match:
                        return match.group(1).strip()
                    lines = [l.strip() for l in content.split('\n') if l.strip()]
                    return lines[0] if lines else "Untitled Report"
                
                markdown = new_output.get("markdown_report")
                if not markdown:
                    for k, v in new_output.items():
                        if k.startswith("formatted_report") and isinstance(v, str):
                            markdown = v
                            break
                
                if markdown and len(markdown) > 100:
                    title = extract_title(markdown)
                    filename = sanitize_filename(title) + ".md"
                    target_path = notes_dir / filename
                    
                    # Write or overwrite (Run Again means user explicitly wants new version)
                    with open(target_path, 'w', encoding='utf-8') as f:
                        f.write(markdown)
                    logger.info("Auto-Saved (Run Again) to Notes: %s", filename)
                    
            except Exception as e:
                logger.warning("Failed to auto-save to Notes: %s", e)
        
        return {
            "status": "success",
            "node_id": node_id,
            "message": "Test results saved to session"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to save agent test results: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
