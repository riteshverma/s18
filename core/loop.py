# flow.py – 100% NetworkX Graph-First (No agentSession)

import networkx as nx
import asyncio
import time
import ast
import json
from pathlib import Path
from memory.context import ExecutionContextManager
from agents.base_agent import AgentRunner
from core.utils import log_step, log_error
from core.event_bus import event_bus
from core.schemas.clinical import extract_request_payload_from_query, validate_cbc_payload
from core.model_manager import ModelManager
from config.settings_loader import get_timeout
from core.prometheus_metrics import (
    ORCHESTRATOR_RUNS_TOTAL,
    ORCHESTRATOR_RUN_LATENCY_MS,
    elapsed_ms,
    now_ms,
)
from integrations.policies.workflow_guards import (
    enforce_cbc_full_mode_minimum_plan,
    enforce_mental_health_plan_guard,
    filter_memory_context_for_cbc,
    filter_memory_context_for_mental_health,
    is_cbc_payload_query,
    is_fast_mode,
    is_mental_health_task_query,
)
from ui.visualizer import ExecutionVisualizer
from rich.live import Live
from rich.console import Console
from datetime import datetime


def sanitize_io_keys_list(keys):
    """Normalize reads/writes to string keys to avoid unhashable dict errors."""
    if keys is None:
        return []
    if not isinstance(keys, list):
        keys = [keys]
    out = []
    for item in keys:
        if isinstance(item, str):
            key = item.strip()
        elif isinstance(item, dict):
            if len(item) == 1:
                _, v = next(iter(item.items()))
                key = v.strip() if isinstance(v, str) and v.strip() else json.dumps(item, sort_keys=True, default=str)
            else:
                key = json.dumps(item, sort_keys=True, default=str)
        else:
            key = str(item).strip()
        if key and key not in out:
            out.append(key)
    return out


# ===== EXPONENTIAL BACKOFF FOR TRANSIENT FAILURES =====

async def retry_with_backoff(
    async_func, 
    max_retries: int = 3, 
    base_delay: float = 1.0,
    retryable_errors: tuple = None
):
    """
    Retry an async function with exponential backoff.
    
    Args:
        async_func: Async callable to execute
        max_retries: Maximum retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        retryable_errors: Tuple of exception types to retry on
        
    Returns:
        Result of async_func on success
        
    Raises:
        Last exception if all retries exhausted
    """
    if retryable_errors is None:
        retryable_errors = (
            asyncio.TimeoutError,
            ConnectionError,
            TimeoutError,
        )
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await async_func()
        except retryable_errors as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
                log_step(f"Transient error: {type(e).__name__}. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})", symbol="🔄")
                await asyncio.sleep(delay)
            else:
                log_error(f"All {max_retries} retry attempts failed: {e}")
        except Exception as e:
            # Non-retryable error, raise immediately
            raise
    
    raise last_exception


class AgentLoop4:
    def __init__(self, multi_mcp, strategy="conservative"):
        self.multi_mcp = multi_mcp
        self.strategy = strategy
        self.agent_runner = AgentRunner(multi_mcp)
        self.context = None  # Reference for external stopping
        self._tasks = set()  # Track active async tasks for immediate cancellation

    def stop(self):
        """Request execution stop"""
        if self.context:
            self.context.stop()
        # Immediately cancel all tracked tasks
        for t in list(self._tasks):
            if not t.done():
                t.cancel()

    async def _track_task(self, coro_or_future):
        """Track an async task or future so it can be cancelled immediately on stop()"""
        if asyncio.iscoroutine(coro_or_future):
            task = asyncio.create_task(coro_or_future)
        else:
            # It's already a task or future (like from asyncio.gather)
            task = coro_or_future
            
        self._tasks.add(task)
        try:
            return await task
        except asyncio.CancelledError:
            raise
        finally:
            self._tasks.discard(task)

    def _apply_memory_context(self, context, memory_context):
        context.memory_context = memory_context
        context.plan_graph.graph["memory_context"] = memory_context
        if memory_context is not None:
            context.plan_graph.graph.setdefault("globals_schema", {})["memory_context"] = memory_context

    def _prepare_resumed_context(self, context, memory_context=None):
        context.set_multi_mcp(self.multi_mcp)
        context.api_mode = bool(context.plan_graph.graph.get("api_mode", True))
        context.user_input_event = asyncio.Event()
        context.user_input_value = None
        context._live_display = None
        restored_memory = memory_context
        if restored_memory is None:
            restored_memory = context.plan_graph.graph.get("memory_context")
        self._apply_memory_context(context, restored_memory)
        context.stop_requested = False

        for _, node_data in context.plan_graph.nodes(data=True):
            if node_data.get("status") == "running":
                node_data["status"] = "pending"
                node_data["error"] = None
                node_data["end_time"] = None

        context.plan_graph.graph["status"] = "running"
        context._save_session()
        return context

    async def resume(self, session_file, memory_context=None):
        context = ExecutionContextManager.load_session(Path(session_file))
        self.context = self._prepare_resumed_context(context, memory_context=memory_context)

        query_node = self.context.plan_graph.nodes["Query"] if "Query" in self.context.plan_graph else {}
        has_expanded_plan = len(self.context.plan_graph.nodes) > 2
        if not has_expanded_plan or query_node.get("status") != "completed":
            return await self.run(
                query=self.context.plan_graph.graph.get("original_query", ""),
                file_manifest=self.context.plan_graph.graph.get("file_manifest", []),
                globals_schema=self.context.plan_graph.graph.get("globals_schema", {}),
                uploaded_files=[],
                session_id=self.context.plan_graph.graph.get("session_id"),
                memory_context=self.context.memory_context,
                existing_context=self.context,
                storage_namespace=self.context.plan_graph.graph.get("storage_namespace", "shared"),
            )

        await self._track_task(self._execute_dag(self.context))
        return self.context

    async def run(
        self,
        query,
        file_manifest,
        globals_schema,
        uploaded_files,
        session_id=None,
        memory_context=None,
        existing_context=None,
        storage_namespace: str = "shared",
    ):
        run_start_ms = now_ms()
        final_status = "failed"
        try:
            if existing_context is None:
                # 🟢 PHASE 0: BOOTSTRAP CONTEXT (Immediate VS Code feedback)
                # We create a temporary graph with just a "Query" node (running Planner) so the UI sees meaningful start
                bootstrap_graph = {
                    "nodes": [
                        {
                            "id": "Query", 
                            "description": "Formulate execution plan", 
                            "agent": "PlannerAgent", 
                            "status": "running",
                            "reads": ["original_query"],
                            "writes": ["plan_graph"]
                        }
                    ],
                    "edges": [
                        {"source": "ROOT", "target": "Query"}
                    ]
                }

                # Create Context & Save Immediately
                self.context = ExecutionContextManager(
                    bootstrap_graph,
                    session_id=session_id,
                    original_query=query,
                    file_manifest=file_manifest,
                    storage_namespace=storage_namespace,
                )
                log_step("✅ Session initialized with Query processing", symbol="🌱")
            else:
                self.context = existing_context
                self.context.set_multi_mcp(self.multi_mcp)
                self.context.plan_graph.graph["file_manifest"] = file_manifest
                self.context.plan_graph.graph["storage_namespace"] = storage_namespace
                query_node = self.context.plan_graph.nodes["Query"] if "Query" in self.context.plan_graph else None
                if query_node and query_node.get("status") != "completed":
                    query_node["status"] = "running"
                    query_node["error"] = None
                    query_node["end_time"] = None
                self.context.plan_graph.graph["status"] = "running"
                log_step("✅ Resuming session from saved bootstrap state", symbol="🌱")

            self._apply_memory_context(self.context, memory_context)
            self.context.multi_mcp = self.multi_mcp
            seeded_query = self.context.plan_graph.graph['globals_schema'].get("original_query")
            self.context.plan_graph.graph['globals_schema'].update(globals_schema or {})
            merged_query = self.context.plan_graph.graph['globals_schema'].get("original_query")
            if (merged_query is None or merged_query == "") and seeded_query not in (None, ""):
                self.context.plan_graph.graph['globals_schema']['original_query'] = seeded_query
            await self.context.save_session_async()
        except Exception as e:
            print(f"❌ ERROR initializing context: {e}")
            raise

        # Phase 1: File Profiling (if files exist)
        file_profiles = self.context.plan_graph.graph.get("file_profiles", {}) or {}
        if uploaded_files and not file_profiles:
            # Wrap with retry for transient failures
            async def run_distiller():
                return await self.agent_runner.run_agent(
                    "DistillerAgent",
                    {
                        "task": "profile_files",
                        "files": uploaded_files,
                        "instruction": "Profile and summarize each file's structure, columns, content type",
                        "writes": ["file_profiles"]
                    }
                )
            file_result = await self._track_task(retry_with_backoff(run_distiller))
            if file_result["success"]:
                file_profiles = file_result["output"]
                self.context.set_file_profiles(file_profiles)

        # Phase 2: Planning and Execution Loop
        try:
            while True:
                if self.context.stop_requested:
                    break

                # Note: The "Query" node is already 'running' in our bootstrap context
                # Validate CBC payloads for both fast AND full modes
                if self._is_cbc_payload_query(query):
                    payload = extract_request_payload_from_query(query)
                    validated, cbc_err = validate_cbc_payload(payload)
                    if cbc_err is not None:
                        msg = f"CBC payload invalid: {cbc_err}"
                        self.context.mark_failed("Query", msg)
                        raise RuntimeError(msg)

                planner_memory_context = self._filter_memory_context_for_cbc(query, memory_context)
                planner_memory_context = self._filter_memory_context_for_mental_health(query, planner_memory_context)

                if self._is_cbc_payload_query(query) and self._is_fast_mode(query):
                    plan_result = {
                        "success": True,
                        "output": {
                            "plan_graph": {
                                "nodes": [
                                    {
                                        "id": "T001",
                                        "agent": "ThinkerAgent",
                                        "description": "Analyze CBC payload and return risk/confidence/flags.",
                                        "reads": ["original_query"],
                                        "writes": ["response"],
                                        "status": "pending",
                                    }
                                ],
                                "edges": [{"source": "Query", "target": "T001"}],
                            },
                            "next_step_id": "T001",
                            "interpretation_confidence": 1.0,
                            "ambiguity_notes": [],
                        },
                    }
                    log_step("⚡ Skipped Planner for CBC payload query (fast mode)", symbol="⚡")
                else:
                    async def run_planner():
                        return await self.agent_runner.run_agent(
                            "PlannerAgent",
                            {
                                "original_query": query,
                                "planning_strategy": self.strategy,
                                "globals_schema": self.context.plan_graph.graph.get("globals_schema", {}),
                                "file_manifest": file_manifest,
                                "file_profiles": file_profiles,
                                "memory_context": planner_memory_context
                            }
                        )
                    plan_result = await self._track_task(retry_with_backoff(run_planner))

                if self.context.stop_requested:
                    break

                if not plan_result["success"]:
                    self.context.mark_failed("Query", plan_result['error'])
                    raise RuntimeError(f"Planning failed: {plan_result['error']}")

                # Normalize planner output: accept "plan" or top-level nodes/edges as plan_graph
                out = plan_result["output"]
                if "plan_graph" not in out:
                    if "plan" in out and isinstance(out["plan"], dict):
                        p = out["plan"]
                        out["plan_graph"] = p.get("plan_graph", p) if isinstance(p.get("plan_graph"), dict) else p
                    elif "nodes" in out:
                        out["plan_graph"] = {
                            "nodes": out["nodes"],
                            "edges": out.get("edges", out.get("links", []))
                        }
                    else:
                        # Hard fallback so planner output contract is always satisfied.
                        out["plan_graph"] = {"nodes": [], "edges": []}
                pg = out["plan_graph"]
                if not isinstance(pg.get("nodes"), list):
                    pg["nodes"] = list(pg["nodes"]) if pg.get("nodes") else []
                if "edges" not in pg and "links" in pg:
                    pg["edges"] = pg["links"]
                pg.setdefault("edges", [])
                # Normalize edge sources so downstream callers never see planner aliases.
                normalized_edges = []
                for edge in pg["edges"]:
                    if not isinstance(edge, dict):
                        continue
                    source = edge.get("source")
                    target = edge.get("target")
                    if source in {"ROOT", "root", "original_query", "query", "user_query"}:
                        source = "Query"
                    if not source or not target:
                        continue
                    normalized_edges.append({"source": source, "target": target})
                pg["edges"] = normalized_edges
                if not out.get("next_step_id") and pg["nodes"]:
                    first_node = pg["nodes"][0]
                    if isinstance(first_node, dict) and first_node.get("id"):
                        out["next_step_id"] = first_node["id"]

                # Fast-path for WISE CBC payloads (fast mode only):
                # avoid expensive multi-step plans that frequently exceed frontend timeout windows.
                if self._is_cbc_payload_query(query) and self._is_fast_mode(query):
                    out["plan_graph"] = {
                        "nodes": [
                            {
                                "id": "T001",
                                "agent": "ThinkerAgent",
                                "description": "Analyze CBC payload and return risk/confidence/flags.",
                                "reads": ["original_query"],
                                "writes": ["response"],
                                "status": "pending",
                            }
                        ],
                        "edges": [{"source": "Query", "target": "T001"}],
                    }
                    out["next_step_id"] = "T001"
                    log_step("⚡ Applied CBC fast-path plan (single ThinkerAgent step)", symbol="⚡")
                    pg = out["plan_graph"]
                # Fallback: if planner returned no steps, create a single ThinkerAgent step
                if not pg["nodes"]:
                    log_step("Planner returned no steps; using single-step fallback", symbol="🔄")
                    pg["nodes"] = [{
                        "id": "T001",
                        "agent": "ThinkerAgent",
                        "description": "Answer the user's query",
                        "reads": ["original_query"],
                        "writes": ["response"],
                        "status": "pending"
                    }]
                    pg["edges"] = [{"source": "Query", "target": "T001"}]
                    out["next_step_id"] = "T001"
                # For CBC full mode, enforce a deterministic minimum multi-step graph.
                self._enforce_cbc_full_mode_minimum_plan(query, out)
                # For mental-health tasks, block CBC/lab-miner routing leakage.
                mh_guard_applied = self._enforce_mental_health_plan_guard(query, out)
                if mh_guard_applied:
                    # Surface a compact marker in planner output so downstream adapters
                    # can expose this in API/debug flags without parsing full graphs.
                    out["mental_health_plan_guard_applied"] = True
                pg = out["plan_graph"]

                # ===== AUTO-CLARIFICATION CHECK =====
                AUTO_CLARYFY_THRESHOLD = 0.7
                confidence = plan_result["output"].get("interpretation_confidence", 1.0)
                ambiguity_notes = plan_result["output"].get("ambiguity_notes", [])
                
                # Check if Planner already added a ClarificationAgent (avoid duplicates)
                plan_nodes = plan_result["output"]["plan_graph"].get("nodes", [])
                has_clarification_agent = any(
                    n.get("agent") == "ClarificationAgent" for n in plan_nodes
                )
                
                if confidence < AUTO_CLARYFY_THRESHOLD and ambiguity_notes and not has_clarification_agent:
                    log_step(f"Low confidence ({confidence:.2f}), auto-triggering clarification", symbol="❓")
                    
                    # Get the first step ID from the plan
                    first_step = plan_result["output"].get("next_step_id", "T001")
                    clarification_write_key = "user_clarification_T000"
                    
                    # Create clarification node
                    clarification_node = {
                        "id": "T000_AutoClarify",
                        "agent": "ClarificationAgent",
                        "description": "Clarify ambiguous requirements before proceeding",
                        "agent_prompt": f"The system has identified ambiguities in the user's request. Please ask for clarification on: {'; '.join(ambiguity_notes)}",
                        "reads": [],
                        "writes": [clarification_write_key],
                        "status": "pending"
                    }
                    
                    # Insert clarification node at beginning
                    plan_result["output"]["plan_graph"]["nodes"].insert(0, clarification_node)
                    
                    # Add edge from ROOT to clarification, and clarification to first step
                    plan_result["output"]["plan_graph"]["edges"].insert(0, {
                        "source": "T000_AutoClarify",
                        "target": first_step
                    })
                    
                    # 🔧 CRITICAL FIX: Wire clarification output into the downstream node's reads
                    # Find the first_step node and add clarification_write_key to its reads
                    for node in plan_result["output"]["plan_graph"]["nodes"]:
                        if node.get("id") == first_step:
                            if "reads" not in node:
                                node["reads"] = []
                            if clarification_write_key not in node["reads"]:
                                node["reads"].append(clarification_write_key)
                                log_step(f"Wired {clarification_write_key} into {first_step}'s reads", symbol="🔗")
                            break
                    
                    # Update next_step_id to start with clarification
                    plan_result["output"]["next_step_id"] = "T000_AutoClarify"
                    
                    log_step(f"Injected ClarificationAgent before {first_step}", symbol="➕")
                elif has_clarification_agent:
                    log_step(f"Planner already added ClarificationAgent, skipping auto-injection", symbol="ℹ️")
                
                # ✅ Mark Query/Planner as Done
                self.context.plan_graph.nodes["Query"]["output"] = plan_result["output"]
                self.context.plan_graph.nodes["Query"]["status"] = "completed"
                self.context.plan_graph.nodes["Query"]["end_time"] = datetime.utcnow().isoformat()
                
                # 🟢 PHASE 3: EXPAND GRAPH
                # Merge the new plan into our existing context
                new_plan_graph = plan_result["output"]["plan_graph"]
                self._merge_plan_into_context(new_plan_graph)

                try:
                    # Phase 4: Execute DAG
                    await self._track_task(self._execute_dag(self.context))

                    if self.context.stop_requested:
                        break

                    # Phase 5: Check for Adaptive Re-Planning (Dead End Discovery)
                    if self._should_replan():
                        log_step("♻️ Adaptive Re-planning: Clarification resolved, formulating next steps...", symbol="🔄")
                        # Reactivate Query node for UI
                        self.context.plan_graph.nodes["Query"]["status"] = "running"
                        await self.context.save_session_async()
                        continue
                    else:
                        # No more work or re-planning needed
                        final_status = "success"
                        return self.context

                except (Exception, asyncio.CancelledError) as e:
                    if isinstance(e, asyncio.CancelledError) or self.context.stop_requested:
                        log_step("🛑 Execution interrupted/stopped.", symbol="🛑")
                        break
                    print(f"❌ ERROR during execution: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        except (Exception, asyncio.CancelledError) as e:
            if self.context:
                # Mark ANY running/pending node as stopped/failed to stop spinners
                final_status = "stopped" if (self.context.stop_requested or isinstance(e, asyncio.CancelledError)) else "failed"
                for node_id in self.context.plan_graph.nodes:
                    if self.context.plan_graph.nodes[node_id].get("status") in ["running", "pending"]:
                        self.context.plan_graph.nodes[node_id]["status"] = final_status
                        if final_status == "failed":
                             self.context.plan_graph.nodes[node_id]["error"] = str(e)
                
                self.context.plan_graph.graph['status'] = final_status
                if final_status == "failed":
                    self.context.plan_graph.graph['error'] = str(e)
                await self.context.save_session_async()
            if not isinstance(e, asyncio.CancelledError) and not self.context.stop_requested:
                raise e
            final_status = "stopped"
            return self.context
        finally:
            integration_id = "default"
            workflow_id = "generic"
            contract_version = "v1"
            try:
                gs = (self.context.plan_graph.graph or {}).get("globals_schema", {}) if self.context else {}
                meta = gs.get("_integration_meta", {}) if isinstance(gs, dict) else {}
                if isinstance(meta, dict):
                    integration_id = str(meta.get("integration_id") or integration_id)
                    workflow_id = str(meta.get("workflow_id") or workflow_id)
                    contract_version = str(meta.get("contract_version") or contract_version)
            except Exception:
                pass
            ORCHESTRATOR_RUNS_TOTAL.labels(
                status=final_status,
                integration_id=integration_id,
                workflow_id=workflow_id,
                contract_version=contract_version,
            ).inc()
            ORCHESTRATOR_RUN_LATENCY_MS.observe(elapsed_ms(run_start_ms))

    def _is_cbc_payload_query(self, query: str) -> bool:
        return is_cbc_payload_query(query)

    def _is_fast_mode(self, query: str) -> bool:
        return is_fast_mode(query)

    def _is_mental_health_task_query(self, query: str) -> bool:
        return is_mental_health_task_query(query)

    def _filter_memory_context_for_cbc(self, query: str, memory_context):
        return filter_memory_context_for_cbc(query, memory_context)

    def _filter_memory_context_for_mental_health(self, query: str, memory_context):
        return filter_memory_context_for_mental_health(query, memory_context)

    def _enforce_cbc_full_mode_minimum_plan(self, query: str, out: dict):
        enforce_cbc_full_mode_minimum_plan(query, out)

    def _enforce_mental_health_plan_guard(self, query: str, out: dict) -> bool:
        return enforce_mental_health_plan_guard(query, out)

    def _should_replan(self):
        """
        Check if the graph needs expansion (re-planning).
        Conditions:
        1. All current nodes are finished (completed/skipped).
        2. At least one ClarificationAgent recently completed.
        3. That ClarificationAgent was a 'leaf' (had no successors in the current graph).
        """
        # If any node is still pending/running, we aren't at a dead end yet
        if not self.context.all_done():
            return False
            
        has_new_leaf_clarification = False
        for node_id, node_data in self.context.plan_graph.nodes(data=True):
            if node_data.get("agent") == "ClarificationAgent" and node_data.get("status") == "completed":
                # Check if it was a leaf node (no arrows coming out)
                if not list(self.context.plan_graph.successors(node_id)):
                    has_new_leaf_clarification = True
                    break
        
        return has_new_leaf_clarification

    def _repair_cbc_plan_dependencies(self, new_nodes, new_edges):
        """
        Repair common CBC planner mistakes:
        - reasoning/summarization node reads only `original_query`
        - miner node exists but no dependency edge is present
        This avoids parallel execution where summary is generated before lab retrieval.
        """
        if not isinstance(new_nodes, list):
            return

        miner_nodes = [
            n for n in new_nodes
            if isinstance(n, dict) and n.get("agent") == "EHRDataMinerAgent"
        ]
        if not miner_nodes:
            return

        miner_write_keys = []
        miner_node_ids = []
        for miner in miner_nodes:
            miner_node_ids.append(miner.get("id"))
            miner_write_keys.extend(sanitize_io_keys_list(miner.get("writes", [])))

        dependent_agents = {
            "SummarizationAgent",
            "SummarizerAgent",
            "ClinicalReasoningAgent",
            "ThinkerAgent",
            "FormatterAgent",
            "ActionAgent",
            "ResponseAgent",
        }

        existing_edges = {
            (
                (edge.get("source") or edge.get("from")),
                (edge.get("target") or edge.get("to")),
            )
            for edge in new_edges
            if isinstance(edge, dict)
        }

        for node in new_nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            agent = node.get("agent")
            if not node_id or agent not in dependent_agents:
                continue
            if node_id in miner_node_ids:
                continue

            reads = sanitize_io_keys_list(node.get("reads", []))
            if reads == ["original_query"]:
                for write_key in miner_write_keys:
                    if write_key not in reads:
                        reads.append(write_key)
                node["reads"] = reads

            for miner_id in miner_node_ids:
                if (miner_id, node_id) not in existing_edges:
                    new_edges.append({"source": miner_id, "target": node_id})
                    existing_edges.add((miner_id, node_id))
                    log_step(
                        f"🔧 Repaired CBC plan dependency {miner_id} -> {node_id}",
                        symbol="🔧",
                    )

    def _merge_plan_into_context(self, new_plan_graph):
        """Merge the planned nodes into the existing bootstrap context"""
        new_nodes = new_plan_graph.get("nodes", [])
        new_edges = new_plan_graph.get("edges", [])
        if self.context and self._is_cbc_payload_query(self.context.plan_graph.graph.get("original_query", "")):
            self._repair_cbc_plan_dependencies(new_nodes, new_edges)
        known_new_node_ids = {n.get("id") for n in new_nodes if isinstance(n, dict) and n.get("id")}
        
        # Track which new nodes have incoming edges to detect orphans
        nodes_with_incoming_edges = set()

        # Sanitize node IO fields in-place so all later merge wiring sees safe keys.
        for node in new_nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id", "unknown")
            raw_reads = node.get("reads", [])
            raw_writes = node.get("writes", [])
            node["reads"] = sanitize_io_keys_list(raw_reads)
            node["writes"] = sanitize_io_keys_list(raw_writes)
            if node["reads"] != raw_reads:
                log_step(f"🧹 Sanitized reads for {node_id}: {node['reads']}", symbol="🧹")
            if node["writes"] != raw_writes:
                log_step(f"🧹 Sanitized writes for {node_id}: {node['writes']}", symbol="🧹")

        # Add new nodes
        for node in new_nodes:
            # Prepare node data with defaults
            node_data = node.copy()
            # Set defaults if not present in the plan
            defaults = {
                'status': 'pending',
                'output': None,
                'error': None,
                'cost': 0.0,
                'start_time': None,
                'end_time': None,
                'execution_time': 0.0
            }
            for k, v in defaults.items():
                node_data.setdefault(k, v)
                
            # Avoid overwriting already completed nodes if they somehow appear in the new plan
            if node["id"] in self.context.plan_graph:
                 existing_status = self.context.plan_graph.nodes[node["id"]].get("status")
                 if existing_status == "completed":
                      continue

            self.context.plan_graph.add_node(node["id"], **node_data)
            
        # Add new edges, redirecting ROOT -> First Step to Query -> First Step
        for edge in new_edges:
            # Robustly handle different edge formats or missing keys
            source = edge.get("source") or edge.get("from")
            target = edge.get("target") or edge.get("to")
            
            if not source or not target:
                log_step(f"⚠️ Skipping malformed edge: {edge}", symbol="⚠️")
                continue
            
            # Redirect common planner aliases to the real planner node id.
            if source in {"ROOT", "root", "original_query", "query", "user_query"}:
                source = "Query"
            # Guard against dangling/unknown edge sources by attaching to Query.
            elif source not in known_new_node_ids and source != "Query":
                log_step(f"🔧 Rewriting unknown edge source '{source}' -> Query", symbol="🔧")
                source = "Query"

            self.context.plan_graph.add_edge(source, target)
            nodes_with_incoming_edges.add(target)
        
        # Build reverse index of produced keys -> producer nodes for dependency inference.
        produced_key_to_nodes = {}
        for node in new_nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if not node_id:
                continue
            for write_key in sanitize_io_keys_list(node.get("writes", [])):
                produced_key_to_nodes.setdefault(write_key, set()).add(node_id)

        # 🛡️ AUTO-CONNECT:
        # If a new node has no incoming edges, infer missing edges from reads->writes first.
        # Only fall back to Query when there are no inferred producer dependencies.
        for node in new_nodes:
            node_id = node.get("id")
            if not node_id or node_id in nodes_with_incoming_edges:
                continue

            inferred_sources = set()
            for read_key in sanitize_io_keys_list(node.get("reads", [])):
                for source_node in produced_key_to_nodes.get(read_key, set()):
                    if source_node != node_id:
                        inferred_sources.add(source_node)

            if inferred_sources:
                for source_node in sorted(inferred_sources):
                    self.context.plan_graph.add_edge(source_node, node_id)
                    nodes_with_incoming_edges.add(node_id)
                    log_step(
                        f"🔗 Inferred missing edge {source_node} -> {node_id} from read dependency",
                        symbol="🔗",
                    )
                continue

            log_step(f"🔗 Auto-connected orphan node {node_id} to Query", symbol="🔗")
            self.context.plan_graph.add_edge("Query", node_id)
        
        # 🔧 SAFETY NET: Ensure ClarificationAgent outputs are wired to successor nodes
        # This fixes cases where Planner adds a ClarificationAgent but forgets to wire reads
        for node in new_nodes:
            if node.get("agent") == "ClarificationAgent":
                clarification_node_id = node["id"]
                clarification_writes = node.get("writes", [])
                
                if not clarification_writes:
                    continue
                    
                # Find all successor nodes (nodes that this ClarificationAgent points to)
                for edge in new_edges:
                    if edge.get("source") == clarification_node_id:
                        successor_id = edge.get("target")
                        if not successor_id:
                            continue
                        
                        # Find the successor node and ensure it reads from clarification
                        for succ_node in new_nodes:
                            if succ_node.get("id") == successor_id:
                                if "reads" not in succ_node:
                                    succ_node["reads"] = []
                                
                                for write_key in clarification_writes:
                                    if write_key not in succ_node["reads"]:
                                        succ_node["reads"].append(write_key)
                                        log_step(f"🔗 Auto-wired {write_key} into {successor_id}'s reads", symbol="🔗")
                                        
                                        # Also update the node in the graph if already added
                                        if successor_id in self.context.plan_graph:
                                            if "reads" not in self.context.plan_graph.nodes[successor_id]:
                                                self.context.plan_graph.nodes[successor_id]["reads"] = []
                                            if write_key not in self.context.plan_graph.nodes[successor_id]["reads"]:
                                                self.context.plan_graph.nodes[successor_id]["reads"].append(write_key)
                                break
        
        self.context._save_session()
        log_step("✅ Plan merged into execution context", symbol="🌳")

    async def _execute_dag(self, context):
        """Execute DAG with visualization - DEBUGGING MODE"""
        
        # Get plan_graph structure for visualization
        plan_graph = {
            "nodes": [
                {"id": node_id, **node_data} 
                for node_id, node_data in context.plan_graph.nodes(data=True)
            ],
            "links": [
                {"source": source, "target": target}
                for source, target in context.plan_graph.edges()
            ]
        }
        
        # Create visualizer
        visualizer = ExecutionVisualizer(plan_graph)
        console = Console()
        
        # 🔧 DEBUGGING MODE: No Live display, just regular prints
        max_iterations = 20
        iteration = 0
        
        # ===== COST THRESHOLD ENFORCEMENT =====
        from config.settings_loader import reload_settings
        settings = reload_settings()
        max_cost = settings.get("agent", {}).get("max_cost_per_run", 0.50)
        warn_cost = settings.get("agent", {}).get("warn_at_cost", 0.25)
        cost_warning_shown = False

        while not context.all_done():
            if context.stop_requested:
                console.print("[yellow]🛑 Aborting execution: Cleaning up nodes...[/yellow]")
                # Cleanup: Mark any 'running' nodes as 'stopped' to prevent zombie spinners in UI
                for n_id in context.plan_graph.nodes:
                    if context.plan_graph.nodes[n_id].get("status") == "running":
                        context.plan_graph.nodes[n_id]["status"] = "stopped"
                await context.save_session_async()
                break
            
            # Get ready nodes
            ready_steps = context.get_ready_steps()
            
            # 🛡️ DEFENSIVE: Filter out steps that are not pending (prevents loops)
            ready_steps = [s for s in ready_steps if context.plan_graph.nodes[s]["status"] == "pending"]
            
            if not ready_steps:
                # Check for running steps or waiting steps
                running_or_waiting = any(
                    context.plan_graph.nodes[n]['status'] in ['running', 'waiting_input']
                    for n in context.plan_graph.nodes
                )
                
                if not running_or_waiting:
                    # If no ready steps, and nothing is running/waiting, and we aren't "all_done" (maybe orphans?)
                    # Check if everything is completed or skipped
                    is_complete = all(
                        context.plan_graph.nodes[n]['status'] in ['completed', 'skipped', 'cost_exceeded']
                        for n in context.plan_graph.nodes
                        if n != "ROOT"
                    )
                    if is_complete:
                        break
                
                # Wait for progress
                await asyncio.sleep(0.5)
                continue

            # Show current state (only when we found work to do)
            try:
                console.print(visualizer.get_layout())
            except Exception as e:
                console.print(f"[dim]Note: Could not refresh terminal UI: {e}[/dim]")

            # Mark running
            for step_id in ready_steps:
                visualizer.mark_running(step_id)
                context.mark_running(step_id)
            
            # ✅ EXECUTE AGENTS FOR REAL
            tasks = []
            for step_id in ready_steps:
                # Log step start with description
                step_data = context.get_step_data(step_id)
                desc = step_data.get("agent_prompt", step_data.get("description", "No description"))[:60]
                log_step(f"🔄 Starting {step_id} ({step_data['agent']}): {desc}...", symbol="🚀")
                
                visualizer.mark_running(step_id)
                context.mark_running(step_id)
                tasks.append(self._track_task(self._execute_step(step_id, context)))

            results = await self._track_task(asyncio.gather(*tasks, return_exceptions=True))

            # Step-level retry configuration
            MAX_STEP_RETRIES = 2
            
            # Process results (with step-level retry)
            for step_id, result in zip(ready_steps, results):
                step_data = context.get_step_data(step_id)
                retry_count = step_data.get('_retry_count', 0)
                
                # ✅ HANDLE AWAITING INPUT
                if isinstance(result, dict) and result.get("status") == "waiting_input":
                     visualizer.mark_waiting(step_id) 
                     context.plan_graph.nodes[step_id]["status"] = "waiting_input"
                     # Preserve partial output
                     if "output" in result:
                         context.plan_graph.nodes[step_id]["output"] = result["output"]
                     await context.save_session_async()
                     log_step(f"⏳ {step_id}: Waiting for user input...", symbol="⏳")
                     continue
                
                if isinstance(result, Exception):
                    # Check if we should retry this step
                    if retry_count < MAX_STEP_RETRIES:
                        step_data['_retry_count'] = retry_count + 1
                        context.plan_graph.nodes[step_id]['status'] = 'pending'  # Reset to pending for retry
                        log_step(f"🔄 Retrying {step_id} (attempt {retry_count + 1}/{MAX_STEP_RETRIES}): {str(result)}", symbol="🔄")
                    else:
                        visualizer.mark_failed(step_id, result)
                        context.mark_failed(step_id, str(result))
                        log_error(f"❌ Failed {step_id} after {MAX_STEP_RETRIES} retries: {str(result)}")
                elif result["success"]:
                    visualizer.mark_completed(step_id)
                    await context.mark_done(step_id, result["output"])
                    log_step(f"✅ Completed {step_id} ({step_data['agent']})", symbol="✅")
                else:
                    # Agent returned failure - also retry
                    if retry_count < MAX_STEP_RETRIES:
                        step_data['_retry_count'] = retry_count + 1
                        context.plan_graph.nodes[step_id]['status'] = 'pending'
                        log_step(f"🔄 Retrying {step_id} (attempt {retry_count + 1}/{MAX_STEP_RETRIES}): {result['error']}", symbol="🔄")
                    else:
                        visualizer.mark_failed(step_id, result["error"])
                        context.mark_failed(step_id, result["error"])
                        log_error(f"❌ Failed {step_id} after {MAX_STEP_RETRIES} retries: {result['error']}")

            # ===== COST THRESHOLD CHECK =====
            accumulated_cost = sum(
                context.plan_graph.nodes[n].get('cost', 0) 
                for n in context.plan_graph.nodes
                if context.plan_graph.nodes[n].get('status') == 'completed'
            )
            
            # Warning threshold
            if not cost_warning_shown and accumulated_cost >= warn_cost:
                log_step(f"⚠️ Cost Warning: ${accumulated_cost:.4f} (threshold: ${warn_cost:.2f})", symbol="💰")
                cost_warning_shown = True
            
            # Hard stop threshold
            if accumulated_cost >= max_cost:
                log_error(f"🛑 Cost Exceeded: ${accumulated_cost:.4f} > ${max_cost:.2f}")
                context.plan_graph.graph['status'] = 'cost_exceeded'
                context.plan_graph.graph['final_cost'] = accumulated_cost
                break

        # Final state
        console.print(visualizer.get_layout())
        
        # Determine and save final status
        if context.stop_requested:
             context.plan_graph.graph['status'] = 'stopped'
        elif any(context.plan_graph.nodes[n]['status'] == 'failed' for n in context.plan_graph.nodes):
             context.plan_graph.graph['status'] = 'failed'
        elif context.all_done():
             context.plan_graph.graph['status'] = 'completed'
        else:
             # Max iterations or stalled
             context.plan_graph.graph['status'] = 'failed'
        
        context._auto_save()
        
        if context.all_done():
            console.print("🎉 All tasks completed!")

    async def _execute_step(self, step_id, context):
        """Execute a single step with call_self support"""
        # 📡 EMIT EVENT
        await event_bus.publish("step_start", "AgentLoop4", {"step_id": step_id})
        step_data = context.get_step_data(step_id)
        # Sanitize reads/writes to string keys (handles planner dicts and session load; prevents unhashable type: 'dict')
        reads = sanitize_io_keys_list(step_data.get("reads", []))
        writes = sanitize_io_keys_list(step_data.get("writes", []))
        context.plan_graph.nodes[step_id]["reads"] = reads
        context.plan_graph.nodes[step_id]["writes"] = writes
        step_data["reads"] = reads
        step_data["writes"] = writes

        agent_type = step_data["agent"]
        # Normalize common planner aliases to configured agent names
        agent_aliases = {
            "SummarizationAgent": "SummarizerAgent",
            "SummaryAgent": "SummarizerAgent",
            "ResearchAgent": "RetrieverAgent",
            "RAG": "RetrieverAgent",
            "RagAgent": "RetrieverAgent",
            "ResponseAgent": "FormatterAgent",
        }
        agent_type = agent_aliases.get(agent_type, agent_type)
        
        # Get inputs from NetworkX graph
        inputs = context.get_inputs(step_data.get("reads", []))
        
        # 🔧 HELPER FUNCTION: Build agent input (consistent for both iterations)
        def build_agent_input(instruction=None, previous_output=None, iteration_context=None):
            # Base payload for all agents
            payload = {
                "step_id": step_id,
                "agent_prompt": instruction or step_data.get("agent_prompt", step_data["description"]),
                "reads": step_data.get("reads", []),
                "writes": step_data.get("writes", []),
                "inputs": inputs,
                "original_query": context.plan_graph.graph['original_query'],
                "session_context": {
                    "session_id": context.plan_graph.graph['session_id'],
                    "created_at": context.plan_graph.graph['created_at'],
                    "file_manifest": context.plan_graph.graph['file_manifest'],
                    "memory_context": getattr(context, 'memory_context', None) # 🧠 Universal Injection
                },
                **({"previous_output": previous_output} if previous_output else {}),
                **({"iteration_context": iteration_context} if iteration_context else {})
            }
            
            # Formatter-specific additions
            if agent_type == "FormatterAgent":
                payload["all_globals_schema"] = context.plan_graph.graph['globals_schema'].copy()
                
            return payload

        # Execute with ReAct Loop (Max 15 turns)
        max_turns = 15
        current_input = build_agent_input()
        iterations_data = []
        
        for turn in range(1, max_turns + 1):
            log_step(f"🔄 {agent_type} Iteration {turn}/{max_turns}", symbol="🔄")
            
            # Run Agent (with retry for transient failures like rate limits)
            # Per-step timeout: 1.5x Ollama timeout so one slow LLM call + overhead can complete
            step_timeout = int(1.5 * get_timeout())
            async def run_agent_step():
                return await self.agent_runner.run_agent(agent_type, current_input)

            try:
                result = await retry_with_backoff(
                    lambda: asyncio.wait_for(run_agent_step(), timeout=step_timeout)
                )
            except Exception as e:
                # All retries exhausted, return failure
                return {"success": False, "error": f"Agent failed after retries: {str(e)}"}
            
            if not result["success"]:
                return result
            
            output = result["output"]

            # Safety net: ensure RetrieverAgent always gets concrete document snippets.
            # If the model returns plain text without calling tools, we do one direct RAG fetch.
            if (
                agent_type == "RetrieverAgent"
                and not output.get("call_tool")
                and "retrieved_documents" not in output
            ):
                try:
                    rag_result = await self.multi_mcp.call_tool(
                        "rag",
                        "search_stored_documents_rag",
                        {"query": context.plan_graph.graph["original_query"]},
                    )
                    retrieved_docs = []
                    if hasattr(rag_result, "content") and isinstance(rag_result.content, list):
                        for item in rag_result.content:
                            if not hasattr(item, "text"):
                                continue
                            raw_text = item.text or ""
                            try:
                                parsed = ast.literal_eval(raw_text)
                                if isinstance(parsed, list):
                                    retrieved_docs.extend([str(x) for x in parsed])
                                elif parsed:
                                    retrieved_docs.append(str(parsed))
                            except Exception:
                                if raw_text.strip():
                                    retrieved_docs.append(raw_text.strip())

                    output["retrieved_documents"] = retrieved_docs
                    if "response" not in output:
                        output["response"] = (
                            f"Retrieved {len(retrieved_docs)} relevant snippets."
                            if retrieved_docs
                            else "No relevant documents found."
                        )
                    log_step(
                        f"Retriever fallback injected {len(retrieved_docs)} snippets",
                        symbol="📚",
                    )
                except Exception as e:
                    output.setdefault("retrieved_documents", [])
                    output.setdefault("response", "No relevant documents found.")
                    log_error(f"Retriever fallback search failed: {e}")
            
            # ✅ CHECK FOR CLARIFICATION REQUEST (HALT)
            if output.get("clarificationMessage"):
                 return {
                    "success": True, 
                    "status": "waiting_input", 
                    "output": output
                 }

            iterations_data.append({"iteration": turn, "output": output})
            
            # ✅ IMMEDIATE STOP CHECK (Between turns)
            if context.stop_requested:
                log_step(f"🛑 {agent_type}: Stop requested, aborting iteration {turn}", symbol="🛑")
                return {"success": False, "error": "Stop requested"}

            # Update step data with iterations so far
            step_data = context.get_step_data(step_id)
            step_data['iterations'] = iterations_data
            
            # 1. Check for 'call_tool' (ReAct)
            if output.get("call_tool"):
                tool_call = output["call_tool"]
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
                
                log_step(f"🛠️ Executing Tool: {tool_name}", payload=tool_args, symbol="⚙️")
                
                try:
                    # Execute tool via MultiMCP
                    tool_result = await self.multi_mcp.route_tool_call(tool_name, tool_args)
                    
                    # Serialize result content
                    if isinstance(tool_result.content, list):
                        result_str = "\n".join([str(item.text) for item in tool_result.content if hasattr(item, "text")])
                    else:
                        result_str = str(tool_result.content)

                    # ✅ SAVE RESULT TO HISTORY
                    iterations_data[-1]["tool_result"] = result_str

                    # Log result (truncated)
                    log_step(f"✅ Tool Result", payload={"result_preview": result_str[:200] + "..."}, symbol="🔌")
                    
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
                    log_error(f"Tool Execution Failed: {e}")
                    # Feed error back to agent
                    current_input = build_agent_input(
                        instruction="The tool execution failed. Try a different approach or tool.",
                        previous_output=output,
                        iteration_context={"tool_result": f"Error: {str(e)}"}
                    )
                    continue

            # 2. Check for call_self (Legacy/Advanced recursion)
            elif output.get("call_self"):
                # Handle code execution if needed
                if context._has_executable_code(output):
                    execution_result = await context._auto_execute_code(step_id, output)
                    
                    # ✅ SAVE RESULT TO HISTORY
                    iterations_data[-1]["execution_result"] = execution_result

                    if execution_result.get("status") == "success":
                        execution_data = execution_result.get("result", {})
                        inputs = {**inputs, **execution_data}  # Update inputs for iteration 2
                
                # Prepare input for next iteration
                current_input = build_agent_input(
                    instruction=output.get("next_instruction", "Continue the task"),
                    previous_output=output,
                    iteration_context=output.get("iteration_context", {})
                )
                continue

            # 3. Success (No tool call, just output) - Execute code for final iteration
            else:
                # ✅ LAST-SECOND STOP CHECK
                if context.stop_requested:
                    return {"success": False, "error": "Stop requested"}
                    
                # Execute code if present and save to iterations_data (same as call_self path)
                if context._has_executable_code(output):
                    execution_result = await context._auto_execute_code(step_id, output)
                    iterations_data[-1]["execution_result"] = execution_result
                return result
        
        # If loop finishes without returning (max turns reached): Return PARTIAL SUCCESS to allow graph continuation
        log_error(f"Max iterations ({max_turns}) reached for {step_id}. Returning last output (incomplete).")
        last_output = iterations_data[-1]["output"] if iterations_data else {"error": "No output produced"}
        # Ensure it has a valid structure if possible, or just pass it through
        return {"success": True, "output": last_output}

    async def _handle_failures(self, context):
        """Handle failures via mid-session replanning"""
        # TODO: Implement mid-session replanning with PlannerAgent
        log_error("Mid-session replanning not yet implemented")
