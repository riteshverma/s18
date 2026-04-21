import time
from typing import Optional

from prometheus_client import Counter, Histogram


class _MirroredChild:
    def __init__(self, primary, legacy):
        self.primary = primary
        self.legacy = legacy

    def inc(self, amount: float = 1.0):
        self.primary.inc(amount)
        self.legacy.inc(amount)

    def observe(self, value: float):
        self.primary.observe(value)
        self.legacy.observe(value)


class _MirroredMetric:
    def __init__(self, primary, legacy):
        self.primary = primary
        self.legacy = legacy

    def labels(self, *args, **kwargs):
        return _MirroredChild(self.primary.labels(*args, **kwargs), self.legacy.labels(*args, **kwargs))

    def inc(self, amount: float = 1.0):
        self.primary.inc(amount)
        self.legacy.inc(amount)

    def observe(self, value: float):
        self.primary.observe(value)
        self.legacy.observe(value)


def _counter_pair(primary_name: str, legacy_name: str, description: str, labels):
    return _MirroredMetric(
        Counter(primary_name, description, labels),
        Counter(legacy_name, f"{description} (legacy wiseai namespace)", labels),
    )


def _histogram_pair(primary_name: str, legacy_name: str, description: str, labels=(), buckets=()):
    return _MirroredMetric(
        Histogram(primary_name, description, labels, buckets=buckets),
        Histogram(legacy_name, f"{description} (legacy wiseai namespace)", labels, buckets=buckets),
    )


API_REQUESTS_TOTAL = _counter_pair(
    "s18_api_requests_total",
    "wiseai_api_requests_total",
    "Total API requests received",
    ["method", "route", "status_class"],
)

API_REQUESTS_SUCCESS_TOTAL = _counter_pair(
    "s18_api_requests_success_total",
    "wiseai_api_requests_success_total",
    "Total successful API requests",
    ["method", "route"],
)

API_REQUEST_LATENCY_MS = _histogram_pair(
    "s18_api_request_latency_ms",
    "wiseai_api_request_latency_ms",
    "API request latency in milliseconds",
    labels=["method", "route"],
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)

ORCHESTRATOR_RUNS_TOTAL = _counter_pair(
    "s18_orchestrator_runs_total",
    "wiseai_orchestrator_runs_total",
    "Total orchestrator runs by final status",
    ["status", "integration_id", "workflow_id", "contract_version"],
)

ORCHESTRATOR_RUN_LATENCY_MS = _histogram_pair(
    "s18_orchestrator_run_latency_ms",
    "wiseai_orchestrator_run_latency_ms",
    "Orchestrator run latency in milliseconds",
    buckets=(50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 60000, 120000),
)

ORCHESTRATOR_PLANNER_LATENCY_MS = _histogram_pair(
    "s18_orchestrator_planner_latency_ms",
    "wiseai_orchestrator_planner_latency_ms",
    "PlannerAgent invocation latency in milliseconds",
    buckets=(100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 60000),
)

ORCHESTRATOR_STEP_LATENCY_MS = _histogram_pair(
    "s18_orchestrator_step_latency_ms",
    "wiseai_orchestrator_step_latency_ms",
    "Per DAG step execution latency in milliseconds",
    labels=["agent"],
    buckets=(50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 60000, 120000),
)

ORCHESTRATOR_DAG_ITERATIONS = _histogram_pair(
    "s18_orchestrator_dag_iterations",
    "wiseai_orchestrator_dag_iterations",
    "Inner DAG scheduler iterations per run",
    buckets=(1, 2, 5, 10, 15, 20, 30, 40, 60, 80),
)

ORCHESTRATOR_REPLAN_TOTAL = _counter_pair(
    "s18_orchestrator_replan_total",
    "wiseai_orchestrator_replan_total",
    "Re-planning events by reason",
    ["reason"],
)

ORCHESTRATOR_STEP_RETRIES_TOTAL = _counter_pair(
    "s18_orchestrator_step_retries_total",
    "wiseai_orchestrator_step_retries_total",
    "Step-level retries by agent",
    ["agent"],
)

ORCHESTRATOR_DAG_STALL_ITERATIONS = _counter_pair(
    "s18_orchestrator_dag_stall_iterations_total",
    "wiseai_orchestrator_dag_stall_iterations_total",
    "DAG iterations where no pending ready steps were found but work remained",
    [],
)

RAG_REQUESTS_TOTAL = _counter_pair(
    "s18_rag_requests_total",
    "wiseai_rag_requests_total",
    "Total RAG search requests",
    ["endpoint", "status"],
)

RAG_SEARCH_LATENCY_MS = _histogram_pair(
    "s18_rag_search_latency_ms",
    "wiseai_rag_search_latency_ms",
    "RAG search endpoint latency in milliseconds",
    labels=["endpoint"],
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)

RAG_RESULTS_COUNT = _histogram_pair(
    "s18_rag_results_count",
    "wiseai_rag_results_count",
    "Number of results returned by RAG endpoints",
    labels=["endpoint"],
    buckets=(0, 1, 2, 5, 10, 20, 50, 100, 250),
)

RAG_EMPTY_RESULT_TOTAL = _counter_pair(
    "s18_rag_empty_result_total",
    "wiseai_rag_empty_result_total",
    "Number of RAG requests that returned empty results",
    ["endpoint"],
)

MCP_TOOL_CALLS_TOTAL = _counter_pair(
    "s18_mcp_tool_calls_total",
    "wiseai_mcp_tool_calls_total",
    "Total MCP tool calls by status",
    ["tool", "status", "integration_id", "workflow_id", "contract_version"],
)

MCP_TOOL_LATENCY_MS = _histogram_pair(
    "s18_mcp_tool_latency_ms",
    "wiseai_mcp_tool_latency_ms",
    "MCP tool call latency in milliseconds",
    labels=["tool", "integration_id", "workflow_id", "contract_version"],
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)

MEMORY_OPERATIONS_TOTAL = _counter_pair(
    "s18_memory_operations_total",
    "wiseai_memory_operations_total",
    "Total memory operations by endpoint and status",
    ["endpoint", "status"],
)

MEMORY_OPERATION_LATENCY_MS = _histogram_pair(
    "s18_memory_operation_latency_ms",
    "wiseai_memory_operation_latency_ms",
    "Memory endpoint operation latency in milliseconds",
    labels=["endpoint"],
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)


def now_ms() -> float:
    return time.perf_counter() * 1000


def elapsed_ms(start_ms: float) -> float:
    return (time.perf_counter() * 1000) - start_ms


def normalize_status_class(status_code: int) -> str:
    return f"{status_code // 100}xx"


def route_template(path: str, scope_route: Optional[object]) -> str:
    if scope_route is not None and hasattr(scope_route, "path"):
        return getattr(scope_route, "path") or path
    return path
