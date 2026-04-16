from core.utils import log_step


def is_cbc_payload_query(query: str) -> bool:
    q = (query or "").lower()
    return (
        "[patient id:" in q
        and "request:" in q
        and "hemoglobin" in q
        and "wbc" in q
        and "platelets" in q
    )


def is_fast_mode(query: str) -> bool:
    return "[execution mode: fast]" in (query or "").lower()


def is_mental_health_task_query(query: str) -> bool:
    q = (query or "").lower()
    return (
        "[task: mental_health]" in q
        or '"task": "mental_health"' in q
        or '"task":"mental_health"' in q
    )


def filter_memory_context_for_cbc(query: str, memory_context):
    if not is_cbc_payload_query(query):
        return memory_context
    if not isinstance(memory_context, str) or not memory_context.strip():
        return memory_context
    mh_markers = (
        "mental_health",
        "task: mental_health",
        "phq9",
        "gad7",
        "suicidal",
        "self_harm",
        "depression",
        "anxiety",
        "local screening",
    )
    kept_lines = []
    removed = 0
    for line in memory_context.splitlines():
        low = line.lower()
        if any(marker in low for marker in mh_markers):
            removed += 1
            continue
        kept_lines.append(line)
    if removed == 0:
        return memory_context
    filtered = "\n".join(kept_lines).strip()
    log_step(f"Filtered {removed} mental-health memory lines for CBC planning")
    return filtered if filtered else None


def filter_memory_context_for_mental_health(query: str, memory_context):
    if not is_mental_health_task_query(query):
        return memory_context
    if not isinstance(memory_context, str) or not memory_context.strip():
        return memory_context
    cbc_markers = (
        "cbc",
        "hemoglobin",
        "wbc",
        "rbc",
        "platelets",
        "anemia",
        "leukocytosis",
        "task: cbc",
        "cbc_results",
    )
    kept_lines = []
    removed = 0
    for line in memory_context.splitlines():
        low = line.lower()
        if any(marker in low for marker in cbc_markers):
            removed += 1
            continue
        kept_lines.append(line)
    if removed == 0:
        return memory_context
    filtered = "\n".join(kept_lines).strip()
    log_step(f"Filtered {removed} CBC memory lines for mental-health planning")
    return filtered if filtered else None


def enforce_cbc_full_mode_minimum_plan(query: str, out: dict):
    if not is_cbc_payload_query(query) or is_fast_mode(query):
        return
    if not isinstance(out, dict):
        return
    pg = out.get("plan_graph")
    if not isinstance(pg, dict):
        pg = {"nodes": [], "edges": []}
        out["plan_graph"] = pg
    nodes = pg.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
        pg["nodes"] = nodes
    has_miner = any(isinstance(n, dict) and n.get("agent") == "EHRDataMinerAgent" for n in nodes)
    if len(nodes) >= 2 and has_miner:
        return
    pg["nodes"] = [
        {
            "id": "T001",
            "agent": "EHRDataMinerAgent",
            "description": "Retrieve patient's CBC context from available records.",
            "reads": ["original_query"],
            "writes": ["cbc_results"],
            "status": "pending",
        },
        {
            "id": "T002",
            "agent": "ClinicalReasoningAgent",
            "description": "Interpret CBC payload/results and produce risk, confidence, and flags.",
            "reads": ["cbc_results"],
            "writes": ["response"],
            "status": "pending",
        },
    ]
    pg["edges"] = [{"source": "Query", "target": "T001"}, {"source": "T001", "target": "T002"}]
    out["next_step_id"] = "T001"
    log_step("Enforced CBC full-mode minimum multi-step plan")


def enforce_mental_health_plan_guard(query: str, out: dict) -> bool:
    if not is_mental_health_task_query(query):
        return False
    if not isinstance(out, dict):
        return False
    pg = out.get("plan_graph")
    if not isinstance(pg, dict):
        pg = {"nodes": [], "edges": []}
        out["plan_graph"] = pg
    nodes = pg.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
        pg["nodes"] = nodes
    blocked_agents = {"CBCAgent", "EHRDataMinerAgent", "TrendAgent", "SearchLabsAgent"}
    has_blocked = any(isinstance(node, dict) and node.get("agent") in blocked_agents for node in nodes)
    if not has_blocked:
        return False
    pg["nodes"] = [
        {
            "id": "T001",
            "agent": "ThinkerAgent",
            "description": "Analyze mental-health payload and produce risk, confidence, and flags.",
            "reads": ["original_query"],
            "writes": ["response"],
            "status": "pending",
        }
    ]
    pg["edges"] = [{"source": "Query", "target": "T001"}]
    out["next_step_id"] = "T001"
    log_step("Enforced mental-health plan guard (removed CBC/lab routing)")
    return True
