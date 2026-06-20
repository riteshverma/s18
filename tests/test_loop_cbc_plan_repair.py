import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.loop import AgentLoop4
from memory.context import ExecutionContextManager


def test_repair_cbc_plan_dependencies_wires_summary_after_ehr():
    loop = AgentLoop4(MagicMock())
    new_nodes = [
        {
            "id": "T001",
            "agent": "SummarizationAgent",
            "reads": ["original_query"],
            "writes": ["clinical_note"],
            "status": "pending",
        },
        {
            "id": "T002",
            "agent": "EHRDataMinerAgent",
            "reads": ["original_query"],
            "writes": ["patient_record"],
            "status": "pending",
        },
    ]
    new_edges = [{"source": "Query", "target": "T001"}]

    loop._repair_cbc_plan_dependencies(new_nodes, new_edges)

    assert "patient_record" in new_nodes[0]["reads"]
    assert {"source": "T002", "target": "T001"} in new_edges


def test_enforce_cbc_full_mode_minimum_plan_when_single_thinker():
    loop = AgentLoop4(MagicMock())
    query = (
        "[Patient ID: abc] [Execution Mode: full] "
        'Request: {"hemoglobin": 7.0, "wbc": 14000, "platelets": 250000}'
    )
    out = {
        "plan_graph": {
            "nodes": [
                {
                    "id": "T001",
                    "agent": "ThinkerAgent",
                    "reads": ["original_query"],
                    "writes": ["response"],
                    "status": "pending",
                }
            ],
            "edges": [{"source": "Query", "target": "T001"}],
        },
        "next_step_id": "T001",
    }

    loop._enforce_cbc_full_mode_minimum_plan(query, out)

    nodes = out["plan_graph"]["nodes"]
    assert len(nodes) == 2
    assert nodes[0]["agent"] == "EHRDataMinerAgent"
    assert nodes[1]["agent"] == "ClinicalReasoningAgent"
    assert out["next_step_id"] == "T001"
    assert {"source": "Query", "target": "T001"} in out["plan_graph"]["edges"]
    assert {"source": "T001", "target": "T002"} in out["plan_graph"]["edges"]


def test_filter_memory_context_for_cbc_removes_mental_health_lines():
    loop = AgentLoop4(MagicMock())
    query = (
        "[Patient ID: abc] [Execution Mode: full] "
        'Request: {"hemoglobin": 8.0, "wbc": 14000, "platelets": 250000}'
    )
    memory_context = (
        "PREVIOUS MEMORIES ABOUT USER:\n"
        "- CBC: low hemoglobin with high wbc.\n"
        "- Task: mental_health PHQ9 12 and GAD7 8.\n"
        "- anxiety and depression screening details.\n"
        "- Keep this CBC trend note."
    )

    filtered = loop._filter_memory_context_for_cbc(query, memory_context)

    assert isinstance(filtered, str)
    assert "CBC: low hemoglobin" in filtered
    assert "Keep this CBC trend note" in filtered
    assert "mental_health" not in filtered
    assert "PHQ9" not in filtered
    assert "anxiety" not in filtered.lower()


def test_filter_memory_context_for_mental_health_removes_cbc_lines():
    loop = AgentLoop4(MagicMock())
    query = (
        "[Task: mental_health] [Patient ID: abc] [Execution Mode: full] "
        'Request: {"task": "mental_health", "patient_payload": {"phq9_total": 11, "gad7_total": 8}}'
    )
    memory_context = (
        "PREVIOUS MEMORIES ABOUT USER:\n"
        "- CBC: low hemoglobin with high wbc.\n"
        "- Platelets trend looked stable.\n"
        "- Task: mental_health PHQ9 12 and GAD7 8.\n"
        "- Keep this anxiety follow-up note."
    )

    filtered = loop._filter_memory_context_for_mental_health(query, memory_context)

    assert isinstance(filtered, str)
    assert "mental_health" in filtered
    assert "anxiety follow-up note" in filtered
    assert "hemoglobin" not in filtered.lower()
    assert "platelets" not in filtered.lower()


def test_enforce_mental_health_plan_guard_rewrites_cbc_agents():
    loop = AgentLoop4(MagicMock())
    query = (
        "[Task: mental_health] [Patient ID: abc] [Execution Mode: full] "
        'Request: {"task": "mental_health", "patient_payload": {"phq9_total": 14, "gad7_total": 9}}'
    )
    out = {
        "plan_graph": {
            "nodes": [
                {
                    "id": "T001",
                    "agent": "CBCAgent",
                    "reads": ["original_query"],
                    "writes": ["cbc_results"],
                    "status": "pending",
                },
                {
                    "id": "T002",
                    "agent": "ClinicalReasoningAgent",
                    "reads": ["cbc_results"],
                    "writes": ["response"],
                    "status": "pending",
                },
            ],
            "edges": [
                {"source": "Query", "target": "T001"},
                {"source": "T001", "target": "T002"},
            ],
        },
        "next_step_id": "T001",
    }

    loop._enforce_mental_health_plan_guard(query, out)

    nodes = out["plan_graph"]["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["agent"] == "ThinkerAgent"
    assert out["next_step_id"] == "T001"
    assert out["plan_graph"]["edges"] == [{"source": "Query", "target": "T001"}]


def test_execute_dag_skips_dependents_after_terminal_step_failure(monkeypatch, tmp_path):
    context = ExecutionContextManager(
        {
            "nodes": [
                {
                    "id": "Query",
                    "agent": "PlannerAgent",
                    "status": "completed",
                    "writes": ["plan_graph"],
                },
                {
                    "id": "T001",
                    "agent": "ThinkerAgent",
                    "description": "Failing upstream step",
                    "reads": ["original_query"],
                    "writes": ["analysis"],
                    "status": "pending",
                },
                {
                    "id": "T002",
                    "agent": "FormatterAgent",
                    "description": "Dependent step",
                    "reads": ["analysis"],
                    "writes": ["response"],
                    "status": "pending",
                },
            ],
            "edges": [
                {"source": "ROOT", "target": "Query"},
                {"source": "Query", "target": "T001"},
                {"source": "T001", "target": "T002"},
            ],
        },
        session_id="dag-failure-test",
        original_query="test query",
    )
    context.session_file = tmp_path / "session.json"

    loop = AgentLoop4(MagicMock())

    async def fail_step(step_id, _context):
        assert step_id == "T001"
        return {"success": False, "error": "provider unavailable"}

    monkeypatch.setattr(loop, "_execute_step", fail_step)

    asyncio.run(loop._execute_dag(context))

    assert context.plan_graph.nodes["T001"]["status"] == "failed"
    assert context.plan_graph.nodes["T002"]["status"] == "skipped"
    assert context.plan_graph.nodes["T002"]["skip_reason"] == "upstream_failed:T001"
    assert context.plan_graph.graph["status"] == "failed"
