import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.loop import AgentLoop4


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
