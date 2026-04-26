import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.adapters.wiseai import WiseAIIntegrationAdapter


def test_wise_adapter_normalizes_defaults():
    adapter = WiseAIIntegrationAdapter()
    canonical = adapter.to_canonical(
        {
            "query": "interpret cbc",
            "source_system": "s18",
            "workflow_id": "generic",
        }
    )
    assert canonical.integration_id == "wiseai"
    assert canonical.workflow_id == "cdss"
    assert canonical.source_system == "wiseai"


def test_wise_adapter_response_envelope_includes_contract_metadata():
    adapter = WiseAIIntegrationAdapter()
    canonical = adapter.to_canonical({"query": "hello"})
    response = adapter.from_canonical({"id": "run-1", "status": "starting"}, canonical)
    assert response["integration_id"] == "wiseai"
    assert response["workflow_id"] == "cdss"
    assert response["contract_version"] == "v1"
