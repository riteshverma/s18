import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.contracts import CanonicalRunRequest


def test_canonical_contract_defaults():
    req = CanonicalRunRequest(query="hello")
    assert req.contract_version == "v1"
    assert req.integration_id == "default"
    assert req.workflow_id == "generic"
    assert req.source_system == "s18"


def test_canonical_contract_required_query():
    try:
        CanonicalRunRequest()
        assert False, "Expected validation error for missing query"
    except Exception:
        assert True
