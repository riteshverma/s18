"""Adapter contract tests for the PowerApps integration."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.adapters.powerapps import PowerAppsIntegrationAdapter
from integrations.contracts import CanonicalRunRequest
from integrations.registry import get_integration_adapter


def test_powerapps_adapter_resolves_via_registry():
    adapter = get_integration_adapter("powerapps")
    assert isinstance(adapter, PowerAppsIntegrationAdapter)


def test_powerapps_alias_powerapp_routes_to_adapter():
    adapter = get_integration_adapter(integration_id="power-apps")
    assert isinstance(adapter, PowerAppsIntegrationAdapter)


def test_to_canonical_normalizes_envelope():
    adapter = PowerAppsIntegrationAdapter()
    payload = {
        "tenant_id": "Acme",
        "tenant_tier": "Enterprise-Health",
        "data_region": "us",
        "workflow_id": "Claims",
        "trigger": {"kind": "dataverse_row_change", "entity": "claim"},
        "record": {"recordId": "abc-123", "tableLogicalName": "claim", "fields": {"amount": 99}},
        "files": [{"name": "x.pdf", "contentBytes": "Zm9v"}],
        "consent_ref": "consent-1",
    }
    canonical = adapter.to_canonical(payload)
    assert isinstance(canonical, CanonicalRunRequest)
    assert canonical.integration_id == "powerapps"
    assert canonical.workflow_id == "claims"
    assert canonical.tenant_id == "acme"
    assert canonical.tenant_tier == "enterprise-health"
    assert canonical.data_region == "us"
    assert canonical.audit["record_id"] == "abc-123"
    assert canonical.audit["file_count"] == 1
    assert canonical.policy["response_profile"] == "powerapps_v1"


def test_to_canonical_synthesizes_query_when_missing():
    adapter = PowerAppsIntegrationAdapter()
    canonical = adapter.to_canonical(
        {
            "tenant_id": "default",
            "record": {"recordId": "row-9", "tableLogicalName": "incident"},
        }
    )
    assert canonical.query.startswith("powerapps:incident:row-9")
