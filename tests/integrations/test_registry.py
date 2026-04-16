import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.adapters.default import DefaultIntegrationAdapter
from integrations.adapters.wiseai import WiseAIIntegrationAdapter
from integrations.registry import get_integration_adapter


def test_registry_returns_wise_adapter():
    adapter = get_integration_adapter("wiseai")
    assert isinstance(adapter, WiseAIIntegrationAdapter)


def test_registry_falls_back_to_default():
    adapter = get_integration_adapter("unknown-system")
    assert isinstance(adapter, DefaultIntegrationAdapter)


def test_registry_uses_source_system_fallback_for_wiseai():
    adapter = get_integration_adapter(integration_id=None, source_system="wiseai")
    assert isinstance(adapter, WiseAIIntegrationAdapter)


def test_registry_maps_wise_alias():
    adapter = get_integration_adapter(integration_id="wise-ai")
    assert isinstance(adapter, WiseAIIntegrationAdapter)
