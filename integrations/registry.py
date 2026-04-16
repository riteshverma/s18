from integrations.adapters.default import DefaultIntegrationAdapter
from integrations.adapters.wiseai import WiseAIIntegrationAdapter
from integrations.base import IntegrationAdapter

_ADAPTERS = {
    "default": DefaultIntegrationAdapter(),
    "wiseai": WiseAIIntegrationAdapter(),
}


def get_integration_adapter(
    integration_id: str | None = None,
    source_system: str | None = None,
) -> IntegrationAdapter:
    key = (integration_id or "").strip().lower()
    if not key:
        key = (source_system or "").strip().lower()
    # Source-system aliases for backward compatibility.
    if key in {"wise", "wise-ai"}:
        key = "wiseai"
    if not key:
        key = "default"
    return _ADAPTERS.get(key, _ADAPTERS["default"])
