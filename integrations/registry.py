from typing import Dict, Optional

from integrations.adapters.default import DefaultIntegrationAdapter
from integrations.adapters.powerapps import PowerAppsIntegrationAdapter
from integrations.adapters.wiseai import WiseAIIntegrationAdapter
from integrations.base import IntegrationAdapter

_ADAPTERS = {
    "default": DefaultIntegrationAdapter(),
    "wiseai": WiseAIIntegrationAdapter(),
    "powerapps": PowerAppsIntegrationAdapter(),
}

# Registry of adapter overrides per tenant_tier. Empty today; kept here so the
# growth/enterprise-health routing path is a documented extension point rather
# than a silent no-op. Populate with e.g. {"enterprise-health": "wiseai"} when
# a tier requires a dedicated adapter.
_TIER_ADAPTER_OVERRIDES: Dict[str, str] = {}


def _resolve_adapter_key(
    integration_id: Optional[str],
    source_system: Optional[str],
    tenant_context: Optional[Dict[str, str]],
) -> str:
    key = (integration_id or "").strip().lower()
    if not key:
        key = (source_system or "").strip().lower()
    # Source-system aliases for backward compatibility.
    if key in {"wise", "wise-ai"}:
        key = "wiseai"
    if key in {"power-apps", "power_apps", "powerapp", "power-platform"}:
        key = "powerapps"
    if not key:
        key = "default"

    # Tier-specific override wins only when caller left adapter selection to
    # defaults; explicit integration_id from the client is always respected.
    if tenant_context and not integration_id:
        tier = (tenant_context.get("tenant_tier") or "").strip().lower()
        override = _TIER_ADAPTER_OVERRIDES.get(tier)
        if override and override in _ADAPTERS:
            key = override
    return key


def get_integration_adapter(
    integration_id: str | None = None,
    source_system: str | None = None,
    tenant_context: Optional[Dict[str, str]] = None,
) -> IntegrationAdapter:
    key = _resolve_adapter_key(integration_id, source_system, tenant_context)
    return _ADAPTERS.get(key, _ADAPTERS["default"])
