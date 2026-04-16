from typing import Any, Dict

from integrations.adapters.default import DefaultIntegrationAdapter


class WiseAIIntegrationAdapter(DefaultIntegrationAdapter):
    integration_id = "wiseai"

    def to_canonical(self, raw_request: Dict[str, Any]):
        raw = dict(raw_request)
        raw["integration_id"] = "wiseai"
        if not raw.get("workflow_id") or raw.get("workflow_id") == "generic":
            raw["workflow_id"] = "cdss"
        canonical = super().to_canonical(raw)
        if not canonical.source_system or canonical.source_system == "s18":
            canonical.source_system = "wiseai"
        return canonical
