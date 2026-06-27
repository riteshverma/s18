from typing import Any, Dict

from integrations.contracts import CanonicalRunRequest, CanonicalRunResponse
from integrations.profiles import load_integration_profile


class DefaultIntegrationAdapter:
    integration_id = "default"

    def to_canonical(self, raw_request: Dict[str, Any]) -> CanonicalRunRequest:
        integration_id = (raw_request.get("integration_id") or "default").strip().lower()
        workflow_id = (raw_request.get("workflow_id") or "generic").strip().lower()
        contract_version = raw_request.get("contract_version") or "v1"
        profile = load_integration_profile(integration_id, workflow_id, contract_version)
        payload = raw_request.get("raw_payload")
        query = (raw_request.get("query") or "").strip()
        if not query:
            raise ValueError("query must be non-empty")
        return CanonicalRunRequest(
            contract_version=contract_version,
            integration_id=integration_id,
            workflow_id=workflow_id,
            query=query,
            model=raw_request.get("model"),
            source_system=(raw_request.get("source_system") or "s18").strip().lower(),
            tenant_id=(raw_request.get("tenant_id") or "default").strip().lower(),
            tenant_tier=(raw_request.get("tenant_tier") or "starter").strip().lower(),
            data_region=(raw_request.get("data_region") or "in").strip().lower(),
            external_event_id=raw_request.get("external_event_id"),
            consent_ref=raw_request.get("consent_ref"),
            raw_payload=payload,
            idempotency_key=raw_request.get("idempotency_key"),
            skill_id=raw_request.get("skill_id"),
            policy={
                "risk_profile": profile.get("risk_profile", "generic_default"),
                "response_profile": profile.get("response_profile", "default_v1"),
            },
            payload={"raw_payload": payload} if payload else {},
            audit={
                "source_system": (raw_request.get("source_system") or "s18").strip().lower(),
                "consent_ref": raw_request.get("consent_ref"),
            },
        )

    def from_canonical(self, run_result: Dict[str, Any], context: CanonicalRunRequest) -> Dict[str, Any]:
        response = dict(run_result)
        response["integration_id"] = context.integration_id
        response["workflow_id"] = context.workflow_id
        response["contract_version"] = context.contract_version
        response["tenant_id"] = context.tenant_id
        response["tenant_tier"] = context.tenant_tier
        response["data_region"] = context.data_region
        canonical = CanonicalRunResponse.model_validate(response)
        return canonical.model_dump()
