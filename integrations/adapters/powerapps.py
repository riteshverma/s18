"""IntegrationAdapter for Power Apps / Power Automate ingest payloads."""

from __future__ import annotations

from typing import Any, Dict

from integrations.contracts import CanonicalRunRequest, CanonicalRunResponse
from integrations.profiles import load_integration_profile


class PowerAppsIntegrationAdapter:
    """Translates Power Automate envelopes into the S18 canonical request.

    Power Automate flows wrap their HTTP action payloads in
    `{ "trigger": {...}, "record": {...}, "files": [...] }`. We accept any
    shape and surface useful fields for routing while keeping the original
    envelope on `raw_payload`.
    """

    integration_id = "powerapps"

    def to_canonical(self, raw_request: Dict[str, Any]) -> CanonicalRunRequest:
        contract_version = raw_request.get("contract_version") or "v1"
        workflow_id = (raw_request.get("workflow_id") or "generic").strip().lower()
        profile = load_integration_profile(self.integration_id, workflow_id, contract_version)

        record = raw_request.get("record") or {}
        files = raw_request.get("files") or []
        trigger = raw_request.get("trigger") or {}

        query = (raw_request.get("query") or trigger.get("description") or "").strip()
        if not query:
            # Generate a deterministic descriptor so the canonical contract is
            # satisfied even when Power Automate omits a free-form prompt.
            entity = record.get("tableLogicalName") or trigger.get("entity") or "record"
            record_id = record.get("recordId") or record.get("id") or "unknown"
            query = f"powerapps:{entity}:{record_id}"

        tenant_id = (
            raw_request.get("tenant_id")
            or trigger.get("tenant_id")
            or "default"
        )
        tenant_tier = raw_request.get("tenant_tier") or "enterprise-health"
        data_region = raw_request.get("data_region") or trigger.get("data_region") or "in"

        payload: Dict[str, Any] = {
            "record": record,
            "files": files,
            "trigger": trigger,
        }

        return CanonicalRunRequest(
            contract_version=contract_version,
            integration_id=self.integration_id,
            workflow_id=workflow_id,
            query=query,
            model=raw_request.get("model"),
            source_system="powerapps",
            tenant_id=str(tenant_id).strip().lower(),
            tenant_tier=str(tenant_tier).strip().lower(),
            data_region=str(data_region).strip().lower(),
            external_event_id=raw_request.get("external_event_id") or trigger.get("event_id"),
            consent_ref=raw_request.get("consent_ref") or trigger.get("consent_ref"),
            raw_payload=raw_request,
            idempotency_key=raw_request.get("idempotency_key") or trigger.get("idempotency_key"),
            policy={
                "risk_profile": profile.get("risk_profile", "powerapps_default"),
                "response_profile": profile.get("response_profile", "powerapps_v1"),
                "redact_pii": bool(profile.get("redact_pii", False)),
                "object_store_provider": profile.get("object_store_provider"),
                "vector_store_provider": profile.get("vector_store_provider"),
                "embedding_provider": profile.get("embedding_provider"),
            },
            payload=payload,
            audit={
                "source_system": "powerapps",
                "consent_ref": raw_request.get("consent_ref") or trigger.get("consent_ref"),
                "trigger_kind": trigger.get("kind") or "manual",
                "record_entity": record.get("tableLogicalName"),
                "record_id": record.get("recordId") or record.get("id"),
                "file_count": len(files),
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
