from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CanonicalRunRequest(BaseModel):
    contract_version: str = "v1"
    integration_id: str = "default"
    workflow_id: str = "generic"
    query: str
    model: Optional[str] = None
    source_system: str = "s18"
    tenant_id: str = "default"
    tenant_tier: str = "starter"
    data_region: Optional[str] = None
    external_event_id: Optional[str] = None
    consent_ref: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    session: Dict[str, Any] = Field(default_factory=dict)
    policy: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)
    audit: Dict[str, Any] = Field(default_factory=dict)


class CanonicalRunResult(BaseModel):
    run_id: str
    status: str
    summary: Optional[str] = None
    output: Optional[str] = None
    contract_version: str = "v1"
    integration_id: str = "default"
    workflow_id: str = "generic"
