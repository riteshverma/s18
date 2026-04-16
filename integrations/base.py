from typing import Any, Dict, Protocol

from integrations.contracts import CanonicalRunRequest


class IntegrationAdapter(Protocol):
    integration_id: str

    def to_canonical(self, raw_request: Dict[str, Any]) -> CanonicalRunRequest:
        ...

    def from_canonical(self, run_result: Dict[str, Any], context: CanonicalRunRequest) -> Dict[str, Any]:
        ...
