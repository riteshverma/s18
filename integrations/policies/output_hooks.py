from typing import Any, Dict, Optional, Protocol, Tuple

from integrations.adapters.wise_output import (
    apply_cbc_evidence_to_wise_output,
    ensure_wise_output_schema,
    sync_wise_response_footer,
)


class OutputPolicy(Protocol):
    def transform(
        self,
        agent_type: str,
        output: Any,
        raw_response: str,
        input_data: Dict[str, Any],
    ) -> Any:
        ...


class DefaultOutputPolicy:
    def transform(
        self,
        agent_type: str,
        output: Any,
        raw_response: str,
        input_data: Dict[str, Any],
    ) -> Any:
        return output


class WiseCdssOutputPolicy:
    def transform(
        self,
        agent_type: str,
        output: Any,
        raw_response: str,
        input_data: Dict[str, Any],
    ) -> Any:
        if not isinstance(output, dict):
            output = {"response": str(output) if output is not None else raw_response}

        if agent_type in {"ThinkerAgent", "SummarizerAgent"}:
            output = ensure_wise_output_schema(output, raw_response)
            output = apply_cbc_evidence_to_wise_output(output, input_data)
            if agent_type == "SummarizerAgent":
                output = sync_wise_response_footer(output)
            return output

        if agent_type == "FormatterAgent":
            gs = input_data.get("all_globals_schema") or {}
            if "risk_level" not in output and gs.get("_wise_risk_level"):
                output["risk_level"] = gs["_wise_risk_level"]
            if "confidence" not in output and gs.get("_wise_confidence") is not None:
                output["confidence"] = gs["_wise_confidence"]
            if not isinstance(output.get("flags"), list) and isinstance(gs.get("_wise_flags"), list):
                output["flags"] = gs["_wise_flags"]
            return sync_wise_response_footer(output)

        return output


_DEFAULT_POLICY = DefaultOutputPolicy()
_WISE_POLICY = WiseCdssOutputPolicy()
_POLICIES: Dict[Tuple[str, str], OutputPolicy] = {
    ("wiseai", "cdss"): _WISE_POLICY,
    ("wiseai", "generic"): _WISE_POLICY,
}


def resolve_integration_meta(input_data: Dict[str, Any]) -> Dict[str, str]:
    meta: Optional[Dict[str, Any]] = None
    if isinstance(input_data.get("integration_meta"), dict):
        meta = input_data.get("integration_meta")
    elif isinstance(input_data.get("_integration_meta"), dict):
        meta = input_data.get("_integration_meta")
    else:
        session_context = input_data.get("session_context")
        if isinstance(session_context, dict) and isinstance(session_context.get("integration_meta"), dict):
            meta = session_context.get("integration_meta")

    base = meta or {}
    return {
        "integration_id": str(base.get("integration_id") or "default").strip().lower(),
        "workflow_id": str(base.get("workflow_id") or "generic").strip().lower(),
        "contract_version": str(base.get("contract_version") or "v1").strip().lower(),
    }


def apply_output_policy(
    agent_type: str,
    output: Any,
    raw_response: str,
    input_data: Dict[str, Any],
) -> Any:
    meta = resolve_integration_meta(input_data)
    policy = _POLICIES.get(
        (meta["integration_id"], meta["workflow_id"]),
        _POLICIES.get((meta["integration_id"], "generic"), _DEFAULT_POLICY),
    )
    return policy.transform(agent_type, output, raw_response, input_data)
