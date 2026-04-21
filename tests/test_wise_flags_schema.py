"""Unit tests for Wise output normalization and policy hooks."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.adapters.wise_output import (
    apply_cbc_evidence_to_wise_output,
    ensure_wise_output_schema,
    merge_wise_flag_lists,
    normalize_wise_flags,
    sync_wise_response_footer,
)
from integrations.policies.output_hooks import apply_output_policy


def test_normalize_wise_flags_list_dedup():
    assert normalize_wise_flags(["a", "a", "b"]) == ["a", "b"]
    assert normalize_wise_flags([]) == []


def test_normalize_wise_flags_dict_truthy_keys():
    assert normalize_wise_flags({"low_hemoglobin": True, "high_wbc": False}) == [
        "low_hemoglobin"
    ]
    assert normalize_wise_flags({"x": 1, "y": 0}) == ["x"]


def test_normalize_wise_flags_string():
    assert normalize_wise_flags("  single  ") == ["single"]
    assert normalize_wise_flags("") == []


def test_merge_wise_flag_lists_order_and_union():
    assert merge_wise_flag_lists(["a"], ["b", "a"]) == ["a", "b"]
    assert merge_wise_flag_lists([], ["x"]) == ["x"]
    assert merge_wise_flag_lists(["x"], []) == ["x"]


def test_ensure_wise_output_schema_dict_flags_merged_with_extracted():
    raw = 'Risk Level: moderate\nFlags: ["from_footer"]'
    out = ensure_wise_output_schema(
        {"response": "x", "flags": {"low_hemoglobin": True, "high_wbc": False}},
        raw,
    )
    assert set(out["flags"]) == {"low_hemoglobin", "from_footer"}


def test_ensure_wise_output_schema_list_parsed_plus_footer():
    raw = 'Flags: ["footer_only"]'
    out = ensure_wise_output_schema(
        {"response": "ok", "flags": ["parsed_first"]},
        raw,
    )
    assert out["flags"] == ["parsed_first", "footer_only"]


def test_normalize_wise_flags_none_returns_empty():
    assert normalize_wise_flags(None) == []


def test_normalize_wise_flags_dict_string_truthy():
    assert normalize_wise_flags({"a": "true", "b": "false", "c": "yes"}) == ["a", "c"]


def test_merge_wise_flag_lists_both_empty():
    assert merge_wise_flag_lists([], []) == []


def test_ensure_wise_output_schema_empty_list_flags_stays_empty():
    """Model explicitly returns flags: [] — normalization should NOT invent flags."""
    out = ensure_wise_output_schema(
        {"response": "all normal", "risk_level": "low", "confidence": 0.95, "flags": []},
        "all normal",
    )
    assert out["flags"] == []
    assert out["risk_level"] == "low"


def test_apply_cbc_evidence_overrides_hallucinated_flags():
    """With a normal CBC in the query, evidence rules strip impossible LLM flags."""
    q = (
        '[Patient ID: x] Request: '
        '{"hemoglobin": 13.5, "wbc": 7.0, "rbc": 4.5, "platelets": 250000}'
    )
    out = apply_cbc_evidence_to_wise_output(
        {
            "response": "text",
            "risk_level": "high",
            "confidence": 0.95,
            "flags": ["low_hemoglobin", "high_wbc"],
        },
        {"original_query": q},
    )
    assert out["flags"] == []
    assert out["risk_level"] == "low"


def test_sync_wise_response_footer_rewrites_contradictory_values():
    out = sync_wise_response_footer(
        {
            "response": '- Risk Level: moderate\n- Confidence: 0.9\n- Flags: ["high_wbc"]',
            "risk_level": "low",
            "confidence": 0.9,
            "flags": [],
        }
    )
    assert "Risk Level: low" in out["response"]
    assert 'Flags: []' in out["response"]


def test_apply_output_policy_uses_wise_profile_for_summarizer():
    input_data = {
        "integration_meta": {"integration_id": "wiseai", "workflow_id": "cdss", "contract_version": "v1"},
        "original_query": '[Patient ID: x] Request: {"hemoglobin": 13.5, "wbc": 7.0, "rbc": 4.5, "platelets": 250000}',
    }
    out = apply_output_policy(
        agent_type="SummarizerAgent",
        output={"response": "Clinical summary", "risk_level": "high", "confidence": 0.95, "flags": ["high_wbc"]},
        raw_response="Risk Level: high",
        input_data=input_data,
    )
    assert out["risk_level"] == "low"
    assert out["flags"] == []


def test_apply_output_policy_default_is_passthrough():
    output = {"response": "ok"}
    out = apply_output_policy(
        agent_type="ThinkerAgent",
        output=output,
        raw_response="ok",
        input_data={"integration_meta": {"integration_id": "default", "workflow_id": "generic"}},
    )
    assert out == output
