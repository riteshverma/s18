"""Unit tests for Wise flags normalization in AgentRunner."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from agents.base_agent import AgentRunner


@pytest.fixture
def runner():
    return AgentRunner(MagicMock())


def test_normalize_wise_flags_list_dedup(runner):
    assert runner._normalize_wise_flags(["a", "a", "b"]) == ["a", "b"]
    assert runner._normalize_wise_flags([]) == []


def test_normalize_wise_flags_dict_truthy_keys(runner):
    assert runner._normalize_wise_flags({"low_hemoglobin": True, "high_wbc": False}) == [
        "low_hemoglobin"
    ]
    assert runner._normalize_wise_flags({"x": 1, "y": 0}) == ["x"]


def test_normalize_wise_flags_string(runner):
    assert runner._normalize_wise_flags("  single  ") == ["single"]
    assert runner._normalize_wise_flags("") == []


def test_merge_wise_flag_lists_order_and_union(runner):
    assert runner._merge_wise_flag_lists(["a"], ["b", "a"]) == ["a", "b"]
    assert runner._merge_wise_flag_lists([], ["x"]) == ["x"]
    assert runner._merge_wise_flag_lists(["x"], []) == ["x"]


def test_ensure_wise_output_schema_dict_flags_merged_with_extracted(runner):
    raw = 'Risk Level: moderate\nFlags: ["from_footer"]'
    out = runner._ensure_wise_output_schema(
        {"response": "x", "flags": {"low_hemoglobin": True, "high_wbc": False}},
        raw,
    )
    assert set(out["flags"]) == {"low_hemoglobin", "from_footer"}


def test_ensure_wise_output_schema_list_parsed_plus_footer(runner):
    raw = 'Flags: ["footer_only"]'
    out = runner._ensure_wise_output_schema(
        {"response": "ok", "flags": ["parsed_first"]},
        raw,
    )
    assert out["flags"] == ["parsed_first", "footer_only"]


def test_normalize_wise_flags_none_returns_empty(runner):
    assert runner._normalize_wise_flags(None) == []


def test_normalize_wise_flags_dict_string_truthy(runner):
    assert runner._normalize_wise_flags({"a": "true", "b": "false", "c": "yes"}) == ["a", "c"]


def test_merge_wise_flag_lists_both_empty(runner):
    assert runner._merge_wise_flag_lists([], []) == []


def test_ensure_wise_output_schema_empty_list_flags_stays_empty(runner):
    """Model explicitly returns flags: [] — normalization should NOT invent flags."""
    out = runner._ensure_wise_output_schema(
        {"response": "all normal", "risk_level": "low", "confidence": 0.95, "flags": []},
        "all normal",
    )
    assert out["flags"] == []
    assert out["risk_level"] == "low"


def test_apply_cbc_evidence_overrides_hallucinated_flags(runner):
    """With a normal CBC in the query, evidence rules strip impossible LLM flags."""
    q = (
        '[Patient ID: x] Request: '
        '{"hemoglobin": 13.5, "wbc": 7.0, "rbc": 4.5, "platelets": 250000}'
    )
    out = runner._apply_cbc_evidence_to_wise_output(
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


def test_sync_wise_response_footer_rewrites_contradictory_values(runner):
    out = runner._sync_wise_response_footer(
        {
            "response": '- Risk Level: moderate\n- Confidence: 0.9\n- Flags: ["high_wbc"]',
            "risk_level": "low",
            "confidence": 0.9,
            "flags": [],
        }
    )
    assert "Risk Level: low" in out["response"]
    assert 'Flags: []' in out["response"]
