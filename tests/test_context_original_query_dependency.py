import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from memory.context import ExecutionContextManager


def _make_context(original_query: str = "cbc query"):
    return ExecutionContextManager(
        plan_graph={"nodes": [], "edges": []},
        session_id="test-session",
        original_query=original_query,
        file_manifest=[],
    )


def test_get_inputs_returns_original_query_from_globals():
    ctx = _make_context("hello")
    inputs = ctx.get_inputs(["original_query"])
    assert inputs["original_query"] == "hello"


def test_get_inputs_falls_back_to_graph_original_query_when_globals_missing():
    ctx = _make_context("fallback query")
    ctx.plan_graph.graph["globals_schema"].pop("original_query", None)

    inputs = ctx.get_inputs(["original_query"])

    assert inputs["original_query"] == "fallback query"


def test_get_inputs_still_warns_for_missing_non_root_key(capsys):
    ctx = _make_context("hello")

    inputs = ctx.get_inputs(["patient_record"])

    assert "patient_record" not in inputs
    captured = capsys.readouterr()
    assert "Missing dependency: 'patient_record'" in captured.out
