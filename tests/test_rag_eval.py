import json
from pathlib import Path

from core.rag_eval import evaluate_groundedness, evaluate_recall_at_k, load_golden_queries


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SET = PROJECT_ROOT / "evals" / "rag" / "golden_set.json"
FIXTURE_RESULTS = PROJECT_ROOT / "tests" / "fixtures" / "rag_eval_results.json"


def _fixture_payload():
    return json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))


def test_golden_rag_set_is_valid():
    golden = load_golden_queries(GOLDEN_SET)

    assert len(golden) >= 5
    assert all(query.expected_sources for query in golden)
    assert all(query.required_terms for query in golden)


def test_recall_at_k_passes_for_seed_fixture_results():
    golden = load_golden_queries(GOLDEN_SET)
    payload = _fixture_payload()

    result = evaluate_recall_at_k(golden, payload["retrieved"], k=3)

    assert result.score == 1.0
    assert result.hits == result.total
    assert result.misses == ()


def test_recall_at_k_reports_missed_expected_source():
    golden = load_golden_queries(GOLDEN_SET)
    payload = _fixture_payload()
    payload["retrieved"]["rag-phoenix-monitoring"] = [
        "This mentions tracing but points at the wrong document.\n[Source: docs/architecture/S18_WORKFLOW_AGNOSTIC_TARGET.md p1]"
    ]

    result = evaluate_recall_at_k(golden, payload["retrieved"], k=1)

    assert result.score < 1.0
    assert len(result.misses) == 1
    assert result.misses[0].query_id == "rag-phoenix-monitoring"
    assert result.misses[0].expected_sources == ("docs/monitoring/phoenix.md",)


def test_groundedness_accepts_answer_with_retrieved_citation_and_supported_claims():
    payload = _fixture_payload()
    answer = payload["answers"]["rag-phoenix-monitoring"]
    retrieved = payload["retrieved"]["rag-phoenix-monitoring"]

    result = evaluate_groundedness(answer, retrieved)

    assert result.is_grounded
    assert result.unsupported_citations == ()
    assert result.unsupported_sentences == ()


def test_groundedness_flags_unsupported_citation_and_claim():
    payload = _fixture_payload()
    retrieved = payload["retrieved"]["rag-phoenix-monitoring"]
    answer = (
        "S18 monitoring uses Kafka lag, payment retries, and proprietary APM sampling as its main RAG signal. "
        "[Source: docs/architecture/GBRAIN_COMPATIBILITY.md p1]"
    )

    result = evaluate_groundedness(answer, retrieved)

    assert not result.is_grounded
    assert result.unsupported_citations == ("docs/architecture/gbrain_compatibility.md",)
    assert result.unsupported_sentences == (
        "S18 monitoring uses Kafka lag, payment retries, and proprietary APM sampling as its main RAG signal.",
    )
