"""Evaluate captured S18 RAG results against a golden set.

Usage:
    python scripts/evaluate_rag.py --results tests/fixtures/rag_eval_results.json

The results JSON should contain:
    {
      "retrieved": {"query-id": ["chunk text\\n[Source: path p1]"]},
      "answers": {"query-id": "answer text with [Source: path p1]"}
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag_eval import evaluate_groundedness, evaluate_recall_at_k, load_golden_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and groundedness from captured results.")
    parser.add_argument("--golden", default="evals/rag/golden_set.json", help="Path to golden set JSON.")
    parser.add_argument("--results", required=True, help="Path to captured RAG results JSON.")
    parser.add_argument("--k", type=int, default=5, help="Recall@k cutoff.")
    parser.add_argument("--min-recall", type=float, default=0.8, help="Minimum passing Recall@k score.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    golden = load_golden_queries(PROJECT_ROOT / args.golden)
    payload = json.loads((PROJECT_ROOT / args.results).read_text(encoding="utf-8"))
    retrieved = payload.get("retrieved", {})
    answers = payload.get("answers", {})

    recall = evaluate_recall_at_k(golden, retrieved, k=args.k)
    groundedness = {
        query_id: evaluate_groundedness(answer, retrieved.get(query_id, ()))
        for query_id, answer in answers.items()
    }

    ungrounded = {
        query_id: result
        for query_id, result in groundedness.items()
        if not result.is_grounded
    }
    passed = recall.score >= args.min_recall and not ungrounded

    report = {
        "passed": passed,
        "recall_at_k": {
            "k": recall.k,
            "score": recall.score,
            "hits": recall.hits,
            "total": recall.total,
            "misses": [
                {
                    "query_id": miss.query_id,
                    "expected_sources": list(miss.expected_sources),
                    "retrieved_sources": list(miss.retrieved_sources),
                }
                for miss in recall.misses
            ],
        },
        "groundedness": {
            query_id: {
                "is_grounded": result.is_grounded,
                "unsupported_citations": list(result.unsupported_citations),
                "unsupported_sentences": list(result.unsupported_sentences),
            }
            for query_id, result in groundedness.items()
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Recall@{recall.k}: {recall.score:.2%} ({recall.hits}/{recall.total})")
        if recall.misses:
            print("Misses:")
            for miss in recall.misses:
                print(f"- {miss.query_id}: expected {', '.join(miss.expected_sources)}")
        if groundedness:
            grounded_count = len(groundedness) - len(ungrounded)
            print(f"Grounded answers: {grounded_count}/{len(groundedness)}")
        if ungrounded:
            print("Ungrounded answers:")
            for query_id, result in ungrounded.items():
                print(f"- {query_id}: {len(result.unsupported_sentences)} unsupported sentences")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
