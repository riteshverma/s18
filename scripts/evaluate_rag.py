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
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag_eval import evaluate_groundedness, evaluate_retrieval_at_k, load_golden_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and groundedness from captured results.")
    parser.add_argument("--golden", default="evals/rag/golden_set.json", help="Path to golden set JSON.")
    parser.add_argument("--results", required=True, help="Path to captured RAG results JSON.")
    parser.add_argument("--k", type=int, default=5, help="Recall@k cutoff.")
    parser.add_argument("--min-recall", type=float, default=0.8, help="Minimum passing Recall@k score.")
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional path for compact per-query retrieval CSV export.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def _resolve_output_path(path_arg: str) -> Path:
    candidate = Path(path_arg)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def write_compact_csv(output_path: Path, retrieval) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "is_hit",
                "first_relevant_rank",
                "reciprocal_rank",
                "recall_at_k",
                "precision_at_k",
            ],
        )
        writer.writeheader()
        for item in retrieval.query_metrics:
            writer.writerow(
                {
                    "query_id": item.query_id,
                    "is_hit": int(item.is_hit),
                    "first_relevant_rank": item.first_relevant_rank if item.first_relevant_rank is not None else "",
                    "reciprocal_rank": f"{item.reciprocal_rank:.6f}",
                    "recall_at_k": f"{item.recall_at_k:.6f}",
                    "precision_at_k": f"{item.precision_at_k:.6f}",
                }
            )


def main() -> int:
    args = parse_args()
    golden = load_golden_queries(PROJECT_ROOT / args.golden)
    payload = json.loads((PROJECT_ROOT / args.results).read_text(encoding="utf-8"))
    retrieved = payload.get("retrieved", {})
    answers = payload.get("answers", {})

    retrieval = evaluate_retrieval_at_k(golden, retrieved, k=args.k)
    groundedness = {
        query_id: evaluate_groundedness(answer, retrieved.get(query_id, ()))
        for query_id, answer in answers.items()
    }

    ungrounded = {
        query_id: result
        for query_id, result in groundedness.items()
        if not result.is_grounded
    }
    passed = retrieval.recall_at_k >= args.min_recall and not ungrounded
    csv_output = None
    if args.out_csv:
        csv_output = _resolve_output_path(args.out_csv)
        write_compact_csv(csv_output, retrieval)

    report = {
        "passed": passed,
        "recall_at_k": {
            "k": retrieval.k,
            "score": retrieval.recall_at_k,
            "misses": [
                {
                    "query_id": miss.query_id,
                    "expected_sources": list(miss.expected_sources),
                    "retrieved_sources": list(miss.retrieved_sources),
                }
                for miss in retrieval.misses
            ],
        },
        "precision_at_k": {
            "k": retrieval.k,
            "score": retrieval.precision_at_k,
        },
        "hit_rate_at_k": {
            "k": retrieval.k,
            "score": retrieval.hit_rate,
            "hits": retrieval.hit_count,
            "total": retrieval.total_queries,
        },
        "mrr": retrieval.mrr,
        "retrieval_metrics": {
            "k": retrieval.k,
            "recall_at_k": retrieval.recall_at_k,
            "precision_at_k": retrieval.precision_at_k,
            "hit_rate_at_k": retrieval.hit_rate,
            "mrr": retrieval.mrr,
            "per_query": [
                {
                    "query_id": item.query_id,
                    "question": item.question,
                    "is_hit": item.is_hit,
                    "expected_sources": list(item.expected_sources),
                    "retrieved_sources": list(item.retrieved_sources),
                    "matched_sources": list(item.matched_sources),
                    "first_relevant_rank": item.first_relevant_rank,
                    "reciprocal_rank": item.reciprocal_rank,
                    "recall_at_k": item.recall_at_k,
                    "precision_at_k": item.precision_at_k,
                }
                for item in retrieval.query_metrics
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
    if csv_output:
        report["compact_csv"] = str(csv_output)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Recall@{retrieval.k}: {retrieval.recall_at_k:.2%}")
        print(f"Precision@{retrieval.k}: {retrieval.precision_at_k:.2%}")
        print(f"HitRate@{retrieval.k}: {retrieval.hit_rate:.2%} ({retrieval.hit_count}/{retrieval.total_queries})")
        print(f"MRR: {retrieval.mrr:.4f}")
        if retrieval.misses:
            print("Misses:")
            for miss in retrieval.misses:
                print(f"- {miss.query_id}: expected {', '.join(miss.expected_sources)}")
        if groundedness:
            grounded_count = len(groundedness) - len(ungrounded)
            print(f"Grounded answers: {grounded_count}/{len(groundedness)}")
        if ungrounded:
            print("Ungrounded answers:")
            for query_id, result in ungrounded.items():
                print(f"- {query_id}: {len(result.unsupported_sentences)} unsupported sentences")
        if csv_output:
            print(f"CSV export: {csv_output}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
