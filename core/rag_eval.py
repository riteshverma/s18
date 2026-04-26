"""Deterministic RAG evaluation helpers.

These utilities intentionally avoid model calls so they can run in CI. They
score retriever outputs and answer grounding from captured RAG results.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_PATTERN = re.compile(r"\[Source:\s*(?P<source>.+?)(?:\s+p(?P<page>\d+))?\]", re.IGNORECASE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    question: str
    expected_sources: tuple[str, ...]
    required_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecallMiss:
    query_id: str
    question: str
    expected_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...]


@dataclass(frozen=True)
class RecallAtKResult:
    k: int
    total: int
    hits: int
    score: float
    misses: tuple[RecallMiss, ...]


@dataclass(frozen=True)
class GroundednessResult:
    cited_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    unsupported_citations: tuple[str, ...]
    supported_sentence_ratio: float
    unsupported_sentences: tuple[str, ...]

    @property
    def has_citation_support(self) -> bool:
        return bool(self.cited_sources) and not self.unsupported_citations

    @property
    def is_grounded(self) -> bool:
        return self.has_citation_support and not self.unsupported_sentences


def load_golden_queries(path: str | Path) -> list[GoldenQuery]:
    """Load and validate a RAG golden set JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw.get("queries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("RAG golden set must contain a non-empty 'queries' list")

    golden: list[GoldenQuery] = []
    seen_ids: set[str] = set()
    for row in rows:
        query_id = str(row.get("id", "")).strip()
        question = str(row.get("question", "")).strip()
        expected_sources = tuple(
            normalize_source(source) for source in row.get("expected_sources", []) if str(source).strip()
        )
        required_terms = tuple(str(term).strip().lower() for term in row.get("required_terms", []) if str(term).strip())

        if not query_id:
            raise ValueError("Each golden query must include an id")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate golden query id: {query_id}")
        if not question:
            raise ValueError(f"Golden query {query_id} must include a question")
        if not expected_sources:
            raise ValueError(f"Golden query {query_id} must include expected_sources")

        seen_ids.add(query_id)
        golden.append(
            GoldenQuery(
                id=query_id,
                question=question,
                expected_sources=expected_sources,
                required_terms=required_terms,
            )
        )

    return golden


def normalize_source(source: str) -> str:
    """Normalize source paths across Windows, POSIX, and data-root prefixes."""

    normalized = str(source).strip().replace("\\", "/").lower()
    normalized = re.sub(r"\s+p\d+$", "", normalized)
    normalized = normalized.removeprefix("./").removeprefix("data/")
    return normalized.strip("/")


def extract_sources(value: Any) -> tuple[str, ...]:
    """Extract source ids from a RAG result or answer string."""

    if isinstance(value, Mapping):
        for key in ("source", "source_id", "doc", "path"):
            source = value.get(key)
            if source:
                return (normalize_source(str(source)),)
        value = value.get("text") or value.get("chunk") or value.get("content") or ""

    text = str(value)
    sources = tuple(normalize_source(match.group("source")) for match in SOURCE_PATTERN.finditer(text))
    return tuple(source for source in sources if source)


def result_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("text", "chunk", "content", "answer"):
            if value.get(key):
                return str(value[key])
    return str(value)


def evaluate_recall_at_k(
    golden_queries: Sequence[GoldenQuery],
    retrieved_by_query_id: Mapping[str, Sequence[Any]],
    *,
    k: int = 5,
) -> RecallAtKResult:
    """Compute source Recall@k where any expected source is a hit."""

    if k <= 0:
        raise ValueError("k must be positive")

    hits = 0
    misses: list[RecallMiss] = []

    for query in golden_queries:
        retrieved_sources = _sources_for_top_k(retrieved_by_query_id.get(query.id, ()), k)
        expected = set(query.expected_sources)
        if expected.intersection(retrieved_sources):
            hits += 1
            continue

        misses.append(
            RecallMiss(
                query_id=query.id,
                question=query.question,
                expected_sources=query.expected_sources,
                retrieved_sources=tuple(retrieved_sources),
            )
        )

    total = len(golden_queries)
    return RecallAtKResult(
        k=k,
        total=total,
        hits=hits,
        score=(hits / total) if total else 0.0,
        misses=tuple(misses),
    )


def evaluate_groundedness(
    answer: str,
    retrieved_results: Sequence[Any],
    *,
    min_sentence_support: float = 0.45,
    min_shared_terms: int = 3,
) -> GroundednessResult:
    """Check that cited sources are retrieved and answer sentences are supported.

    This is a lightweight CI gate, not a substitute for human or LLM-as-judge
    evals. It catches common regressions: missing citations, citations outside
    retrieved context, and sentences with little overlap against the context.
    """

    retrieved_sources = tuple(dict.fromkeys(source for item in retrieved_results for source in extract_sources(item)))
    cited_sources = tuple(dict.fromkeys(extract_sources(answer)))
    retrieved_source_set = set(retrieved_sources)
    unsupported_citations = tuple(source for source in cited_sources if source not in retrieved_source_set)

    context_tokens = _content_tokens(" ".join(result_text(item) for item in retrieved_results))
    unsupported_sentences: list[str] = []
    supported_sentences = 0
    checked_sentences = 0

    for sentence in _answer_sentences(answer):
        sentence_tokens = _content_tokens(sentence)
        if len(sentence_tokens) < min_shared_terms:
            continue
        checked_sentences += 1
        shared = sentence_tokens.intersection(context_tokens)
        support_ratio = len(shared) / len(sentence_tokens)
        if len(shared) >= min_shared_terms or support_ratio >= min_sentence_support:
            supported_sentences += 1
        else:
            unsupported_sentences.append(sentence)

    supported_sentence_ratio = (supported_sentences / checked_sentences) if checked_sentences else 1.0
    return GroundednessResult(
        cited_sources=cited_sources,
        retrieved_sources=retrieved_sources,
        unsupported_citations=unsupported_citations,
        supported_sentence_ratio=supported_sentence_ratio,
        unsupported_sentences=tuple(unsupported_sentences),
    )


def _sources_for_top_k(results: Sequence[Any], k: int) -> set[str]:
    sources: set[str] = set()
    for item in results[:k]:
        sources.update(extract_sources(item))
    return sources


def _content_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text) if token.lower() not in STOPWORDS}


def _answer_sentences(answer: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(answer.strip()) if sentence.strip()]
