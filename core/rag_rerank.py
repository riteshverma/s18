"""Stage-2 RAG reranker abstractions.

Phase 1 introduces configuration parsing and a provider abstraction only.
Execution wiring into retrieval paths is handled in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Mapping, Protocol, Sequence

from config.settings_loader import load_settings


logger = logging.getLogger(__name__)


_DEFAULT_PROVIDER = "local_oss"
_DEFAULT_CANDIDATE_K = 40
_DEFAULT_TIMEOUT_SECONDS = 8.0
_DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class RerankConfig:
    """Runtime configuration for stage-2 reranking."""

    enabled: bool = False
    provider: str = _DEFAULT_PROVIDER
    model: str = ""
    candidate_k: int = _DEFAULT_CANDIDATE_K
    top_k: int | None = None
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    batch_size: int = _DEFAULT_BATCH_SIZE


@dataclass(frozen=True)
class RerankCandidate:
    """Single candidate item passed to the reranker."""

    candidate_id: str
    text: str
    base_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    """Reranked candidate output with final score."""

    candidate: RerankCandidate
    score: float


class Reranker(Protocol):
    """Provider contract for stage-2 rerankers."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        ...


class NoopReranker:
    """Fallback reranker used until concrete provider integrations are added."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query

        results = [
            RerankResult(candidate=candidate, score=float(candidate.base_score or 0.0))
            for candidate in candidates
        ]
        if top_k is not None and top_k > 0:
            return results[:top_k]
        return results


class LocalHeuristicReranker:
    """Lightweight lexical reranker for local/offline environments."""

    _TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
    _STOPWORDS = {
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
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        query_terms = self._extract_terms(query)
        normalized_query = self._normalize_text(query)
        indexed = list(enumerate(candidates))

        scored: list[tuple[int, RerankResult]] = []
        for original_index, candidate in indexed:
            normalized_text = self._normalize_text(candidate.text)
            text_terms = self._extract_terms(candidate.text)
            overlap_hits = len(query_terms.intersection(text_terms))
            overlap_ratio = (overlap_hits / len(query_terms)) if query_terms else 0.0

            phrase_bonus = 0.0
            if normalized_query and len(normalized_query) >= 10 and normalized_query in normalized_text:
                phrase_bonus = 0.35

            score = overlap_ratio + phrase_bonus
            scored.append((original_index, RerankResult(candidate=candidate, score=score)))

        ranked = sorted(
            scored,
            key=lambda item: (
                -item[1].score,
                -(item[1].candidate.base_score if item[1].candidate.base_score is not None else float("-inf")),
                item[0],
            ),
        )
        results = [result for _, result in ranked]
        if top_k is not None and top_k > 0:
            return results[:top_k]
        return results

    def _extract_terms(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in self._TOKEN_PATTERN.findall(text or "")
            if len(token) > 2 and token.lower() not in self._STOPWORDS
        }

    def _normalize_text(self, text: str) -> str:
        cleaned = " ".join((text or "").strip().lower().split())
        return re.sub(r"\s+", " ", cleaned).strip()


def load_rerank_config(runtime_settings: Mapping[str, Any] | None = None) -> RerankConfig:
    """Load reranker config from settings with safe defaults."""

    source = runtime_settings or load_settings()
    rag_settings = source.get("rag", {}) if isinstance(source, Mapping) else {}
    rerank_raw = rag_settings.get("rerank", {}) if isinstance(rag_settings, Mapping) else {}
    if not isinstance(rerank_raw, Mapping):
        rerank_raw = {}

    return RerankConfig(
        enabled=bool(rerank_raw.get("enabled", False)),
        provider=_normalize_provider(rerank_raw.get("provider")),
        model=str(rerank_raw.get("model", "") or "").strip(),
        candidate_k=_to_positive_int(rerank_raw.get("candidate_k"), _DEFAULT_CANDIDATE_K),
        top_k=_to_optional_positive_int(rerank_raw.get("top_k")),
        timeout_seconds=_to_positive_float(
            rerank_raw.get("timeout_seconds"),
            _DEFAULT_TIMEOUT_SECONDS,
        ),
        batch_size=_to_positive_int(rerank_raw.get("batch_size"), _DEFAULT_BATCH_SIZE),
    )


def build_reranker(config: RerankConfig | None = None) -> Reranker:
    """Create a reranker provider instance.

    Phase 1 intentionally returns NoopReranker for all providers.
    """

    active = config or load_rerank_config()
    provider = _normalize_provider(active.provider)
    if provider == "local_oss":
        return LocalHeuristicReranker()
    if provider == "noop":
        return NoopReranker()
    if provider not in {"local_oss", "noop"}:
        logger.warning("Unknown rerank provider '%s'; using noop provider.", provider)
    return NoopReranker()


def _normalize_provider(raw_provider: Any) -> str:
    provider = str(raw_provider or _DEFAULT_PROVIDER).strip().lower()
    if provider in {"none", "disabled", "off"}:
        return "noop"
    return provider or _DEFAULT_PROVIDER


def _to_positive_int(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _to_optional_positive_int(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _to_positive_float(raw: Any, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default
