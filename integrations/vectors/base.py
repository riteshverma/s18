"""Vector store contract used by the ingest pipeline and the RAG router."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class Chunk:
    """A single text chunk + its embedding ready for upsert."""

    chunk_id: str
    doc_id: str
    text: str
    embedding: List[float]
    tenant_id: str = "default"
    integration_id: str = "default"
    source_uri: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source_uri: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    provider: str

    def ensure_index(
        self,
        *,
        dimension: int,
        metric: str = "cosine",
    ) -> None:
        ...

    def upsert(self, chunks: List[Chunk]) -> int:
        ...

    def query(
        self,
        *,
        embedding: Optional[List[float]] = None,
        text: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        ...

    def delete_by_doc(self, doc_id: str, *, tenant_id: Optional[str] = None) -> int:
        ...

    def stats(self) -> Dict[str, Any]:
        ...
