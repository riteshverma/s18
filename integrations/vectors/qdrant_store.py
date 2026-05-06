"""Qdrant vector backend for local and self-hosted deployments."""

from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional

from integrations.vectors.base import Chunk, SearchHit, VectorStore


class QdrantVectorStore(VectorStore):
    provider = "qdrant"

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        api_key: str = "",
        distance: str = "Cosine",
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.6,
        upsert_batch_size: int = 128,
    ) -> None:
        if not url:
            raise ValueError("QdrantVectorStore requires url")
        if not collection:
            raise ValueError("QdrantVectorStore requires collection")
        self.url = url.rstrip("/")
        self.collection = collection
        self.api_key = api_key
        self.distance = distance
        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_backoff_seconds = max(0.05, float(retry_backoff_seconds))
        self._upsert_batch_size = max(1, int(upsert_batch_size))
        self._client = None

    def _call_with_retry(self, fn):
        last_error: Optional[Exception] = None
        for idx in range(self._retry_attempts):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover
                last_error = exc
                text = str(exc).lower()
                retryable = any(
                    token in text
                    for token in [
                        "timeout",
                        "tempor",
                        "throttl",
                        "503",
                        "502",
                        "429",
                        "connection",
                        "unavailable",
                    ]
                )
                if idx >= self._retry_attempts - 1 or not retryable:
                    raise
                sleep_for = self._retry_backoff_seconds * (2 ** idx) + random.uniform(0, 0.1)
                time.sleep(min(sleep_for, 4.0))
        raise RuntimeError(f"Qdrant operation failed: {last_error}") from last_error

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "qdrant-client must be installed for the qdrant backend."
            ) from exc
        self._client = QdrantClient(
            url=self.url,
            api_key=self.api_key or None,
            timeout=30.0,
        )
        return self._client

    def ensure_index(self, *, dimension: int, metric: str = "cosine") -> None:
        from qdrant_client.http import models as qm  # type: ignore[reportMissingImports]

        client = self._client_lazy()
        if self._call_with_retry(lambda: client.collection_exists(self.collection)):
            existing = self._call_with_retry(lambda: client.get_collection(self.collection))
            vectors_cfg = getattr(getattr(existing, "config", None), "params", None)
            vectors_cfg = getattr(vectors_cfg, "vectors", None)
            size = getattr(vectors_cfg, "size", None)
            if size is not None and int(size) != int(dimension):
                raise ValueError(
                    f"Qdrant collection dimension mismatch (existing={size}, requested={dimension})"
                )
            return

        dist = _to_qdrant_distance(self.distance or metric)
        self._call_with_retry(
            lambda: client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=int(dimension), distance=dist),
            )
        )

    def upsert(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        from qdrant_client.http import models as qm  # type: ignore[reportMissingImports]

        client = self._client_lazy()
        points: List[Any] = []
        for chunk in chunks:
            payload = _chunk_payload(chunk)
            points.append(
                qm.PointStruct(
                    id=_point_id_for_chunk(chunk.chunk_id),
                    vector=list(chunk.embedding),
                    payload=payload,
                )
            )
        for i in range(0, len(points), self._upsert_batch_size):
            batch = points[i : i + self._upsert_batch_size]
            self._call_with_retry(
                lambda b=batch: client.upsert(
                    collection_name=self.collection,
                    points=b,
                    wait=False,
                )
            )
        return len(chunks)

    def query(
        self,
        *,
        embedding: Optional[List[float]] = None,
        text: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        if embedding is None:
            return []
        client = self._client_lazy()
        query_filter = _to_qdrant_filter(filters or {})
        if hasattr(client, "search"):
            scored = self._call_with_retry(
                lambda: client.search(
                    collection_name=self.collection,
                    query_vector=list(embedding),
                    query_filter=query_filter,
                    limit=int(k),
                    with_payload=True,
                )
            )
        else:
            query_result = self._call_with_retry(
                lambda: client.query_points(
                    collection_name=self.collection,
                    query=list(embedding),
                    query_filter=query_filter,
                    limit=int(k),
                    with_payload=True,
                )
            )
            scored = getattr(query_result, "points", None) or []
        hits: List[SearchHit] = []
        for item in scored:
            payload = item.payload or {}
            metadata = payload.get("metadata") if isinstance(payload, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            if "page" in payload and "page" not in metadata:
                metadata["page"] = payload.get("page")
            if "doc" in payload and "doc" not in metadata:
                metadata["doc"] = payload.get("doc")
            hits.append(
                SearchHit(
                    chunk_id=str(payload.get("chunk_id") or item.id),
                    doc_id=str(payload.get("doc_id") or ""),
                    text=str(payload.get("chunk") or payload.get("text") or ""),
                    score=float(getattr(item, "score", 0.0) or 0.0),
                    source_uri=str(payload.get("source_uri") or payload.get("doc") or ""),
                    metadata=metadata,
                )
            )
        return hits

    def delete_by_doc(self, doc_id: str, *, tenant_id: Optional[str] = None) -> int:
        from qdrant_client.http import models as qm  # type: ignore[reportMissingImports]

        client = self._client_lazy()
        conditions: List[Any] = []
        if tenant_id:
            conditions.append(
                qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(tenant_id)))
            )

        conditions.append(
            qm.Filter(
                should=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=str(doc_id))),
                    qm.FieldCondition(key="doc", match=qm.MatchValue(value=str(doc_id))),
                ]
            )
        )
        selector = qm.Filter(must=conditions)
        points, _ = self._call_with_retry(
            lambda: client.scroll(
                collection_name=self.collection,
                scroll_filter=selector,
                limit=10_000,
                with_payload=False,
                with_vectors=False,
            )
        )
        if not points:
            return 0
        self._call_with_retry(
            lambda: client.delete(
                collection_name=self.collection,
                points_selector=selector,
                wait=True,
            )
        )
        return len(points)

    def stats(self) -> Dict[str, Any]:
        client = self._client_lazy()
        count = None
        try:
            counted = self._call_with_retry(
                lambda: client.count(collection_name=self.collection, exact=False)
            )
            count = getattr(counted, "count", None)
        except Exception:
            count = None
        return {
            "provider": self.provider,
            "collection": self.collection,
            "url": self.url,
            "document_count": count,
        }

    def export_metadata(self, *, tenant_id: Optional[str] = None, limit: int = 10_000) -> List[Dict[str, Any]]:
        client = self._client_lazy()
        scroll_filter = _to_qdrant_filter({"tenant_id": tenant_id} if tenant_id else {})
        records: List[Dict[str, Any]] = []
        offset = None
        remaining = max(1, int(limit))
        while remaining > 0:
            page_size = min(256, remaining)
            points, next_offset = self._call_with_retry(
                lambda: client.scroll(
                    collection_name=self.collection,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                    with_vectors=False,
                    limit=page_size,
                    offset=offset,
                )
            )
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") if isinstance(payload, dict) else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                records.append(
                    {
                        "chunk_id": str(payload.get("chunk_id") or point.id),
                        "chunk": str(payload.get("chunk") or payload.get("text") or ""),
                        "doc": str(payload.get("doc") or payload.get("doc_id") or ""),
                        "doc_id": str(payload.get("doc_id") or payload.get("doc") or ""),
                        "tenant_id": str(payload.get("tenant_id") or metadata.get("tenant_id") or ""),
                        "integration_id": str(payload.get("integration_id") or metadata.get("integration_id") or ""),
                        "source_uri": str(payload.get("source_uri") or ""),
                        "page": payload.get("page") if payload.get("page") is not None else metadata.get("page"),
                        **metadata,
                    }
                )
            remaining -= len(points)
            if next_offset is None:
                break
            offset = next_offset
        return records


def _to_qdrant_distance(metric: str):
    from qdrant_client.http import models as qm  # type: ignore[reportMissingImports]

    normalized = (metric or "cosine").strip().lower()
    if normalized in {"cosine", "cosinesim", "cosinesimil"}:
        return qm.Distance.COSINE
    if normalized in {"ip", "innerproduct", "dot"}:
        return qm.Distance.DOT
    if normalized in {"l2", "euclid", "euclidean"}:
        return qm.Distance.EUCLID
    return qm.Distance.COSINE


def _to_qdrant_filter(filters: Dict[str, Any]):
    if not filters:
        return None
    from qdrant_client.http import models as qm  # type: ignore[reportMissingImports]

    conditions = []
    for key, expected in filters.items():
        if expected is None:
            continue
        if isinstance(expected, (list, tuple, set)):
            values = [value for value in expected if value is not None]
            if not values:
                continue
            conditions.append(
                qm.FieldCondition(key=key, match=qm.MatchAny(any=[str(v) for v in values]))
            )
        else:
            conditions.append(
                qm.FieldCondition(key=key, match=qm.MatchValue(value=str(expected)))
            )
    if not conditions:
        return None
    return qm.Filter(must=conditions)


def _chunk_payload(chunk: Chunk) -> Dict[str, Any]:
    metadata = dict(chunk.metadata or {})
    doc_rel = str(metadata.get("doc") or chunk.doc_id)
    page = metadata.get("page")
    payload: Dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "chunk": chunk.text,
        "text": chunk.text,
        "doc_id": chunk.doc_id,
        "doc": doc_rel,
        "tenant_id": chunk.tenant_id,
        "integration_id": chunk.integration_id,
        "source_uri": chunk.source_uri or doc_rel,
        "metadata": metadata,
    }
    if page is not None:
        payload["page"] = page
    return payload


def _point_id_for_chunk(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"s18share:{chunk_id}"))

