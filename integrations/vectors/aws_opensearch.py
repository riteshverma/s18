"""AWS OpenSearch / OpenSearch Serverless (AOSS) vector backend.

Mapping created on ``ensure_index``::

    chunk_id   keyword, _id
    doc_id     keyword
    tenant_id  keyword
    integration_id keyword
    source_uri keyword
    text       text
    embedding  knn_vector
    metadata   object
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from integrations.vectors.base import Chunk, SearchHit, VectorStore


class AwsOpenSearchVectorStore(VectorStore):
    provider = "aws_opensearch"

    def __init__(
        self,
        *,
        endpoint: str,
        region: str = "us-east-1",
        index_name: str,
        service: str = "aoss",
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.6,
        bulk_batch_size: int = 300,
    ) -> None:
        if not endpoint:
            raise ValueError("AwsOpenSearchVectorStore requires endpoint")
        self.endpoint = endpoint
        self.region = region
        self.index_name = index_name
        # `aoss` for serverless, `es` for managed OpenSearch.
        self.service = service
        self._client = None
        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_backoff_seconds = max(0.05, float(retry_backoff_seconds))
        self._bulk_batch_size = max(1, int(bulk_batch_size))

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
                    for token in ["timeout", "tempor", "throttl", "503", "502", "429", "connection", "unavailable"]
                )
                if idx >= self._retry_attempts - 1 or not retryable:
                    raise
                sleep_for = self._retry_backoff_seconds * (2 ** idx) + random.uniform(0, 0.1)
                time.sleep(min(sleep_for, 4.0))
        raise RuntimeError(f"OpenSearch operation failed: {last_error}") from last_error

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[reportMissingImports]
            from opensearchpy import OpenSearch, RequestsHttpConnection  # type: ignore[reportMissingImports]
            from requests_aws4auth import AWS4Auth  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "opensearch-py, boto3, and requests-aws4auth must be installed for "
                "the aws_opensearch backend."
            ) from exc

        if self.service not in {"aoss", "es"}:
            raise ValueError("aws_opensearch service must be 'aoss' or 'es'")

        creds = boto3.Session().get_credentials()
        if creds is None:
            raise RuntimeError("No AWS credentials were found for aws_opensearch backend.")
        auth = AWS4Auth(
            creds.access_key,
            creds.secret_key,
            self.region,
            self.service,
            session_token=creds.token,
        )
        host = self.endpoint.replace("https://", "").replace("http://", "")
        self._client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=self._retry_attempts,
            retry_on_timeout=True,
        )
        return self._client

    def ensure_index(self, *, dimension: int, metric: str = "cosine") -> None:
        client = self._client_lazy()
        if self._call_with_retry(lambda: client.indices.exists(index=self.index_name)):
            return
        space_type = {"cosine": "cosinesimil", "l2": "l2", "ip": "innerproduct"}.get(metric, "cosinesimil")
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "integration_id": {"type": "keyword"},
                    "source_uri": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": space_type,
                            "engine": "nmslib",
                        },
                    },
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        }
        self._call_with_retry(lambda: client.indices.create(index=self.index_name, body=body))

    def upsert(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        client = self._client_lazy()
        actions = []
        for chunk in chunks:
            actions.append({"index": {"_index": self.index_name, "_id": chunk.chunk_id}})
            actions.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "tenant_id": chunk.tenant_id,
                    "integration_id": chunk.integration_id,
                    "source_uri": chunk.source_uri,
                    "text": chunk.text,
                    "embedding": list(chunk.embedding),
                    "metadata": chunk.metadata or {},
                }
            )
        for i in range(0, len(actions), self._bulk_batch_size * 2):
            batch = actions[i : i + self._bulk_batch_size * 2]
            resp = self._call_with_retry(lambda b=batch: client.bulk(body=b, refresh=False))
            if resp.get("errors"):
                raise RuntimeError(f"OpenSearch bulk upsert reported errors: {resp}")
        return len(chunks)

    def query(
        self,
        *,
        embedding: Optional[List[float]] = None,
        text: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        client = self._client_lazy()
        must: List[Dict[str, Any]] = []
        if text:
            must.append({"match": {"text": text}})
        if embedding is not None:
            must.append({"knn": {"embedding": {"vector": list(embedding), "k": k}}})

        filter_clauses: List[Dict[str, Any]] = []
        for key, val in (filters or {}).items():
            if isinstance(val, (list, tuple, set)):
                filter_clauses.append({"terms": {key: list(val)}})
            else:
                filter_clauses.append({"term": {key: val}})

        body: Dict[str, Any] = {
            "size": k,
            "query": {"bool": {"must": must, "filter": filter_clauses}},
        }
        resp = self._call_with_retry(lambda: client.search(index=self.index_name, body=body))
        hits: List[SearchHit] = []
        for h in resp.get("hits", {}).get("hits", []):
            src = h.get("_source", {})
            hits.append(
                SearchHit(
                    chunk_id=src.get("chunk_id", h.get("_id", "")),
                    doc_id=src.get("doc_id", ""),
                    text=src.get("text", ""),
                    score=float(h.get("_score") or 0.0),
                    source_uri=src.get("source_uri", ""),
                    metadata=src.get("metadata") or {},
                )
            )
        return hits

    def delete_by_doc(self, doc_id: str, *, tenant_id: Optional[str] = None) -> int:
        client = self._client_lazy()
        filters: List[Dict[str, Any]] = [{"term": {"doc_id": doc_id}}]
        if tenant_id:
            filters.append({"term": {"tenant_id": tenant_id}})
        body = {"query": {"bool": {"filter": filters}}}
        resp = self._call_with_retry(
            lambda: client.delete_by_query(index=self.index_name, body=body)
        )
        return int(resp.get("deleted") or 0)

    def stats(self) -> Dict[str, Any]:
        client = self._client_lazy()
        try:
            count = self._call_with_retry(
                lambda: client.count(index=self.index_name).get("count")
            )
        except Exception:
            count = None
        return {
            "provider": self.provider,
            "index_name": self.index_name,
            "endpoint": self.endpoint,
            "document_count": count,
        }
