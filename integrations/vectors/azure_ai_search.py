"""Azure AI Search vector backend.

Schema (created on `ensure_index`)::

    chunk_id     Edm.String  key, filterable
    doc_id       Edm.String  filterable, facetable
    tenant_id    Edm.String  filterable
    integration_id Edm.String filterable
    source_uri   Edm.String  retrievable
    text         Edm.String  searchable, retrievable
    embedding    Collection(Edm.Single) vector field, HNSW
    metadata     Edm.String  retrievable JSON-encoded extras
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Optional

from integrations.vectors.base import Chunk, SearchHit, VectorStore


class AzureAiSearchVectorStore(VectorStore):
    provider = "azure_ai_search"

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        index_name: str,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.6,
        upsert_batch_size: int = 500,
    ) -> None:
        if not endpoint:
            raise ValueError("AzureAiSearchVectorStore requires endpoint")
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.index_name = index_name
        self._search = None
        self._mgmt = None
        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_backoff_seconds = max(0.05, float(retry_backoff_seconds))
        self._upsert_batch_size = max(1, int(upsert_batch_size))

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
                    for token in ["timeout", "tempor", "throttl", "503", "502", "429", "connection"]
                )
                if idx >= self._retry_attempts - 1 or not retryable:
                    raise
                sleep_for = self._retry_backoff_seconds * (2 ** idx) + random.uniform(0, 0.1)
                time.sleep(min(sleep_for, 4.0))
        raise RuntimeError(f"Azure AI Search operation failed: {last_error}") from last_error

    def _credential(self):
        try:
            from azure.core.credentials import AzureKeyCredential  # type: ignore[reportMissingImports]
            from azure.identity import DefaultAzureCredential  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "azure-search-documents and azure-identity must be installed for "
                "the azure_ai_search backend."
            ) from exc
        return AzureKeyCredential(self.api_key) if self.api_key else DefaultAzureCredential()

    def _search_client(self):
        if self._search is not None:
            return self._search
        try:
            from azure.search.documents import SearchClient  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "azure-search-documents must be installed: `pip install azure-search-documents`"
            ) from exc
        self._search = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self._credential(),
        )
        return self._search

    def _mgmt_client(self):
        if self._mgmt is not None:
            return self._mgmt
        try:
            from azure.search.documents.indexes import SearchIndexClient  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "azure-search-documents must be installed: `pip install azure-search-documents`"
            ) from exc
        self._mgmt = SearchIndexClient(
            endpoint=self.endpoint, credential=self._credential()
        )
        return self._mgmt

    def ensure_index(self, *, dimension: int, metric: str = "cosine") -> None:
        from azure.search.documents.indexes.models import (  # type: ignore[reportMissingImports]
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        client = self._mgmt_client()
        existing = [idx.name for idx in client.list_indexes()]
        if self.index_name in existing:
            return

        algo_name = "hnsw-default"
        profile_name = "vector-profile"
        index = SearchIndex(
            name=self.index_name,
            fields=[
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True),
                SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="tenant_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="integration_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="source_uri", type=SearchFieldDataType.String, retrievable=True),
                SearchableField(name="text", type=SearchFieldDataType.String),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=dimension,
                    vector_search_profile_name=profile_name,
                ),
                SimpleField(name="metadata", type=SearchFieldDataType.String, retrievable=True),
            ],
            vector_search=VectorSearch(
                profiles=[VectorSearchProfile(name=profile_name, algorithm_configuration_name=algo_name)],
                algorithms=[HnswAlgorithmConfiguration(name=algo_name, parameters={"metric": metric})],
            ),
        )
        self._call_with_retry(lambda: client.create_index(index))

    def upsert(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        client = self._search_client()
        docs = []
        for chunk in chunks:
            docs.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "tenant_id": chunk.tenant_id,
                    "integration_id": chunk.integration_id,
                    "source_uri": chunk.source_uri,
                    "text": chunk.text,
                    "embedding": list(chunk.embedding),
                    "metadata": json.dumps(chunk.metadata or {}),
                }
            )
        for i in range(0, len(docs), self._upsert_batch_size):
            batch = docs[i : i + self._upsert_batch_size]
            self._call_with_retry(lambda b=batch: client.upload_documents(documents=b))
        return len(docs)

    def query(
        self,
        *,
        embedding: Optional[List[float]] = None,
        text: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        from azure.search.documents.models import VectorizedQuery  # type: ignore[reportMissingImports]

        client = self._search_client()
        filter_str = _filters_to_odata(filters or {})

        kwargs: Dict[str, Any] = {"top": k, "filter": filter_str}
        if text:
            kwargs["search_text"] = text
        if embedding is not None:
            kwargs["vector_queries"] = [
                VectorizedQuery(vector=list(embedding), k_nearest_neighbors=k, fields="embedding")
            ]
        results = self._call_with_retry(
            lambda: client.search(**{key: val for key, val in kwargs.items() if val is not None})
        )

        hits: List[SearchHit] = []
        for r in results:
            md_raw = r.get("metadata") or "{}"
            try:
                md = json.loads(md_raw)
            except Exception:
                md = {}
            hits.append(
                SearchHit(
                    chunk_id=r.get("chunk_id", ""),
                    doc_id=r.get("doc_id", ""),
                    text=r.get("text", ""),
                    score=float(r.get("@search.score", 0.0)),
                    source_uri=r.get("source_uri", ""),
                    metadata=md,
                )
            )
        return hits

    def delete_by_doc(self, doc_id: str, *, tenant_id: Optional[str] = None) -> int:
        client = self._search_client()
        flt = f"doc_id eq '{_odata_escape(doc_id)}'"
        if tenant_id:
            flt += f" and tenant_id eq '{_odata_escape(tenant_id)}'"
        rows = list(
            self._call_with_retry(
                lambda: client.search(search_text="*", filter=flt, select=["chunk_id"])
            )
        )
        if not rows:
            return 0
        self._call_with_retry(
            lambda: client.delete_documents(documents=[{"chunk_id": r["chunk_id"]} for r in rows])
        )
        return len(rows)

    def stats(self) -> Dict[str, Any]:
        client = self._mgmt_client()
        try:
            stats = self._call_with_retry(lambda: client.get_index_statistics(self.index_name))
        except Exception:
            stats = {}
        return {
            "provider": self.provider,
            "index_name": self.index_name,
            "endpoint": self.endpoint,
            "document_count": stats.get("document_count") if isinstance(stats, dict) else None,
        }


def _odata_escape(value: str) -> str:
    return value.replace("'", "''")


def _filters_to_odata(filters: Dict[str, Any]) -> Optional[str]:
    if not filters:
        return None
    parts: List[str] = []
    for key, val in filters.items():
        if isinstance(val, (list, tuple, set)):
            ors = " or ".join(f"{key} eq '{_odata_escape(str(v))}'" for v in val)
            if ors:
                parts.append(f"({ors})")
        else:
            parts.append(f"{key} eq '{_odata_escape(str(val))}'")
    return " and ".join(parts) if parts else None
