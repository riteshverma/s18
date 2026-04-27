"""Bedrock Knowledge Base read-side adapter.

Bedrock KB is fully managed: ingestion is driven by S3 events on the data
source bucket and Bedrock handles parsing/chunking/embedding internally. This
backend therefore implements ``query`` and ``stats`` against the runtime API
and treats ``upsert`` as a no-op (objects appear automatically once they land
in the configured S3 data source).

Use this backend when the tenant prefers fully-managed RAG and the ingest
pipeline only needs to upload bytes to S3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from integrations.vectors.base import Chunk, SearchHit, VectorStore


class BedrockKnowledgeBaseVectorStore(VectorStore):
    provider = "bedrock_kb"

    def __init__(
        self,
        *,
        kb_id: str,
        region: str = "us-east-1",
        data_source_id: Optional[str] = None,
    ) -> None:
        if not kb_id:
            raise ValueError("BedrockKnowledgeBaseVectorStore requires kb_id")
        self.kb_id = kb_id
        self.region = region
        self.data_source_id = data_source_id
        self._runtime = None
        self._agent = None

    def _runtime_client(self):
        if self._runtime is not None:
            return self._runtime
        try:
            import boto3  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "boto3 must be installed for the bedrock_kb backend."
            ) from exc
        self._runtime = boto3.client("bedrock-agent-runtime", region_name=self.region)
        return self._runtime

    def _agent_client(self):
        if self._agent is not None:
            return self._agent
        try:
            import boto3  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "boto3 must be installed for the bedrock_kb backend."
            ) from exc
        self._agent = boto3.client("bedrock-agent", region_name=self.region)
        return self._agent

    def ensure_index(self, *, dimension: int, metric: str = "cosine") -> None:
        # Knowledge bases are provisioned by IaC (Terraform / CloudFormation).
        # We expose a hook only to trigger an ingestion sync if the caller
        # already uploaded new objects to the S3 data source.
        del dimension, metric
        if not self.data_source_id:
            return
        try:
            self._agent_client().start_ingestion_job(
                knowledgeBaseId=self.kb_id, dataSourceId=self.data_source_id
            )
        except Exception:
            # Sync may already be running; non-fatal.
            pass

    def upsert(self, chunks: List[Chunk]) -> int:
        # Bedrock KB ingests directly from S3; the ingest pipeline writes the
        # bytes there, so explicit upsert is a no-op. We trigger a sync if a
        # data source is configured.
        del chunks
        self.ensure_index(dimension=0)
        return 0

    def query(
        self,
        *,
        embedding: Optional[List[float]] = None,
        text: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        if not text:
            raise ValueError("BedrockKnowledgeBaseVectorStore.query requires `text`")
        del embedding  # KB owns the embedding step

        client = self._runtime_client()
        params: Dict[str, Any] = {
            "knowledgeBaseId": self.kb_id,
            "retrievalQuery": {"text": text},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {"numberOfResults": k}
            },
        }
        # Best-effort metadata filter mapping; user should match Bedrock's schema.
        if filters:
            equals_clauses = [
                {"equals": {"key": k_, "value": v_}}
                for k_, v_ in filters.items()
                if not isinstance(v_, (list, tuple, set))
            ]
            if equals_clauses:
                params["retrievalConfiguration"]["vectorSearchConfiguration"][
                    "filter"
                ] = {"andAll": equals_clauses} if len(equals_clauses) > 1 else equals_clauses[0]

        resp = client.retrieve(**params)
        hits: List[SearchHit] = []
        for r in resp.get("retrievalResults", []) or []:
            content = r.get("content", {}).get("text", "")
            location = r.get("location", {})
            uri = (
                location.get("s3Location", {}).get("uri")
                or location.get("webLocation", {}).get("url")
                or ""
            )
            hits.append(
                SearchHit(
                    chunk_id=r.get("metadata", {}).get("chunk_id", uri or content[:32]),
                    doc_id=r.get("metadata", {}).get("doc_id", uri),
                    text=content,
                    score=float(r.get("score") or 0.0),
                    source_uri=uri,
                    metadata=r.get("metadata") or {},
                )
            )
        return hits

    def delete_by_doc(self, doc_id: str, *, tenant_id: Optional[str] = None) -> int:
        del doc_id, tenant_id
        # Deletion is by object lifecycle on the underlying S3 data source.
        return 0

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "kb_id": self.kb_id,
            "region": self.region,
            "data_source_id": self.data_source_id,
        }
