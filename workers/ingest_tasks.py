"""Celery tasks for the Power Apps ingest pipeline.

Three independent tasks make it easy to retry parsing or embedding without
re-uploading bytes:

* `s18share.ingest.materialize` -> normalize records + persist files
* `s18share.ingest.parse_and_chunk` -> turn objects into chunk dicts
* `s18share.ingest.embed_and_index` -> compute embeddings + upsert to vector store
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from celery import chain

from core.celery_app import celery_app
from core.embedding import get_batch_normalized_embeddings
from integrations.ingest import (
    IngestRecord,
    chunk_record,
    chunk_text,
    get_job_store,
    parse_file_to_text,
)
from integrations.ingest.pipeline import iter_chunks_with_embeddings, materialize_record
from integrations.storage import get_object_store
from integrations.vectors import Chunk, get_vector_store


def _tenant_context(payload: Dict[str, Any]) -> Dict[str, str]:
    return {
        "tenant_id": str(payload.get("tenant_id") or "default"),
        "tenant_tier": str(payload.get("tenant_tier") or "starter"),
        "data_region": str(payload.get("data_region") or "in"),
    }


def enqueue_ingest(
    *,
    job_id: str,
    canonical_payload: Dict[str, Any],
    raw_payload: Dict[str, Any],
) -> str:
    """Helper used by the FastAPI router to start the 3-stage chain."""

    pipeline = chain(
        materialize_task.s(job_id, canonical_payload, raw_payload),
        parse_and_chunk_task.s(job_id, canonical_payload),
        embed_and_index_task.s(job_id, canonical_payload),
    )
    result = pipeline.apply_async()
    return result.id


@celery_app.task(name="s18share.ingest.materialize", bind=True)
def materialize_task(
    self,
    job_id: str,
    canonical_payload: Dict[str, Any],
    raw_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 1: persist files into the ObjectStore and normalize records."""

    store = get_job_store()
    tenant_ctx = _tenant_context(canonical_payload)
    object_store = get_object_store(tenant_ctx)

    job = store.update(job_id, status="materializing")
    if job is None:
        store.create(
            tenant_id=tenant_ctx["tenant_id"],
            integration_id=canonical_payload.get("integration_id", "powerapps"),
            workflow_id=canonical_payload.get("workflow_id", "generic"),
            metadata={"task_id": self.request.id},
        )

    records: List[Dict[str, Any]] = list(raw_payload.get("record") or raw_payload.get("records") or [])
    if isinstance(records, dict):  # tolerate single record shorthand
        records = [records]
    files: List[Dict[str, Any]] = list(raw_payload.get("files") or [])

    integration_id = canonical_payload.get("integration_id", "powerapps")
    tenant_id = tenant_ctx["tenant_id"]

    materialized_files: List[Dict[str, Any]] = []
    object_uris: List[str] = []

    for f in files:
        filename = f.get("name") or f.get("filename") or "blob"
        content_type = f.get("contentType") or f.get("content_type")
        content_b64 = f.get("contentBytes") or f.get("content_b64")
        if not content_b64:
            store.append_error(
                job_id,
                {"stage": "materialize", "filename": filename, "reason": "missing_contentBytes"},
            )
            continue
        try:
            data = base64.b64decode(content_b64)
        except Exception as exc:
            store.append_error(
                job_id,
                {"stage": "materialize", "filename": filename, "reason": f"base64: {exc}"},
            )
            continue
        key = f"{integration_id}/{job_id}/{filename}"
        ref = object_store.put(
            key,
            data,
            content_type=content_type,
            metadata={
                "tenant_id": tenant_id,
                "integration_id": integration_id,
                "job_id": job_id,
            },
        )
        materialized_files.append(
            {
                "filename": filename,
                "uri": ref.uri,
                "size": ref.size,
                "sha256": ref.sha256,
                "content_type": content_type,
            }
        )
        object_uris.append(ref.uri)

    if object_uris:
        store.append_uris(job_id, object_uris)
    store.update(
        job_id,
        record_count=len(records),
        file_count=len(materialized_files),
    )

    return {
        "records": records,
        "files": materialized_files,
        "tenant_context": tenant_ctx,
    }


@celery_app.task(name="s18share.ingest.parse_and_chunk", bind=True)
def parse_and_chunk_task(
    self,
    materialized: Dict[str, Any],
    job_id: str,
    canonical_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 2: produce chunk dicts (no embeddings yet)."""

    store = get_job_store()
    tenant_ctx = materialized.get("tenant_context") or _tenant_context(canonical_payload)
    object_store = get_object_store(tenant_ctx)

    chunk_cfg = (canonical_payload.get("policy") or {}).get("chunk") or {}
    chunk_size = int(chunk_cfg.get("size") or 512)
    overlap = int(chunk_cfg.get("overlap") or 64)
    max_length = int(chunk_cfg.get("max_length") or 1024)

    store.update(job_id, status="parsing")

    chunks: List[Dict[str, Any]] = []

    for record in materialized.get("records", []) or []:
        ingest_record = materialize_record(record)
        chunks.extend(
            chunk_record(
                ingest_record,
                chunk_size=chunk_size,
                overlap=overlap,
                max_length=max_length,
            )
        )

    for file_meta in materialized.get("files", []) or []:
        try:
            data = object_store.get(file_meta["uri"])
        except Exception as exc:
            store.append_error(
                job_id,
                {"stage": "parse", "uri": file_meta["uri"], "reason": str(exc)},
            )
            continue
        text = parse_file_to_text(
            filename=file_meta.get("filename", ""),
            data=data,
            content_type=file_meta.get("content_type"),
        )
        if not text:
            store.append_error(
                job_id,
                {
                    "stage": "parse",
                    "uri": file_meta["uri"],
                    "reason": "no_text_extracted",
                },
            )
            continue
        from integrations.ingest.pipeline import hash_id

        for idx, piece in enumerate(
            chunk_text(text, chunk_size=chunk_size, overlap=overlap, max_length=max_length)
        ):
            chunk_id = hash_id(file_meta["uri"], str(idx), piece[:64])
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": file_meta["uri"],
                    "text": piece,
                    "metadata": {
                        "filename": file_meta.get("filename"),
                        "content_type": file_meta.get("content_type"),
                        "sha256": file_meta.get("sha256"),
                        "chunk_index": idx,
                        "record_kind": "file",
                    },
                }
            )

    store.update(job_id, chunk_count=len(chunks))
    return {"chunks": chunks, "tenant_context": tenant_ctx}


@celery_app.task(name="s18share.ingest.embed_and_index", bind=True)
def embed_and_index_task(
    self,
    parsed: Dict[str, Any],
    job_id: str,
    canonical_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 3: embed each chunk and upsert into the tenant vector store."""

    store = get_job_store()
    tenant_ctx = parsed.get("tenant_context") or _tenant_context(canonical_payload)
    integration_id = canonical_payload.get("integration_id", "powerapps")
    tenant_id = tenant_ctx["tenant_id"]

    raw_chunks = parsed.get("chunks") or []
    if not raw_chunks:
        store.update(job_id, status="completed")
        return {"indexed": 0}

    store.update(job_id, status="embedding")

    def _embed(texts: List[str]) -> List[List[float]]:
        vectors = get_batch_normalized_embeddings(texts, task_type="search_document")
        return [v.tolist() for v in vectors]

    embedded = list(iter_chunks_with_embeddings(raw_chunks, embedder=_embed))

    vector_store = get_vector_store(tenant_ctx)
    if embedded:
        vector_store.ensure_index(dimension=len(embedded[0]["embedding"]))

    chunks: List[Chunk] = []
    chunk_ids: List[str] = []
    for c in embedded:
        chunks.append(
            Chunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                text=c["text"],
                embedding=c["embedding"],
                tenant_id=tenant_id,
                integration_id=integration_id,
                source_uri=c.get("metadata", {}).get("source_uri") or c["doc_id"],
                metadata=c.get("metadata") or {},
            )
        )
        chunk_ids.append(c["chunk_id"])

    indexed = vector_store.upsert(chunks)
    store.append_chunk_ids(job_id, chunk_ids)
    store.update(job_id, indexed_count=indexed, status="completed")
    return {"indexed": indexed, "vector_provider": vector_store.provider}


def run_ingest_inline(
    *,
    job_id: str,
    canonical_payload: Dict[str, Any],
    raw_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Synchronous fallback used when ``S18_RUN_EXECUTOR=in_process``.

    Runs the same three steps in-process so dev environments without a Redis
    broker still process ingest requests.
    """

    materialized = materialize_task.run(job_id, canonical_payload, raw_payload)
    parsed = parse_and_chunk_task.run(materialized, job_id, canonical_payload)
    return embed_and_index_task.run(parsed, job_id, canonical_payload)
