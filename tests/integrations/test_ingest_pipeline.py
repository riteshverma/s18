"""End-to-end smoke for the ingest pipeline using local_fs + FAISS + a stub embedder."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.ingest import IngestRecord, chunk_record, chunk_text, parse_file_to_text
from integrations.ingest.jobs import IngestJobStore
from integrations.ingest.pipeline import iter_chunks_with_embeddings, materialize_record


def test_chunk_text_respects_overlap():
    words = " ".join(f"w{i}" for i in range(50))
    pieces = chunk_text(words, chunk_size=10, overlap=2)
    assert pieces, "expected non-empty chunks"
    assert len(pieces) > 1
    # every word should appear in at least one chunk
    joined = " ".join(pieces).split()
    for w in words.split():
        assert w in joined


def test_materialize_record_handles_dataverse_row():
    rec = materialize_record(
        {
            "recordId": "abc",
            "tableLogicalName": "claim",
            "fields": {"amount": 1200, "status": "open"},
        }
    )
    assert rec.record_id == "abc"
    assert "table: claim" in rec.text
    assert "amount: 1200" in rec.text


def test_chunk_record_emits_unique_ids():
    rec = IngestRecord(record_id="r1", record_kind="dataverse", text="alpha beta gamma " * 200)
    chunks = chunk_record(rec, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_parse_file_to_text_csv():
    csv = b"a,b,c\n1,2,3\n4,5,6\n"
    out = parse_file_to_text(filename="x.csv", data=csv, content_type="text/csv")
    assert "a: 1" in out and "b: 2" in out


def test_iter_chunks_with_embeddings_uses_injected_embedder():
    chunks = [{"chunk_id": "c1", "doc_id": "d", "text": "hello", "metadata": {}}]
    fake = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    out = list(iter_chunks_with_embeddings(chunks, embedder=fake))
    assert out[0]["embedding"] == [1.0, 0.0, 0.0]


def test_ingest_job_store_lifecycle(tmp_path: Path):
    store = IngestJobStore(root=tmp_path)
    job = store.create(tenant_id="acme", integration_id="powerapps", workflow_id="claims")
    assert job.status == "queued"
    store.update(job.job_id, status="parsing")
    store.increment(job.job_id, chunk_count=3)
    store.append_uris(job.job_id, ["file://a", "file://b"])
    store.append_error(job.job_id, {"stage": "parse", "reason": "bad pdf"})
    fetched = store.get(job.job_id)
    assert fetched.status == "parsing"
    assert fetched.chunk_count == 3
    assert fetched.object_uris == ["file://a", "file://b"]
    assert fetched.errors and fetched.errors[0]["reason"] == "bad pdf"


def test_pipeline_local_roundtrip(tmp_path: Path, monkeypatch):
    """Run materialize -> parse -> embed -> upsert with all dependencies stubbed."""
    pytest.importorskip("faiss")

    from integrations.storage import factory as storage_factory
    from integrations.vectors import factory as vectors_factory

    fake_settings = {
        "tenancy": {"growth_routing_enabled": False},
        "ingest": {
            "object_store": {
                "provider": "local_fs",
                "local_fs": {"root": str(tmp_path / "obj")},
                "tenant_overrides": {},
            },
            "vector_store": {
                "provider": "faiss",
                "faiss": {"index_dir": str(tmp_path / "vec")},
                "tenant_overrides": {},
            },
        },
    }
    monkeypatch.setattr(storage_factory, "load_settings", lambda: fake_settings)
    monkeypatch.setattr(vectors_factory, "load_settings", lambda: fake_settings)

    # Stub the embedding function to avoid Ollama / Azure calls in CI.
    from core import embedding as embedding_mod

    def _fake_batch(texts, **_kw):
        return [np.array([1.0 if i == 0 else 0.0 for i in range(8)], dtype=np.float32) for _ in texts]

    monkeypatch.setattr(embedding_mod, "get_batch_normalized_embeddings", _fake_batch)
    # Patch the worker's import too.
    import workers.ingest_tasks as ingest_tasks

    monkeypatch.setattr(ingest_tasks, "get_batch_normalized_embeddings", _fake_batch)

    job_store = ingest_tasks.get_job_store()
    job = job_store.create(tenant_id="default", integration_id="powerapps", workflow_id="generic")

    canonical_payload = {
        "integration_id": "powerapps",
        "workflow_id": "generic",
        "tenant_id": "default",
        "tenant_tier": "starter",
        "data_region": "in",
        "policy": {"chunk": {"size": 32, "overlap": 4, "max_length": 256}},
    }
    raw_payload = {
        "record": [
            {
                "recordId": "abc-1",
                "tableLogicalName": "claim",
                "fields": {"summary": "patient reports persistent cough and fever for 3 days"},
            }
        ],
        "files": [],
    }

    result = ingest_tasks.run_ingest_inline(
        job_id=job.job_id,
        canonical_payload=canonical_payload,
        raw_payload=raw_payload,
    )
    assert result["indexed"] >= 1

    final = job_store.get(job.job_id)
    assert final.status == "completed"
    assert final.indexed_count >= 1
