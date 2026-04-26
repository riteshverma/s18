"""Smoke tests for the VectorStore facade (FAISS path)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.vectors import Chunk
from integrations.vectors.faiss_local import FaissLocalVectorStore


pytest.importorskip("faiss")


def _unit(vec):
    arr = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(arr)
    return (arr / n).tolist()


def test_faiss_upsert_and_query(tmp_path: Path):
    store = FaissLocalVectorStore(index_dir=tmp_path)
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            doc_id=f"d{i}",
            text=f"chunk {i}",
            embedding=_unit([1.0 if j == i else 0.0 for j in range(8)]),
            tenant_id="acme",
            integration_id="powerapps",
            source_uri=f"file://test/{i}",
            metadata={"chunk_index": i},
        )
        for i in range(4)
    ]
    inserted = store.upsert(chunks)
    assert inserted == 4
    assert store.stats()["ntotal"] == 4

    hits = store.query(embedding=_unit([1.0, 0, 0, 0, 0, 0, 0, 0]), k=2)
    assert hits, "expected at least one hit"
    assert hits[0].chunk_id == "c0"
    assert hits[0].source_uri == "file://test/0"


def test_faiss_filters_by_tenant(tmp_path: Path):
    store = FaissLocalVectorStore(index_dir=tmp_path)
    store.upsert(
        [
            Chunk(
                chunk_id="t1-c0",
                doc_id="t1-d0",
                text="t1",
                embedding=_unit([1, 0, 0, 0]),
                tenant_id="t1",
                integration_id="powerapps",
            ),
            Chunk(
                chunk_id="t2-c0",
                doc_id="t2-d0",
                text="t2",
                embedding=_unit([1, 0, 0, 0]),
                tenant_id="t2",
                integration_id="powerapps",
            ),
        ]
    )
    hits = store.query(embedding=_unit([1, 0, 0, 0]), k=5, filters={"tenant_id": "t2"})
    assert hits and all(h.metadata["tenant_id"] == "t2" for h in hits)


def test_faiss_dimension_mismatch_raises(tmp_path: Path):
    store = FaissLocalVectorStore(index_dir=tmp_path)
    store.ensure_index(dimension=4)
    with pytest.raises(ValueError):
        store.ensure_index(dimension=8)


def test_faiss_delete_by_doc(tmp_path: Path):
    store = FaissLocalVectorStore(index_dir=tmp_path)
    store.upsert(
        [
            Chunk(chunk_id="c1", doc_id="doc-1", text="x", embedding=_unit([1, 0, 0])),
            Chunk(chunk_id="c2", doc_id="doc-2", text="y", embedding=_unit([0, 1, 0])),
        ]
    )
    removed = store.delete_by_doc("doc-1")
    assert removed == 1
