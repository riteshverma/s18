"""FAISS-backed vector store kept compatible with mcp_servers/faiss_index/.

The dev path keeps using the existing ``index.bin`` + ``metadata.json`` layout
that ``mcp_servers/server_rag.py`` already reads, so retrieval keeps working
unchanged for tenants on the shared starter tier.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.faiss_runtime import (
    create_index_flat_ip,
    read_index,
    runtime_info,
    write_index,
)
from integrations.vectors.base import Chunk, SearchHit, VectorStore


class FaissLocalVectorStore(VectorStore):
    provider = "faiss"

    def __init__(self, *, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.bin"
        self.metadata_path = self.index_dir / "metadata.json"
        # Re-entrant: upsert() holds the lock and calls ensure_index(), which
        # also acquires it. A plain Lock would deadlock under that pattern.
        self._lock = threading.RLock()
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._dimension: Optional[int] = None
        self._load()

    def _load(self) -> None:
        if self.metadata_path.exists():
            try:
                self._metadata = json.loads(self.metadata_path.read_text())
            except Exception:
                self._metadata = []
        else:
            self._metadata = []

        if self.index_path.exists():
            try:
                self._index = read_index(self.index_path)
                self._dimension = self._index.d
            except Exception:
                self._index = None

    def _save(self) -> None:
        if self._index is not None:
            write_index(self._index, self.index_path)
        self.metadata_path.write_text(json.dumps(self._metadata, indent=2))

    def ensure_index(self, *, dimension: int, metric: str = "cosine") -> None:
        del metric  # FAISS index type is set by caller; we use IndexFlatIP for cosine on normalized vectors
        with self._lock:
            if self._index is None:
                # Inner product on L2-normalized vectors == cosine similarity.
                self._index = create_index_flat_ip(dimension)
                self._dimension = dimension
            elif self._dimension is not None and self._dimension != dimension:
                raise ValueError(
                    f"FAISS index dimension mismatch: existing={self._dimension}, requested={dimension}. "
                    "Rebuild the index before switching embedding models."
                )

    def upsert(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        with self._lock:
            self.ensure_index(dimension=len(chunks[0].embedding))
            existing = {m["chunk_id"]: i for i, m in enumerate(self._metadata) if "chunk_id" in m}
            vectors = []
            new_meta = []
            for chunk in chunks:
                vec = np.array(chunk.embedding, dtype=np.float32)
                if vec.shape[0] != self._dimension:
                    raise ValueError(
                        f"Chunk {chunk.chunk_id} dim {vec.shape[0]} != index dim {self._dimension}"
                    )
                vectors.append(vec)
                row = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc": chunk.metadata.get("doc") or chunk.doc_id,
                    "chunk": chunk.text,
                    "tenant_id": chunk.tenant_id,
                    "integration_id": chunk.integration_id,
                    "source_uri": chunk.source_uri,
                    **{k: v for k, v in chunk.metadata.items() if k != "doc"},
                }
                if chunk.chunk_id in existing:
                    self._metadata[existing[chunk.chunk_id]] = row
                else:
                    new_meta.append(row)
            if vectors:
                arr = np.vstack(vectors).astype(np.float32)
                self._index.add(arr)
                self._metadata.extend(new_meta)
            self._save()
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
            raise ValueError("FaissLocalVectorStore.query requires `embedding`")
        if self._index is None or self._index.ntotal == 0:
            return []
        del text  # FAISS-only path; hybrid search lives in the caller (server_rag.py)

        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        # Pull more candidates so post-filter still has k hits.
        fan_out = max(k * 4, 16)
        scores, idxs = self._index.search(vec, fan_out)
        hits: List[SearchHit] = []
        for rank, idx in enumerate(idxs[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            if filters and not _passes_filters(meta, filters):
                continue
            hits.append(
                SearchHit(
                    chunk_id=meta.get("chunk_id", f"idx_{idx}"),
                    doc_id=meta.get("doc_id") or meta.get("doc", ""),
                    text=meta.get("chunk", ""),
                    score=float(scores[0][rank]),
                    source_uri=meta.get("source_uri", ""),
                    metadata=meta,
                )
            )
            if len(hits) >= k:
                break
        return hits

    def delete_by_doc(self, doc_id: str, *, tenant_id: Optional[str] = None) -> int:
        with self._lock:
            keep = []
            removed = 0
            for row in self._metadata:
                same_doc = row.get("doc_id") == doc_id or row.get("doc") == doc_id
                same_tenant = tenant_id is None or row.get("tenant_id") == tenant_id
                if same_doc and same_tenant:
                    removed += 1
                    continue
                keep.append(row)
            if removed:
                # Soft-delete: rebuild metadata; the orphan vectors remain in
                # FAISS but their chunk_ids are gone, so search filters skip
                # them. A periodic compaction job (`server_rag.py` reindex)
                # rebuilds the index cleanly.
                self._metadata = keep
                self._save()
            return removed

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "ntotal": int(self._index.ntotal) if self._index is not None else 0,
            "dimension": self._dimension,
            "metadata_count": len(self._metadata),
            "index_dir": str(self.index_dir),
            "faiss": runtime_info(),
        }


def _passes_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if isinstance(expected, (list, tuple, set)):
            if meta.get(key) not in expected:
                return False
        elif meta.get(key) != expected:
            return False
    return True
