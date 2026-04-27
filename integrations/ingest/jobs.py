"""File-backed job store for ingest pipelines.

We deliberately avoid pulling in another database dependency: a small JSON
ledger under ``data/ingest/_jobs/`` is enough for the canonical observability
surface (`GET /ingest/jobs/{id}`). Production deployments wire the same
:class:`IngestJobStore` API to Supabase / DynamoDB / Cosmos DB by subclassing.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_JOB_LOCK = threading.Lock()
_DEFAULT_JOB_DIR = Path("data/ingest/_jobs")


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class IngestJob:
    job_id: str
    tenant_id: str
    integration_id: str
    workflow_id: str
    status: str = "queued"
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    record_count: int = 0
    file_count: int = 0
    chunk_count: int = 0
    indexed_count: int = 0
    object_uris: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IngestJobStore:
    """File-backed job store. Thread-safe within a single process."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root or _DEFAULT_JOB_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        clean = job_id.replace("/", "_").replace("..", "")
        return self.root / f"{clean}.json"

    def create(
        self,
        *,
        tenant_id: str,
        integration_id: str,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestJob:
        job = IngestJob(
            job_id=f"ing_{uuid.uuid4().hex[:16]}",
            tenant_id=tenant_id,
            integration_id=integration_id,
            workflow_id=workflow_id,
            metadata=metadata or {},
        )
        self._write(job)
        return job

    def get(self, job_id: str) -> Optional[IngestJob]:
        path = self._path(job_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None
        return IngestJob(**data)

    def update(self, job_id: str, **changes: Any) -> Optional[IngestJob]:
        with _JOB_LOCK:
            job = self.get(job_id)
            if job is None:
                return None
            for key, value in changes.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = _utcnow()
            self._write(job)
            return job

    def append_error(self, job_id: str, error: Dict[str, Any]) -> None:
        with _JOB_LOCK:
            job = self.get(job_id)
            if job is None:
                return
            job.errors.append({"at": _utcnow(), **error})
            job.updated_at = _utcnow()
            self._write(job)

    def increment(self, job_id: str, **deltas: int) -> Optional[IngestJob]:
        with _JOB_LOCK:
            job = self.get(job_id)
            if job is None:
                return None
            for key, delta in deltas.items():
                if hasattr(job, key) and isinstance(getattr(job, key), int):
                    setattr(job, key, getattr(job, key) + int(delta))
            job.updated_at = _utcnow()
            self._write(job)
            return job

    def append_uris(self, job_id: str, uris: List[str]) -> None:
        with _JOB_LOCK:
            job = self.get(job_id)
            if job is None:
                return
            job.object_uris.extend(uris)
            job.updated_at = _utcnow()
            self._write(job)

    def append_chunk_ids(self, job_id: str, chunk_ids: List[str]) -> None:
        with _JOB_LOCK:
            job = self.get(job_id)
            if job is None:
                return
            job.chunk_ids.extend(chunk_ids)
            job.updated_at = _utcnow()
            self._write(job)

    def _write(self, job: IngestJob) -> None:
        path = self._path(job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job.to_dict(), indent=2))


_JOB_STORE: Optional[IngestJobStore] = None


def get_job_store() -> IngestJobStore:
    global _JOB_STORE
    if _JOB_STORE is None:
        _JOB_STORE = IngestJobStore()
    return _JOB_STORE
