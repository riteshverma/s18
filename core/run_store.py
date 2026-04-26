import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


RUN_STORE_PATH = Path("data/system/agent_runs.sqlite")
_RUN_STATUSES = {
    "accepted",
    "starting",
    "running",
    "waiting_input",
    "completed",
    "failed",
    "stopped",
    "interrupted",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_status(status: Optional[str]) -> str:
    normalized = (status or "accepted").strip().lower()
    if normalized == "success":
        normalized = "completed"
    if normalized == "paused":
        normalized = "stopped"
    if normalized not in _RUN_STATUSES:
        return "failed"
    return normalized


class RunStore:
    """SQLite-backed run metadata registry for durable status tracking."""

    def __init__(self, db_path: Path = RUN_STORE_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        query TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        request_id TEXT,
                        idempotency_key TEXT,
                        integration_id TEXT,
                        workflow_id TEXT,
                        tenant_id TEXT,
                        tenant_tier TEXT,
                        data_region TEXT,
                        summary TEXT,
                        error TEXT,
                        session_file TEXT,
                        metadata_json TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at ON agent_runs(created_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status)"
                )
                conn.commit()
            finally:
                conn.close()

    def upsert_run(
        self,
        *,
        run_id: str,
        status: str,
        query: Optional[str] = None,
        created_at: Optional[str] = None,
        request_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        integration_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        tenant_tier: Optional[str] = None,
        data_region: Optional[str] = None,
        summary: Optional[str] = None,
        error: Optional[str] = None,
        session_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _now_iso()
        effective_created_at = created_at or now
        payload = (
            run_id,
            _normalize_status(status),
            query,
            effective_created_at,
            now,
            request_id,
            idempotency_key,
            integration_id,
            workflow_id,
            tenant_id,
            tenant_tier,
            data_region,
            summary,
            error,
            session_file,
            json.dumps(metadata or {}, ensure_ascii=False),
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO agent_runs (
                        run_id, status, query, created_at, updated_at,
                        request_id, idempotency_key, integration_id, workflow_id,
                        tenant_id, tenant_tier, data_region, summary, error,
                        session_file, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status=excluded.status,
                        query=COALESCE(excluded.query, agent_runs.query),
                        created_at=COALESCE(agent_runs.created_at, excluded.created_at),
                        updated_at=excluded.updated_at,
                        request_id=COALESCE(excluded.request_id, agent_runs.request_id),
                        idempotency_key=COALESCE(excluded.idempotency_key, agent_runs.idempotency_key),
                        integration_id=COALESCE(excluded.integration_id, agent_runs.integration_id),
                        workflow_id=COALESCE(excluded.workflow_id, agent_runs.workflow_id),
                        tenant_id=COALESCE(excluded.tenant_id, agent_runs.tenant_id),
                        tenant_tier=COALESCE(excluded.tenant_tier, agent_runs.tenant_tier),
                        data_region=COALESCE(excluded.data_region, agent_runs.data_region),
                        summary=COALESCE(excluded.summary, agent_runs.summary),
                        error=COALESCE(excluded.error, agent_runs.error),
                        session_file=COALESCE(excluded.session_file, agent_runs.session_file),
                        metadata_json=COALESCE(excluded.metadata_json, agent_runs.metadata_json)
                    """,
                    payload,
                )
                conn.commit()
            finally:
                conn.close()
        return self.get_run(run_id) or {"id": run_id, "status": _normalize_status(status)}

    def update_run(self, run_id: str, **fields: Any) -> Dict[str, Any]:
        if not fields:
            return self.get_run(run_id) or {"id": run_id, "status": "accepted"}

        allowed = {
            "status",
            "query",
            "request_id",
            "idempotency_key",
            "integration_id",
            "workflow_id",
            "tenant_id",
            "tenant_tier",
            "data_region",
            "summary",
            "error",
            "session_file",
            "metadata_json",
        }
        patch: Dict[str, Any] = {}
        for key, value in fields.items():
            if key == "metadata":
                patch["metadata_json"] = json.dumps(value or {}, ensure_ascii=False)
                continue
            if key in allowed:
                patch[key] = _normalize_status(value) if key == "status" else value
        patch["updated_at"] = _now_iso()

        if not patch:
            return self.get_run(run_id) or {"id": run_id, "status": "accepted"}

        assignments = ", ".join([f"{col}=?" for col in patch.keys()])
        values = list(patch.values()) + [run_id]

        needs_insert = False
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(f"UPDATE agent_runs SET {assignments} WHERE run_id=?", values)
                if cur.rowcount == 0:
                    needs_insert = True
                else:
                    conn.commit()
            finally:
                if conn:
                    conn.close()
        if needs_insert:
            return self.upsert_run(
                run_id=run_id,
                status=patch.get("status", "accepted"),
                query=patch.get("query"),
                request_id=patch.get("request_id"),
                idempotency_key=patch.get("idempotency_key"),
                integration_id=patch.get("integration_id"),
                workflow_id=patch.get("workflow_id"),
                tenant_id=patch.get("tenant_id"),
                tenant_tier=patch.get("tenant_tier"),
                data_region=patch.get("data_region"),
                summary=patch.get("summary"),
                error=patch.get("error"),
                session_file=patch.get("session_file"),
                metadata=json.loads(patch.get("metadata_json") or "{}"),
            )
        return self.get_run(run_id) or {"id": run_id, "status": patch.get("status", "accepted")}

    def update_status(self, run_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        fields = dict(fields)
        fields["status"] = status
        return self.update_run(run_id, **fields)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
            finally:
                conn.close()
        return self._row_to_run(row) if row else None

    def list_runs(self, limit: int = 200) -> list[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?",
                    (max(1, int(limit)),),
                ).fetchall()
            finally:
                conn.close()
        return [self._row_to_run(row) for row in rows]

    def mark_orphaned_inflight_as_interrupted(self, active_run_ids: Iterable[str]) -> int:
        active_ids = [str(run_id) for run_id in active_run_ids if run_id]
        now = _now_iso()
        params: list[Any] = [now, "interrupted", "starting", "running", "waiting_input"]
        sql = (
            "UPDATE agent_runs SET updated_at=?, status=? "
            "WHERE status IN (?, ?, ?)"
        )
        if active_ids:
            placeholders = ", ".join(["?"] * len(active_ids))
            sql += f" AND run_id NOT IN ({placeholders})"
            params.extend(active_ids)

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(sql, params)
                conn.commit()
                return cur.rowcount or 0
            finally:
                conn.close()

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        raw_metadata = row["metadata_json"]
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata)
            except Exception:
                metadata = {}

        return {
            "id": row["run_id"],
            "status": row["status"],
            "query": row["query"] or "Unknown Query",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "request_id": row["request_id"],
            "idempotency_key": row["idempotency_key"],
            "integration_id": row["integration_id"],
            "workflow_id": row["workflow_id"],
            "tenant_id": row["tenant_id"],
            "tenant_tier": row["tenant_tier"],
            "data_region": row["data_region"],
            "summary": row["summary"],
            "error": row["error"],
            "session_file": row["session_file"],
            "metadata": metadata,
        }


_run_store: Optional[RunStore] = None


def get_run_store() -> RunStore:
    global _run_store
    if _run_store is None:
        _run_store = RunStore()
    return _run_store
