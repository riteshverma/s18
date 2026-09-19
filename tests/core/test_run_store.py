import logging
import sqlite3

import pytest

from core.run_store import RunStore, generate_run_id, merge_run_metadata


def test_run_store_upsert_and_list(tmp_path):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(
        run_id="run-1",
        status="accepted",
        query="demo query",
        request_id="req-1",
        metadata={"source_system": "s18"},
    )
    store.update_status("run-1", "running")
    stored = store.get_run("run-1")
    assert stored is not None
    assert stored["id"] == "run-1"
    assert stored["status"] == "running"
    assert stored["query"] == "demo query"
    assert stored["request_id"] == "req-1"
    assert stored["metadata"]["source_system"] == "s18"

    listed = store.list_runs()
    assert listed
    assert listed[0]["id"] == "run-1"


def test_mark_orphaned_inflight_as_interrupted(tmp_path):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(run_id="run-a", status="running", query="a")
    store.upsert_run(run_id="run-b", status="waiting_input", query="b")
    store.upsert_run(run_id="run-c", status="completed", query="c")

    changed = store.mark_orphaned_inflight_as_interrupted(active_run_ids={"run-a"})
    assert changed == 1
    assert store.get_run("run-a")["status"] == "running"
    assert store.get_run("run-b")["status"] == "interrupted"
    assert store.get_run("run-c")["status"] == "completed"


def test_generate_run_id_is_unique_under_same_millisecond():
    ids = {generate_run_id() for _ in range(500)}
    assert len(ids) == 500
    for run_id in ids:
        epoch_ms, suffix = run_id.split("_", 1)
        assert epoch_ms.isdigit() and len(epoch_ms) == 13
        assert len(suffix) == 8


def test_merge_run_metadata_preserves_existing_keys(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    monkeypatch.setattr("core.run_store.get_run_store", lambda: store)
    store.upsert_run(
        run_id="run-1",
        status="accepted",
        query="q",
        metadata={"source_system": "s18", "skill_id": "old"},
    )

    merged = merge_run_metadata("run-1", {"skill_id": "web_clipper", "skill_source": "intent"})

    assert merged["source_system"] == "s18"
    assert merged["skill_id"] == "web_clipper"
    assert merged["skill_source"] == "intent"
    assert store.get_run("run-1")["metadata"] == merged


def test_merge_run_metadata_creates_metadata_when_missing(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    monkeypatch.setattr("core.run_store.get_run_store", lambda: store)
    store.upsert_run(run_id="run-2", status="accepted", query="q")

    merged = merge_run_metadata("run-2", {"execution_backend": "celery"})

    assert merged == {"execution_backend": "celery"}
    assert store.get_run("run-2")["metadata"]["execution_backend"] == "celery"


def test_get_run_by_idempotency_key_returns_newest_match(tmp_path):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(
        run_id="run-old",
        status="accepted",
        query="q",
        idempotency_key="evt-1",
        tenant_id="tenant-a",
        created_at="2026-01-01T00:00:00",
    )
    store.upsert_run(
        run_id="run-new",
        status="running",
        query="q",
        idempotency_key="evt-1",
        tenant_id="tenant-b",
        created_at="2026-01-02T00:00:00",
    )

    found = store.get_run_by_idempotency_key("evt-1")

    assert found is not None
    assert found["id"] == "run-new"
    assert found["status"] == "running"


def test_get_run_by_idempotency_key_respects_tenant_scoping(tmp_path):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(
        run_id="run-a", status="accepted", query="q", idempotency_key="evt-1", tenant_id="tenant-a"
    )
    store.upsert_run(
        run_id="run-b", status="accepted", query="q", idempotency_key="evt-1", tenant_id="tenant-b"
    )

    assert store.get_run_by_idempotency_key("evt-1", tenant_id="tenant-a")["id"] == "run-a"
    assert store.get_run_by_idempotency_key("evt-1", tenant_id="tenant-b")["id"] == "run-b"
    assert store.get_run_by_idempotency_key("evt-1", tenant_id="tenant-c") is None


def test_get_run_by_idempotency_key_returns_none_when_absent(tmp_path):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(run_id="run-a", status="accepted", query="q")

    assert store.get_run_by_idempotency_key("missing-key") is None
    assert store.get_run_by_idempotency_key("") is None


def test_duplicate_key_for_same_tenant_is_rejected(tmp_path):
    """The partial unique index backstops concurrent creates for one tenant."""
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(
        run_id="run-a", status="accepted", query="q", idempotency_key="evt-1", tenant_id="tenant-a"
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_run(
            run_id="run-b",
            status="accepted",
            query="q",
            idempotency_key="evt-1",
            tenant_id="tenant-a",
        )


def test_ensure_db_warns_and_continues_when_duplicate_keys_exist(tmp_path, caplog):
    db_path = tmp_path / "agent_runs.sqlite"
    RunStore(db_path)
    # Simulate a legacy database that predates the unique index and already
    # holds duplicate (idempotency_key, tenant_id) rows.
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX IF EXISTS idx_agent_runs_idempotency")
        for run_id in ("run-a", "run-b"):
            conn.execute(
                "INSERT INTO agent_runs (run_id, status, created_at, updated_at, "
                "idempotency_key, tenant_id) VALUES (?, 'accepted', ?, ?, 'evt-1', 'tenant-a')",
                (run_id, "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
            )

    caplog.set_level(logging.WARNING)
    reopened = RunStore(db_path)

    assert reopened.get_run("run-a")["id"] == "run-a"
    assert any("idempotency" in record.getMessage().lower() for record in caplog.records)
