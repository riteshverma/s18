from core.run_store import RunStore


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
