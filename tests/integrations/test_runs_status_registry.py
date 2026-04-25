from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.run_store import RunStore
from core.supabase_auth import require_supabase_user
from routers import runs


async def _fake_user():
    return {"sub": "test-user", "email": "test@example.com"}


def _make_client():
    app = FastAPI()
    app.include_router(runs.router)
    app.dependency_overrides[require_supabase_user] = _fake_user
    return TestClient(app)


def test_get_run_uses_durable_registry_when_graph_missing(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(run_id="run-42", status="interrupted", query="recover me")
    monkeypatch.setattr(runs, "run_store", store)

    client = _make_client()
    response = client.get("/runs/run-42", headers={"Authorization": "Bearer token"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "run-42"
    assert payload["status"] == "interrupted"
    assert payload["graph"] == {"nodes": [], "edges": []}


def test_list_runs_reads_durable_registry(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(run_id="run-1", status="completed", query="done")
    store.upsert_run(run_id="run-2", status="running", query="in progress")
    monkeypatch.setattr(runs, "run_store", store)

    client = _make_client()
    response = client.get("/runs", headers={"Authorization": "Bearer token"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {row["id"] for row in body} == {"run-1", "run-2"}


def test_stop_run_updates_durable_registry(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    store.upsert_run(run_id="run-stop", status="running", query="stop me")
    monkeypatch.setattr(runs, "run_store", store)

    client = _make_client()
    response = client.post("/runs/run-stop/stop", headers={"Authorization": "Bearer token"})
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"
    assert store.get_run("run-stop")["status"] == "stopped"
