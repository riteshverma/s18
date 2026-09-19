import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.run_store import RunStore
from core.supabase_auth import require_supabase_user
from integrations.contracts import CanonicalRunRequest
from routers import runs


async def _fake_user():
    return {"sub": "test-user", "email": "test@example.com"}


def _make_client():
    app = FastAPI()
    app.include_router(runs.router)
    app.dependency_overrides[require_supabase_user] = _fake_user
    return TestClient(app)


def test_runs_accepts_explicit_canonical_fields():
    client = _make_client()
    with patch("routers.runs.log_inbound_request", new=AsyncMock(return_value=None)):
        with patch("routers.runs.execute_run", new=AsyncMock(return_value=None)):
            resp = client.post(
                "/runs",
                json={
                    "query": "interpret cbc",
                    "integration_id": "wiseai",
                    "workflow_id": "cdss",
                    "contract_version": "v1",
                    "source_system": "wiseai",
                },
                headers={"Authorization": "Bearer token"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "starting"
    assert body["integration_id"] == "wiseai"
    assert body["workflow_id"] == "cdss"
    assert body["contract_version"] == "v1"
    assert body["tenant_id"] == "default"
    assert body["tenant_tier"] == "starter"


def test_runs_supports_default_non_wise_integration_path():
    client = _make_client()
    with patch("routers.runs.log_inbound_request", new=AsyncMock(return_value=None)):
        with patch("routers.runs.execute_run", new=AsyncMock(return_value=None)):
            resp = client.post(
                "/runs",
                json={
                    "query": "generic integration request",
                    "integration_id": "default",
                    "workflow_id": "generic",
                    "contract_version": "v1",
                    "source_system": "s18",
                },
                headers={"Authorization": "Bearer token"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "starting"
    assert body["integration_id"] == "default"
    assert body["workflow_id"] == "generic"
    assert body["contract_version"] == "v1"


def test_runs_accepts_explicit_tenant_context():
    client = _make_client()
    with patch("routers.runs.log_inbound_request", new=AsyncMock(return_value=None)):
        with patch("routers.runs.execute_run", new=AsyncMock(return_value=None)):
            resp = client.post(
                "/runs",
                json={
                    "query": "interpret cbc",
                    "integration_id": "wiseai",
                    "workflow_id": "cdss",
                    "contract_version": "v1",
                    "source_system": "wiseai",
                    "tenant_id": "acme-health",
                    "tenant_tier": "growth",
                    "data_region": "in",
                },
                headers={"Authorization": "Bearer token"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "acme-health"
    assert body["tenant_tier"] == "growth"
    assert body["data_region"] == "in"


def test_build_memory_context_parses_rag_snippet_literals():
    rag_payload = SimpleNamespace(content=[SimpleNamespace(text='["snippet one", "snippet two"]')])
    with patch("core.run_service._retrieve_memories_sync", return_value=[]):
        with patch("core.run_service.multi_mcp.call_tool", new=AsyncMock(return_value=rag_payload)):
            memory_context, _ = asyncio.run(runs._build_memory_context("run-1", "hello"))
    assert "snippet one" in memory_context


def test_has_idempotency_signal_requires_client_key_or_event_id():
    assert runs._has_idempotency_signal(
        CanonicalRunRequest(query="q", idempotency_key="client-key")
    )
    assert runs._has_idempotency_signal(
        CanonicalRunRequest(query="q", external_event_id="evt-1")
    )
    assert not runs._has_idempotency_signal(CanonicalRunRequest(query="q"))


WISE_POST = {
    "query": "interpret cbc",
    "integration_id": "wiseai",
    "workflow_id": "cdss",
    "contract_version": "v1",
    "source_system": "wiseai",
}


def _post_runs(client, payloads):
    """Post each payload once, sharing one patched execute_run mock so the
    total execution count is observable across requests."""
    responses = []
    with patch("routers.runs.log_inbound_request", new=AsyncMock(return_value=None)):
        with patch("routers.runs.execute_run", new=AsyncMock(return_value=None)) as mock_exec:
            for payload in payloads:
                responses.append(
                    client.post("/runs", json=payload, headers={"Authorization": "Bearer token"})
                )
    return responses, mock_exec


def test_runs_dedupes_retry_with_same_external_event_id(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    monkeypatch.setattr(runs, "run_store", store)
    client = _make_client()
    payload = {**WISE_POST, "external_event_id": "evt-dedupe-1", "tenant_id": "acme-health"}

    responses, mock_exec = _post_runs(client, [payload, payload])
    first, second = responses

    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert second_body["id"] == first_body["id"]
    assert second_body["request_id"] == first_body["request_id"]
    assert second_body["status"] == "accepted"
    assert second_body["deduplicated"] is True
    assert first_body["deduplicated"] is None
    assert [row["id"] for row in store.list_runs()] == [first_body["id"]]
    assert mock_exec.await_count == 1


def test_runs_does_not_dedupe_plain_repeated_queries(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    monkeypatch.setattr(runs, "run_store", store)
    client = _make_client()

    responses, mock_exec = _post_runs(client, [dict(WISE_POST), dict(WISE_POST)])
    first, second = responses

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["id"] != first.json()["id"]
    assert second.json()["deduplicated"] is None
    assert len(store.list_runs()) == 2
    assert mock_exec.await_count == 2


def test_runs_same_event_from_two_tenants_is_not_deduped(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "agent_runs.sqlite")
    monkeypatch.setattr(runs, "run_store", store)
    client = _make_client()

    responses, mock_exec = _post_runs(
        client,
        [
            {**WISE_POST, "external_event_id": "evt-shared-1", "tenant_id": "tenant-a"},
            {**WISE_POST, "external_event_id": "evt-shared-1", "tenant_id": "tenant-b"},
        ],
    )
    first, second = responses

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["id"] != first.json()["id"]
    assert second.json()["deduplicated"] is None
    assert sorted(row["tenant_id"] for row in store.list_runs()) == ["tenant-a", "tenant-b"]
    assert mock_exec.await_count == 2
