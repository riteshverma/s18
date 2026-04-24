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

from core.supabase_auth import require_supabase_user
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
        with patch("routers.runs.process_run", new=AsyncMock(return_value=None)):
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
        with patch("routers.runs.process_run", new=AsyncMock(return_value=None)):
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
        with patch("routers.runs.process_run", new=AsyncMock(return_value=None)):
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
    with patch("routers.runs._retrieve_memories_sync", return_value=[]):
        with patch("routers.runs.multi_mcp.call_tool", new=AsyncMock(return_value=rag_payload)):
            memory_context, _ = asyncio.run(runs._build_memory_context("run-1", "hello"))
    assert "snippet one" in memory_context
