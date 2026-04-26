import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


def test_runs_source_only_payload_stays_compatible():
    client = _make_client()
    with patch("routers.runs.log_inbound_request", new=AsyncMock(return_value=None)):
        with patch("routers.runs.process_run", new=AsyncMock(return_value=None)):
            resp = client.post(
                "/runs",
                json={
                    "query": "legacy wise query payload",
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
