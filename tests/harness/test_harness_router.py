from __future__ import annotations

import sys
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.supabase_auth import require_supabase_user
from routers import harness


async def _fake_user():
    return {"sub": "test-user", "email": "test@example.com"}


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(harness.router)
    app.dependency_overrides[require_supabase_user] = _fake_user
    return TestClient(app)


def test_create_harness_job_endpoint():
    client = _make_client()
    fake_state = SimpleNamespace(
        id="1",
        model_dump=lambda mode="json": {"id": "1", "status": "accepted", "provider": "claude"},
    )
    with patch.object(harness.runtime, "create_job", new=AsyncMock(return_value=fake_state)):
        with patch.object(harness.runtime, "run_job", new=AsyncMock(return_value=None)):
            response = client.post(
                "/harness/jobs",
                json={"provider": "claude", "prompt": "hello harness"},
                headers={"Authorization": "Bearer token"},
            )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "1"
    assert body["provider"] == "claude"


def test_get_harness_job_not_found():
    client = _make_client()
    with patch.object(harness.runtime, "get_job", new=AsyncMock(return_value=None)):
        response = client.get("/harness/jobs/missing", headers={"Authorization": "Bearer token"})
    assert response.status_code == 404

