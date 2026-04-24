"""Root pytest configuration for S18Share.

Includes:
- marker auto-assignment for CI lanes
- shared fixtures for API tests (TestClient factory)
- Supabase client/user doubles
- temporary FAISS index directory
- fake LLM client for deterministic unit tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_INTEGRATION_STEMS: frozenset[str] = frozenset(
    {
        "agent_automated_test",
        "real_verification_suite",
        "audit_features",
    }
)

_SLOW_STEMS: frozenset[str] = frozenset(
    {
        "verify_scheduler_suite",
        "test_multi_mcp_modes",
    }
)

_CONTRACT_STEMS: frozenset[str] = frozenset(
    {
        "test_contracts",
        "test_registry",
        "test_runs_backward_compat",
        "test_runs_router_contract",
        "test_wiseai_adapter",
    }
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        stem = item.path.stem
        if stem in _INTEGRATION_STEMS:
            item.add_marker(pytest.mark.integration)
        if stem in _SLOW_STEMS:
            item.add_marker(pytest.mark.slow)
        if stem in _CONTRACT_STEMS:
            item.add_marker(pytest.mark.contract)


class _FakeSupabaseTable:
    """Minimal chainable table adapter used by tests."""

    def __init__(self) -> None:
        self._result: dict[str, Any] = {"data": [], "error": None}

    def upsert(self, *_args: Any, **_kwargs: Any) -> "_FakeSupabaseTable":
        return self

    def insert(self, *_args: Any, **_kwargs: Any) -> "_FakeSupabaseTable":
        return self

    def select(self, *_args: Any, **_kwargs: Any) -> "_FakeSupabaseTable":
        return self

    def eq(self, *_args: Any, **_kwargs: Any) -> "_FakeSupabaseTable":
        return self

    def execute(self) -> dict[str, Any]:
        return self._result


class FakeSupabaseClient:
    """Simple fake Supabase client fixture for unit tests."""

    def table(self, _name: str) -> _FakeSupabaseTable:
        return _FakeSupabaseTable()


class FakeLLMClient:
    """Deterministic fake LLM client for prompt/loop unit tests."""

    def complete(self, prompt: str, **_kwargs: Any) -> dict[str, str]:
        return {"text": f"fake-llm-complete:{prompt[:40]}"}

    def chat(self, messages: list[dict[str, Any]], **_kwargs: Any) -> dict[str, str]:
        last = messages[-1]["content"] if messages else ""
        return {"text": f"fake-llm-chat:{str(last)[:40]}"}

    def embed(self, text: str, **_kwargs: Any) -> list[float]:
        # Deterministic pseudo-embedding with fixed length.
        base = float(len(text) % 7)
        return [base, base + 1.0, base + 2.0]


@pytest.fixture
def test_client_factory() -> Callable[..., TestClient]:
    """Factory to build a TestClient with optional router/dependency overrides."""

    def _factory(
        *,
        router=None,
        dependencies: dict[Callable[..., Any], Callable[..., Any]] | None = None,
    ) -> TestClient:
        app = FastAPI()
        if router is not None:
            app.include_router(router)
        if dependencies:
            app.dependency_overrides.update(dependencies)
        return TestClient(app)

    return _factory


@pytest.fixture
def test_client(test_client_factory: Callable[..., TestClient]) -> TestClient:
    """Default empty TestClient for simple endpoint unit tests."""
    return test_client_factory()


@pytest.fixture
def mock_supabase_client() -> FakeSupabaseClient:
    return FakeSupabaseClient()


@pytest.fixture
def mock_supabase_user() -> dict[str, str]:
    return {
        "sub": "test-user-id",
        "email": "test.user@s18.local",
        "role": "authenticated",
    }


@pytest.fixture
def temp_faiss_dir(tmp_path: Path) -> Path:
    """Create a temporary FAISS-like index folder for test isolation."""
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Keep files lightweight while matching common path expectations.
    (index_dir / "metadata.json").write_text("[]", encoding="utf-8")
    (index_dir / "ledger.json").write_text("{}", encoding="utf-8")
    (index_dir / "doc_index_cache.json").write_text("{}", encoding="utf-8")
    (index_dir / "captions.json").write_text("{}", encoding="utf-8")
    (index_dir / "index.bin").write_bytes(b"")
    (index_dir / "bm25_index.pkl").write_bytes(b"")
    return index_dir


@pytest.fixture
def fake_llm_client() -> FakeLLMClient:
    return FakeLLMClient()
