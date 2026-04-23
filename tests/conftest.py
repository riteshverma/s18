"""Root pytest configuration for S18Share.

Scope: marker auto-assignment only.
Shared fixtures (TestClient factory, Supabase mock, FAISS dir, fake LLM)
land in a follow-up change (hardening plan P0 #4).
"""

from __future__ import annotations

import pytest


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
