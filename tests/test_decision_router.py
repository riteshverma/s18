"""Unit tests for the Jev-backed DecisionRouter and its fallback behavior.

All HTTP is served by httpx.MockTransport; no network access happens.
Async entry points are driven with asyncio.run() to stay independent of
the pytest-asyncio mode configuration.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.decisions.jev_client import JevClient, JevError
from core.decisions.router import (
    Decision,
    DecisionRouter,
    decision_choice,
    decision_flag,
    resolve_skill_intent,
)
from integrations.policies.workflow_guards import (
    is_cbc_payload_query,
    is_fast_mode,
    is_mental_health_task_query,
)


@pytest.fixture(autouse=True)
def _reset_jev_breaker():
    """The circuit breaker registry is global; isolate it per test."""
    from core import circuit_breaker

    circuit_breaker._breakers.pop("jev_decisions", None)
    yield
    circuit_breaker._breakers.pop("jev_decisions", None)


def _jev_answer(choice=None, noul=None, probabilities=None, confidence=None):
    ans = {}
    if choice is not None:
        ans["choice"] = choice
    if noul is not None:
        ans["noul"] = noul
    if probabilities is not None:
        ans["probabilities"] = probabilities
    if confidence is not None:
        ans["confidence"] = confidence
    return ans


def _router_with(answer_body, config=None):
    """Build a DecisionRouter whose HTTP layer returns answer_body (dict or int status)."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if isinstance(answer_body, int):
            return httpx.Response(answer_body, text="boom")
        if not request.headers.get("authorization", "").startswith("Bearer "):
            return httpx.Response(401, text="unauthorized")
        return httpx.Response(
            200,
            json={"model": "jev-latest", "answers": answer_body, "usage": {}},
        )

    router = DecisionRouter(
        config={
            "enabled": True,
            "endpoint": "https://mock.typesafe.ai/v1/systemone",
            "api_key": "test-key",
            **(config or {}),
        },
        transport=httpx.MockTransport(handler),
    )
    return router, calls


# ---------------------------------------------------------------------------
# Router behavior
# ---------------------------------------------------------------------------


def test_ask_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    router, calls = _router_with({}, config={"api_key": ""})
    assert router.enabled is False

    async def run():
        return await router.ask("state", {"q": {"type": "noul", "instructions": "x"}})

    assert asyncio.run(run()) == {}
    assert calls["count"] == 0


def test_env_var_disables_router_even_with_key(monkeypatch):
    monkeypatch.setenv("S18_DECISIONS_ENABLED", "0")
    router, calls = _router_with({})

    async def run():
        return await router.ask("state", {"q": {"type": "noul", "instructions": "x"}})

    assert asyncio.run(run()) == {}
    assert calls["count"] == 0


def test_ask_parses_noul_and_choice_and_caches():
    answers = {
        "is_cbc_payload": _jev_answer(noul=0.93),
        "query_type": _jev_answer(
            choice="cbc", probabilities={"cbc": 0.95, "general": 0.05}, confidence=0.7
        ),
    }
    router, calls = _router_with(answers)
    questions = {"is_cbc_payload": {"type": "noul", "instructions": "x"}}

    async def run():
        first = await router.ask("some query", questions)
        second = await router.ask("some query", questions)
        return first, second

    first, second = asyncio.run(run())
    assert calls["count"] == 1  # second ask was a cache hit
    assert first["is_cbc_payload"].value == 0.93
    assert first["query_type"].value == "cbc"
    assert first["query_type"].probabilities["cbc"] == 0.95
    assert second == first


def test_http_failure_falls_back_and_opens_breaker():
    router, calls = _router_with(500, config={"failure_threshold": 3, "failure_cooldown_seconds": 60})
    questions = {"q": {"type": "noul", "instructions": "x"}}

    async def run():
        results = []
        for _ in range(5):
            results.append(await router.ask("state", questions))
        return results

    results = asyncio.run(run())
    assert all(r == {} for r in results)
    assert calls["count"] == 3  # breaker opens after threshold, later asks skip HTTP


def test_jev_client_wraps_errors():
    def handler(request):
        return httpx.Response(500, text="nope")

    client = JevClient(api_key="k", transport=httpx.MockTransport(handler))

    async def run():
        return await client.ask("state", {"q": {"type": "noul", "instructions": "x"}})

    with pytest.raises(JevError):
        asyncio.run(run())


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------


def test_decision_flag_threshold():
    strong = Decision("is_cbc_payload", value=0.93)
    weak = Decision("is_cbc_payload", value=0.5)
    assert decision_flag({"x": strong}, "x") is True
    assert decision_flag({"x": weak}, "x") is False
    assert decision_flag({}, "missing") is False


def test_decision_choice_uses_probability_of_picked_option():
    high = Decision("skill", value="a", probabilities={"a": 0.91}, confidence=0.4)
    low = Decision("skill", value="a", probabilities={"a": 0.4}, confidence=0.9)
    assert decision_choice({"s": high}, "s") == "a"
    assert decision_choice({"s": low}, "s") is None


# ---------------------------------------------------------------------------
# Guard integration (regex primary, Jev as upgrade)
# ---------------------------------------------------------------------------


def test_guards_fall_back_to_regex_without_decisions():
    assert is_fast_mode("plain query") is False
    assert is_fast_mode("[execution mode: fast]") is True
    assert is_cbc_payload_query("plain query") is False
    assert is_mental_health_task_query("plain query") is False


def test_guards_upgrade_from_decisions():
    decisions = {
        "wants_fast_mode": Decision("wants_fast_mode", value=0.9),
        "is_cbc_payload": Decision("is_cbc_payload", value=0.85),
        "is_mental_health": Decision("is_mental_health", value=0.2),
    }
    assert is_fast_mode("plain query", decisions) is True
    assert is_cbc_payload_query("plain query", decisions) is True
    assert is_mental_health_task_query("plain query", decisions) is False


def test_guards_ignore_low_confidence_decisions():
    decisions = {"wants_fast_mode": Decision("wants_fast_mode", value=0.5)}
    assert is_fast_mode("plain query", decisions) is False


# ---------------------------------------------------------------------------
# Skill intent resolution
# ---------------------------------------------------------------------------


class _StubSkillManager:
    def __init__(self, registry_path, regex_hit=None):
        self.registry_file = registry_path
        self._regex_hit = regex_hit

    def match_intent(self, query):
        return self._regex_hit


def _write_registry(tmp_path):
    registry = {
        "jev_skill": {"description": "Handles J things"},
        "regex_skill": {"description": "Handles R things"},
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    return path


def test_resolve_skill_intent_uses_jev_when_confident(tmp_path, monkeypatch):
    path = _write_registry(tmp_path)
    answers = {"skill": _jev_answer(choice="jev_skill", probabilities={"jev_skill": 0.95}, confidence=0.8)}
    router, _ = _router_with(answers)
    monkeypatch.setattr("core.decisions.router.get_decision_router", lambda: router)

    manager = _StubSkillManager(path)  # regex finds nothing
    picked = asyncio.run(resolve_skill_intent("mystery query", manager))
    assert picked == "jev_skill"


def test_resolve_skill_intent_keeps_regex_on_low_confidence(tmp_path, monkeypatch):
    path = _write_registry(tmp_path)
    answers = {"skill": _jev_answer(choice="none", probabilities={"none": 0.99}, confidence=0.9)}
    router, _ = _router_with(answers)
    monkeypatch.setattr("core.decisions.router.get_decision_router", lambda: router)

    manager = _StubSkillManager(path, regex_hit="regex_skill")
    picked = asyncio.run(resolve_skill_intent("alpha query", manager))
    assert picked == "regex_skill"


def test_resolve_skill_intent_falls_back_when_disabled(tmp_path, monkeypatch):
    path = _write_registry(tmp_path)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    router, calls = _router_with({}, config={"api_key": ""})
    monkeypatch.setattr("core.decisions.router.get_decision_router", lambda: router)

    manager = _StubSkillManager(path, regex_hit="regex_skill")
    picked = asyncio.run(resolve_skill_intent("alpha query", manager))
    assert picked == "regex_skill"
    assert calls["count"] == 0
