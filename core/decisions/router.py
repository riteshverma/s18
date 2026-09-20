"""DecisionRouter: calibrated fast decisions (Jev) with heuristic fallback.

The router asks Jev — TypeSafe AI's System One model — to answer typed
yes/no and choice questions about a query. Every caller keeps its
existing regex/keyword path: Jev decisions are an upgrade layered on
top, applied only when the API is reachable and the answer clears the
configured confidence threshold. Any failure returns ``{}`` so callers
silently fall back to heuristics.

Configuration lives under the ``decisions`` key of config/settings.json.
The API key is read at runtime from the ``JEV_API_KEY`` env var (or
``TYPESAFE_API_KEY``, matching the official SDK) — never hardcode it.
``S18_DECISIONS_ENABLED=0|1`` overrides the enabled flag at runtime.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from config.settings_loader import load_settings
from core.circuit_breaker import get_breaker
from core.decisions.jev_client import JevClient
from core.utils import log_step

_ENV_ENABLE = "S18_DECISIONS_ENABLED"


@dataclass
class Decision:
    """One typed answer from Jev."""

    name: str
    value: Any  # str for choice, float for score/noul
    confidence: Optional[float] = None  # None for noul: value IS the calibration
    probabilities: Dict[str, float] = field(default_factory=dict)
    source: str = "jev"


def _as_float(raw) -> Optional[float]:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _env_flag() -> Optional[bool]:
    raw = os.getenv(_ENV_ENABLE)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_threshold() -> float:
    try:
        cfg = load_settings().get("decisions", {}) or {}
        return float(cfg.get("confidence_threshold", 0.8))
    except Exception:
        return 0.8


def decision_flag(
    decisions: Dict[str, Decision],
    name: str,
    threshold: Optional[float] = None,
) -> bool:
    """True when a noul decision is a strong 'yes' (value >= threshold)."""
    decision = decisions.get(name)
    if decision is None:
        return False
    limit = _default_threshold() if threshold is None else threshold
    return _as_float(decision.value) is not None and float(decision.value) >= limit


def decision_choice(
    decisions: Dict[str, Decision],
    name: str,
    threshold: Optional[float] = None,
) -> Optional[str]:
    """Chosen option when its probability clears the threshold, else None."""
    decision = decisions.get(name)
    if decision is None or not isinstance(decision.value, str):
        return None
    limit = _default_threshold() if threshold is None else threshold
    score = decision.probabilities.get(decision.value)
    if score is None:
        score = decision.confidence
    score = _as_float(score)
    if score is not None and score >= limit:
        return decision.value
    return None


def _parse_answers(answers: Dict[str, dict]) -> Dict[str, Decision]:
    parsed: Dict[str, Decision] = {}
    for name, ans in answers.items():
        if not isinstance(ans, dict):
            continue
        if "choice" in ans:
            parsed[name] = Decision(
                name=name,
                value=ans["choice"],
                confidence=_as_float(ans.get("confidence")),
                probabilities={
                    k: (_as_float(v) or 0.0)
                    for k, v in (ans.get("probabilities") or {}).items()
                },
            )
        elif "noul" in ans:
            value = _as_float(ans.get("noul"))
            if value is not None:
                parsed[name] = Decision(name=name, value=value)
        elif "score" in ans:
            value = _as_float(ans.get("score"))
            if value is not None:
                parsed[name] = Decision(
                    name=name,
                    value=value,
                    confidence=_as_float(ans.get("confidence")),
                )
    return parsed


class DecisionRouter:
    """Ask Jev typed questions about a state, with cache + circuit breaker."""

    def __init__(self, config: Optional[dict] = None, transport=None):
        self._config_override = config
        self._transport = transport
        self._cache: "OrderedDict[Tuple, Dict[str, Decision]]" = OrderedDict()
        self._client: Optional[JevClient] = None
        self._client_key: Optional[Tuple] = None

    def _config(self) -> dict:
        defaults = {
            "enabled": True,
            "endpoint": "https://api.typesafe.ai/v1/systemone",
            "model": "jev-latest",
            "api_key": "",
            "api_key_env": "JEV_API_KEY",
            "api_key_env_fallback": "TYPESAFE_API_KEY",
            "confidence_threshold": 0.8,
            "timeout_seconds": 4.0,
            "state_char_limit": 2000,
            "cache_size": 256,
            "failure_threshold": 3,
            "failure_cooldown_seconds": 120.0,
        }
        cfg = dict(defaults)
        if self._config_override is not None:
            cfg.update(self._config_override)
        else:
            try:
                cfg.update(load_settings().get("decisions", {}) or {})
            except Exception:
                pass
        env_flag = _env_flag()
        if env_flag is not None:
            cfg["enabled"] = env_flag
        return cfg

    def _api_key(self, cfg: dict) -> Optional[str]:
        if cfg.get("api_key"):
            return str(cfg["api_key"]).strip()
        for env_name in (cfg.get("api_key_env"), cfg.get("api_key_env_fallback")):
            if env_name and os.getenv(env_name, "").strip():
                return os.getenv(env_name).strip()
        return None

    @property
    def enabled(self) -> bool:
        cfg = self._config()
        return bool(cfg.get("enabled")) and self._api_key(cfg) is not None

    def _get_client(self, cfg: dict, api_key: str) -> JevClient:
        key = (cfg.get("endpoint"), cfg.get("model"), cfg.get("timeout_seconds"))
        if self._client is None or self._client_key != key:
            self._client = JevClient(
                api_key=api_key,
                endpoint=cfg["endpoint"],
                model=cfg["model"],
                timeout_seconds=float(cfg["timeout_seconds"]),
                transport=self._transport,
            )
            self._client_key = key
        return self._client

    async def ask(
        self, state: str, questions: Dict[str, dict]
    ) -> Dict[str, Decision]:
        """Ask Jev; returns {} whenever it cannot answer reliably."""
        cfg = self._config()
        api_key = self._api_key(cfg)
        if not cfg.get("enabled") or not api_key or not questions:
            return {}

        cache_key = (
            hash(state),
            tuple(
                sorted(
                    (name, json.dumps(q, sort_keys=True))
                    for name, q in questions.items()
                )
            ),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        breaker = get_breaker(
            "jev_decisions",
            failure_threshold=int(cfg.get("failure_threshold", 3)),
            recovery_timeout=float(cfg.get("failure_cooldown_seconds", 120.0)),
        )
        if not breaker.can_execute():
            return {}

        limit = int(cfg.get("state_char_limit", 2000) or 0)
        trimmed = state if limit <= 0 else state[:limit]
        try:
            answers, _usage = await self._get_client(cfg, api_key).ask(
                trimmed, questions
            )
        except Exception as exc:
            breaker.record_failure()
            log_step(
                f"Jev decisions unavailable, falling back to heuristics: {exc}",
                symbol="⚠️",
            )
            return {}
        breaker.record_success()

        decisions = _parse_answers(answers)
        self._cache[cache_key] = decisions
        cache_size = int(cfg.get("cache_size", 256) or 0)
        while cache_size and len(self._cache) > cache_size:
            self._cache.popitem(last=False)

        summary = ", ".join(
            f"{n}={d.value if isinstance(d.value, str) else round(d.value, 2)}"
            for n, d in decisions.items()
        )
        log_step(f"Jev decisions: {summary}", symbol="🧭")
        return decisions


_router: Optional[DecisionRouter] = None


def get_decision_router() -> DecisionRouter:
    global _router
    if _router is None:
        _router = DecisionRouter()
    return _router


async def ask_query_decisions(query: str) -> Dict[str, Decision]:
    """One batched Jev call with all query-routing questions for AgentLoop4."""
    if not query:
        return {}
    from core.decisions.questions import query_routing_questions

    try:
        return await get_decision_router().ask(query, query_routing_questions())
    except Exception as exc:  # decisions must never break the loop
        log_step(f"Query decision routing failed: {exc}", symbol="⚠️")
        return {}


async def resolve_skill_intent(query: str, skill_manager=None) -> Optional[str]:
    """Regex-first skill matching, upgraded by a Jev choice when confident."""
    if skill_manager is None:
        from core.skills.manager import skill_manager as default_manager

        skill_manager = default_manager

    if not skill_manager.registry_file.exists():
        skill_manager.initialize()
    regex_hit = skill_manager.match_intent(query)

    registry: Dict[str, dict] = {}
    if skill_manager.registry_file.exists():
        try:
            registry = json.loads(skill_manager.registry_file.read_text()) or {}
        except Exception:
            registry = {}
    if not registry:
        return regex_hit

    router = get_decision_router()
    if not router.enabled:
        return regex_hit

    from core.decisions.questions import skill_intent_question

    try:
        decisions = await router.ask(query, skill_intent_question(registry))
    except Exception:
        return regex_hit
    pick = decision_choice(decisions, "skill")
    if pick and pick != "none" and pick in registry:
        if regex_hit and regex_hit != pick:
            log_step(
                f"Jev skill routing overrode regex match '{regex_hit}' -> '{pick}'",
                symbol="🧭",
            )
        return pick
    return regex_hit
