"""Typed fast decisions (System One / Jev) with heuristic fallback.

Public surface:
    Decision, DecisionRouter, get_decision_router
    ask_query_decisions(query)      -> planning-guard decisions for AgentLoop4
    resolve_skill_intent(query)     -> skill routing upgrade over regex
    decision_flag / decision_choice -> confidence-threshold helpers
    JevClient, JevError             -> raw System One API access
"""

from core.decisions.jev_client import JevClient, JevError
from core.decisions.router import (
    Decision,
    DecisionRouter,
    ask_query_decisions,
    decision_choice,
    decision_flag,
    get_decision_router,
    resolve_skill_intent,
)

__all__ = [
    "Decision",
    "DecisionRouter",
    "JevClient",
    "JevError",
    "ask_query_decisions",
    "decision_choice",
    "decision_flag",
    "get_decision_router",
    "resolve_skill_intent",
]
