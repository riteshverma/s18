"""Public import surface for S18 orchestration core."""

from core.loop import AgentLoop4
from integrations.contracts import (
    CanonicalRunRequest,
    CanonicalRunResponse,
    CanonicalRunResult,
)
from integrations.registry import get_integration_adapter

__all__ = [
    "AgentLoop4",
    "CanonicalRunRequest",
    "CanonicalRunResponse",
    "CanonicalRunResult",
    "get_integration_adapter",
]
