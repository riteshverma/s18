"""
REMME - "Remember Me"

The single source of truth for user knowledge and preferences.

REMME collects signals from multiple sources (conversations, notes, sessions, news),
extracts structured preferences, and serves all agents with user context.

Uses a two-LLM pipeline:
1. Extractor (frequent) - Free-form preference extraction to staging
2. Normalizer (batched) - Maps to schema via LLM + BeliefUpdateEngine

Usage:
    from remme import get_preferences_hub, get_staging_store
    from remme.normalizer import run_normalizer
"""

__all__ = [
    # Core
    "RemmeStore",
    "RemmeExtractor",
    "StagingStore",
    "get_staging_store",
    # Hubs
    "get_preferences_hub",
    "get_operating_context_hub", 
    "get_soft_identity_hub",
    "PreferencesHub",
    "OperatingContextHub",
    "SoftIdentityHub",
    # Engines
    "get_evidence_log",
    "get_belief_engine",
    "EvidenceLog",
    "BeliefUpdateEngine",
    # Bootstrap
    "bootstrap_from_remme",
    "extract_from_memories",
    "apply_extraction_to_hubs",
    # Normalizer
    "Normalizer",
    "run_normalizer",
]


def __getattr__(name):
    if name == "RemmeStore":
        from remme.store import RemmeStore

        return RemmeStore
    if name == "RemmeExtractor":
        from remme.extractor import RemmeExtractor

        return RemmeExtractor
    if name in {"StagingStore", "get_staging_store"}:
        from remme import staging

        return getattr(staging, name)
    if name in {
        "get_preferences_hub",
        "get_operating_context_hub",
        "get_soft_identity_hub",
        "PreferencesHub",
        "OperatingContextHub",
        "SoftIdentityHub",
    }:
        from remme import hubs

        return getattr(hubs, name)
    if name in {"get_evidence_log", "get_belief_engine", "EvidenceLog", "BeliefUpdateEngine"}:
        from remme import engines

        return getattr(engines, name)
    if name in {"bootstrap_from_remme", "extract_from_memories", "apply_extraction_to_hubs"}:
        from remme import bootstrap

        return getattr(bootstrap, name)
    if name in {"Normalizer", "run_normalizer"}:
        from remme import normalizer

        return getattr(normalizer, name)
    raise AttributeError(name)
