from __future__ import annotations

import shutil
import uuid

import numpy as np
import pytest

import remme.store as remme_store_module
from remme.store import RemmeStore


def _settings_with_policy(policy: dict) -> dict:
    return {
        "azure_openai": {"embedding_dimension": 3},
        "remme": {
            "gbrain": {
                "enabled": False,
                "dual_write": False,
                "read_from_bridge": False,
                "mirror_dir": "memory/test_gbrain_bridge",
            },
            "policy": policy,
        },
    }


@pytest.fixture()
def store_factory(monkeypatch):
    created_paths: list = []

    def _create(policy: dict) -> RemmeStore:
        settings_payload = _settings_with_policy(policy)
        monkeypatch.setattr(remme_store_module, "load_settings", lambda: settings_payload)
        rel_dir = f"memory/test_remme_{uuid.uuid4().hex}"
        store = RemmeStore(persistence_dir=rel_dir)
        created_paths.append(store.root)
        return store

    yield _create

    for path in created_paths:
        shutil.rmtree(path, ignore_errors=True)


def _vec(values: list[float]) -> np.ndarray:
    return np.array(values, dtype="float32")


def test_write_policy_rejects_oversized_text(store_factory):
    store = store_factory(
        {
            "enabled": True,
            "write": {
                "max_text_length": 5,
                "blocked_patterns": [],
                "allowed_sources": [],
                "denied_sources": [],
                "allowed_categories": [],
                "denied_categories": [],
                "default_ttl_seconds": None,
                "source_ttl_overrides": {},
            },
            "read": {
                "allowed_sources": [],
                "denied_sources": [],
                "allowed_categories": [],
                "denied_categories": [],
                "exclude_expired": True,
                "requester_overrides": {},
            },
        }
    )

    with pytest.raises(ValueError, match="blocked add"):
        store.add("too-long-text", _vec([1.0, 0.0, 0.0]), source="manual")


def test_ttl_expired_memories_are_filtered_and_cleaned(store_factory):
    store = store_factory(
        {
            "enabled": True,
            "write": {
                "max_text_length": 2000,
                "blocked_patterns": [],
                "allowed_sources": [],
                "denied_sources": [],
                "allowed_categories": [],
                "denied_categories": [],
                "default_ttl_seconds": None,
                "source_ttl_overrides": {"run_*": 60},
            },
            "read": {
                "allowed_sources": [],
                "denied_sources": [],
                "allowed_categories": [],
                "denied_categories": [],
                "exclude_expired": True,
                "requester_overrides": {},
            },
        }
    )

    item = store.add("short lived memory", _vec([1.0, 0.0, 0.0]), source="run_123")
    assert item.get("expires_at")

    for memory in store.memories:
        if memory["id"] == item["id"]:
            memory["expires_at"] = "2000-01-01T00:00:00"
    store.save()

    results = store.search(_vec([1.0, 0.0, 0.0]), requester="default")
    assert results == []

    deleted = store.cleanup_expired(source="manual_cleanup")
    assert deleted == 1
    assert all(memory["id"] != item["id"] for memory in store.memories)


def test_requester_override_filters_sources(store_factory):
    store = store_factory(
        {
            "enabled": True,
            "write": {
                "max_text_length": 2000,
                "blocked_patterns": [],
                "allowed_sources": [],
                "denied_sources": [],
                "allowed_categories": [],
                "denied_categories": [],
                "default_ttl_seconds": None,
                "source_ttl_overrides": {},
            },
            "read": {
                "allowed_sources": [],
                "denied_sources": [],
                "allowed_categories": [],
                "denied_categories": [],
                "exclude_expired": True,
                "requester_overrides": {
                    "run_context": {
                        "allowed_sources": [],
                        "denied_sources": ["manual"],
                        "allowed_categories": [],
                        "denied_categories": [],
                    }
                },
            },
        }
    )

    store.add("manual memory", _vec([1.0, 0.0, 0.0]), source="manual")
    store.add("run memory", _vec([0.0, 1.0, 0.0]), source="run_abc")

    run_results = store.search(_vec([1.0, 0.0, 0.0]), requester="run_context", score_threshold=10.0)
    run_ids = {item["source"] for item in run_results}
    assert all("manual" not in src for src in run_ids)


def test_consolidation_merges_duplicate_like_entries(store_factory):
    store = store_factory(
        {
            "enabled": True,
            "write": {
                "max_text_length": 2000,
                "blocked_patterns": [],
                "allowed_sources": [],
                "denied_sources": [],
                "allowed_categories": [],
                "denied_categories": [],
                "default_ttl_seconds": None,
                "source_ttl_overrides": {},
            },
            "read": {
                "allowed_sources": [],
                "denied_sources": [],
                "allowed_categories": [],
                "denied_categories": [],
                "exclude_expired": True,
                "requester_overrides": {},
            },
        }
    )

    store.add(
        "User prefers concise markdown responses for architecture updates.",
        _vec([1.0, 0.0, 0.0]),
        source="manual",
    )
    store.add(
        "User prefers concise markdown responses for architecture update notes.",
        _vec([0.0, 1.0, 0.0]),
        source="run_1",
    )

    before = len(store.get_all(apply_read_policy=False))
    report = store.consolidate_memories(dry_run=False, token_overlap_threshold=0.5, min_text_length=10)
    after = len(store.get_all(apply_read_policy=False))

    assert report["duplicates_removed"] >= 1
    assert after < before
