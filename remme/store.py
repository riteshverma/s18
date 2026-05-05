import json
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import sys
import re
from typing import Any
from remme.gbrain_bridge import GBrainBridge

from config.settings_loader import load_settings

class RemmeStore:
    def __init__(self, persistence_dir: str = "memory/remme_index"):
        self.root = Path(__file__).parent.parent / persistence_dir
        self.root.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.root / "index.bin"
        self.metadata_path = self.root / "memories.json"
        self.scanned_runs_path = self.root / "scanned_runs.json"
        
        default_dim = load_settings().get("azure_openai", {}).get("embedding_dimension", 768)
        self.dimension = int(default_dim or 768)
        self.index = None
        self.memories = []
        self.scanned_run_ids = set()
        self.gbrain_bridge = GBrainBridge()
        
        self.load()

    def _load_policy_config(self) -> dict[str, Any]:
        runtime_settings = load_settings()
        remme_raw = runtime_settings.get("remme", {})
        remme_cfg = remme_raw if isinstance(remme_raw, dict) else {}
        policy_cfg = remme_cfg.get("policy", {}) if isinstance(remme_cfg, dict) else {}
        if not isinstance(policy_cfg, dict):
            policy_cfg = {}

        write_cfg = policy_cfg.get("write", {}) if isinstance(policy_cfg.get("write", {}), dict) else {}
        read_cfg = policy_cfg.get("read", {}) if isinstance(policy_cfg.get("read", {}), dict) else {}
        requester_overrides = read_cfg.get("requester_overrides", {})
        if not isinstance(requester_overrides, dict):
            requester_overrides = {}

        return {
            "enabled": bool(policy_cfg.get("enabled", False)),
            "write": {
                "max_text_length": self._positive_int(write_cfg.get("max_text_length"), default=2000),
                "blocked_patterns": self._normalize_str_list(write_cfg.get("blocked_patterns")),
                "allowed_sources": self._normalize_str_list(write_cfg.get("allowed_sources")),
                "denied_sources": self._normalize_str_list(write_cfg.get("denied_sources")),
                "allowed_categories": self._normalize_str_list(write_cfg.get("allowed_categories")),
                "denied_categories": self._normalize_str_list(write_cfg.get("denied_categories")),
                "default_ttl_seconds": self._optional_positive_int(write_cfg.get("default_ttl_seconds")),
                "source_ttl_overrides": {
                    str(key).strip().lower(): self._optional_positive_int(value)
                    for key, value in (write_cfg.get("source_ttl_overrides", {}) or {}).items()
                    if str(key).strip()
                },
            },
            "read": {
                "allowed_sources": self._normalize_str_list(read_cfg.get("allowed_sources")),
                "denied_sources": self._normalize_str_list(read_cfg.get("denied_sources")),
                "allowed_categories": self._normalize_str_list(read_cfg.get("allowed_categories")),
                "denied_categories": self._normalize_str_list(read_cfg.get("denied_categories")),
                "exclude_expired": bool(read_cfg.get("exclude_expired", True)),
                "requester_overrides": {
                    str(key).strip().lower(): {
                        "allowed_sources": self._normalize_str_list((value or {}).get("allowed_sources")),
                        "denied_sources": self._normalize_str_list((value or {}).get("denied_sources")),
                        "allowed_categories": self._normalize_str_list((value or {}).get("allowed_categories")),
                        "denied_categories": self._normalize_str_list((value or {}).get("denied_categories")),
                    }
                    for key, value in requester_overrides.items()
                    if isinstance(value, dict)
                },
            },
        }

    @staticmethod
    def _normalize_str_list(raw_values: Any) -> list[str]:
        if not isinstance(raw_values, list):
            return []
        out: list[str] = []
        for value in raw_values:
            item = str(value or "").strip().lower()
            if item:
                out.append(item)
        return out

    @staticmethod
    def _positive_int(raw: Any, default: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    @staticmethod
    def _optional_positive_int(raw: Any) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, str) and not raw.strip():
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _source_tags(source: str | None) -> list[str]:
        raw = str(source or "")
        parts = [p.strip().lower() for p in raw.split(",")]
        return [p for p in parts if p]

    @staticmethod
    def _source_matches_rule(source_tag: str, rule: str) -> bool:
        if rule.endswith("*"):
            return source_tag.startswith(rule[:-1])
        return source_tag == rule

    def _resolve_ttl_seconds(
        self,
        *,
        source_tags: list[str],
        write_cfg: dict[str, Any],
    ) -> int | None:
        ttl_overrides = write_cfg.get("source_ttl_overrides", {}) or {}
        for tag in source_tags:
            for rule, ttl in ttl_overrides.items():
                if ttl is None:
                    continue
                if self._source_matches_rule(tag, rule):
                    return ttl
        return write_cfg.get("default_ttl_seconds")

    def _is_allowed_by_rules(
        self,
        value: str,
        *,
        allowed: list[str],
        denied: list[str],
    ) -> bool:
        if denied and value in denied:
            return False
        if allowed and value not in allowed:
            return False
        return True

    def _are_sources_allowed(
        self,
        source_tags: list[str],
        *,
        allowed_sources: list[str],
        denied_sources: list[str],
    ) -> bool:
        if not source_tags:
            source_tags = ["unknown"]

        for tag in source_tags:
            for denied_rule in denied_sources:
                if self._source_matches_rule(tag, denied_rule):
                    return False

        if allowed_sources:
            for tag in source_tags:
                if any(self._source_matches_rule(tag, allowed_rule) for allowed_rule in allowed_sources):
                    return True
            return False
        return True

    def _check_write_policy(
        self,
        *,
        action: str,
        text: str | None,
        category: str | None,
        source: str | None,
    ) -> tuple[bool, str]:
        policy = self._load_policy_config()
        if not policy["enabled"]:
            return True, ""

        write_cfg = policy["write"]
        normalized_category = str(category or "general").strip().lower()
        normalized_text = str(text or "").strip()
        source_tags = self._source_tags(source)

        if action in {"add", "update"}:
            if not normalized_text:
                return False, "memory text is empty"
            if len(normalized_text) > write_cfg["max_text_length"]:
                return False, f"memory text exceeds max length {write_cfg['max_text_length']}"
            for pattern in write_cfg["blocked_patterns"]:
                try:
                    if re.search(pattern, normalized_text, flags=re.IGNORECASE):
                        return False, f"memory text matched blocked pattern '{pattern}'"
                except re.error:
                    continue

        if not self._are_sources_allowed(
            source_tags,
            allowed_sources=write_cfg["allowed_sources"],
            denied_sources=write_cfg["denied_sources"],
        ):
            return False, "memory source is not allowed by policy"

        if not self._is_allowed_by_rules(
            normalized_category,
            allowed=write_cfg["allowed_categories"],
            denied=write_cfg["denied_categories"],
        ):
            return False, "memory category is not allowed by policy"

        return True, ""

    @staticmethod
    def _is_expired(memory: dict) -> bool:
        expires_at = str(memory.get("expires_at", "") or "").strip()
        if not expires_at:
            return False
        try:
            parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            return parsed <= datetime.now(parsed.tzinfo)
        except Exception:
            return False

    def _read_policy_for_requester(self, requester: str) -> dict[str, Any]:
        policy = self._load_policy_config()
        read_cfg = policy["read"]
        requester_key = str(requester or "default").strip().lower()
        override = read_cfg.get("requester_overrides", {}).get(requester_key, {})
        merged = {
            "enabled": bool(policy["enabled"]),
            "allowed_sources": override.get("allowed_sources") or read_cfg["allowed_sources"],
            "denied_sources": override.get("denied_sources") or read_cfg["denied_sources"],
            "allowed_categories": override.get("allowed_categories") or read_cfg["allowed_categories"],
            "denied_categories": override.get("denied_categories") or read_cfg["denied_categories"],
            "exclude_expired": bool(read_cfg["exclude_expired"]),
        }
        return merged

    def _allow_read_memory(
        self,
        memory: dict,
        *,
        requester: str,
        apply_read_policy: bool,
    ) -> bool:
        if not apply_read_policy:
            return True

        read_cfg = self._read_policy_for_requester(requester)
        if not read_cfg["enabled"]:
            return True

        if read_cfg["exclude_expired"] and self._is_expired(memory):
            return False

        source_tags = self._source_tags(memory.get("source"))
        if not self._are_sources_allowed(
            source_tags,
            allowed_sources=read_cfg["allowed_sources"],
            denied_sources=read_cfg["denied_sources"],
        ):
            return False

        category = str(memory.get("category", "general") or "general").strip().lower()
        return self._is_allowed_by_rules(
            category,
            allowed=read_cfg["allowed_categories"],
            denied=read_cfg["denied_categories"],
        )

    def _filter_results_by_read_policy(
        self,
        results: list[dict],
        *,
        requester: str,
        apply_read_policy: bool,
    ) -> list[dict]:
        if not apply_read_policy:
            return results
        return [
            result
            for result in results
            if self._allow_read_memory(result, requester=requester, apply_read_policy=True)
        ]

    def load(self):
        """Load index and metadata from disk."""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception as e:
                print(f"Failed to load FAISS index: {e}", file=sys.stderr)
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        if self.metadata_path.exists():
            try:
                self.memories = json.loads(self.metadata_path.read_text())
            except Exception as e:
                print(f"Failed to load memories JSON: {e}", file=sys.stderr)
                self.memories = []
        else:
            self.memories = []

        if self.scanned_runs_path.exists():
            try:
                self.scanned_run_ids = set(json.loads(self.scanned_runs_path.read_text()))
            except Exception as e:
                print(f"Failed to load scanned runs JSON: {e}", file=sys.stderr)
                self.scanned_run_ids = set()
        else:
            self.scanned_run_ids = set()

    def save(self):
        """Save index and metadata to disk."""
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
        
        self.metadata_path.write_text(json.dumps(self.memories, indent=2))
        self.scanned_runs_path.write_text(json.dumps(list(self.scanned_run_ids), indent=2))

    def add(
        self,
        text: str,
        embedding: np.ndarray,
        category: str = "general",
        source: str = "manual",
    ):
        """Add a new memory with deduplication."""
        allowed, reason = self._check_write_policy(
            action="add",
            text=text,
            category=category,
            source=source,
        )
        if not allowed:
            raise ValueError(f"RemMe policy blocked add: {reason}")
        policy_cfg = self._load_policy_config()
        write_cfg = policy_cfg["write"]
        source_tags = self._source_tags(source)
        ttl_seconds = self._resolve_ttl_seconds(source_tags=source_tags, write_cfg=write_cfg)
        now = datetime.now()

        if self.index is None:
            self.dimension = len(embedding)
            self.index = faiss.IndexFlatL2(self.dimension)
            
        # DEDUPLICATION CHECK
        # Search for exact or very similar matches
        # threshold 0.15 is more aggressive than 0.1
        matches = self.search(
            embedding,
            k=1,
            score_threshold=0.15,
            requester="dedup",
            apply_read_policy=False,
        )
        if matches:
            # Update existing memory's timestamp
            memory_id = matches[0]["id"]
            for m in self.memories:
                if m["id"] == memory_id:
                    m["updated_at"] = now.isoformat()
                    # Optionally append source if it's different?
                    if source not in m.get("source", ""):
                        m["source"] = f"{m['source']}, {source}"
                    self.save()
                    return m

        # Add to FAISS
        if self.index is not None and getattr(self.index, "d", len(embedding)) != len(embedding):
            raise ValueError(
                f"Embedding dimension mismatch: index dimension={self.index.d}, incoming={len(embedding)}. "
                "Rebuild REMME index before switching embedding models."
            )
        self.index.add(embedding.reshape(1, -1))
        
        # Add to Metadata
        memory_id = str(uuid.uuid4())
        memory_item = {
            "id": memory_id,
            "text": text,
            "category": category,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "source": source,
            "faiss_id": self.index.ntotal - 1  # 0-indexed ID in FAISS
        }
        if ttl_seconds:
            memory_item["expires_at"] = (now + timedelta(seconds=ttl_seconds)).isoformat()
        self.memories.append(memory_item)
        self.save()
        if self.gbrain_bridge.dual_write_enabled():
            try:
                self.gbrain_bridge.upsert_memory(memory_item)
            except Exception as e:
                print(f"GBrain dual-write add failed: {e}", file=sys.stderr)
        return memory_item

    def search(
        self,
        query_vector: np.ndarray,
        query_text: str = None,
        k: int = 10,
        score_threshold: float = 1.5,
        requester: str = "default",
        apply_read_policy: bool = True,
    ):
        """Search memories by vector similarity with optional keyword boosting."""
        if not self.index or self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(query_vector.reshape(1, -1), k * 2) # Get more candidates for merging
        
        # 1. Gather Vector Results
        vector_results = {}
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            memory = next((m for m in self.memories if m.get("faiss_id") == int(idx)), None)
            if memory:
                if not self._allow_read_memory(
                    memory,
                    requester=requester,
                    apply_read_policy=apply_read_policy,
                ):
                    continue
                score = float(distances[0][i])
                if score < score_threshold:
                    res = memory.copy()
                    res["score"] = score
                    vector_results[memory["id"]] = res

        # 2. Keyword Search & Boosting
        final_results = []
        if query_text:
            import re
            query_words = set(re.findall(r'\b\w+\b', query_text.lower()))
            # Expanded stop words for better precision
            stop_words = {
                "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "you", "your", 
                "have", "has", "had", "any", "about", "of", "our", "to", "what", "we", "in", 
                "with", "from", "for", "and", "or", "but", "so", "how", "when", "where", "why",
                "this", "that", "these", "those", "it", "its", "they", "them", "their",
                "be", "been", "being", "can", "could", "should", "would", "may", "might", "must",
                "shall", "will", "on", "at", "by", "at", "as", "if"
            }
            keywords = query_words - stop_words
            
            if keywords:
                for memory in self.memories:
                    if not self._allow_read_memory(
                        memory,
                        requester=requester,
                        apply_read_policy=apply_read_policy,
                    ):
                        continue
                    text_lower = memory["text"].lower()
                    m_id = memory["id"]
                    
                    # Count whole-word matches only
                    match_count = 0
                    for kw in keywords:
                        if re.search(rf'\b{re.escape(kw)}\b', text_lower):
                            match_count += 1
                    
                    if match_count > 0:
                        # Success! This memory has a keyword match.
                        if m_id in vector_results:
                            # 🚀 BOOST: If found in both, slash the score (lower is better)
                            boost = 1.0 + (match_count * 0.7) # Slightly stronger boost
                            vector_results[m_id]["score"] /= (boost * 2.0)
                            vector_results[m_id]["source"] = f"{vector_results[m_id].get('source', '')} (hybrid_boost)"
                        else:
                            # 💡 INJECT: If only found via keyword, add with competitive score
                            res = memory.copy()
                            res["score"] = 0.6 / (1.0 + match_count) # Competitive synthetic score
                            res["source"] = f"{res.get('source', '')} (keyword_only)"
                            vector_results[m_id] = res

        # 3. Final Sort and Trim
        final_results = sorted(vector_results.values(), key=lambda x: x["score"])
        return final_results[:k]

    def get_all(self, requester: str = "default", apply_read_policy: bool = True):
        """Return memories, optionally filtered by read policy."""
        if not apply_read_policy:
            return [m.copy() for m in self.memories]
        return [
            m.copy()
            for m in self.memories
            if self._allow_read_memory(m, requester=requester, apply_read_policy=True)
        ]

    def cleanup_expired(self, source: str = "system_cleanup") -> int:
        """Delete expired memories from metadata and mirror."""
        expired_ids = [m.get("id") for m in self.memories if self._is_expired(m)]
        deleted = 0
        for memory_id in expired_ids:
            if not memory_id:
                continue
            try:
                self.delete(memory_id, source=source)
                deleted += 1
            except ValueError:
                # Respect policy blocks during cleanup; skip if not allowed.
                continue
        return deleted

    @staticmethod
    def _normalize_memory_text(text: str) -> str:
        cleaned = " ".join((text or "").strip().lower().split())
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _memory_tokens(text: str) -> set[str]:
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "for", "and",
            "or", "in", "on", "with", "as", "by", "this", "that", "it", "be",
        }
        tokens = re.findall(r"\b[a-z0-9][a-z0-9_-]*\b", (text or "").lower())
        return {token for token in tokens if len(token) > 2 and token not in stop_words}

    @classmethod
    def _memorys_are_duplicates(
        cls,
        left_text: str,
        right_text: str,
        *,
        token_overlap_threshold: float,
    ) -> bool:
        left_norm = cls._normalize_memory_text(left_text)
        right_norm = cls._normalize_memory_text(right_text)
        if not left_norm or not right_norm:
            return False
        if left_norm in right_norm or right_norm in left_norm:
            return True
        left_tokens = cls._memory_tokens(left_norm)
        right_tokens = cls._memory_tokens(right_norm)
        if not left_tokens or not right_tokens:
            return False
        intersection = len(left_tokens.intersection(right_tokens))
        union = len(left_tokens.union(right_tokens))
        if union == 0:
            return False
        return (intersection / union) >= token_overlap_threshold

    @staticmethod
    def _merge_source_values(primary_source: str, secondary_source: str) -> str:
        seen: list[str] = []
        for raw in [primary_source, secondary_source]:
            for part in str(raw or "").split(","):
                tag = part.strip()
                if tag and tag not in seen:
                    seen.append(tag)
        return ", ".join(seen)

    def consolidate_memories(
        self,
        *,
        dry_run: bool = False,
        source: str = "system_consolidation",
        token_overlap_threshold: float = 0.82,
        min_text_length: int = 24,
    ) -> dict[str, int]:
        """Merge near-duplicate memory entries and optionally persist changes."""
        memories = [m.copy() for m in self.memories if len(str(m.get("text", "")).strip()) >= min_text_length]
        memories.sort(
            key=lambda item: (
                -len(str(item.get("text", "") or "")),
                str(item.get("updated_at", "") or ""),
            )
        )

        canonical: list[dict] = []
        duplicate_map: dict[str, str] = {}
        updated_canonical = 0

        for memory in memories:
            matched = None
            for candidate in canonical:
                if self._memorys_are_duplicates(
                    str(memory.get("text", "")),
                    str(candidate.get("text", "")),
                    token_overlap_threshold=token_overlap_threshold,
                ):
                    matched = candidate
                    break

            if matched is None:
                canonical.append(memory)
                continue

            duplicate_id = str(memory.get("id", "")).strip()
            canonical_id = str(matched.get("id", "")).strip()
            if duplicate_id and canonical_id and duplicate_id != canonical_id:
                duplicate_map[duplicate_id] = canonical_id

            merged_source = self._merge_source_values(matched.get("source", ""), memory.get("source", ""))
            if merged_source != matched.get("source", ""):
                matched["source"] = merged_source
                updated_canonical += 1

            if len(str(memory.get("text", ""))) > len(str(matched.get("text", ""))):
                matched["text"] = memory.get("text", "")
                updated_canonical += 1

            latest_update = max(
                str(matched.get("updated_at", "") or ""),
                str(memory.get("updated_at", "") or ""),
            )
            if latest_update and latest_update != matched.get("updated_at"):
                matched["updated_at"] = latest_update
                updated_canonical += 1

        duplicates_removed = len(duplicate_map)
        if dry_run or duplicates_removed == 0:
            return {
                "scanned": len(memories),
                "duplicates_removed": duplicates_removed,
                "canonical_updated": updated_canonical,
            }

        canonical_map = {str(item.get("id", "")): item for item in canonical if str(item.get("id", "")).strip()}

        # Respect policy for delete/update operations.
        blocked_deletes = 0
        blocked_updates = 0
        for duplicate_id in list(duplicate_map.keys()):
            original = next((m for m in self.memories if str(m.get("id", "")) == duplicate_id), None)
            allowed, _reason = self._check_write_policy(
                action="delete",
                text=(original or {}).get("text"),
                category=(original or {}).get("category"),
                source=source,
            )
            if not allowed:
                blocked_deletes += 1
                if original is not None:
                    canonical_map[str(original.get("id", ""))] = original
                duplicate_map.pop(duplicate_id, None)

        for memory_id, updated in list(canonical_map.items()):
            original = next((m for m in self.memories if str(m.get("id", "")) == memory_id), None)
            if not original:
                continue
            if original.get("text") == updated.get("text") and original.get("source") == updated.get("source"):
                continue
            allowed, _reason = self._check_write_policy(
                action="update",
                text=updated.get("text"),
                category=updated.get("category"),
                source=source,
            )
            if not allowed:
                blocked_updates += 1
                canonical_map[memory_id] = original

        next_memories = []
        for memory in self.memories:
            memory_id = str(memory.get("id", "")).strip()
            if memory_id in duplicate_map:
                continue
            if memory_id in canonical_map:
                next_memories.append(canonical_map[memory_id])
            else:
                next_memories.append(memory)

        kept_ids = {str(m.get("id", "")) for m in next_memories if str(m.get("id", "")).strip()}
        self.memories = next_memories
        self.save()
        if self.gbrain_bridge.dual_write_enabled():
            for duplicate_id in duplicate_map.keys():
                try:
                    self.gbrain_bridge.mark_deleted(duplicate_id)
                except Exception as e:
                    print(f"GBrain dual-write consolidate delete failed: {e}", file=sys.stderr)
            for memory_id in kept_ids:
                updated = next((m for m in self.memories if str(m.get("id", "")) == memory_id), None)
                if not updated:
                    continue
                try:
                    self.gbrain_bridge.upsert_memory(updated)
                except Exception as e:
                    print(f"GBrain dual-write consolidate update failed: {e}", file=sys.stderr)

        return {
            "scanned": len(memories),
            "duplicates_removed": len(duplicate_map),
            "canonical_updated": updated_canonical,
            "blocked_deletes": blocked_deletes,
            "blocked_updates": blocked_updates,
        }

    def search_text(self, query_text: str, limit: int = 5, requester: str = "default"):
        """Search memories with text input only."""
        if self.gbrain_bridge.read_from_bridge_enabled():
            bridge_results = self.gbrain_bridge.search(query_text, k=limit * 2)
            filtered = self._filter_results_by_read_policy(
                bridge_results,
                requester=requester,
                apply_read_policy=True,
            )
            return filtered[:limit]
        from remme.utils import get_embedding

        query_vector = get_embedding(query_text, task_type="search_query")
        return self.search(
            query_vector,
            query_text=query_text,
            k=limit,
            requester=requester,
            apply_read_policy=True,
        )

    def get_scanned_run_ids(self):
        """Return a set of run IDs that have already been scanned."""
        # 1. Start with dedicated tracking file (Best source)
        ids = set(self.scanned_run_ids)
        
        # 2. Backfill from existing memories if not already there (Legacy support)
        for m in self.memories:
            source = m.get("source", "")
            parts = source.split(", ")
            for part in parts:
                if part.startswith("run_"):
                    ids.add(part.replace("run_", ""))
                elif part.startswith("manual_scan_"):
                    ids.add(part.replace("manual_scan_", ""))
        return ids

    def mark_run_scanned(self, run_id: str):
        """Explicitly mark a run as scanned and persist."""
        if run_id not in self.scanned_run_ids:
            self.scanned_run_ids.add(run_id)
            self.save()

    def delete(self, memory_id: str, source: str = "unknown"):
        """Delete a memory.
        Note: FAISS deletion is complex (requires IDMap or rebuild).
        For simplicity in this V1, we will remove from metadata and rebuild index.
        """
        existing = next((m for m in self.memories if m["id"] == memory_id), None)
        category = existing.get("category") if existing else "general"
        text = existing.get("text") if existing else ""
        allowed, reason = self._check_write_policy(
            action="delete",
            text=text,
            category=category,
            source=source,
        )
        if not allowed:
            raise ValueError(f"RemMe policy blocked delete: {reason}")

        # Remove from memories list
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        
        # Rebuild Index
        new_index = faiss.IndexFlatL2(self.dimension)
        if self.memories:
            # We need embeddings to rebuild. 
            # OPTION 1: Store embeddings in a separate .npy file (Better for large scale)
            # OPTION 2: Re-embed everything (Bad)
            # OPTION 3: Don't support delete yet in FAISS, just soft delete in metadata.
            
            # Going with Option 3/Hybrid for MVP: We accept that the vector exists but we filter it out?
            # No, that affects Top-K.
            
            # Better strategy for Teaching/MVP:
            # We assume we have the embeddings available or re-calculate.
            # Since we didn't store embeddings in JSON (too big), and we want to avoid re-embed cost...
            # We will implement a "Soft Delete" workflow where we filter search results.
            pass
            
        # Re-save metadata (so it's gone from UI)
        self.save()
        if self.gbrain_bridge.dual_write_enabled():
            try:
                self.gbrain_bridge.mark_deleted(memory_id)
            except Exception as e:
                print(f"GBrain dual-write delete failed: {e}", file=sys.stderr)
        
        # Ideally we should rebuild the index cleanly. 
        # For now, let's just mark it deleted in metadata and handle filtering in search
        # or implement a full rebuild if the user edits. 
        # Let's keep it simple: Just Metadata Update. The "Ghost" vector might return but we filter it.
        return True

    def update_text(
        self,
        memory_id: str,
        new_text: str,
        new_embedding: np.ndarray,
        source: str = "unknown",
    ):
        """Update the text of a memory."""
        # 1. Soft delete the old vector (by removing metadata mapping)
        # 2. Add new vector
        
        original_idx = -1
        for i, m in enumerate(self.memories):
            if m["id"] == memory_id:
                original_idx = i
                break
        
        if original_idx != -1:
            category = self.memories[original_idx].get("category", "general")
            allowed, reason = self._check_write_policy(
                action="update",
                text=new_text,
                category=category,
                source=source,
            )
            if not allowed:
                raise ValueError(f"RemMe policy blocked update: {reason}")

            # Modify in place (preserving ID/Created At)
            self.memories[original_idx]["text"] = new_text
            self.memories[original_idx]["updated_at"] = datetime.now().isoformat()
            
            # Update FAISS:
            # As explained in delete, we can't easily "replace" a vector in FlatL2 without IDMap.
            # We will append the new vector and update the faiss_id pointer.
            # The old vector becomes "garbage" (unreachable).
            self.index.add(new_embedding.reshape(1, -1))
            self.memories[original_idx]["faiss_id"] = self.index.ntotal - 1
            
            self.save()
            if self.gbrain_bridge.dual_write_enabled():
                try:
                    self.gbrain_bridge.upsert_memory(self.memories[original_idx])
                except Exception as e:
                    print(f"GBrain dual-write update failed: {e}", file=sys.stderr)
            return True
        return False
