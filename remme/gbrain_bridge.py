import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config.settings_loader import load_settings


class GBrainBridge:
    """
    Local markdown mirror for REMME -> GBrain compatibility.

    This bridge intentionally starts as file-based so dual-write can be validated
    without requiring a live GBrain runtime. MCP wiring can be enabled separately.
    """

    def __init__(self) -> None:
        cfg = load_settings().get("remme", {}).get("gbrain", {})
        root = cfg.get("mirror_dir", "memory/gbrain_bridge")
        self.root = Path(__file__).parent.parent / root
        self.pages_dir = self.root / "pages"
        self.meta_path = self.root / "mapping.json"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.mapping = self._load_mapping()

    @staticmethod
    def is_enabled() -> bool:
        return bool(load_settings().get("remme", {}).get("gbrain", {}).get("enabled", False))

    @staticmethod
    def dual_write_enabled() -> bool:
        cfg = load_settings().get("remme", {}).get("gbrain", {})
        return bool(cfg.get("enabled", False) and cfg.get("dual_write", False))

    @staticmethod
    def read_from_bridge_enabled() -> bool:
        cfg = load_settings().get("remme", {}).get("gbrain", {})
        return bool(cfg.get("enabled", False) and cfg.get("read_from_bridge", False))

    def _load_mapping(self) -> Dict[str, Dict]:
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_mapping(self) -> None:
        self.meta_path.write_text(json.dumps(self.mapping, indent=2), encoding="utf-8")

    @staticmethod
    def _slugify(text: str) -> str:
        lowered = text.lower().strip()
        lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
        return lowered.strip("-") or "untitled"

    def _memory_slug(self, memory_id: str) -> str:
        return f"remme-memory-{memory_id}"

    def _memory_path(self, memory_id: str) -> Path:
        return self.pages_dir / f"{self._memory_slug(memory_id)}.md"

    def upsert_memory(self, memory: Dict) -> Path:
        memory_id = str(memory.get("id", "unknown"))
        created_at = memory.get("created_at") or datetime.now().isoformat()
        updated_at = memory.get("updated_at") or created_at
        text = str(memory.get("text", "")).strip()
        category = str(memory.get("category", "general"))
        source = str(memory.get("source", "unknown"))
        title = self._slugify(text[:80] if text else memory_id)

        content = (
            f"---\n"
            f"type: remme_memory\n"
            f"title: {title}\n"
            f"remme_id: {memory_id}\n"
            f"category: {category}\n"
            f"source: {source}\n"
            f"created_at: {created_at}\n"
            f"updated_at: {updated_at}\n"
            f"---\n\n"
            f"{text}\n\n"
            f"---\n\n"
            f"- {updated_at}: synced from REMME ({category})\n"
        )
        path = self._memory_path(memory_id)
        path.write_text(content, encoding="utf-8")

        self.mapping[memory_id] = {
            "slug": self._memory_slug(memory_id),
            "path": str(path),
            "updated_at": updated_at,
            "deleted": False,
        }
        self._save_mapping()
        return path

    def mark_deleted(self, memory_id: str) -> bool:
        memory_id = str(memory_id)
        path = self._memory_path(memory_id)
        if path.exists():
            deleted_at = datetime.now().isoformat()
            path.write_text(
                (
                    "---\n"
                    "type: remme_memory_tombstone\n"
                    f"remme_id: {memory_id}\n"
                    f"deleted_at: {deleted_at}\n"
                    "---\n\n"
                    "This memory was deleted from REMME.\n"
                ),
                encoding="utf-8",
            )

        current = self.mapping.get(memory_id, {})
        current.update({"deleted": True, "updated_at": datetime.now().isoformat()})
        self.mapping[memory_id] = current
        self._save_mapping()
        return True

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Bridge read mode fallback: lightweight keyword search over mirrored pages.
        """
        query_terms = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 1]
        if not query_terms:
            return []

        scored: List[Dict] = []
        for memory_id, meta in self.mapping.items():
            if meta.get("deleted"):
                continue
            page = Path(meta.get("path", ""))
            if not page.exists():
                continue
            text = page.read_text(encoding="utf-8", errors="ignore")
            lowered = text.lower()
            matches = sum(1 for t in query_terms if re.search(rf"\b{re.escape(t)}\b", lowered))
            if matches <= 0:
                continue
            scored.append(
                {
                    "id": memory_id,
                    "text": text.split("\n---\n", maxsplit=1)[0].split("\n\n", maxsplit=1)[-1].strip() or text[:240],
                    "score": 1.0 / (1.0 + matches),
                    "source": "gbrain_bridge",
                }
            )
        scored.sort(key=lambda x: x["score"])
        return scored[:k]

    def sync_hubs_snapshot(self, snapshot: Dict) -> Path:
        """
        Persist a GBrain-compatible profile page from REMME structured hubs.
        """
        ts = datetime.now().isoformat()
        path = self.pages_dir / "remme-user-profile.md"
        body = json.dumps(snapshot, indent=2, ensure_ascii=True)
        content = (
            "---\n"
            "type: remme_profile\n"
            "title: remme-user-profile\n"
            f"updated_at: {ts}\n"
            "---\n\n"
            "Current structured user profile exported from REMME hubs.\n\n"
            "```json\n"
            f"{body}\n"
            "```\n"
        )
        path.write_text(content, encoding="utf-8")
        return path
