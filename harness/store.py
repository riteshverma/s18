from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from harness.models import HarnessJobState


class HarnessJobStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.base_dir = self._resolve_base_dir(project_root)
        self.jobs_dir = self.base_dir / "jobs"
        self.index_file = self.base_dir / "index.json"

    @staticmethod
    def _resolve_base_dir(project_root: Path) -> Path:
        explicit = os.getenv("S18_HARNESS_STATE_DIR", "").strip()
        if explicit:
            return Path(explicit)

        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / "S18Share" / "harness_jobs"

        # Fallback for non-Windows or restricted environments.
        return Path(tempfile.gettempdir()) / "S18Share" / "harness_jobs"

    def save(self, state: HarnessJobState) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump(mode="json")
        tmp_path = self.jobs_dir / f"{state.id}.json.tmp"
        final_path = self.jobs_dir / f"{state.id}.json"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp_path.replace(final_path)
        self._refresh_index()

    def load(self, job_id: str) -> Optional[HarnessJobState]:
        job_file = self.jobs_dir / f"{job_id}.json"
        if not job_file.exists():
            return None
        data = json.loads(job_file.read_text(encoding="utf-8", errors="ignore"))
        return HarnessJobState.model_validate(data)

    def list_jobs(self, limit: int = 100) -> List[HarnessJobState]:
        rows = self._load_index()
        selected = rows[:limit] if limit > 0 else rows
        result: List[HarnessJobState] = []
        for row in selected:
            job_id = str(row.get("id", "")).strip()
            if not job_id:
                continue
            state = self.load(job_id)
            if state:
                result.append(state)
        return result

    def _refresh_index(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                data = json.loads(job_file.read_text(encoding="utf-8", errors="ignore"))
                rows.append(
                    {
                        "id": data.get("id"),
                        "created_at": data.get("created_at"),
                        "status": data.get("status"),
                        "provider": data.get("provider"),
                    }
                )
            except Exception:
                continue
        rows.sort(key=lambda item: str(item.get("id", "")), reverse=True)
        tmp = self.index_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        tmp.replace(self.index_file)

    def _load_index(self) -> List[dict]:
        if not self.index_file.exists():
            self._refresh_index()
        if not self.index_file.exists():
            return []
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

