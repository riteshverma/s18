import json
from pathlib import Path
from typing import Any, Dict


def load_integration_profile(integration_id: str, workflow_id: str, contract_version: str) -> Dict[str, Any]:
    config_dir = Path(__file__).resolve().parents[1] / "config" / "integrations"
    integration_key = (integration_id or "default").strip().lower()
    workflow_key = (workflow_id or "generic").strip().lower()
    version_key = (contract_version or "v1").strip().lower()
    profile_path = config_dir / f"{integration_key}_{workflow_key}_{version_key}.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
