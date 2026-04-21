import json
import logging
from pathlib import Path
from typing import Any, Dict

from prometheus_client import Counter

logger = logging.getLogger("integration_profiles")
PROFILE_LOAD_FAILURES_TOTAL = Counter(
    "wiseai_integration_profile_load_failures_total",
    "Total integration profile load failures",
    ["integration_id", "workflow_id", "contract_version"],
)


def load_integration_profile(integration_id: str, workflow_id: str, contract_version: str) -> Dict[str, Any]:
    config_dir = Path(__file__).resolve().parents[1] / "config" / "integrations"
    integration_key = (integration_id or "default").strip().lower()
    workflow_key = (workflow_id or "generic").strip().lower()
    version_key = (contract_version or "v1").strip().lower()
    profile_path = config_dir / f"{integration_key}_{workflow_key}_{version_key}.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            PROFILE_LOAD_FAILURES_TOTAL.labels(
                integration_id=integration_key,
                workflow_id=workflow_key,
                contract_version=version_key,
            ).inc()
            logger.warning("Failed to load profile '%s': %s", profile_path, exc)
            return {}
    return {}
