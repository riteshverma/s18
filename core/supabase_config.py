import os
from typing import Any, Dict

from config.settings_loader import settings


def get_supabase_config() -> Dict[str, Any]:
    auth_cfg = settings.get("auth", {})
    log_cfg = settings.get("supabase_logging", {})
    return {
        "url": (os.getenv("SUPABASE_URL") or auth_cfg.get("supabase_url") or log_cfg.get("supabase_url") or "").rstrip("/"),
        "anon_key": os.getenv("SUPABASE_ANON_KEY") or auth_cfg.get("supabase_anon_key", ""),
        "jwt_audience": os.getenv("SUPABASE_JWT_AUDIENCE") or auth_cfg.get("supabase_jwt_audience", "authenticated"),
        "service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY") or log_cfg.get("service_role_key", ""),
        "logging_enabled": (
            os.getenv("SUPABASE_LOGGING_ENABLED", str(log_cfg.get("enabled", False))).strip().lower() in {"1", "true", "yes", "on"}
        ),
        "auth_enabled": (os.getenv("AUTH_ENABLED", str(auth_cfg.get("enabled", False))).strip().lower() in {"1", "true", "yes", "on"}),
        "request_table": log_cfg.get("request_table", "ehr_request_log"),
        "result_table": log_cfg.get("result_table", "ehr_clinical_result"),
    }
