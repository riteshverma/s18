import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from config.settings_loader import settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _logging_settings() -> Dict[str, Any]:
    return settings.get("supabase_logging", {})


def is_logging_enabled() -> bool:
    env_override = os.getenv("SUPABASE_LOGGING_ENABLED")
    if env_override is not None:
        return env_override.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_logging_settings().get("enabled", False))


def _config() -> Dict[str, str]:
    cfg = _logging_settings()
    return {
        "url": (os.getenv("SUPABASE_URL") or cfg.get("supabase_url") or "").rstrip("/"),
        "service_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY") or cfg.get("service_role_key", ""),
        "request_table": cfg.get("request_table", "ehr_request_log"),
        "result_table": cfg.get("result_table", "ehr_clinical_result"),
    }


def compute_payload_hash(query: str, raw_payload: Optional[Dict[str, Any]]) -> str:
    payload = {"query": query, "raw_payload": raw_payload or {}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_idempotency_key(source_system: str, external_event_id: str, payload_hash: str) -> str:
    if external_event_id:
        return f"{source_system}:{external_event_id}"
    return f"{source_system}:sha256:{payload_hash}"


async def _rest_upsert(table: str, row: Dict[str, Any], on_conflict: str) -> Optional[Dict[str, Any]]:
    if not is_logging_enabled():
        return None

    cfg = _config()
    if not cfg["url"] or not cfg["service_key"]:
        return None

    endpoint = f"{cfg['url']}/rest/v1/{table}"
    headers = {
        "apikey": cfg["service_key"],
        "Authorization": f"Bearer {cfg['service_key']}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    params = {"on_conflict": on_conflict}

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(endpoint, headers=headers, params=params, json=[row])
        if resp.status_code >= 300:
            print(f"⚠️ Supabase upsert failed [{table}]: {resp.status_code} {resp.text[:250]}")
            return None
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0]
        return None
    except Exception as exc:
        print(f"⚠️ Supabase upsert error [{table}]: {exc}")
        return None


async def log_inbound_request(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not is_logging_enabled():
        return None
    payload = {
        "run_id": row.get("run_id"),
        "request_id": row.get("request_id"),
        "source_system": row.get("source_system", "s18"),
        "external_event_id": row.get("external_event_id"),
        "idempotency_key": row.get("idempotency_key"),
        "payload_hash": row.get("payload_hash"),
        "query": row.get("query"),
        "raw_payload": row.get("raw_payload"),
        "auth_sub": row.get("auth_sub"),
        "auth_email": row.get("auth_email"),
        "consent_ref": row.get("consent_ref"),
        "status": row.get("status", "accepted"),
        "error_code": row.get("error_code"),
        "created_at": row.get("created_at") or _now_iso(),
        "updated_at": row.get("updated_at") or _now_iso(),
    }
    return await _rest_upsert(_config()["request_table"], payload, on_conflict="idempotency_key")


async def update_request_status(
    *,
    idempotency_key: str,
    run_id: str,
    status: str,
    error_code: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not is_logging_enabled():
        return None
    payload = {
        "idempotency_key": idempotency_key,
        "run_id": run_id,
        "status": status,
        "error_code": error_code,
        "updated_at": _now_iso(),
    }
    return await _rest_upsert(_config()["request_table"], payload, on_conflict="idempotency_key")


async def log_clinical_result(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not is_logging_enabled():
        return None
    payload = {
        "run_id": row.get("run_id"),
        "request_id": row.get("request_id"),
        "idempotency_key": row.get("idempotency_key"),
        "query_type": row.get("query_type", "generic"),
        "normalized_result": row.get("normalized_result"),
        "summary": row.get("summary"),
        "triage_flag": row.get("triage_flag"),
        "status": row.get("status", "completed"),
        "error_code": row.get("error_code"),
        "generated_at": row.get("generated_at") or _now_iso(),
        "updated_at": row.get("updated_at") or _now_iso(),
    }
    return await _rest_upsert(_config()["result_table"], payload, on_conflict="run_id")

