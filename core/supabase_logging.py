import hashlib
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from core.supabase_config import get_supabase_config

logger = logging.getLogger("supabase_logging")
_http_client: Optional[httpx.AsyncClient] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_logging_enabled() -> bool:
    return bool(get_supabase_config().get("logging_enabled", False))


def _config() -> Dict[str, str]:
    cfg = get_supabase_config()
    return {
        "url": cfg.get("url", ""),
        "service_key": cfg.get("service_role_key", ""),
        "request_table": cfg.get("request_table", "ehr_request_log"),
        "result_table": cfg.get("result_table", "ehr_clinical_result"),
    }


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=12.0)
    return _http_client


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

    retries = 2
    for attempt in range(retries + 1):
        try:
            resp = await _get_http_client().post(endpoint, headers=headers, params=params, json=[row])
            if resp.status_code >= 500 and attempt < retries:
                await asyncio.sleep(0.25 * (attempt + 1))
                continue
            if resp.status_code >= 300:
                logger.warning(
                    "Supabase upsert failed table=%s status=%s run_id=%s body=%s",
                    table,
                    resp.status_code,
                    row.get("run_id"),
                    resp.text[:250],
                )
                return None
            data = resp.json()
            if isinstance(data, list) and data:
                logger.debug("Supabase upsert ok table=%s status=%s run_id=%s", table, resp.status_code, row.get("run_id"))
                return data[0]
            return None
        except Exception as exc:
            if attempt < retries:
                await asyncio.sleep(0.25 * (attempt + 1))
                continue
            logger.warning("Supabase upsert error table=%s run_id=%s error=%s", table, row.get("run_id"), exc)
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

