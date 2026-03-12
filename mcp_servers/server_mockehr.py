import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from mcp.server.fastmcp import FastMCP


# MCP protocol safety: avoid stdout noise.
def print(*args, **kwargs):
    sys.stderr.write(" ".join(map(str, args)) + "\n")
    sys.stderr.flush()


mcp = FastMCP("mockehr")
PROJECT_ROOT = Path(__file__).parent.parent
CONVERSATION_ROOT = PROJECT_ROOT / "data" / "conversation_history"
WISE_MOCKEHR_BASE_URL = os.getenv("WISE_MOCKEHR_BASE_URL", "").rstrip("/")
HTTP_TIMEOUT_SECONDS = float(os.getenv("MOCKEHR_HTTP_TIMEOUT_SECONDS", "6"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    value = ts.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _normalize_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_session_state(session_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = _normalize_dict(session_state)
    if not parsed:
        return {
            "clinical_full_view": False,
            "last_timestamp": None,
            "retrieved_cache": {},
            "last_records": {},
        }

    return {
        "clinical_full_view": bool(parsed.get("Clinical_Full_View", False)),
        "last_timestamp": parsed.get("last_timestamp"),
        "retrieved_cache": _normalize_dict(parsed.get("retrieved_cache")),
        "last_records": _normalize_dict(parsed.get("last_records")),
    }


def _cached_value(state: Dict[str, Any], cache_key: str) -> Optional[Any]:
    cache = _normalize_dict(state.get("retrieved_cache"))
    return cache.get(cache_key)


def _default_patient(patient_id: str) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "id": patient_id,
        "external_id": f"ext-{patient_id}",
        "created_at": now,
        "updated_at": now,
    }


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_patient_id_from_query(query: str) -> Optional[str]:
    if not query:
        return None
    match = re.search(r"\[Patient ID:\s*([^\]]+)\]", query)
    if match:
        return match.group(1).strip()
    return None


def _extract_payload_from_query(query: str) -> Dict[str, Any]:
    if not query:
        return {}
    marker = "Request:"
    idx = query.find(marker)
    if idx == -1:
        return {}
    payload_text = query[idx + len(marker) :].strip()
    try:
        payload = json.loads(payload_text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _internal_patient_from_history(patient_id: str) -> Optional[Dict[str, Any]]:
    if not CONVERSATION_ROOT.exists():
        return None

    # Search recent sessions first for a matching patient id.
    candidates = sorted(CONVERSATION_ROOT.glob("*/*/*/session_*.json"), reverse=True)
    for path in candidates[:200]:
        content = _safe_load_json(path)
        if not content:
            continue
        context = _normalize_dict(content.get("context"))
        original_query = context.get("original_query", "")
        found_id = _extract_patient_id_from_query(original_query)
        if found_id != patient_id:
            continue
        return _default_patient(patient_id)
    return None


def _map_payload_to_labs(patient_id: str, payload: Dict[str, Any], ts: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mapping = {
        "hemoglobin": ("Hemoglobin", "g/dL"),
        "wbc": ("WBC", "10^3/uL"),
        "rbc": ("RBC", "10^6/uL"),
        "platelets": ("Platelets", "/uL"),
    }
    for key, (name, unit) in mapping.items():
        value = payload.get(key)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except Exception:
            continue
        rows.append(
            {
                "id": f"internal-{key}-{patient_id}",
                "name": name,
                "value": numeric_value,
                "unit": unit,
                "date": ts,
            }
        )
    return rows


def _internal_labs_from_history(patient_id: str) -> List[Dict[str, Any]]:
    if not CONVERSATION_ROOT.exists():
        return []

    rows: List[Dict[str, Any]] = []
    candidates = sorted(CONVERSATION_ROOT.glob("*/*/*/session_*.json"), reverse=True)
    for path in candidates[:240]:
        content = _safe_load_json(path)
        if not content:
            continue
        context = _normalize_dict(content.get("context"))
        original_query = context.get("original_query", "")
        found_id = _extract_patient_id_from_query(original_query)
        if found_id != patient_id:
            continue

        payload = _extract_payload_from_query(original_query)
        if not payload:
            continue
        timestamp = content.get("timestamp") or _now_iso()
        rows.extend(_map_payload_to_labs(patient_id, payload, timestamp))

    # Deduplicate by (name, date) while preserving order.
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("name", "")), str(row.get("date", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _redact_patient_record(record: Dict[str, Any], clinical_full_view: bool) -> Dict[str, Any]:
    if clinical_full_view:
        return record
    redacted = dict(record)
    if "id" in redacted:
        redacted["id"] = "REDACTED"
    if "external_id" in redacted:
        redacted["external_id"] = "REDACTED"
    return redacted


def _redact_labs(labs: List[Dict[str, Any]], clinical_full_view: bool) -> List[Dict[str, Any]]:
    if clinical_full_view:
        return labs
    redacted: List[Dict[str, Any]] = []
    for item in labs:
        row = dict(item)
        if "id" in row:
            row["id"] = "REDACTED"
        redacted.append(row)
    return redacted


def _apply_lab_filters(
    labs: List[Dict[str, Any]],
    query: Optional[str],
    from_ts: Optional[str],
) -> List[Dict[str, Any]]:
    filtered = labs
    if query:
        needle = query.lower()
        filtered = [row for row in filtered if needle in str(row.get("name", "")).lower()]
    if from_ts:
        from_dt = _parse_iso(from_ts)
        if from_dt:
            filtered = [
                row
                for row in filtered
                if (_parse_iso(str(row.get("date"))) or datetime.min.replace(tzinfo=timezone.utc)) >= from_dt
            ]
    return filtered


def _dict_delta(new_obj: Dict[str, Any], old_obj: Dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for key, value in new_obj.items():
        if old_obj.get(key) != value:
            delta[key] = {"old": old_obj.get(key), "new": value}
    return delta


def _lab_delta(new_labs: List[Dict[str, Any]], old_labs: List[Dict[str, Any]], last_ts: Optional[str]) -> List[Dict[str, Any]]:
    old_index = {
        (str(x.get("name", "")), str(x.get("date", ""))): x
        for x in old_labs
    }
    last_dt = _parse_iso(last_ts)
    changes: List[Dict[str, Any]] = []
    for row in new_labs:
        row_dt = _parse_iso(str(row.get("date")))
        if last_dt and row_dt and row_dt <= last_dt:
            continue
        key = (str(row.get("name", "")), str(row.get("date", "")))
        old = old_index.get(key)
        if old is None:
            changes.append({"change": "new", "record": row})
        elif old != row:
            changes.append({"change": "updated", "record": row, "previous": old})
    return changes


# Header sent when calling wise-ai Mock EHR; wise-ai skips nested run_wise_agent when present
_WISE_REQUEST_HEADERS = {"X-Request-Source": "s18"}


async def _fetch_primary_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    if not WISE_MOCKEHR_BASE_URL:
        return None
    url = f"{WISE_MOCKEHR_BASE_URL}/patients/{patient_id}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=_WISE_REQUEST_HEADERS)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


async def _fetch_primary_labs(patient_id: str) -> Optional[List[Dict[str, Any]]]:
    if not WISE_MOCKEHR_BASE_URL:
        return None
    url = f"{WISE_MOCKEHR_BASE_URL}/patients/{patient_id}/labs"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=_WISE_REQUEST_HEADERS)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return None
        return [row for row in payload if isinstance(row, dict)]
    except Exception:
        return None


@mcp.tool()
async def get_patient_records(
    patient_id: str,
    fresh_sync: bool = False,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Retrieve patient demographic/meta record with runtime cache, privacy, and delta handling.
    Returns "No record found" for 404/empty responses.
    """
    state = _parse_session_state(session_state)
    cache_key = f"patient:{patient_id}"
    clinical_full_view = bool(state["clinical_full_view"])

    if not fresh_sync:
        cached = _cached_value(state, cache_key)
        if isinstance(cached, dict) and cached:
            redacted = _redact_patient_record(cached, clinical_full_view)
            return json.dumps(
                {
                    "status": "ok",
                    "source": "session_cache",
                    "delta": _dict_delta(redacted, _normalize_dict(state["last_records"].get(cache_key))),
                    "patient_record": redacted,
                    "timestamp": _now_iso(),
                }
            )

    primary = await _fetch_primary_patient(patient_id)
    source = "wise_mock_api"
    if not primary:
        primary = _internal_patient_from_history(patient_id)
        source = "s18_internal_history"

    if not primary:
        return json.dumps(
            {
                "status": "No record found",
                "source": source,
                "patient_record": None,
                "delta": {},
                "timestamp": _now_iso(),
            }
        )

    redacted = _redact_patient_record(primary, clinical_full_view)
    previous = _normalize_dict(state["last_records"].get(cache_key))
    return json.dumps(
        {
            "status": "ok",
            "source": source,
            "delta": _dict_delta(redacted, previous),
            "patient_record": redacted,
            "timestamp": _now_iso(),
        }
    )


@mcp.tool()
async def search_labs(
    patient_id: str,
    query: Optional[str] = None,
    from_ts: Optional[str] = None,
    fresh_sync: bool = False,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Retrieve patient labs with query/time filtering plus privacy and delta tracking.
    Returns "No record found" for 404/empty responses.
    """
    state = _parse_session_state(session_state)
    cache_key = f"labs:{patient_id}"
    clinical_full_view = bool(state["clinical_full_view"])

    labs: Optional[List[Dict[str, Any]]] = None
    source = "wise_mock_api"
    if not fresh_sync:
        cached = _cached_value(state, cache_key)
        if isinstance(cached, list):
            labs = [x for x in cached if isinstance(x, dict)]
            source = "session_cache"

    if labs is None:
        labs = await _fetch_primary_labs(patient_id)
        if labs is None:
            labs = _internal_labs_from_history(patient_id)
            source = "s18_internal_history"

    filtered = _apply_lab_filters(labs or [], query=query, from_ts=from_ts)
    if not filtered:
        return json.dumps(
            {
                "status": "No record found",
                "source": source,
                "labs": [],
                "delta": [],
                "timestamp": _now_iso(),
            }
        )

    redacted = _redact_labs(filtered, clinical_full_view)
    previous = state["last_records"].get(cache_key)
    previous_rows = previous if isinstance(previous, list) else []
    delta = _lab_delta(redacted, previous_rows, state.get("last_timestamp"))
    return json.dumps(
        {
            "status": "ok",
            "source": source,
            "labs": redacted,
            "delta": delta,
            "timestamp": _now_iso(),
        }
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
