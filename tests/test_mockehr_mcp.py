import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers.multi_mcp import MultiMCP
from mcp_servers.server_mockehr import get_patient_records, search_labs, _map_payload_to_labs
from core.schemas.clinical import validate_cbc_payload


class _DummyTool:
    def __init__(self, name: str):
        self.name = name


def test_get_patient_records_uses_cache_and_redacts():
    session_state = {
        "Clinical_Full_View": False,
        "retrieved_cache": {
            "patient:p-123": {
                "id": "p-123",
                "external_id": "ext-p-123",
                "created_at": "2026-03-11T10:00:00+00:00",
                "updated_at": "2026-03-11T10:00:00+00:00",
            }
        },
        "last_records": {},
    }

    raw = asyncio.run(get_patient_records("p-123", fresh_sync=False, session_state=session_state))
    payload = json.loads(raw)

    assert payload["status"] == "ok"
    assert payload["source"] == "session_cache"
    assert payload["patient_record"]["id"] == "REDACTED"
    assert payload["patient_record"]["external_id"] == "REDACTED"


def test_get_patient_records_full_view_no_redaction():
    session_state = {
        "Clinical_Full_View": True,
        "retrieved_cache": {
            "patient:p-123": {
                "id": "p-123",
                "external_id": "ext-p-123",
            }
        },
        "last_records": {},
    }

    raw = asyncio.run(get_patient_records("p-123", fresh_sync=False, session_state=session_state))
    payload = json.loads(raw)

    assert payload["status"] == "ok"
    assert payload["patient_record"]["id"] == "p-123"
    assert payload["patient_record"]["external_id"] == "ext-p-123"


def test_search_labs_filters_and_reports_delta():
    session_state = {
        "Clinical_Full_View": False,
        "last_timestamp": "2026-03-11T10:02:00+00:00",
        "retrieved_cache": {
            "labs:p-123": [
                {
                    "id": "lab-1",
                    "name": "Hemoglobin",
                    "value": 10.2,
                    "unit": "g/dL",
                    "date": "2026-03-11T10:05:00+00:00",
                },
                {
                    "id": "lab-2",
                    "name": "WBC",
                    "value": 7.1,
                    "unit": "10^3/uL",
                    "date": "2026-03-11T10:01:00+00:00",
                },
            ]
        },
        "last_records": {"labs:p-123": []},
    }

    raw = asyncio.run(
        search_labs(
            "p-123",
            query="hemo",
            from_ts="2026-03-11T10:03:00+00:00",
            fresh_sync=False,
            session_state=session_state,
        )
    )
    payload = json.loads(raw)

    assert payload["status"] == "ok"
    assert payload["source"] == "session_cache"
    assert len(payload["labs"]) == 1
    assert payload["labs"][0]["name"] == "Hemoglobin"
    assert payload["labs"][0]["id"] == "REDACTED"
    assert len(payload["delta"]) == 1
    assert payload["delta"][0]["change"] == "new"


def test_search_labs_returns_no_record_found():
    session_state = {
        "Clinical_Full_View": False,
        "retrieved_cache": {"labs:p-123": []},
        "last_records": {},
    }

    raw = asyncio.run(search_labs("p-123", query="platelets", fresh_sync=False, session_state=session_state))
    payload = json.loads(raw)
    assert payload["status"] == "No record found"
    assert payload["labs"] == []


def test_get_patient_records_fresh_sync_bypasses_cache():
    """fresh_sync=True bypasses session cache and fetches from primary/fallback."""
    session_state = {
        "Clinical_Full_View": False,
        "retrieved_cache": {"patient:p-999": {"id": "p-999"}},
        "last_records": {},
    }
    raw = asyncio.run(get_patient_records("p-999", fresh_sync=True, session_state=session_state))
    payload = json.loads(raw)
    assert payload["status"] in ("ok", "No record found")
    assert payload["source"] in ("external_mockehr", "s18_internal_history")


def test_search_labs_internal_fallback_from_conversation_history():
    """When the upstream provider is unavailable, labs can come from S18 history."""
    patient_id = "54ce03ff-9186-445c-b8cd-96d3950c7bd1"
    raw = asyncio.run(
        search_labs(
            patient_id,
            fresh_sync=True,
            session_state={"Clinical_Full_View": False},
        )
    )
    payload = json.loads(raw)
    assert payload["status"] in ("ok", "No record found")
    if payload["status"] == "ok":
        assert len(payload["labs"]) > 0
        assert any("Hemoglobin" in str(r.get("name")) for r in payload["labs"])


def test_cbc_payload_validation_normalizes_g_l_to_g_dl_in_labs():
    """Validated CBC payload with unit g/L is normalized; _map_payload_to_labs then gets 14.5 g/dL."""
    raw = {"hemoglobin": 145, "unit": "g/L", "wbc": 7000, "platelets": 250}
    validated, err = validate_cbc_payload(raw)
    assert err is None
    assert validated.hemoglobin == 14.5
    assert validated.wbc == 7.0
    assert validated.platelets == 250000.0
    rows = _map_payload_to_labs(
        "p-123",
        validated.model_dump(),
        "2026-03-17T00:00:00+00:00",
    )
    hemoglobin_rows = [r for r in rows if r.get("name") == "Hemoglobin"]
    assert len(hemoglobin_rows) == 1
    assert hemoglobin_rows[0]["value"] == 14.5
    assert hemoglobin_rows[0]["unit"] == "g/dL"


def test_route_tool_call_prefers_mockehr_for_ehr_tools():
    mm = MultiMCP()
    mm.tools = {
        "alpha": [_DummyTool("get_patient_records")],
        "mockehr": [_DummyTool("get_patient_records")],
    }
    mm.sessions = {"alpha": object(), "mockehr": object()}
    called = {"server": None}

    async def _fake_call_tool(server_name, tool_name, arguments):
        called["server"] = server_name
        return {"ok": True, "server": server_name, "tool": tool_name, "arguments": arguments}

    mm.call_tool = _fake_call_tool  # type: ignore[assignment]

    result = asyncio.run(mm.route_tool_call("get_patient_records", {"patient_id": "p-123"}))
    assert called["server"] == "mockehr"
    assert result["ok"] is True


def test_route_tool_call_forces_trusted_workspace_root():
    mm = MultiMCP()
    mm.tools = {"sandbox": [_DummyTool("write_workspace_file")]}
    mm.sessions = {"sandbox": object()}
    called = {"arguments": None}

    async def _fake_call_tool(server_name, tool_name, arguments):
        called["arguments"] = arguments
        return {"ok": True}

    mm.call_tool = _fake_call_tool  # type: ignore[assignment]
    token = mm.set_trace_context({"workspace": "/trusted/workspace"})
    try:
        result = asyncio.run(
            mm.route_tool_call(
                "write_workspace_file",
                {
                    "workspace_root": "/workspace",
                    "path": "config/settings.json",
                    "content": "{}",
                },
            )
        )
    finally:
        mm.reset_trace_context(token)

    assert result["ok"] is True
    assert called["arguments"]["workspace_root"] == "/trusted/workspace"

