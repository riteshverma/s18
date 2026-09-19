"""S18 computer-use safety proxy over the Cua Driver (trycua/cua).

Spawns ``cua-driver mcp`` as a child process, discovers its tool
registry at runtime, and re-exposes a filtered subset as FastMCP tools
named ``computer_<driver_tool>`` with their real input schemas, plus
``computer_driver_status`` / ``computer_refresh_tools``.

Safety posture:
- A default denylist drops system-level tools (kill processes, rewrite
  driver config, read the clipboard, replay recorded trajectories that
  would bypass this proxy's audit).
- Every call is appended to a JSONL audit log.
- ``S18_COMPUTER_PERMISSION_MODE`` maps to the child's
  ``CUA_DRIVER_PERMISSION_MODE`` (see cua.ai docs for bounded mode and
  capability manifests; other ``CUA_DRIVER_*`` envs pass through).

Environment overrides:
- ``S18_COMPUTER_DRIVER_CMD``        JSON list, default ["cua-driver", "mcp"]
- ``S18_COMPUTER_TOOL_ALLOWLIST``    comma-separated names or "*" (default *)
- ``S18_COMPUTER_TOOL_DENYLIST``     comma-separated names (default below)
- ``S18_COMPUTER_AUDIT_PATH``        audit JSONL path; "" disables
- ``S18_COMPUTER_CONNECT_TIMEOUT``   startup connect seconds (default 15)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import keyword
import logging
import os
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.fastmcp import FastMCP

try:
    from stdio_safety import configure_mcp_stdio_logging
except ImportError:  # package-style import (tests)
    from .stdio_safety import configure_mcp_stdio_logging

configure_mcp_stdio_logging()

logger = logging.getLogger("computer_proxy")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DRIVER_CMD = ["cua-driver", "mcp"]

# Tools never exposed to planner agents unless explicitly allowlisted.
DEFAULT_DENYLIST = (
    "set_config",        # rewrite persistent driver config
    "kill_app",          # force-terminate arbitrary pids
    "clipboard_read",    # privacy-sensitive clipboard contents
    "install_ffmpeg",    # installs binaries
    "replay_trajectory", # bypasses per-call proxy audit
    "check_for_update",
    "escalate_session",  # deprecated
    "get_session_state", # deprecated
    "page",              # deprecated legacy browser tool
)

server = FastMCP("s18-computer")

_lock = asyncio.Lock()
_stack: Optional[AsyncExitStack] = None
_session: Optional[ClientSession] = None
_registered: set = set()


def _env_list(name: str) -> List[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _driver_command() -> List[str]:
    raw = os.getenv("S18_COMPUTER_DRIVER_CMD")
    if raw:
        try:
            cmd = json.loads(raw)
            if isinstance(cmd, list) and cmd:
                return [str(part) for part in cmd]
        except json.JSONDecodeError:
            logger.warning("S18_COMPUTER_DRIVER_CMD is not a JSON list; using default")
    return list(DEFAULT_DRIVER_CMD)


def _child_env() -> dict:
    env = dict(os.environ)
    mode = os.getenv("S18_COMPUTER_PERMISSION_MODE")
    if mode:
        env["CUA_DRIVER_PERMISSION_MODE"] = mode
    return env


def _audit_path() -> Optional[Path]:
    raw = os.getenv("S18_COMPUTER_AUDIT_PATH")
    if raw == "":
        return None
    if raw:
        return Path(raw)
    return REPO_ROOT / "data" / "system" / "computer_audit.jsonl"


def _is_denied(tool_name: str) -> bool:
    """Single policy check shared by discovery and the call path."""
    allow = _env_list("S18_COMPUTER_TOOL_ALLOWLIST") or ["*"]
    deny = set(_env_list("S18_COMPUTER_TOOL_DENYLIST")) or set(DEFAULT_DENYLIST)
    if tool_name in deny:
        return True
    return not ("*" in allow or tool_name in allow)


def _filter_tools(tools) -> list:
    kept = [t for t in tools if not _is_denied(t.name)]
    dropped = sorted({t.name for t in tools} - {t.name for t in kept})
    if dropped:
        logger.info("Denylisted driver tools not exposed: %s", ", ".join(dropped))
    return kept


_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _py_annotation(json_type: Any) -> type:
    if isinstance(json_type, list):
        for candidate in json_type:
            if candidate != "null":
                json_type = candidate
                break
    return _JSON_TYPE_MAP.get(json_type, Any)


def _make_wrapper(tool):
    """Build an async callable whose signature mirrors the driver tool schema."""
    driver_name = tool.name

    async def _call(**kwargs):
        return await _invoke(driver_name, kwargs)

    schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = []
    for prop_name, prop_meta in properties.items():
        if not prop_name.isidentifier() or keyword.iskeyword(prop_name):
            continue  # non-Python-safe names stay unreachable through the proxy
        annotation = _py_annotation((prop_meta or {}).get("type"))
        if prop_name in required:
            params.append(inspect.Parameter(prop_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation))
        else:
            params.append(inspect.Parameter(prop_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation, default=None))
    _call.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    _call.__name__ = f"computer_{driver_name}"
    _call.__doc__ = (tool.description or f"Cua driver tool {driver_name}.")[:4000]
    return _call


def _register_tools(tools) -> int:
    added = 0
    for tool in tools:
        exposed = f"computer_{tool.name}"
        if exposed in _registered:
            continue
        server.add_tool(_make_wrapper(tool), name=exposed)
        _registered.add(exposed)
        added += 1
    return added


async def _ensure_session() -> ClientSession:
    global _stack, _session
    if _session is not None:
        return _session
    cmd = _driver_command()
    params = StdioServerParameters(command=cmd[0], args=cmd[1:], env=_child_env())
    stack = AsyncExitStack()
    try:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
    except BaseException:
        await stack.aclose()
        raise
    _stack, _session = stack, session
    logger.info("Connected to cua driver: %s", " ".join(cmd))
    return _session


async def _teardown() -> None:
    global _stack, _session
    if _stack is not None:
        try:
            await _stack.aclose()
        except Exception as exc:
            logger.warning("Error closing driver connection: %s", exc)
    _stack, _session = None, None


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:200] + "...[truncated]"
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _audit(tool: str, arguments: Dict[str, Any], ok: bool, error: str = "") -> None:
    path = _audit_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "arguments": _sanitize(arguments),
            "ok": ok,
        }
        if error:
            entry["error"] = error[:300]
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError as exc:
        logger.warning("Failed to write computer-use audit log: %s", exc)


async def _invoke(tool: str, arguments: Dict[str, Any]) -> str:
    if _is_denied(tool):
        _audit(tool, arguments, ok=False, error="blocked by proxy allow/deny policy")
        return f"ERROR: tool '{tool}' is blocked by the computer proxy policy"
    session = await _ensure_session()
    arguments = {k: v for k, v in arguments.items() if v is not None}
    try:
        result = await session.call_tool(tool, arguments or None)
    except Exception as exc:
        _audit(tool, arguments, ok=False, error=str(exc))
        return f"ERROR: driver call failed: {exc}"
    texts = [getattr(block, "text", "") for block in (result.content or [])]
    payload = "\n".join(t for t in texts if t)
    if not payload and getattr(result, "structuredContent", None):
        payload = json.dumps(result.structuredContent, default=str)
    if getattr(result, "isError", False):
        _audit(tool, arguments, ok=False, error=payload)
        return f"ERROR: {payload or 'driver tool reported an error'}"
    _audit(tool, arguments, ok=True)
    return payload or "OK"


async def discover_and_register() -> Dict[str, Any]:
    """Connect to the driver, filter, and expose its tools. Idempotent."""
    async with _lock:
        session = await _ensure_session()
        result = await session.list_tools()
        kept = _filter_tools(result.tools)
        added = _register_tools(kept)
        logger.info("Exposing %d cua driver tools (%d newly registered)", len(kept), added)
        return {
            "driver_tools": len(result.tools),
            "exposed": len(kept),
            "registered_total": len(_registered),
        }


@server.tool()
async def computer_driver_status() -> str:
    """Report whether the cua driver is reachable and which tools are exposed."""
    global _session
    status: Dict[str, Any] = {
        "connected": _session is not None,
        "driver_command": _driver_command(),
        "permission_mode": os.getenv("S18_COMPUTER_PERMISSION_MODE") or os.environ.get("CUA_DRIVER_PERMISSION_MODE", "driver-default"),
        "exposed_tools": len(_registered),
        "audit_path": str(_audit_path() or "disabled"),
    }
    if _session is None:
        status["available"] = False
        return json.dumps(status)
    try:
        result = await asyncio.wait_for(_session.list_tools(), timeout=5.0)
        status["available"] = True
        status["driver_tools"] = len(result.tools)
    except Exception as exc:
        status["available"] = False
        status["error"] = str(exc)[:200]
    return json.dumps(status)


@server.tool()
async def computer_refresh_tools() -> str:
    """Re-discover cua driver tools (e.g. after installing or updating the driver)."""
    try:
        return json.dumps(await discover_and_register())
    except Exception as exc:
        return f"ERROR: refresh failed: {exc}"


async def _main() -> None:
    timeout = float(os.getenv("S18_COMPUTER_CONNECT_TIMEOUT") or 15)
    try:
        summary = await asyncio.wait_for(discover_and_register(), timeout=timeout)
        logger.info("cua driver startup: %s", summary)
    except Exception as exc:
        logger.warning(
            "cua driver unavailable at startup (%s); serving status/refresh tools only", exc
        )
    await server.run_stdio_async()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
