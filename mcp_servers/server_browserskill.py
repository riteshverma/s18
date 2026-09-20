"""S18 authenticated-browser proxy over the BrowserSkill CLI (Tencent/BrowserSkill).

Drives the user's *real, logged-in* Chromium through the ``bsk`` CLI, which
talks to a local daemon and browser extension. Each command is exposed as a
``bsk_<name>`` FastMCP tool, plus ``bsk_cli_status`` and a policy-checked
``bsk_run`` passthrough for commands without a dedicated wrapper.

This is deliberately separate from ``server_browser.py``: that server browses
anonymously (search + extraction + headless vision), while this one reuses the
user's existing login state in a visible Agent Window.

Safety posture:
- A default denylist drops the commands that can exfiltrate or bypass audit
  (``evaluate`` runs arbitrary JS against authenticated origins, ``record``
  captures the user's own keystrokes, ``upload`` discloses local files to the
  site, ``daemon``/``update``/``install-skill`` touch shared state).
- Policy is enforced at ONE chokepoint shared by every wrapper and by
  ``bsk_run``, so the denylist is a real boundary rather than a naming
  convention.
- Every call is appended to a JSONL audit log. ``fill``/``select`` values are
  recorded as a length only unless ``S18_BSK_AUDIT_VALUES=1``, since form
  values carry passwords and OTPs.
- Files written by ``screenshot --out`` and ``download --out`` are confined to
  an artifact directory, so site-controlled bytes cannot land in the repo.

Environment overrides:
- ``S18_BSK_BIN``               CLI binary, default "bsk"
- ``S18_BSK_HOME``              maps to the child's BSK_HOME
- ``S18_BSK_AUTO_START``        maps to the child's BSK_AUTO_START
- ``S18_BSK_COMMAND_ALLOWLIST`` comma-separated commands or "*" (default *)
- ``S18_BSK_COMMAND_DENYLIST``  comma-separated commands (default below)
- ``S18_BSK_AUDIT_PATH``        audit JSONL path; "" disables
- ``S18_BSK_AUDIT_VALUES``      "1" to record raw fill/select values
- ``S18_BSK_ARTIFACT_DIR``      screenshot/download output root
- ``S18_BSK_TIMEOUT``           default per-command seconds (default 120)
- ``S18_BSK_MAX_TIMEOUT``       per-command ceiling (default 600)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

try:
    from stdio_safety import configure_mcp_stdio_logging
except ImportError:  # package-style import (tests)
    from .stdio_safety import configure_mcp_stdio_logging

configure_mcp_stdio_logging()

# Windows: ProactorEventLoop required for asyncio subprocess.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = logging.getLogger("browserskill_proxy")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = "bsk"

# Commands never exposed unless explicitly allowlisted. Matching is on the
# command head ("daemon" covers "daemon start") or the full command string.
DEFAULT_DENYLIST = (
    "evaluate",       # arbitrary JS on authenticated origins -> token/cookie theft
    "record",         # records the user's own actions, incl. password/SSO pages
    "upload",         # discloses agent-local files to the site
    "daemon",         # lifecycle of a daemon shared with the user's other agents
    "update",         # self-updating binary
    "install-skill",  # rewrites harness skill files outside this repo
    "config",         # persistent CLI configuration
)

# Commands whose output is structured when --json is passed.
JSON_CAPABLE = frozenset({"session start", "session list", "status", "browsers"})

_COMMAND_TOKEN = re.compile(r"^[a-z][a-z0-9-]*$")
_REF = re.compile(r"^@[A-Za-z0-9_.:-]+$")

server = FastMCP("s18-browserskill")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _env_list(name: str) -> List[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_denied(command: str) -> bool:
    """Single policy check shared by every wrapper and by bsk_run."""
    allow = _env_list("S18_BSK_COMMAND_ALLOWLIST") or ["*"]
    deny = set(_env_list("S18_BSK_COMMAND_DENYLIST")) or set(DEFAULT_DENYLIST)
    parts = command.split()
    head = parts[0] if parts else command
    if command in deny or head in deny:
        return True
    return not ("*" in allow or command in allow or head in allow)


def _command_prefix() -> List[str]:
    """The argv prefix that invokes the CLI; a JSON list allows a wrapper."""
    raw = (os.getenv("S18_BSK_BIN") or "").strip()
    if not raw:
        return [DEFAULT_BIN]
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return [str(part) for part in parsed]
        except json.JSONDecodeError:
            logger.warning("S18_BSK_BIN looks like JSON but does not parse; using it as a path")
    return [raw]


def _child_env() -> dict:
    env = dict(os.environ)
    for src, dst in (("S18_BSK_HOME", "BSK_HOME"), ("S18_BSK_AUTO_START", "BSK_AUTO_START")):
        value = os.getenv(src)
        if value:
            env[dst] = value
    return env


def _timeout(requested: Optional[float]) -> float:
    default = float(os.getenv("S18_BSK_TIMEOUT") or 120)
    ceiling = float(os.getenv("S18_BSK_MAX_TIMEOUT") or 600)
    return min(float(requested or default), ceiling)


def _artifact_dir() -> Path:
    raw = os.getenv("S18_BSK_ARTIFACT_DIR")
    return Path(raw) if raw else REPO_ROOT / "data" / "system" / "browserskill"


def _confined_out(out: str) -> str:
    """Resolve an --out path, refusing anything outside the artifact dir.

    Downloads carry site-controlled bytes and screenshots are model-named, so
    neither may choose where in the filesystem it lands.
    """
    base = _artifact_dir().resolve()
    candidate = Path(out)
    candidate = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"output path must stay under {base}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def _check_ref(ref: str) -> str:
    """Refs go in argv positionally; a '--flag'-shaped ref would be parsed as one."""
    if not _REF.match(ref or ""):
        raise ValueError(f"invalid element ref {ref!r}; expected the @eN form from observe")
    return ref


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _audit_path() -> Optional[Path]:
    raw = os.getenv("S18_BSK_AUDIT_PATH")
    if raw == "":
        return None
    if raw:
        return Path(raw)
    return REPO_ROOT / "data" / "system" / "browserskill_audit.jsonl"


def _sanitize(argv: List[str]) -> List[str]:
    """Truncate long argv entries and redact form values unless opted in."""
    keep_values = (os.getenv("S18_BSK_AUDIT_VALUES") or "").strip() == "1"
    out: List[str] = []
    redact_next = False
    for item in argv:
        item = str(item)
        if redact_next and not keep_values:
            out.append(f"[redacted len={len(item)}]")
            redact_next = False
            continue
        redact_next = item == "--value"
        out.append(item if len(item) <= 200 else item[:200] + "...[truncated]")
    return out


def _audit(command: str, argv: List[str], ok: bool, error: str = "") -> None:
    path = _audit_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "argv": _sanitize(argv),
            "ok": ok,
        }
        if error:
            entry["error"] = error[:300]
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError as exc:
        logger.warning("Failed to write browserskill audit log: %s", exc)


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def _flags(**kwargs: Any) -> List[str]:
    """Build CLI flags, dropping None/False and rendering True as a bare flag."""
    argv: List[str] = []
    for name, value in kwargs.items():
        if value is None or value is False:
            continue
        flag = "--" + name.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


async def _run(command: str, args: Optional[List[str]] = None, timeout: Optional[float] = None) -> str:
    """Run one bsk command. The only path to the binary; policy is enforced here."""
    args = [str(a) for a in (args or [])]
    if _is_denied(command):
        _audit(command, args, ok=False, error="blocked by proxy allow/deny policy")
        return f"ERROR: command '{command}' is blocked by the browserskill proxy policy"

    parts = command.split()
    if not parts or not all(_COMMAND_TOKEN.match(p) for p in parts):
        _audit(command, args, ok=False, error="malformed command")
        return f"ERROR: malformed bsk command {command!r}"

    prefix = _command_prefix()
    argv = [*prefix, *parts, *args]
    if command in JSON_CAPABLE and "--json" not in args:
        argv.append("--json")

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_child_env(),
        )
    except FileNotFoundError:
        _audit(command, args, ok=False, error="bsk binary not found")
        return (
            f"ERROR: '{prefix[0]}' not found. Install the BrowserSkill CLI "
            "(https://github.com/Tencent/BrowserSkill) and its browser extension, "
            "or set S18_BSK_BIN."
        )
    except OSError as exc:
        _audit(command, args, ok=False, error=str(exc))
        return f"ERROR: could not launch bsk: {exc}"

    limit = _timeout(timeout)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=limit)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        _audit(command, args, ok=False, error="timeout")
        return f"ERROR: 'bsk {command}' timed out after {limit:.0f}s"

    out = (stdout or b"").decode("utf-8", "replace").strip()
    err = (stderr or b"").decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        _audit(command, args, ok=False, error=err or out)
        return f"ERROR: bsk {command} exited {proc.returncode}: {err or out or 'no output'}"
    _audit(command, args, ok=True)
    return out or err or "OK"


async def _guarded(command: str, build, timeout: Optional[float] = None) -> str:
    """Run `build()` (which may raise on bad input) then invoke the command."""
    try:
        args = build()
    except ValueError as exc:
        _audit(command, [], ok=False, error=str(exc))
        return f"ERROR: {exc}"
    return await _run(command, args, timeout)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


@server.tool()
async def bsk_session_start(browser: str = None, no_focus: bool = False) -> str:
    """Start a BrowserSkill session in the Agent Window; returns JSON with session_id.

    Pass `browser` (id or label from bsk_browsers) when several are connected.
    Use `no_focus` for background work. Always finish with bsk_session_stop.
    """
    return await _run("session start", _flags(browser=browser, no_focus=no_focus))


@server.tool()
async def bsk_session_stop(session: str) -> str:
    """Stop a session and return borrowed user tabs. Call on success and failure alike."""
    return await _run("session stop", [session])


@server.tool()
async def bsk_session_list() -> str:
    """List active BrowserSkill sessions as JSON."""
    return await _run("session list")


@server.tool()
async def bsk_browsers() -> str:
    """List connected browsers (ids and labels) available to start a session against."""
    return await _run("browsers")


@server.tool()
async def bsk_doctor() -> str:
    """Run the bsk end-to-end setup diagnostic (CLI, daemon, extension)."""
    return await _run("doctor")


# ---------------------------------------------------------------------------
# Navigation and observation
# ---------------------------------------------------------------------------


@server.tool()
async def bsk_navigate(url: str, session: str, tab_id: str = None) -> str:
    """Navigate the session's current tab to a URL. Observe afterwards for fresh refs."""
    return await _run("navigate", [url, *_flags(session=session, tab_id=tab_id)])


@server.tool()
async def bsk_observe(
    session: str,
    tab_id: str = None,
    max_tokens: int = None,
    cursor: str = None,
    probe_hover: bool = False,
) -> str:
    """Read the page as text, controls and @eN refs. The main way to see a page.

    Refs are invalidated by navigation and large DOM changes, so re-observe
    before each interaction. With `max_tokens`, follow a returned next_cursor
    via `cursor`. `probe_hover` touches the live page; use it once, only when a
    control is missing and no hover marker identifies the trigger.
    """
    return await _run(
        "observe",
        _flags(
            session=session,
            tab_id=tab_id,
            max_tokens=max_tokens,
            cursor=cursor,
            probe_hover=probe_hover,
        ),
    )


@server.tool()
async def bsk_snapshot(session: str, tab_id: str = None) -> str:
    """Capture the static accessibility tree for the current page."""
    return await _run("snapshot", _flags(session=session, tab_id=tab_id))


@server.tool()
async def bsk_get_html(session: str, ref: str = None, tab_id: str = None) -> str:
    """Get exact page markup or hidden metadata. Re-observe before acting on what you find."""

    def build():
        head = [_check_ref(ref)] if ref else []
        return head + _flags(session=session, tab_id=tab_id)

    return await _guarded("get-html", build)


@server.tool()
async def bsk_screenshot(
    session: str,
    out: str = None,
    ref: str = None,
    full_page: bool = False,
    scope: str = None,
    tab_id: str = None,
    timeout: float = None,
) -> str:
    """Capture a PNG of the viewport, one element (`ref`), or the full page.

    `out` is a filename resolved under the artifact directory. `ref` and
    `full_page` cannot be combined. `scope` is "follow" (default) or "current".
    Full-page capture can take minutes; raise `timeout` for long pages.
    """

    def build():
        if ref and full_page:
            raise ValueError("ref and full_page cannot be combined")
        head = [_check_ref(ref)] if ref else []
        return head + _flags(
            session=session,
            tab_id=tab_id,
            full_page=full_page,
            scope=scope,
            out=_confined_out(out) if out else None,
            json=True,
        )

    return await _guarded("screenshot", build, timeout)


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------


@server.tool()
async def bsk_click(ref: str, session: str, tab_id: str = None) -> str:
    """Click an element by a ref taken from the most recent observation."""
    return await _guarded(
        "click", lambda: [_check_ref(ref), *_flags(session=session, tab_id=tab_id)]
    )


@server.tool()
async def bsk_fill(ref: str, value: str, session: str, tab_id: str = None) -> str:
    """Fill a form field. The value is redacted in the audit log by default."""
    return await _guarded(
        "fill", lambda: [_check_ref(ref), *_flags(value=value, session=session, tab_id=tab_id)]
    )


@server.tool()
async def bsk_select(ref: str, value: str, session: str, tab_id: str = None) -> str:
    """Select a dropdown option by its value (not its visible label)."""
    return await _guarded(
        "select", lambda: [_check_ref(ref), *_flags(value=value, session=session, tab_id=tab_id)]
    )


@server.tool()
async def bsk_press(key: str, session: str, ref: str = None, tab_id: str = None) -> str:
    """Press a key such as Enter or Tab, optionally targeting an element ref."""

    def build():
        target = _check_ref(ref) if ref else None
        return [key, *_flags(ref=target, session=session, tab_id=tab_id)]

    return await _guarded("press", build)


@server.tool()
async def bsk_hover(ref: str, session: str, tab_id: str = None) -> str:
    """Hover an element to reveal a menu, then observe to get the revealed items' refs."""
    return await _guarded(
        "hover", lambda: [_check_ref(ref), *_flags(session=session, tab_id=tab_id)]
    )


@server.tool()
async def bsk_scroll_to(ref: str, session: str, tab_id: str = None) -> str:
    """Scroll an element into view; returns its bounds in viewport CSS pixels."""
    return await _guarded(
        "scroll-to", lambda: [_check_ref(ref), *_flags(session=session, tab_id=tab_id)]
    )


@server.tool()
async def bsk_wheel(
    session: str,
    delta_y: int = None,
    delta_x: int = None,
    ref: str = None,
    tab_id: str = None,
) -> str:
    """Send wheel input (at least one nonzero delta). Observe to see the page's response."""

    def build():
        if not delta_x and not delta_y:
            raise ValueError("wheel needs a nonzero delta_x or delta_y")
        target = _check_ref(ref) if ref else None
        return _flags(
            delta_x=delta_x,
            delta_y=delta_y,
            ref=target,
            session=session,
            tab_id=tab_id,
        )

    return await _guarded("wheel", build)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


@server.tool()
async def bsk_tab_list(session: str, scope: str = "user") -> str:
    """List tabs. Scope "user" shows the user's own tabs, which must be borrowed to use."""
    return await _run("tab list", _flags(scope=scope, session=session))


@server.tool()
async def bsk_tab_borrow(tab_id: str, session: str, timeout: str = None) -> str:
    """Borrow a user tab into the Agent Window; the user confirms unless they disabled it.

    List tabs first and never invent an id. Do not repeat a pending, denied or
    timed-out borrow. `timeout` (e.g. "120s") changes only the confirmation wait.
    """
    return await _run("tab borrow", [tab_id, *_flags(session=session, timeout=timeout)])


@server.tool()
async def bsk_tab_return(tab_id: str, session: str) -> str:
    """Return a borrowed tab to the user as soon as the step that needed it ends."""
    return await _run("tab return", [tab_id, *_flags(session=session)])


@server.tool()
async def bsk_tab_create(session: str, url: str = None, no_active: bool = False) -> str:
    """Create a tab in the Agent Window. Retain its tab_id when using `no_active`."""
    head = [url] if url else []
    return await _run("tab create", [*head, *_flags(session=session, no_active=no_active)])


# ---------------------------------------------------------------------------
# Human help and diagnostics
# ---------------------------------------------------------------------------


@server.tool()
async def bsk_request_help(session: str, prompt: str, target: str = None) -> str:
    """Ask the user to complete a human-only step: login, CAPTCHA, OTP, payment, consent.

    Use a precise prompt and a fresh `target` ref. If the result is cancelled or
    timed_out, respect it; do not repeat the request.
    """

    def build():
        ref = _check_ref(target) if target else None
        return _flags(session=session, prompt=prompt, target=ref)

    return await _guarded("request-help", build)


@server.tool()
async def bsk_console(session: str, tab_id: str = None) -> str:
    """Read bounded console output for debugging. Follow returned sequence cursors."""
    return await _run("console", _flags(session=session, tab_id=tab_id))


@server.tool()
async def bsk_network(session: str, tab_id: str = None) -> str:
    """Read bounded network activity for debugging. Follow returned sequence cursors."""
    return await _run("network", _flags(session=session, tab_id=tab_id))


@server.tool()
async def bsk_download(
    ref: str, out: str, session: str, overwrite: bool = False, tab_id: str = None
) -> str:
    """Download via an element into the artifact directory. Bytes are site-controlled."""

    def build():
        return [
            _check_ref(ref),
            *_flags(
                out=_confined_out(out), session=session, overwrite=overwrite, tab_id=tab_id
            ),
        ]

    return await _guarded("download", build)


# ---------------------------------------------------------------------------
# Status and passthrough
# ---------------------------------------------------------------------------


@server.tool()
async def bsk_cli_status() -> str:
    """Report whether the bsk CLI and daemon are reachable, and the active policy.

    Works even when the CLI is absent, so planners can detect the capability.
    """
    prefix = _command_prefix()
    found = shutil.which(prefix[0])
    status: Dict[str, Any] = {
        "binary": " ".join(prefix),
        "installed": bool(found),
        "path": found,
        "denylist": sorted(set(_env_list("S18_BSK_COMMAND_DENYLIST")) or set(DEFAULT_DENYLIST)),
        "allowlist": _env_list("S18_BSK_COMMAND_ALLOWLIST") or ["*"],
        "audit_path": str(_audit_path() or "disabled"),
        "artifact_dir": str(_artifact_dir()),
    }
    if not found:
        status["available"] = False
        status["hint"] = "Install from https://github.com/Tencent/BrowserSkill, then run 'bsk doctor'."
        return json.dumps(status)
    daemon = await _run("status", timeout=20)
    status["available"] = not daemon.startswith("ERROR:")
    status["daemon"] = daemon[:2000]
    return json.dumps(status)


@server.tool()
async def bsk_run(command: str, args: List[str] = None, timeout: float = None) -> str:
    """Run a bsk command that has no dedicated tool (wait, window, history, ...).

    `command` is the subcommand path without the binary, e.g. "tab activate".
    Subject to the same allow/deny policy and audit log as every other tool;
    consult `bsk_run("help")` rather than guessing flags.
    """
    return await _run(command, args or [], timeout)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
