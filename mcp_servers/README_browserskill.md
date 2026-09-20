# Authenticated Browser Use (BrowserSkill proxy)

`server_browserskill.py` is a safety proxy over the [BrowserSkill](https://github.com/Tencent/BrowserSkill)
`bsk` CLI, letting planner agents drive the user's **real, logged-in Chromium**. Work
happens in a separate, visible Agent Window; the user's own tabs must be explicitly
borrowed. No credentials are ever extracted — the agent reuses the browser's existing
login state.

## How this differs from `server_browser.py`

| | `server_browser.py` (hybrid-browser) | `server_browserskill.py` |
| --- | --- | --- |
| Identity | Anonymous / headless | The user's logged-in profile |
| Mechanism | Playwright, Trafilatura, `browser-use` | `bsk` CLI → local daemon → browser extension |
| Best for | Search, scraping, public pages | Sites behind a login, form flows, UI regression checks |

They are complementary. Prefer `web_search` / `web_extract_text` for public research and
reach for `bsk_*` only when the task needs the user's session.

## Requirements

1. **CLI** — Windows: `irm https://raw.githubusercontent.com/Tencent/BrowserSkill/main/install.ps1 | iex`
   (macOS/Linux: the matching `install.sh`). Installs to `~/.local/bin`.
2. **Extension** — install BrowserSkill for Chrome or Edge from the official store.
3. Verify with `bsk doctor`, or call the `bsk_cli_status` tool.

If the CLI is absent the server still starts; `bsk_cli_status` reports
`installed: false` with an install hint, and every other tool returns a clear error.

## Tools

- **Session** — `bsk_session_start`, `bsk_session_stop`, `bsk_session_list`, `bsk_browsers`, `bsk_doctor`
- **Read** — `bsk_navigate`, `bsk_observe`, `bsk_snapshot`, `bsk_get_html`, `bsk_screenshot`
- **Interact** — `bsk_click`, `bsk_fill`, `bsk_select`, `bsk_press`, `bsk_hover`, `bsk_scroll_to`, `bsk_wheel`
- **Tabs** — `bsk_tab_list`, `bsk_tab_borrow`, `bsk_tab_return`, `bsk_tab_create`
- **Human-in-the-loop** — `bsk_request_help` (login, CAPTCHA, OTP, payment, consent)
- **Diagnostics** — `bsk_console`, `bsk_network`, `bsk_download`
- **Meta** — `bsk_cli_status`, `bsk_run` (policy-checked passthrough for commands without a wrapper)

### Workflow

`bsk_session_start` → `bsk_navigate` → `bsk_observe` → act on the returned `@eN` refs →
`bsk_session_stop`. Refs come from the most recent observation and are invalidated by
navigation or large DOM changes, so re-observe before each interaction. Always stop the
session, on failure as well as success — that also returns borrowed tabs.

## Safety controls

- **Default denylist** — `evaluate` (arbitrary JS against authenticated origins, i.e.
  cookie/token theft), `record` (captures the user's own keystrokes), `upload`
  (discloses agent-local files to the site), `daemon` (lifecycle of a daemon shared with
  the user's other agents), `update`, `install-skill`, `config`. Entries match a command
  head, so `daemon` also covers `daemon start`. Override with
  `S18_BSK_COMMAND_DENYLIST`; the denylist always wins over the allowlist.
- **One chokepoint** — every wrapper *and* the `bsk_run` passthrough route through the
  same policy check before any subprocess starts, so the denylist is a real boundary.
  Command strings are validated token by token, so shell-shaped input never reaches argv.
- **Audit log** — every call is appended as JSON to `data/system/browserskill_audit.jsonl`
  (override `S18_BSK_AUDIT_PATH`; empty string disables). `--value` arguments are recorded
  as a length only, since form fills carry passwords and OTPs; set `S18_BSK_AUDIT_VALUES=1`
  to record them in full.
- **Confined writes** — `bsk_screenshot --out` and `bsk_download --out` resolve under
  `data/system/browserskill/` and refuse traversal, so site-controlled bytes cannot be
  written elsewhere in the repo.
- **User consent stays with the user** — tab borrowing and human-help prompts are governed
  by the extension's own Automation settings. This proxy never tries to bypass them.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `S18_BSK_BIN` | `bsk` | CLI path, or a JSON list to wrap it (`["python","bsk.py"]`) |
| `S18_BSK_HOME` | unset | Maps to the child's `BSK_HOME` |
| `S18_BSK_AUTO_START` | unset | Maps to the child's `BSK_AUTO_START` (`0` to require a running daemon) |
| `S18_BSK_COMMAND_ALLOWLIST` | `*` | Comma-separated commands or `*` |
| `S18_BSK_COMMAND_DENYLIST` | see above | Comma-separated blocked commands |
| `S18_BSK_AUDIT_PATH` | `data/system/browserskill_audit.jsonl` | Audit JSONL path (`""` disables) |
| `S18_BSK_AUDIT_VALUES` | unset | `1` records raw `--value` arguments |
| `S18_BSK_ARTIFACT_DIR` | `data/system/browserskill/` | Screenshot/download output root |
| `S18_BSK_TIMEOUT` | `120` | Default per-command seconds |
| `S18_BSK_MAX_TIMEOUT` | `600` | Per-command ceiling |

The daemon normally auto-starts. Set `S18_BSK_AUTO_START=0` only if you manage the daemon
yourself; see BrowserSkill's [sandboxed-agents guide](https://github.com/Tencent/BrowserSkill/blob/main/docs/sandboxed-agents.md).

## Model note

Reading pages via `bsk_observe` is a text task that any planner arm can handle. Only
`bsk_screenshot` output and Canvas work need vision — route those runs to the Gemini arm
rather than the small Ollama default.
