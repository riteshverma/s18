# Computer Use (Cua Driver proxy)

`server_computer.py` is a safety proxy over the [Cua Driver](https://github.com/trycua/cua)
(`cua-driver mcp`), giving planner agents native desktop control on Windows/macOS/Linux.
It spawns the driver as a child process, discovers its tool registry at runtime, and
re-exposes each tool as `computer_<name>` with the driver's own input schema
(e.g. `computer_click`, `computer_type_text`, `computer_get_window_state`), plus:

- `computer_driver_status` — is the driver reachable, how many tools exposed
- `computer_refresh_tools` — re-discover tools after a driver update

## Requirements

Install the driver (Windows): `irm https://cua.ai/driver/install.ps1 | iex`, then verify
with `cua-driver call list_apps`. If the driver is missing, the server still starts and
exposes only the status/refresh tools.

## Safety controls

- **Default denylist** — `set_config`, `kill_app`, `clipboard_read`, `install_ffmpeg`,
  `replay_trajectory`, `check_for_update`, deprecated legacy tools. Override with
  `S18_COMPUTER_TOOL_DENYLIST` (comma-separated); the denylist always wins.
- **Allowlist** — `S18_COMPUTER_TOOL_ALLOWLIST` (comma-separated or `*`, the default).
- **Audit log** — every call is appended as JSON to
  `data/system/computer_audit.jsonl` (override `S18_COMPUTER_AUDIT_PATH`; empty string
  disables). Argument strings are truncated to 200 chars.
- **Permission mode** — set `S18_COMPUTER_PERMISSION_MODE` to pass a
  `CUA_DRIVER_PERMISSION_MODE` to the child (see cua.ai docs for `bounded` mode and
  capability manifests; all other `CUA_DRIVER_*` envs pass through untouched).

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `S18_COMPUTER_DRIVER_CMD` | `["cua-driver", "mcp"]` | JSON list overriding the driver command |
| `S18_COMPUTER_TOOL_ALLOWLIST` | `*` | Comma-separated tool names or `*` |
| `S18_COMPUTER_TOOL_DENYLIST` | see above | Comma-separated blocked tool names |
| `S18_COMPUTER_AUDIT_PATH` | `data/system/computer_audit.jsonl` | Audit JSONL path (`""` disables) |
| `S18_COMPUTER_CONNECT_TIMEOUT` | `15` | Startup handshake timeout (seconds) |
| `S18_COMPUTER_PERMISSION_MODE` | unset | Maps to child `CUA_DRIVER_PERMISSION_MODE` |

## Model note

Computer use is a vision task. The Ollama default (`gemma3:4b`) is too weak to drive a
GUI reliably — route computer-use runs to the Gemini arm or another strong vision model,
otherwise expect the planner to misuse `element_index` addressing.
