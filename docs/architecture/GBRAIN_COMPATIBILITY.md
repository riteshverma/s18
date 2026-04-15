# GBrain Compatibility Audit (REMME -> GBrain)

This document defines the concrete interoperability model implemented in
`feature/gbrain-memory-bridge`.

## 1) Field Mapping

### REMME memory item -> GBrain page (`type: remme_memory`)

- `id` -> `remme_id` (frontmatter)
- `text` -> compiled truth body
- `category` -> `category` (frontmatter)
- `source` -> `source` (frontmatter)
- `created_at` -> `created_at` (frontmatter)
- `updated_at` -> `updated_at` (frontmatter) + timeline event
- delete action -> tombstone page (`type: remme_memory_tombstone`)

### REMME hubs -> GBrain profile page (`type: remme_profile`)

- `PreferencesHub.to_dict()` -> `preferences` JSON block
- `OperatingContextHub.to_dict()` -> `operating_context` JSON block
- `SoftIdentityHub.to_dict()` -> `soft_identity` JSON block

## 2) Non-transferable or lossy fields

- FAISS internals (`faiss_id`, vector distances) are not represented as native
  GBrain semantic index artifacts.
- Exact REMME dedupe behavior (L2 threshold) does not map 1:1 to GBrain's
  retrieval/ranking behavior.
- REMME staging queue internals are not mirrored; only post-application hub
  snapshots are exported.
- MCP execution traces/circuit-breaker state are intentionally not serialized
  into page content.

## 3) Dual-write and Read Cutover Flags

Configured under `remme.gbrain` in `config/settings.defaults.json`.

- `enabled`: master toggle
- `dual_write`: when true, REMME write operations also mirror to markdown pages
- `read_from_bridge`: when true, run-time memory retrieval reads from bridge
  markdown index instead of FAISS search
- `mirror_dir`: local mirror directory
- `server_id`: expected MCP server key for external `gbrain serve` wiring

## 4) MCP Bridge Registration

`mcp_servers/mcp_config.json` includes a `gbrain` server entry that runs the
official CLI in MCP stdio mode via Bun from a local checkout:

- id: `gbrain`
- command: `bun` (resolved to `~/.bun/bin/bun.exe` on Windows if Bun is not on `PATH`)
- args: `["run", "src/cli.ts", "serve"]`
- `cwd`: `gbrain` (resolved relative to the S18 project root by `MultiMCP`)

### Local install (one-time)

1. Install [Bun](https://bun.sh/) (GBrain is Bun-first).
2. From the S18 project root:

   ```bash
   git clone https://github.com/garrytan/gbrain.git gbrain
   cd gbrain && bun install && bun run src/cli.ts init && cd ..
   ```

   The last command initializes the PGLite brain under the user profile (e.g.
   `%USERPROFILE%\.gbrain\` on Windows).

3. Keep `gbrain/` out of git if you prefer: it is listed in `.gitignore`.

### Verify MCP from this repo

```bash
uv run python scripts/test_gbrain_mcp_registration.py
uv run python scripts/test_gbrain_mcp_live.py
```

The live test expects `gbrain` to be **enabled** in `mcp_config.json` and reports
`PASS` when the server connects and exposes tools (30+).

## 5) Rollout Plan (phased)

1. **Phase A (safe default)**: `enabled=false`.
2. **Phase B (dual-write)**: `enabled=true`, `dual_write=true`,
   `read_from_bridge=false` to validate mirrored page quality.
3. **Phase C (read cutover)**: set `read_from_bridge=true` for canary traffic.
4. **Rollback**: set `read_from_bridge=false` (and optionally `dual_write=false`)
   to immediately revert to REMME FAISS reads.
