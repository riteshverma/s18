"""
Smoke test: start MultiMCP and verify the gbrain server connects and exposes tools.
Requires: Bun installed, ./gbrain with bun install, gbrain MCP enabled in mcp_config.json.
"""
import asyncio
import json
import sys
from pathlib import Path


async def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    cfg_path = root / "mcp_servers" / "mcp_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    g = cfg.get("gbrain") or {}
    if not g.get("enabled"):
        print("SKIP: gbrain MCP disabled in mcp_config.json")
        return 0

    # Import after cwd is irrelevant
    from mcp_servers.multi_mcp import MultiMCP

    mcp = MultiMCP()
    await mcp.start()
    try:
        if "gbrain" not in mcp.sessions:
            print("FAIL: gbrain session not connected", file=sys.stderr)
            print("connected:", mcp.get_connected_servers(), file=sys.stderr)
            return 1
        result = await mcp.sessions["gbrain"].list_tools()
        n = len(result.tools)
        if n <= 0:
            print("FAIL: gbrain listed 0 tools", file=sys.stderr)
            return 1
        print(f"PASS: gbrain MCP connected, {n} tools")
        return 0
    finally:
        await mcp.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
