"""Minimal cua-driver MCP stub used by tests/test_computer_server.py.

Speaks real MCP over stdio via FastMCP so the proxy's discovery,
filtering, and call paths are exercised without the actual driver
binary. ``kill_app`` exists to prove the default denylist removes it.
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("fake-cua-driver")


@server.tool()
def list_apps() -> str:
    return "calculator,paint"


@server.tool()
def click(pid: int, element_index: int = -1, x: int = -1, y: int = -1) -> str:
    return f"clicked pid={pid} element_index={element_index} x={x} y={y}"


@server.tool()
def kill_app(pid: int) -> str:
    return f"killed {pid}"


if __name__ == "__main__":
    server.run()
