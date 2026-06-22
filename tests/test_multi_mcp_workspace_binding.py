import asyncio

from mcp_servers.multi_mcp import MultiMCP


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return {"tool": tool_name, "arguments": arguments}


def test_workspace_root_argument_is_removed_without_trace_context():
    session = _FakeSession()
    multi_mcp = MultiMCP()
    multi_mcp.sessions = {"sandbox": session}

    result = asyncio.run(
        multi_mcp.call_tool(
            "sandbox",
            "read_workspace_file",
            {"path": "notes.md", "workspace_root": "/tmp"},
        )
    )

    assert result["arguments"] == {"path": "notes.md"}
    assert session.calls[0][1] == {"path": "notes.md"}


def test_workspace_root_argument_is_overwritten_by_trace_context():
    session = _FakeSession()
    multi_mcp = MultiMCP()
    multi_mcp.sessions = {"sandbox": session}
    token = multi_mcp.set_trace_context({"workspace": "/safe/workspace"})
    try:
        result = asyncio.run(
            multi_mcp.call_tool(
                "sandbox",
                "write_workspace_file",
                {
                    "path": "notes.md",
                    "content": "hello",
                    "workspace_root": "/tmp",
                },
            )
        )
    finally:
        multi_mcp.reset_trace_context(token)

    assert result["arguments"]["workspace_root"] == "/safe/workspace"
    assert session.calls[0][1]["workspace_root"] == "/safe/workspace"
