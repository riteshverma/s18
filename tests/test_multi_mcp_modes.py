import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings_loader import reload_settings
from mcp_servers.multi_mcp import MultiMCP


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return "ok"


class MultiMcpModeTests(unittest.TestCase):
    def tearDown(self):
        reload_settings()

    def test_legacy_mode_reports_ready_even_before_start_completes(self):
        with patch.dict("os.environ", {"MCP_MODE": "legacy"}, clear=False):
            reload_settings()
            mm = MultiMCP()
            health = mm.get_health_status()

        self.assertEqual(health["mode"], "legacy")
        self.assertTrue(health["mcp_ready"])
        self.assertFalse(health["start_completed"])
        self.assertTrue(mm.should_use_cached_metadata())

    def test_strict_mode_requires_required_servers_for_readiness(self):
        with patch.dict(
            "os.environ",
            {"MCP_MODE": "strict", "MCP_REQUIRED_SERVERS": "rag,mockehr"},
            clear=False,
        ):
            reload_settings()
            mm = MultiMCP()
            health = mm.get_health_status()
            self.assertEqual(health["required_servers"], ["rag", "mockehr"])
            self.assertFalse(health["mcp_ready"])
            self.assertFalse(mm.should_use_cached_metadata())

            mm.sessions = {"rag": object(), "mockehr": object()}
            mm.start_completed = True
            health = mm.get_health_status()

        self.assertTrue(health["mcp_ready"])
        self.assertEqual(sorted(health["connected_servers"]), ["mockehr", "rag"])

    def test_strict_mode_start_raises_when_required_server_missing(self):
        with patch.dict(
            "os.environ",
            {"MCP_MODE": "strict", "MCP_REQUIRED_SERVERS": "rag,mockehr"},
            clear=False,
        ):
            reload_settings()
            mm = MultiMCP()
            mm.server_configs = {"rag": {"enabled": True}, "mockehr": {"enabled": True}}

            async def _fake_start_server(name, config):
                if name == "rag":
                    mm.sessions[name] = object()
                    mm._set_server_result(name, "connected", "metadata_source=live")
                else:
                    mm._set_server_result(name, "failed", "boom")

            mm._start_server = _fake_start_server  # type: ignore[assignment]

            with self.assertRaisesRegex(RuntimeError, "mockehr"):
                asyncio.run(mm.start())

    def test_workspace_trace_context_overrides_caller_supplied_root(self):
        async def _run(tool_name: str, arguments: dict):
            mm = MultiMCP()
            session = _FakeSession()
            mm.sessions = {"sandbox": session}
            mm.tools = {"sandbox": [SimpleNamespace(name=tool_name)]}
            token = mm.set_trace_context({"workspace": "/trusted/workspace"})
            try:
                await mm.route_tool_call(tool_name, arguments)
            finally:
                mm.reset_trace_context(token)
            return session.calls[0][1]

        for tool_name, arguments in (
            ("read_workspace_file", {"path": "notes.md", "workspace_root": "/"}),
            (
                "write_workspace_file",
                {"path": "notes.md", "content": "safe", "workspace_root": "/"},
            ),
        ):
            with self.subTest(tool_name=tool_name):
                routed_args = asyncio.run(_run(tool_name, arguments))
                self.assertEqual(routed_args["workspace_root"], "/trusted/workspace")
                self.assertEqual(routed_args["path"], "notes.md")


if __name__ == "__main__":
    unittest.main()
