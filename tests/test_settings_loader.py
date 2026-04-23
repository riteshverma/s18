import unittest
from unittest.mock import patch

from config.settings_loader import (
    get_model,
    get_mcp_mode,
    get_mcp_required_servers,
    get_mcp_startup_timeout,
    normalize_runtime_ollama_base_url,
    reload_settings,
    reset_settings,
    validate_ollama_base_url,
)


class ValidateOllamaBaseUrlTests(unittest.TestCase):
    def test_accepts_loopback_hosts_and_normalizes_port(self):
        self.assertEqual(
            validate_ollama_base_url("http://127.0.0.1"),
            "http://127.0.0.1:11434",
        )
        self.assertEqual(
            validate_ollama_base_url("http://localhost:11434"),
            "http://localhost:11434",
        )
        self.assertEqual(
            validate_ollama_base_url("https://[::1]:11435"),
            "https://::1:11435",
        )

    def test_rejects_non_loopback_hosts(self):
        with self.assertRaisesRegex(ValueError, "host must be loopback"):
            validate_ollama_base_url("http://169.254.169.254")

        with self.assertRaisesRegex(ValueError, "host must be loopback"):
            validate_ollama_base_url("http://example.com")

    def test_rejects_path_query_and_credentials(self):
        with self.assertRaisesRegex(ValueError, "must not include a path"):
            validate_ollama_base_url("http://127.0.0.1/api")

        with self.assertRaisesRegex(ValueError, "cannot include query or fragment"):
            validate_ollama_base_url("http://127.0.0.1:11434?x=1")

        with self.assertRaisesRegex(ValueError, "cannot include credentials"):
            validate_ollama_base_url("http://user:pass@127.0.0.1:11434")

    def test_rejects_invalid_scheme_and_empty_values(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            validate_ollama_base_url("")

        with self.assertRaisesRegex(ValueError, "must use http or https"):
            validate_ollama_base_url("file://127.0.0.1:11434")


class NormalizeRuntimeOllamaBaseUrlTests(unittest.TestCase):
    def test_accepts_docker_hostnames_for_trusted_runtime(self):
        self.assertEqual(
            normalize_runtime_ollama_base_url("http://ollama:11434"),
            "http://ollama:11434",
        )
        self.assertEqual(
            normalize_runtime_ollama_base_url("http://s18share-ollama"),
            "http://s18share-ollama:11434",
        )

    def test_runtime_path_still_rejects_malformed_urls(self):
        with self.assertRaisesRegex(ValueError, "must not include a path"):
            normalize_runtime_ollama_base_url("http://ollama:11434/api")

        with self.assertRaisesRegex(ValueError, "cannot include query or fragment"):
            normalize_runtime_ollama_base_url("http://ollama:11434?x=1")

        with self.assertRaisesRegex(ValueError, "cannot include credentials"):
            normalize_runtime_ollama_base_url("http://user:pass@ollama:11434")


class McpSettingsTests(unittest.TestCase):
    def tearDown(self):
        reload_settings()

    def test_defaults_to_legacy_mode(self):
        reload_settings()
        self.assertEqual(get_mcp_mode(), "legacy")
        self.assertEqual(get_mcp_required_servers(), [])
        self.assertEqual(get_mcp_startup_timeout(), 5.0)

    def test_env_overrides_mcp_mode_required_servers_and_timeout(self):
        with patch.dict(
            "os.environ",
            {
                "MCP_MODE": "strict",
                "MCP_REQUIRED_SERVERS": "rag,mockehr",
                "MCP_STARTUP_TIMEOUT_SECONDS": "12.5",
            },
            clear=False,
        ):
            reload_settings()
            self.assertEqual(get_mcp_mode(), "strict")
            self.assertEqual(get_mcp_required_servers(), ["rag", "mockehr"])
            self.assertEqual(get_mcp_startup_timeout(), 12.5)


class ProfileSettingsTests(unittest.TestCase):
    def tearDown(self):
        reload_settings()

    def test_profile_overrides_defaults(self):
        with patch.dict("os.environ", {"S18_PROFILE": "local-laptop-gemma"}, clear=False):
            reload_settings()
            self.assertEqual(get_model("semantic_chunking"), "gemma3:4b")

    def test_privacy_first_profile_enables_strict_mcp_mode(self):
        with patch.dict("os.environ", {"S18_PROFILE": "privacy-first"}, clear=False):
            reload_settings()
            self.assertEqual(get_mcp_mode(), "strict")
            self.assertIn("rag", get_mcp_required_servers())


if __name__ == "__main__":
    unittest.main()
