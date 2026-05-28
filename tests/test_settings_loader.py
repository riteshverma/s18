import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config.settings_loader as settings_loader
from config.settings_loader import (
    get_model,
    get_mcp_mode,
    get_mcp_required_servers,
    get_mcp_startup_timeout,
    normalize_runtime_llama_cpp_base_url,
    normalize_runtime_ollama_base_url,
    redact_settings_for_client,
    reload_settings,
    reset_settings,
    validate_llama_cpp_base_url,
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


class ValidateLlamaCppBaseUrlTests(unittest.TestCase):
    def test_accepts_loopback_hosts_and_normalizes_port(self):
        self.assertEqual(
            validate_llama_cpp_base_url("http://127.0.0.1"),
            "http://127.0.0.1:8080",
        )
        self.assertEqual(
            validate_llama_cpp_base_url("http://localhost:8080"),
            "http://localhost:8080",
        )

    def test_rejects_non_loopback_hosts(self):
        with self.assertRaisesRegex(ValueError, "host must be loopback"):
            validate_llama_cpp_base_url("http://example.com")

    def test_rejects_paths_query_and_credentials(self):
        with self.assertRaisesRegex(ValueError, "must not include a path"):
            validate_llama_cpp_base_url("http://127.0.0.1:8080/v1")
        with self.assertRaisesRegex(ValueError, "cannot include query or fragment"):
            validate_llama_cpp_base_url("http://127.0.0.1:8080?x=1")
        with self.assertRaisesRegex(ValueError, "cannot include credentials"):
            validate_llama_cpp_base_url("http://user:pass@127.0.0.1:8080")


class NormalizeRuntimeLlamaCppBaseUrlTests(unittest.TestCase):
    def test_accepts_docker_hostnames_for_trusted_runtime(self):
        self.assertEqual(
            normalize_runtime_llama_cpp_base_url("http://llama-cpp:8080"),
            "http://llama-cpp:8080",
        )
        self.assertEqual(
            normalize_runtime_llama_cpp_base_url("http://llama-server"),
            "http://llama-server:8080",
        )


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
            self.assertEqual(get_model("semantic_chunking"), "gemma4:e4b")

    def test_privacy_first_profile_enables_strict_mcp_mode(self):
        with patch.dict("os.environ", {"S18_PROFILE": "privacy-first"}, clear=False):
            reload_settings()
            self.assertEqual(get_mcp_mode(), "strict")
            self.assertIn("rag", get_mcp_required_servers())
            self.assertIn("sandbox", get_mcp_required_servers())

    def test_distribution_runs_profile_requires_rag_and_sandbox(self):
        with patch.dict("os.environ", {"S18_PROFILE": "distribution-runs"}, clear=False):
            reload_settings()
            self.assertEqual(get_mcp_mode(), "strict")
            self.assertEqual(get_mcp_required_servers(), ["rag", "sandbox"])
            self.assertEqual(get_mcp_startup_timeout(), 10.0)

    def test_env_overrides_llama_cpp_connection(self):
        with patch.dict(
            "os.environ",
            {
                "LLAMA_CPP_BASE_URL": "http://llama-cpp:8080",
                "LLAMA_CPP_TIMEOUT": "420",
            },
            clear=False,
        ):
            loaded = reload_settings()
            self.assertEqual(loaded.get("llama_cpp", {}).get("base_url"), "http://llama-cpp:8080")
            self.assertEqual(loaded.get("llama_cpp", {}).get("timeout"), 420)

    def test_railway_forces_gemini_over_ollama_profile(self):
        with patch.dict(
            "os.environ",
            {
                "RAILWAY_ENVIRONMENT_NAME": "production",
                "GEMINI_API_KEY": "test-key",
                "S18_PROFILE": "local-laptop-gemma",
            },
            clear=False,
        ):
            loaded = reload_settings()
            self.assertEqual(loaded.get("agent", {}).get("model_provider"), "gemini")
            self.assertTrue(
                str(loaded.get("agent", {}).get("default_model", "")).lower().startswith("gemini")
            )

    def test_loopback_ollama_with_gemini_key_forces_gemini_without_railway_env(self):
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "test-key"},
            clear=False,
        ):
            loaded = reload_settings()
            self.assertEqual(loaded.get("agent", {}).get("model_provider"), "gemini")
            self.assertEqual(
                str(loaded.get("models", {}).get("embedding_provider", "")).lower(),
                "sentence_transformers",
            )


class SettingsSecretHandlingTests(unittest.TestCase):
    def tearDown(self):
        settings_loader._settings_cache = None
        reload_settings()

    def test_redact_settings_for_client_hides_secret_values(self):
        redacted = redact_settings_for_client(
            {
                "auth": {
                    "supabase_url": "https://example.supabase.co",
                    "supabase_anon_key": "anon-secret",
                },
                "supabase_logging": {
                    "service_role_key": "service-role-secret",
                },
            }
        )

        self.assertEqual(redacted["auth"]["supabase_url"], "https://example.supabase.co")
        self.assertEqual(redacted["auth"]["supabase_anon_key"], "[redacted]")
        self.assertEqual(redacted["supabase_logging"]["service_role_key"], "[redacted]")

    def test_save_settings_does_not_persist_env_secret_values(self):
        initial_settings = {
            "auth": {"supabase_anon_key": ""},
            "supabase_logging": {"service_role_key": ""},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            defaults_path = Path(tmpdir) / "settings.defaults.json"
            settings_path.write_text(json.dumps(initial_settings))
            defaults_path.write_text(json.dumps(initial_settings))

            with patch.object(settings_loader, "SETTINGS_FILE", settings_path), patch.object(
                settings_loader, "DEFAULTS_FILE", defaults_path
            ), patch.dict(
                "os.environ",
                {
                    "SUPABASE_ANON_KEY": "anon-secret",
                    "SUPABASE_SERVICE_ROLE_KEY": "service-role-secret",
                },
                clear=True,
            ):
                settings_loader._settings_cache = None
                loaded = settings_loader.reload_settings()
                self.assertEqual(loaded["auth"]["supabase_anon_key"], "anon-secret")
                self.assertEqual(
                    loaded["supabase_logging"]["service_role_key"],
                    "service-role-secret",
                )

                settings_loader.save_settings()

            written = json.loads(settings_path.read_text())

        self.assertEqual(written["auth"]["supabase_anon_key"], "")
        self.assertEqual(written["supabase_logging"]["service_role_key"], "")


if __name__ == "__main__":
    unittest.main()
