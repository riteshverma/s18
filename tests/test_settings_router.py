import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import settings as settings_router


class SettingsRouterValidationTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(settings_router.router)
        self.client = TestClient(self.app)

    def test_put_settings_rejects_unsafe_ollama_base_url(self):
        with patch("routers.settings.reload_settings", return_value={"ollama": {"base_url": "http://127.0.0.1:11434"}}), patch(
            "routers.settings.save_settings"
        ):
            response = self.client.put(
                "/settings",
                json={"settings": {"ollama": {"base_url": "http://169.254.169.254"}}},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("host must be loopback", response.json().get("detail", ""))

    def test_put_settings_rejects_docker_hostname_ollama_base_url(self):
        with patch("routers.settings.reload_settings", return_value={"ollama": {"base_url": "http://127.0.0.1:11434"}}), patch(
            "routers.settings.save_settings"
        ):
            response = self.client.put(
                "/settings",
                json={"settings": {"ollama": {"base_url": "http://s18share-ollama:11434"}}},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("host must be loopback", response.json().get("detail", ""))

    def test_put_settings_rejects_unsafe_llama_cpp_base_url(self):
        with patch("routers.settings.reload_settings", return_value={"llama_cpp": {"base_url": "http://127.0.0.1:8080"}}), patch(
            "routers.settings.save_settings"
        ):
            response = self.client.put(
                "/settings",
                json={"settings": {"llama_cpp": {"base_url": "http://example.com:8080"}}},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("host must be loopback", response.json().get("detail", ""))

    def test_put_settings_rejects_invalid_llama_cpp_endpoint_path(self):
        with patch("routers.settings.reload_settings", return_value={"llama_cpp": {"base_url": "http://127.0.0.1:8080"}}), patch(
            "routers.settings.save_settings"
        ):
            response = self.client.put(
                "/settings",
                json={"settings": {"llama_cpp": {"endpoints": {"chat_completions": "v1/chat/completions"}}}},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("must start with '/'", response.json().get("detail", ""))

    def test_get_settings_redacts_secret_values(self):
        with patch(
            "routers.settings.reload_settings",
            return_value={
                "auth": {"supabase_anon_key": "anon-secret"},
                "supabase_logging": {"service_role_key": "service-role-secret"},
            },
        ):
            response = self.client.get("/settings")

        self.assertEqual(response.status_code, 200)
        returned_settings = response.json()["settings"]
        self.assertEqual(returned_settings["auth"]["supabase_anon_key"], "[redacted]")
        self.assertEqual(
            returned_settings["supabase_logging"]["service_role_key"],
            "[redacted]",
        )

    def test_put_settings_preserves_redacted_secret_placeholders(self):
        current_settings = {
            "auth": {"supabase_anon_key": "anon-secret"},
            "supabase_logging": {"service_role_key": "service-role-secret"},
        }

        with patch("routers.settings.reload_settings", return_value=current_settings), patch(
            "routers.settings.save_settings"
        ):
            response = self.client.put(
                "/settings",
                json={
                    "settings": {
                        "auth": {"supabase_anon_key": "[redacted]"},
                        "supabase_logging": {"service_role_key": "[redacted]"},
                    }
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(settings_router.settings["auth"]["supabase_anon_key"], "anon-secret")
        self.assertEqual(
            settings_router.settings["supabase_logging"]["service_role_key"],
            "service-role-secret",
        )

    def test_settings_routes_require_auth_when_enabled(self):
        with patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "SUPABASE_URL": "https://demo.supabase.co"},
            clear=False,
        ):
            response = self.client.get("/settings")

        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
