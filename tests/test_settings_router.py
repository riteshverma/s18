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


if __name__ == "__main__":
    unittest.main()
