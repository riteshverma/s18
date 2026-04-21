import unittest

from config.settings_loader import validate_ollama_base_url


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


if __name__ == "__main__":
    unittest.main()
