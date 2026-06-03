"""Unit tests for Ollama-only routing (no llama_cpp)."""
import asyncio
from types import SimpleNamespace

import numpy as np


def test_ollama_embedding_provider_routes_to_ollama_embed(monkeypatch):
    from core import embedding as embedding_mod

    fake_settings = {
        "ollama": {"base_url": "http://127.0.0.1:11434", "timeout": 360},
        "models": {
            "embedding_provider": "ollama",
            "embedding_retry_attempts": 1,
            "embedding_retry_backoff_seconds": 0.01,
            "embedding": "nomic-embed-text",
        },
    }
    monkeypatch.setattr(embedding_mod, "load_settings", lambda: fake_settings)
    monkeypatch.setattr(embedding_mod, "get_model", lambda _purpose: "nomic-embed-text")
    def fake_get_ollama_url(endpoint="generate"):
        base = "http://127.0.0.1:11434"
        paths = {
            "generate": "/api/generate",
            "chat": "/api/chat",
            "embed": "/api/embed",
            "embeddings": "/api/embeddings",
        }
        return f"{base}{paths.get(endpoint, '/api/' + endpoint)}"

    monkeypatch.setattr(embedding_mod, "get_ollama_url", fake_get_ollama_url)

    captured = {}

    class Resp:
        status_code = 200
        response = SimpleNamespace(status_code=200)

        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": [1.0, 0.0, 0.0]}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(embedding_mod.requests, "post", fake_post)
    vec = embedding_mod.get_normalized_embedding("hello ollama")

    assert isinstance(vec, np.ndarray)
    assert captured["url"] == "http://127.0.0.1:11434/api/embed"
    assert captured["json"]["model"] == "nomic-embed-text"
    assert captured["json"]["input"] == "hello ollama"


def test_model_manager_ollama_generate_uses_api_generate(monkeypatch):
    from config import settings_loader
    from core.model_manager import ModelManager

    monkeypatch.setattr(
        settings_loader,
        "settings",
        {"ollama": {"base_url": "http://127.0.0.1:11434", "timeout": 360}},
    )
    monkeypatch.setattr(settings_loader, "get_timeout", lambda: 360)

    manager = ModelManager("gemma4:e4b", provider="ollama")

    captured = {}

    class FakeResponse:
        status = 200
        reason = "OK"
        url = "http://127.0.0.1:11434/api/generate"

        async def json(self):
            return {"response": "ok-from-ollama"}

        async def text(self):
            return ""

        def raise_for_status(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeSession:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        def post(self, url, json=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeClientTimeout:
        def __init__(self, total):
            self.total = total

    fake_aiohttp = SimpleNamespace(ClientSession=FakeSession, ClientTimeout=FakeClientTimeout)
    monkeypatch.setitem(__import__("sys").modules, "aiohttp", fake_aiohttp)

    result = asyncio.run(manager.generate_text("ping"))
    assert result == "ok-from-ollama"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["json"]["model"] == "gemma4:e4b"
    assert captured["json"]["prompt"] == "ping"
    assert captured["json"]["stream"] is False
