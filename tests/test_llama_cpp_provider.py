import asyncio
from types import SimpleNamespace

import numpy as np


def test_llama_cpp_embedding_provider_routes_to_openai_embeddings(monkeypatch):
    from core import embedding as embedding_mod

    fake_settings = {
        "models": {
            "embedding_provider": "llama_cpp",
            "embedding_retry_attempts": 1,
            "embedding_retry_backoff_seconds": 0.01,
            "embedding": "text-embedding-model",
        },
        "llama_cpp": {
            "base_url": "http://127.0.0.1:8080",
            "timeout": 360,
            "endpoints": {"embeddings": "/v1/embeddings"},
        },
    }
    monkeypatch.setattr(embedding_mod, "load_settings", lambda: fake_settings)
    monkeypatch.setattr(embedding_mod, "get_model", lambda _purpose: "text-embedding-model")
    monkeypatch.setattr(embedding_mod, "get_llama_cpp_url", lambda endpoint="embeddings": f"http://127.0.0.1:8080/v1/{endpoint}")

    captured = {}

    class Resp:
        status_code = 200
        response = SimpleNamespace(status_code=200)

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(embedding_mod.requests, "post", fake_post)
    vec = embedding_mod.get_normalized_embedding("hello llama")

    assert isinstance(vec, np.ndarray)
    assert captured["url"] == "http://127.0.0.1:8080/v1/embeddings"
    assert captured["json"]["model"] == "text-embedding-model"
    assert captured["json"]["input"] == ["hello llama"]


def test_model_manager_llama_cpp_generate_uses_chat_completions(monkeypatch):
    from config import settings_loader
    from core.model_manager import ModelManager

    monkeypatch.setattr(
        settings_loader,
        "settings",
        {
            "llama_cpp": {
                "base_url": "http://127.0.0.1:8080",
                "timeout": 360,
                "endpoints": {
                    "chat_completions": "/v1/chat/completions",
                    "embeddings": "/v1/embeddings",
                },
            }
        },
    )
    monkeypatch.setattr(settings_loader, "get_llama_cpp_timeout", lambda: 360)

    manager = ModelManager("Llama-3.2-3B-Instruct", provider="llama_cpp")

    captured = {}

    class FakeResponse:
        async def json(self):
            return {"choices": [{"message": {"content": "ok-from-llama"}}]}

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
    assert result == "ok-from-llama"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["json"]["model"] == "Llama-3.2-3B-Instruct"
    assert captured["json"]["messages"][0]["content"] == "ping"
