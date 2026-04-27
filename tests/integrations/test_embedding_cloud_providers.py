"""Mock-only tests for cloud embedding providers and batching/retries."""

import os
import sys
import types
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_azure_embedding_batches_inputs(monkeypatch):
    from core import embedding as embedding_mod

    fake_settings = {
        "models": {
            "embedding_provider": "azure_openai",
            "embedding_retry_attempts": 1,
            "embedding_retry_backoff_seconds": 0.01,
        },
        "azure_openai": {
            "endpoint": "https://example.openai.azure.com",
            "api_version": "2024-10-21",
            "embedding_deployment": "emb",
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "batch_size": 2,
        },
    }
    monkeypatch.setattr(embedding_mod, "load_settings", lambda: fake_settings)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")

    calls = {"n": 0}

    class Resp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200
            self.response = types.SimpleNamespace(status_code=200)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        del url, headers, timeout
        calls["n"] += 1
        # Return one embedding per input with explicit index ordering.
        data = [{"index": i, "embedding": [float(i + 1), 0.0]} for i, _ in enumerate(json["input"])]
        return Resp({"data": data})

    monkeypatch.setattr(embedding_mod.requests, "post", fake_post)
    vectors = embedding_mod.get_batch_normalized_embeddings(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert calls["n"] == 3  # batch_size=2 for 5 inputs


def test_azure_embedding_retry_on_503(monkeypatch):
    from core import embedding as embedding_mod

    fake_settings = {
        "models": {
            "embedding_provider": "azure_openai",
            "embedding_retry_attempts": 2,
            "embedding_retry_backoff_seconds": 0.01,
        },
        "azure_openai": {
            "endpoint": "https://example.openai.azure.com",
            "api_version": "2024-10-21",
            "embedding_deployment": "emb",
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "batch_size": 8,
        },
    }
    monkeypatch.setattr(embedding_mod, "load_settings", lambda: fake_settings)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")

    calls = {"n": 0}

    class Resp:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.response = types.SimpleNamespace(status_code=status_code)

        def raise_for_status(self):
            if self.status_code >= 400:
                err = requests.HTTPError("error")
                err.response = types.SimpleNamespace(status_code=self.status_code)
                raise err

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        del url, headers, timeout, json
        calls["n"] += 1
        if calls["n"] == 1:
            return Resp(503)
        return Resp(200, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr(embedding_mod.requests, "post", fake_post)
    vec = embedding_mod.get_normalized_embedding("hello")
    assert isinstance(vec, np.ndarray)
    assert calls["n"] == 2


def test_vertex_embedding_provider_uses_mocked_sdk(monkeypatch):
    from core import embedding as embedding_mod

    class FakeEmbedding:
        def __init__(self, values):
            self.values = values

    class FakeModel:
        def get_embeddings(self, reqs, output_dimensionality=None, auto_truncate=True):
            del output_dimensionality, auto_truncate
            return [FakeEmbedding([1.0, 0.0, 0.0]) for _ in reqs]

    class FakeTextEmbeddingModel:
        @classmethod
        def from_pretrained(cls, _name):
            return FakeModel()

    class FakeTextEmbeddingInput:
        def __init__(self, text, task_type=None):
            self.text = text
            self.task_type = task_type

    fake_aiplatform = types.ModuleType("google.cloud.aiplatform")
    fake_aiplatform.init = lambda project=None, location=None: (project, location)
    fake_vertex_language = types.ModuleType("vertexai.language_models")
    fake_vertex_language.TextEmbeddingModel = FakeTextEmbeddingModel
    fake_vertex_language.TextEmbeddingInput = FakeTextEmbeddingInput
    fake_google = types.ModuleType("google")
    fake_google_cloud = types.ModuleType("google.cloud")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_google_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", fake_aiplatform)
    monkeypatch.setitem(sys.modules, "vertexai.language_models", fake_vertex_language)

    fake_settings = {
        "models": {"embedding_provider": "vertex_ai", "embedding_retry_attempts": 1},
        "vertex_ai": {
            "project": "acme-proj",
            "location": "us-central1",
            "embedding_model": "text-embedding-005",
            "embedding_dimension": 3,
            "batch_size": 2,
        },
    }
    monkeypatch.setattr(embedding_mod, "load_settings", lambda: fake_settings)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "acme-proj")
    out = embedding_mod.get_batch_normalized_embeddings(["a", "b", "c"])
    assert len(out) == 3
    assert all(isinstance(v, np.ndarray) for v in out)
