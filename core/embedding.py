import os
import sys
from typing import Any, Optional

import numpy as np
import requests

from config.settings_loader import get_model, get_ollama_url, get_timeout, load_settings

EMBED_MODEL = get_model("embedding")
OLLAMA_TIMEOUT = get_timeout()


def _extract_embedding(body: dict[str, Any]) -> Optional[list[float]]:
    if "embedding" in body:
        return body["embedding"]
    if "embeddings" in body and body["embeddings"]:
        first = body["embeddings"][0]
        if isinstance(first, dict):
            return first.get("embedding")
        return first
    return None


def _embedding_provider() -> str:
    provider = str(load_settings().get("models", {}).get("embedding_provider", "")).strip().lower()
    return provider or "ollama"


def _azure_embed_request(inputs: list[str], timeout: int) -> list[list[float]]:
    cfg = load_settings().get("azure_openai", {})
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or cfg.get("endpoint", "")).rstrip("/")
    api_version = os.getenv("OPENAI_API_VERSION") or cfg.get("api_version", "2024-10-21")
    deployment = (
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        or cfg.get("embedding_deployment")
        or get_model("embedding")
    )
    key_env = cfg.get("api_key_env", "AZURE_OPENAI_API_KEY")
    api_key = os.getenv(key_env) or os.getenv("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not deployment or not api_key:
        raise RuntimeError("Azure OpenAI embedding is selected but not fully configured.")

    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
    response = requests.post(
        url,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        json={"input": inputs},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    vectors = [item.get("embedding", []) for item in data]
    if not vectors:
        raise RuntimeError("Azure OpenAI returned no embedding vectors.")
    return vectors


def _normalize_vector(values: list[float]) -> np.ndarray:
    vec = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 0:
        raise RuntimeError("Embedding norm is zero")
    return vec / norm


def get_normalized_embedding(
    text: str,
    *,
    task_type: Optional[str] = None,
    timeout: Optional[int] = None,
) -> np.ndarray:
    prefix = f"{task_type}: " if task_type else ""
    full_text = prefix + text if task_type and not text.startswith(prefix) else text
    request_timeout = timeout or OLLAMA_TIMEOUT

    if _embedding_provider() == "azure_openai":
        vectors = _azure_embed_request([full_text], request_timeout)
        return _normalize_vector(vectors[0])

    last_error: Optional[Exception] = None
    embed_url = get_ollama_url("embed")
    try:
        response = requests.post(
            embed_url,
            json={"model": EMBED_MODEL, "input": full_text},
            timeout=request_timeout,
        )
        response.raise_for_status()
        embedding = _extract_embedding(response.json())
        if embedding is not None:
            return _normalize_vector(embedding)
    except Exception as exc:
        last_error = exc

    embeddings_url = get_ollama_url("embeddings")
    try:
        response = requests.post(
            embeddings_url,
            json={"model": EMBED_MODEL, "prompt": full_text},
            timeout=request_timeout,
        )
        response.raise_for_status()
        embedding = _extract_embedding(response.json())
        if embedding is not None:
            return _normalize_vector(embedding)
    except Exception as exc:
        last_error = exc

    raise RuntimeError(f"Embedding generation failed: {last_error}") from last_error


def get_batch_normalized_embeddings(
    texts: list[str],
    *,
    task_type: Optional[str] = None,
    timeout: Optional[int] = None,
) -> list[np.ndarray]:
    if not texts:
        return []
    request_timeout = timeout or OLLAMA_TIMEOUT
    prefix = f"{task_type}: " if task_type else ""
    prepared = [
        prefix + t if task_type and not t.startswith(prefix) else t for t in texts
    ]

    if _embedding_provider() == "azure_openai":
        vectors = _azure_embed_request(prepared, request_timeout)
        return [_normalize_vector(v) for v in vectors]

    embed_url = get_ollama_url("embed")
    response = requests.post(
        embed_url,
        json={"model": EMBED_MODEL, "input": prepared},
        timeout=request_timeout,
    )
    response.raise_for_status()
    body = response.json()
    raw_vectors = body.get("embeddings") or body.get("embedding") or []
    if not raw_vectors:
        raise RuntimeError("No embeddings returned from Ollama batch endpoint.")
    return [_normalize_vector(v) for v in raw_vectors]


def try_get_normalized_embedding(
    text: str,
    *,
    task_type: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Optional[np.ndarray]:
    try:
        return get_normalized_embedding(text, task_type=task_type, timeout=timeout)
    except Exception as exc:
        print(f"Embedding generation failed: {exc}", file=sys.stderr)
        return None
