import sys
from typing import Any, Optional

import numpy as np
import requests

from config.settings_loader import get_model, get_ollama_url, get_timeout

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


def get_normalized_embedding(
    text: str,
    *,
    task_type: Optional[str] = None,
    timeout: Optional[int] = None,
) -> np.ndarray:
    prefix = f"{task_type}: " if task_type else ""
    full_text = prefix + text if task_type and not text.startswith(prefix) else text
    last_error: Optional[Exception] = None
    request_timeout = timeout or OLLAMA_TIMEOUT

    # Keep URL and user text flows separate so URL construction stays server-controlled.
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
            vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm <= 0:
                raise RuntimeError("Embedding norm is zero")
            return vec / norm
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
            vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm <= 0:
                raise RuntimeError("Embedding norm is zero")
            return vec / norm
    except Exception as exc:
        last_error = exc

    raise RuntimeError(f"Embedding generation failed: {last_error}") from last_error


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
