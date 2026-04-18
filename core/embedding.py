import sys
from typing import Any, Optional

import numpy as np
import requests

from config.settings_loader import get_model, get_ollama_url, get_timeout

EMBED_MODEL = get_model("embedding")
OLLAMA_TIMEOUT = get_timeout()


def _build_embed_candidates(text: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (get_ollama_url("embed"), {"model": EMBED_MODEL, "input": text}),
        (get_ollama_url("embeddings"), {"model": EMBED_MODEL, "prompt": text}),
    ]


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

    for embed_url, payload in _build_embed_candidates(full_text):
        try:
            response = requests.post(embed_url, json=payload, timeout=timeout or OLLAMA_TIMEOUT)
            response.raise_for_status()
            embedding = _extract_embedding(response.json())
            if embedding is None:
                continue
            vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm <= 0:
                raise RuntimeError("Embedding norm is zero")
            return vec / norm
        except Exception as exc:
            last_error = exc
            continue

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
