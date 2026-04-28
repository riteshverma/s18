import os
import random
import sys
import time
from typing import Any, Optional

import numpy as np
import requests

from config.settings_loader import (
    get_llama_cpp_timeout,
    get_llama_cpp_url,
    get_model,
    get_ollama_url,
    get_timeout,
    load_settings,
)

EMBED_MODEL = get_model("embedding")
OLLAMA_TIMEOUT = get_timeout()
LLAMA_CPP_TIMEOUT = get_llama_cpp_timeout()


def _embedding_retry_config() -> tuple[int, float]:
    cfg = load_settings().get("models", {}) or {}
    attempts = int(cfg.get("embedding_retry_attempts") or 3)
    backoff = float(cfg.get("embedding_retry_backoff_seconds") or 0.6)
    return max(1, attempts), max(0.05, backoff)


def _retry_call(fn, *, retryable, attempts: int, backoff: float):
    last_error: Optional[Exception] = None
    for idx in range(attempts):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - behavior tested via call sites
            last_error = exc
            if idx >= attempts - 1 or not retryable(exc):
                raise
            sleep_for = backoff * (2 ** idx) + random.uniform(0, backoff / 5.0)
            time.sleep(min(sleep_for, 5.0))
    raise RuntimeError(f"Embedding call failed: {last_error}") from last_error


def _requests_retryable(exc: Exception) -> bool:
    if isinstance(exc, requests.Timeout):
        return True
    if isinstance(exc, requests.ConnectionError):
        return True
    if isinstance(exc, requests.HTTPError):
        code = getattr(exc.response, "status_code", 0) or 0
        return code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


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


def _bedrock_embed_request(inputs: list[str], timeout: int) -> list[list[float]]:
    """AWS Bedrock embedding provider.

    Lazily imports boto3 and uses the model id from settings. Defaults to
    Titan Text Embeddings V2 which returns 1024-dim vectors.
    """
    try:
        import boto3  # type: ignore[reportMissingImports]
        from botocore.config import Config  # type: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "boto3 must be installed for the bedrock embedding provider. "
            "Install: `pip install boto3`"
        ) from exc

    cfg = load_settings().get("bedrock", {}) or {}
    region = os.getenv("AWS_REGION") or cfg.get("region", "us-east-1")
    model_id = (
        os.getenv("BEDROCK_EMBEDDING_MODEL_ID")
        or cfg.get("embedding_model_id")
        or "amazon.titan-embed-text-v2:0"
    )

    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=timeout, connect_timeout=timeout),
    )

    attempts, backoff = _embedding_retry_config()
    vectors: list[list[float]] = []
    batch_size = int(cfg.get("batch_size") or 16)
    prepared_batches = [
        inputs[i : i + batch_size] for i in range(0, len(inputs), max(1, batch_size))
    ]
    for batch in prepared_batches:
        for text in batch:
            body = {"inputText": text}
            if model_id.startswith("amazon.titan-embed-text-v2"):
                body["dimensions"] = int(cfg.get("embedding_dimension") or 1024)
                body["normalize"] = True
            import json as _json

            def _invoke():
                resp = client.invoke_model(
                    modelId=model_id,
                    body=_json.dumps(body).encode("utf-8"),
                    accept="application/json",
                    contentType="application/json",
                )
                return _json.loads(resp["body"].read())

            payload = _retry_call(
                _invoke,
                retryable=lambda exc: "thrott" in str(exc).lower() or "timeout" in str(exc).lower(),
                attempts=attempts,
                backoff=backoff,
            )
            embedding = payload.get("embedding") or payload.get("embeddings")
            if embedding is None:
                raise RuntimeError(f"Bedrock returned no embedding: {payload}")
            if isinstance(embedding[0], list):
                embedding = embedding[0]
            vectors.append(embedding)
    return vectors


def _vertex_embed_request(inputs: list[str], timeout: int) -> list[list[float]]:
    del timeout
    try:
        from google.cloud import aiplatform  # type: ignore[reportMissingImports]
        from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel  # type: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-aiplatform must be installed for the vertex_ai embedding "
            "provider. Install: `pip install google-cloud-aiplatform`"
        ) from exc

    cfg = load_settings().get("vertex_ai", {}) or {}
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or cfg.get("project", "")
    location = os.getenv("VERTEX_AI_LOCATION") or cfg.get("location", "us-central1")
    model = (
        os.getenv("VERTEX_AI_EMBEDDING_MODEL")
        or cfg.get("embedding_model")
        or "text-embedding-005"
    )
    dimension = int(
        os.getenv("VERTEX_AI_EMBEDDING_DIMENSION")
        or cfg.get("embedding_dimension")
        or 768
    )
    batch_size = int(cfg.get("batch_size") or 32)
    if not project:
        raise RuntimeError("Vertex AI embedding provider requires GOOGLE_CLOUD_PROJECT or vertex_ai.project.")

    aiplatform.init(project=project, location=location)
    model_client = TextEmbeddingModel.from_pretrained(model)
    attempts, backoff = _embedding_retry_config()

    vectors: list[list[float]] = []
    for i in range(0, len(inputs), max(1, batch_size)):
        batch = inputs[i : i + batch_size]

        def _call():
            reqs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in batch]
            return model_client.get_embeddings(
                reqs,
                output_dimensionality=dimension,
                auto_truncate=True,
            )

        result = _retry_call(
            _call,
            retryable=lambda exc: "quota" in str(exc).lower() or "429" in str(exc) or "503" in str(exc),
            attempts=attempts,
            backoff=backoff,
        )
        for emb in result:
            vectors.append(list(emb.values))
    return vectors


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
    batch_size = int(cfg.get("batch_size") or 64)
    attempts, backoff = _embedding_retry_config()
    vectors: list[list[float]] = []
    for i in range(0, len(inputs), max(1, batch_size)):
        chunk = inputs[i : i + batch_size]

        def _post():
            response = requests.post(
                url,
                headers={"api-key": api_key, "Content-Type": "application/json"},
                json={"input": chunk},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()

        payload = _retry_call(
            _post,
            retryable=_requests_retryable,
            attempts=attempts,
            backoff=backoff,
        )
        data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        vectors.extend(item.get("embedding", []) for item in data)
    if not vectors:
        raise RuntimeError("Azure OpenAI returned no embedding vectors.")
    return vectors


def _llama_cpp_embed_request(inputs: list[str], timeout: int) -> list[list[float]]:
    model_name = get_model("embedding")
    url = get_llama_cpp_url("embeddings")
    attempts, backoff = _embedding_retry_config()

    def _post():
        response = requests.post(
            url,
            json={"model": model_name, "input": inputs},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    payload = _retry_call(
        _post,
        retryable=_requests_retryable,
        attempts=attempts,
        backoff=backoff,
    )
    data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    vectors = [item.get("embedding", []) for item in data]
    if not vectors:
        raise RuntimeError("llama.cpp returned no embedding vectors.")
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

    if _embedding_provider() == "bedrock":
        vectors = _bedrock_embed_request([full_text], request_timeout)
        return _normalize_vector(vectors[0])

    if _embedding_provider() == "vertex_ai":
        vectors = _vertex_embed_request([full_text], request_timeout)
        return _normalize_vector(vectors[0])
    if _embedding_provider() == "llama_cpp":
        request_timeout = timeout or LLAMA_CPP_TIMEOUT
        vectors = _llama_cpp_embed_request([full_text], request_timeout)
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

    if _embedding_provider() == "bedrock":
        vectors = _bedrock_embed_request(prepared, request_timeout)
        return [_normalize_vector(v) for v in vectors]

    if _embedding_provider() == "vertex_ai":
        vectors = _vertex_embed_request(prepared, request_timeout)
        return [_normalize_vector(v) for v in vectors]
    if _embedding_provider() == "llama_cpp":
        request_timeout = timeout or LLAMA_CPP_TIMEOUT
        vectors = _llama_cpp_embed_request(prepared, request_timeout)
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
