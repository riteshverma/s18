import requests
import numpy as np
import sys
import os
from pathlib import Path

# Import from centralized settings
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings_loader import get_ollama_url, get_model, get_timeout, load_settings

EMBED_URL = get_ollama_url("embeddings")
EMBED_MODEL = get_model("embedding")
OLLAMA_TIMEOUT = get_timeout()

def _build_embed_candidates(text: str) -> list[tuple[str, dict]]:
    """Support both modern and legacy Ollama embedding endpoints."""
    return [
        (get_ollama_url("embed"), {"model": EMBED_MODEL, "input": text}),
        (get_ollama_url("embeddings"), {"model": EMBED_MODEL, "prompt": text}),
    ]

def _azure_embedding_request(inputs: list[str]) -> list[list[float]]:
    cfg = load_settings().get("azure_openai", {})
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", cfg.get("endpoint", "")).rstrip("/")
    api_version = os.getenv("OPENAI_API_VERSION", cfg.get("api_version", "2024-10-21"))
    deployment = (
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        or cfg.get("embedding_deployment")
        or get_model("embedding")
    )
    key_env = cfg.get("api_key_env", "AZURE_OPENAI_API_KEY")
    api_key = os.getenv(key_env) or os.getenv("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not deployment or not api_key:
        raise RuntimeError("Azure OpenAI embedding is not configured.")

    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
    response = requests.post(
        url,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        json={"input": inputs},
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    data = sorted(data, key=lambda item: item.get("index", 0))
    vectors = [item.get("embedding", []) for item in data]
    if not vectors:
        raise RuntimeError("Azure OpenAI returned no embeddings.")
    return vectors

def _get_embedding_provider() -> str:
    cfg = load_settings().get("models", {})
    provider = str(cfg.get("embedding_provider", "ollama")).strip().lower()
    if provider:
        return provider
    return "ollama"

def get_embedding(text: str, task_type: str = "search_document") -> np.ndarray:
    """Generate embedding for text using local Ollama instance with Nomic prefixes."""
    try:
        # 🏷️ Nomic Embed v1.5 requires task-specific prefixes
        # search_query: for the query
        # search_document: for the facts/documents
        prefix = f"{task_type}: "
        full_text = prefix + text if not text.startswith(prefix) else text
        
        embedding = None
        last_error = None
        provider = _get_embedding_provider()

        if provider == "azure_openai":
            embedding = _azure_embedding_request([full_text])[0]
        else:
            for embed_url, payload in _build_embed_candidates(full_text):
                try:
                    response = requests.post(embed_url, json=payload, timeout=OLLAMA_TIMEOUT)
                    response.raise_for_status()
                    body = response.json()

                    # /api/embed may return {"embeddings":[...]} (batch-friendly)
                    if "embedding" in body:
                        embedding = body["embedding"]
                    elif "embeddings" in body and body["embeddings"]:
                        first = body["embeddings"][0]
                        embedding = first.get("embedding", first) if isinstance(first, dict) else first

                    if embedding is not None:
                        break
                except Exception as e:
                    last_error = e
                    continue

        if embedding is None:
            raise RuntimeError(last_error or "No embedding returned from local model server")
        vec = np.array(embedding, dtype=np.float32)
        
        # 📐 L2 Normalization (ensures distances are in [0, 4] range for IndexFlatL2)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        return vec
    except Exception as e:
        print(f"Error generating embedding: {e}", file=sys.stderr)
        fallback_dim = int(load_settings().get("azure_openai", {}).get("embedding_dimension", 768) or 768)
        return np.zeros(fallback_dim, dtype=np.float32) # Fallback to empty vector
