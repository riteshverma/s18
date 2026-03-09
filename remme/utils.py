import requests
import numpy as np
import sys
from pathlib import Path

# Import from centralized settings
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings_loader import get_ollama_url, get_model, get_timeout

EMBED_URL = get_ollama_url("embeddings")
EMBED_MODEL = get_model("embedding")
OLLAMA_TIMEOUT = get_timeout()

def _build_embed_candidates(text: str) -> list[tuple[str, dict]]:
    """Support both modern and legacy Ollama embedding endpoints."""
    return [
        (get_ollama_url("embed"), {"model": EMBED_MODEL, "input": text}),
        (get_ollama_url("embeddings"), {"model": EMBED_MODEL, "prompt": text}),
    ]

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
        return np.zeros(768, dtype=np.float32) # Fallback to empty vector
