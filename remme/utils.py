import numpy as np

from core.embedding import get_normalized_embedding

def get_embedding(text: str, task_type: str = "search_document") -> np.ndarray:
    """Generate embedding for text using local Ollama instance with Nomic prefixes."""
    return get_normalized_embedding(text, task_type=task_type)
