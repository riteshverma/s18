import numpy as np

from core.embedding import get_normalized_embedding


def get_embedding(text: str, task_type: str = "search_document") -> np.ndarray:
    """Generate embedding for text. Routes to Azure OpenAI when configured,
    otherwise falls back to local Ollama (Nomic) via core.embedding."""
    return get_normalized_embedding(text, task_type=task_type)
