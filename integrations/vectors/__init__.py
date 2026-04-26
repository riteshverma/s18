"""Cloud-agnostic vector store facade.

Tenants choose FAISS, Azure AI Search, AWS OpenSearch Serverless, or Bedrock
Knowledge Base; callers work against the :class:`VectorStore` Protocol.
"""

from integrations.vectors.base import Chunk, SearchHit, VectorStore
from integrations.vectors.factory import get_vector_store

__all__ = ["Chunk", "SearchHit", "VectorStore", "get_vector_store"]
