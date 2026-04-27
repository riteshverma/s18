"""Factory that resolves VectorStore backends from settings + tenant context."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings_loader import load_settings
from integrations.tenancy import storage_namespace_for_tenant
from integrations.vectors.base import VectorStore

_BACKEND_KEY = "vector_store"
_DEFAULT_PROVIDER = "faiss"


def _resolve_provider(
    tenant_context: Optional[Dict[str, str]],
    settings: Dict[str, Any],
) -> str:
    ingest_cfg = settings.get("ingest", {}) or {}
    backend_cfg = ingest_cfg.get(_BACKEND_KEY, {}) or {}
    base_provider = str(backend_cfg.get("provider") or _DEFAULT_PROVIDER).lower()

    overrides = backend_cfg.get("tenant_overrides") or {}
    if tenant_context:
        tenant_id = tenant_context.get("tenant_id")
        if tenant_id and tenant_id in overrides:
            return str(overrides[tenant_id]).lower()
    return base_provider


def get_vector_store(tenant_context: Optional[Dict[str, str]] = None) -> VectorStore:
    settings = load_settings()
    provider = _resolve_provider(tenant_context, settings)
    backend_cfg = (settings.get("ingest", {}) or {}).get(_BACKEND_KEY, {}) or {}
    namespace = storage_namespace_for_tenant(
        tenant_context or {}, settings.get("tenancy", {}) or {}
    )

    if provider == "faiss":
        from integrations.vectors.faiss_local import FaissLocalVectorStore

        cfg = backend_cfg.get("faiss", {}) or {}
        index_dir = Path(cfg.get("index_dir") or "mcp_servers/faiss_index").resolve()
        # Each tenant gets its own subfolder unless the deployment is the
        # shared starter tier (namespace == "shared").
        if namespace and namespace != "shared":
            index_dir = index_dir / namespace
        return FaissLocalVectorStore(index_dir=index_dir)

    if provider == "azure_ai_search":
        from integrations.vectors.azure_ai_search import AzureAiSearchVectorStore

        cfg = backend_cfg.get("azure_ai_search", {}) or {}
        endpoint = (
            os.getenv("AZURE_SEARCH_ENDPOINT") or cfg.get("endpoint", "")
        ).rstrip("/")
        api_key = os.getenv(cfg.get("api_key_env", "AZURE_SEARCH_KEY")) or os.getenv(
            "AZURE_SEARCH_KEY", ""
        )
        index_name = cfg.get("index_name") or f"s18-rag-{namespace}".replace("_", "-")
        return AzureAiSearchVectorStore(
            endpoint=endpoint, api_key=api_key, index_name=index_name
        )

    if provider == "aws_opensearch":
        from integrations.vectors.aws_opensearch import AwsOpenSearchVectorStore

        cfg = backend_cfg.get("aws_opensearch", {}) or {}
        endpoint = os.getenv("OPENSEARCH_ENDPOINT") or cfg.get("endpoint", "")
        region = cfg.get("region", "us-east-1")
        index_name = cfg.get("index_name") or f"s18-rag-{namespace}"
        return AwsOpenSearchVectorStore(
            endpoint=endpoint, region=region, index_name=index_name,
            service=cfg.get("service", "aoss"),
        )

    if provider == "bedrock_kb":
        from integrations.vectors.bedrock_kb import BedrockKnowledgeBaseVectorStore

        cfg = backend_cfg.get("bedrock_kb", {}) or {}
        return BedrockKnowledgeBaseVectorStore(
            kb_id=cfg.get("knowledge_base_id", ""),
            region=cfg.get("region", "us-east-1"),
            data_source_id=cfg.get("data_source_id"),
        )

    raise ValueError(f"Unknown ingest.vector_store.provider: {provider!r}")
