"""Factory that resolves ObjectStore backends from settings + tenant context."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from config.settings_loader import load_settings
from integrations.storage.base import ObjectStore
from integrations.tenancy import storage_namespace_for_tenant

_BACKEND_KEY = "object_store"
_DEFAULT_PROVIDER = "local_fs"


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


def get_object_store(tenant_context: Optional[Dict[str, str]] = None) -> ObjectStore:
    """Return an :class:`ObjectStore` configured for the active tenant.

    The factory selects a provider in this order:

    1. ``ingest.object_store.tenant_overrides[tenant_id]`` if set
    2. ``ingest.object_store.provider``
    3. fallback to ``local_fs``
    """

    settings = load_settings()
    provider = _resolve_provider(tenant_context, settings)
    backend_cfg = (settings.get("ingest", {}) or {}).get(_BACKEND_KEY, {}) or {}
    namespace = storage_namespace_for_tenant(
        tenant_context or {}, settings.get("tenancy", {}) or {}
    )

    if provider == "local_fs":
        from integrations.storage.local_fs import LocalFsObjectStore

        root_cfg = backend_cfg.get("local_fs", {}) or {}
        root = Path(root_cfg.get("root") or "data/ingest").resolve()
        return LocalFsObjectStore(root=root, namespace=namespace)

    if provider == "azure_blob":
        from integrations.storage.azure_blob import AzureBlobObjectStore

        cfg = backend_cfg.get("azure_blob", {}) or {}
        return AzureBlobObjectStore(
            account_url=cfg.get("account_url", ""),
            container=cfg.get("container", ""),
            namespace=namespace,
            sas_token_env=cfg.get("sas_token_env"),
            connection_string_env=cfg.get("connection_string_env"),
        )

    if provider == "aws_s3":
        from integrations.storage.aws_s3 import AwsS3ObjectStore

        cfg = backend_cfg.get("aws_s3", {}) or {}
        return AwsS3ObjectStore(
            bucket=cfg.get("bucket", ""),
            region=cfg.get("region", "us-east-1"),
            namespace=namespace,
            kms_key_id=cfg.get("kms_key_id"),
            endpoint_url=cfg.get("endpoint_url"),
        )

    raise ValueError(f"Unknown ingest.object_store.provider: {provider!r}")
