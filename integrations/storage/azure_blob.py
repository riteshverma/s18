"""Azure Blob Storage backend.

Imports the Azure SDK lazily so deployments that do not need it can skip the
``azure-storage-blob`` dependency.
"""

from __future__ import annotations

import datetime as dt
import os
import random
import time
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote, urlparse

from integrations.storage.base import ObjectRef, ObjectStore, compute_sha256


class AzureBlobObjectStore(ObjectStore):
    provider = "azure_blob"

    def __init__(
        self,
        *,
        account_url: str,
        container: str,
        namespace: str = "shared",
        sas_token_env: Optional[str] = None,
        connection_string_env: Optional[str] = None,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.6,
    ) -> None:
        if not container:
            raise ValueError("AzureBlobObjectStore requires a container name")
        self.account_url = account_url.rstrip("/")
        self.container = container
        self.namespace = namespace
        self._sas_env = sas_token_env
        self._conn_env = connection_string_env
        self._client = None
        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_backoff_seconds = max(0.05, float(retry_backoff_seconds))

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:  # pragma: no cover - exercised in deployment
            raise RuntimeError(
                "azure-storage-blob and azure-identity must be installed for "
                "the azure_blob ObjectStore. Install: "
                "`pip install azure-storage-blob azure-identity`"
            ) from exc

        if self._conn_env and os.getenv(self._conn_env):
            self._client = BlobServiceClient.from_connection_string(os.environ[self._conn_env])
            return self._client

        if self._sas_env and os.getenv(self._sas_env):
            sas = os.environ[self._sas_env].lstrip("?")
            self._client = BlobServiceClient(account_url=f"{self.account_url}?{sas}")
            return self._client

        # Default: Managed Identity / az login / workload identity.
        self._client = BlobServiceClient(
            account_url=self.account_url, credential=DefaultAzureCredential()
        )
        return self._client

    def _call_azure(self, fn):
        last_error: Optional[Exception] = None
        for idx in range(self._retry_attempts):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_error = exc
                text = str(exc).lower()
                retryable = any(
                    token in text
                    for token in [
                        "timeout",
                        "tempor",
                        "throttl",
                        "503",
                        "502",
                        "429",
                        "connection",
                    ]
                )
                if idx >= self._retry_attempts - 1 or not retryable:
                    raise
                sleep_for = self._retry_backoff_seconds * (2 ** idx) + random.uniform(0, 0.1)
                time.sleep(min(sleep_for, 4.0))
        raise RuntimeError(f"Azure Blob operation failed: {last_error}") from last_error

    def _full_key(self, key: str) -> str:
        clean = key.strip("/").replace("..", "")
        return f"{self.namespace}/{clean}" if self.namespace else clean

    def _uri(self, full_key: str) -> str:
        return f"{self.account_url}/{self.container}/{quote(full_key)}"

    def _container_client(self):
        return self._client_lazy().get_container_client(self.container)

    def _blob_client(self, uri: str):
        full_key = self._key_from_uri(uri)
        return self._client_lazy().get_blob_client(self.container, full_key)

    def _key_from_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        prefix = f"/{self.container}/"
        path = parsed.path
        if path.startswith(prefix):
            return path[len(prefix):]
        return path.lstrip("/")

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectRef:
        from azure.storage.blob import ContentSettings

        full_key = self._full_key(key)
        sha = compute_sha256(data)
        meta = {k: str(v) for k, v in (metadata or {}).items()}
        meta["sha256"] = sha
        client = self._client_lazy().get_blob_client(self.container, full_key)
        self._call_azure(
            lambda: client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type or "application/octet-stream"),
                metadata=meta,
            )
        )
        return ObjectRef(
            provider=self.provider,
            uri=self._uri(full_key),
            key=key,
            size=len(data),
            sha256=sha,
            metadata=meta,
        )

    def get(self, uri: str) -> bytes:
        return self._call_azure(lambda: self._blob_client(uri).download_blob().readall())

    def presign(self, uri: str, *, ttl_seconds: int = 900) -> str:
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        full_key = self._key_from_uri(uri)
        client = self._client_lazy()
        # User delegation key works for Managed Identity. Account-key path is
        # used when the client was built from a connection string.
        try:
            udk = client.get_user_delegation_key(
                key_start_time=dt.datetime.utcnow() - dt.timedelta(minutes=1),
                key_expiry_time=dt.datetime.utcnow() + dt.timedelta(seconds=ttl_seconds + 60),
            )
            sas = generate_blob_sas(
                account_name=client.account_name,
                container_name=self.container,
                blob_name=full_key,
                user_delegation_key=udk,
                permission=BlobSasPermissions(read=True),
                expiry=dt.datetime.utcnow() + dt.timedelta(seconds=ttl_seconds),
            )
        except Exception:
            sas = generate_blob_sas(
                account_name=client.account_name,
                container_name=self.container,
                blob_name=full_key,
                account_key=client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=dt.datetime.utcnow() + dt.timedelta(seconds=ttl_seconds),
            )
        return f"{self._uri(full_key)}?{sas}"

    def list(self, prefix: str) -> Iterable[ObjectRef]:
        container = self._container_client()
        full_prefix = self._full_key(prefix)
        out: list[ObjectRef] = []
        for blob in container.list_blobs(name_starts_with=full_prefix):
            md = blob.metadata or {}
            out.append(
                ObjectRef(
                    provider=self.provider,
                    uri=self._uri(blob.name),
                    key=blob.name,
                    size=int(blob.size or 0),
                    sha256=str(md.get("sha256") or ""),
                    metadata={k: str(v) for k, v in md.items()},
                )
            )
        return out

    def delete(self, uri: str) -> None:
        self._call_azure(
            lambda: self._blob_client(uri).delete_blob(delete_snapshots="include")
        )

    def stat(self, uri: str) -> Dict[str, Any]:
        props = self._call_azure(lambda: self._blob_client(uri).get_blob_properties())
        return {
            "size": props.size,
            "sha256": (props.metadata or {}).get("sha256"),
            "metadata": dict(props.metadata or {}),
            "content_type": getattr(props.content_settings, "content_type", None),
            "last_modified": getattr(props, "last_modified", None),
        }
