"""Google Cloud Storage backend.

Uses lazy imports so non-GCP deployments do not require google-cloud-storage.
"""

from __future__ import annotations

import datetime as dt
import random
import time
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote, unquote, urlparse

from integrations.storage.base import ObjectRef, ObjectStore, compute_sha256


class GcsObjectStore(ObjectStore):
    provider = "gcs"

    def __init__(
        self,
        *,
        bucket: str,
        namespace: str = "shared",
        project: Optional[str] = None,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.6,
    ) -> None:
        if not bucket:
            raise ValueError("GcsObjectStore requires a bucket name")
        self.bucket = bucket
        self.namespace = namespace
        self.project = project
        self._client = None
        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_backoff_seconds = max(0.05, float(retry_backoff_seconds))

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            from google.cloud import storage  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-cloud-storage must be installed for the gcs ObjectStore. "
                "Install: `pip install google-cloud-storage`"
            ) from exc
        self._client = storage.Client(project=self.project) if self.project else storage.Client()
        return self._client

    def _bucket_obj(self):
        return self._client_lazy().bucket(self.bucket)

    def _call_gcs(self, fn):
        last_error: Optional[Exception] = None
        for idx in range(self._retry_attempts):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover
                last_error = exc
                text = str(exc).lower()
                retryable = any(
                    token in text
                    for token in ["timeout", "tempor", "throttl", "503", "502", "429", "connection"]
                )
                if idx >= self._retry_attempts - 1 or not retryable:
                    raise
                sleep_for = self._retry_backoff_seconds * (2 ** idx) + random.uniform(0, 0.1)
                time.sleep(min(sleep_for, 4.0))
        raise RuntimeError(f"GCS operation failed: {last_error}") from last_error

    def _full_key(self, key: str) -> str:
        clean = key.strip("/").replace("..", "")
        return f"{self.namespace}/{clean}" if self.namespace else clean

    def _uri(self, full_key: str) -> str:
        return f"gs://{self.bucket}/{quote(full_key, safe='/')}"

    def _key_from_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "gs":
            raise ValueError(f"gcs cannot resolve uri scheme: {parsed.scheme}")
        if parsed.netloc and parsed.netloc != self.bucket:
            raise ValueError(f"GCS URI bucket mismatch: expected {self.bucket}, got {parsed.netloc}")
        return unquote(parsed.path.lstrip("/"))

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectRef:
        full_key = self._full_key(key)
        sha = compute_sha256(data)
        meta = {k: str(v) for k, v in (metadata or {}).items()}
        meta["sha256"] = sha
        blob = self._bucket_obj().blob(full_key)
        blob.metadata = meta
        self._call_gcs(
            lambda: blob.upload_from_string(
                data, content_type=content_type or "application/octet-stream"
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
        full_key = self._key_from_uri(uri)
        return self._call_gcs(lambda: self._bucket_obj().blob(full_key).download_as_bytes())

    def presign(self, uri: str, *, ttl_seconds: int = 900) -> str:
        full_key = self._key_from_uri(uri)
        blob = self._bucket_obj().blob(full_key)
        return self._call_gcs(
            lambda: blob.generate_signed_url(
            version="v4",
            expiration=dt.timedelta(seconds=ttl_seconds),
            method="GET",
            )
        )

    def list(self, prefix: str) -> Iterable[ObjectRef]:
        full_prefix = self._full_key(prefix)
        out: list[ObjectRef] = []
        for blob in self._client_lazy().list_blobs(self.bucket, prefix=full_prefix):
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
        full_key = self._key_from_uri(uri)
        self._call_gcs(lambda: self._bucket_obj().blob(full_key).delete())

    def stat(self, uri: str) -> Dict[str, Any]:
        full_key = self._key_from_uri(uri)
        blob = self._call_gcs(lambda: self._bucket_obj().get_blob(full_key))
        if blob is None:
            raise FileNotFoundError(f"GCS object not found: {uri}")
        md = blob.metadata or {}
        return {
            "size": int(blob.size or 0),
            "sha256": md.get("sha256"),
            "metadata": {k: str(v) for k, v in md.items()},
            "content_type": blob.content_type,
            "updated": blob.updated,
            "etag": blob.etag,
        }
