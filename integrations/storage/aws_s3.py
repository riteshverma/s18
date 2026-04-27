"""AWS S3 backend.

Imports boto3 lazily so Azure-only deployments do not need it on the path.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote, urlparse

from integrations.storage.base import ObjectRef, ObjectStore, compute_sha256


class AwsS3ObjectStore(ObjectStore):
    provider = "aws_s3"

    def __init__(
        self,
        *,
        bucket: str,
        region: str = "us-east-1",
        namespace: str = "shared",
        kms_key_id: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.6,
    ) -> None:
        if not bucket:
            raise ValueError("AwsS3ObjectStore requires a bucket name")
        self.bucket = bucket
        self.region = region
        self.namespace = namespace
        self.kms_key_id = kms_key_id
        self.endpoint_url = endpoint_url
        self._client = None
        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_backoff_seconds = max(0.05, float(retry_backoff_seconds))

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[reportMissingImports]
            from botocore.config import Config  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - exercised in deployment
            raise RuntimeError(
                "boto3 must be installed for the aws_s3 ObjectStore. "
                "Install: `pip install boto3`"
            ) from exc
        self._client = boto3.client(
            "s3",
            region_name=self.region,
            endpoint_url=self.endpoint_url,
            config=Config(
                retries={"mode": "standard", "max_attempts": self._retry_attempts},
                read_timeout=30,
                connect_timeout=10,
            ),
        )
        return self._client

    def _call_s3(self, fn):
        try:
            from botocore.exceptions import ClientError, ConnectionError, ReadTimeoutError  # type: ignore[reportMissingImports]
        except Exception:  # pragma: no cover
            ClientError = Exception  # type: ignore[assignment]
            ConnectionError = Exception  # type: ignore[assignment]
            ReadTimeoutError = Exception  # type: ignore[assignment]

        last_error: Optional[Exception] = None
        for idx in range(self._retry_attempts):
            try:
                return fn()
            except (ConnectionError, ReadTimeoutError) as exc:
                last_error = exc
            except ClientError as exc:
                last_error = exc
                status = int((exc.response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode") or 0)
                code = str((exc.response.get("Error", {}) or {}).get("Code") or "").lower()
                retryable = status in {408, 409, 425, 429, 500, 502, 503, 504} or code in {
                    "throttling",
                    "throttlingexception",
                    "requesttimeout",
                    "slowdown",
                    "serviceunavailable",
                }
                if not retryable:
                    raise
            if idx >= self._retry_attempts - 1:
                raise RuntimeError(f"S3 operation failed after retries: {last_error}") from last_error
            sleep_for = self._retry_backoff_seconds * (2 ** idx) + random.uniform(0, 0.1)
            time.sleep(min(sleep_for, 4.0))
        raise RuntimeError(f"S3 operation failed: {last_error}") from last_error

    def _full_key(self, key: str) -> str:
        clean = key.strip("/").replace("..", "")
        return f"{self.namespace}/{clean}" if self.namespace else clean

    def _uri(self, full_key: str) -> str:
        return f"s3://{self.bucket}/{quote(full_key, safe='/')}"

    def _key_from_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"aws_s3 cannot resolve uri scheme: {parsed.scheme}")
        return parsed.path.lstrip("/")

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

        params: Dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": full_key,
            "Body": data,
            "ContentType": content_type or "application/octet-stream",
            "Metadata": meta,
        }
        if self.kms_key_id:
            params["ServerSideEncryption"] = "aws:kms"
            params["SSEKMSKeyId"] = self.kms_key_id
        self._call_s3(lambda: self._client_lazy().put_object(**params))

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
        resp = self._call_s3(
            lambda: self._client_lazy().get_object(Bucket=self.bucket, Key=full_key)
        )
        return resp["Body"].read()

    def presign(self, uri: str, *, ttl_seconds: int = 900) -> str:
        full_key = self._key_from_uri(uri)
        return self._client_lazy().generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": full_key},
            ExpiresIn=ttl_seconds,
        )

    def list(self, prefix: str) -> Iterable[ObjectRef]:
        full_prefix = self._full_key(prefix)
        client = self._client_lazy()
        paginator = client.get_paginator("list_objects_v2")
        out: list[ObjectRef] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for entry in page.get("Contents", []) or []:
                head = self._call_s3(
                    lambda key=entry["Key"]: client.head_object(Bucket=self.bucket, Key=key)
                )
                meta = {k: str(v) for k, v in (head.get("Metadata") or {}).items()}
                out.append(
                    ObjectRef(
                        provider=self.provider,
                        uri=self._uri(entry["Key"]),
                        key=entry["Key"],
                        size=int(entry.get("Size") or 0),
                        sha256=meta.get("sha256", ""),
                        metadata=meta,
                    )
                )
        return out

    def delete(self, uri: str) -> None:
        full_key = self._key_from_uri(uri)
        self._call_s3(
            lambda: self._client_lazy().delete_object(Bucket=self.bucket, Key=full_key)
        )

    def stat(self, uri: str) -> Dict[str, Any]:
        full_key = self._key_from_uri(uri)
        head = self._call_s3(
            lambda: self._client_lazy().head_object(Bucket=self.bucket, Key=full_key)
        )
        meta = {k: str(v) for k, v in (head.get("Metadata") or {}).items()}
        return {
            "size": int(head.get("ContentLength") or 0),
            "sha256": meta.get("sha256"),
            "metadata": meta,
            "content_type": head.get("ContentType"),
            "last_modified": head.get("LastModified"),
        }
