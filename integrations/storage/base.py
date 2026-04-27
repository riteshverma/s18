"""ObjectStore Protocol shared by every storage backend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol


@dataclass(frozen=True)
class ObjectRef:
    """Stable handle to a stored object.

    `uri` is what we persist downstream (FAISS metadata, vector store, audit
    log). The provider is included so retrieval can route back to the correct
    backend when a tenant migrates clouds.
    """

    provider: str
    uri: str
    key: str
    size: int
    sha256: str
    metadata: Dict[str, str] = field(default_factory=dict)


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ObjectStore(Protocol):
    """Minimum surface every storage backend must implement.

    Backends are deliberately thin: parsing, chunking, embedding all live in
    the worker pipeline. The store only moves bytes and lets the caller
    reference them by URI.
    """

    provider: str

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectRef:
        ...

    def get(self, uri: str) -> bytes:
        ...

    def presign(self, uri: str, *, ttl_seconds: int = 900) -> str:
        ...

    def list(self, prefix: str) -> Iterable[ObjectRef]:
        ...

    def delete(self, uri: str) -> None:
        ...

    def stat(self, uri: str) -> Dict[str, Any]:
        ...
