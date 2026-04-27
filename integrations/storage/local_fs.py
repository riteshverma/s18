"""Local-filesystem ObjectStore for dev / CI / air-gapped runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import unquote, urlparse

from integrations.storage.base import ObjectRef, ObjectStore, compute_sha256


class LocalFsObjectStore(ObjectStore):
    """Stores objects under a tenant-namespaced directory.

    URI scheme: ``file://<absolute path>``. Sidecar ``.meta.json`` files keep
    user metadata + content hash next to the blob so we can return real
    :class:`ObjectRef` values from :meth:`stat` and :meth:`list`.
    """

    provider = "local_fs"

    def __init__(self, root: Path, *, namespace: str = "shared") -> None:
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace

    def _resolve(self, key: str) -> Path:
        # Scrub `..` traversals plus any leading slash/backslash so the
        # second operand of `self.root / clean` can never become absolute
        # (which on POSIX or Windows would silently discard the root).
        clean = key.replace("..", "").replace("\\", "/").lstrip("/")
        return (self.root / clean).resolve()

    def _ensure_inside_root(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise PermissionError(f"Path escapes namespace root: {path}") from exc

    @staticmethod
    def _path_to_uri(path: Path) -> str:
        """`Path.as_uri` handles the Windows drive-letter / forward-slash dance."""
        return path.resolve().as_uri()

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectRef:
        target = self._resolve(key)
        self._ensure_inside_root(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

        sha = compute_sha256(data)
        meta = dict(metadata or {})
        meta.setdefault("content_type", content_type or "application/octet-stream")
        meta["sha256"] = sha
        meta["size"] = str(len(data))
        target.with_suffix(target.suffix + ".meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        return ObjectRef(
            provider=self.provider,
            uri=self._path_to_uri(target),
            key=key,
            size=len(data),
            sha256=sha,
            metadata=meta,
        )

    def get(self, uri: str) -> bytes:
        path = self._uri_to_path(uri)
        return path.read_bytes()

    def presign(self, uri: str, *, ttl_seconds: int = 900) -> str:
        # Local backend has no real presigning; return the URI itself so the
        # contract is honored. Production deployments should never use this
        # backend for external Power Apps callers.
        del ttl_seconds
        return uri

    def list(self, prefix: str) -> Iterable[ObjectRef]:
        base = self._resolve(prefix)
        if not base.exists():
            return []
        out: list[ObjectRef] = []
        for path in base.rglob("*"):
            if path.is_file() and not path.name.endswith(".meta.json"):
                rel = path.relative_to(self.root).as_posix()
                meta_path = path.with_suffix(path.suffix + ".meta.json")
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                    except Exception:
                        meta = {}
                out.append(
                    ObjectRef(
                        provider=self.provider,
                        uri=self._path_to_uri(path),
                        key=rel,
                        size=int(meta.get("size") or path.stat().st_size),
                        sha256=str(meta.get("sha256") or ""),
                        metadata=meta,
                    )
                )
        return out

    def delete(self, uri: str) -> None:
        path = self._uri_to_path(uri)
        self._ensure_inside_root(path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
            meta = path.with_suffix(path.suffix + ".meta.json")
            if meta.exists():
                meta.unlink()

    def stat(self, uri: str) -> Dict[str, Any]:
        path = self._uri_to_path(uri)
        st = path.stat()
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        return {
            "size": st.st_size,
            "sha256": meta.get("sha256"),
            "metadata": meta,
        }

    def _uri_to_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"local_fs cannot resolve uri scheme: {parsed.scheme}")
        raw_path = unquote(parsed.path)
        # Windows file URIs round-trip as `/C:/Users/...`; strip the leading
        # slash before constructing a Path so the drive letter parses.
        if len(raw_path) > 2 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        path = Path(raw_path).resolve()
        self._ensure_inside_root(path)
        return path
