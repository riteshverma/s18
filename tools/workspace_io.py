"""Resolve and read/write files within a bounded workspace root."""

from __future__ import annotations

import os
from pathlib import Path


def get_workspace_root(workspace_root: str | None = None) -> Path:
    """Return the configured workspace root or raise if missing."""
    raw = (workspace_root or os.environ.get("S18_WORKSPACE_ROOT") or "").strip()
    if not raw:
        raise ValueError(
            "Workspace root not configured. Pass workspace_root or set S18_WORKSPACE_ROOT."
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Workspace root is not a directory: {root}")
    return root


def resolve_workspace_path(path: str, workspace_root: str | None = None) -> Path:
    """Resolve path relative to workspace root; reject escapes outside the root."""
    if not path or not str(path).strip():
        raise ValueError("path must be a non-empty string")

    root = get_workspace_root(workspace_root)
    candidate = Path(str(path).strip())
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path {path!r} escapes workspace root {root}") from exc

    return resolved


def read_workspace_file(path: str, workspace_root: str | None = None) -> str:
    """Read a UTF-8 text file from the workspace."""
    target = resolve_workspace_path(path, workspace_root)
    if not target.is_file():
        raise FileNotFoundError(f"Workspace file not found: {target}")
    return target.read_text(encoding="utf-8")


def write_workspace_file(path: str, content: str, workspace_root: str | None = None) -> str:
    """Write UTF-8 text to a file inside the workspace (creating parent dirs)."""
    target = resolve_workspace_path(path, workspace_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {target}"
