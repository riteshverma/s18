"""Reusable ingest helpers: parsing, chunking, record materialization.

Stays pure-Python with light optional deps so unit tests run without cloud
SDKs. PDF/DOCX paths use lazy imports.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


_TEXT_EXTENSIONS = {
    "txt", "md", "markdown", "rst", "html", "htm", "json", "yaml", "yml",
    "csv", "tsv", "log", "py", "ts", "tsx", "js", "jsx", "java", "go", "rb",
    "c", "cpp", "h", "hpp", "cs", "kt", "swift", "rs",
}


@dataclass
class IngestRecord:
    """A normalized Power Apps payload row.

    `record_kind` distinguishes structured rows ("dataverse", "sharepoint",
    "custom") from generic blobs ("file") so chunking can specialize.
    """

    record_id: str
    record_kind: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def hash_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def chunk_text(
    text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    max_length: int = 1024,
) -> List[str]:
    """Word-window chunker.

    Mirrors the behavior of the existing FAISS indexer in
    ``mcp_servers/server_rag.py`` so retrieval quality is consistent across
    the new and legacy paths.
    """

    if not text or not text.strip():
        return []
    words = re.split(r"\s+", text.strip())
    if len(words) <= chunk_size:
        joined = " ".join(words)
        return [joined[:max_length]]

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        piece = " ".join(words[start:end])
        if piece.strip():
            chunks.append(piece[:max_length])
        if end == len(words):
            break
    return chunks


def chunk_record(
    record: IngestRecord,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    max_length: int = 1024,
) -> List[Dict[str, Any]]:
    """Chunk a structured record. Adds chunk_id + position metadata."""

    pieces = chunk_text(
        record.text, chunk_size=chunk_size, overlap=overlap, max_length=max_length
    )
    out: List[Dict[str, Any]] = []
    for idx, piece in enumerate(pieces):
        chunk_id = hash_id(record.record_id, record.record_kind, str(idx), piece[:64])
        out.append(
            {
                "chunk_id": chunk_id,
                "doc_id": record.record_id,
                "text": piece,
                "metadata": {
                    **record.metadata,
                    "record_kind": record.record_kind,
                    "chunk_index": idx,
                    "chunk_total": len(pieces),
                },
            }
        )
    return out


def materialize_record(
    record: Dict[str, Any], *, default_kind: str = "dataverse"
) -> IngestRecord:
    """Flatten a structured Power Apps record into ``IngestRecord``.

    The resulting `text` block uses `<field>: <value>` lines so embeddings
    capture both field name and value; this is critical for tabular data.
    """

    record_id = (
        record.get("recordId")
        or record.get("id")
        or record.get("primaryKey")
        or uuid.uuid4().hex[:16]
    )
    kind = record.get("kind") or default_kind
    fields = record.get("fields")
    if fields is None:
        # Treat the whole dict as fields when no envelope was supplied.
        fields = {k: v for k, v in record.items() if k not in {"recordId", "id", "kind"}}

    lines: List[str] = []
    if "tableLogicalName" in record:
        lines.append(f"table: {record['tableLogicalName']}")
    if "title" in record:
        lines.append(f"title: {record['title']}")
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {value}")

    return IngestRecord(
        record_id=str(record_id),
        record_kind=str(kind),
        text="\n".join(lines),
        metadata={
            "tableLogicalName": record.get("tableLogicalName"),
            "title": record.get("title"),
            "raw_keys": list(fields.keys()),
        },
    )


def parse_file_to_text(
    *, filename: str, data: bytes, content_type: Optional[str] = None
) -> str:
    """Best-effort extraction of textual content from common file formats.

    Returns an empty string when extraction is not possible; caller decides
    whether to skip or persist a placeholder chunk pointing at the raw object
    URI.
    """

    name = (filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""

    if ext in _TEXT_EXTENSIONS or (content_type or "").startswith("text/"):
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            return ""
        if ext == "csv":
            return _csv_to_text(text)
        if ext == "json":
            return _json_to_text(text)
        return text

    if ext == "pdf" or content_type == "application/pdf":
        return _pdf_to_text(data)

    if ext == "docx":
        return _docx_to_text(data)

    return ""


def _csv_to_text(text: str) -> str:
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except Exception:
        return text
    if not rows:
        return ""
    header = rows[0]
    out_lines: List[str] = []
    for row in rows[1:]:
        line = "; ".join(
            f"{header[i] if i < len(header) else f'col{i}'}: {cell}"
            for i, cell in enumerate(row)
            if cell
        )
        if line:
            out_lines.append(line)
    return "\n".join(out_lines)


def _json_to_text(text: str) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return text
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _pdf_to_text(data: bytes) -> str:
    try:
        import pymupdf  # type: ignore[reportMissingImports]
    except Exception:
        return ""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        return ""
    try:
        return "\n\n".join(page.get_text("text") for page in doc)
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _docx_to_text(data: bytes) -> str:
    try:
        import docx  # type: ignore[reportMissingImports]
    except Exception:
        return ""
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception:
        return ""
    return "\n".join(p.text for p in document.paragraphs if p.text)


def iter_chunks_with_embeddings(
    chunks: List[Dict[str, Any]],
    *,
    embedder,
    batch_size: int = 16,
) -> Iterable[Dict[str, Any]]:
    """Yield chunk dicts with an `embedding` field attached.

    `embedder` is a callable accepting a list of strings and returning a list
    of vectors (lists of floats). Provided as a parameter so unit tests can
    inject a fake.
    """

    if not chunks:
        return []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder([c["text"] for c in batch])
        for chunk, vec in zip(batch, vectors):
            yield {**chunk, "embedding": list(vec)}
