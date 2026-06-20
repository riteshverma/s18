from pathlib import Path

from mcp_servers.index_scheduler import IndexScheduler


def test_callback_error_result_marks_file_retryable(tmp_path: Path):
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "faiss_index"
    data_dir.mkdir()
    index_dir.mkdir()
    doc = data_dir / "doc.md"
    doc.write_text("important content", encoding="utf-8")

    def fail_callback(_abs_path: Path, _rel_path: str) -> dict:
        return {
            "status": "error",
            "message": "Index compatibility guard triggered",
            "chunk_count": 0,
        }

    scheduler = IndexScheduler(data_dir, index_dir, process_callback=fail_callback)
    scheduler._handle_index("doc.md")

    entry = scheduler.ledger.get("doc.md")
    assert entry is not None
    assert entry.status == "error"
    assert "compatibility" in (entry.error or "")
    assert scheduler.ledger.needs_indexing("doc.md", scheduler._compute_hash(doc))


def test_zero_chunk_success_can_complete_empty_file(tmp_path: Path):
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "faiss_index"
    data_dir.mkdir()
    index_dir.mkdir()
    doc = data_dir / "empty.md"
    doc.write_text("", encoding="utf-8")

    scheduler = IndexScheduler(
        data_dir,
        index_dir,
        process_callback=lambda _abs_path, _rel_path: {"chunk_count": 0},
    )
    scheduler._handle_index("empty.md")

    entry = scheduler.ledger.get("empty.md")
    assert entry is not None
    assert entry.status == "complete"
    assert entry.chunk_count == 0
