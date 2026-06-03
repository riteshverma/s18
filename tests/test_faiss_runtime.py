"""Tests for optional FAISS GPU runtime helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("faiss")

from core import faiss_runtime


def test_read_write_roundtrip_cpu(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("S18_FAISS_USE_GPU", "0")
    faiss_runtime._use_gpu = None
    faiss_runtime._gpu_device = None
    faiss_runtime._gpu_min_vectors = None
    faiss_runtime._gpu_status_logged = False

    index = faiss_runtime.create_index_flat_l2(4)
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    index.add(vectors)

    path = tmp_path / "index.bin"
    faiss_runtime.write_index(index, path)
    loaded = faiss_runtime.read_index(path)
    D, I = loaded.search(vectors[:1], 2)
    assert loaded.ntotal == 2
    assert int(I[0][0]) in {0, 1}


def test_runtime_info_reports_cpu_when_gpu_disabled(monkeypatch):
    monkeypatch.setenv("S18_FAISS_USE_GPU", "0")
    faiss_runtime._use_gpu = None
    faiss_runtime._gpu_device = None
    faiss_runtime._gpu_min_vectors = None

    info = faiss_runtime.runtime_info()
    assert info["use_gpu_requested"] is False
    assert info["gpu_active"] is False
