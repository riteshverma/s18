"""FAISS helpers with optional CUDA acceleration (faiss-gpu).

Install a CUDA-enabled faiss build (mutually exclusive with faiss-cpu).
On Linux, ``pip install faiss-gpu`` when wheels match your Python version; on
Windows use WSL/Linux or conda-forge ``faiss-gpu``. Then::

    uv pip uninstall faiss-cpu
    uv pip install faiss-gpu

Enable via settings ``integrations.vector_store.faiss.use_gpu`` or env
``S18_FAISS_USE_GPU=1``. Indexes are still persisted as CPU ``index.bin`` files;
GPU indices are used in memory for search/build when configured.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

_faiss: Any = None
_gpu_resources: dict[int, Any] = {}
_config_lock = threading.Lock()
_use_gpu: Optional[bool] = None
_gpu_device: Optional[int] = None
_gpu_min_vectors: Optional[int] = None
_gpu_status_logged = False


def _env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _load_config() -> None:
    global _use_gpu, _gpu_device, _gpu_min_vectors
    with _config_lock:
        if _use_gpu is not None:
            return
        env_flag = _env_bool("S18_FAISS_USE_GPU")
        env_device = os.getenv("S18_FAISS_GPU_DEVICE")
        env_min = os.getenv("S18_FAISS_GPU_MIN_VECTORS")
        if env_flag is not None:
            _use_gpu = env_flag
        if env_device is not None and str(env_device).strip():
            _gpu_device = int(env_device)
        if env_min is not None and str(env_min).strip():
            _gpu_min_vectors = max(0, int(env_min))

        try:
            from config.settings_loader import load_settings

            cfg = (load_settings().get("integrations") or {}).get("vector_store") or {}
            faiss_cfg = cfg.get("faiss") if isinstance(cfg, dict) else {}
            if not isinstance(faiss_cfg, dict):
                faiss_cfg = {}
            if _use_gpu is None:
                _use_gpu = bool(faiss_cfg.get("use_gpu", False))
            if _gpu_device is None:
                _gpu_device = int(faiss_cfg.get("gpu_device", 0) or 0)
            if _gpu_min_vectors is None:
                _gpu_min_vectors = max(0, int(faiss_cfg.get("gpu_min_vectors", 0) or 0))
        except Exception:
            if _use_gpu is None:
                _use_gpu = False
            if _gpu_device is None:
                _gpu_device = 0
            if _gpu_min_vectors is None:
                _gpu_min_vectors = 0


def faiss_use_gpu_requested() -> bool:
    _load_config()
    return bool(_use_gpu)


def faiss_gpu_device() -> int:
    _load_config()
    return int(_gpu_device or 0)


def faiss_gpu_min_vectors() -> int:
    _load_config()
    return int(_gpu_min_vectors or 0)


def get_faiss():
    """Return the faiss module (faiss-cpu or faiss-gpu)."""
    global _faiss
    if _faiss is None:
        try:
            import faiss as _imported
        except ImportError as exc:
            raise RuntimeError(
                "faiss is not installed. Default: `uv pip install faiss-cpu`. "
                "For CUDA: uninstall faiss-cpu, then `uv pip install faiss-gpu`, "
                "and set S18_FAISS_USE_GPU=1."
            ) from exc
        _faiss = _imported
        _log_gpu_status_once()
    return _faiss


def _log_gpu_status_once() -> None:
    global _gpu_status_logged
    if _gpu_status_logged:
        return
    _gpu_status_logged = True
    f = _faiss
    try:
        num = int(f.get_num_gpus())
    except Exception:
        num = 0
    if faiss_use_gpu_requested():
        if num > 0:
            print(
                f"[faiss] GPU enabled (device {faiss_gpu_device()}, "
                f"{num} GPU(s) visible to faiss)",
                file=sys.stderr,
            )
        else:
            print(
                "[faiss] S18_FAISS_USE_GPU is set but faiss reports 0 GPUs; using CPU. "
                "Install faiss-gpu and ensure CUDA drivers are available.",
                file=sys.stderr,
            )


def gpu_available() -> bool:
    if not faiss_use_gpu_requested():
        return False
    f = get_faiss()
    try:
        return int(f.get_num_gpus()) > 0
    except Exception:
        return False


def _should_use_gpu_for_index(index: Any) -> bool:
    if not gpu_available():
        return False
    try:
        ntotal = int(getattr(index, "ntotal", 0))
    except Exception:
        ntotal = 0
    return ntotal >= faiss_gpu_min_vectors()


def _gpu_resources(device: int) -> Any:
    f = get_faiss()
    with _config_lock:
        if device not in _gpu_resources:
            _gpu_resources[device] = f.StandardGpuResources()
        return _gpu_resources[device]


def to_gpu(index: Any, *, device: Optional[int] = None) -> Any:
    if index is None or not _should_use_gpu_for_index(index):
        return index
    f = get_faiss()
    dev = faiss_gpu_device() if device is None else int(device)
    try:
        return f.index_cpu_to_gpu(_gpu_resources(dev), dev, index)
    except Exception as exc:
        print(f"[faiss] index_cpu_to_gpu failed, using CPU: {exc}", file=sys.stderr)
        return index


def to_cpu(index: Any) -> Any:
    if index is None or not gpu_available():
        return index
    f = get_faiss()
    try:
        return f.index_gpu_to_cpu(index)
    except Exception:
        return index


def read_index(path: str | Path) -> Any:
    f = get_faiss()
    index = f.read_index(str(path))
    return to_gpu(index)


def write_index(index: Any, path: str | Path) -> None:
    f = get_faiss()
    f.write_index(to_cpu(index), str(path))


def create_index_flat_l2(dimension: int) -> Any:
    f = get_faiss()
    return to_gpu(f.IndexFlatL2(int(dimension)))


def create_index_flat_ip(dimension: int) -> Any:
    f = get_faiss()
    return to_gpu(f.IndexFlatIP(int(dimension)))


def runtime_info() -> dict[str, Any]:
    _load_config()
    f = get_faiss()
    try:
        num_gpus = int(f.get_num_gpus())
    except Exception:
        num_gpus = 0
    return {
        "use_gpu_requested": faiss_use_gpu_requested(),
        "gpu_active": gpu_available(),
        "gpu_device": faiss_gpu_device(),
        "gpu_min_vectors": faiss_gpu_min_vectors(),
        "num_gpus": num_gpus,
    }
