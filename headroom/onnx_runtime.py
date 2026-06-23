"""ONNX Runtime helpers for long-running Headroom processes."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Any


def _cpuset_size() -> int | None:
    """Number of CPUs this process is actually allowed to run on.

    Honors cgroup/cpuset limits via ``os.sched_getaffinity`` on Linux so the
    ONNX thread pool is sized to the container quota rather than the host's
    total core count. Falls back to ``os.cpu_count`` where affinity is not
    available (Windows/macOS).
    """
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            count = len(sched_getaffinity(0))
            if count > 0:
                return count
        except OSError:
            pass
    return os.cpu_count()


def _env_thread_count(name: str) -> int | None:
    """Read a positive ONNX thread-count override from the environment."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def hf_hub_download_local_first(repo_id: str, filename: str, *, allow_network: bool = True) -> str:
    """Download a file from HuggingFace Hub, preferring the local cache.

    Tries ``local_files_only=True`` first to avoid a network HEAD request when
    the model is already cached.  Falls back to a normal (network-allowed)
    download on the first cold start.

    Args:
        repo_id: HuggingFace Hub repository identifier (e.g. ``"org/model"``).
        filename: Filename within the repository.
        allow_network: When ``False``, never fall back to a network download —
            a cache miss re-raises the local-lookup error. Used by startup
            preload so a cold cache cannot block (or, via native crashes in the
            download stack, kill) the process before it binds its port.

    Returns:
        Absolute path to the local cached file.

    Raises:
        Any exception raised by ``hf_hub_download`` on a genuine download failure,
        or the local-lookup error when ``allow_network`` is ``False`` and the
        file is not cached.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        return str(hf_hub_download(repo_id, filename, local_files_only=True))
    except (LocalEntryNotFoundError, EntryNotFoundError, OSError):
        if not allow_network:
            raise
        return str(hf_hub_download(repo_id, filename))


def create_cpu_session_options(
    ort: Any,
    *,
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int | None = None,
) -> Any:
    """Create CPU-oriented ONNX Runtime session options.

    Headroom runs as a long-lived proxy process, so we bias toward predictable
    memory usage over peak ONNX throughput. Disabling ORT's CPU arena and memory
    pattern caches reduces retained anonymous RSS after variable-size inference
    workloads, which is especially important on small VMs.
    """
    sess_options = ort.SessionOptions()

    # Size the ONNX thread pools to the container's cpuset when the caller does
    # not pass explicit values. This keeps ORT from spawning host-topology
    # threads it then tries (and fails) to pin outside the cgroup cpuset
    # (``pthread_setaffinity_np ... error 22``). Callers that pass explicit
    # values (e.g. Kompress) are left untouched.
    if intra_op_num_threads is None:
        intra_op_num_threads = _env_thread_count("HEADROOM_ONNX_INTRA_OP_THREADS")
        if intra_op_num_threads is None:
            intra_op_num_threads = _cpuset_size()
    if inter_op_num_threads is None:
        inter_op_num_threads = _env_thread_count("HEADROOM_ONNX_INTER_OP_THREADS")
        if inter_op_num_threads is None:
            inter_op_num_threads = 1

    if intra_op_num_threads is not None:
        sess_options.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads is not None:
        sess_options.inter_op_num_threads = inter_op_num_threads

    if hasattr(sess_options, "enable_cpu_mem_arena"):
        sess_options.enable_cpu_mem_arena = False
    if hasattr(sess_options, "enable_mem_pattern"):
        sess_options.enable_mem_pattern = False

    return sess_options


def trim_process_heap() -> bool:
    """Ask glibc to return unused heap pages to the OS when available."""
    if not sys.platform.startswith("linux"):
        return False

    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return False

    try:
        return bool(libc.malloc_trim(0))
    except Exception:
        return False
