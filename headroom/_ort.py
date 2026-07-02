"""Pin the ONNX Runtime dylib for the Rust core.

Why this module exists
----------------------
``headroom._core`` consumers of the ``ort`` crate (magika content
detection, fastembed embeddings) are built with ``ort-load-dynamic`` on
every platform: the native ONNX Runtime library is resolved at
*runtime* rather than statically linked. (Static `ort-download-binaries`
linking was dropped on Linux/macOS too because Microsoft's prebuilt
x86_64 ORT requires AVX2 and executes at extension load, SIGILLing
`import headroom._core` on pre-AVX2 CPUs — #1278.) Unless
``ORT_DYLIB_PATH`` is set, ort falls back to a bare dlopen /
``LoadLibrary("onnxruntime.dll")``; on Windows the DLL search order
applies — and ``C:\\Windows\\System32`` wins.

Windows 11 24H2+ ships ``System32\\onnxruntime.dll`` as part of Windows
ML (observed: 1.17.2603 "os-germanium"). Initializing an ort 2.x
session against that OS build does not fail — it deadlocks
indefinitely at 0% CPU, which the tiered detection fallback cannot
catch (a hang is not an ``Err``). Reproduced and bracketed with
``scripts/diag_magika_windows.py``: the identical session inits in
~400ms when ``ORT_DYLIB_PATH`` points at the ``onnxruntime`` pip
package's DLL (which ``headroom-ai[proxy]`` already depends on).

The fix: before anything can import ``headroom._core``, resolve the
pip-installed ``onnxruntime`` package's shared library
(``capi/onnxruntime.dll`` / ``capi/libonnxruntime.so*`` /
``capi/libonnxruntime*.dylib``) and export it via ``ORT_DYLIB_PATH``.
``headroom/__init__.py`` calls this hook, which guarantees ordering for
every package-level consumer.

Behavior contract
-----------------
- All platforms; pins only when the ``onnxruntime`` package is present.
- Respects a pre-set ``ORT_DYLIB_PATH`` (user override wins).
- Locates the ``onnxruntime`` package via ``find_spec`` WITHOUT
  importing it (importing would load its native code; this hook must
  stay ~microseconds and side-effect free).
- Never raises: import-time failure of an optional accelerator must
  not break ``import headroom``. Without a pin, detection still
  degrades gracefully through HEADROOM_MAGIKA_INIT_TIMEOUT_SECS and
  the non-ML tiers.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_VAR = "ORT_DYLIB_PATH"

# Tri-state module cache: unset sentinel / resolved path / None (no pin).
_UNSET = object()
_pinned: object = _UNSET


def ensure_ort_dylib_pinned() -> str | None:
    """Export ``ORT_DYLIB_PATH`` for the Rust core's ort runtime.

    Returns the effective dylib path (pinned now or already present in
    the environment), or ``None`` when no pin applies (no ``onnxruntime``
    package to point at). Idempotent and exception-free.
    """
    global _pinned
    if _pinned is not _UNSET:
        return _pinned  # type: ignore[return-value]
    _pinned = _resolve_and_pin()
    return _pinned  # type: ignore[return-value]


def _find_dylib(capi: Path) -> Path | None:
    """Return the platform's ONNX Runtime shared library inside ``capi``."""
    if sys.platform.startswith("win"):
        dll = capi / "onnxruntime.dll"
        return dll if dll.is_file() else None
    # Linux ships `libonnxruntime.so.<version>`, macOS `libonnxruntime.dylib`
    # (sometimes versioned). Glob rather than hardcode the suffix.
    for pattern in ("libonnxruntime.so*", "libonnxruntime*.dylib"):
        for candidate in sorted(capi.glob(pattern)):
            if candidate.is_file():
                return candidate
    return None


def _resolve_and_pin() -> str | None:
    try:
        existing = os.environ.get(_ENV_VAR)
        if existing:
            logger.debug("%s already set; respecting user override: %s", _ENV_VAR, existing)
            return existing

        spec = importlib.util.find_spec("onnxruntime")
        if spec is None or not spec.origin:
            logger.debug(
                "onnxruntime package not found; %s left unset. The Rust ML detection "
                "cannot load ONNX Runtime and degrades to non-ML tiers (on Windows it "
                "may instead pick up the Windows ML System32 onnxruntime.dll, which is "
                "known to deadlock ort init on Windows 11 24H2+ and then degrades via "
                "HEADROOM_MAGIKA_INIT_TIMEOUT_SECS). Install onnxruntime or set %s "
                "explicitly.",
                _ENV_VAR,
                _ENV_VAR,
            )
            return None

        dll = _find_dylib(Path(spec.origin).parent / "capi")
        if dll is None:
            logger.debug(
                "onnxruntime package found but its shared library is missing; %s left unset",
                _ENV_VAR,
            )
            return None

        os.environ[_ENV_VAR] = str(dll)
        logger.info("Pinned %s to bundled ONNX Runtime: %s", _ENV_VAR, dll)
        return str(dll)
    except Exception as exc:  # never break `import headroom` over an accelerator pin
        logger.debug("ort dylib pin skipped: %s: %s", type(exc).__name__, exc)
        return None
