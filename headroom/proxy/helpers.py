"""Top-level helper functions and constants for the Headroom proxy.

Contains lazy loaders, file logging setup, request body decompression,
and safety-limit constants.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from headroom import paths as _paths

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger("headroom.proxy")

RTK_STATS_CACHE_TTL_SECONDS = 5.0
_rtk_stats_cache_lock = threading.Lock()
_rtk_stats_cache: dict[str, Any] = {
    "expires_at": 0.0,
    "has_value": False,
    "value": None,
}

# Maximum request body size (100MB - increased to support image-heavy requests)
MAX_REQUEST_BODY_SIZE = 100 * 1024 * 1024

# Maximum SSE buffer size (10MB - prevents memory exhaustion from malformed streams)
MAX_SSE_BUFFER_SIZE = 10 * 1024 * 1024

# Maximum message array length (prevents DoS from deeply nested payloads)
MAX_MESSAGE_ARRAY_LENGTH = 10000

# Compression pipeline timeout in seconds
COMPRESSION_TIMEOUT_SECONDS = 30

# Maximum compression cache sessions (prevents unbounded memory growth)
MAX_COMPRESSION_CACHE_SESSIONS = 500


def jitter_delay_ms(base_ms: int, max_ms: int, attempt: int) -> float:
    """Exponential backoff with 50-150% jitter.

    Returns ``min(base_ms * 2**attempt, max_ms) * (0.5 + random())`` — the
    canonical formula used across proxy retry loops. Extracted so every
    retry site shares one implementation.
    """
    capped: float = min(base_ms * (2**attempt), max_ms)
    return capped * (0.5 + random.random())


# Image compression (lazy-loaded to avoid heavy dependencies at startup)
_image_compressor = None


def _get_image_compressor():
    """Lazy load image compressor to avoid startup overhead."""
    global _image_compressor
    if _image_compressor is None:
        try:
            from headroom.image import ImageCompressor

            _image_compressor = ImageCompressor()
            logger.info("Image compression enabled (model: chopratejas/technique-router)")
        except ImportError as e:
            logger.warning(f"Image compression not available: {e}")
            _image_compressor = False  # Mark as unavailable
    return _image_compressor if _image_compressor else None


# Always-on file logging to the workspace logs directory for `headroom perf` analysis.
# Resolved lazily so HEADROOM_WORKSPACE_DIR env-var changes are honored.


def _headroom_log_dir() -> Path:
    return _paths.log_dir()


def _setup_file_logging() -> None:
    """Add a RotatingFileHandler to the headroom root logger.

    Writes to ~/.headroom/logs/proxy.log with automatic rotation:
    - Rotates at 10 MB
    - Keeps 5 backups (~50 MB max)
    """
    from logging.handlers import RotatingFileHandler

    try:
        log_dir = _headroom_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "proxy.log"
        handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        # Attach to the headroom root logger so all sub-loggers are captured.
        # Disable propagation to root to avoid duplicate writes when
        # wrap.py redirects stderr to the same log file.
        headroom_logger = logging.getLogger("headroom")
        if not any(isinstance(h, RotatingFileHandler) for h in headroom_logger.handlers):
            headroom_logger.addHandler(handler)
        headroom_logger.propagate = False
    except OSError:
        # Non-fatal: can't write logs (read-only fs, permissions, etc.)
        pass


def _get_rtk_stats() -> dict[str, Any] | None:
    """Get rtk (Rust Token Killer) savings stats if rtk is installed.

    Reads from rtk's tracking database via `rtk gain --format json`.
    Results are memoized briefly so dashboard polling does not spawn a new
    subprocess on every refresh.
    """
    import shutil
    import subprocess as _sp

    now = time.monotonic()
    with _rtk_stats_cache_lock:
        if _rtk_stats_cache["has_value"] and now < float(_rtk_stats_cache["expires_at"]):
            return cast(dict[str, Any] | None, _rtk_stats_cache["value"])

    payload: dict[str, Any] | None
    rtk_bin = shutil.which("rtk")
    if not rtk_bin:
        # Check headroom-managed install. Preserve the historical Unix-name
        # behavior here (bin_dir()/"rtk") rather than switching to
        # paths.rtk_path() which would become rtk.exe on Windows.
        rtk_managed = _paths.bin_dir() / "rtk"
        if rtk_managed.exists():
            rtk_bin = str(rtk_managed)
        else:
            payload = None
            with _rtk_stats_cache_lock:
                _rtk_stats_cache.update(
                    {
                        "expires_at": time.monotonic() + RTK_STATS_CACHE_TTL_SECONDS,
                        "has_value": True,
                        "value": payload,
                    }
                )
            return payload

    try:
        result = _sp.run(
            [rtk_bin, "gain", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            summary = data.get("summary", {})
            payload = {
                "installed": True,
                "total_commands": summary.get("total_commands", 0),
                "tokens_saved": summary.get("total_saved", 0),
                "avg_savings_pct": summary.get("avg_savings_pct", 0.0),
            }
        else:
            payload = {
                "installed": True,
                "total_commands": 0,
                "tokens_saved": 0,
                "avg_savings_pct": 0.0,
            }
    except Exception:
        payload = {
            "installed": True,
            "total_commands": 0,
            "tokens_saved": 0,
            "avg_savings_pct": 0.0,
        }

    with _rtk_stats_cache_lock:
        _rtk_stats_cache.update(
            {
                "expires_at": time.monotonic() + RTK_STATS_CACHE_TTL_SECONDS,
                "has_value": True,
                "value": payload,
            }
        )
    return payload


def is_anthropic_auth(headers: dict[str, str]) -> bool:
    """Detect Anthropic auth signals in request headers."""
    if headers.get("x-api-key") or headers.get("anthropic-version"):
        return True
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer sk-ant-"):
        return True
    return False


async def _read_request_json(request: Request) -> dict[str, Any]:
    """Read and parse JSON from a request, handling compressed bodies.

    Clients like OpenAI Codex may send zstd, gzip, or deflate-compressed
    request bodies.  Starlette's ``request.json()`` does not decompress
    automatically, causing a UnicodeDecodeError on compressed bytes.

    This helper inspects ``Content-Encoding``, decompresses if needed,
    then JSON-decodes the result.  It raises ``ValueError`` on any
    decompression or parse failure so callers can return a clean 400.
    """
    encoding = (request.headers.get("content-encoding") or "").lower().strip()
    raw = await request.body()

    if encoding in ("zstd", "zstandard"):
        try:
            import zstandard

            dctx = zstandard.ZstdDecompressor()
            # Use stream_reader for streaming zstd frames (no content size in header).
            # Plain decompress() fails when the frame header omits the size, which
            # is common with clients like OpenAI Codex.
            reader = dctx.stream_reader(raw)
            raw = reader.read()
            reader.close()
        except ImportError:
            raise ValueError(
                "Request body is zstd-compressed but the 'zstandard' package is not installed. "
                "Install it with: pip install zstandard"
            ) from None
        except Exception as exc:
            raise ValueError(f"Failed to decompress zstd request body: {exc}") from exc
    elif encoding == "gzip":
        import gzip as _gzip

        try:
            raw = _gzip.decompress(raw)
        except Exception as exc:
            raise ValueError(f"Failed to decompress gzip request body: {exc}") from exc
    elif encoding == "deflate":
        import zlib

        try:
            raw = zlib.decompress(raw)
        except Exception as exc:
            raise ValueError(f"Failed to decompress deflate request body: {exc}") from exc
    elif encoding == "br":
        try:
            import brotli

            raw = brotli.decompress(raw)
        except ImportError:
            raise ValueError(
                "Request body is brotli-compressed but the 'brotli' package is not installed."
            ) from None
        except Exception as exc:
            raise ValueError(f"Failed to decompress brotli request body: {exc}") from exc
    elif encoding and encoding != "identity":
        raise ValueError(f"Unsupported Content-Encoding: {encoding}")

    # Decode and parse JSON
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Request body is not valid UTF-8 (possibly compressed?): {exc}") from exc

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Request body must be a JSON object, not " + type(result).__name__)
    return result
