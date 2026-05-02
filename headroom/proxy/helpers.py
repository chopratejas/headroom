"""Top-level helper functions and constants for the Headroom proxy.

Contains lazy loaders, file logging setup, request body decompression,
and safety-limit constants.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from headroom import paths as _paths

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger("headroom.proxy")

# Memory injection mode (P0-1 fix in PR-A2).
#
# Values:
#   - "live_zone_tail" (default): Memory context appends to the first text block
#     of the latest non-frozen user message. Cache hot zone (system + frozen
#     prefix) is never mutated.
#   - "disabled": Memory context lookup is skipped entirely; the request
#     forwards untouched.
#
# Configurable via HEADROOM_MEMORY_INJECTION_MODE env var. There is no
# "system_prompt" option — that path is permanently retired by I2 (cache hot
# zone never modified). See REALIGNMENT/02-architecture.md §2.2.
_MEMORY_INJECTION_MODE_ENV = "HEADROOM_MEMORY_INJECTION_MODE"
_MEMORY_INJECTION_MODE_DEFAULT: Literal["live_zone_tail", "disabled"] = "live_zone_tail"
MemoryInjectionMode = Literal["live_zone_tail", "disabled"]


def get_memory_injection_mode() -> MemoryInjectionMode:
    """Return the active memory-injection routing mode.

    Read at request time so the env var can be flipped without restart for
    smoke tests. Unknown values are rejected loudly (no silent fallback).
    """
    raw = os.environ.get(_MEMORY_INJECTION_MODE_ENV, "").strip().lower()
    if not raw:
        return _MEMORY_INJECTION_MODE_DEFAULT
    if raw in ("live_zone_tail", "disabled"):
        return cast(MemoryInjectionMode, raw)
    raise ValueError(
        f"Invalid {_MEMORY_INJECTION_MODE_ENV}={raw!r}; expected 'live_zone_tail' or 'disabled'"
    )


def hash_query_for_log(query: str) -> str:
    """Stable short hash of a memory-context query, safe to log.

    Uses BLAKE2b truncated to 16 hex chars. Never logs the raw query content.
    """
    h = hashlib.blake2b(query.encode("utf-8", errors="replace"), digest_size=8)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Byte-faithful Python forwarder support (PR-A3 — fixes P0-2).
# ---------------------------------------------------------------------------
#
# Every Python forwarder (server.py:_retry_request, streaming.py,
# openai.py:_ws_http_fallback, batch.py) historically used
# ``httpx.AsyncClient.post(url, json=body)``. httpx's default JSON encoder
# uses ``separators=(", ", ": ")`` and ``ensure_ascii=True`` so the bytes
# leaving the proxy never byte-equal the bytes that arrived from a
# well-behaved client (Claude Code, Codex CLI emit compact + UTF-8). Every
# such request collapses Anthropic prefix-cache hit-rate.
#
# PR-A3 switches every forwarder to byte-faithful forwarding:
#   * unmutated body → forward original ``await request.body()`` verbatim;
#   * mutated body  → re-serialize once via ``serialize_body_canonical``.
#
# A ``BodyMutationTracker`` accompanies each request so the forwarder can
# pick the right path. Memory-injection / compression / image-rewrite sites
# call ``tracker.mark_mutated(reason)``.

_PYTHON_FORWARDER_MODE_ENV = "HEADROOM_PROXY_PYTHON_FORWARDER_MODE"
PythonForwarderMode = Literal["byte_faithful", "legacy_json_kwarg"]
_PYTHON_FORWARDER_MODE_DEFAULT: PythonForwarderMode = "byte_faithful"


def get_python_forwarder_mode() -> PythonForwarderMode:
    """Return the active Python-forwarder mode.

    Read at request time. Unknown values raise loudly per the no-silent-
    fallback build constraint. The ``legacy_json_kwarg`` value is an
    explicit operator opt-in for emergency rollback — NOT a fallback.
    """
    raw = os.environ.get(_PYTHON_FORWARDER_MODE_ENV, "").strip().lower()
    if not raw:
        return _PYTHON_FORWARDER_MODE_DEFAULT
    if raw in ("byte_faithful", "legacy_json_kwarg"):
        return cast(PythonForwarderMode, raw)
    raise ValueError(
        f"Invalid {_PYTHON_FORWARDER_MODE_ENV}={raw!r}; "
        "expected 'byte_faithful' or 'legacy_json_kwarg'"
    )


def serialize_body_canonical(body: dict[str, Any]) -> bytes:
    """Re-serialize a request body deterministically with cache-stable formatting.

    Uses compact separators and preserves UTF-8 (no ``\\uXXXX`` escapes), so
    byte output matches what well-behaved API clients (Claude Code, Codex
    CLI) emit. Python 3.7+ dict insertion order is preserved by
    ``json.dumps`` so message ordering is stable.

    This is the canonical re-serialization for any forwarder path that did
    mutate the body (memory injection, compression, etc.).
    """
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class BodyMutationTracker:
    """Records whether a request body was mutated and why.

    The forwarder reads ``mutated`` to decide between byte-faithful
    passthrough and canonical re-serialization. Reasons are logged with
    each outbound request to make cache-affecting decisions auditable.

    Thread-safety: a single tracker instance is owned by exactly one
    request task. No locking needed.
    """

    __slots__ = ("_mutated", "_reasons")

    def __init__(self) -> None:
        self._mutated: bool = False
        self._reasons: list[str] = []

    def mark_mutated(self, reason: str) -> None:
        """Mark the body as mutated and record the reason.

        ``reason`` should be a stable identifier (snake_case) suitable for
        log aggregation, e.g. ``memory_injection`` or
        ``compression_smart_crusher``.
        """
        if not reason:
            raise ValueError("BodyMutationTracker.mark_mutated: reason must be non-empty")
        self._mutated = True
        if reason not in self._reasons:
            self._reasons.append(reason)

    @property
    def mutated(self) -> bool:
        return self._mutated

    @property
    def reasons(self) -> list[str]:
        return list(self._reasons)


def prepare_outbound_body_bytes(
    *,
    body: dict[str, Any],
    original_body_bytes: bytes | None,
    body_mutated: bool,
    forwarder_mode: PythonForwarderMode | None = None,
) -> tuple[bytes, str]:
    """Pick the outbound body bytes for a forwarder call.

    Returns ``(outbound_bytes, source)`` where ``source`` is one of
    ``passthrough`` (original bytes verbatim), ``canonical`` (re-serialized
    deterministically because body was mutated), or ``legacy`` (rollback
    mode — old ``json=body`` behavior).

    * ``forwarder_mode == "byte_faithful"`` (default): unmutated → passthrough,
      mutated → canonical.
    * ``forwarder_mode == "legacy_json_kwarg"``: always re-encode via the old
      httpx-style separators (operator opt-in, for rollback only).
    """
    mode = forwarder_mode if forwarder_mode is not None else get_python_forwarder_mode()
    if mode == "legacy_json_kwarg":
        # Old httpx default: separators=(", ", ": "), ensure_ascii=True.
        legacy_bytes = json.dumps(body, separators=(", ", ": "), ensure_ascii=True).encode("utf-8")
        return legacy_bytes, "legacy"

    # byte_faithful path
    if body_mutated or original_body_bytes is None:
        return serialize_body_canonical(body), "canonical"
    return original_body_bytes, "passthrough"


def log_outbound_request(
    *,
    forwarder: str,
    method: str,
    path: str,
    body_bytes_count: int,
    body_mutated: bool,
    mutation_reasons: list[str],
    request_id: str | None,
    source: str,
) -> None:
    """Structured log line for every outbound forwarder call.

    Per realignment build constraints: every cache-affecting decision is
    logged. Never includes ``Authorization``/``x-api-key`` content or full
    body bytes.
    """
    logger.info(
        "event=outbound_request forwarder=%s method=%s path=%s body_bytes=%d "
        "body_mutated=%s mutation_reasons=%s source=%s request_id=%s",
        forwarder,
        method,
        path,
        body_bytes_count,
        "true" if body_mutated else "false",
        ",".join(mutation_reasons) if mutation_reasons else "",
        source,
        request_id or "",
    )


def log_memory_injection(
    *,
    request_id: str,
    session_id: str | None,
    decision: str,
    bytes_injected: int,
    query: str | None = None,
) -> None:
    """Emit a structured log line for every memory-context routing decision.

    Per realignment build constraints: log every cache-affecting decision.
    Never log raw query content or Authorization header — only a stable
    hash of the query.
    """
    query_hash = hash_query_for_log(query) if query else ""
    logger.info(
        "event=memory_injection request_id=%s session_id=%s decision=%s "
        "bytes_injected=%d query_hash=%s",
        request_id,
        session_id or "",
        decision,
        bytes_injected,
        query_hash,
    )


def append_text_to_latest_user_chat_message(
    messages: list[dict[str, Any]],
    context_text: str,
) -> tuple[list[dict[str, Any]], int]:
    """Append context text to the first text block of the latest user chat message.

    OpenAI Chat Completions ``body["messages"]`` shape: each message is
    ``{"role": ..., "content": str | list[{"type": "text"|"input_text", "text": ...}]}``.

    This is the OpenAI Chat Completions analog of
    ``_append_context_to_latest_non_frozen_user_turn`` (Anthropic) and
    ``append_text_to_latest_user_input_item`` (OpenAI Responses). Used by
    PR-A3 to retire the legacy system-prepend memory-injection path
    (P0-equivalent for /v1/chat/completions).

    Returns ``(new_messages, bytes_appended)``. ``bytes_appended == 0``
    when no eligible user message was found (no mutation occurred).
    """
    if not messages or not context_text:
        return messages, 0

    new_messages = list(messages)
    for idx in range(len(new_messages) - 1, -1, -1):
        msg = new_messages[idx]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            updated_msg = {**msg, "content": content + "\n\n" + context_text}
            new_messages[idx] = updated_msg
            return new_messages, len(context_text)

        if isinstance(content, list) and content:
            new_content: list[dict[str, Any]] = []
            appended = False
            for part in content:
                if (
                    not appended
                    and isinstance(part, dict)
                    and part.get("type") in ("text", "input_text")
                ):
                    existing_text = part.get("text", "")
                    new_part = {**part, "text": existing_text + "\n\n" + context_text}
                    new_content.append(new_part)
                    appended = True
                else:
                    new_content.append(part)
            if appended:
                updated_msg = {**msg, "content": new_content}
                new_messages[idx] = updated_msg
                return new_messages, len(context_text)

        # User message but no eligible text block — leave untouched and stop.
        return messages, 0

    return messages, 0


def append_text_to_latest_user_input_item(
    body_input: list[dict[str, Any]],
    context_text: str,
) -> tuple[list[dict[str, Any]], int]:
    """Append context text to the first text block of the latest user input item.

    Mirrors ``_append_context_to_latest_non_frozen_user_turn`` but for the
    OpenAI Responses API ``body["input"]`` shape, which uses a flat item list
    where each user item's content is a list like
    ``[{"type": "input_text", "text": "..."}]``.

    Returns a tuple ``(new_input, bytes_appended)`` where ``bytes_appended``
    is 0 when the item list was unchanged (no eligible user item).
    """
    if not body_input or not context_text:
        return body_input, 0

    new_input = list(body_input)

    for idx in range(len(new_input) - 1, -1, -1):
        item = new_input[idx]
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue

        content = item.get("content")
        if isinstance(content, str):
            updated_item = {**item, "content": content + "\n\n" + context_text}
            new_input[idx] = updated_item
            return new_input, len(context_text)

        if isinstance(content, list) and content:
            new_content: list[dict[str, Any]] = []
            appended = False
            for part in content:
                if (
                    not appended
                    and isinstance(part, dict)
                    and part.get("type") in ("input_text", "text")
                ):
                    existing_text = part.get("text", "")
                    new_part = {**part, "text": existing_text + "\n\n" + context_text}
                    new_content.append(new_part)
                    appended = True
                else:
                    new_content.append(part)
            if appended:
                updated_item = {**item, "content": new_content}
                new_input[idx] = updated_item
                return new_input, len(context_text)

        # User item but no eligible text block — leave untouched and stop.
        return body_input, 0

    return body_input, 0


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


# Image compression availability (do not retain a global compressor instance)
_image_compressor_available: bool | None = None


def _get_image_compressor():
    """Create a short-lived image compressor on demand."""
    global _image_compressor_available
    if _image_compressor_available is False:
        return None

    try:
        from headroom.image import ImageCompressor

        # Callers own closing the compressor; this helper only memoizes whether
        # the optional image stack is importable.
        compressor = ImageCompressor()
        if _image_compressor_available is None:
            logger.info("Image compression enabled (model: chopratejas/technique-router)")
        _image_compressor_available = True
        return compressor
    except ImportError as e:
        if _image_compressor_available is not False:
            logger.warning(f"Image compression not available: {e}")
        _image_compressor_available = False
        return None


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


# ---------------------------------------------------------------------------
# Internal-header stripping (PR-A5 — fixes P5-49).
# ---------------------------------------------------------------------------
#
# `x-headroom-*` request headers (e.g. ``x-headroom-bypass``,
# ``x-headroom-mode``, ``x-headroom-user-id``, ``x-headroom-stack``,
# ``x-headroom-base-url``) are internal control flags consumed by the
# proxy itself. They MUST NOT leak upstream — leaking them would (a)
# fingerprint the proxy to subscription enforcers and (b) expose the
# user-id/stack/base-url internals to whichever vendor terminates the
# request.
#
# Inbound read paths (bypass gating, ``_extract_tags`` reading
# ``x-headroom-*``, memory ``x-headroom-user-id`` lookup) keep using
# the original dict / ``request.headers``. The stripped copy is what
# every upstream-bound forwarder receives.
#
# Note: response-side ``X-Headroom-*`` injection (e.g.
# ``x-headroom-tokens-saved``) is unrelated — the proxy is allowed to
# tell its client about its own work. This helper only filters
# request-side headers.

_INTERNAL_HEADER_PREFIX = "x-headroom-"

# Operator opt-in env var. ``enabled`` (default) strips internal
# ``x-headroom-*`` headers from every upstream-bound forwarder.
# ``disabled`` is an explicit operator opt-in for diagnostic shadow
# tracing — NOT a fallback. Per realignment build constraint #4 the
# behaviour is loud, configurable, and never silent.
_STRIP_INTERNAL_HEADERS_ENV = "HEADROOM_STRIP_INTERNAL_HEADERS"
StripInternalHeadersMode = Literal["enabled", "disabled"]
_STRIP_INTERNAL_HEADERS_DEFAULT: StripInternalHeadersMode = "enabled"


def get_strip_internal_headers_mode() -> StripInternalHeadersMode:
    """Return the active internal-header strip mode.

    Read at request time so operators can flip behaviour without a
    restart. Unknown values raise loudly per the no-silent-fallback
    build constraint.
    """
    raw = os.environ.get(_STRIP_INTERNAL_HEADERS_ENV, "").strip().lower()
    if not raw:
        return _STRIP_INTERNAL_HEADERS_DEFAULT
    if raw in ("enabled", "disabled"):
        return cast(StripInternalHeadersMode, raw)
    raise ValueError(
        f"Invalid {_STRIP_INTERNAL_HEADERS_ENV}={raw!r}; expected 'enabled' or 'disabled'"
    )


def _strip_internal_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with internal ``x-headroom-*`` keys stripped.

    Used at every upstream call site to prevent fingerprinting / leakage of
    internal flags like ``x-headroom-bypass``, ``x-headroom-mode``,
    ``x-headroom-user-id``, ``x-headroom-stack``, ``x-headroom-base-url``.
    Case-insensitive on the prefix. Returns a NEW dict; never mutates the
    caller's mapping. Pure function. No regex.

    When the operator opt-in ``HEADROOM_STRIP_INTERNAL_HEADERS=disabled``
    is set, returns a shallow copy unchanged. That mode is for diagnostic
    shadow tracing only and is documented as a per-deploy choice.
    """
    mode = get_strip_internal_headers_mode()
    if mode == "disabled":
        # Always return a copy so callers can mutate without surprise.
        return dict(headers)
    return {k: v for k, v in headers.items() if not k.lower().startswith(_INTERNAL_HEADER_PREFIX)}


def log_outbound_headers(
    *,
    forwarder: str,
    stripped_count: int,
    request_id: str | None,
) -> None:
    """Structured log line for every upstream forwarder header strip.

    Emitted once per outbound request (paired with ``log_outbound_request``).
    Per realignment build constraint #8 we log every cache-affecting
    decision; per #8/#11 we never log header values, only the count of
    stripped internal headers.
    """
    logger.info(
        "event=outbound_headers forwarder=%s stripped_count=%d request_id=%s",
        forwarder,
        stripped_count,
        request_id or "",
    )


async def _read_request_body_bytes(request: Request) -> bytes:
    """Read and (if needed) decompress the request body, returning raw UTF-8 bytes.

    Mirrors ``_read_request_json`` but returns the bytes pre-parse so
    forwarders can implement byte-faithful passthrough (PR-A3, fixes P0-2).
    Raises ``ValueError`` on any decompression failure.
    """
    encoding = (request.headers.get("content-encoding") or "").lower().strip()
    raw = await request.body()

    if encoding in ("zstd", "zstandard"):
        try:
            import zstandard

            dctx = zstandard.ZstdDecompressor()
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

    return cast(bytes, raw)


async def _read_request_json(request: Request) -> dict[str, Any]:
    """Read and parse JSON from a request, handling compressed bodies.

    Clients like OpenAI Codex may send zstd, gzip, or deflate-compressed
    request bodies.  Starlette's ``request.json()`` does not decompress
    automatically, causing a UnicodeDecodeError on compressed bytes.

    This helper inspects ``Content-Encoding``, decompresses if needed,
    then JSON-decodes the result.  It raises ``ValueError`` on any
    decompression or parse failure so callers can return a clean 400.
    """
    raw = await _read_request_body_bytes(request)

    # Decode and parse JSON
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Request body is not valid UTF-8 (possibly compressed?): {exc}") from exc

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Request body must be a JSON object, not " + type(result).__name__)
    return result


async def read_request_json_with_bytes(request: Request) -> tuple[dict[str, Any], bytes]:
    """Read JSON body AND return the original (decompressed) bytes.

    Returned bytes are post-content-decoding (zstd/gzip/deflate/br are
    decompressed) so they represent the body as the upstream API will
    receive it. Forwarders pair this with a ``BodyMutationTracker`` to
    decide between passthrough and canonical re-serialization.
    """
    raw = await _read_request_body_bytes(request)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Request body is not valid UTF-8 (possibly compressed?): {exc}") from exc

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Request body must be a JSON object, not " + type(result).__name__)
    return result, raw


def _strip_per_call_annotations(obj: Any) -> Any:
    """Remove annotations that clients mutate between calls in one agent loop.

    ``cache_control`` is the main offender: clients (notably Claude Code)
    move the cache breakpoint to the newest message on each call, which
    means the exact same user-text message carries ``cache_control`` on
    call 1 and not on call 2. Hashing the raw message dicts therefore
    produces a different turn_id for every iteration of a single agent
    loop, collapsing ``turn_id`` to effectively ``request_id`` and
    breaking prompt-level aggregation downstream.
    """
    if isinstance(obj, dict):
        return {k: _strip_per_call_annotations(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_per_call_annotations(item) for item in obj]
    return obj


def compute_turn_id(
    model: str,
    system: Any,
    messages: list[dict[str, Any]] | None,
) -> str | None:
    """Group all agent-loop API calls triggered by a single user prompt.

    A turn spans the user's text prompt plus every assistant tool-use and
    user tool-result message the agent appends while executing that prompt.
    Hashing the prefix up to and including the last user *text* message yields
    an id that is stable across the turn but rolls over when the user sends a
    new prompt.

    Returns None when no user-text message is present (nothing to identify).
    """
    if not messages:
        return None

    last_text_user_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            last_text_user_idx = i
            break
        if isinstance(content, list):
            has_text = any(
                isinstance(block, dict) and block.get("type") == "text" for block in content
            )
            has_tool_result = any(
                isinstance(block, dict) and block.get("type") == "tool_result" for block in content
            )
            # An agent-loop continuation carries tool_result blocks; only a
            # fresh user turn is text-only.
            if has_text and not has_tool_result:
                last_text_user_idx = i
                break

    if last_text_user_idx is None:
        return None

    prefix = _strip_per_call_annotations(messages[: last_text_user_idx + 1])
    try:
        prefix_json = json.dumps(prefix, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None

    h = hashlib.sha256()
    h.update(model.encode("utf-8", errors="replace"))
    h.update(b"\0")
    if isinstance(system, str):
        h.update(system.encode("utf-8", errors="replace"))
    elif system is not None:
        try:
            normalized_system = _strip_per_call_annotations(system)
            h.update(json.dumps(normalized_system, sort_keys=True, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            pass
    h.update(b"\0")
    h.update(prefix_json.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]
