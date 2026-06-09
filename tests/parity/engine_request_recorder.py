"""Characterization golden recorder for Chunk 4.1.

Captures the exact outbound bytes the current Anthropic proxy handler sends
upstream, over a controlled corpus, and writes them as golden fixtures under
tests/parity/fixtures/engine_request_golden/<name>.json.

The recorded ``outbound_b64`` is the byte-exact parity oracle: later steps
(Chunk 4.2 and beyond) run the same corpus through the refactored engine path
and assert identical bytes.

Interception point
------------------
The proxy's ``http_client`` (an ``httpx.AsyncClient``) is swapped for one
backed by ``_CapturingTransport`` — an ``httpx.AsyncBaseTransport`` that
collects every ``content=...`` value before returning a synthetic 200.  This
captures the canonical forwarder output regardless of whether the call goes
through ``_retry_request`` (non-streaming) or ``_stream_response`` (streaming),
because both ultimately call ``self.http_client.post(url, content=...)`` or
``self.http_client.build_request(..., content=...)``.

Determinism
-----------
The following nondeterministic sources are suppressed:
  * ``request_id`` / ``trace_session_id`` — used for logging only; they do
    not appear in the outbound body.  Confirmed by tracing the handler: the
    only body mutations that touch ``request_id`` are purely logging calls.
  * ``session_tracker_store`` — replaced with a controlled ``_FixedTracker``
    so ``PrefixCacheTracker`` state does not leak across cases.
  * ``_get_compression_cache`` — replaced with a fresh ``_FreshCache`` per case
    so no session-scoped compression history leaks between runs.
  * Wall-clock timestamps — the handler records them for telemetry but they
    never reach the outbound body.
  * Python compression ML paths (ProactiveExpansion, LLMLingua) — excluded
    from the byte-exact corpus; flagged explicitly in this file and the fixture
    registry.

Usage
-----
Called from ``tests/parity/test_engine_request_golden.py`` (via
``seed_all_golden_fixtures()``), or directly from a helper script.  In CI,
the fixture files are committed; recording is only re-run when a case spec
changes (``overwrite=True``) or a fixture is missing.
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture locations
# ---------------------------------------------------------------------------

_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "engine_request_golden"
# Separate directory for CCR-ON fixtures so load_all_golden_fixtures (which
# scans _FIXTURES_ROOT) does not pick them up and feed them to the non-CCR
# parity test (test_engine_request_parity.py), which would fail because the
# non-CCR engine does not inject CCR tools.
_CCR_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "ccr_request_golden"


# ---------------------------------------------------------------------------
# _CapturingTransport — intercepts http_client calls at the transport level
# ---------------------------------------------------------------------------


class _CapturingTransport(httpx.AsyncBaseTransport):
    """Records the exact ``content=`` bytes delivered to the upstream client.

    Works for both non-streaming (``http_client.post(url, content=...)``) and
    streaming (``http_client.build_request(..., content=...)`` /
    ``http_client.send(..., stream=True)``).  The transport-level handler is
    the same code path for both call styles.
    """

    def __init__(self) -> None:
        self.captured_body: bytes | None = None
        self.captured_headers: dict[str, str] | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body
        self.captured_headers = dict(request.headers.items())
        # Return a minimal valid Anthropic response so the handler proceeds
        # without error.
        return httpx.Response(
            200,
            json={
                "id": "msg_parity_golden",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )


class _StreamingCapturingTransport(httpx.AsyncBaseTransport):
    """Like ``_CapturingTransport`` but returns a minimal SSE body so the
    streaming path does not raise on response parsing."""

    def __init__(self) -> None:
        self.captured_body: bytes | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body

        sse_body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"id":"msg_parity_stream",'
            b'"type":"message","role":"assistant","content":[],"model":"claude",'
            b'"usage":{"input_tokens":10,"output_tokens":0,'
            b'"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}\n\n'
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"output_tokens":3}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )
        return httpx.Response(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )


# ---------------------------------------------------------------------------
# Session-state stubs — isolate per-case state
# ---------------------------------------------------------------------------


class _FixedTracker:
    """Deterministic stand-in for PrefixCacheTracker.

    ``frozen_count`` controls how many prefix messages appear frozen to the
    handler, which affects the cache-aligner's injection site.
    """

    def __init__(self, frozen_count: int = 0) -> None:
        self._frozen_count = frozen_count
        self._cached_token_count = 0
        self._last_original_messages: list[Any] = []
        self._last_forwarded_messages: list[Any] = []

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    def get_last_original_messages(self) -> list[Any]:
        return list(self._last_original_messages)

    def get_last_forwarded_messages(self) -> list[Any]:
        return list(self._last_forwarded_messages)

    def update_from_response(self, **kwargs: Any) -> None:
        self._last_original_messages = list(
            kwargs.get("original_messages", kwargs.get("messages", []))
        )
        self._last_forwarded_messages = list(kwargs.get("messages", []))


class _FreshCompressionCache:
    """Minimal stub for CompressionCache — returns messages unmodified.

    Fresh per-case so no compression-history state leaks between runs.
    """

    def apply_cached(self, messages: list[Any]) -> list[Any]:
        return list(messages)

    def compute_frozen_count(self, messages: list[Any]) -> int:
        return 0

    def update_from_result(self, originals: list[Any], compressed: list[Any]) -> None:
        pass

    def mark_stable_from_messages(self, messages: list[Any], up_to: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Case spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenCaseSpec:
    """Input specification for one golden fixture.

    Fields
    ------
    name:
        Unique identifier; becomes the fixture filename.
    inbound_headers:
        HTTP headers the client sends (``x-api-key``, ``anthropic-version``, etc.)
        These control auth-mode classification and bypass detection.
    body:
        The JSON-parsed request body to send as compact JSON.
    proxy_config:
        ``ProxyConfig`` kwargs passed to ``create_app`` for this case.
        Each case carries its own config so mode/optimize are explicit.
    frozen_count:
        Number of messages the ``_FixedTracker`` reports as frozen.
        Only relevant when the handler reads ``get_frozen_message_count()``.
    streaming:
        Whether to send ``stream=True`` (uses _StreamingCapturingTransport).
    notes:
        Human-readable explanation of what this case exercises.
    nondeterministic_flag:
        If True: this case is NOT byte-exact (e.g. ML randomness).
        Set this to skip the byte-parity assertion and record for reference.
        A loud comment must accompany any case with this flag set.
    prev_original_messages:
        If set, pre-seed ``_FixedTracker._last_original_messages`` so the
        handler's multi-turn delta path sees a prior turn.
    prev_forwarded_messages:
        Companion to ``prev_original_messages``; pre-seeds
        ``_FixedTracker._last_forwarded_messages``.
    """

    name: str
    inbound_headers: dict[str, str]
    body: dict[str, Any]
    proxy_config: dict[str, Any] = field(default_factory=dict)
    frozen_count: int = 0
    streaming: bool = False
    notes: str = ""
    nondeterministic_flag: bool = False
    prev_original_messages: list[Any] = field(default_factory=list)
    prev_forwarded_messages: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_KWARGS = {
    "cache_enabled": False,
    "rate_limit_enabled": False,
    "cost_tracking_enabled": False,
    "log_requests": False,
    "ccr_inject_tool": False,
    "ccr_handle_responses": False,
    "ccr_context_tracking": False,
    "image_optimize": False,
}


def _make_app_and_transport(
    spec: GoldenCaseSpec,
) -> tuple[TestClient, _CapturingTransport | _StreamingCapturingTransport]:
    """Build a proxy app wired to a capturing transport for ``spec``."""
    config_kwargs = {**_DEFAULT_CONFIG_KWARGS, **spec.proxy_config}
    config = ProxyConfig(**config_kwargs)
    app = create_app(config)

    transport: _CapturingTransport | _StreamingCapturingTransport
    if spec.streaming:
        transport = _StreamingCapturingTransport()
    else:
        transport = _CapturingTransport()

    proxy = app.state.proxy
    proxy.http_client = httpx.AsyncClient(transport=transport)

    # Pin session tracker — deterministic frozen_count, no leakage.
    tracker = _FixedTracker(frozen_count=spec.frozen_count)
    if spec.prev_original_messages:
        tracker._last_original_messages = list(spec.prev_original_messages)
    if spec.prev_forwarded_messages:
        tracker._last_forwarded_messages = list(spec.prev_forwarded_messages)

    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
        f"golden-{spec.name}"
    )
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker

    # Pin compression cache — fresh per case, no history leakage.
    proxy._get_compression_cache = lambda session_id: _FreshCompressionCache()

    return TestClient(app), transport


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

_STANDARD_HEADERS = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}


def _infer_auth_mode(headers: dict[str, str]) -> str:
    from headroom.proxy.auth_mode import classify_auth_mode

    return classify_auth_mode(headers).value


def record_golden_fixture(
    spec: GoldenCaseSpec,
    *,
    root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Drive the real proxy handler with a capturing transport and write a fixture.

    Parameters
    ----------
    spec:
        Input specification for this golden case.
    root:
        Override fixture output directory (for tests).
    overwrite:
        If False (default) and fixture exists, skip and return existing path.

    Returns
    -------
    Path to the written (or existing) fixture file.

    Raises
    ------
    RuntimeError
        If the transport captured no body — indicates the handler did not
        call the upstream client (e.g. raised before reaching the forward
        step).  A partial fixture is worse than no fixture.

    Determinism note — ``inbound_b64``
    ------------------------------------
    The fixture stores the inbound body bytes as ``inbound_b64`` alongside
    the outbound golden ``outbound_b64``.  On replay, ``inbound_b64`` is used
    to reconstruct the exact bytes sent to the handler — this eliminates
    key-order sensitivity from JSON ``sort_keys=True`` serialization of the
    fixture file.  Without it, loading ``body`` from a ``sort_keys=True``
    fixture would reorder keys alphabetically, producing different inbound
    bytes and therefore different (non-golden) outbound bytes for passthrough
    cases.
    """
    out_dir = root or _FIXTURES_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}.json"

    if out_path.exists() and not overwrite:
        return out_path

    client, transport = _make_app_and_transport(spec)

    # Use insertion-order serialization so inbound bytes match the spec dict
    # ordering exactly.  This is what gets stored as inbound_b64 and what
    # the replay uses to reconstruct the exact bytes sent to the handler.
    body_bytes = json.dumps(spec.body, separators=(",", ":"), ensure_ascii=False).encode()

    if spec.streaming:
        # For streaming, drain the response so the SSE finalizer runs.
        with client.stream(
            "POST",
            "/v1/messages",
            headers=spec.inbound_headers,
            content=body_bytes,
        ) as resp:
            for _ in resp.iter_bytes():
                pass
        captured_body = transport.captured_body  # type: ignore[union-attr]
    else:
        resp = client.post(
            "/v1/messages",
            headers=spec.inbound_headers,
            content=body_bytes,
        )
        if resp.status_code not in (200, 400):
            raise RuntimeError(
                f"Golden recording for '{spec.name}' got HTTP {resp.status_code}: {resp.text[:200]}"
            )
        captured_body = transport.captured_body  # type: ignore[union-attr]

    if captured_body is None:
        raise RuntimeError(
            f"Golden recording for '{spec.name}': transport captured no body. "
            "The handler may have returned before reaching the upstream send call. "
            "Check for early-exit conditions (validation errors, missing fields, etc.)."
        )

    auth_mode = _infer_auth_mode(spec.inbound_headers)

    fixture: dict[str, Any] = {
        "name": spec.name,
        "auth_mode": auth_mode,
        "headers": dict(spec.inbound_headers),
        # ``inbound_b64`` is the canonical replay key — stores exact bytes sent
        # to the handler so replay is independent of JSON key-order in the file.
        "inbound_b64": base64.b64encode(body_bytes).decode(),
        # ``body`` is stored for human readability only; not used on replay.
        "body": spec.body,
        "proxy_config": spec.proxy_config,
        "frozen_count": spec.frozen_count,
        "streaming": spec.streaming,
        "notes": spec.notes,
        "nondeterministic_flag": spec.nondeterministic_flag,
        "prev_original_messages": spec.prev_original_messages,
        "prev_forwarded_messages": spec.prev_forwarded_messages,
        "outbound_b64": base64.b64encode(captured_body).decode(),
        "recorded_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }

    out_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Fixture loader (for the characterization test)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenFixture:
    """Parsed representation of one engine_request_golden fixture."""

    name: str
    auth_mode: str
    headers: dict[str, str]
    body: dict[str, Any]  # human-readable; not used for replay
    inbound_bytes: bytes  # decoded from inbound_b64; used for exact replay
    proxy_config: dict[str, Any]
    frozen_count: int
    streaming: bool
    notes: str
    nondeterministic_flag: bool
    prev_original_messages: list[Any]
    prev_forwarded_messages: list[Any]
    outbound_bytes: bytes  # decoded from outbound_b64
    recorded_at: str


def load_golden_fixture(path: Path) -> GoldenFixture:
    """Parse one engine_request_golden fixture.  Raises loudly on malformed input."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in golden fixture {path}: {exc}") from exc

    required = {
        "name",
        "auth_mode",
        "headers",
        "inbound_b64",
        "outbound_b64",
        "recorded_at",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Golden fixture {path} missing required keys: {missing!r}")

    try:
        inbound_bytes = base64.b64decode(data["inbound_b64"])
    except Exception as exc:
        raise ValueError(f"Golden fixture {path}: bad inbound_b64: {exc}") from exc

    try:
        outbound_bytes = base64.b64decode(data["outbound_b64"])
    except Exception as exc:
        raise ValueError(f"Golden fixture {path}: bad outbound_b64: {exc}") from exc

    return GoldenFixture(
        name=data["name"],
        auth_mode=data["auth_mode"],
        headers=dict(data["headers"]),
        body=dict(data.get("body", {})),
        inbound_bytes=inbound_bytes,
        proxy_config=dict(data.get("proxy_config", {})),
        frozen_count=int(data.get("frozen_count", 0)),
        streaming=bool(data.get("streaming", False)),
        notes=str(data.get("notes", "")),
        nondeterministic_flag=bool(data.get("nondeterministic_flag", False)),
        prev_original_messages=list(data.get("prev_original_messages", [])),
        prev_forwarded_messages=list(data.get("prev_forwarded_messages", [])),
        outbound_bytes=outbound_bytes,
        recorded_at=data["recorded_at"],
    )


def load_all_golden_fixtures(root: Path | None = None) -> list[GoldenFixture]:
    """Load all *.json fixtures from the engine_request_golden directory."""
    fixture_dir = root or _FIXTURES_ROOT
    paths = sorted(fixture_dir.glob("*.json"))
    return [load_golden_fixture(p) for p in paths]


# ---------------------------------------------------------------------------
# Replay helper — used by the characterization test
# ---------------------------------------------------------------------------


def replay_golden_fixture(fix: GoldenFixture) -> bytes:
    """Re-drive the current handler for ``fix`` and return captured outbound bytes.

    Uses ``fix.inbound_bytes`` directly (not ``fix.body``) to preserve the exact
    byte order recorded at fixture creation time.  This eliminates key-order
    sensitivity from JSON ``sort_keys=True`` serialization of the fixture file.

    Used by ``test_engine_request_golden.py`` to assert byte parity with the
    recorded golden.
    """
    spec = GoldenCaseSpec(
        name=fix.name,
        inbound_headers=fix.headers,
        body=fix.body,  # used only by _make_app_and_transport for config; not re-serialized
        proxy_config=fix.proxy_config,
        frozen_count=fix.frozen_count,
        streaming=fix.streaming,
        notes=fix.notes,
        nondeterministic_flag=fix.nondeterministic_flag,
        prev_original_messages=fix.prev_original_messages,
        prev_forwarded_messages=fix.prev_forwarded_messages,
    )
    client, transport = _make_app_and_transport(spec)

    # Use inbound_bytes directly — the exact bytes recorded at fixture creation.
    # Never re-serialize fix.body; JSON sort_keys would reorder keys.
    body_bytes = fix.inbound_bytes

    if spec.streaming:
        with client.stream(
            "POST",
            "/v1/messages",
            headers=spec.inbound_headers,
            content=body_bytes,
        ) as resp:
            for _ in resp.iter_bytes():
                pass
        captured = transport.captured_body  # type: ignore[union-attr]
    else:
        resp = client.post(
            "/v1/messages",
            headers=spec.inbound_headers,
            content=body_bytes,
        )
        captured = transport.captured_body  # type: ignore[union-attr]

    if captured is None:
        raise RuntimeError(
            f"Replay for golden fixture '{fix.name}': transport captured no body. "
            "The handler may have exited before reaching the upstream send call."
        )
    return captured


# ---------------------------------------------------------------------------
# Corpus definition
# ---------------------------------------------------------------------------

_LARGE_TOOL_RESULT_CONTENT = (
    # ~1 KB of log output — large enough that the ContentRouter routes to
    # LogCompressor if optimize=True / mode=token.
    "INFO [2026-05-29T10:00:00Z] Starting batch job batch-9821\n"
    "INFO [2026-05-29T10:00:01Z] Loading model weights from /data/models/v7\n"
    "INFO [2026-05-29T10:00:05Z] Model loaded (3.2 GB) in 4.1s\n"
    "WARN [2026-05-29T10:00:05Z] GPU memory fragmented; defragmenting\n"
    "INFO [2026-05-29T10:00:07Z] Processing shard 1/8 (125k records)\n"
    "INFO [2026-05-29T10:00:15Z] Shard 1 done: 124,982 processed, 18 skipped\n"
    "INFO [2026-05-29T10:00:15Z] Processing shard 2/8 (125k records)\n"
    "INFO [2026-05-29T10:00:23Z] Shard 2 done: 124,999 processed, 1 skipped\n"
    "ERROR [2026-05-29T10:00:24Z] Shard 3: connection reset by peer during write\n"
    "WARN [2026-05-29T10:00:24Z] Retrying shard 3 (attempt 1/3)\n"
    "INFO [2026-05-29T10:00:28Z] Shard 3 retry succeeded\n"
    "INFO [2026-05-29T10:00:28Z] Processing shard 4/8 (125k records)\n"
    "INFO [2026-05-29T10:00:37Z] Shard 4 done: 125,000 processed, 0 skipped\n"
    "INFO [2026-05-29T10:00:37Z] Processing shard 5/8 (125k records)\n"
    "INFO [2026-05-29T10:00:46Z] Shard 5 done: 125,000 processed, 0 skipped\n"
    "INFO [2026-05-29T10:00:46Z] Processing shard 6/8 (125k records)\n"
    "INFO [2026-05-29T10:00:55Z] Shard 6 done: 124,950 processed, 50 skipped\n"
    "INFO [2026-05-29T10:00:55Z] Processing shard 7/8 (125k records)\n"
    "INFO [2026-05-29T10:01:05Z] Shard 7 done: 125,000 processed, 0 skipped\n"
    "INFO [2026-05-29T10:01:05Z] Processing shard 8/8 (125k records)\n"
    "INFO [2026-05-29T10:01:14Z] Shard 8 done: 125,000 processed, 0 skipped\n"
    "INFO [2026-05-29T10:01:14Z] All shards complete. Total: 999,931 processed\n"
    "INFO [2026-05-29T10:01:14Z] Writing results to s3://data-prod/batch-9821/\n"
    "INFO [2026-05-29T10:01:18Z] Upload complete (42 MB in 4s)\n"
    "INFO [2026-05-29T10:01:18Z] Job batch-9821 finished OK\n"
)

_CORPUS: list[GoldenCaseSpec] = [
    # ── 1. Passthrough — PAYG, bypass header, no optimize ───────────────────
    GoldenCaseSpec(
        name="anthropic_payg_bypass_header",
        inbound_headers={
            **_STANDARD_HEADERS,
            "x-headroom-bypass": "true",
        },
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        },
        proxy_config={"optimize": False},
        notes="PAYG auth; bypass header → handler forwards original bytes verbatim",
    ),
    # ── 2. Passthrough — PAYG, optimize=False (compression disabled) ─────────
    GoldenCaseSpec(
        name="anthropic_payg_no_optimize",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        proxy_config={"optimize": False},
        notes="optimize=False → passthrough; canonical serialization of small body",
    ),
    # ── 3. Passthrough — OAuth bearer, optimize=False ────────────────────────
    GoldenCaseSpec(
        name="anthropic_oauth_no_optimize",
        inbound_headers={
            "authorization": "Bearer sk-ant-oat-abc123def456",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Summarize this document."}],
        },
        proxy_config={"optimize": False},
        notes="OAuth auth mode (sk-ant-oat-*); optimize=False → passthrough",
    ),
    # ── 4. Passthrough — subscription CLI token, optimize=False ─────────────
    GoldenCaseSpec(
        name="anthropic_subscription_no_optimize",
        inbound_headers={
            "authorization": "Bearer sk-ant-oat-sub-xyz789",
            "x-headroom-client": "claude-code",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        body={
            "model": "claude-opus-4-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Write a poem."}],
        },
        proxy_config={"optimize": False},
        notes="Subscription client token; optimize=False → passthrough",
    ),
    # ── 5. Cache mode — no tools, frozen_count=0 ────────────────────────────
    GoldenCaseSpec(
        name="anthropic_cache_mode_no_tools",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "Turn 1 question"},
                {"role": "assistant", "content": "Turn 1 answer"},
                {"role": "user", "content": "Turn 2 question"},
            ],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        frozen_count=0,
        notes="cache mode; frozen=0; all messages live; CacheAligner may add cache_control",
    ),
    # ── 6. Cache mode — with tools (deterministic sort) ─────────────────────
    GoldenCaseSpec(
        name="anthropic_cache_mode_tools_sorted",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Run the search tool."}],
            "tools": [
                {"name": "zeta_tool", "description": "z", "input_schema": {"type": "object"}},
                {"name": "alpha_tool", "description": "a", "input_schema": {"type": "object"}},
                {"name": "mu_tool", "description": "m", "input_schema": {"type": "object"}},
            ],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        frozen_count=0,
        notes="cache mode; tools must be sorted alpha before forwarding",
    ),
    # ── 7. Cache mode — tools absent ─────────────────────────────────────────
    GoldenCaseSpec(
        name="anthropic_cache_mode_no_tools_single_turn",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        frozen_count=0,
        notes="cache mode; no tools field; single turn",
    ),
    # ── 8. Token mode — multi-turn, frozen_count=2 ───────────────────────────
    GoldenCaseSpec(
        name="anthropic_token_mode_frozen_prefix",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "First message (frozen)"},
                {"role": "assistant", "content": "First reply (frozen)"},
                {"role": "user", "content": "Current question"},
            ],
        },
        proxy_config={"optimize": True, "mode": "token"},
        frozen_count=2,
        notes="token mode; prefix_tracker reports 2 frozen; pipeline sees only suffix",
    ),
    # ── 9. Non-streaming, bypass passthrough — unicode content ───────────────
    GoldenCaseSpec(
        name="anthropic_payg_unicode_bypass",
        inbound_headers={
            **_STANDARD_HEADERS,
            "x-headroom-bypass": "true",
        },
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hello 🔥 — 世界 — emoji is 🚀"}],
        },
        proxy_config={"optimize": False},
        notes="Unicode in body; bypass; confirms no \\uXXXX escaping in outbound bytes",
    ),
    # ── 10. Cache mode with system prompt ────────────────────────────────────
    GoldenCaseSpec(
        name="anthropic_cache_mode_with_system",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "system": "You are a helpful assistant. Always be concise.",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        frozen_count=0,
        notes="cache mode; system prompt present; locks cache_control injection site",
    ),
    # ── 11. Streaming — passthrough (no optimize) ────────────────────────────
    GoldenCaseSpec(
        name="anthropic_streaming_no_optimize",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "Stream me a response."}],
        },
        proxy_config={"optimize": False},
        streaming=True,
        notes="Streaming path; optimize=False; confirms streaming transport interception works",
    ),
    # ── 12. Token mode — large tool_result (exercises ContentRouter) ─────────
    GoldenCaseSpec(
        name="anthropic_token_mode_large_tool_result",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_batch_job",
                            "name": "run_batch_job",
                            "input": {"job_id": "batch-9821"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_batch_job",
                            "content": _LARGE_TOOL_RESULT_CONTENT,
                        }
                    ],
                },
            ],
        },
        proxy_config={"optimize": True, "mode": "token"},
        frozen_count=0,
        notes=(
            "token mode; large tool_result log text exercises ContentRouter→LogCompressor; "
            "real compression path — deterministic because empty query → BM25 short-circuits"
        ),
    ),
    # ── 13. Cache mode — frozen_count=2 (prefix stability) ──────────────────
    GoldenCaseSpec(
        name="anthropic_cache_mode_frozen_prefix",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "Frozen turn 1"},
                {"role": "assistant", "content": "Frozen answer 1"},
                {"role": "user", "content": "Frozen turn 2"},
                {"role": "assistant", "content": "Frozen answer 2"},
                {"role": "user", "content": "Current live turn"},
            ],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        frozen_count=4,
        notes=(
            "cache mode; frozen_count=4 forces prefix_tracker to report 4 frozen; "
            "only the last message is live"
        ),
    ),
    # ── 14. Streaming — cache mode ───────────────────────────────────────────
    GoldenCaseSpec(
        name="anthropic_streaming_cache_mode",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Previous"},
                {"role": "assistant", "content": "Ack"},
                {"role": "user", "content": "Now"},
            ],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        frozen_count=0,
        streaming=True,
        notes="streaming + cache mode; CacheAligner may inject cache_control on live turns",
    ),
    # ── 15. Token mode — multi-turn with prior forwarded messages (delta) ────
    GoldenCaseSpec(
        name="anthropic_token_mode_delta_compression",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "Shared prefix text for turn 1"},
                {"role": "assistant", "content": "Turn 1 answer"},
                {"role": "user", "content": "New current question turn 2"},
            ],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        frozen_count=0,
        prev_original_messages=[
            {"role": "user", "content": "Shared prefix text for turn 1"},
            {"role": "assistant", "content": "Turn 1 answer"},
        ],
        prev_forwarded_messages=[
            {"role": "user", "content": "Shared prefix text for turn 1"},
            {"role": "assistant", "content": "Turn 1 answer"},
        ],
        notes=(
            "cache mode delta path; tracker has prior original+forwarded messages; "
            "handler should reuse prior forwarded prefix and only act on new suffix"
        ),
    ),
    # ── 16. PAYG — no optimize, large body (float/int precision) ─────────────
    GoldenCaseSpec(
        name="anthropic_payg_numeric_precision",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "temperature": 1.0,
            "top_p": 0.95,
            "messages": [{"role": "user", "content": "hi"}],
        },
        proxy_config={"optimize": False},
        notes=(
            "No optimize; numeric fields (temperature=1.0, top_p=0.95) in body; "
            "confirms canonical serializer preserves float precision"
        ),
    ),
    # ── 17. Cache mode — no optimize, x-headroom-mode=passthrough ───────────
    GoldenCaseSpec(
        name="anthropic_passthrough_mode_header",
        inbound_headers={
            **_STANDARD_HEADERS,
            "x-headroom-mode": "passthrough",
        },
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "passthrough via mode header"}],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        notes="x-headroom-mode=passthrough triggers bypass just like x-headroom-bypass=true",
    ),
    # ── 18. Token mode — no messages body ────────────────────────────────────
    GoldenCaseSpec(
        name="anthropic_token_mode_no_messages",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [],
        },
        proxy_config={"optimize": True, "mode": "token"},
        frozen_count=0,
        notes="empty messages array → handler should passthrough without compression",
    ),
    # ── 19. No-compress passthrough with unsorted tools (D1 fix) ─────────────
    # optimize=False → CompressionDecision.should_compress=False (no-compress path).
    # Tools arrive in reverse-alphabetical order; legacy handler sorts them at
    # the pre-send site (~line 1634) unconditionally, producing sorted outbound.
    # Engine D1 fix applies the same sort + byte-faithful serialize.
    GoldenCaseSpec(
        name="anthropic_payg_no_optimize_unsorted_tools",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Use a tool"}],
            "tools": [
                {"name": "zeta_tool", "description": "z", "input_schema": {"type": "object"}},
                {"name": "alpha_tool", "description": "a", "input_schema": {"type": "object"}},
            ],
        },
        proxy_config={"optimize": False},
        frozen_count=0,
        notes=(
            "D1 fix: no-compress path (optimize=False) with unsorted tools; "
            "legacy handler sorts tools unconditionally at pre-send site; "
            "engine must match — byte-exact after tool-sort"
        ),
    ),
]

# Cases that are intentionally EXCLUDED from the byte-exact corpus and why:
#
# CCR proactive-expansion:
#   CCRContextTracker._proactive_expansion uses ML scoring that is not
#   deterministic across Python version / onnxruntime builds.  To include
#   a CCR case we would need to seed the expansion result; that is deferred.
#
# Image compression:
#   image_optimize=True paths call PIL/ONNX for WebP conversion; file-level
#   nondeterminism in image codecs (quality rounding) makes byte-exact golden
#   unstable.  Deferred until a seeded/mocked path is wired.
#
# Memory injection:
#   memory_handler.search_and_format_context is async and makes SQLite/HNSW
#   calls; mocking it here is straightforward but the resulting outbound bytes
#   depend on what "MEMCTX" the mock returns, not the current handler behavior.
#   Add if a controlled mock-injection case is needed.
#
# LLMLingua / IntelligentContext:
#   These transforms use ML models with float-level nondeterminism.
#   Excluded from byte-exact golden corpus.

DEFERRED_CASES = [
    "ccr_proactive_expansion",
    "image_compression",
    "memory_injection",
    "llmlingua",
    "intelligent_context",
]


def seed_all_golden_fixtures(
    *,
    root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Record all corpus cases and return {name: path} for recorded fixtures."""
    results: dict[str, Path] = {}
    for spec in _CORPUS:
        path = record_golden_fixture(spec, root=root, overwrite=overwrite)
        results[spec.name] = path
    return results


# ---------------------------------------------------------------------------
# CCR corpus — deterministic CCR-ON golden cases (Chunk 4.2b)
#
# These fixtures use ccr_inject_tool=True and/or ccr_inject_system_instructions=True.
# To make the marker scan deterministic without running actual compression,
# the body contains a pre-formed CCR marker (exact 24-hex hash) in a
# tool_result — scan_for_markers detects it and the tool/system injection fires.
#
# Proactive expansion is excluded from this corpus (ccr_proactive_expansion=False)
# because analyze_query is ML-nondeterministic; that behaviour is covered by
# structural (mock-based) tests in tests/engine/test_facade_ccr.py.
# ---------------------------------------------------------------------------

# A valid 24-character hex hash that scan_for_markers will detect.
_CCR_TEST_HASH = "abcdef123456789012345678"
# A CCR marker in the exact format scan_for_markers recognises.
_CCR_MARKER = f"[100 items compressed to 10. Retrieve more: hash={_CCR_TEST_HASH}]"

_CCR_CORPUS: list[GoldenCaseSpec] = [
    # ── CCR-1. tool injection fires (frozen=0) ────────────────────────────────
    # Body has a CCR marker in a tool_result + no existing tools → handler
    # injects the headroom_retrieve tool. frozen=0 so tool injection is allowed.
    # No system message → system instructions are injected (ccr_inject_system_instructions=True).
    GoldenCaseSpec(
        name="ccr_tool_inject_frozen0",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_search_01",
                            "name": "search_files",
                            "input": {"pattern": "*.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_search_01",
                            "content": f"Found 100 Python files. {_CCR_MARKER}",
                        }
                    ],
                },
            ],
        },
        proxy_config={
            "optimize": True,
            "mode": "token",
            "ccr_inject_tool": True,
            "ccr_inject_system_instructions": True,
            "ccr_handle_responses": False,
            "ccr_context_tracking": False,
        },
        frozen_count=0,
        notes=(
            "CCR-ON; frozen=0; body contains a CCR marker in tool_result; "
            "handler injects headroom_retrieve tool + system instructions; "
            "byte-exact because tool injection is deterministic"
        ),
    ),
    # ── CCR-2. frozen prefix in cache mode — tool injection suppressed ────────
    # In cache mode with frozen_count=2, _strict_previous_turn_frozen_count raises
    # the frozen count further. In token mode, _FreshCompressionCache.compute_frozen_count
    # returns 0, which clamps frozen_count to min(2,0)=0 — so frozen guard never fires.
    # Use cache mode to actually exercise frozen suppression of system instructions.
    GoldenCaseSpec(
        name="ccr_tool_inject_frozen_suppressed",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "Frozen turn 1 (cache hot zone)"},
                {"role": "assistant", "content": "Reply to frozen turn 1"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_search_02",
                            "content": f"Results compressed. {_CCR_MARKER}",
                        }
                    ],
                },
            ],
        },
        proxy_config={
            "optimize": True,
            "mode": "cache",
            "ccr_inject_tool": True,
            "ccr_inject_system_instructions": True,
            "ccr_handle_responses": False,
            "ccr_context_tracking": False,
        },
        frozen_count=2,
        notes=(
            "CCR-ON; cache mode; frozen_count=2; _strict_previous_turn_frozen_count pushes "
            "frozen_message_count to max(2,2)=2 (final turn is user → idx=2); "
            "inject_tool and inject_system_instructions are both suppressed by the frozen gate; "
            "body has CCR marker but neither tool nor system-instruction is injected; "
            "byte-exact — deterministic"
        ),
    ),
    # ── CCR-3. tool injection only (system instructions disabled) ─────────────
    # ccr_inject_system_instructions=False → only tool injection fires.
    # Existing tools=[] in body (no pre-existing tools).
    GoldenCaseSpec(
        name="ccr_tool_inject_no_system_instr",
        inbound_headers=_STANDARD_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "tools": [],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_search_03",
                            "content": f"Compressed output. {_CCR_MARKER}",
                        }
                    ],
                },
            ],
        },
        proxy_config={
            "optimize": True,
            "mode": "token",
            "ccr_inject_tool": True,
            "ccr_inject_system_instructions": False,
            "ccr_handle_responses": False,
            "ccr_context_tracking": False,
        },
        frozen_count=0,
        notes=(
            "CCR-ON; ccr_inject_system_instructions=False; "
            "tool injection fires but no system-instruction block is prepended; "
            "byte-exact"
        ),
    ),
]


def seed_ccr_golden_fixtures(
    *,
    root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Record all CCR corpus cases and return {name: path}.

    Separate from ``seed_all_golden_fixtures`` so the CCR-ON cases can be
    recorded with ccr_inject_tool/ccr_inject_system_instructions enabled
    without touching the 18-case baseline corpus.

    CCR fixtures are written to ``_CCR_FIXTURES_ROOT`` (a sibling of
    ``_FIXTURES_ROOT``) so that ``load_all_golden_fixtures`` (which scans
    ``_FIXTURES_ROOT``) never picks them up and passes them to the non-CCR
    parity test, which would fail since the non-CCR engine does not inject tools.
    """
    ccr_root = root or _CCR_FIXTURES_ROOT
    results: dict[str, Path] = {}
    for spec in _CCR_CORPUS:
        path = record_golden_fixture(spec, root=ccr_root, overwrite=overwrite)
        results[spec.name] = path
    return results


def load_ccr_golden_fixtures(root: Path | None = None) -> list[GoldenFixture]:
    """Load all CCR golden fixtures from ``_CCR_FIXTURES_ROOT``."""
    fixture_dir = root or _CCR_FIXTURES_ROOT
    paths = sorted(fixture_dir.glob("*.json"))
    return [load_golden_fixture(p) for p in paths]


__all__ = [
    "GoldenCaseSpec",
    "GoldenFixture",
    "_CORPUS",
    "_CCR_CORPUS",
    "_CCR_TEST_HASH",
    "_CCR_MARKER",
    "_FIXTURES_ROOT",
    "_CCR_FIXTURES_ROOT",
    "DEFERRED_CASES",
    "load_all_golden_fixtures",
    "load_ccr_golden_fixtures",
    "load_golden_fixture",
    "record_golden_fixture",
    "replay_golden_fixture",
    "seed_all_golden_fixtures",
    "seed_ccr_golden_fixtures",
]
