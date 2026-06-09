"""Characterization golden recorder for Chunk 5 — OpenAI handler.

Mirrors ``engine_request_recorder.py`` (Chunk 4.1) but drives the
CURRENT LEGACY OpenAI handler (``/v1/chat/completions`` and
``/v1/responses``) and writes golden fixtures under
``tests/parity/fixtures/engine_request_golden_openai/``.

The recorded ``outbound_b64`` is the byte-exact parity oracle for the
upcoming OpenAI engine path (next task): once the engine takes over
``/v1/chat/completions`` and ``/v1/responses``, the same parametrized
test (``test_openai_request_golden.py``) must still pass 100%.

Interception point
------------------
Identical to the Anthropic oracle: ``proxy.http_client`` is swapped for
an ``httpx.AsyncClient`` backed by ``_CapturingTransport`` /
``_StreamingCapturingTransport``.  Both non-streaming (``_retry_request``)
and streaming (``_stream_response`` via ``http_client.build_request`` +
``http_client.send``) call through the same transport layer, so the
capture is exhaustive.

Response mocks
--------------
- Chat non-streaming: OpenAI ``chat.completion`` JSON shape (``choices``,
  ``usage.prompt_tokens`` / ``completion_tokens``).
- Chat streaming: OpenAI SSE with ``data: [DONE]`` terminator + a final
  chunk that carries ``usage`` (mimics ``stream_options.include_usage``).
- Responses non-streaming: OpenAI Responses API JSON shape (``output[]``,
  ``usage.input_tokens`` / ``output_tokens``).
- Responses streaming: minimal SSE (same ``data: [DONE]`` pattern).

Determinism
-----------
The following nondeterministic sources are suppressed (same strategy as
the Anthropic oracle):
  * ``session_tracker_store`` — replaced with a controlled ``_FixedTracker``
    so ``PrefixCacheTracker`` state does not leak across cases.
  * ``_get_compression_cache`` — replaced with ``_FreshCompressionCache``
    per case.
  * ``request_id`` / UUIDs — appear only in logs, never in the outbound body.
  * Wall-clock timestamps — telemetry only; never in the outbound body.
  * Nondeterministic features (image, memory, LLMLingua, CCR proactive-
    expansion) — excluded from the byte-exact corpus; flagged explicitly.

The test re-runs each case twice and confirms run1 == run2 (determinism
self-check) before asserting the golden.

Usage
-----
Called from ``tests/parity/test_openai_request_golden.py`` (via
``seed_all_openai_golden_fixtures()``), or directly via helper script.
In CI the fixture files are committed; recording is only re-run when a
case spec changes (``overwrite=True``) or a fixture is missing.
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

_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "engine_request_golden_openai"

# ---------------------------------------------------------------------------
# _CapturingTransport — intercepts http_client calls at the transport level
# ---------------------------------------------------------------------------


class _CapturingTransport(httpx.AsyncBaseTransport):
    """Records the exact ``content=`` bytes delivered to the upstream client.

    Returns a minimal OpenAI ``chat.completion`` JSON response so the
    handler can proceed without error.
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
        # Return a minimal valid OpenAI Chat Completions response so the
        # handler proceeds without error and parser does not raise.
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-parity-golden",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )


class _ResponsesCapturingTransport(httpx.AsyncBaseTransport):
    """Like ``_CapturingTransport`` but returns an OpenAI Responses API shape.

    Used for ``/v1/responses`` cases so the Responses handler's JSON parser
    does not raise on missing ``output`` / ``usage`` fields.
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
        return httpx.Response(
            200,
            json={
                "id": "resp-parity-golden",
                "object": "response",
                "created_at": 1700000000,
                "model": "gpt-4o",
                "output": [
                    {
                        "id": "msg_parity",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        )


class _StreamingCapturingTransport(httpx.AsyncBaseTransport):
    """Returns a minimal OpenAI chat SSE body (with ``data: [DONE]``)."""

    def __init__(self) -> None:
        self.captured_body: bytes | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body

        # OpenAI streaming format: a delta chunk, then a usage chunk, then [DONE].
        # The streaming path reads SSE lines; ``[DONE]`` signals termination.
        sse_body = (
            b'data: {"id":"chatcmpl-parity-stream","object":"chat.completion.chunk",'
            b'"created":1700000000,"model":"gpt-4o","choices":[{"index":0,'
            b'"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-parity-stream","object":"chat.completion.chunk",'
            b'"created":1700000000,"model":"gpt-4o","choices":[{"index":0,'
            b'"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,'
            b'"prompt_tokens_details":{"cached_tokens":0}}}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )


class _ResponsesStreamingCapturingTransport(httpx.AsyncBaseTransport):
    """Minimal SSE body for the Responses API streaming path."""

    def __init__(self) -> None:
        self.captured_body: bytes | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body

        sse_body = (
            b'data: {"type":"response.created","response":{"id":"resp-parity-stream",'
            b'"object":"response","created_at":1700000000,"model":"gpt-4o",'
            b'"status":"in_progress","output":[]}}\n\n'
            b'data: {"type":"response.output_item.added","output_index":0,'
            b'"item":{"id":"msg_p","type":"message","role":"assistant","content":[]}}\n\n'
            b'data: {"type":"response.output_text.delta","output_index":0,'
            b'"content_index":0,"delta":"ok"}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp-parity-stream",'
            b'"object":"response","created_at":1700000000,"model":"gpt-4o",'
            b'"status":"completed","output":[{"id":"msg_p","type":"message",'
            b'"role":"assistant","content":[{"type":"output_text","text":"ok"}]}],'
            b'"usage":{"input_tokens":10,"output_tokens":3,'
            b'"input_tokens_details":{"cached_tokens":0}}}}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )


# ---------------------------------------------------------------------------
# Session-state stubs — deterministic, isolated per case
# ---------------------------------------------------------------------------


class _FixedTracker:
    """Deterministic stand-in for PrefixCacheTracker.

    ``frozen_count`` controls how many prefix messages appear frozen to
    the handler, which affects the cache-aligner's injection site.
    """

    def __init__(self, frozen_count: int = 0) -> None:
        self._frozen_count = frozen_count
        self._cached_token_count = 0

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    def update_from_response(self, **kwargs: Any) -> None:
        pass  # state not needed for golden capture


class _FreshCompressionCache:
    """Minimal stub for CompressionCache — returns messages unmodified.

    Fresh per case so no compression-history state leaks between runs.
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
class OpenAIGoldenCaseSpec:
    """Input specification for one OpenAI golden fixture.

    Fields
    ------
    name:
        Unique identifier; becomes the fixture filename.
    endpoint:
        ``"/v1/chat/completions"`` or ``"/v1/responses"``.
    inbound_headers:
        HTTP headers the client sends.  These control auth-mode
        classification and bypass detection.
    body:
        The JSON-parsed request body (sent as compact JSON).
    proxy_config:
        ``ProxyConfig`` kwargs passed to ``create_app``.
    frozen_count:
        Number of messages the ``_FixedTracker`` reports as frozen.
    streaming:
        Whether to send the request with ``stream=True``.
    notes:
        Human-readable explanation of what this case exercises.
    nondeterministic_flag:
        If True: NOT byte-exact (e.g. ML randomness).  Records for
        reference only; determinism self-check is skipped.
    """

    name: str
    endpoint: str
    inbound_headers: dict[str, str]
    body: dict[str, Any]
    proxy_config: dict[str, Any] = field(default_factory=dict)
    frozen_count: int = 0
    streaming: bool = False
    notes: str = ""
    nondeterministic_flag: bool = False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_KWARGS: dict[str, Any] = {
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
    spec: OpenAIGoldenCaseSpec,
) -> tuple[
    TestClient,
    _CapturingTransport
    | _ResponsesCapturingTransport
    | _StreamingCapturingTransport
    | _ResponsesStreamingCapturingTransport,
]:
    """Build a proxy app wired to a capturing transport for ``spec``."""
    config_kwargs = {**_DEFAULT_CONFIG_KWARGS, **spec.proxy_config}
    config = ProxyConfig(**config_kwargs)
    app = create_app(config)

    is_responses = spec.endpoint == "/v1/responses"

    transport: (
        _CapturingTransport
        | _ResponsesCapturingTransport
        | _StreamingCapturingTransport
        | _ResponsesStreamingCapturingTransport
    )
    if spec.streaming and is_responses:
        transport = _ResponsesStreamingCapturingTransport()
    elif spec.streaming:
        transport = _StreamingCapturingTransport()
    elif is_responses:
        transport = _ResponsesCapturingTransport()
    else:
        transport = _CapturingTransport()

    proxy = app.state.proxy
    proxy.http_client = httpx.AsyncClient(transport=transport)

    # Pin session tracker — deterministic frozen_count, no leakage.
    tracker = _FixedTracker(frozen_count=spec.frozen_count)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
        f"golden-openai-{spec.name}"
    )
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker

    # Pin compression cache — fresh per case, no history leakage.
    proxy._get_compression_cache = lambda session_id: _FreshCompressionCache()

    return TestClient(app), transport


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

_STANDARD_OPENAI_HEADERS = {
    "authorization": "Bearer sk-test-openai-key",
    "content-type": "application/json",
}

_STANDARD_OAUTH_HEADERS = {
    "authorization": "Bearer sk-ant-oat-abc123def456",
    "content-type": "application/json",
}

_STANDARD_SUBSCRIPTION_HEADERS = {
    "authorization": "Bearer eyJ.abc.def",  # JWT shape → OAuth classification
    "user-agent": "claude-code/1.0",
    "content-type": "application/json",
}


def _infer_auth_mode(headers: dict[str, str]) -> str:
    from headroom.proxy.auth_mode import classify_auth_mode

    return classify_auth_mode(headers).value


def record_openai_golden_fixture(
    spec: OpenAIGoldenCaseSpec,
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
        Override fixture output directory.
    overwrite:
        If False (default) and fixture exists, skip and return existing path.

    Returns
    -------
    Path to the written (or existing) fixture file.

    Raises
    ------
    RuntimeError
        If the transport captured no body — the handler exited before
        reaching the upstream send call (validation error, etc.).
    """
    out_dir = root or _FIXTURES_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}.json"

    if out_path.exists() and not overwrite:
        return out_path

    client, transport = _make_app_and_transport(spec)

    # Use insertion-order serialization so inbound bytes match the spec dict
    # exactly.  Stored as inbound_b64 for byte-faithful replay.
    body_bytes = json.dumps(spec.body, separators=(",", ":"), ensure_ascii=False).encode()

    if spec.streaming:
        with client.stream(
            "POST",
            spec.endpoint,
            headers=spec.inbound_headers,
            content=body_bytes,
        ) as resp:
            for _ in resp.iter_bytes():
                pass
        captured_body = transport.captured_body  # type: ignore[union-attr]
    else:
        resp = client.post(
            spec.endpoint,
            headers=spec.inbound_headers,
            content=body_bytes,
        )
        if resp.status_code not in (200, 400):
            raise RuntimeError(
                f"Golden recording for '{spec.name}' got HTTP {resp.status_code}: {resp.text[:400]}"
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
        "endpoint": spec.endpoint,
        "auth_mode": auth_mode,
        "headers": dict(spec.inbound_headers),
        # ``inbound_b64`` is canonical for replay — preserves exact byte order.
        "inbound_b64": base64.b64encode(body_bytes).decode(),
        # ``body`` is stored for human readability only; not used on replay.
        "body": spec.body,
        "proxy_config": spec.proxy_config,
        "frozen_count": spec.frozen_count,
        "streaming": spec.streaming,
        "notes": spec.notes,
        "nondeterministic_flag": spec.nondeterministic_flag,
        "outbound_b64": base64.b64encode(captured_body).decode(),
        "recorded_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }

    out_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenAIGoldenFixture:
    """Parsed representation of one engine_request_golden_openai fixture."""

    name: str
    endpoint: str
    auth_mode: str
    headers: dict[str, str]
    body: dict[str, Any]  # human-readable; not used for replay
    inbound_bytes: bytes  # decoded from inbound_b64; used for exact replay
    proxy_config: dict[str, Any]
    frozen_count: int
    streaming: bool
    notes: str
    nondeterministic_flag: bool
    outbound_bytes: bytes  # decoded from outbound_b64
    recorded_at: str


def load_openai_golden_fixture(path: Path) -> OpenAIGoldenFixture:
    """Parse one fixture.  Raises loudly on malformed input."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in OpenAI golden fixture {path}: {exc}") from exc

    required = {
        "name",
        "endpoint",
        "auth_mode",
        "headers",
        "inbound_b64",
        "outbound_b64",
        "recorded_at",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"OpenAI golden fixture {path} missing required keys: {missing!r}")

    try:
        inbound_bytes = base64.b64decode(data["inbound_b64"])
    except Exception as exc:
        raise ValueError(f"OpenAI golden fixture {path}: bad inbound_b64: {exc}") from exc

    try:
        outbound_bytes = base64.b64decode(data["outbound_b64"])
    except Exception as exc:
        raise ValueError(f"OpenAI golden fixture {path}: bad outbound_b64: {exc}") from exc

    return OpenAIGoldenFixture(
        name=data["name"],
        endpoint=data["endpoint"],
        auth_mode=data["auth_mode"],
        headers=dict(data["headers"]),
        body=dict(data.get("body", {})),
        inbound_bytes=inbound_bytes,
        proxy_config=dict(data.get("proxy_config", {})),
        frozen_count=int(data.get("frozen_count", 0)),
        streaming=bool(data.get("streaming", False)),
        notes=str(data.get("notes", "")),
        nondeterministic_flag=bool(data.get("nondeterministic_flag", False)),
        outbound_bytes=outbound_bytes,
        recorded_at=data["recorded_at"],
    )


def load_all_openai_golden_fixtures(root: Path | None = None) -> list[OpenAIGoldenFixture]:
    """Load all ``*.json`` fixtures from the openai golden directory."""
    fixture_dir = root or _FIXTURES_ROOT
    paths = sorted(fixture_dir.glob("*.json"))
    return [load_openai_golden_fixture(p) for p in paths]


# ---------------------------------------------------------------------------
# Replay helper
# ---------------------------------------------------------------------------


def replay_openai_golden_fixture(fix: OpenAIGoldenFixture) -> bytes:
    """Re-drive the current OpenAI handler for ``fix`` and return outbound bytes.

    Uses ``fix.inbound_bytes`` directly (not ``fix.body``) to preserve the
    exact byte order recorded at fixture creation time.
    """
    spec = OpenAIGoldenCaseSpec(
        name=fix.name,
        endpoint=fix.endpoint,
        inbound_headers=fix.headers,
        body=fix.body,
        proxy_config=fix.proxy_config,
        frozen_count=fix.frozen_count,
        streaming=fix.streaming,
        notes=fix.notes,
        nondeterministic_flag=fix.nondeterministic_flag,
    )
    client, transport = _make_app_and_transport(spec)

    # Use inbound_bytes directly — never re-serialize fix.body.
    body_bytes = fix.inbound_bytes

    if spec.streaming:
        with client.stream(
            "POST",
            spec.endpoint,
            headers=spec.inbound_headers,
            content=body_bytes,
        ) as resp:
            for _ in resp.iter_bytes():
                pass
        captured = transport.captured_body  # type: ignore[union-attr]
    else:
        resp = client.post(
            spec.endpoint,
            headers=spec.inbound_headers,
            content=body_bytes,
        )
        captured = transport.captured_body  # type: ignore[union-attr]

    if captured is None:
        raise RuntimeError(
            f"Replay for OpenAI golden fixture '{fix.name}': transport captured no body. "
            "The handler may have exited before reaching the upstream send call."
        )
    return captured


# ---------------------------------------------------------------------------
# Corpus definition
# ---------------------------------------------------------------------------

# ~1 KB of log output — large enough to trigger ContentRouter → LogCompressor
# when optimize=True / mode=token.
_LARGE_TOOL_LOG_CONTENT = (
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

# Large function_call_output for /v1/responses compression cases.
_LARGE_FCO_CONTENT = (
    '{"status":"ok","results":['
    + ",".join(
        f'{{"id":{i},"text":"Log entry {i}: processed shard result from worker node",'
        f'"level":"INFO","ts":1748477414{i}}}'
        for i in range(30)
    )
    + "]}"
)

_OPENAI_CORPUS: list[OpenAIGoldenCaseSpec] = [
    # ── 1. Chat — passthrough, PAYG, bypass header ──────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_payg_bypass",
        endpoint="/v1/chat/completions",
        inbound_headers={
            **_STANDARD_OPENAI_HEADERS,
            "x-headroom-bypass": "true",
        },
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        },
        proxy_config={"optimize": False},
        notes="PAYG auth; bypass header → handler forwards original bytes verbatim",
    ),
    # ── 2. Chat — passthrough, PAYG, optimize=False ─────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_payg_no_optimize",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        proxy_config={"optimize": False},
        notes="optimize=False → passthrough; canonical serialization of small body",
    ),
    # ── 3. Chat — passthrough, OAuth bearer ─────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_oauth_no_optimize",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OAUTH_HEADERS,
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Summarize this document."}],
        },
        proxy_config={"optimize": False},
        notes="OAuth auth mode (sk-ant-oat-*); optimize=False → passthrough",
    ),
    # ── 4. Chat — passthrough, subscription (JWT bearer + subscription UA) ──
    OpenAIGoldenCaseSpec(
        name="openai_chat_subscription_no_optimize",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_SUBSCRIPTION_HEADERS,
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Write a poem."}],
        },
        proxy_config={"optimize": False},
        notes=(
            "Subscription auth (JWT shape Bearer + claude-code UA); optimize=False → passthrough"
        ),
    ),
    # ── 5. Chat — token mode, single turn, no tools ─────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_token_mode_simple",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Turn 1 question"},
                {"role": "assistant", "content": "Turn 1 answer"},
                {"role": "user", "content": "Turn 2 question"},
            ],
        },
        proxy_config={"optimize": True, "mode": "token"},
        frozen_count=0,
        notes="token mode; frozen=0; small body → passthrough (under compression threshold)",
    ),
    # ── 6. Chat — cache mode, tools sorted ─────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_cache_mode_tools_sorted",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Run the search tool."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "zeta_tool",
                        "description": "z",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "alpha_tool",
                        "description": "a",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mu_tool",
                        "description": "m",
                        "parameters": {"type": "object"},
                    },
                },
            ],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        frozen_count=0,
        notes=(
            "cache mode; tools arrive unsorted (z, a, m); "
            "characterizes whether/how handler sorts tools at the pre-send site"
        ),
    ),
    # ── 7. Chat — no optimize, unsorted tools (passthrough path) ───────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_payg_no_optimize_unsorted_tools",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "zeta_tool",
                        "description": "z",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "alpha_tool",
                        "description": "a",
                        "parameters": {"type": "object"},
                    },
                },
            ],
        },
        proxy_config={"optimize": False},
        frozen_count=0,
        notes=(
            "no-compress path (optimize=False) with unsorted tools; "
            "characterizes tool-sort behavior on the passthrough path"
        ),
    ),
    # ── 8. Chat — token mode, role:tool message large enough to compress ────
    OpenAIGoldenCaseSpec(
        name="openai_chat_token_mode_role_tool_large",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_batch_job",
                            "type": "function",
                            "function": {
                                "name": "run_batch_job",
                                "arguments": '{"job_id":"batch-9821"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": _LARGE_TOOL_LOG_CONTENT,
                    "tool_call_id": "call_batch_job",
                },
            ],
        },
        proxy_config={"optimize": True, "mode": "token"},
        frozen_count=0,
        notes=(
            "token mode; large role:tool message with log output; "
            "exercises ContentRouter → LogCompressor on OpenAI body shape; "
            "deterministic: empty query → BM25 short-circuits"
        ),
    ),
    # ── 9. Chat — cache mode, frozen_count=2 ────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_cache_mode_frozen_prefix",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
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
        notes="cache mode; frozen_count=4; only the last message is live",
    ),
    # ── 10. Chat — streaming, no optimize ──────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_streaming_no_optimize",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "stream": True,
            "messages": [{"role": "user", "content": "Stream me a response."}],
        },
        proxy_config={"optimize": False},
        streaming=True,
        notes=(
            "streaming path; optimize=False; confirms transport interception "
            "captures bytes on the streaming code path"
        ),
    ),
    # ── 11. Chat — streaming, cache mode ────────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_streaming_cache_mode",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
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
    # ── 12. Chat — bypass via x-headroom-mode=passthrough ──────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_passthrough_mode_header",
        endpoint="/v1/chat/completions",
        inbound_headers={
            **_STANDARD_OPENAI_HEADERS,
            "x-headroom-mode": "passthrough",
        },
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "passthrough via mode header"}],
        },
        proxy_config={"optimize": True, "mode": "cache"},
        notes="x-headroom-mode=passthrough triggers bypass; body forwarded verbatim",
    ),
    # ── 13. Chat — unicode content ──────────────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_unicode_bypass",
        endpoint="/v1/chat/completions",
        inbound_headers={
            **_STANDARD_OPENAI_HEADERS,
            "x-headroom-bypass": "true",
        },
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello 🔥 — 世界 — emoji is 🚀"}],
        },
        proxy_config={"optimize": False},
        notes=("unicode in body; bypass; confirms no \\uXXXX escaping in outbound bytes"),
    ),
    # ── 14. Chat — numeric precision ────────────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_numeric_precision",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "temperature": 1.0,
            "top_p": 0.95,
            "messages": [{"role": "user", "content": "hi"}],
        },
        proxy_config={"optimize": False},
        notes=(
            "numeric fields (temperature=1.0, top_p=0.95); confirms canonical "
            "serializer preserves float precision"
        ),
    ),
    # ── 15. Chat — empty messages array ─────────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_chat_empty_messages",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "messages": [],
        },
        proxy_config={"optimize": True, "mode": "token"},
        frozen_count=0,
        notes="empty messages array → handler passthrough without compression",
    ),
    # ── 16. Chat — multi-turn with assistant tool_calls (no compress) ───────
    OpenAIGoldenCaseSpec(
        name="openai_chat_assistant_tool_calls_passthrough",
        endpoint="/v1/chat/completions",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Call a tool"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query":"cats"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "Found: cats are great.",
                    "tool_call_id": "call_abc",
                },
            ],
        },
        proxy_config={"optimize": False},
        notes=(
            "assistant with tool_calls + role:tool follow-up; optimize=False; "
            "byte-faithful passthrough of OpenAI chat shape with null content"
        ),
    ),
    # ── 17. Responses — passthrough, PAYG ──────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_responses_payg_passthrough",
        endpoint="/v1/responses",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "input": "Hello from the Responses API",
        },
        proxy_config={"optimize": False},
        notes="Responses API; PAYG; optimize=False → passthrough; string input",
    ),
    # ── 18. Responses — with instructions passthrough ───────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_responses_instructions_passthrough",
        endpoint="/v1/responses",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "instructions": "You are a helpful assistant.",
            "input": "What is the weather?",
        },
        proxy_config={"optimize": False},
        notes=(
            "Responses API; instructions field present (cache hot zone); "
            "optimize=False → passthrough; confirms instructions forwarded verbatim"
        ),
    ),
    # ── 19. Responses — large function_call_output (compression path) ───────
    OpenAIGoldenCaseSpec(
        name="openai_responses_fco_large",
        endpoint="/v1/responses",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_fcout_01",
                    "output": _LARGE_FCO_CONTENT,
                }
            ],
        },
        proxy_config={"optimize": True},
        notes=(
            "Responses API; large function_call_output in input[] array; "
            "exercises _compress_openai_responses_payload_in_executor; "
            "optimize=True; deterministic because ContentRouter routes on content type"
        ),
    ),
    # ── 20. Responses — bypass header ───────────────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_responses_bypass",
        endpoint="/v1/responses",
        inbound_headers={
            **_STANDARD_OPENAI_HEADERS,
            "x-headroom-bypass": "true",
        },
        body={
            "model": "gpt-4o",
            "input": "Bypass me",
        },
        proxy_config={"optimize": True},
        notes=("Responses API; bypass header → compression skipped; bytes forwarded verbatim"),
    ),
    # ── 21. Responses — streaming passthrough ───────────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_responses_streaming_passthrough",
        endpoint="/v1/responses",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "gpt-4o",
            "stream": True,
            "input": "Stream this response",
        },
        proxy_config={"optimize": False},
        streaming=True,
        notes=(
            "Responses API; streaming path; optimize=False; confirms streaming "
            "transport interception works on /v1/responses"
        ),
    ),
    # ── 22. Responses — reasoning items passthrough ─────────────────────────
    OpenAIGoldenCaseSpec(
        name="openai_responses_reasoning_items_passthrough",
        endpoint="/v1/responses",
        inbound_headers=_STANDARD_OPENAI_HEADERS,
        body={
            "model": "o3",
            "input": [
                {"type": "message", "role": "user", "content": "What is 2+2?"},
            ],
            "reasoning": {"effort": "low"},
        },
        proxy_config={"optimize": False},
        notes=(
            "Responses API; reasoning field present; optimize=False → passthrough; "
            "byte-faithful preservation of reasoning items"
        ),
    ),
]

# Cases intentionally excluded from the byte-exact corpus:
#
# CCR proactive-expansion:
#   ccr_proactive_expansion is ML-nondeterministic (ONNX scoring). Deferred.
#
# Image compression:
#   image_optimize=True calls PIL/ONNX; codec nondeterminism. Deferred.
#
# Memory injection:
#   memory_handler.search_and_format_context is async I/O. Deferred.
#
# LLMLingua / IntelligentContext:
#   ML models with float-level nondeterminism. Deferred.
#
# WebSocket paths (/v1/responses WS):
#   WS handler has a completely different transport path (not http_client.post).
#   Deferred — requires a WS-aware capturing transport.

DEFERRED_OPENAI_CASES = [
    "ccr_proactive_expansion",
    "image_compression",
    "memory_injection",
    "llmlingua",
    "intelligent_context",
    "websocket_responses",
]


def seed_all_openai_golden_fixtures(
    *,
    root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Record all corpus cases and return {name: path} for recorded fixtures."""
    results: dict[str, Path] = {}
    for spec in _OPENAI_CORPUS:
        path = record_openai_golden_fixture(spec, root=root, overwrite=overwrite)
        results[spec.name] = path
    return results


__all__ = [
    "OpenAIGoldenCaseSpec",
    "OpenAIGoldenFixture",
    "_FIXTURES_ROOT",
    "_OPENAI_CORPUS",
    "DEFERRED_OPENAI_CASES",
    "load_all_openai_golden_fixtures",
    "load_openai_golden_fixture",
    "record_openai_golden_fixture",
    "replay_openai_golden_fixture",
    "seed_all_openai_golden_fixtures",
]
