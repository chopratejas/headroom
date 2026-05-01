"""Unit 4: bounded pre-upstream concurrency for Anthropic replay storms.

Verifies that ``HeadroomProxy`` gates the pre-upstream phase of
``handle_anthropic_messages`` with a semaphore, so cold-start replay
storms cannot starve ``/livez`` or new Codex WS opens.

Covers:
- happy path (single request, no contention)
- N+1 contention (only the (N+1)th waiter records ``pre_upstream_wait`` > 0)
- strict serialization under concurrency=1
- unbounded mode (``anthropic_pre_upstream_concurrency=0`` -> no semaphore)
- acquire timeout fails fast with ``503`` + ``Retry-After``
- memory-context timeout fails open without leaking the semaphore
- exception-safety (semaphore released when the critical section raises)
- ``/livez`` unaffected under Anthropic backpressure
- compression is not bypassed (the Unit 4 gate is additive, not a shortcut)
- CLI flag ``--anthropic-pre-upstream-concurrency`` wires into ``ProxyConfig``
- env var ``HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY`` with flag override
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest
from click.testing import CliRunner
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from headroom.cli.proxy import proxy as proxy_cli
from headroom.pipeline import PipelineStage
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.helpers import MAX_MESSAGE_ARRAY_LENGTH, MAX_REQUEST_BODY_SIZE
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import HeadroomProxy, create_app

# --------------------------------------------------------------------------- #
# Dummy handler that gives tests control over the ``_retry_request`` duration #
# so we can simulate long pre-upstream work (semaphore contention).           #
# --------------------------------------------------------------------------- #


class _DummyTokenizer:
    def count(self, messages) -> int:  # noqa: D401 - stub
        return 1

    def count_messages(self, messages) -> int:  # noqa: D401 - stub
        return 1

    def count_tokens(self, text) -> int:  # noqa: D401 - stub
        return 1


class _DummyMetrics:
    def __init__(self) -> None:
        self.stage_timings: list[tuple[str, dict]] = []

    async def record_request(self, **kwargs):
        return None

    async def record_stage_timings(self, path: str, timings: dict) -> None:
        self.stage_timings.append((path, timings))

    async def record_rate_limited(self, **kwargs) -> None:
        return None

    async def record_failed(self, **kwargs) -> None:
        return None


class _ResponseStub:
    def __init__(self) -> None:
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self._text = json.dumps(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-3-5-sonnet-latest",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    @property
    def text(self) -> str:
        return self._text

    @property
    def content(self) -> bytes:
        return self._text.encode("utf-8")

    def json(self) -> dict:
        return json.loads(self._text)


class _DummyAnthropicHandler(AnthropicHandlerMixin):
    """Minimal handler used across tests; allows controlling upstream delay."""

    ANTHROPIC_API_URL = "https://api.anthropic.com"

    def _extract_anthropic_cache_ttl_metrics(self, usage):  # noqa: D401
        return (0, 0)

    def __init__(
        self,
        *,
        anthropic_pre_upstream_sem: asyncio.Semaphore | None = None,
        upstream_delay_s: float = 0.0,
        raise_during_critical: bool = False,
    ) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = ProxyConfig(
            optimize=False,
            image_optimize=False,
            retry_max_attempts=1,
            retry_base_delay_ms=1,
            retry_max_delay_ms=1,
            connect_timeout_seconds=10,
            mode="token",
            cache_enabled=False,
            rate_limit_enabled=False,
            fallback_enabled=False,
            fallback_provider=None,
            prefix_freeze_enabled=False,
            memory_enabled=False,
        )
        self.usage_reporter = None
        self.anthropic_provider = SimpleNamespace(get_context_limit=lambda model: 200_000)
        self.anthropic_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.cache = None
        self.security = None
        self.ccr_context_tracker = None
        self.ccr_injector = None
        self.ccr_response_handler = None
        self.ccr_feedback = None
        self.ccr_batch_processor = None
        self.ccr_mcp_server = None
        self.traffic_learner = None
        self.tool_injector = None
        self.read_lifecycle_manager = None
        self.logger = SimpleNamespace(log=lambda *a, **k: None)
        self.request_logger = self.logger
        self.usage_observer = None
        self.image_compressor = None
        self.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-1",
            get_or_create=lambda *a, **k: SimpleNamespace(
                _cached_token_count=0,
                get_frozen_message_count=lambda: 0,
                get_last_original_messages=lambda: [],
                get_last_forwarded_messages=lambda: [],
                update_from_response=lambda *a, **k: None,
                record_request=lambda *a, **k: None,
            ),
        )
        # Unit 4: the only field this test cares about.
        self.anthropic_pre_upstream_sem = anthropic_pre_upstream_sem
        self.anthropic_pre_upstream_concurrency = (
            0 if anthropic_pre_upstream_sem is None else anthropic_pre_upstream_sem._value
        )
        self._upstream_delay_s = upstream_delay_s
        self._raise_during_critical = raise_during_critical
        self.upstream_enter_times: list[float] = []
        self.upstream_exit_times: list[float] = []

    async def _next_request_id(self) -> str:
        # Unique IDs so log assertions remain disambiguated under parallelism.
        return f"req-{id(object()):x}"

    def _extract_tags(self, headers):
        return {}

    async def _retry_request(self, method: str, url: str, headers: dict, body: dict):
        if self._raise_during_critical:
            raise RuntimeError("synthetic pre-upstream failure")
        enter = time.perf_counter()
        self.upstream_enter_times.append(enter)
        if self._upstream_delay_s > 0:
            await asyncio.sleep(self._upstream_delay_s)
        self.upstream_exit_times.append(time.perf_counter())
        return _ResponseStub()

    def _get_compression_cache(self, session_id):
        return SimpleNamespace(
            apply_cached=lambda m: m,
            compute_frozen_count=lambda m: 0,
            mark_stable_from_messages=lambda *a, **k: None,
            should_defer_compression=lambda h: False,
            mark_stable=lambda h: None,
            content_hash=lambda c: "h",
            update_from_result=lambda *a, **k: None,
            _cache={},
            _stable_hashes=set(),
        )


def _build_request(body: dict, headers: dict[str, str]) -> Request:
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def stage_log_capture():
    target = logging.getLogger("headroom.proxy")
    handler = _CapturingHandler()
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)


def _parse_all_stage_logs(handler: _CapturingHandler) -> list[dict]:
    payloads: list[dict] = []
    for record in handler.records:
        msg = record.getMessage()
        if "STAGE_TIMINGS" in msg:
            payload_start = msg.index("STAGE_TIMINGS ") + len("STAGE_TIMINGS ")
            payloads.append(json.loads(msg[payload_start:]))
    return payloads


def _tokenizer_patch():
    import headroom.tokenizers as _tk

    orig_get = _tk.get_tokenizer

    class _Ctx:
        def __enter__(self):
            _tk.get_tokenizer = lambda model: _DummyTokenizer()
            return self

        def __exit__(self, *exc):
            _tk.get_tokenizer = orig_get

    return _Ctx()


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #


def test_happy_path_single_request_negligible_wait(stage_log_capture):
    sem = asyncio.Semaphore(2)
    handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": "hello"}],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )

    with _tokenizer_patch():
        anyio.run(handler.handle_anthropic_messages, request)

    payloads = _parse_all_stage_logs(stage_log_capture)
    assert len(payloads) == 1
    stages = payloads[0]["stages"]
    assert "pre_upstream_wait" in stages
    # Single request -> no contention, wait ms must be tiny.
    assert stages["pre_upstream_wait"] is not None
    assert stages["pre_upstream_wait"] < 25.0, stages
    # Sanity: semaphore was released cleanly.
    assert sem._value == 2


# --------------------------------------------------------------------------- #
# N+1 contention: with concurrency=2 and 3 concurrent requests,               #
# exactly one of them must observe a non-trivial ``pre_upstream_wait``.       #
# --------------------------------------------------------------------------- #


def test_n_plus_one_contention_only_waiter_has_nonzero_wait(stage_log_capture):
    async def _run() -> None:
        sem = asyncio.Semaphore(2)
        # Each request hogs the semaphore for ~150 ms. With concurrency=2,
        # 3 concurrent requests mean exactly one waits ~150 ms.
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem, upstream_delay_s=0.15)
        reqs = [
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": f"hello {i}"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
            for i in range(3)
        ]
        await asyncio.gather(*(handler.handle_anthropic_messages(r) for r in reqs))
        assert sem._value == 2  # semaphore fully released

    with _tokenizer_patch():
        anyio.run(_run)

    payloads = _parse_all_stage_logs(stage_log_capture)
    assert len(payloads) == 3
    waits = sorted(p["stages"]["pre_upstream_wait"] for p in payloads)
    # Exactly one request must have waited noticeably; the first two should
    # be near zero (they acquired the sem immediately).
    assert waits[0] < 25.0, waits
    assert waits[1] < 25.0, waits
    # The waiter should have waited roughly the upstream-delay budget.
    assert waits[2] > 75.0, waits


# --------------------------------------------------------------------------- #
# Serialization: concurrency=1 => strict ordering of upstream enter timestamps #
# --------------------------------------------------------------------------- #


def test_concurrency_one_serializes_requests():
    async def _run() -> float:
        sem = asyncio.Semaphore(1)
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem, upstream_delay_s=0.10)
        reqs = [
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": f"msg {i}"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
            for i in range(2)
        ]
        start = time.perf_counter()
        await asyncio.gather(*(handler.handle_anthropic_messages(r) for r in reqs))
        elapsed = time.perf_counter() - start
        # Strict ordering: second request enters upstream only AFTER the first exits.
        assert len(handler.upstream_enter_times) == 2
        assert handler.upstream_enter_times[1] >= handler.upstream_exit_times[0] - 1e-6, (
            handler.upstream_enter_times,
            handler.upstream_exit_times,
        )
        return elapsed

    with _tokenizer_patch():
        elapsed = anyio.run(_run)
    # Two back-to-back 100 ms upstream calls under serialization: must take
    # at least ~2 * 100 ms. (Give a little slack for scheduler jitter.)
    assert elapsed >= 0.18, elapsed


# --------------------------------------------------------------------------- #
# Unbounded mode: ``anthropic_pre_upstream_concurrency=0`` disables the sem.   #
# --------------------------------------------------------------------------- #


def test_unbounded_mode_no_semaphore_instance():
    config = ProxyConfig(anthropic_pre_upstream_concurrency=0)
    proxy = HeadroomProxy(config)
    assert proxy.anthropic_pre_upstream_sem is None
    assert proxy.anthropic_pre_upstream_concurrency == 0


def test_unbounded_mode_requests_run_concurrently():
    """With concurrency=0 (sem disabled), two slow requests overlap."""

    async def _run() -> float:
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=None, upstream_delay_s=0.10)
        reqs = [
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": f"msg {i}"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
            for i in range(2)
        ]
        start = time.perf_counter()
        await asyncio.gather(*(handler.handle_anthropic_messages(r) for r in reqs))
        return time.perf_counter() - start

    with _tokenizer_patch():
        elapsed = anyio.run(_run)
    # Unbounded -> both sleeps run in parallel. Total should be ~0.10 s,
    # nowhere near 0.20 s.
    assert elapsed < 0.18, elapsed


# --------------------------------------------------------------------------- #
# Exception releases the semaphore.                                            #
# --------------------------------------------------------------------------- #


def test_exception_inside_critical_section_releases_semaphore():
    async def _run() -> None:
        sem = asyncio.Semaphore(2)
        baseline = sem._value
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem, raise_during_critical=True)
        # Drive several cycles to ensure we don't leak on any path.
        for i in range(5):
            req = _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": f"msg {i}"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
            # The handler catches upstream RuntimeError internally and
            # returns a 5xx JSONResponse; this is the expected behaviour.
            await handler.handle_anthropic_messages(req)
            # After each cycle the semaphore must be fully restored.
            assert sem._value == baseline, (i, sem._value, baseline)

    with _tokenizer_patch():
        anyio.run(_run)


def test_acquire_timeout_returns_503_with_retry_after(stage_log_capture):
    async def _run() -> None:
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
        handler.config.anthropic_pre_upstream_acquire_timeout_seconds = 0.01
        req = _build_request(
            {
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
            {"authorization": "Bearer sk-ant-api-test"},
        )
        try:
            response = await handler.handle_anthropic_messages(req)
            assert response.status_code == 503
            assert response.headers["retry-after"] == "1"
            body = json.loads(response.body)
            assert body["error"]["type"] == "service_unavailable"
            assert sem._value == 0
        finally:
            sem.release()
        assert sem._value == 1

    with _tokenizer_patch():
        anyio.run(_run)

    payloads = _parse_all_stage_logs(stage_log_capture)
    assert len(payloads) == 1
    assert payloads[0]["stages"]["pre_upstream_wait"] >= 0.0


def test_memory_context_timeout_fails_open_and_releases_semaphore():
    class _MemoryHandler:
        def __init__(self) -> None:
            self.config = SimpleNamespace(inject_context=True, inject_tools=False)
            self.initialized = False
            self.backend = None

        async def search_and_format_context(self, _user_id, _messages):
            await asyncio.sleep(5.0)
            return "should-timeout"

        def inject_tools(self, tools, _provider):
            return tools, False

        def get_beta_headers(self) -> dict[str, str]:
            return {}

        def has_memory_tool_calls(self, _response, _provider) -> bool:
            return False

        async def handle_memory_tool_calls(self, _response, _user_id, _provider):
            return []

    async def _run() -> None:
        sem = asyncio.Semaphore(1)
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
        handler.memory_handler = _MemoryHandler()
        handler.config.anthropic_pre_upstream_memory_context_timeout_seconds = 0.01
        req = _build_request(
            {
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
            {
                "authorization": "Bearer sk-ant-api-test",
                "x-headroom-user-id": "user-1",
            },
        )
        response = await handler.handle_anthropic_messages(req)
        assert response.status_code == 200
        assert sem._value == 1

    with _tokenizer_patch():
        anyio.run(_run)


def test_request_guardrails_reject_oversized_body_and_message_array():
    async def _run() -> None:
        handler = _DummyAnthropicHandler()

        oversized = _build_request(
            {
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
            {
                "authorization": "Bearer sk-ant-api-test",
                "content-length": str(MAX_REQUEST_BODY_SIZE + 1),
            },
        )
        response = await handler.handle_anthropic_messages(oversized)
        assert response.status_code == 413

        too_many_messages = _build_request(
            {
                "model": "claude-3-5-sonnet-latest",
                "messages": [
                    {"role": "user", "content": f"msg-{idx}"}
                    for idx in range(MAX_MESSAGE_ARRAY_LENGTH + 1)
                ],
            },
            {"authorization": "Bearer sk-ant-api-test"},
        )
        response = await handler.handle_anthropic_messages(too_many_messages)
        assert response.status_code == 400
        assert b"Message array too large" in response.body

    with _tokenizer_patch():
        anyio.run(_run)


def test_cache_hit_short_circuits_upstream_and_drops_compression_headers():
    async def _run() -> None:
        handler = _DummyAnthropicHandler()
        upstream_called = False

        async def _unexpected_retry(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal upstream_called
            upstream_called = True
            raise AssertionError("cache hit should bypass upstream")

        class _CachedResponse:
            response_headers = {
                "content-encoding": "gzip",
                "content-length": "99",
                "x-cache": "hit",
            }
            response_body = b'{"cached":true}'

        handler._retry_request = _unexpected_retry
        handler.cache = SimpleNamespace(
            get=lambda messages, model: _return_cached_response(_CachedResponse())
        )

        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
        )
        assert response.status_code == 200
        assert response.headers["x-cache"] == "hit"
        assert "content-encoding" not in response.headers
        assert upstream_called is False

    async def _return_cached_response(response):  # noqa: ANN001, ANN201
        return response

    with _tokenizer_patch():
        anyio.run(_run)


def test_security_block_returns_403_and_generic_security_and_hook_fail_open(monkeypatch):
    async def _run() -> None:
        notified_auth: list[str] = []
        monkeypatch.setattr(
            "headroom.subscription.tracker.get_subscription_tracker",
            lambda: SimpleNamespace(notify_active=lambda auth: notified_auth.append(auth)),
        )

        class _BlockedSecurity:
            def scan_request(self, messages, metadata):  # noqa: ANN001, ANN201
                class _Blocked(Exception):
                    reason = "blocked"

                raise _Blocked("policy denied")

        handler = _DummyAnthropicHandler()
        handler.security = _BlockedSecurity()
        blocked = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                {"authorization": "Bearer oauth-token"},
            )
        )
        assert blocked.status_code == 403
        assert notified_auth == ["Bearer oauth-token"]

        class _GenericSecurity:
            def scan_request(self, messages, metadata):  # noqa: ANN001, ANN201
                raise RuntimeError("scanner unavailable")

        class _Hooks:
            def pre_compress(self, messages, ctx):  # noqa: ANN001, ANN201
                raise RuntimeError("hook failed")

        handler = _DummyAnthropicHandler()
        handler.security = _GenericSecurity()
        handler.config.hooks = _Hooks()
        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
        )
        assert response.status_code == 200

    with _tokenizer_patch():
        anyio.run(_run)


def test_rate_limit_and_budget_denials_release_pre_upstream_semaphore():
    async def _run() -> None:
        sem = asyncio.Semaphore(1)
        rate_limited: list[dict] = []
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)

        class _Metrics(_DummyMetrics):
            async def record_rate_limited(self, **kwargs) -> None:
                rate_limited.append(kwargs)

        class _RateLimiter:
            async def check_request(self, key):  # noqa: ANN001, ANN201
                assert key.startswith("oauth-token:")
                return (False, 1.25)

        handler.metrics = _Metrics()
        handler.rate_limiter = _RateLimiter()
        with pytest.raises(HTTPException) as exc_info:
            await handler.handle_anthropic_messages(
                _build_request(
                    {
                        "model": "claude-3-5-sonnet-latest",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    {"authorization": "Bearer oauth-token"},
                )
            )
        assert exc_info.value.status_code == 429
        assert sem._value == 1
        assert rate_limited == [{"provider": "anthropic"}]

        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
        handler.cost_tracker = SimpleNamespace(check_budget=lambda: (False, 0))
        with pytest.raises(HTTPException) as exc_info:
            await handler.handle_anthropic_messages(
                _build_request(
                    {
                        "model": "claude-3-5-sonnet-latest",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    {"authorization": "Bearer sk-ant-api-test"},
                )
            )
        assert exc_info.value.status_code == 429
        assert sem._value == 1

    with _tokenizer_patch():
        anyio.run(_run)


def test_optimize_pipeline_events_and_presend_adjust_forwarded_request(monkeypatch):
    async def _run() -> None:
        post_events = []

        class _Hooks:
            def pre_compress(self, messages, ctx):  # noqa: ANN001, ANN201
                return messages + [{"role": "assistant", "content": "hooked"}]

            def compute_biases(self, messages, ctx):  # noqa: ANN001, ANN201
                return {"focus": "recent"}

            def post_compress(self, event) -> None:  # noqa: ANN001
                post_events.append(event)

        class _ImageCompressor:
            def __init__(self) -> None:
                self.last_result = SimpleNamespace(
                    technique=SimpleNamespace(value="webp"),
                    savings_percent=50,
                    original_tokens=20,
                    compressed_tokens=10,
                )

            def has_images(self, messages):  # noqa: ANN001, ANN201
                return True

            def compress(self, messages, provider):  # noqa: ANN001, ANN201
                assert provider == "anthropic"
                return messages

        class _PipelineExtensions:
            @staticmethod
            def emit(stage, **kwargs):  # noqa: ANN001, ANN003, ANN201
                messages = kwargs.get("messages")
                tools = kwargs.get("tools")
                headers = kwargs.get("headers")
                if stage == PipelineStage.INPUT_RECEIVED:
                    return SimpleNamespace(messages=messages, tools=tools, headers=None)
                if stage == PipelineStage.INPUT_ROUTED:
                    return SimpleNamespace(
                        messages=messages + [{"role": "assistant", "content": "routed"}],
                        tools=None,
                        headers=None,
                    )
                if stage == PipelineStage.INPUT_COMPRESSED:
                    return SimpleNamespace(
                        messages=messages + [{"role": "assistant", "content": "compressed"}],
                        tools=None,
                        headers=None,
                    )
                if stage == PipelineStage.PRE_SEND:
                    return SimpleNamespace(
                        messages=messages + [{"role": "assistant", "content": "presend"}],
                        tools=[{"name": "zeta"}, {"name": "alpha"}],
                        headers={**headers, "x-pre": "1"},
                    )
                return SimpleNamespace(messages=None, tools=None, headers=None)

        monkeypatch.setattr(
            "headroom.proxy.helpers._get_image_compressor",
            lambda: _ImageCompressor(),
        )
        monkeypatch.setattr("headroom.proxy.modes.is_token_mode", lambda mode: False)
        monkeypatch.setattr("headroom.proxy.modes.is_cache_mode", lambda mode: False)

        handler = _DummyAnthropicHandler()
        handler.config.mode = "adaptive"
        handler.config.optimize = True
        handler.config.image_optimize = True
        handler.config.hooks = _Hooks()
        handler.pipeline_extensions = _PipelineExtensions()

        async def _capture_retry(method: str, url: str, headers: dict, body: dict):
            handler.captured = (method, url, headers, body)
            return _ResponseStub()

        handler._retry_request = _capture_retry
        handler.anthropic_pipeline = SimpleNamespace(
            apply=MagicMock(
                return_value=SimpleNamespace(
                    messages=[{"role": "user", "content": "optimized"}],
                    transforms_applied=["router:text:kompress"],
                    timing={"compression_ms": 1.0},
                    tokens_before=10,
                    tokens_after=4,
                    waste_signals=SimpleNamespace(to_dict=lambda: {"compress": 1}),
                )
            )
        )

        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [{"name": "zeta"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
        )
        assert response.status_code == 200
        assert handler.anthropic_pipeline.apply.called
        message_contents = [message["content"] for message in handler.captured[3]["messages"]]
        assert "optimized" in message_contents
        assert "routed" in message_contents
        assert "compressed" in message_contents
        assert message_contents[-1] == "presend"
        assert [tool["name"] for tool in handler.captured[3]["tools"]] == ["alpha", "zeta"]
        assert handler.captured[2]["x-pre"] == "1"
        assert post_events

    with _tokenizer_patch():
        anyio.run(_run)


def test_memory_and_traffic_learning_update_forwarded_messages_and_headers():
    async def _run() -> None:
        backend = object()

        class _MemoryHandler:
            def __init__(self) -> None:
                self.config = SimpleNamespace(inject_context=True, inject_tools=True)
                self.initialized = True
                self.backend = backend

            async def search_and_format_context(self, user_id, messages):  # noqa: ANN001, ANN201
                assert user_id == "user-1"
                return "memory-context"

            def inject_tools(self, tools, provider):  # noqa: ANN001, ANN201
                assert provider == "anthropic"
                return ((tools or []) + [{"name": "memory_lookup"}], True)

            def get_beta_headers(self) -> dict[str, str]:
                return {"anthropic-beta": "memory-v1"}

            def has_memory_tool_calls(self, _response, _provider) -> bool:
                return False

            async def handle_memory_tool_calls(self, _response, _user_id, _provider):
                return []

        class _TrafficLearner:
            def __init__(self) -> None:
                self._backend = None
                self.tool_results: list[tuple] = []
                self.message_batches: list[list[dict[str, str]]] = []

            def set_backend(self, value) -> None:  # noqa: ANN001
                self._backend = value

            def extract_tool_results_from_messages(self, messages):  # noqa: ANN001, ANN201
                return [
                    {
                        "tool_name": "search",
                        "input": {"q": "hello"},
                        "output": "ok",
                        "is_error": False,
                    }
                ]

            async def on_tool_result(self, **kwargs) -> None:
                self.tool_results.append(
                    (
                        kwargs["tool_name"],
                        kwargs["tool_input"],
                        kwargs["tool_output"],
                        kwargs["is_error"],
                    )
                )

            async def on_messages(self, messages) -> None:  # noqa: ANN001
                self.message_batches.append(messages)

        prefix_tracker = SimpleNamespace(
            _cached_token_count=0,
            get_frozen_message_count=lambda: 1,
            get_last_original_messages=lambda: [],
            get_last_forwarded_messages=lambda: [],
            update_from_response=lambda *a, **k: None,
            record_request=lambda *a, **k: None,
        )

        handler = _DummyAnthropicHandler()
        handler.memory_handler = _MemoryHandler()
        handler.traffic_learner = _TrafficLearner()
        handler.pipeline_extensions = SimpleNamespace(
            emit=lambda *a, **k: SimpleNamespace(messages=None, tools=None, headers=None)
        )

        async def _capture_retry(method: str, url: str, headers: dict, body: dict):
            handler.captured = (method, url, headers, body)
            return _ResponseStub()

        handler._retry_request = _capture_retry
        handler.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-1",
            get_or_create=lambda *a, **k: prefix_tracker,
        )

        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [
                        {"role": "system", "content": "keep"},
                        {"role": "user", "content": "hello"},
                    ],
                },
                {
                    "authorization": "Bearer sk-ant-api-test",
                    "x-headroom-user-id": "user-1",
                    "anthropic-beta": "existing",
                },
            )
        )
        assert response.status_code == 200
        assert handler.captured[3]["messages"][-1]["content"] == "hello\n\nmemory-context"
        assert [tool["name"] for tool in handler.captured[3]["tools"]] == ["memory_lookup"]
        assert handler.captured[2]["anthropic-beta"] == "existing,memory-v1"
        assert handler.traffic_learner._backend is backend
        assert handler.traffic_learner.tool_results == [("search", {"q": "hello"}, "ok", False)]
        assert handler.traffic_learner.message_batches

    with _tokenizer_patch():
        anyio.run(_run)


def test_ccr_proactive_expansion_appends_context_to_latest_user_turn():
    async def _run() -> None:
        handler = _DummyAnthropicHandler()
        handler._turn_counter = 7
        handler.ccr_context_tracker = SimpleNamespace(
            analyze_query=lambda query, turn: (
                ["relevant"] if query == "hello" and turn == 7 else []
            ),
            execute_expansions=lambda recommendations: (
                [{"kind": "memory"}] if recommendations else []
            ),
            format_expansions_for_context=lambda expansions: "expanded-context",
        )
        handler.config.ccr_proactive_expansion = True

        async def _capture_retry(method: str, url: str, headers: dict, body: dict):
            handler.captured = (method, url, headers, body)
            return _ResponseStub()

        handler._retry_request = _capture_retry

        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
        )
        assert response.status_code == 200
        assert handler.captured[3]["messages"] == [
            {"role": "user", "content": "hello\n\nexpanded-context"}
        ]

    with _tokenizer_patch():
        anyio.run(_run)


def test_traffic_learner_fail_open_and_memory_system_context_injection():
    async def _run() -> None:
        handler = _DummyAnthropicHandler()
        handler.traffic_learner = SimpleNamespace(
            _backend=None,
            extract_tool_results_from_messages=lambda messages: (_ for _ in ()).throw(
                RuntimeError("traffic failed")
            ),
        )
        handler._inject_system_context = lambda messages, context, body: [
            {"role": "system", "content": context},
            *messages,
        ]
        handler.memory_handler = SimpleNamespace(
            config=SimpleNamespace(inject_context=True, inject_tools=False),
            initialized=False,
            backend=None,
            search_and_format_context=lambda user_id, messages: _return_memory_context(
                "system-memory"
            ),
            has_memory_tool_calls=lambda response, provider: False,
            handle_memory_tool_calls=lambda response, user_id, provider: _return_tool_calls([]),
        )
        handler.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-1",
            get_or_create=lambda *a, **k: SimpleNamespace(
                _cached_token_count=0,
                get_frozen_message_count=lambda: 0,
                get_last_original_messages=lambda: [],
                get_last_forwarded_messages=lambda: [],
                update_from_response=lambda *a, **k: None,
                record_request=lambda *a, **k: None,
            ),
        )

        async def _capture_retry(method: str, url: str, headers: dict, body: dict):
            handler.captured = (method, url, headers, body)
            return _ResponseStub()

        handler._retry_request = _capture_retry

        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                {
                    "authorization": "Bearer sk-ant-api-test",
                    "x-headroom-user-id": "user-1",
                },
            )
        )
        assert response.status_code == 200
        assert handler.captured[3]["messages"][0] == {
            "role": "system",
            "content": "system-memory",
        }

    async def _return_memory_context(context):  # noqa: ANN001, ANN201
        return context

    async def _return_tool_calls(result):  # noqa: ANN001, ANN201
        return result

    with _tokenizer_patch():
        anyio.run(_run)


def test_bedrock_backend_success_and_error_paths():
    async def _run_success() -> None:
        emitted: list[tuple[PipelineStage, dict]] = []
        logged: list[object] = []
        metric_calls: list[dict[str, object]] = []
        cost_calls: list[tuple] = []

        async def _send_message(body, headers):  # noqa: ANN001, ANN201
            return SimpleNamespace(
                status_code=200,
                body={
                    "content": [{"type": "text", "text": "bedrock ok"}],
                    "usage": {"output_tokens": 7},
                },
                error=False,
            )

        handler = _DummyAnthropicHandler()
        handler.anthropic_backend = SimpleNamespace(name="bedrock", send_message=_send_message)
        handler.pipeline_extensions = SimpleNamespace(
            emit=lambda stage, **kwargs: (
                emitted.append((stage, kwargs))
                or SimpleNamespace(messages=None, tools=None, headers=None)
            )
        )
        handler.metrics.record_request = lambda **kwargs: (
            metric_calls.append(kwargs) or _return_none()
        )
        handler.cost_tracker = SimpleNamespace(
            check_budget=lambda: (True, None),
            record_tokens=lambda *args: cost_calls.append(args),
        )
        handler.logger = SimpleNamespace(log=lambda entry: logged.append(entry))
        handler.request_logger = handler.logger

        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
        )
        assert response.status_code == 200
        assert (
            response.body
            == b'{"content":[{"type":"text","text":"bedrock ok"}],"usage":{"output_tokens":7}}'
        )
        assert metric_calls[-1]["provider"] == "bedrock"
        assert metric_calls[-1]["output_tokens"] == 7
        assert cost_calls[-1] == ("claude-3-5-sonnet-latest", 0, 1)
        assert logged
        assert any(stage == PipelineStage.POST_SEND for stage, _ in emitted)
        assert any(stage == PipelineStage.RESPONSE_RECEIVED for stage, _ in emitted)

    async def _run_error() -> None:
        async def _boom(body, headers):  # noqa: ANN001, ANN201
            raise RuntimeError("bedrock boom")

        handler = _DummyAnthropicHandler()
        handler.anthropic_backend = SimpleNamespace(name="bedrock", send_message=_boom)
        handler.pipeline_extensions = SimpleNamespace(
            emit=lambda *a, **k: SimpleNamespace(messages=None, tools=None, headers=None)
        )

        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
        )
        assert response.status_code == 500
        assert "bedrock boom" in response.body.decode("utf-8")

    async def _return_none():  # noqa: ANN001, ANN201
        return None

    with _tokenizer_patch():
        anyio.run(_run_success)
        anyio.run(_run_error)


def test_direct_api_error_dump_and_redacted_headers(monkeypatch, tmp_path):
    async def _run() -> None:
        cache_busts: list[int] = []
        cache_sets: list[tuple] = []
        metric_calls: list[dict[str, object]] = []

        class _ErrorResponse:
            def __init__(self) -> None:
                self.status_code = 400
                self.headers = {
                    "content-type": "application/json",
                    "x-api-key": "upstream-secret",
                    "authorization": "Bearer upstream-secret",
                }
                self._payload = {
                    "error": {"message": "bad upstream request", "type": "invalid_request_error"},
                    "usage": {"input_tokens": 11, "output_tokens": 0},
                }

            @property
            def text(self) -> str:
                return json.dumps(self._payload)

            @property
            def content(self) -> bytes:
                return self.text.encode("utf-8")

            def json(self) -> dict[str, object]:
                return self._payload

        prefix_tracker = SimpleNamespace(
            _cached_token_count=25,
            get_frozen_message_count=lambda: 0,
            get_last_original_messages=lambda: [],
            get_last_forwarded_messages=lambda: [],
            update_from_response=lambda **kwargs: None,
            record_request=lambda *a, **k: None,
        )

        handler = _DummyAnthropicHandler()
        handler.metrics.record_cache_bust = lambda bust_tokens: (
            cache_busts.append(bust_tokens) or _return_none()
        )
        handler.metrics.record_request = lambda **kwargs: (
            metric_calls.append(kwargs) or _return_none()
        )
        handler.cache = SimpleNamespace(
            get=lambda messages, model: _return_none(),
            set=lambda *args, **kwargs: cache_sets.append((args, kwargs)) or _return_none(),
        )
        handler.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-1",
            get_or_create=lambda *a, **k: prefix_tracker,
        )
        handler._retry_request = lambda method, url, headers, body: _return_response(
            _ErrorResponse()
        )
        handler.pipeline_extensions = SimpleNamespace(
            emit=lambda *a, **k: SimpleNamespace(messages=None, tools=None, headers=None)
        )
        monkeypatch.setattr("headroom.paths.debug_400_dir", lambda: tmp_path)

        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                {
                    "authorization": "Bearer sk-ant-api-test-123456789",
                    "x-api-key": "sk-ant-api-test-123456789",
                },
            )
        )
        assert response.status_code == 400
        debug_files = list(tmp_path.glob("*.json"))
        assert len(debug_files) == 1
        payload = json.loads(debug_files[0].read_text())
        assert payload["status_code"] == 400
        assert payload["headers"]["authorization"].endswith("...")
        assert payload["headers"]["x-api-key"].endswith("...")
        assert payload["compression"]["original_tokens"] == 1
        assert metric_calls[-1]["provider"] == "anthropic"
        assert cache_busts == []
        assert cache_sets == []

    async def _return_none():  # noqa: ANN001, ANN201
        return None

    async def _return_response(value):  # noqa: ANN001, ANN201
        return value

    with _tokenizer_patch():
        anyio.run(_run)


def test_direct_api_success_handles_ccr_memory_continuation_and_cache_bust(monkeypatch):
    async def _run() -> None:
        cache_busts: list[int] = []
        cache_sets: list[tuple] = []
        metric_calls: list[dict[str, object]] = []
        cost_calls: list[tuple] = []
        sub_updates: list[dict[str, int]] = []
        logger_entries: list[object] = []
        continuation_bodies: list[dict] = []
        http_client_calls: list[dict[str, object]] = []
        prefix_updates: list[dict[str, object]] = []

        class _JsonResponse:
            def __init__(
                self,
                payload: dict,
                *,
                status_code: int = 200,
                headers: dict[str, str] | None = None,
            ):
                self.status_code = status_code
                self.headers = headers or {"content-type": "application/json"}
                self._payload = payload

            @property
            def text(self) -> str:
                return json.dumps(self._payload)

            @property
            def content(self) -> bytes:
                return self.text.encode("utf-8")

            def json(self) -> dict:
                return self._payload

        initial_response = _JsonResponse(
            {
                "content": [
                    {"type": "tool_use", "id": "call1", "name": "headroom_retrieve", "input": {}}
                ],
                "usage": {
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "input_tokens": 10,
                },
            }
        )
        continuation_response = _JsonResponse(
            {
                "content": [{"type": "text", "text": "final answer"}],
                "usage": {
                    "output_tokens": 7,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 1,
                    "input_tokens": 8,
                },
            }
        )

        prefix_tracker = SimpleNamespace(
            _cached_token_count=20,
            get_frozen_message_count=lambda: 0,
            get_last_original_messages=lambda: [],
            get_last_forwarded_messages=lambda: [],
            update_from_response=lambda **kwargs: prefix_updates.append(kwargs),
            record_request=lambda *a, **k: None,
        )

        handler = _DummyAnthropicHandler()
        handler.config.optimize = True
        handler.anthropic_pipeline = SimpleNamespace(
            apply=MagicMock(
                return_value=SimpleNamespace(
                    messages=[{"role": "user", "content": "optimized"}],
                    transforms_applied=["router:text:kompress"],
                    timing={"compression_ms": 1.0},
                    tokens_before=10,
                    tokens_after=4,
                    waste_signals=SimpleNamespace(to_dict=lambda: {"compress": 1}),
                )
            )
        )
        handler.metrics.record_cache_bust = lambda bust_tokens: (
            cache_busts.append(bust_tokens) or _return_none()
        )
        handler.metrics.record_request = lambda **kwargs: (
            metric_calls.append(kwargs) or _return_none()
        )
        handler.cache = SimpleNamespace(
            get=lambda messages, model: _return_none(),
            set=lambda *args, **kwargs: cache_sets.append((args, kwargs)) or _return_none(),
        )
        handler.cost_tracker = SimpleNamespace(
            check_budget=lambda: (True, None),
            record_tokens=lambda *args, **kwargs: cost_calls.append((args, kwargs)),
        )
        handler.logger = SimpleNamespace(log=lambda entry: logger_entries.append(entry))
        handler.request_logger = handler.logger
        handler.pipeline_extensions = SimpleNamespace(
            emit=lambda *a, **k: SimpleNamespace(messages=None, tools=None, headers=None)
        )
        handler.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-1",
            get_or_create=lambda *a, **k: prefix_tracker,
        )
        handler.ccr_response_handler = SimpleNamespace(
            has_ccr_tool_calls=lambda response, provider: True,
            handle_response=_handle_ccr,
        )
        handler.memory_handler = SimpleNamespace(
            config=SimpleNamespace(inject_context=False, inject_tools=False),
            has_memory_tool_calls=lambda response, provider: True,
            handle_memory_tool_calls=lambda response, user_id, provider: _return_tool_results(
                [{"type": "tool_result", "tool_use_id": "call1", "content": "resolved"}]
            ),
        )

        async def _http_post(url, json, headers, timeout):  # noqa: ANN001, ANN201
            http_client_calls.append({"url": url, "json": json, "headers": headers})
            return _JsonResponse({"continued": True}, headers={"content-encoding": "gzip"})

        handler.http_client = SimpleNamespace(post=_http_post)

        async def _capture_retry(method: str, url: str, headers: dict, body: dict):
            continuation_bodies.append(body)
            if len(continuation_bodies) == 1:
                return initial_response
            return continuation_response

        handler._retry_request = _capture_retry

        monkeypatch.setitem(
            sys.modules,
            "headroom.subscription.tracker",
            SimpleNamespace(
                get_subscription_tracker=lambda: SimpleNamespace(
                    notify_active=lambda auth_header: None,
                    update_contribution=lambda **kwargs: sub_updates.append(kwargs),
                )
            ),
        )

        response = await handler.handle_anthropic_messages(
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                {"authorization": "Bearer oauth-token"},
            )
        )
        assert response.status_code == 200
        assert response.body == continuation_response.content
        assert http_client_calls[0]["json"]["messages"] == [
            {"role": "user", "content": "continued"}
        ]
        assert http_client_calls[0]["json"]["tools"] == [{"name": "tool"}]
        assert "content-encoding" not in http_client_calls[0]["headers"]
        memory_continuation = continuation_bodies[-1]["messages"]
        assert memory_continuation[-2]["role"] == "assistant"
        assert memory_continuation[-1]["content"] == [
            {"type": "tool_result", "tool_use_id": "call1", "content": "resolved"}
        ]
        assert metric_calls[-1]["cache_read_tokens"] == 3
        assert metric_calls[-1]["tokens_saved"] == 0
        assert cache_sets and cache_sets[-1][1]["tokens_saved"] == 0
        assert cost_calls[-1][1]["cache_read_tokens"] == 3
        assert sub_updates[-1]["tokens_saved_compression"] == 0
        assert prefix_updates[-1]["cache_read_tokens"] == 3
        assert logger_entries

    async def _handle_ccr(resp_json, optimized_messages, tools, api_call_fn, provider):  # noqa: ANN001, ANN201
        assert provider == "anthropic"
        continuation = await api_call_fn(
            [{"role": "user", "content": "continued"}], [{"name": "tool"}]
        )
        assert continuation == {"continued": True}
        return {
            "content": [{"type": "tool_use", "id": "call1", "name": "memory_lookup", "input": {}}],
            "usage": {
                "output_tokens": 4,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 2,
                "input_tokens": 9,
            },
        }

    async def _return_none():  # noqa: ANN001, ANN201
        return None

    async def _return_tool_results(result):  # noqa: ANN001, ANN201
        return result

    with _tokenizer_patch():
        anyio.run(_run)


# --------------------------------------------------------------------------- #
# /livez stays fast under Anthropic pre-upstream contention.                   #
# --------------------------------------------------------------------------- #


def test_livez_unaffected_under_anthropic_backpressure():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        anthropic_pre_upstream_concurrency=2,
    )
    app = create_app(config)
    assert app.state.proxy.anthropic_pre_upstream_sem is not None

    # Drain the semaphore so any simulated Anthropic request would block.
    proxy = app.state.proxy

    async def _drain_sem() -> None:
        # Acquire both permits — no request can enter the pre-upstream region.
        await proxy.anthropic_pre_upstream_sem.acquire()
        await proxy.anthropic_pre_upstream_sem.acquire()

    # Run an event loop just to drain the semaphore.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_drain_sem())
    finally:
        loop.close()

    latencies: list[float] = []
    with TestClient(app) as client:
        # Warm up: the first requests pay one-time costs (TestClient ASGI
        # lifespan, route resolution, lazy imports the restructured proxy
        # triggers on first-request paths). Three warmups was not enough on
        # Python 3.10 under full-suite load; ten is comfortably past every
        # lazy-init boundary observed in CI traces (the rogue sample landed
        # at measured-index 2, i.e. request #6 overall).
        for _ in range(10):
            client.get("/livez")
        for _ in range(20):
            t0 = time.perf_counter()
            resp = client.get("/livez")
            latencies.append((time.perf_counter() - t0) * 1000.0)
            assert resp.status_code == 200
            assert resp.json()["alive"] is True

    # With only 20 samples `statistics.quantiles(n=100)[98]` collapses to
    # max(latencies), so any single CI hiccup trips the assertion. Drop the
    # one worst outlier and assert on the next-worst — that still fails hard
    # if /livez is genuinely being blocked by the drained semaphore (every
    # sample would cluster near the drained timeout) but tolerates a single
    # GC pause or scheduler jitter in the 20-sample window.
    sorted_latencies = sorted(latencies)
    p95_like = sorted_latencies[-2] if len(sorted_latencies) >= 2 else sorted_latencies[-1]
    assert p95_like < 100.0, (p95_like, latencies)


# --------------------------------------------------------------------------- #
# Compression is NOT bypassed by the gate.                                     #
# --------------------------------------------------------------------------- #


def test_compression_is_not_bypassed_when_gated(stage_log_capture):
    """With ``optimize=True`` the first compression stage must still run."""

    class _Pipeline:
        def __init__(self) -> None:
            self.called = False

        def apply(self, messages, *args, **kwargs):
            self.called = True
            return SimpleNamespace(messages=messages, metadata={"applied_steps": ["first"]})

    sem = asyncio.Semaphore(2)
    handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
    handler.config = ProxyConfig(
        optimize=True,
        image_optimize=False,
        retry_max_attempts=1,
        retry_base_delay_ms=1,
        retry_max_delay_ms=1,
        connect_timeout_seconds=10,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        fallback_enabled=False,
        fallback_provider=None,
        prefix_freeze_enabled=False,
        memory_enabled=False,
        anthropic_pre_upstream_concurrency=2,
    )
    pipeline = _Pipeline()
    handler.anthropic_pipeline = pipeline

    # Large synthetic body to ensure the pipeline triggers.
    big_text = "x" * 50_000
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": big_text}],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )

    with _tokenizer_patch():
        anyio.run(handler.handle_anthropic_messages, request)

    assert pipeline.called, "compression pipeline must still run under backpressure"
    # Semaphore restored after the request.
    assert sem._value == 2


# --------------------------------------------------------------------------- #
# CLI: --anthropic-pre-upstream-concurrency plumbs into ProxyConfig.           #
# --------------------------------------------------------------------------- #


def _run_cli_capture(args: list[str], env: dict | None = None) -> ProxyConfig:
    """Invoke the proxy CLI, intercepting ``run_server`` to capture config.

    We do NOT want the CLI to actually start a server — monkeypatching the
    ``run_server`` entry point (imported lazily inside the click command
    via ``from headroom.proxy.server import ... run_server``) short-
    circuits it and lets us inspect the ``ProxyConfig`` that was built.
    """
    import headroom.proxy.server as server_mod

    captured: dict[str, ProxyConfig] = {}
    orig_run = server_mod.run_server

    def _fake_run(config: ProxyConfig):  # noqa: D401 - stub
        captured["config"] = config
        return 0

    server_mod.run_server = _fake_run
    try:
        runner = CliRunner()
        result = runner.invoke(proxy_cli, args, env=env or {})
    finally:
        server_mod.run_server = orig_run

    assert result.exit_code == 0, (result.output, result.exception)
    assert "config" in captured, "run_server was not called"
    return captured["config"]


def test_cli_flag_sets_pre_upstream_concurrency():
    config = _run_cli_capture(["--anthropic-pre-upstream-concurrency", "3"])
    assert config.anthropic_pre_upstream_concurrency == 3


def test_env_var_sets_pre_upstream_concurrency():
    # Must set in the env passed to the runner (click reads envvar).
    # Also strip the corresponding CLI flag.
    env = {"HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY": "4"}
    # Also make sure we don't pick up a host user env that could override.
    config = _run_cli_capture([], env=env)
    assert config.anthropic_pre_upstream_concurrency == 4


def test_cli_flag_overrides_env_var():
    env = {"HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY": "4"}
    config = _run_cli_capture(["--anthropic-pre-upstream-concurrency", "7"], env=env)
    assert config.anthropic_pre_upstream_concurrency == 7


def test_cli_env_sets_pre_upstream_timeouts():
    env = {
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS": "9.5",
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS": "3.25",
    }
    config = _run_cli_capture([], env=env)
    assert config.anthropic_pre_upstream_acquire_timeout_seconds == pytest.approx(9.5)
    assert config.anthropic_pre_upstream_memory_context_timeout_seconds == pytest.approx(3.25)


def test_cli_flags_override_pre_upstream_timeout_env_vars():
    env = {
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS": "9.5",
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS": "3.25",
    }
    config = _run_cli_capture(
        [
            "--anthropic-pre-upstream-acquire-timeout-seconds",
            "4.5",
            "--anthropic-pre-upstream-memory-context-timeout-seconds",
            "1.5",
        ],
        env=env,
    )
    assert config.anthropic_pre_upstream_acquire_timeout_seconds == pytest.approx(4.5)
    assert config.anthropic_pre_upstream_memory_context_timeout_seconds == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# Sanity: HeadroomProxy auto-computes default when config value is None.       #
# --------------------------------------------------------------------------- #


def test_auto_computed_default_on_this_machine():
    config = ProxyConfig()  # field left at None -> auto-compute.
    proxy = HeadroomProxy(config)
    expected = max(2, min(8, os.cpu_count() or 4))
    assert proxy.anthropic_pre_upstream_concurrency == expected
    assert proxy.anthropic_pre_upstream_sem is not None
    assert proxy.anthropic_pre_upstream_sem._value == expected


# --------------------------------------------------------------------------- #
# Semaphore released on HTTPException / early-exit paths even with an         #
# already-held permit. Explicitly covers the 4 pre-upstream early exits:     #
#   - rate_limiter deny (429)                                                 #
#   - cost_tracker block (429)                                                #
#   - security scan block (403)                                               #
#   - cache hit (200)                                                         #
# Each test holds 1 permit of a Semaphore(2) with a concurrent request,      #
# then verifies the handler restores ``_value`` to the original after        #
# the early return.                                                           #
# --------------------------------------------------------------------------- #


class _RateLimiterDeny:
    async def check_request(self, _rate_key):
        return False, 1.0


class _CostTrackerBlock:
    def check_budget(self):
        return False, 0

    def record_tokens(self, *a, **k):
        return None


class _SecurityBlock:
    class _Err(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.reason = "blocked-by-security"

    def scan_request(self, _messages, _ctx):
        raise self._Err("blocked by security policy")


class _CacheHit:
    class _Entry:
        response_headers: dict = {}
        response_body: bytes = b'{"id":"cached","type":"message","role":"assistant","content":[{"type":"text","text":"hit"}]}'

    def __init__(self) -> None:
        self._entry = self._Entry()

    async def get(self, _messages, _model):
        return self._entry

    async def set(self, *a, **k):
        return None


@pytest.mark.parametrize(
    "scenario",
    ["rate_limiter", "cost_tracker", "security", "cache"],
)
def test_early_exit_paths_release_semaphore_under_contention(scenario):
    """Hold one permit of a Semaphore(1) with a concurrent request, trigger
    the early-exit path, verify the semaphore value is restored.
    """

    async def _run() -> None:
        sem = asyncio.Semaphore(1)
        original_value = sem._value

        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
        if scenario == "rate_limiter":
            handler.rate_limiter = _RateLimiterDeny()
        elif scenario == "cost_tracker":
            handler.cost_tracker = _CostTrackerBlock()
        elif scenario == "security":
            handler.security = _SecurityBlock()
        elif scenario == "cache":
            handler.cache = _CacheHit()

        req = _build_request(
            {
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
            {"authorization": "Bearer sk-ant-api-test"},
        )

        # Drive several iterations to confirm each early-exit call fully
        # releases the semaphore rather than leaking a permit AND that the
        # exception type or response status matches the contract for this
        # scenario. `except Exception: pass` would mask the 62d0a50 regression
        # where HTTPException got swallowed and turned into a 502 JSONResponse.
        from fastapi import HTTPException

        for _ in range(3):
            raised: BaseException | None = None
            result = None
            try:
                result = await handler.handle_anthropic_messages(req)
            except HTTPException as exc:
                raised = exc

            if scenario in ("rate_limiter", "cost_tracker"):
                # These paths MUST surface HTTPException(429) so FastAPI's
                # exception handler emits the proper status + Retry-After.
                assert isinstance(raised, HTTPException), (
                    f"{scenario}: expected HTTPException to propagate, got "
                    f"raised={raised!r} result={result!r}"
                )
                assert raised.status_code == 429, (
                    f"{scenario}: wrong status code — got {raised.status_code}"
                )
            else:
                # security returns a JSONResponse; cache returns a Response.
                assert raised is None, f"{scenario}: unexpected exception {raised!r}"
                assert result is not None
            assert sem._value == original_value, (
                f"{scenario}: semaphore leak got={sem._value}, want={original_value}"
            )

    with _tokenizer_patch():
        anyio.run(_run)
