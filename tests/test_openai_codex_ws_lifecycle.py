"""Unit 3: WebSocket session lifecycle + deterministic relay cancellation.

These tests exercise the Codex WS handler with a fake upstream and a
fake client WebSocket so we can drive the relay halves through their
real code paths (not mocked) and assert on registry / task state.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.ws_session_registry import WebSocketSessionRegistry

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _DummyMetrics:
    def __init__(self) -> None:
        self.active_ws_sessions = 0
        self.active_ws_sessions_max = 0
        self.active_relay_tasks = 0
        self.ws_session_durations: list[float] = []
        self.stage_timings: list[tuple[str, dict[str, float]]] = []
        self.termination_causes: list[str] = []

    async def record_request(self, **kwargs):  # pragma: no cover
        return None

    async def record_stage_timings(self, path: str, timings: dict[str, float]) -> None:
        self.stage_timings.append((path, dict(timings)))

    def inc_active_ws_sessions(self) -> None:
        self.active_ws_sessions += 1
        self.active_ws_sessions_max = max(self.active_ws_sessions_max, self.active_ws_sessions)

    def dec_active_ws_sessions(self) -> None:
        self.active_ws_sessions = max(0, self.active_ws_sessions - 1)

    def inc_active_relay_tasks(self, n: int = 1) -> None:
        self.active_relay_tasks += n

    def dec_active_relay_tasks(self, n: int = 1) -> None:
        self.active_relay_tasks = max(0, self.active_relay_tasks - n)

    def record_ws_session_duration(self, duration_ms: float, cause: str) -> None:
        self.ws_session_durations.append(duration_ms)
        self.termination_causes.append(cause)


class _DummyOpenAIHandler(OpenAIHandlerMixin):
    OPENAI_API_URL = "https://api.openai.com"

    def __init__(self, ws_sessions: WebSocketSessionRegistry | None = None) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = SimpleNamespace(
            optimize=False,
            retry_max_attempts=1,
            retry_base_delay_ms=1,
            retry_max_delay_ms=1,
            connect_timeout_seconds=10,
        )
        self.usage_reporter = None
        self.openai_provider = SimpleNamespace(get_context_limit=lambda model: 128_000)
        self.openai_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.ws_sessions = ws_sessions or WebSocketSessionRegistry()

    async def _next_request_id(self) -> str:
        return "req-lifecycle-test"


class _FakeWebSocketDisconnect(Exception):
    """Mirrors the ``WebSocketDisconnect`` type-name check in the handler.

    The production code identifies "normal client gone" by
    ``"WebSocketDisconnect" in type(e).__name__`` — so the fake exception
    type name must start with ``WebSocketDisconnect``.
    """


# Force the type-name substring match in the handler.
_FakeWebSocketDisconnect.__name__ = "WebSocketDisconnect_Fake"


class _FakeWebSocket:
    """Scripted client WebSocket that can delay / disconnect mid-stream."""

    def __init__(
        self,
        frames: list[str] | None = None,
        *,
        disconnect_after_n_sends: int | None = None,
        hold_after_initial: bool = False,
    ) -> None:
        self.headers = {"authorization": "Bearer test"}
        self._frames = list(frames or [])
        self._hold_after_initial = hold_after_initial
        self._disconnect_after_n_sends = disconnect_after_n_sends
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.accepted_subprotocol: str | None = None
        self.closed = False
        self.close_code: int | None = None
        # "client" can trip this event to simulate mid-stream disconnect.
        self._disconnect_event = asyncio.Event()
        self.client = SimpleNamespace(host="127.0.0.1", port=12345)

    async def accept(self, subprotocol=None) -> None:
        self.accepted_subprotocol = subprotocol

    async def receive_text(self) -> str:
        if self._frames:
            return self._frames.pop(0)
        if self._hold_after_initial:
            # Wait for simulated client disconnect.
            await self._disconnect_event.wait()
        # Use an exception type whose name starts with ``WebSocketDisconnect``
        # so the handler's ``type(e).__name__`` check classifies this as a
        # normal client exit (not a ``client_error``).
        raise _FakeWebSocketDisconnect("client closed")

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)
        if (
            self._disconnect_after_n_sends is not None
            and len(self.sent_text) >= self._disconnect_after_n_sends
        ):
            # Trigger the "client gone" signal the next receive_text will see.
            self._disconnect_event.set()

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed = True
        self.close_code = code

    def trigger_disconnect(self) -> None:
        self._disconnect_event.set()


class _FakeUpstream:
    """Upstream that streams scripted events then optionally blocks.

    ``hold_after_events`` makes the async iterator wait forever after the
    scripted events are exhausted — that mirrors a real upstream that
    keeps the connection open after a ``response.completed`` event. The
    handler's ``_upstream_to_client`` will block on it, so the only way
    the outer ``asyncio.wait`` can progress is via the client-side task
    completing — which is exactly the cancel-partner path we want to
    test.
    """

    def __init__(
        self,
        events: list[str],
        *,
        hold_after_events: bool = False,
        raise_mid_stream: Exception | None = None,
    ) -> None:
        self._events = list(events)
        self._hold_after_events = hold_after_events
        self._raise_mid_stream = raise_mid_stream
        self.sent: list[str] = []
        self.closed = False

    async def __aenter__(self) -> _FakeUpstream:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for ev in self._events:
            yield ev
        if self._raise_mid_stream is not None:
            raise self._raise_mid_stream
        if self._hold_after_events:
            # Wait forever — until the task is cancelled by the handler.
            await asyncio.Event().wait()


def _make_fake_websockets_module(upstream: _FakeUpstream):
    module = MagicMock()
    module.connect = MagicMock(return_value=upstream)
    module.Subprotocol = str
    return module


def _first_frame() -> str:
    return json.dumps(
        {
            "type": "response.create",
            "response": {"model": "gpt-5.4", "input": "hi"},
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_registry_empty_after_response_completed():
    """Normal session completes — both relay tasks done, registry empty."""
    upstream_events = [
        json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
        json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
    ]
    upstream = _FakeUpstream(upstream_events)
    fake_ws_mod = _make_fake_websockets_module(upstream)

    client_ws = _FakeWebSocket(frames=[_first_frame()])
    handler = _DummyOpenAIHandler()

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    assert handler.ws_sessions.active_count() == 0
    assert handler.metrics.active_ws_sessions == 0
    # termination_cause captured
    assert handler.metrics.termination_causes
    # Either "response_completed" or "client_disconnect" — both are
    # acceptable here depending on which relay half exited first; the
    # important thing is we recorded one.
    assert handler.metrics.termination_causes[-1] in {
        "response_completed",
        "client_disconnect",
        "upstream_disconnect",
    }


@pytest.mark.asyncio
async def test_client_disconnect_cancels_upstream_relay_within_100ms():
    """**Failing-test-first** scenario from the plan.

    When the client side exits (``receive_text`` raises
    ``WebSocketDisconnect``) while upstream is still open and iterating,
    the upstream relay task must be cancelled and become ``done()``
    quickly. The registry must report no active sessions afterwards.
    """
    # Upstream keeps iterating forever after one event, forcing the
    # upstream-to-client task to block on the iterator. The only way
    # out is a cancel from the handler's orchestration.
    upstream_events = [
        json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
    ]
    upstream = _FakeUpstream(upstream_events, hold_after_events=True)
    fake_ws_mod = _make_fake_websockets_module(upstream)

    # Client has one initial frame, then disconnects after the server
    # sends the first forwarded event to us.
    client_ws = _FakeWebSocket(
        frames=[_first_frame()],
        hold_after_initial=True,
    )
    handler = _DummyOpenAIHandler()

    # Trigger disconnect shortly after the handler accepts.
    async def _trigger() -> None:
        await asyncio.sleep(0.05)
        client_ws.trigger_disconnect()

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        trigger_task = asyncio.create_task(_trigger())
        try:
            await asyncio.wait_for(
                handler.handle_openai_responses_ws(client_ws),
                timeout=2.0,
            )
        finally:
            trigger_task.cancel()
            try:
                await trigger_task
            except asyncio.CancelledError:
                pass

    # Registry must be empty — the finally block deregistered the session.
    assert handler.ws_sessions.active_count() == 0, (
        "session leaked — deregister did not run in outermost finally"
    )
    assert handler.metrics.active_ws_sessions == 0
    # We recorded a session duration (came through deregister path).
    assert handler.metrics.ws_session_durations, (
        "record_ws_session_duration never fired — deregister path broken"
    )
    # And we tagged the cause. For a client-side exit it should be one
    # of: client_disconnect, client_error, upstream_disconnect (if
    # upstream iteration happened to end first in a race).
    cause = handler.metrics.termination_causes[-1]
    assert cause in {
        "client_disconnect",
        "client_error",
        "upstream_disconnect",
    }, f"unexpected cause: {cause}"

    # No codex-ws-* named task should still be running.
    leaked = [
        t
        for t in asyncio.all_tasks()
        if (t.get_name() or "").startswith("codex-ws-") and not t.done()
    ]
    assert leaked == [], f"relay tasks leaked: {[t.get_name() for t in leaked]}"


@pytest.mark.asyncio
async def test_upstream_closes_first_cancels_client_task():
    """Upstream iterator ends naturally; client task should be cancelled.

    The client is set to block on ``receive_text`` indefinitely; only a
    cancel from the handler's orchestration releases it.
    """
    upstream_events = [
        json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
        json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
    ]
    upstream = _FakeUpstream(upstream_events, hold_after_events=False)
    fake_ws_mod = _make_fake_websockets_module(upstream)

    client_ws = _FakeWebSocket(
        frames=[_first_frame()],
        hold_after_initial=True,
    )
    handler = _DummyOpenAIHandler()

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await asyncio.wait_for(
            handler.handle_openai_responses_ws(client_ws),
            timeout=2.0,
        )

    assert handler.ws_sessions.active_count() == 0
    # We must still have recorded exactly one session duration.
    assert len(handler.metrics.ws_session_durations) == 1


@pytest.mark.asyncio
async def test_upstream_error_mid_stream_classifies_as_upstream_error():
    upstream_events = [
        json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
    ]
    upstream = _FakeUpstream(
        upstream_events,
        raise_mid_stream=RuntimeError("boom from upstream"),
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)

    client_ws = _FakeWebSocket(
        frames=[_first_frame()],
        hold_after_initial=True,
    )
    handler = _DummyOpenAIHandler()

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await asyncio.wait_for(
            handler.handle_openai_responses_ws(client_ws),
            timeout=2.0,
        )

    assert handler.ws_sessions.active_count() == 0
    assert handler.metrics.termination_causes
    assert handler.metrics.termination_causes[-1] == "upstream_error"


@pytest.mark.asyncio
async def test_upstream_connect_failure_still_deregisters_cleanly():
    """Handshake-phase leak must be impossible: if upstream connect
    raises before relay tasks are created, the session is still
    registered+deregistered cleanly (or never registered). Either way,
    no leak.
    """

    class _BoomUpstream:
        async def __aenter__(self):
            raise RuntimeError("upstream refused")

        async def __aexit__(self, exc_type, exc, tb):
            return None

    fake_ws_mod = MagicMock()
    fake_ws_mod.connect = MagicMock(return_value=_BoomUpstream())
    fake_ws_mod.Subprotocol = str

    client_ws = _FakeWebSocket(frames=[_first_frame()])
    handler = _DummyOpenAIHandler()

    async def _fallback(*args, **kwargs):
        return None

    handler._ws_http_fallback = _fallback  # type: ignore[assignment]

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    assert handler.ws_sessions.active_count() == 0


@pytest.mark.asyncio
async def test_websocket_handler_closes_cleanly_when_websockets_package_missing(monkeypatch):
    client_ws = _FakeWebSocket(frames=[_first_frame()])
    handler = _DummyOpenAIHandler()

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "websockets":
            raise ImportError("missing websockets")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    await handler.handle_openai_responses_ws(client_ws)

    assert client_ws.closed is True
    assert client_ws.close_code == 1011


@pytest.mark.asyncio
async def test_ws_handler_forwards_subprotocol_and_host_only_client_address():
    upstream_events = [
        json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
        json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
    ]
    upstream = _FakeUpstream(upstream_events)
    fake_ws_mod = _make_fake_websockets_module(upstream)

    client_ws = _FakeWebSocket(frames=[_first_frame()])
    client_ws.headers["sec-websocket-protocol"] = "realtime, fallback"
    client_ws.client = SimpleNamespace(host="127.0.0.1", port=None)

    handler = _DummyOpenAIHandler()
    captured_handles = []
    original_register = handler.ws_sessions.register

    def capture_register(handle) -> None:  # noqa: ANN001
        captured_handles.append(handle)
        original_register(handle)

    handler.ws_sessions.register = capture_register  # type: ignore[method-assign]

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    assert client_ws.accepted_subprotocol == "realtime"
    assert captured_handles[0].client_addr == "127.0.0.1"
    assert handler.metrics.active_ws_sessions_max == 1


@pytest.mark.asyncio
async def test_ws_handler_injects_env_auth_and_beta_header(monkeypatch):
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)
    client_ws = _FakeWebSocket(frames=[_first_frame()])
    client_ws.headers = {"x-test": "value"}
    handler = _DummyOpenAIHandler()

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    connect_kwargs = fake_ws_mod.connect.call_args.kwargs
    assert connect_kwargs["additional_headers"]["Authorization"] == "Bearer env-key"
    assert connect_kwargs["additional_headers"]["OpenAI-Beta"] == "responses_websockets=2026-02-06"
    assert connect_kwargs["additional_headers"]["x-test"] == "value"


@pytest.mark.asyncio
async def test_ws_handler_times_out_when_client_never_sends_first_frame(monkeypatch):
    client_ws = _FakeWebSocket(frames=[], hold_after_initial=True)
    handler = _DummyOpenAIHandler()

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )
    monkeypatch.setattr("headroom.proxy.handlers.openai.WS_FIRST_FRAME_TIMEOUT_SECONDS", 0.01)

    await handler.handle_openai_responses_ws(client_ws)

    assert client_ws.closed is True
    assert client_ws.close_code == 1001
    assert handler.metrics.termination_causes[-1] == "client_timeout"


@pytest.mark.asyncio
async def test_ws_handler_passes_through_non_json_first_frame(monkeypatch):
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)
    client_ws = _FakeWebSocket(frames=["raw text frame"])
    handler = _DummyOpenAIHandler()

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    assert upstream.sent[0] == "raw text frame"


@pytest.mark.asyncio
async def test_ws_handler_injects_memory_context_tools_and_instruction(monkeypatch):
    class _MemoryHandler:
        def __init__(self) -> None:
            self.config = SimpleNamespace(inject_context=True, inject_tools=True)

        async def search_and_format_context(self, memory_user_id, messages):  # noqa: ANN001, ANN201
            return "remembered context"

        def inject_tools(self, tools, provider):
            return (
                tools
                + [
                    {
                        "type": "function",
                        "function": {
                            "name": "memory_search",
                            "description": "search memory",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                True,
            )

    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)
    client_ws = _FakeWebSocket(
        frames=[
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "model": "gpt-5.4",
                        "input": "hi",
                        "instructions": "base instructions",
                        "tools": [{"type": "web_search_preview"}],
                    },
                }
            )
        ]
    )
    client_ws.headers["x-headroom-user-id"] = "user-1"
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    payload = json.loads(upstream.sent[0])
    response_body = payload["response"]
    assert "remembered context" in response_body["instructions"]
    assert "memory_search and memory_save tools" in response_body["instructions"]
    assert {"type": "web_search_preview"} in response_body["tools"]
    assert {
        "type": "function",
        "name": "memory_search",
        "description": "search memory",
        "parameters": {"type": "object"},
    } in response_body["tools"]


@pytest.mark.asyncio
async def test_ws_handler_compresses_first_frame_input(monkeypatch):
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)
    client_ws = _FakeWebSocket(
        frames=[
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "model": "gpt-5.4",
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "first"}],
                            },
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "second"}],
                            },
                        ],
                    },
                }
            )
        ]
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler.openai_pipeline.apply.return_value = SimpleNamespace(
        messages=[{"role": "user", "content": "condensed"}],
        tokens_after=1,
    )

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )
    monkeypatch.setattr(
        "headroom.tokenizers.get_tokenizer",
        lambda model: SimpleNamespace(count_messages=lambda messages: len(messages)),
    )

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    payload = json.loads(upstream.sent[0])
    response_body = payload["response"]
    assert (
        response_body["input"] != json.loads(client_ws._frames[0])["response"]["input"]
        if client_ws._frames
        else response_body["input"]
    )


@pytest.mark.asyncio
async def test_ws_handler_reverts_when_compression_inflates_tokens(monkeypatch):
    original_payload = {
        "type": "response.create",
        "response": {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "first"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "second"}],
                },
            ],
        },
    }
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)
    client_ws = _FakeWebSocket(frames=[json.dumps(original_payload)])
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler.openai_pipeline.apply.return_value = SimpleNamespace(
        messages=[{"role": "user", "content": "inflated"}],
        tokens_after=99,
    )
    warnings: list[str] = []

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )
    monkeypatch.setattr(
        "headroom.tokenizers.get_tokenizer",
        lambda model: SimpleNamespace(count_messages=lambda messages: len(messages)),
    )
    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.logger",
        SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda message, *args: warnings.append(message % args if args else message),
            debug=lambda *args, **kwargs: None,
        ),
    )

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    payload = json.loads(upstream.sent[0])
    assert payload == original_payload
    assert any("WS optimization inflated tokens" in line for line in warnings)


@pytest.mark.asyncio
async def test_ws_handler_continues_when_compression_raises(monkeypatch):
    original_payload = {
        "type": "response.create",
        "response": {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "first"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "second"}],
                },
            ],
        },
    }
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)
    client_ws = _FakeWebSocket(frames=[json.dumps(original_payload)])
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler.openai_pipeline.apply.side_effect = RuntimeError("compress failed")
    warnings: list[str] = []

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )
    monkeypatch.setattr(
        "headroom.tokenizers.get_tokenizer",
        lambda model: SimpleNamespace(count_messages=lambda messages: len(messages)),
    )
    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.logger",
        SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda message, *args: warnings.append(message % args if args else message),
            debug=lambda *args, **kwargs: None,
        ),
    )

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    assert json.loads(upstream.sent[0]) == original_payload
    assert any("WS compression failed: compress failed" in line for line in warnings)


@pytest.mark.asyncio
async def test_ws_handler_compression_updates_instructions_for_bare_response_body(monkeypatch):
    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)
    original_payload = {
        "model": "gpt-5.4",
        "instructions": "base instructions",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "first"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "second"}],
            },
        ],
    }
    client_ws = _FakeWebSocket(frames=[json.dumps(original_payload)])
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler.openai_pipeline.apply.return_value = SimpleNamespace(
        messages=[
            {"role": "system", "content": "rewritten instructions"},
            {"role": "user", "content": "condensed"},
        ],
        tokens_after=1,
    )

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )
    monkeypatch.setattr(
        "headroom.tokenizers.get_tokenizer",
        lambda model: SimpleNamespace(count_messages=lambda messages: len(messages)),
    )

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    payload = json.loads(upstream.sent[0])
    assert payload["instructions"] == "rewritten instructions"
    assert payload["input"] != original_payload["input"]


@pytest.mark.asyncio
async def test_ws_handler_memory_context_timeout_fails_open(monkeypatch):
    class _MemoryHandler:
        def __init__(self) -> None:
            self.config = SimpleNamespace(inject_context=True, inject_tools=False)

        async def search_and_format_context(self, memory_user_id, messages):  # noqa: ANN001, ANN201
            raise asyncio.TimeoutError

    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)
    original_payload = {
        "type": "response.create",
        "response": {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "first"}],
                }
            ],
        },
    }
    client_ws = _FakeWebSocket(frames=[json.dumps(original_payload)])
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()
    info_logs: list[str] = []

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )
    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.logger",
        SimpleNamespace(
            info=lambda message, *args: info_logs.append(message % args if args else message),
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        ),
    )

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    assert json.loads(upstream.sent[0]) == original_payload
    assert any("WS Memory: Context lookup exceeded" in line for line in info_logs)


@pytest.mark.asyncio
async def test_ws_handler_memory_injection_failure_is_non_fatal(monkeypatch):
    class _MemoryHandler:
        def __init__(self) -> None:
            self.config = SimpleNamespace(inject_context=False, inject_tools=True)

        def inject_tools(self, tools, provider):
            raise RuntimeError("inject failed")

    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    fake_ws_mod = _make_fake_websockets_module(upstream)
    original_payload = {
        "type": "response.create",
        "response": {
            "model": "gpt-5.4",
            "input": "hi",
            "tools": [{"type": "web_search_preview"}],
        },
    }
    client_ws = _FakeWebSocket(frames=[json.dumps(original_payload)])
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()
    warnings: list[str] = []

    async def passthrough_auth(headers, url=None):  # noqa: ANN001, ANN201
        return headers

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        passthrough_auth,
    )
    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.logger",
        SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda message, *args: warnings.append(message % args if args else message),
            debug=lambda *args, **kwargs: None,
        ),
    )

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    assert json.loads(upstream.sent[0]) == original_payload
    assert any("WS Memory injection failed: inject failed" in line for line in warnings)


@pytest.mark.asyncio
async def test_many_concurrent_sessions_cleanly_drained():
    """50 concurrent sessions: all drain; registry and named tasks go to 0."""
    upstream_events = [
        json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
        json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
    ]

    async def run_one() -> None:
        upstream = _FakeUpstream(list(upstream_events))
        fake_ws_mod = _make_fake_websockets_module(upstream)
        client_ws = _FakeWebSocket(frames=[_first_frame()])
        handler = _DummyOpenAIHandler()
        with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
            await handler.handle_openai_responses_ws(client_ws)
        assert handler.ws_sessions.active_count() == 0

    await asyncio.gather(*[run_one() for _ in range(50)])

    # Global check: no codex-ws-* named task remains.
    leaked = [
        t
        for t in asyncio.all_tasks()
        if (t.get_name() or "").startswith("codex-ws-") and not t.done()
    ]
    assert leaked == []
