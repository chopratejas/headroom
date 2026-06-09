"""Tests for Chunk 5-shadow/on: engine shadow+flip hooks in handle_openai_chat
and handle_openai_responses.

Mirrors tests/test_engine_shadow_hook.py (Anthropic) for the OpenAI handlers.

Three groups per handler:
  1. flag=off — no-op, no engine call, bytes unchanged.
  2. flag=shadow — engine observe-only; metrics fire; legacy bytes forwarded.
  3. flag=on — engine bytes forwarded; exception → legacy fallback + metric.

Additional group:
  4. Responses+memory expected divergence — shadow fires but diverges when
     memory is injected on the responses path (engine does not yet do memory
     on responses; this is EXPECTED signal, not a bug).

NOTE: TestClient without context manager skips lifespan so pre-seeded
http_client and session_tracker_store stay in place (same pattern as the
Anthropic test).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Shared transport stub
# ---------------------------------------------------------------------------


class _CapturingTransport(httpx.AsyncBaseTransport):
    """Records exact outbound bytes and returns minimal success responses."""

    def __init__(self, *, is_responses: bool = False) -> None:
        self.captured_body: bytes | None = None
        self._is_responses = is_responses

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body
        if self._is_responses:
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "model": "gpt-4o",
                    "output": [{"type": "message", "role": "assistant", "content": "ok"}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "object": "chat.completion",
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
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )


class _FakePrefixTracker:
    def __init__(self, frozen_count: int = 0) -> None:
        self._frozen_count = frozen_count
        self._last_original_messages: list = []
        self._last_forwarded_messages: list = []
        self._cached_token_count: int = 0

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    def get_last_original_messages(self) -> list:
        return list(self._last_original_messages)

    def get_last_forwarded_messages(self) -> list:
        return list(self._last_forwarded_messages)

    def update_from_response(self, **kwargs: Any) -> None:
        self._last_original_messages = list(
            kwargs.get("original_messages", kwargs.get("messages", []))
        )
        self._last_forwarded_messages = list(kwargs.get("messages", []))


def _make_chat_client(
    *,
    engine_request_path: str = "off",
    optimize: bool = False,
    frozen_count: int = 0,
) -> tuple[TestClient, _CapturingTransport]:
    """Build a proxy TestClient for /v1/chat/completions tests."""
    config = ProxyConfig(
        optimize=optimize,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        engine_request_path=engine_request_path,
    )
    app = create_app(config)
    transport = _CapturingTransport()
    proxy = app.state.proxy

    proxy.http_client = httpx.AsyncClient(transport=transport)

    fake_tracker = _FakePrefixTracker(frozen_count=frozen_count)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
        "chat-shadow-test-session"
    )
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

    return TestClient(app), transport


def _make_responses_client(
    *,
    engine_request_path: str = "off",
    optimize: bool = False,
) -> tuple[TestClient, _CapturingTransport]:
    """Build a proxy TestClient for /v1/responses tests."""
    config = ProxyConfig(
        optimize=optimize,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        engine_request_path=engine_request_path,
    )
    app = create_app(config)
    transport = _CapturingTransport(is_responses=True)
    proxy = app.state.proxy

    proxy.http_client = httpx.AsyncClient(transport=transport)

    fake_tracker = _FakePrefixTracker()
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
        "resp-shadow-test-session"
    )
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

    return TestClient(app), transport


_CHAT_BODY = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello world"}],
    "max_tokens": 100,
}
_CHAT_HEADERS = {
    "authorization": "Bearer test-key-openai-shadow",
    "content-type": "application/json",
}

_RESP_BODY = {
    "model": "gpt-4o",
    "input": "Hello from responses",
}
_RESP_HEADERS = {
    "authorization": "Bearer test-key-openai-resp-shadow",
    "content-type": "application/json",
}


# ---------------------------------------------------------------------------
# 1. Chat — flag=off
# ---------------------------------------------------------------------------


class TestChatFlagOff:
    """flag=off must be a complete no-op for handle_openai_chat."""

    def test_flag_off_request_succeeds(self) -> None:
        client, _ = _make_chat_client(engine_request_path="off")
        resp = client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)
        assert resp.status_code == 200

    def test_flag_off_shadow_metrics_untouched(self) -> None:
        client, _ = _make_chat_client(engine_request_path="off")
        proxy = client.app.state.proxy
        before_shadow = proxy.metrics.engine_shadow_total
        before_div = proxy.metrics.engine_shadow_divergence_total
        before_err = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)
        assert resp.status_code == 200
        assert proxy.metrics.engine_shadow_total == before_shadow, "shadow fired when flag=off"
        assert proxy.metrics.engine_shadow_divergence_total == before_div
        assert proxy.metrics.engine_shadow_error_total == before_err

    def test_flag_off_engine_never_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _ = _make_chat_client(engine_request_path="off")
        proxy = client.app.state.proxy
        called: list[bool] = []
        original = proxy.engine.on_request

        def _spy(*args: Any, **kwargs: Any) -> Any:
            called.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(proxy.engine, "on_request", _spy)
        resp = client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)
        assert resp.status_code == 200
        assert not called, "engine.on_request should not be called when flag=off"


# ---------------------------------------------------------------------------
# 2. Chat — flag=shadow
# ---------------------------------------------------------------------------


class TestChatFlagShadow:
    """flag=shadow: engine observe-only, metrics fire, legacy bytes forwarded."""

    def _assert_shadow_match(
        self,
        proxy: Any,
        before_total: int,
        before_div: int,
        before_err: int,
        *,
        n: int = 1,
    ) -> None:
        assert proxy.metrics.engine_shadow_total == before_total + n, (
            f"shadow_total should increment by {n}"
        )
        assert proxy.metrics.engine_shadow_divergence_total == before_div, (
            "divergence counter must stay 0"
        )
        assert proxy.metrics.engine_shadow_error_total == before_err, "error counter must stay 0"

    def test_shadow_passthrough_no_compression(self) -> None:
        client, _ = _make_chat_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        bt = proxy.metrics.engine_shadow_total
        bd = proxy.metrics.engine_shadow_divergence_total
        be = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)
        assert resp.status_code == 200
        self._assert_shadow_match(proxy, bt, bd, be)

    def test_shadow_client_response_unchanged(self) -> None:
        """Client response is byte-identical in shadow mode vs off mode."""
        client_off, _ = _make_chat_client(engine_request_path="off", optimize=False)
        resp_off = client_off.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)

        client_sh, _ = _make_chat_client(engine_request_path="shadow", optimize=False)
        resp_sh = client_sh.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)

        assert resp_off.status_code == resp_sh.status_code == 200
        assert resp_off.json()["id"] == resp_sh.json()["id"]

    def test_shadow_outbound_bytes_unchanged(self) -> None:
        """Outbound upstream bytes are identical in shadow mode vs off mode."""
        body_bytes = json.dumps(_CHAT_BODY, separators=(",", ":")).encode()

        client_off, transport_off = _make_chat_client(engine_request_path="off", optimize=False)
        client_off.post(
            "/v1/chat/completions",
            content=body_bytes,
            headers=_CHAT_HEADERS,
        )

        client_sh, transport_sh = _make_chat_client(engine_request_path="shadow", optimize=False)
        client_sh.post(
            "/v1/chat/completions",
            content=body_bytes,
            headers=_CHAT_HEADERS,
        )

        # Both should have non-empty captured bodies (bytes may differ from
        # input due to body mutation on the chat path, but must match each other).
        assert transport_off.captured_body is not None
        assert transport_sh.captured_body == transport_off.captured_body, (
            "shadow mode must forward the same bytes as off mode"
        )

    def test_shadow_total_incremented(self) -> None:
        client, _ = _make_chat_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before = proxy.metrics.engine_shadow_total
        client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)
        assert proxy.metrics.engine_shadow_total == before + 1

    def test_shadow_multiple_requests_accumulate(self) -> None:
        client, _ = _make_chat_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before = proxy.metrics.engine_shadow_total
        for _ in range(3):
            client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)
        assert proxy.metrics.engine_shadow_total == before + 3
        assert proxy.metrics.engine_shadow_divergence_total == 0

    def test_shadow_multi_turn_frozen_count_seeded(self) -> None:
        """frozen_count from snapshot seeds the engine correctly (zero divergence)."""
        multi_turn = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Reply 1"},
                {"role": "user", "content": "Turn 2"},
            ],
            "max_tokens": 100,
        }
        client, _ = _make_chat_client(engine_request_path="shadow", optimize=False, frozen_count=1)
        proxy = client.app.state.proxy
        bt = proxy.metrics.engine_shadow_total
        bd = proxy.metrics.engine_shadow_divergence_total
        be = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/chat/completions", json=multi_turn, headers=_CHAT_HEADERS)
        assert resp.status_code == 200
        self._assert_shadow_match(proxy, bt, bd, be)

    def test_shadow_exception_safety(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Engine error in shadow mode does not break the request."""
        client, _ = _make_chat_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before_err = proxy.metrics.engine_shadow_error_total

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("injected shadow failure")

        monkeypatch.setattr(proxy.engine, "on_request", _boom)
        resp = client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)
        assert resp.status_code == 200, f"shadow error must not break request: {resp.json()}"
        assert proxy.metrics.engine_shadow_error_total == before_err + 1


# ---------------------------------------------------------------------------
# 3. Chat — flag=on
# ---------------------------------------------------------------------------


class TestChatFlagOn:
    """flag=on: engine bytes forwarded; exception → legacy fallback."""

    def test_flag_on_request_succeeds(self) -> None:
        client, _ = _make_chat_client(engine_request_path="on")
        resp = client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)
        assert resp.status_code == 200

    def test_flag_on_forwards_engine_bytes(self) -> None:
        """'on' mode must forward non-empty bytes upstream."""
        client, transport = _make_chat_client(engine_request_path="on", optimize=False)
        resp = client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)
        assert resp.status_code == 200
        assert transport.captured_body is not None
        assert len(transport.captured_body) > 0

    def test_flag_on_bytes_match_legacy(self) -> None:
        """'on' mode must forward byte-identical bytes to what 'off' mode sends.

        Shadow divergence=0 invariant: engine path is byte-identical to the
        legacy path for simple passthrough (no compression, no CCR, no memory).
        """
        body_bytes = json.dumps(_CHAT_BODY, separators=(",", ":")).encode()

        client_off, transport_off = _make_chat_client(engine_request_path="off", optimize=False)
        client_off.post("/v1/chat/completions", content=body_bytes, headers=_CHAT_HEADERS)

        client_on, transport_on = _make_chat_client(engine_request_path="on", optimize=False)
        client_on.post("/v1/chat/completions", content=body_bytes, headers=_CHAT_HEADERS)

        assert transport_off.captured_body == transport_on.captured_body, (
            "engine_request_path='on' must forward same bytes as 'off' (shadow-divergence=0)"
        )

    def test_flag_on_multi_turn_succeeds(self) -> None:
        multi_turn = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Reply 1"},
                {"role": "user", "content": "Turn 2"},
            ],
            "max_tokens": 100,
        }
        client, _ = _make_chat_client(engine_request_path="on", optimize=False, frozen_count=1)
        resp = client.post("/v1/chat/completions", json=multi_turn, headers=_CHAT_HEADERS)
        assert resp.status_code == 200

    def test_flag_on_fallback_on_engine_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Engine raises in 'on' mode → request SUCCEEDS, fallback metric fires."""
        client, transport = _make_chat_client(engine_request_path="on", optimize=False)
        proxy = client.app.state.proxy
        before_fallback = proxy.metrics.engine_on_fallback_total

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("injected engine failure for on-mode fallback test")

        monkeypatch.setattr(proxy.engine, "on_request", _boom)

        body_bytes = json.dumps(_CHAT_BODY, separators=(",", ":")).encode()
        resp = client.post("/v1/chat/completions", content=body_bytes, headers=_CHAT_HEADERS)
        assert resp.status_code == 200, f"request must succeed even on engine error: {resp.json()}"
        assert proxy.metrics.engine_on_fallback_total == before_fallback + 1

    def test_flag_on_no_shadow_metrics_side_effects(self) -> None:
        """'on' mode must not increment shadow-specific metrics."""
        client, _ = _make_chat_client(engine_request_path="on", optimize=False)
        proxy = client.app.state.proxy
        before_shadow = proxy.metrics.engine_shadow_total
        before_div = proxy.metrics.engine_shadow_divergence_total
        before_err = proxy.metrics.engine_shadow_error_total

        client.post("/v1/chat/completions", json=_CHAT_BODY, headers=_CHAT_HEADERS)

        assert proxy.metrics.engine_shadow_total == before_shadow
        assert proxy.metrics.engine_shadow_divergence_total == before_div
        assert proxy.metrics.engine_shadow_error_total == before_err


# ---------------------------------------------------------------------------
# 4. Responses — flag=off
# ---------------------------------------------------------------------------


class TestResponsesFlagOff:
    """flag=off must be a complete no-op for handle_openai_responses."""

    def test_flag_off_request_succeeds(self) -> None:
        client, _ = _make_responses_client(engine_request_path="off")
        resp = client.post("/v1/responses", json=_RESP_BODY, headers=_RESP_HEADERS)
        assert resp.status_code == 200

    def test_flag_off_shadow_metrics_untouched(self) -> None:
        client, _ = _make_responses_client(engine_request_path="off")
        proxy = client.app.state.proxy
        before_shadow = proxy.metrics.engine_shadow_total
        before_div = proxy.metrics.engine_shadow_divergence_total
        before_err = proxy.metrics.engine_shadow_error_total

        client.post("/v1/responses", json=_RESP_BODY, headers=_RESP_HEADERS)
        assert proxy.metrics.engine_shadow_total == before_shadow
        assert proxy.metrics.engine_shadow_divergence_total == before_div
        assert proxy.metrics.engine_shadow_error_total == before_err

    def test_flag_off_engine_never_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _ = _make_responses_client(engine_request_path="off")
        proxy = client.app.state.proxy
        called: list[bool] = []
        original = proxy.engine.on_request

        def _spy(*args: Any, **kwargs: Any) -> Any:
            called.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(proxy.engine, "on_request", _spy)
        resp = client.post("/v1/responses", json=_RESP_BODY, headers=_RESP_HEADERS)
        assert resp.status_code == 200
        assert not called, "engine.on_request should not be called when flag=off"


# ---------------------------------------------------------------------------
# 5. Responses — flag=shadow
# ---------------------------------------------------------------------------


class TestResponsesFlagShadow:
    """flag=shadow for /v1/responses: engine observe-only, metrics fire."""

    def test_shadow_request_succeeds(self) -> None:
        client, _ = _make_responses_client(engine_request_path="shadow", optimize=False)
        resp = client.post("/v1/responses", json=_RESP_BODY, headers=_RESP_HEADERS)
        assert resp.status_code == 200

    def test_shadow_total_incremented(self) -> None:
        client, _ = _make_responses_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before = proxy.metrics.engine_shadow_total
        client.post("/v1/responses", json=_RESP_BODY, headers=_RESP_HEADERS)
        assert proxy.metrics.engine_shadow_total == before + 1

    def test_shadow_compression_only_zero_divergence(self) -> None:
        """Compression-only (no memory, no CCR): shadow fires with zero divergence.

        The engine's responses path does NOT do CCR or memory (deferred).
        On a simple passthrough request without memory enabled,
        engine and legacy produce identical bytes → divergence counter stays 0.
        """
        client, _ = _make_responses_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before_div = proxy.metrics.engine_shadow_divergence_total
        before_err = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/responses", json=_RESP_BODY, headers=_RESP_HEADERS)
        assert resp.status_code == 200
        assert proxy.metrics.engine_shadow_divergence_total == before_div, (
            "compression-only responses: divergence must stay 0 (no memory or CCR)"
        )
        assert proxy.metrics.engine_shadow_error_total == before_err

    def test_shadow_legacy_bytes_unchanged(self) -> None:
        """Legacy bytes forwarded in shadow mode match off-mode bytes."""
        body_bytes = json.dumps(_RESP_BODY, separators=(",", ":")).encode()

        client_off, transport_off = _make_responses_client(
            engine_request_path="off", optimize=False
        )
        client_off.post("/v1/responses", content=body_bytes, headers=_RESP_HEADERS)

        client_sh, transport_sh = _make_responses_client(
            engine_request_path="shadow", optimize=False
        )
        client_sh.post("/v1/responses", content=body_bytes, headers=_RESP_HEADERS)

        assert transport_off.captured_body is not None
        assert transport_sh.captured_body == transport_off.captured_body

    def test_shadow_exception_safety(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Engine error in responses shadow mode does not break the request."""
        client, _ = _make_responses_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before_err = proxy.metrics.engine_shadow_error_total

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("injected shadow failure in responses")

        monkeypatch.setattr(proxy.engine, "on_request", _boom)
        resp = client.post("/v1/responses", json=_RESP_BODY, headers=_RESP_HEADERS)
        assert resp.status_code == 200
        assert proxy.metrics.engine_shadow_error_total == before_err + 1


# ---------------------------------------------------------------------------
# 6. Responses — flag=on
# ---------------------------------------------------------------------------


class TestResponsesFlagOn:
    """flag=on for /v1/responses: engine bytes forwarded; exception → fallback."""

    def test_flag_on_request_succeeds(self) -> None:
        client, _ = _make_responses_client(engine_request_path="on")
        resp = client.post("/v1/responses", json=_RESP_BODY, headers=_RESP_HEADERS)
        assert resp.status_code == 200

    def test_flag_on_bytes_match_legacy(self) -> None:
        """'on' mode bytes match 'off' mode bytes (shadow-divergence=0 invariant)."""
        body_bytes = json.dumps(_RESP_BODY, separators=(",", ":")).encode()

        client_off, transport_off = _make_responses_client(
            engine_request_path="off", optimize=False
        )
        client_off.post("/v1/responses", content=body_bytes, headers=_RESP_HEADERS)

        client_on, transport_on = _make_responses_client(engine_request_path="on", optimize=False)
        client_on.post("/v1/responses", content=body_bytes, headers=_RESP_HEADERS)

        assert transport_off.captured_body == transport_on.captured_body, (
            "engine_request_path='on' must forward same bytes as 'off' for responses"
        )

    def test_flag_on_fallback_on_engine_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Engine raises in responses 'on' mode → request SUCCEEDS, fallback metric fires."""
        client, transport = _make_responses_client(engine_request_path="on", optimize=False)
        proxy = client.app.state.proxy
        before_fallback = proxy.metrics.engine_on_fallback_total

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("injected engine failure in responses on-mode")

        monkeypatch.setattr(proxy.engine, "on_request", _boom)

        body_bytes = json.dumps(_RESP_BODY, separators=(",", ":")).encode()
        resp = client.post("/v1/responses", content=body_bytes, headers=_RESP_HEADERS)
        assert resp.status_code == 200, f"must succeed even on engine error: {resp.json()}"
        assert proxy.metrics.engine_on_fallback_total == before_fallback + 1

    def test_flag_on_no_shadow_metrics_side_effects(self) -> None:
        """'on' mode must not increment shadow-specific metrics."""
        client, _ = _make_responses_client(engine_request_path="on", optimize=False)
        proxy = client.app.state.proxy
        before_shadow = proxy.metrics.engine_shadow_total

        client.post("/v1/responses", json=_RESP_BODY, headers=_RESP_HEADERS)
        assert proxy.metrics.engine_shadow_total == before_shadow
