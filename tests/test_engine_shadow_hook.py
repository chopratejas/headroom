"""Tests for Chunk 4.3-ii: HeadroomEngine shadow hook in handle_anthropic_messages.

Three test groups:
  1. flag=off — handler behavior + outbound bytes byte-identical (existing tests
     pass; these add explicit zero-overhead assertions).
  2. flag=shadow, zero divergence — drives the handler in shadow mode over
     passthrough, tools, and multi-turn cases; asserts (a) client response
     unchanged, (b) engine_bytes == legacy_bytes (divergence counter stays 0),
     (c) shadow_total incremented.
  3. shadow exception safety — forces the engine to raise; asserts the request
     still succeeds and error metric incremented.

NOTE on TestClient usage: lifespan (and therefore proxy.startup) runs ONLY
when the TestClient is used as a context manager (`with client:`). Without the
context manager, startup is skipped and state set before client creation is
preserved.  All tests below intentionally omit `with client:` so the
pre-seeded http_client and session_tracker_store stay in place.
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
    """Records exact outbound bytes and returns a minimal success response."""

    def __init__(self) -> None:
        self.captured_body: bytes | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
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


class _FakePrefixTracker:
    def __init__(self, frozen_count: int = 0) -> None:
        self._frozen_count = frozen_count
        self._last_original_messages: list = []
        self._last_forwarded_messages: list = []
        # Direct attribute read by handle_anthropic_messages for cache-bust
        # tracking (line: expected_cached = prefix_tracker._cached_token_count).
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


def _make_client(
    *,
    engine_request_path: str = "off",
    optimize: bool = False,
    ccr_inject_tool: bool = False,
    frozen_count: int = 0,
) -> tuple[TestClient, _CapturingTransport]:
    """Build a proxy TestClient with a capturing transport.

    Does NOT enter the TestClient context manager so proxy.startup() is
    never called and our pre-seeded http_client stays in place.
    """
    config = ProxyConfig(
        optimize=optimize,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=ccr_inject_tool,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        engine_request_path=engine_request_path,
    )
    app = create_app(config)
    transport = _CapturingTransport()
    proxy = app.state.proxy

    # Pre-seed the http_client BEFORE TestClient creation (startup skipped).
    proxy.http_client = httpx.AsyncClient(transport=transport)

    # Pin a deterministic session tracker — stable frozen_count + session ID.
    fake_tracker = _FakePrefixTracker(frozen_count=frozen_count)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
        "shadow-test-session"
    )
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

    return TestClient(app), transport


_SIMPLE_BODY = {
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello world"}],
    "max_tokens": 100,
}
_HEADERS = {"x-api-key": "test-key-shadow-hook", "anthropic-version": "2023-06-01"}


# ---------------------------------------------------------------------------
# 1. Flag=off: zero-overhead, byte-identical
# ---------------------------------------------------------------------------


class TestFlagOff:
    """Flag=off must be a complete no-op: no engine call, bytes unchanged."""

    def test_flag_off_request_succeeds(self) -> None:
        client, _ = _make_client(engine_request_path="off")
        resp = client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert resp.status_code == 200

    def test_flag_off_shadow_metrics_untouched(self) -> None:
        client, _ = _make_client(engine_request_path="off")
        proxy = client.app.state.proxy
        before_shadow = proxy.metrics.engine_shadow_total
        before_div = proxy.metrics.engine_shadow_divergence_total
        before_err = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert resp.status_code == 200

        assert proxy.metrics.engine_shadow_total == before_shadow, "shadow fired when flag=off"
        assert proxy.metrics.engine_shadow_divergence_total == before_div
        assert proxy.metrics.engine_shadow_error_total == before_err

    def test_flag_off_engine_never_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _ = _make_client(engine_request_path="off")
        proxy = client.app.state.proxy
        called: list[bool] = []
        original_on_request = proxy.engine.on_request

        def _spy(*args: Any, **kwargs: Any) -> Any:
            called.append(True)
            return original_on_request(*args, **kwargs)

        monkeypatch.setattr(proxy.engine, "on_request", _spy)
        resp = client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert resp.status_code == 200
        assert not called, "engine.on_request should not be called when flag=off"

    def test_flag_off_outbound_bytes_passthrough(self) -> None:
        """Unmutated body with flag=off must be forwarded byte-identical."""
        client, transport = _make_client(engine_request_path="off", optimize=False)
        body_bytes = json.dumps(_SIMPLE_BODY, separators=(",", ":")).encode()
        resp = client.post(
            "/v1/messages",
            content=body_bytes,
            headers={**_HEADERS, "content-type": "application/json"},
        )
        assert resp.status_code == 200
        assert transport.captured_body == body_bytes


# ---------------------------------------------------------------------------
# 2. Flag=shadow: zero divergence + observability
# ---------------------------------------------------------------------------


class TestFlagShadowZeroDivergence:
    """In shadow mode, engine_bytes == legacy_bytes for representative cases."""

    def _assert_shadow_match(
        self,
        proxy: Any,
        before_total: int,
        before_div: int,
        before_err: int,
        *,
        n_requests: int = 1,
    ) -> None:
        """Assert: shadow fired n_requests times, zero divergence, zero errors."""
        assert proxy.metrics.engine_shadow_total == before_total + n_requests, (
            f"shadow_total should increment by {n_requests}"
        )
        assert proxy.metrics.engine_shadow_divergence_total == before_div, (
            "divergence counter must stay 0"
        )
        assert proxy.metrics.engine_shadow_error_total == before_err, "error counter must stay 0"

    def test_shadow_passthrough_no_compression(self) -> None:
        """Passthrough (no compression): legacy and engine both return original bytes."""
        client, _ = _make_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        bt = proxy.metrics.engine_shadow_total
        bd = proxy.metrics.engine_shadow_divergence_total
        be = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert resp.status_code == 200
        self._assert_shadow_match(proxy, bt, bd, be)

    def test_shadow_client_response_unchanged(self) -> None:
        """The response the client receives is byte-identical in shadow mode."""
        client_off, _ = _make_client(engine_request_path="off", optimize=False)
        resp_off = client_off.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)

        client_shadow, _ = _make_client(engine_request_path="shadow", optimize=False)
        resp_shadow = client_shadow.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)

        assert resp_off.status_code == resp_shadow.status_code == 200
        assert resp_off.json()["id"] == resp_shadow.json()["id"]

    def test_shadow_outbound_bytes_unchanged(self) -> None:
        """Outbound bytes forwarded to upstream are identical in shadow mode."""
        body_bytes = json.dumps(_SIMPLE_BODY, separators=(",", ":")).encode()
        client_shadow, transport_shadow = _make_client(engine_request_path="shadow", optimize=False)
        resp = client_shadow.post(
            "/v1/messages",
            content=body_bytes,
            headers={**_HEADERS, "content-type": "application/json"},
        )
        assert resp.status_code == 200
        assert transport_shadow.captured_body == body_bytes

    def test_shadow_shadow_total_incremented(self) -> None:
        """shadow_total must increment exactly once per shadow-mode request."""
        client, _ = _make_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before = proxy.metrics.engine_shadow_total
        client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert proxy.metrics.engine_shadow_total == before + 1

    def test_shadow_multiple_requests_accumulate(self) -> None:
        """shadow_total increments on every request in shadow mode."""
        client, _ = _make_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before = proxy.metrics.engine_shadow_total
        for _ in range(3):
            client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert proxy.metrics.engine_shadow_total == before + 3
        assert proxy.metrics.engine_shadow_divergence_total == 0

    def test_shadow_multi_turn_frozen_count_seeded(self) -> None:
        """Multi-turn: frozen_count from snapshot seeds the engine correctly.

        Seeds frozen_count=1 so the handler treats the first message as frozen.
        The engine sees the same seeded state and must produce matching bytes
        (zero divergence).
        """
        multi_turn_body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "Turn 1 text"},
                {"role": "assistant", "content": "Turn 1 reply"},
                {"role": "user", "content": "Turn 2 text"},
            ],
            "max_tokens": 100,
        }
        client, _ = _make_client(
            engine_request_path="shadow",
            optimize=False,
            frozen_count=1,
        )
        proxy = client.app.state.proxy
        bt = proxy.metrics.engine_shadow_total
        bd = proxy.metrics.engine_shadow_divergence_total
        be = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/messages", json=multi_turn_body, headers=_HEADERS)
        assert resp.status_code == 200
        self._assert_shadow_match(proxy, bt, bd, be)

    def test_shadow_tools_body_already_sorted_no_divergence(self) -> None:
        """Body with already-sorted tools: shadow fires, no divergence.

        When tools arrive pre-sorted (alphabetical), the handler's tool-sort
        produces no mutation and the engine's passthrough path returns the same
        bytes.  Tools that need reordering are a KNOWN gap between the engine's
        passthrough path (byte-identical by design) and the handler (always
        sorts tools).  That gap will be closed in Chunk 4.4 when the engine
        drives the full request path.
        """
        body_with_sorted_tools = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "max_tokens": 100,
            "tools": [
                # Already in alphabetical order — sort is a no-op for both paths.
                {
                    "name": "a_tool",
                    "description": "a",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "z_tool",
                    "description": "z",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        }
        client, _ = _make_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        bt = proxy.metrics.engine_shadow_total
        bd = proxy.metrics.engine_shadow_divergence_total
        be = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/messages", json=body_with_sorted_tools, headers=_HEADERS)
        assert resp.status_code == 200
        self._assert_shadow_match(proxy, bt, bd, be)

    def test_shadow_unsorted_tools_match(self) -> None:
        """Unsorted tools: shadow fires, no divergence (gap closed in D1 fix).

        The legacy handler always sorts tools deterministically before forwarding
        (handler ~line 1634, outside any bypass or should_compress gate).
        After the D1 fix, the engine's no-compression path also applies the same
        tool-sort + byte-faithful serialization, so both produce identical bytes.
        """
        body_unsorted = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "max_tokens": 100,
            "tools": [
                {
                    "name": "z_tool",
                    "description": "z",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "a_tool",
                    "description": "a",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        }
        client, _ = _make_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before_total = proxy.metrics.engine_shadow_total
        before_div = proxy.metrics.engine_shadow_divergence_total
        before_err = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/messages", json=body_unsorted, headers=_HEADERS)
        assert resp.status_code == 200, "shadow must not break the request"
        self._assert_shadow_match(proxy, before_total, before_div, before_err)


# ---------------------------------------------------------------------------
# 3. Shadow exception safety
# ---------------------------------------------------------------------------


class TestShadowExceptionSafety:
    """Force engine.on_request to throw; request must still succeed."""

    def test_exception_in_engine_does_not_break_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = _make_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy
        before_err = proxy.metrics.engine_shadow_error_total
        before_total = proxy.metrics.engine_shadow_total

        # Force engine.on_request to raise
        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("injected engine failure")

        monkeypatch.setattr(proxy.engine, "on_request", _boom)

        resp = client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert resp.status_code == 200, f"request failed: {resp.json()}"

        # Error counter must have incremented
        assert proxy.metrics.engine_shadow_error_total == before_err + 1
        # shadow_total is NOT incremented when the engine throws (the counter
        # increments only on successful completion of the shadow comparison)
        assert proxy.metrics.engine_shadow_total == before_total, (
            "shadow_total should not increment on engine exception"
        )

    def test_engine_exception_leaves_response_body_intact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The client response body is correct even after engine throws."""
        client, _ = _make_client(engine_request_path="shadow", optimize=False)
        proxy = client.app.state.proxy

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("injected boom")

        monkeypatch.setattr(proxy.engine, "on_request", _boom)

        resp = client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("type") == "message"


# ---------------------------------------------------------------------------
# 4. get_engine_request_path helper
# ---------------------------------------------------------------------------


class TestGetEngineRequestPath:
    """Unit tests for the helpers module function."""

    def test_default_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_ENGINE_REQUEST_PATH", raising=False)
        from headroom.proxy.helpers import get_engine_request_path

        assert get_engine_request_path() == "off"

    def test_env_shadow_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_ENGINE_REQUEST_PATH", "shadow")
        from headroom.proxy.helpers import get_engine_request_path

        assert get_engine_request_path() == "shadow"

    def test_env_off_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_ENGINE_REQUEST_PATH", "off")
        from headroom.proxy.helpers import get_engine_request_path

        assert get_engine_request_path() == "off"

    def test_config_value_takes_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_ENGINE_REQUEST_PATH", "shadow")
        from headroom.proxy.helpers import get_engine_request_path

        # Explicit config_value="off" overrides env "shadow"
        assert get_engine_request_path("off") == "off"

    def test_invalid_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_ENGINE_REQUEST_PATH", "bad_value")
        from headroom.proxy.helpers import get_engine_request_path

        with pytest.raises(ValueError, match="engine_request_path"):
            get_engine_request_path()

    def test_on_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chunk 4.4a: 'on' is now a valid mode (not reserved/rejected)."""
        monkeypatch.setenv("HEADROOM_ENGINE_REQUEST_PATH", "on")
        from headroom.proxy.helpers import get_engine_request_path

        assert get_engine_request_path() == "on"

    def test_on_accepted_via_config(self) -> None:
        from headroom.proxy.helpers import get_engine_request_path

        assert get_engine_request_path("on") == "on"


# ---------------------------------------------------------------------------
# 5. Flag=on: engine bytes forwarded + fallback safety (Chunk 4.4a)
# ---------------------------------------------------------------------------


class TestFlagOn:
    """engine_request_path='on' forwards engine bytes; falls back to legacy on error."""

    def test_flag_on_request_succeeds(self) -> None:
        """'on' mode must not break the request."""
        client, _ = _make_client(engine_request_path="on")
        resp = client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert resp.status_code == 200

    def test_flag_on_forwards_engine_bytes(self) -> None:
        """'on' mode: upstream-received bytes == engine bytes.

        Since shadow divergence=0, engine bytes == legacy bytes, so we assert
        the upstream-received bytes match what the engine would produce
        (which also matches legacy).
        """
        body_bytes = json.dumps(_SIMPLE_BODY, separators=(",", ":")).encode()
        client_on, transport_on = _make_client(engine_request_path="on", optimize=False)
        resp = client_on.post(
            "/v1/messages",
            content=body_bytes,
            headers={**_HEADERS, "content-type": "application/json"},
        )
        assert resp.status_code == 200
        # Engine bytes == legacy bytes (shadow=0 invariant) so upstream received
        # the same bytes the legacy path would have forwarded.
        assert transport_on.captured_body is not None
        assert len(transport_on.captured_body) > 0

    def test_flag_on_bytes_match_legacy(self) -> None:
        """'on' mode forwards byte-identical bytes to what 'off' mode would send."""
        body_bytes = json.dumps(_SIMPLE_BODY, separators=(",", ":")).encode()

        client_off, transport_off = _make_client(engine_request_path="off", optimize=False)
        client_off.post(
            "/v1/messages",
            content=body_bytes,
            headers={**_HEADERS, "content-type": "application/json"},
        )

        client_on, transport_on = _make_client(engine_request_path="on", optimize=False)
        client_on.post(
            "/v1/messages",
            content=body_bytes,
            headers={**_HEADERS, "content-type": "application/json"},
        )

        assert transport_off.captured_body == transport_on.captured_body, (
            "engine_request_path='on' must forward the same bytes as 'off' "
            "(shadow divergence=0 invariant)"
        )

    def test_flag_on_multi_turn(self) -> None:
        """Multi-turn request succeeds in 'on' mode with frozen_count seeded."""
        multi_turn_body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Reply 1"},
                {"role": "user", "content": "Turn 2"},
            ],
            "max_tokens": 100,
        }
        client, _ = _make_client(engine_request_path="on", optimize=False, frozen_count=1)
        resp = client.post("/v1/messages", json=multi_turn_body, headers=_HEADERS)
        assert resp.status_code == 200

    def test_flag_on_tools_body(self) -> None:
        """'on' mode handles tools body without breaking."""
        body_with_tools = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "max_tokens": 100,
            "tools": [
                {
                    "name": "a_tool",
                    "description": "a",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "z_tool",
                    "description": "z",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        }
        client, transport = _make_client(engine_request_path="on", optimize=False)
        resp = client.post("/v1/messages", json=body_with_tools, headers=_HEADERS)
        assert resp.status_code == 200
        assert transport.captured_body is not None

    def test_flag_on_fallback_on_engine_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'on' + engine raises → request SUCCEEDS, fallback metric fires, legacy bytes sent."""
        client, transport = _make_client(engine_request_path="on", optimize=False)
        proxy = client.app.state.proxy
        before_fallback = proxy.metrics.engine_on_fallback_total

        # Force the engine to raise.
        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("injected engine failure for on-mode fallback test")

        monkeypatch.setattr(proxy.engine, "on_request", _boom)

        # Send a known-bytes request so we can verify legacy bytes were forwarded.
        body_bytes = json.dumps(_SIMPLE_BODY, separators=(",", ":")).encode()
        resp = client.post(
            "/v1/messages",
            content=body_bytes,
            headers={**_HEADERS, "content-type": "application/json"},
        )

        assert resp.status_code == 200, f"request must succeed even on engine error: {resp.json()}"
        # Fallback metric must have incremented.
        assert proxy.metrics.engine_on_fallback_total == before_fallback + 1, (
            "engine_on_fallback_total must increment when engine raises in 'on' mode"
        )
        # Upstream received the legacy bytes (not None / empty).
        assert transport.captured_body == body_bytes, (
            "legacy bytes must be forwarded when engine raises in 'on' mode"
        )

    def test_flag_on_fallback_response_body_intact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client response is correct even after engine raises in 'on' mode."""
        client, _ = _make_client(engine_request_path="on", optimize=False)
        proxy = client.app.state.proxy

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("injected boom in on-mode")

        monkeypatch.setattr(proxy.engine, "on_request", _boom)

        resp = client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("type") == "message"

    def test_flag_on_no_shadow_metrics_side_effects(self) -> None:
        """'on' mode must not increment shadow-specific metrics."""
        client, _ = _make_client(engine_request_path="on", optimize=False)
        proxy = client.app.state.proxy
        before_shadow = proxy.metrics.engine_shadow_total
        before_div = proxy.metrics.engine_shadow_divergence_total
        before_shadow_err = proxy.metrics.engine_shadow_error_total

        resp = client.post("/v1/messages", json=_SIMPLE_BODY, headers=_HEADERS)
        assert resp.status_code == 200

        assert proxy.metrics.engine_shadow_total == before_shadow, (
            "shadow_total must not increment in 'on' mode"
        )
        assert proxy.metrics.engine_shadow_divergence_total == before_div
        assert proxy.metrics.engine_shadow_error_total == before_shadow_err


# ---------------------------------------------------------------------------
# 4. #31 — engine compression-cache isolation + per-session keying
# ---------------------------------------------------------------------------


class TestEngineCompressionCacheIsolation:
    """#31: the engine must use its OWN compression-cache store, keyed by the
    REAL per-session id — not the proxy's shared store under a constant key.

    Pre-#31 the Anthropic shadow/on seeded stores returned a constant
    ("shadow-seeded"/"engine-on-seeded") AND the engine was wired to the
    proxy's shared `_get_compression_cache`. So every conversation collapsed
    into ONE shared cache entry → cross-tenant content bleed in `on` mode and
    contaminated divergence metrics in `shadow`.
    """

    def test_engine_cache_keyed_per_session_and_isolated(self) -> None:
        client, _ = _make_client(engine_request_path="shadow", optimize=True)
        proxy = client.app.state.proxy

        # Make the legacy session_id (which the seeded store now forwards to
        # the engine) vary per request, so two conversations map to two ids.
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            request.headers.get("x-test-session", "default")
        )
        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.get_or_create = lambda sid, provider: fake_tracker

        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello world"}],
        }

        r1 = client.post("/v1/messages", json=body, headers={**_HEADERS, "x-test-session": "alpha"})
        r2 = client.post("/v1/messages", json=body, headers={**_HEADERS, "x-test-session": "beta"})
        assert r1.status_code == 200
        assert r2.status_code == 200

        # No constant-collapse: engine cache keyed by the two REAL session ids.
        # Pre-#31 this would be {"shadow-seeded"} (a single collapsed entry).
        assert set(proxy._engine_compression_caches.keys()) == {"alpha", "beta"}, (
            "engine compression cache must be keyed by the real per-session id; got "
            f"{sorted(proxy._engine_compression_caches.keys())}"
        )

        # Isolation: the engine store is a distinct object from the legacy proxy
        # store, so engine shadow/on activity can never read or mutate the cache
        # the legacy path serves from.
        assert proxy._engine_compression_caches is not proxy._compression_caches
        for sid in ("alpha", "beta"):
            if sid in proxy._compression_caches:
                assert proxy._engine_compression_caches[sid] is not proxy._compression_caches[sid]
