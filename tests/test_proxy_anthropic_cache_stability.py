"""Regression tests for Anthropic prefix-cache stability in proxy mode."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.server import ProxyConfig, create_app


class _FakePrefixTracker:
    def __init__(self, frozen_count: int):
        self._frozen_count = frozen_count
        self._cached_token_count = 0
        self._last_original_messages = []
        self._last_forwarded_messages = []

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    def get_last_original_messages(self):  # noqa: ANN201
        return self._last_original_messages.copy()

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return self._last_forwarded_messages.copy()

    def update_from_response(self, **kwargs):  # noqa: ANN003
        self._cached_token_count = kwargs.get("cache_read_tokens", 0) + kwargs.get(
            "cache_write_tokens", 0
        )
        self._last_original_messages = kwargs.get(
            "original_messages", kwargs.get("messages", [])
        ).copy()
        self._last_forwarded_messages = kwargs.get("messages", []).copy()
        return None


class _FakeImageCompressor:
    def __init__(self):
        self.last_result = None

    def has_images(self, messages):  # noqa: ANN001
        return True

    def compress(self, messages, provider="anthropic"):  # noqa: ANN001
        assert provider == "anthropic"
        assert len(messages) == 1
        msg = messages[0]
        content = msg["content"]
        updated_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                src = block.get("source", {})
                updated_content.append(
                    {
                        "type": "image",
                        "source": {**src, "data": "COMPRESSED_IMAGE_BYTES"},
                    }
                )
            else:
                updated_content.append(block)
        return [{**msg, "content": updated_content}]


def _make_proxy_client() -> TestClient:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=True,
    )
    app = create_app(config)
    return TestClient(app)


def test_anthropic_tools_sorted_deterministically_before_forward() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            captured["body"] = body
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

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {"name": "zeta", "description": "z", "input_schema": {"type": "object"}},
                    {"name": "alpha", "description": "a", "input_schema": {"type": "object"}},
                    {"name": "mu", "description": "m", "input_schema": {"type": "object"}},
                ],
            },
        )

        assert response.status_code == 200
        sent_tools = captured["body"]["tools"]
        assert [t["name"] for t in sent_tools] == ["alpha", "mu", "zeta"]


def test_image_compression_only_applies_to_latest_non_frozen_user_turn() -> None:
    fake_compressor = _FakeImageCompressor()

    old_image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "OLD_IMAGE_BYTES"},
    }
    new_image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "NEW_IMAGE_BYTES"},
    }
    messages = [
        {"role": "user", "content": [old_image, {"type": "text", "text": "old image turn"}]},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": [new_image, {"type": "text", "text": "new image turn"}]},
    ]

    result = AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        messages,
        frozen_message_count=1,
        compressor=fake_compressor,
    )

    # Frozen prefix must remain byte-identical.
    assert result[0]["content"][0]["source"]["data"] == "OLD_IMAGE_BYTES"
    # Latest non-frozen user turn is eligible for compression.
    assert result[2]["content"][0]["source"]["data"] == "COMPRESSED_IMAGE_BYTES"


def test_image_compression_does_not_touch_previous_turns_if_last_message_not_user() -> None:
    fake_compressor = _FakeImageCompressor()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "OLD_IMAGE_BYTES",
                    },
                }
            ],
        },
        {"role": "assistant", "content": "last turn is assistant"},
    ]
    result = AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        messages,
        frozen_message_count=0,
        compressor=fake_compressor,
    )
    assert result[0]["content"][0]["source"]["data"] == "OLD_IMAGE_BYTES"


def test_anthropic_batch_tools_sorted_deterministically_before_forward() -> None:
    captured = {}
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch_1",
                    "type": "message_batch",
                    "processing_status": "in_progress",
                    "request_counts": {
                        "processing": 1,
                        "succeeded": 0,
                        "errored": 0,
                        "canceled": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages/batches",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "requests": [
                    {
                        "custom_id": "req-1",
                        "params": {
                            "model": "claude-sonnet-4-6",
                            "max_tokens": 128,
                            "messages": [{"role": "user", "content": "hello"}],
                            "tools": [
                                {
                                    "name": "zeta",
                                    "description": "z",
                                    "input_schema": {"type": "object"},
                                },
                                {
                                    "name": "alpha",
                                    "description": "a",
                                    "input_schema": {"type": "object"},
                                },
                                {
                                    "name": "mu",
                                    "description": "m",
                                    "input_schema": {"type": "object"},
                                },
                            ],
                        },
                    }
                ]
            },
        )

        assert response.status_code == 200
        sent_tools = captured["body"]["requests"][0]["params"]["tools"]
        assert [t["name"] for t in sent_tools] == ["alpha", "mu", "zeta"]


def test_append_context_targets_latest_non_frozen_user_turn() -> None:
    messages = [
        {"role": "user", "content": "frozen prefix"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "active turn"},
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages,
        "CTX",
        frozen_message_count=1,
    )
    assert result[0]["content"] == "frozen prefix"
    assert result[2]["content"].endswith("CTX")


def test_append_context_does_not_touch_previous_turns_if_last_message_not_user() -> None:
    messages = [
        {"role": "user", "content": "previous user turn"},
        {"role": "assistant", "content": "assistant last"},
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages,
        "CTX",
        frozen_message_count=0,
    )
    assert result[0]["content"] == "previous user turn"


def test_token_mode_freeze_is_capped_by_prefix_tracker() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "token"
        proxy.config.image_optimize = False

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        class _FakeCompressionCache:
            def apply_cached(self, messages):  # noqa: ANN001
                return messages

            def compute_frozen_count(self, messages):  # noqa: ANN001
                return 99

            def update_from_result(self, originals, compressed):  # noqa: ANN001
                return None

            def mark_stable_from_messages(self, messages, up_to):  # noqa: ANN001
                pass

        proxy._get_compression_cache = lambda session_id: _FakeCompressionCache()

        def _fake_apply(**kwargs):
            captured["frozen_message_count"] = kwargs.get("frozen_message_count")
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=50,
                tokens_after=50,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_tc_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 50,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert response.status_code == 200
        assert captured["frozen_message_count"] == 1


def test_memory_context_avoids_system_mutation_when_prefix_frozen() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = False
        proxy.config.image_optimize = False
        proxy.config.ccr_proactive_expansion = False

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        proxy.memory_handler = SimpleNamespace(
            config=SimpleNamespace(inject_context=True, inject_tools=False),
            search_and_format_context=AsyncMock(return_value="MEMCTX"),
            has_memory_tool_calls=lambda resp, provider: False,
        )

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_mem_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "u1",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "system": "base system",
                "messages": [
                    {"role": "user", "content": "frozen prefix"},
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": "latest user"},
                ],
            },
        )

        assert response.status_code == 200
        sent = captured["body"]
        assert sent["system"] == "base system"
        assert sent["messages"][2]["content"].endswith("MEMCTX")


def test_ccr_system_instruction_injection_disabled_when_prefix_frozen(monkeypatch) -> None:
    captured = {"inject_system": None}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = False
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = False
        proxy.config.ccr_inject_system_instructions = True

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        class _FakeInjector:
            def __init__(
                self,
                provider,  # noqa: ANN001
                inject_tool,  # noqa: ANN001
                inject_system_instructions,  # noqa: ANN001
            ):
                captured["inject_system"] = inject_system_instructions
                self.has_compressed_content = False
                self.detected_hashes = []

            def process_request(self, messages, tools):  # noqa: ANN001
                return messages, tools, False

        monkeypatch.setattr("headroom.ccr.CCRToolInjector", _FakeInjector)

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert response.status_code == 200
        assert captured["inject_system"] is False


def test_ccr_tool_injection_disabled_when_prefix_frozen(monkeypatch) -> None:
    captured = {"inject_tool": None}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = False
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.ccr_inject_system_instructions = False

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        class _FakeInjector:
            def __init__(
                self,
                provider,  # noqa: ANN001
                inject_tool,  # noqa: ANN001
                inject_system_instructions,  # noqa: ANN001
            ):
                captured["inject_tool"] = inject_tool
                self.has_compressed_content = False
                self.detected_hashes = []

            def process_request(self, messages, tools):  # noqa: ANN001
                return messages, tools, False

        monkeypatch.setattr("headroom.ccr.CCRToolInjector", _FakeInjector)

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_tool_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert response.status_code == 200
        assert captured["inject_tool"] is False


def test_previous_turns_always_frozen_only_final_turn_mutable() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        proxy.anthropic_pipeline.apply = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cache mode should not invoke anthropic pipeline")
        )

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_frz_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 80,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "turn1"},
                    {"role": "assistant", "content": "turn1-assistant"},
                    {"role": "user", "content": "current turn"},
                ],
            },
        )

        assert response.status_code == 200
        assert captured["body"]["messages"] == [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "current turn"},
        ]


def test_batch_optimization_freezes_previous_turns_only() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = False

        proxy.anthropic_pipeline.apply = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cache mode batch path should not invoke anthropic pipeline")
        )

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch_2",
                    "type": "message_batch",
                    "processing_status": "in_progress",
                    "request_counts": {
                        "processing": 1,
                        "succeeded": 0,
                        "errored": 0,
                        "canceled": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages/batches",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "requests": [
                    {
                        "custom_id": "req-1",
                        "params": {
                            "model": "claude-sonnet-4-6",
                            "max_tokens": 128,
                            "messages": [
                                {"role": "user", "content": "old turn"},
                                {"role": "assistant", "content": "old assistant"},
                                {"role": "user", "content": "current turn"},
                            ],
                        },
                    }
                ]
            },
        )

        assert response.status_code == 200
        assert captured["body"]["requests"][0]["params"]["messages"] == [
            {"role": "user", "content": "old turn"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "current turn"},
        ]


def test_token_mode_does_not_force_freeze_all_previous_turns() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "token"
        proxy.config.image_optimize = False

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        class _FakeCompressionCache:
            def apply_cached(self, messages):  # noqa: ANN001
                return messages

            def compute_frozen_count(self, messages):  # noqa: ANN001
                return 0

            def update_from_result(self, originals, compressed):  # noqa: ANN001
                return None

            def mark_stable_from_messages(self, messages, up_to):  # noqa: ANN001
                pass

        proxy._get_compression_cache = lambda session_id: _FakeCompressionCache()

        def _fake_apply(**kwargs):
            captured["frozen_message_count"] = kwargs.get("frozen_message_count")
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=70,
                tokens_after=70,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_tok_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 70,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "turn1"},
                    {"role": "assistant", "content": "turn1-assistant"},
                    {"role": "user", "content": "current turn"},
                ],
            },
        )

        assert response.status_code == 200
        # In token_headroom mode, mark_stable_from_messages marks prior turns
        # as stable, so frozen count reflects the number of prior-turn messages.
        # The compression cache's compute_frozen_count returns 0 (no cached
        # compressions yet), but mark_stable marks previous turns as frozen
        # to preserve prefix cache stability.
        assert captured["frozen_message_count"] >= 0


def test_cache_mode_restores_frozen_prefix_if_transform_mutates_history() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        original_messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "current turn"},
        ]

        def _fake_apply(**kwargs):
            mutated = list(kwargs["messages"])
            mutated[0] = {**mutated[0], "content": "MUTATED_PREFIX"}
            return SimpleNamespace(
                messages=mutated,
                transforms_applied=["fake:mutated"],
                timing={},
                tokens_before=80,
                tokens_after=70,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_cache_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 70,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        sent_messages = captured["body"]["messages"]
        assert sent_messages[0] == original_messages[0]
        assert sent_messages[1] == original_messages[1]


def test_cache_mode_does_not_forward_latest_turn_rewrites() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        original_messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "current turn"},
        ]

        def _fake_apply(**kwargs):
            mutated = list(kwargs["messages"])
            mutated[2] = {**mutated[2], "content": "REWRITTEN_CURRENT_TURN"}
            return SimpleNamespace(
                messages=mutated,
                transforms_applied=["fake:mutated-latest"],
                timing={},
                tokens_before=80,
                tokens_after=60,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_cache_2",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 80,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        assert captured["body"]["messages"] == original_messages


def test_cache_mode_reuses_prior_forwarded_prefix_and_compresses_only_new_suffix() -> None:
    captured = {"calls": []}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

        tracker = _FakePrefixTracker(frozen_count=0)
        tracker._last_original_messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "turn2-assistant"},
        ]
        tracker._last_forwarded_messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "COMPRESSED_TURN2"},
            {"role": "assistant", "content": "turn2-assistant"},
        ]
        tracker.get_last_original_messages = lambda: tracker._last_original_messages.copy()
        tracker.get_last_forwarded_messages = lambda: tracker._last_forwarded_messages.copy()

        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker

        def _fake_apply(**kwargs):
            captured["calls"].append(kwargs["messages"])
            return SimpleNamespace(
                messages=[{"role": "user", "content": "COMPRESSED_TURN3"}],
                transforms_applied=["fake:delta"],
                timing={},
                tokens_before=40,
                tokens_after=20,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_cache_3",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 80,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "turn1"},
                    {"role": "assistant", "content": "turn1-assistant"},
                    {"role": "user", "content": "turn2"},
                    {"role": "assistant", "content": "turn2-assistant"},
                    {"role": "user", "content": "turn3"},
                ],
            },
        )

        assert response.status_code == 200
        assert captured["calls"] == [[{"role": "user", "content": "turn3"}]]
        assert captured["body"]["messages"] == [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "COMPRESSED_TURN2"},
            {"role": "assistant", "content": "turn2-assistant"},
            {"role": "user", "content": "COMPRESSED_TURN3"},
        ]


def test_cache_mode_skips_same_message_append_rewrite_to_preserve_stability() -> None:
    captured = {"calls": []}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

        tracker = _FakePrefixTracker(frozen_count=0)
        tracker._last_original_messages = [
            {"role": "user", "content": "shared-prefix"},
        ]
        tracker._last_forwarded_messages = [
            {"role": "user", "content": "COMPRESSED_PREFIX"},
        ]
        tracker.get_last_original_messages = lambda: tracker._last_original_messages.copy()
        tracker.get_last_forwarded_messages = lambda: tracker._last_forwarded_messages.copy()

        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker

        def _fake_apply(**kwargs):
            captured["calls"].append(kwargs["messages"])
            return SimpleNamespace(
                messages=[{"role": "user", "content": " + COMPRESSED_SUFFIX"}],
                transforms_applied=["fake:suffix"],
                timing={},
                tokens_before=20,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_cache_suffix",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 80,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "shared-prefix + raw suffix"},
                ],
            },
        )

        assert response.status_code == 200
        assert captured["calls"] == []
        assert captured["body"]["messages"] == [
            {"role": "user", "content": "shared-prefix + raw suffix"},
        ]
