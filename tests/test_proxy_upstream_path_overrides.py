from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


class _DummyTokenizer:
    def count_messages(self, messages):
        return len(messages)


def _config(**overrides) -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        openai_api_url="https://openai-upstream.test",
        anthropic_api_url="https://anthropic-upstream.test",
        **overrides,
    )


def test_proxy_config_normalizes_endpoint_path_overrides() -> None:
    config = _config(
        anthropic_messages_path="anthropic/messages",
        openai_chat_path="litellm/chat/completions",
        openai_responses_path="/custom/responses",
    )

    assert config.anthropic_messages_path == "/anthropic/messages"
    assert config.openai_chat_path == "/litellm/chat/completions"
    assert config.openai_responses_path == "/custom/responses"


def test_proxy_config_rejects_full_url_endpoint_path_overrides() -> None:
    with pytest.raises(ValueError, match="paths, not full URLs"):
        _config(openai_chat_path="https://proxy.example/v1/chat/completions")


def test_proxy_config_from_env_reads_endpoint_path_overrides(monkeypatch) -> None:
    from headroom.proxy.server import _MULTI_WORKER_CONFIG_ENV, _proxy_config_from_env

    monkeypatch.delenv(_MULTI_WORKER_CONFIG_ENV, raising=False)
    monkeypatch.setenv("HEADROOM_ANTHROPIC_MESSAGES_PATH", "anthropic/messages")
    monkeypatch.setenv("HEADROOM_OPENAI_CHAT_PATH", "/litellm/chat/completions")
    monkeypatch.setenv("HEADROOM_OPENAI_RESPONSES_PATH", "/litellm/responses")

    config = _proxy_config_from_env()

    assert config.anthropic_messages_path == "/anthropic/messages"
    assert config.openai_chat_path == "/litellm/chat/completions"
    assert config.openai_responses_path == "/litellm/responses"


def test_openai_chat_forwards_to_configured_upstream_path(monkeypatch) -> None:
    seen: dict[str, str] = {}
    app = create_app(_config(openai_chat_path="/litellm/chat/completions"))

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            seen["url"] = url
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
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert seen["url"] == "https://openai-upstream.test/litellm/chat/completions"


def test_openai_responses_forwards_to_configured_upstream_path(monkeypatch) -> None:
    seen: dict[str, str] = {}
    app = create_app(_config(openai_responses_path="/litellm/responses"))

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            seen["url"] = url
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test"},
            json={"model": "gpt-4o-mini", "input": "hello"},
        )

    assert response.status_code == 200
    assert seen["url"] == "https://openai-upstream.test/litellm/responses"


def test_anthropic_messages_forwards_to_configured_upstream_path(monkeypatch) -> None:
    seen: dict[str, str] = {}
    app = create_app(_config(anthropic_messages_path="/proxy/anthropic/messages"))

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            seen["url"] = url
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert seen["url"] == "https://anthropic-upstream.test/proxy/anthropic/messages"
