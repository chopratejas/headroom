import base64
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anyio
import pytest
from fastapi import Request

from headroom.proxy.handlers.openai import (
    OpenAIHandlerMixin,
    _resolve_codex_routing_headers,
)


def _jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def encode(part: dict) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}."


def test_resolve_codex_routing_prefers_explicit_header():
    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "Authorization": "Bearer sk-test",
            "ChatGPT-Account-ID": "acct-explicit",
        }
    )

    assert is_chatgpt is True
    assert headers["ChatGPT-Account-ID"] == "acct-explicit"


def test_resolve_codex_routing_derives_account_id_from_oauth_jwt():
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )

    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "authorization": f"Bearer {token}",
        }
    )

    assert is_chatgpt is True
    assert headers["ChatGPT-Account-ID"] == "acct-from-jwt"


def test_resolve_codex_routing_leaves_regular_openai_bearer_tokens_unchanged():
    token = _jwt({"aud": ["https://api.openai.com/v1"]})

    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "authorization": f"Bearer {token}",
        }
    )

    assert is_chatgpt is False
    assert "ChatGPT-Account-ID" not in headers


def test_resolve_codex_routing_returns_none_without_bearer_auth():
    headers, is_chatgpt = _resolve_codex_routing_headers({})

    assert is_chatgpt is False
    assert headers == {}


def test_resolve_codex_routing_ignores_non_jwt_bearer_tokens():
    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "authorization": "Bearer not-a-jwt",
        }
    )

    assert is_chatgpt is False
    assert headers["authorization"] == "Bearer not-a-jwt"


def test_resolve_codex_routing_ignores_invalid_jwt_payloads():
    invalid_payload = base64.urlsafe_b64encode(b"not-json").decode("ascii").rstrip("=")
    token = f"test-header.{invalid_payload}.signature"

    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "authorization": f"Bearer {token}",
        }
    )

    assert is_chatgpt is False
    assert headers["authorization"] == f"Bearer {token}"


class _DummyMetrics:
    def __init__(self) -> None:
        self.failed_calls = 0
        self.rate_limited_calls = 0

    async def record_request(self, **kwargs):  # noqa: ANN003
        return None

    async def record_failed(self, **kwargs):  # noqa: ANN003
        self.failed_calls += 1
        return None

    async def record_rate_limited(self, **kwargs):  # noqa: ANN003
        self.rate_limited_calls += 1
        return None


class _DummyTokenizer:
    def count_messages(self, messages):
        return len(messages)


class _DummyPipelineExtensions:
    def emit(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return SimpleNamespace(messages=None, tools=None, headers=None)


class _DummyPrefixTracker:
    def __init__(self) -> None:
        self.updated_with = None

    def get_frozen_message_count(self) -> int:
        return 0

    def update_from_response(self, **kwargs) -> None:  # noqa: ANN003
        self.updated_with = kwargs


class _ResponseStub:
    def __init__(
        self,
        body: dict | None = None,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json", "content-length": "42"}
        self._body = body or {
            "id": "resp_123",
            "output": [{"type": "message"}],
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        self.content = json.dumps(self._body).encode("utf-8")

    def json(self):
        return self._body


class _DummyOpenAIHandler(OpenAIHandlerMixin):
    OPENAI_API_URL = "https://api.openai.com"

    def __init__(self) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = SimpleNamespace(
            optimize=False,
            image_optimize=False,
            hooks=None,
            mode="optimize",
            ccr_inject_tool=False,
            ccr_inject_system_instructions=False,
            retry_max_attempts=3,
            retry_base_delay_ms=10,
            retry_max_delay_ms=50,
            connect_timeout_seconds=10,
        )
        self.usage_reporter = None
        self.openai_provider = SimpleNamespace(get_context_limit=lambda model: 128_000)
        self.openai_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.cache = None
        self.pipeline_extensions = _DummyPipelineExtensions()
        self.session_prefix_tracker = _DummyPrefixTracker()
        self.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda request, model, messages: "sess-1",
            get_or_create=lambda session_id, provider: self.session_prefix_tracker,
        )
        self.response_queue: list[_ResponseStub] = []
        self.captured_request: tuple[str, str, dict, dict] | None = None
        self.captured_requests: list[tuple[str, str, dict, dict]] = []
        self.captured_stream_request: tuple[str, dict, dict] | None = None

    async def _next_request_id(self) -> str:
        return "req-1"

    def _extract_tags(self, headers: dict[str, str]) -> dict[str, str]:
        return {}

    async def _retry_request(self, method: str, url: str, headers: dict, body: dict):
        self.captured_request = (method, url, headers, body)
        self.captured_requests.append((method, url, headers, body))
        if self.response_queue:
            return self.response_queue.pop(0)
        return _ResponseStub()

    async def _stream_response(
        self,
        url: str,
        headers: dict,
        body: dict,
        provider: str,
        model: str,
        request_id: str,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: dict[str, str],
        optimization_latency: float,
        memory_user_id: str | None = None,
        **kwargs,
    ):
        self.captured_stream_request = (url, headers, body)
        return SimpleNamespace(
            status_code=200,
            url=url,
            headers=headers,
            body=body,
            memory_user_id=memory_user_id,
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
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


def _build_chat_request(body: dict, headers: dict[str, str]) -> Request:
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


def test_handle_openai_responses_routes_chatgpt_auth_to_backend_api(monkeypatch):
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )
    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": f"Bearer {token}"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = SimpleNamespace(
        config=SimpleNamespace(inject_context=False, inject_tools=False),
        has_memory_tool_calls=lambda response, provider: False,
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert handler.captured_request is not None
    method, url, headers, body = handler.captured_request
    assert method == "POST"
    assert url == "https://chatgpt.com/backend-api/codex/responses"
    assert headers["ChatGPT-Account-ID"] == "acct-from-jwt"
    assert body["input"] == "hello"
    assert response.status_code == 200


def test_handle_openai_responses_stream_keeps_compression(monkeypatch):
    request = _build_request(
        {
            "model": "gpt-5.4",
            "stream": True,
            "instructions": "Keep it short",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler.openai_pipeline.apply.return_value = SimpleNamespace(
        messages=[
            {"role": "system", "content": "Keep it short"},
            {"role": "user", "content": "hello"},
        ],
        transforms_applied=[],
        tokens_before=2,
        tokens_after=2,
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_stream_request is not None
    assert handler.openai_pipeline.apply.call_count == 1
    assert handler.captured_stream_request[2]["stream"] is True


def test_handle_openai_responses_memory_timeout_fails_open(monkeypatch):
    class _SlowMemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=True, inject_tools=False)

        async def search_and_format_context(self, memory_user_id, messages):
            return "should not be used"

        def has_memory_tool_calls(self, response, provider):
            return False

    async def _timeout_wait_for(awaitable, timeout):
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise TimeoutError

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _SlowMemoryHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.asyncio.wait_for", _timeout_wait_for)

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body.get("instructions") is None


def test_handle_openai_responses_appends_memory_context_and_converts_tools(monkeypatch):
    class _MemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=True, inject_tools=True)

        async def search_and_format_context(self, memory_user_id, messages):
            assert memory_user_id == "user-1"
            assert messages[0]["role"] == "system"
            return "Remember prior project details."

        def inject_tools(self, tools, provider):
            assert provider == "openai"
            return (
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "memory_search",
                            "description": "Search memory",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                True,
            )

        def has_memory_tool_calls(self, response, provider):
            return False

    request = _build_request(
        {
            "model": "gpt-5.4",
            "instructions": "Keep it short",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body["instructions"] == "Keep it short\n\nRemember prior project details."
    assert body["tools"] == [
        {
            "type": "function",
            "name": "memory_search",
            "description": "Search memory",
            "parameters": {"type": "object"},
        }
    ]


def test_handle_openai_responses_converts_optimized_system_message_back_to_instructions(
    monkeypatch,
):
    request = _build_request(
        {
            "model": "gpt-5.4",
            "instructions": "Old instructions",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler.openai_pipeline.apply.return_value = SimpleNamespace(
        messages=[
            {"role": "system", "content": "New instructions"},
            {"role": "user", "content": "hello"},
        ],
        transforms_applied=["content_router"],
        tokens_before=2,
        tokens_after=1,
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.openai_pipeline.apply.call_count == 1
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body["instructions"] == "New instructions"
    assert isinstance(body["input"], list)
    assert body["input"][0]["type"] == "message"


def test_handle_openai_responses_regular_auth_uses_openai_upstream_url(monkeypatch):
    seen = {}

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        seen["url"] = url
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert seen["url"] == "https://api.openai.com/v1/responses"
    assert handler.captured_request is not None
    _, url, headers, _ = handler.captured_request
    assert url == "https://api.openai.com/v1/responses"
    assert headers["authorization"] == "Bearer upstream"


def test_handle_openai_responses_continues_after_memory_tool_calls(monkeypatch):
    class _MemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=False, inject_tools=False)
            self._backend = object()
            self.initialized = False

        def inject_tools(self, tools, provider):
            return tools, False

        def has_memory_tool_calls(self, response, provider):
            return True

        async def _ensure_initialized(self):
            self.initialized = True

        async def _execute_memory_tool(self, name, args, memory_user_id, provider):
            assert name == "memory_search"
            assert args == {"query": "hello"}
            assert memory_user_id == "user-1"
            assert provider == "openai"
            return json.dumps({"result": "found"})

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()
    handler.response_queue = [
        _ResponseStub(
            {
                "id": "resp-first",
                "usage": {"input_tokens": 2, "output_tokens": 0},
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-1",
                        "call_id": "call-1",
                        "name": "memory_search",
                        "arguments": json.dumps({"query": "hello"}),
                    }
                ],
            }
        ),
        _ResponseStub(
            {
                "id": "resp-second",
                "usage": {"input_tokens": 3, "output_tokens": 1},
                "output": [{"type": "message", "id": "msg-1"}],
            }
        ),
    ]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr(
        "headroom.proxy.memory_handler.MEMORY_TOOL_NAMES",
        {"memory_search"},
        raising=False,
    )

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert len(handler.captured_requests) == 2
    _, _, _, continuation_body = handler.captured_requests[1]
    assert continuation_body["previous_response_id"] == "resp-first"
    assert continuation_body["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": json.dumps({"result": "found"}),
        }
    ]
    assert handler.memory_handler.initialized is True


def test_handle_openai_responses_reverts_inflated_optimization_result(monkeypatch):
    request = _build_request(
        {
            "model": "gpt-5.4",
            "instructions": "Original instructions",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler.openai_pipeline.apply.return_value = SimpleNamespace(
        messages=[
            {"role": "system", "content": "Inflated instructions"},
            {"role": "user", "content": "hello hello hello"},
        ],
        transforms_applied=["tool_crusher"],
        tokens_before=2,
        tokens_after=5,
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body["instructions"] == "Original instructions"
    assert body["input"][0]["content"][0]["text"] == "hello"


def test_handle_openai_responses_returns_sanitized_error_on_upstream_failure(monkeypatch):
    async def raise_request(method: str, url: str, headers: dict, body: dict):
        raise RuntimeError("boom")

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler._retry_request = raise_request  # type: ignore[method-assign]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 502
    assert handler.metrics.failed_calls == 1
    assert json.loads(response.body)["error"]["code"] == "proxy_error"


class _DummyWebSocket:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers
        self.accepted_subprotocol = None

    async def accept(self, subprotocol=None):
        self.accepted_subprotocol = subprotocol


def test_handle_openai_responses_ws_resolves_codex_routing_headers():
    class SentinelError(RuntimeError):
        pass

    handler = _DummyOpenAIHandler()
    websocket = _DummyWebSocket({"authorization": "Bearer token"})

    with patch.dict(sys.modules, {"websockets": MagicMock()}):
        with patch(
            "headroom.proxy.handlers.openai._resolve_codex_routing_headers",
            side_effect=SentinelError("resolved"),
        ):
            with pytest.raises(SentinelError, match="resolved"):
                anyio.run(handler.handle_openai_responses_ws, websocket)


def test_handle_openai_chat_rate_limit_raises_http_exception(monkeypatch):
    class _RateLimiter:
        async def check_request(self, rate_key):
            return False, 2.5

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.rate_limiter = _RateLimiter()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    with pytest.raises(Exception) as exc_info:
        anyio.run(handler.handle_openai_chat, request)

    assert exc_info.value.status_code == 429
    assert handler.metrics.rate_limited_calls == 1


def test_handle_openai_chat_returns_cached_response_without_compression_headers(monkeypatch):
    class _Cache:
        async def get(self, messages, model):
            return SimpleNamespace(
                response_body=b'{"id":"cached"}',
                response_headers={
                    "content-type": "application/json",
                    "content-encoding": "gzip",
                    "content-length": "999",
                    "x-test": "ok",
                },
            )

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.cache = _Cache()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert response.headers["x-test"] == "ok"
    assert "content-encoding" not in response.headers
    assert response.headers["content-length"] != "999"


def test_handle_openai_chat_reverts_inflated_optimization_result(monkeypatch):
    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler.openai_pipeline.apply.return_value = SimpleNamespace(
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "extra"},
        ],
        transforms_applied=["tool_crusher"],
        timing={"router": 1.0},
        waste_signals=None,
        tokens_before=1,
        tokens_after=5,
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_handle_openai_chat_injects_memory_context_and_tools(monkeypatch):
    class _MemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=True, inject_tools=True)

        async def search_and_format_context(self, memory_user_id, messages):
            assert memory_user_id == "user-1"
            return "remember this"

        def inject_tools(self, tools, provider):
            assert provider == "openai"
            return ([{"type": "function", "function": {"name": "memory_search"}}], True)

        def has_memory_tool_calls(self, response, provider):
            return False

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body["messages"][0] == {"role": "system", "content": "remember this"}
    assert body["tools"] == [{"type": "function", "function": {"name": "memory_search"}}]


def test_handle_openai_chat_returns_sanitized_error_on_retry_failure(monkeypatch):
    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return headers

    async def raise_request(method: str, url: str, headers: dict, body: dict):
        raise RuntimeError("boom")

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler._retry_request = raise_request  # type: ignore[method-assign]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 502
    assert handler.metrics.failed_calls == 1
    assert json.loads(response.body)["error"]["code"] == "proxy_error"


def test_handle_openai_chat_allows_pipeline_to_override_memory_and_presend_payload(monkeypatch):
    class _MemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=True, inject_tools=True)

        async def search_and_format_context(self, memory_user_id, messages):
            return "remember this"

        def inject_tools(self, tools, provider):
            return ([{"type": "function", "function": {"name": "memory_search"}}], True)

        def has_memory_tool_calls(self, response, provider):
            return False

    class _PipelineExtensions:
        def emit(self, stage, **kwargs):  # noqa: ANN003, ANN201
            stage_name = getattr(stage, "name", "")
            if stage_name == "INPUT_REMEMBERED":
                return SimpleNamespace(
                    messages=[{"role": "system", "content": "remembered"}],
                    tools=[{"type": "function", "function": {"name": "remembered_tool"}}],
                    headers=None,
                )
            if stage_name == "PRE_SEND":
                return SimpleNamespace(
                    messages=[{"role": "system", "content": "presend"}],
                    tools=[{"type": "function", "function": {"name": "presend_tool"}}],
                    headers={**kwargs["headers"], "x-pre-send": "1"},
                )
            return SimpleNamespace(messages=None, tools=None, headers=None)

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()
    handler.pipeline_extensions = _PipelineExtensions()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, headers, body = handler.captured_request
    assert body["messages"] == [{"role": "system", "content": "presend"}]
    assert body["tools"] == [{"type": "function", "function": {"name": "presend_tool"}}]
    assert headers["x-pre-send"] == "1"


def test_handle_openai_chat_handles_hooks_ccr_and_cache_restore(monkeypatch):
    hook_calls: list[object] = []

    class _Hooks:
        def post_compress(self, event) -> None:  # noqa: ANN001
            hook_calls.append(event)
            raise RuntimeError("hook failed")

    class _PipelineExtensions:
        def emit(self, stage, **kwargs):  # noqa: ANN003, ANN201
            stage_name = getattr(stage, "name", "")
            if stage_name == "INPUT_ROUTED":
                return SimpleNamespace(
                    messages=[{"role": "assistant", "content": "routed"}],
                    tools=None,
                    headers=None,
                )
            if stage_name == "INPUT_COMPRESSED":
                return SimpleNamespace(
                    messages=[{"role": "assistant", "content": "compressed"}],
                    tools=None,
                    headers=None,
                )
            if stage_name == "PRE_SEND":
                return SimpleNamespace(
                    messages=None,
                    tools=[{"type": "function", "function": {"name": "pre_send_tool"}}],
                    headers={**kwargs["headers"], "x-pre-send": "1"},
                )
            return SimpleNamespace(messages=None, tools=None, headers=None)

    class _PrefixTracker:
        def get_frozen_message_count(self) -> int:
            return 1

        def update_from_response(self, **kwargs) -> None:  # noqa: ANN003
            return None

    class _CCRToolInjector:
        def __init__(
            self, provider: str, inject_tool: bool, inject_system_instructions: bool
        ) -> None:
            self.has_compressed_content = True
            self.detected_hashes = ["hash-1"]

        def process_request(self, messages, tools):  # noqa: ANN001, ANN201
            return messages, [{"type": "function", "function": {"name": "ccr_tool"}}], True

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_chat_request(
        {
            "model": "gpt-5.4",
            "messages": [
                {"role": "system", "content": "original system"},
                {"role": "user", "content": "hello"},
            ],
        },
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler.config.mode = "cache"
    handler.config.ccr_inject_tool = True
    handler.config.hooks = _Hooks()
    handler.pipeline_extensions = _PipelineExtensions()
    handler.session_prefix_tracker = _PrefixTracker()
    handler.openai_pipeline.apply.return_value = SimpleNamespace(
        messages=[{"role": "assistant", "content": "optimized"}],
        transforms_applied=["tool_crusher"],
        timing={"router": 1.0},
        waste_signals=None,
        tokens_before=2,
        tokens_after=1,
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)
    monkeypatch.setattr("headroom.ccr.CCRToolInjector", _CCRToolInjector)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert len(hook_calls) == 1
    assert handler.captured_request is not None
    _, _, headers, body = handler.captured_request
    assert body["messages"] == [{"role": "system", "content": "original system"}]
    assert body["tools"] == [{"type": "function", "function": {"name": "pre_send_tool"}}]
    assert headers["x-pre-send"] == "1"


def test_handle_openai_chat_fails_open_when_memory_injection_raises(monkeypatch):
    class _BrokenMemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=True, inject_tools=True)

        async def search_and_format_context(self, memory_user_id, messages):
            raise RuntimeError("memory boom")

        def inject_tools(self, tools, provider):
            raise AssertionError("should not inject tools after context failure")

        def has_memory_tool_calls(self, response, provider):
            return False

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _BrokenMemoryHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in body


def test_handle_openai_chat_routes_non_streaming_requests_via_backend(monkeypatch):
    class _PipelineExtensions:
        def __init__(self) -> None:
            self.stages: list[str] = []

        def emit(self, stage, **kwargs):  # noqa: ANN003, ANN201
            self.stages.append(getattr(stage, "name", ""))
            return SimpleNamespace(messages=None, tools=None, headers=None)

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    async def send_openai_message(body: dict, headers: dict):  # noqa: ANN001, ANN201
        return SimpleNamespace(
            status_code=202,
            body={
                "id": "backend-response",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            error=False,
        )

    recorded: list[dict[str, object]] = []

    async def record_request(**kwargs):  # noqa: ANN003
        recorded.append(kwargs)

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.pipeline_extensions = _PipelineExtensions()
    handler.anthropic_backend = SimpleNamespace(
        send_openai_message=send_openai_message,
        name="anyllm",
    )
    handler.metrics.record_request = record_request

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 202
    assert json.loads(response.body)["id"] == "backend-response"
    assert handler.pipeline_extensions.stages.count("POST_SEND") == 1
    assert handler.pipeline_extensions.stages.count("RESPONSE_RECEIVED") == 1
    assert recorded[0]["provider"] == "anyllm"
    assert recorded[0]["input_tokens"] == 3
    assert recorded[0]["output_tokens"] == 2


def test_handle_openai_chat_returns_backend_errors_verbatim(monkeypatch):
    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    async def send_openai_message(body: dict, headers: dict):  # noqa: ANN001, ANN201
        return SimpleNamespace(
            status_code=503,
            body={"error": {"message": "backend unavailable"}},
            error=True,
        )

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.anthropic_backend = SimpleNamespace(
        send_openai_message=send_openai_message,
        name="anyllm",
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 503
    assert json.loads(response.body) == {"error": {"message": "backend unavailable"}}


def test_handle_openai_chat_routes_streaming_requests_via_backend(monkeypatch):
    class _PipelineExtensions:
        def __init__(self) -> None:
            self.stages: list[str] = []

        def emit(self, stage, **kwargs):  # noqa: ANN003, ANN201
            self.stages.append(getattr(stage, "name", ""))
            return SimpleNamespace(messages=None, tools=None, headers=None)

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    async def fake_stream_backend(
        body: dict,
        headers: dict,
        model: str,
        request_id: str,
        start_time: float,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: dict[str, str],
        optimization_latency: float,
        pipeline_timing=None,
    ):
        return SimpleNamespace(
            status_code=200,
            model=model,
            body=body,
            headers=headers,
            pipeline_timing=pipeline_timing,
        )

    request = _build_chat_request(
        {"model": "gpt-5.4", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.pipeline_extensions = _PipelineExtensions()
    handler.anthropic_backend = SimpleNamespace(name="anyllm")
    handler._stream_openai_via_backend = fake_stream_backend  # type: ignore[method-assign]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert response.model == "gpt-5.4"
    assert response.body["stream"] is True
    assert handler.pipeline_extensions.stages.count("POST_SEND") == 1


def test_handle_openai_chat_returns_backend_error_payload_on_exception(monkeypatch):
    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    async def send_openai_message(body: dict, headers: dict):  # noqa: ANN001, ANN201
        raise RuntimeError("backend boom")

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.anthropic_backend = SimpleNamespace(
        send_openai_message=send_openai_message,
        name="anyllm",
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 500
    payload = json.loads(response.body)
    assert payload["error"]["code"] == "backend_error"
    assert payload["error"]["message"] == "backend boom"


def test_handle_openai_chat_streaming_adds_usage_stream_options(monkeypatch):
    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_chat_request(
        {"model": "gpt-5.4", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert handler.captured_stream_request is not None
    _, _, body = handler.captured_stream_request
    assert body["stream_options"] == {"include_usage": True}


def test_handle_openai_chat_streaming_updates_existing_stream_options(monkeypatch):
    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_chat_request(
        {
            "model": "gpt-5.4",
            "stream": True,
            "stream_options": {"other": True},
            "messages": [{"role": "user", "content": "hello"}],
        },
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert handler.captured_stream_request is not None
    _, _, body = handler.captured_stream_request
    assert body["stream_options"] == {"other": True, "include_usage": True}


def test_handle_openai_chat_continues_after_memory_tool_calls(monkeypatch):
    class _MemoryHandler:
        def has_memory_tool_calls(self, response, provider):
            return True

        async def handle_memory_tool_calls(self, response, memory_user_id, provider):
            assert memory_user_id == "user-1"
            assert provider == "openai"
            return [{"role": "tool", "tool_call_id": "call-1", "content": "found"}]

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()
    handler.response_queue = [
        _ResponseStub(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": "call-1", "type": "function"}],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        ),
        _ResponseStub(
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        ),
    ]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert len(handler.captured_requests) == 2
    _, _, _, continuation_body = handler.captured_requests[1]
    assert continuation_body["messages"] == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "found"},
    ]


def test_handle_openai_responses_sets_instructions_when_memory_context_has_no_existing_system(
    monkeypatch,
):
    class _MemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=True, inject_tools=False)

        async def search_and_format_context(self, memory_user_id, messages):
            return "remember without instructions"

        def has_memory_tool_calls(self, response, provider):
            return False

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body["instructions"] == "remember without instructions"


def test_handle_openai_responses_preserves_non_function_memory_tools(monkeypatch):
    class _MemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=False, inject_tools=True)

        def inject_tools(self, tools, provider):
            return ([{"type": "web_search_preview"}], True)

        def has_memory_tool_calls(self, response, provider):
            return False

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body["tools"] == [{"type": "web_search_preview"}]


def test_handle_openai_responses_warns_on_memory_injection_failure_and_ignores_backend_override(
    monkeypatch,
):
    class _BrokenMemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=True, inject_tools=False)

        async def search_and_format_context(self, memory_user_id, messages):
            raise RuntimeError("responses memory boom")

        def has_memory_tool_calls(self, response, provider):
            return False

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _BrokenMemoryHandler()
    handler.anthropic_backend = SimpleNamespace(name="anyllm")

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, url, _, body = handler.captured_request
    assert url == "https://api.openai.com/v1/responses"
    assert body.get("instructions") is None


def test_handle_openai_chat_records_costs_and_caches_successful_response(monkeypatch):
    class _Cache:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        async def get(self, messages, model):
            return None

        async def set(self, messages, model, content, headers, tokens_saved):  # noqa: ANN001
            self.calls.append((messages, model, content, headers, tokens_saved))

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    cost_calls: list[tuple] = []
    metric_calls: list[dict[str, object]] = []

    async def record_request(**kwargs):  # noqa: ANN003
        metric_calls.append(kwargs)

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.cache = _Cache()
    handler.cost_tracker = SimpleNamespace(
        record_tokens=lambda *args, **kwargs: cost_calls.append((args, kwargs))
    )
    handler.metrics.record_request = record_request
    handler.response_queue = [
        _ResponseStub(
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            }
        )
    ]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert cost_calls == [(("gpt-5.4", 0, 1), {"cache_read_tokens": 3, "uncached_tokens": 7})]
    assert handler.cache.calls and handler.cache.calls[0][1] == "gpt-5.4"
    assert metric_calls[0]["cache_read_tokens"] == 3
    assert metric_calls[0]["uncached_input_tokens"] == 7


def test_handle_openai_chat_fails_open_when_memory_tool_handling_raises(monkeypatch):
    class _MemoryHandler:
        def has_memory_tool_calls(self, response, provider):
            return True

        async def handle_memory_tool_calls(self, response, memory_user_id, provider):
            raise RuntimeError("tool boom")

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "authorization": "Bearer upstream"}

    request = _build_chat_request(
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()
    handler.response_queue = [
        _ResponseStub(
            {
                "choices": [{"message": {"role": "assistant", "content": None}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        )
    ]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.apply_copilot_api_auth", fake_apply)

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert len(handler.captured_requests) == 1


def test_handle_openai_responses_stream_passes_memory_user_id(monkeypatch):
    request = _build_request(
        {
            "model": "gpt-5.4",
            "stream": True,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_stream_request is not None
    assert handler.captured_stream_request[1]["x-headroom-user-id"] == "user-1"


def test_handle_openai_responses_records_costs_metrics_and_strips_encoding_headers(monkeypatch):
    metric_calls: list[dict[str, object]] = []
    cost_calls: list[tuple] = []

    async def record_request(**kwargs):  # noqa: ANN003
        metric_calls.append(kwargs)

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.metrics.record_request = record_request
    handler.cost_tracker = SimpleNamespace(record_tokens=lambda *args: cost_calls.append(args))
    handler.response_queue = [
        _ResponseStub(
            {
                "id": "resp-1",
                "usage": {"input_tokens": 9, "output_tokens": 4},
                "output": [{"type": "message"}],
            },
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
                "content-length": "999",
                "x-test": "ok",
            },
        )
    ]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert response.headers["x-test"] == "ok"
    assert "content-encoding" not in response.headers
    assert cost_calls == [("gpt-5.4", 0, 9)]
    assert metric_calls[0]["input_tokens"] == 9
    assert metric_calls[0]["output_tokens"] == 4


def test_handle_openai_responses_continues_memory_calls_with_error_payload_when_backend_missing(
    monkeypatch,
):
    class _MemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=False, inject_tools=False)
            self._backend = None
            self.initialized = False

        def inject_tools(self, tools, provider):
            return tools, False

        def has_memory_tool_calls(self, response, provider):
            return True

        async def _ensure_initialized(self):
            self.initialized = True

    request = _build_request(
        {
            "model": "gpt-5.4",
            "input": "hello",
            "tools": [{"type": "function", "name": "memory_search"}],
        },
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()
    handler.response_queue = [
        _ResponseStub(
            {
                "id": "resp-first",
                "usage": {"input_tokens": 2, "output_tokens": 0},
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-1",
                        "name": "memory_search",
                        "arguments": "{bad json",
                    }
                ],
            }
        ),
        _ResponseStub(
            {
                "id": "resp-second",
                "usage": {"input_tokens": 3, "output_tokens": 1},
                "output": [{"type": "message", "id": "msg-1"}],
            }
        ),
    ]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr(
        "headroom.proxy.memory_handler.MEMORY_TOOL_NAMES",
        {"memory_search"},
        raising=False,
    )

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert len(handler.captured_requests) == 2
    _, _, _, continuation_body = handler.captured_requests[1]
    assert continuation_body["previous_response_id"] == "resp-first"
    assert continuation_body["tools"] == [{"type": "function", "name": "memory_search"}]
    assert continuation_body["input"] == [
        {
            "type": "function_call_output",
            "call_id": "fc-1",
            "output": json.dumps({"error": "Memory backend not initialized"}),
        }
    ]
    assert handler.memory_handler.initialized is True


def test_handle_openai_responses_fails_open_when_memory_tool_handling_raises(monkeypatch):
    class _MemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=False, inject_tools=False)

        def inject_tools(self, tools, provider):
            return tools, False

        def has_memory_tool_calls(self, response, provider):
            return True

        async def _ensure_initialized(self):
            return None

        async def _execute_memory_tool(self, name, args, memory_user_id, provider):
            raise RuntimeError("tool failed")

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _MemoryHandler()
    handler.response_queue = [
        _ResponseStub(
            {
                "id": "resp-first",
                "usage": {"input_tokens": 2, "output_tokens": 0},
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "memory_search",
                        "arguments": json.dumps({"query": "hello"}),
                    }
                ],
            }
        )
    ]

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr(
        "headroom.proxy.memory_handler.MEMORY_TOOL_NAMES",
        {"memory_search"},
        raising=False,
    )

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert len(handler.captured_requests) == 1
