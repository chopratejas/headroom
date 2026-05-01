from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from headroom.proxy.handlers import gemini as gemini_module


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        content: bytes = b"{}",
        headers: dict[str, str] | None = None,
        json_data=None,  # noqa: ANN001
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self._json_data = json_data

    def json(self):  # noqa: ANN201
        if self._json_data is not None:
            return self._json_data
        return json.loads(self.content.decode("utf-8"))


class FakeRequest:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        query: str = "",
    ) -> None:
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.url = SimpleNamespace(query=query)

    async def body(self) -> bytes:
        return b"{}"


class FakeMetrics:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.failures: list[dict[str, object]] = []
        self.rate_limited: list[dict[str, object]] = []

    async def record_request(self, **kwargs) -> None:  # noqa: ANN003
        self.requests.append(kwargs)

    async def record_failed(self, **kwargs) -> None:  # noqa: ANN003
        self.failures.append(kwargs)

    async def record_rate_limited(self, **kwargs) -> None:  # noqa: ANN003
        self.rate_limited.append(kwargs)


class DummyGeminiHandler(gemini_module.GeminiHandlerMixin):
    GEMINI_API_URL = "https://gemini.example"
    CLOUDCODE_API_URL = "https://cloudcode.example/"

    def __init__(self) -> None:
        self.config = SimpleNamespace(optimize=True)
        self.metrics = FakeMetrics()
        self.cost_tracker = SimpleNamespace(calls=[])
        self.cost_tracker.record_tokens = lambda *args, **kwargs: self.cost_tracker.calls.append(
            {"args": args, "kwargs": kwargs}
        )
        self.usage_reporter = SimpleNamespace(should_compress=True)
        self.openai_provider = SimpleNamespace(get_context_limit=lambda model: 4096)
        self.openai_pipeline = SimpleNamespace(
            apply=lambda **kwargs: SimpleNamespace(
                messages=[{"role": "user", "content": "optimized user"}],
                transforms_applied=["trim"],
                tokens_before=40,
                tokens_after=20,
                waste_signals=SimpleNamespace(to_dict=lambda: {"verbosity": 1}),
            )
        )
        self.memory_handler = None
        self.rate_limiter = None
        self._request_id = 0
        self.retry_calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.retry_response = FakeResponse()
        self.stream_response = {"streamed": True}

    async def _next_request_id(self) -> str:
        self._request_id += 1
        return f"req-{self._request_id}"

    async def _retry_request(self, method, url, headers, body):  # noqa: ANN001, ANN201
        self.retry_calls.append({"method": method, "url": url, "headers": headers, "body": body})
        if isinstance(self.retry_response, Exception):
            raise self.retry_response
        return self.retry_response

    async def _stream_response(self, *args):  # noqa: ANN002, ANN201
        self.stream_calls.append({"args": args})
        return self.stream_response

    def _extract_tags(self, headers):  # noqa: ANN001, ANN201
        return {"tagged": headers.get("x-tag", "none")}


class FakeTokenizer:
    def count_messages(self, messages) -> int:  # noqa: ANN001
        return sum(len(message.get("content", "")) for message in messages)

    def count_text(self, text: str) -> int:
        return len(text)


def install_gemini_modules(monkeypatch: pytest.MonkeyPatch, payload) -> None:  # noqa: ANN001
    async def read_request_json(request):  # noqa: ANN001
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", read_request_json)
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: FakeTokenizer())
    monkeypatch.setattr("headroom.utils.extract_user_query", lambda messages: "question")


@pytest.mark.asyncio
async def test_handle_gemini_generate_content_covers_success_passthrough_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = DummyGeminiHandler()

    assert handler._is_cloudcode_antigravity_request({"requestType": "agent"}, {}) is True
    assert handler._is_cloudcode_antigravity_request({"userAgent": "antigravity"}, {}) is True
    assert handler._is_cloudcode_antigravity_request({}, {"user-agent": "antigravity/1.0"}) is True
    assert handler._resolve_cloudcode_base_url(False) == "https://cloudcode.example"
    assert handler._resolve_cloudcode_base_url(True) == gemini_module.ANTIGRAVITY_DAILY_API_URL

    oversized = await handler.handle_gemini_generate_content(
        FakeRequest(headers={"content-length": "9999999999"}),
        "gemini-pro",
    )
    assert oversized.status_code == 413

    install_gemini_modules(monkeypatch, ValueError("bad json"))
    invalid = await handler.handle_gemini_generate_content(FakeRequest(), "gemini-pro")
    assert invalid.status_code == 400

    install_gemini_modules(
        monkeypatch,
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": "aGk="}}],
                }
            ]
        },
    )
    passthrough = await handler.handle_gemini_generate_content(
        FakeRequest(query_params={"alt": "sse", "key": "secret"}),
        "gemini-pro",
    )
    assert passthrough == {"streamed": True}
    assert "streamGenerateContent?key=secret&alt=sse" in handler.stream_calls[-1]["args"][0]

    handler = DummyGeminiHandler()
    handler.openai_pipeline = SimpleNamespace(
        apply=lambda **kwargs: SimpleNamespace(
            messages=[
                {"role": "user", "content": "optimized user"},
                {"role": "user", "content": "optimized preserved slot"},
            ],
            transforms_applied=["trim"],
            tokens_before=40,
            tokens_after=20,
            waste_signals=SimpleNamespace(to_dict=lambda: {"verbosity": 1}),
        )
    )
    handler.memory_handler = SimpleNamespace(
        config=SimpleNamespace(inject_context=True),
        search_and_format_context=lambda user_id, messages: "memory context",
    )
    handler.memory_handler.search_and_format_context = AsyncMock(return_value="memory context")
    handler.retry_response = FakeResponse(
        content=b'{"usageMetadata":{"promptTokenCount":30,"candidatesTokenCount":4,"cachedContentTokenCount":10}}',
        headers={"content-encoding": "gzip", "content-length": "999", "x-upstream": "1"},
        json_data={
            "usageMetadata": {
                "promptTokenCount": 30,
                "candidatesTokenCount": 4,
                "cachedContentTokenCount": 10,
            }
        },
    )
    install_gemini_modules(
        monkeypatch,
        {
            "contents": [
                {"role": "user", "parts": [{"text": "hello"}]},
                {
                    "role": "user",
                    "parts": [
                        {"text": "describe"},
                        {"inlineData": {"mimeType": "image/png", "data": "aGk="}},
                    ],
                },
            ],
            "systemInstruction": {"parts": [{"text": "original system"}]},
        },
    )
    response = await handler.handle_gemini_generate_content(
        FakeRequest(
            headers={"x-goog-api-key": "secret", "x-headroom-user-id": "user-123", "host": "proxy"},
            query_params={"key": "secret"},
        ),
        "gemini-pro",
    )
    assert response.status_code == 200
    assert response.headers["x-headroom-tokens-before"] == "40"
    assert response.headers["x-headroom-tokens-after"] == "20"
    assert response.headers["x-headroom-tokens-saved"] == "20"
    assert response.headers["x-headroom-transforms"] == "trim"
    assert response.headers["x-headroom-cached"] == "true"
    assert "content-encoding" not in response.headers
    assert response.headers["x-upstream"] == "1"
    assert handler.metrics.requests[-1]["cache_read_tokens"] == 10
    assert handler.cost_tracker.calls[-1]["kwargs"]["uncached_tokens"] == 20
    sent_body = handler.retry_calls[-1]["body"]
    assert sent_body["contents"][0]["parts"][0]["text"] == "optimized user"
    assert sent_body["contents"][1]["parts"][1]["inlineData"]["mimeType"] == "image/png"
    assert sent_body["systemInstruction"]["parts"][0]["text"] == "memory context"

    handler = DummyGeminiHandler()
    handler.rate_limiter = SimpleNamespace(check_request=AsyncMock(return_value=(False, 3.2)))
    install_gemini_modules(
        monkeypatch, {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
    )
    with pytest.raises(HTTPException) as exc_info:
        await handler.handle_gemini_generate_content(
            FakeRequest(headers={"x-goog-api-key": "secret"}),
            "gemini-pro",
        )
    assert exc_info.value.status_code == 429
    assert handler.metrics.rate_limited[-1]["provider"] == "gemini"

    handler = DummyGeminiHandler()
    handler.openai_pipeline = SimpleNamespace(
        apply=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    handler.retry_response = RuntimeError("upstream down")
    install_gemini_modules(
        monkeypatch, {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
    )
    failed = await handler.handle_gemini_generate_content(FakeRequest(), "gemini-pro")
    assert failed.status_code == 502
    assert handler.metrics.failures[-1]["provider"] == "gemini"


@pytest.mark.asyncio
async def test_handle_google_cloudcode_stream_and_stream_generate_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = DummyGeminiHandler()

    install_gemini_modules(monkeypatch, json.JSONDecodeError("bad", "x", 0))
    invalid = await handler.handle_google_cloudcode_stream(
        FakeRequest(),
    )
    assert invalid.status_code == 400

    install_gemini_modules(monkeypatch, {"model": "gemini-pro", "request": None})
    missing_request = await handler.handle_google_cloudcode_stream(
        FakeRequest(),
    )
    assert missing_request.status_code == 400

    handler = DummyGeminiHandler()
    handler.openai_pipeline = SimpleNamespace(
        apply=lambda **kwargs: SimpleNamespace(
            messages=[{"role": "user", "content": "optimized cloud"}],
            transforms_applied=["trim"],
            tokens_before=20,
            tokens_after=10,
        )
    )
    install_gemini_modules(
        monkeypatch,
        {
            "requestType": "agent",
            "model": "gemini-pro",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "systemInstruction": {"parts": [{"text": "keep me"}]},
            },
        },
    )
    streamed = await handler.handle_google_cloudcode_stream(
        FakeRequest(headers={"user-agent": "antigravity/1.0"}, query="key=secret"),
    )
    assert streamed == {"streamed": True}
    stream_args = handler.stream_calls[-1]["args"]
    assert (
        stream_args[0]
        == f"{gemini_module.ANTIGRAVITY_DAILY_API_URL}/v1internal:streamGenerateContent?key=secret"
    )
    assert stream_args[2]["request"]["systemInstruction"]["parts"][0]["text"] == "keep me"
    assert stream_args[2]["request"]["contents"][0]["parts"][0]["text"] == "optimized cloud"

    handler = DummyGeminiHandler()
    install_gemini_modules(monkeypatch, ValueError("bad json"))
    invalid_stream = await handler.handle_gemini_stream_generate_content(
        FakeRequest(), "gemini-pro"
    )
    assert invalid_stream.status_code == 400

    install_gemini_modules(
        monkeypatch,
        {"contents": [{"parts": [{"text": "hello"}, {"text": "world"}]}]},
    )
    streamed_native = await handler.handle_gemini_stream_generate_content(
        FakeRequest(query_params={"key": "secret"}),
        "gemini-pro",
    )
    assert streamed_native == {"streamed": True}
    assert handler.stream_calls[-1]["args"][0].endswith("streamGenerateContent?key=secret&alt=sse")
    assert handler.stream_calls[-1]["args"][6] == 10


@pytest.mark.asyncio
async def test_handle_gemini_count_tokens_covers_passthrough_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = DummyGeminiHandler()

    install_gemini_modules(monkeypatch, ValueError("bad json"))
    invalid = await handler.handle_gemini_count_tokens(FakeRequest(), "gemini-pro")
    assert invalid.status_code == 400

    install_gemini_modules(
        monkeypatch,
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": "aGk="}}],
                }
            ]
        },
    )
    handler.retry_response = FakeResponse(
        headers={"content-encoding": "gzip", "content-length": "123"}
    )
    passthrough = await handler.handle_gemini_count_tokens(
        FakeRequest(query_params={"key": "secret"}),
        "gemini-pro",
    )
    assert passthrough.status_code == 200
    assert handler.retry_calls[-1]["url"].endswith("countTokens?key=secret")
    assert "content-encoding" not in passthrough.headers

    handler = DummyGeminiHandler()
    handler.retry_response = FakeResponse(
        content=b'{"totalTokens":12}', json_data={"totalTokens": 12}
    )
    install_gemini_modules(
        monkeypatch,
        {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "systemInstruction": {"parts": [{"text": "original system"}]},
        },
    )
    response = await handler.handle_gemini_count_tokens(FakeRequest(), "gemini-pro")
    assert response.status_code == 200
    assert handler.metrics.requests[-1]["tokens_saved"] == 8
    assert handler.retry_calls[-1]["body"]["contents"][0]["parts"][0]["text"] == "optimized user"

    handler = DummyGeminiHandler()
    handler.retry_response = RuntimeError("count failed")
    install_gemini_modules(
        monkeypatch, {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
    )
    failed = await handler.handle_gemini_count_tokens(FakeRequest(), "gemini-pro")
    assert failed.status_code == 502
    assert handler.metrics.failures[-1]["provider"] == "gemini"
