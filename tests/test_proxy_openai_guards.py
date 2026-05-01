from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import headroom.proxy.helpers as proxy_helpers
from headroom.proxy.handlers.openai import OpenAIHandlerMixin


class _FakePipelineExtensions:
    def emit(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return SimpleNamespace(messages=None, tools=None)


class _FakeHandler(OpenAIHandlerMixin):
    def __init__(self) -> None:
        self.pipeline_extensions = _FakePipelineExtensions()
        self.config = SimpleNamespace(image_optimize=False)

    async def _next_request_id(self) -> str:
        return "req-1"


class _FakeRequest:
    def __init__(self, payload: bytes, headers: dict[str, str]) -> None:
        self._payload = payload
        self.headers = headers

    async def body(self) -> bytes:
        return self._payload


def _decode_response(response) -> dict[str, object]:  # noqa: ANN001
    return json.loads(response.body)


def test_openai_chat_rejects_oversized_body(monkeypatch) -> None:
    monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 10)
    request = _FakeRequest(
        payload=b'{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}]}',
        headers={"content-length": "70", "authorization": "Bearer sk-test"},
    )

    response = asyncio.run(_FakeHandler().handle_openai_chat(request))

    assert response.status_code == 413
    payload = _decode_response(response)
    assert payload["error"]["code"] == "request_too_large"
    assert "Request body too large" in payload["error"]["message"]


def test_openai_chat_rejects_invalid_json_body() -> None:
    request = _FakeRequest(payload=b"{", headers={"authorization": "Bearer sk-test"})

    response = asyncio.run(_FakeHandler().handle_openai_chat(request))

    assert response.status_code == 400
    payload = _decode_response(response)
    assert payload["error"]["code"] == "invalid_json"
    assert "Invalid request body" in payload["error"]["message"]


def test_openai_chat_rejects_message_arrays_above_limit(monkeypatch) -> None:
    monkeypatch.setattr(proxy_helpers, "MAX_MESSAGE_ARRAY_LENGTH", 1)
    request = _FakeRequest(
        payload=json.dumps(
            {
                "model": "gpt-5.4",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "too many"},
                ],
            }
        ).encode("utf-8"),
        headers={"authorization": "Bearer sk-test"},
    )

    response = asyncio.run(_FakeHandler().handle_openai_chat(request))

    assert response.status_code == 400
    payload = _decode_response(response)
    assert payload["error"]["code"] == "invalid_request"
    assert "Message array too large" in payload["error"]["message"]


def test_openai_responses_rejects_oversized_body(monkeypatch) -> None:
    monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 10)
    request = _FakeRequest(
        payload=b'{"model":"gpt-5.4","input":"hello"}',
        headers={"content-length": "40", "authorization": "Bearer sk-test"},
    )

    response = asyncio.run(_FakeHandler().handle_openai_responses(request))

    assert response.status_code == 413
    payload = _decode_response(response)
    assert payload["error"]["code"] == "request_too_large"
    assert "Request body too large" in payload["error"]["message"]


def test_openai_responses_rejects_invalid_json_body() -> None:
    request = _FakeRequest(payload=b"{", headers={"authorization": "Bearer sk-test"})

    response = asyncio.run(_FakeHandler().handle_openai_responses(request))

    assert response.status_code == 400
    payload = _decode_response(response)
    assert payload["error"]["code"] == "invalid_json"
    assert "Invalid request body" in payload["error"]["message"]
