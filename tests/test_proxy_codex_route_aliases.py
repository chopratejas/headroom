import httpx
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


def test_codex_responses_aliases_delegate_to_openai_handler(monkeypatch):
    async def fake_handle(self, request):  # type: ignore[no-untyped-def]
        return JSONResponse({"ok": True, "path": request.url.path})

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses", fake_handle)

    with TestClient(create_app(ProxyConfig())) as client:
        for path in ("/backend-api/responses", "/backend-api/codex/responses"):
            response = client.post(path, json={"model": "gpt-5.3-codex"})
            assert response.status_code == 200
            assert response.json() == {"ok": True, "path": path}


def test_codex_responses_websocket_aliases_delegate_to_openai_handler(monkeypatch):
    seen_paths: list[str] = []

    async def fake_handle_ws(self, websocket: WebSocket):  # type: ignore[no-untyped-def]
        seen_paths.append(websocket.url.path)
        await websocket.accept()
        await websocket.send_json({"ok": True, "path": websocket.url.path})
        await websocket.close()

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses_ws", fake_handle_ws)

    with TestClient(create_app(ProxyConfig())) as client:
        for path in ("/backend-api/responses", "/backend-api/codex/responses"):
            with client.websocket_connect(path) as websocket:
                assert websocket.receive_json() == {"ok": True, "path": path}

    assert seen_paths == ["/backend-api/responses", "/backend-api/codex/responses"]


def test_codex_responses_subpath_aliases_delegate_to_passthrough():
    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def request(self, method, url, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((method, url))
            return httpx.Response(200, json={"method": method, "url": url})

        async def aclose(self) -> None:
            return None

    with TestClient(create_app(ProxyConfig())) as client:
        fake_http_client = FakeAsyncClient()
        client.app.state.proxy.http_client = fake_http_client
        client.app.state.proxy.OPENAI_API_URL = "https://api.openai.test"

        api_key_response = client.post(
            "/backend-api/responses/compact?trace=1",
            json={"model": "gpt-5.3-codex"},
        )
        chatgpt_response = client.post(
            "/backend-api/codex/responses/compact?trace=2",
            headers={"chatgpt-account-id": "acct_123"},
            json={"model": "gpt-5.3-codex"},
        )

    assert api_key_response.status_code == 200
    assert chatgpt_response.status_code == 200
    assert fake_http_client.calls == [
        ("POST", "https://api.openai.test/v1/responses/compact?trace=1"),
        ("POST", "https://chatgpt.com/backend-api/codex/responses/compact?trace=2"),
    ]
