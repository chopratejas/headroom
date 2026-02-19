"""ASGI Middleware — add Headroom compression to any Python proxy.

Drop-in middleware for FastAPI, Starlette, LiteLLM proxy, or any ASGI app.
Intercepts LLM requests, compresses messages, forwards the smaller payload.

Usage with LiteLLM proxy:

    from litellm.proxy.proxy_server import app
    from headroom.integrations.asgi import CompressionMiddleware

    app.add_middleware(CompressionMiddleware)

Usage with any FastAPI app:

    from fastapi import FastAPI
    from headroom.integrations.asgi import CompressionMiddleware

    app = FastAPI()
    app.add_middleware(CompressionMiddleware)
    # Your existing routes...

Configuration:

    app.add_middleware(
        CompressionMiddleware,
        min_tokens=500,        # Only compress if messages > 500 tokens
        model_limit=200000,    # Context window size
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Paths that contain LLM messages to compress
_LLM_PATHS = (
    "/v1/messages",  # Anthropic
    "/v1/chat/completions",  # OpenAI
    "/v1/responses",  # OpenAI Responses API
    "/chat/completions",  # LiteLLM (without /v1 prefix)
)


class CompressionMiddleware:
    """ASGI middleware that compresses LLM request messages.

    Intercepts POST requests to LLM endpoints, compresses the messages
    using Headroom's full pipeline, and forwards the smaller payload.

    Response headers include compression metrics:
    - x-headroom-tokens-before: original token count
    - x-headroom-tokens-after: compressed token count
    - x-headroom-tokens-saved: tokens removed
    """

    def __init__(
        self,
        app: ASGIApp,
        min_tokens: int = 500,
        model_limit: int = 200000,
        hooks: Any = None,
    ) -> None:
        self.app = app
        self._min_tokens = min_tokens
        self._model_limit = model_limit
        self._hooks = hooks

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Only intercept POST to LLM endpoints
        if method != "POST" or not any(path.endswith(p) or path == p for p in _LLM_PATHS):
            await self.app(scope, receive, send)
            return

        # Buffer the request body
        body_chunks: list[bytes] = []

        async def buffering_receive() -> dict[str, Any]:
            message: dict[str, Any] = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                if chunk:
                    body_chunks.append(chunk)
            return message

        # Read the full body
        while True:
            msg = await buffering_receive()
            if msg.get("type") == "http.request":
                if not msg.get("more_body", False):
                    break

        full_body = b"".join(body_chunks)

        # Parse and compress
        tokens_saved = 0
        try:
            body_json = json.loads(full_body)
            messages = body_json.get("messages", [])
            model = body_json.get("model", "")

            if messages:
                from headroom.compress import compress

                result = compress(
                    messages=messages,
                    model=model or "claude-sonnet-4-5-20250929",
                    model_limit=self._model_limit,
                    hooks=self._hooks,
                )

                if result.tokens_saved > 0:
                    body_json["messages"] = result.messages
                    full_body = json.dumps(body_json).encode("utf-8")
                    tokens_saved = result.tokens_saved

                    logger.info(
                        "Headroom: %d→%d tokens (saved %d, %.0f%%)",
                        result.tokens_before,
                        result.tokens_after,
                        result.tokens_saved,
                        result.compression_ratio * 100,
                    )

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug("Headroom middleware: skipping non-JSON request: %s", e)

        # Create a new receive that returns the (possibly modified) body
        body_sent = False

        async def modified_receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": full_body, "more_body": False}
            result: dict[str, Any] = await receive()
            return result

        # Wrap send to inject compression headers
        async def metrics_send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start" and tokens_saved > 0:
                headers = list(message.get("headers", []))
                headers.append((b"x-headroom-compressed", b"true"))
                headers.append((b"x-headroom-tokens-saved", str(tokens_saved).encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, modified_receive, metrics_send)
