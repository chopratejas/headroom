"""Tests for ratelimit header forwarding in streaming responses.

Verifies that anthropic-ratelimit-* headers from the upstream API response
are forwarded to the client in StreamingResponse, even in SSE streaming mode.

This was a bug where non-streaming responses correctly forwarded all headers
via dict(response.headers), but streaming responses used StreamingResponse
without passing any upstream headers — silently dropping ratelimit info.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from headroom.proxy.server import HeadroomProxy


class TestStreamingRatelimitHeaderForwarding:
    """Test that upstream ratelimit headers are forwarded in streaming responses."""

    def _create_mock_proxy(self):
        """Create a HeadroomProxy with mocked internals for unit testing."""
        proxy = object.__new__(HeadroomProxy)
        proxy.http_client = MagicMock(spec=httpx.AsyncClient)
        proxy.metrics = MagicMock()
        proxy.metrics.record_request = AsyncMock(return_value=None)
        proxy.metrics.record_failed = AsyncMock(return_value=None)
        proxy.cost_tracker = MagicMock()
        proxy.cost_tracker.estimate_cost.return_value = 0.001
        proxy.cost_tracker.record_request.return_value = None
        proxy.stats = {
            "requests_total": 0,
            "requests_optimized": 0,
            "tokens": {"original": 0, "optimized": 0, "saved": 0},
            "cost": {"total_usd": 0, "savings_usd": 0},
            "errors": 0,
            "active_requests": 0,
            "requests_per_model": {},
        }
        proxy.memory_manager = None
        proxy._config = MagicMock()
        proxy._config.memory_enabled = False
        proxy._config.ccr_inject_tool = False
        proxy._config.retry_max_attempts = 3
        proxy._config.retry_base_delay_ms = 0
        proxy._config.retry_max_delay_ms = 0
        proxy.config = proxy._config
        proxy._parse_sse_usage_from_buffer = MagicMock(return_value=None)
        proxy.memory_handler = None
        return proxy

    def _create_mock_upstream_response(self, extra_headers=None):
        """Create a mock httpx streaming response with ratelimit headers."""
        mock_response = AsyncMock()
        headers = {
            "content-type": "text/event-stream",
            "anthropic-ratelimit-tokens-limit": "80000",
            "anthropic-ratelimit-tokens-remaining": "75000",
            "anthropic-ratelimit-tokens-reset": "2026-03-25T12:00:00Z",
            "anthropic-ratelimit-requests-limit": "60",
            "anthropic-ratelimit-requests-remaining": "59",
            "anthropic-ratelimit-requests-reset": "2026-03-25T12:00:00Z",
            "anthropic-ratelimit-input-tokens-limit": "50000",
            "anthropic-ratelimit-input-tokens-remaining": "48000",
            "anthropic-ratelimit-input-tokens-reset": "2026-03-25T12:00:00Z",
            "anthropic-ratelimit-output-tokens-limit": "30000",
            "anthropic-ratelimit-output-tokens-remaining": "27000",
            "anthropic-ratelimit-output-tokens-reset": "2026-03-25T12:00:00Z",
            # Non-ratelimit headers that should NOT be forwarded
            "x-request-id": "req-12345",
            "cf-ray": "abc123",
        }
        if extra_headers:
            headers.update(extra_headers)
        mock_response.headers = httpx.Headers(headers)

        # Simulate a simple SSE stream
        sse_data = (
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01"}}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )

        async def aiter_bytes():
            yield sse_data

        mock_response.aiter_bytes = aiter_bytes
        mock_response.aclose = AsyncMock()
        return mock_response

    @pytest.mark.asyncio
    async def test_ratelimit_headers_forwarded_in_streaming(self):
        """Ratelimit headers from upstream should appear in the StreamingResponse."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()

        # Mock build_request + send to return our mock response
        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test", "anthropic-version": "2023-06-01"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-123",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        # Verify ratelimit headers are present in the StreamingResponse
        assert result.headers.get("anthropic-ratelimit-tokens-limit") == "80000"
        assert result.headers.get("anthropic-ratelimit-tokens-remaining") == "75000"
        assert result.headers.get("anthropic-ratelimit-tokens-reset") == "2026-03-25T12:00:00Z"
        assert result.headers.get("anthropic-ratelimit-requests-limit") == "60"
        assert result.headers.get("anthropic-ratelimit-requests-remaining") == "59"
        assert result.headers.get("anthropic-ratelimit-input-tokens-limit") == "50000"
        assert result.headers.get("anthropic-ratelimit-output-tokens-limit") == "30000"

    @pytest.mark.asyncio
    async def test_non_ratelimit_headers_not_forwarded(self):
        """Only ratelimit headers should be forwarded, not arbitrary upstream headers."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-456",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        # Non-ratelimit headers should NOT be in the response
        assert result.headers.get("x-request-id") is None
        assert result.headers.get("cf-ray") is None

    @pytest.mark.asyncio
    async def test_no_ratelimit_headers_still_works(self):
        """When upstream has no ratelimit headers, streaming should still work."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()
        # Remove all ratelimit headers
        mock_response.headers = httpx.Headers(
            {
                "content-type": "text/event-stream",
                "x-request-id": "req-999",
            }
        )

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-789",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        # Should still return a valid StreamingResponse
        assert result.media_type == "text/event-stream"
        # No ratelimit headers to forward
        assert result.headers.get("anthropic-ratelimit-tokens-limit") is None

    @pytest.mark.asyncio
    async def test_connect_error_returns_sse_error(self):
        """Connection errors should return an SSE error event (not crash)."""
        proxy = self._create_mock_proxy()

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-error",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        # Should return a StreamingResponse with error SSE event
        assert result.media_type == "text/event-stream"

        # Consume the generator to get the error event
        chunks = []
        async for chunk in result.body_iterator:
            chunks.append(chunk)

        assert len(chunks) == 1
        raw = chunks[0].decode("utf-8")
        assert "event: error" in raw
        error_data = json.loads(raw.split("data: ")[1].strip())
        assert error_data["error"]["type"] == "connection_error"
        assert "Connection refused" in error_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_connect_timeout_retries_before_returning_stream(self):
        """Transient connect timeouts should retry before failing the stream."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        attempts = {"count": 0}

        async def flaky_send(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ConnectTimeout("timed out")
            return mock_response

        proxy.http_client.send = AsyncMock(side_effect=flaky_send)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-retry",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        chunks = []
        async for chunk in result.body_iterator:
            chunks.append(chunk)

        assert attempts["count"] == 2
        assert chunks
