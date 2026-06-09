"""Unit tests for headroom.subscription.codex_rate_limits."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import headroom.subscription.codex_rate_limits as codex_rate_limits
from headroom.proxy.server import HeadroomProxy
from headroom.subscription.codex_rate_limits import (
    CodexRateLimitSnapshot,
    CodexRateLimitState,
    CodexRateLimitWindow,
    get_codex_rate_limit_state,
    parse_codex_rate_limits,
)


@pytest.fixture(autouse=True)
def _isolate_persist_path(tmp_path, monkeypatch):
    """Redirect the on-disk persistence file to a temp location.

    Without this, ``CodexRateLimitState`` would read/write the real
    ``~/.headroom/codex_rate_limits.json`` during tests, polluting the
    developer's environment and making ``test_initial_state_is_none``-style
    assertions order/run dependent.
    """
    monkeypatch.setattr(
        codex_rate_limits, "_PERSIST_PATH", tmp_path / "codex_rate_limits.json"
    )
    # Reset the process-global singleton so each test starts clean.
    monkeypatch.setattr(codex_rate_limits, "_state", None)
    yield

# ---------------------------------------------------------------------------
# CodexRateLimitWindow helpers
# ---------------------------------------------------------------------------


class TestCodexRateLimitWindow:
    def test_window_label_minutes(self):
        w = CodexRateLimitWindow(used_percent=10.0, window_minutes=45)
        assert w.window_label == "45m"

    def test_window_label_hours(self):
        w = CodexRateLimitWindow(used_percent=10.0, window_minutes=60)
        assert w.window_label == "1h"

    def test_window_label_hours_with_minutes(self):
        w = CodexRateLimitWindow(used_percent=10.0, window_minutes=90)
        assert w.window_label == "1h30m"

    def test_window_label_unknown(self):
        w = CodexRateLimitWindow(used_percent=10.0, window_minutes=None)
        assert w.window_label == "unknown"

    def test_seconds_until_reset_future(self):
        future = int(time.time()) + 3600
        w = CodexRateLimitWindow(used_percent=10.0, resets_at=future)
        secs = w.seconds_until_reset
        assert secs is not None
        assert 3590 <= secs <= 3600

    def test_seconds_until_reset_past(self):
        past = int(time.time()) - 100
        w = CodexRateLimitWindow(used_percent=10.0, resets_at=past)
        assert w.seconds_until_reset == 0

    def test_seconds_until_reset_none(self):
        w = CodexRateLimitWindow(used_percent=10.0, resets_at=None)
        assert w.seconds_until_reset is None

    def test_to_dict_keys(self):
        w = CodexRateLimitWindow(used_percent=42.5, window_minutes=60, resets_at=9999999)
        d = w.to_dict()
        assert set(d.keys()) == {
            "used_percent",
            "window_minutes",
            "window_label",
            "resets_at",
            "seconds_until_reset",
        }
        assert d["used_percent"] == 42.5
        assert d["window_label"] == "1h"


# ---------------------------------------------------------------------------
# parse_codex_rate_limits
# ---------------------------------------------------------------------------


class TestParseCodexRateLimits:
    def test_returns_none_for_empty_headers(self):
        assert parse_codex_rate_limits({}) is None

    def test_returns_none_for_non_codex_headers(self):
        headers = {"content-type": "application/json", "x-request-id": "abc"}
        assert parse_codex_rate_limits(headers) is None

    def test_parses_primary_window(self):
        headers = {
            "x-codex-primary-used-percent": "35.5",
            "x-codex-primary-window-minutes": "60",
            "x-codex-primary-reset-at": "1704069000",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.limit_id == "codex"
        assert snap.primary is not None
        assert snap.primary.used_percent == 35.5
        assert snap.primary.window_minutes == 60
        assert snap.primary.resets_at == 1704069000
        assert snap.secondary is None

    def test_parses_secondary_window(self):
        headers = {
            "x-codex-primary-used-percent": "10.0",
            "x-codex-secondary-used-percent": "80.0",
            "x-codex-secondary-window-minutes": "1440",
            "x-codex-secondary-reset-at": "1704100000",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.secondary is not None
        assert snap.secondary.used_percent == 80.0
        assert snap.secondary.window_minutes == 1440

    def test_parses_credits(self):
        headers = {
            "x-codex-primary-used-percent": "5.0",
            "x-codex-credits-has-credits": "true",
            "x-codex-credits-unlimited": "false",
            "x-codex-credits-balance": "$12.50",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.credits is not None
        assert snap.credits.has_credits is True
        assert snap.credits.unlimited is False
        assert snap.credits.balance == "$12.50"

    def test_parses_unlimited_credits(self):
        headers = {
            "x-codex-primary-used-percent": "0.0",
            "x-codex-credits-has-credits": "true",
            "x-codex-credits-unlimited": "true",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.credits is not None
        assert snap.credits.unlimited is True
        assert snap.credits.balance is None

    def test_parses_limit_name(self):
        headers = {
            "x-codex-primary-used-percent": "20.0",
            "x-codex-limit-name": "gpt-5.2-codex-sonic",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.limit_name == "gpt-5.2-codex-sonic"

    def test_parses_promo_message(self):
        headers = {
            "x-codex-primary-used-percent": "50.0",
            "x-codex-promo-message": "Try our new model!",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.promo_message == "Try our new model!"

    def test_only_credits_header_triggers_parse(self):
        headers = {
            "x-codex-credits-has-credits": "true",
            "x-codex-credits-unlimited": "false",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.primary is None
        assert snap.credits is not None

    def test_invalid_float_ignored(self):
        headers = {"x-codex-primary-used-percent": "not_a_number"}
        assert parse_codex_rate_limits(headers) is None

    def test_to_dict_structure(self):
        headers = {
            "x-codex-primary-used-percent": "42.0",
            "x-codex-primary-window-minutes": "60",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        d = snap.to_dict()
        assert "limit_id" in d
        assert "primary" in d
        assert "secondary" in d
        assert "credits" in d
        assert "captured_at" in d


# ---------------------------------------------------------------------------
# CodexRateLimitState
# ---------------------------------------------------------------------------


class TestCodexRateLimitState:
    def test_initial_state_is_none(self):
        state = CodexRateLimitState()
        assert state.latest is None
        assert state.get_stats() is None

    def test_update_from_headers_stores_snapshot(self):
        state = CodexRateLimitState()
        headers = {
            "x-codex-primary-used-percent": "55.0",
            "x-codex-primary-window-minutes": "60",
        }
        state.update_from_headers(headers)
        snap = state.latest
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 55.0

    def test_update_from_empty_headers_is_noop(self):
        state = CodexRateLimitState()
        state.update_from_headers({})
        assert state.latest is None

    def test_update_from_non_codex_headers_is_noop(self):
        state = CodexRateLimitState()
        state.update_from_headers({"content-type": "application/json"})
        assert state.latest is None

    def test_get_stats_returns_dict_when_data_present(self):
        state = CodexRateLimitState()
        state.update_from_headers({"x-codex-primary-used-percent": "10.0"})
        stats = state.get_stats()
        assert stats is not None
        assert isinstance(stats, dict)
        assert stats["limit_id"] == "codex"

    def test_update_overwrites_previous_snapshot(self):
        state = CodexRateLimitState()
        state.update_from_headers({"x-codex-primary-used-percent": "10.0"})
        state.update_from_headers({"x-codex-primary-used-percent": "90.0"})
        snap = state.latest
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 90.0


# ---------------------------------------------------------------------------
# Persistence (survives restart)
# ---------------------------------------------------------------------------


class TestCodexRateLimitPersistence:
    def test_update_writes_snapshot_to_disk(self):
        state = CodexRateLimitState()
        state.update_from_headers(
            {
                "x-codex-primary-used-percent": "45.2",
                "x-codex-primary-window-minutes": "15",
            }
        )
        assert codex_rate_limits._PERSIST_PATH.exists()
        data = json.loads(codex_rate_limits._PERSIST_PATH.read_text())
        assert data["primary"]["used_percent"] == 45.2

    def test_new_instance_loads_persisted_snapshot(self):
        # First instance writes a snapshot to disk.
        first = CodexRateLimitState()
        first.update_from_headers(
            {
                "x-codex-primary-used-percent": "33.3",
                "x-codex-secondary-used-percent": "66.6",
            }
        )
        # A fresh instance (simulating a proxy restart) should reload it.
        second = CodexRateLimitState()
        snap = second.latest
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 33.3
        assert snap.secondary is not None
        assert snap.secondary.used_percent == 66.6

    def test_corrupt_persist_file_is_ignored(self):
        codex_rate_limits._PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        codex_rate_limits._PERSIST_PATH.write_text("{not valid json")
        # Should not raise — just start with empty state.
        state = CodexRateLimitState()
        assert state.latest is None

    def test_snapshot_roundtrips_through_dict(self):
        snap = parse_codex_rate_limits(
            {
                "x-codex-primary-used-percent": "12.5",
                "x-codex-primary-window-minutes": "60",
                "x-codex-primary-reset-at": "1749043200",
                "x-codex-secondary-used-percent": "80.0",
                "x-codex-credits-has-credits": "true",
                "x-codex-credits-balance": "$5.00",
                "x-codex-limit-name": "gpt-5.4-codex",
            }
        )
        assert snap is not None
        restored = CodexRateLimitSnapshot.from_dict(snap.to_dict())
        assert restored.primary is not None
        assert restored.primary.used_percent == 12.5
        assert restored.primary.window_minutes == 60
        assert restored.secondary is not None
        assert restored.secondary.used_percent == 80.0
        assert restored.credits is not None
        assert restored.credits.balance == "$5.00"
        assert restored.limit_name == "gpt-5.4-codex"


# ---------------------------------------------------------------------------
# Streaming SSE path integration
# ---------------------------------------------------------------------------


def _make_streaming_proxy() -> HeadroomProxy:
    """Create a HeadroomProxy with mocked internals for streaming unit tests."""
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


def _make_codex_streaming_response() -> AsyncMock:
    """Mock httpx streaming response carrying x-codex-* headers."""
    mock_response = AsyncMock()
    mock_response.headers = httpx.Headers(
        {
            "content-type": "text/event-stream",
            "x-codex-primary-used-percent": "45.2",
            "x-codex-primary-window-minutes": "15",
            "x-codex-primary-reset-at": "1749043200",
            "x-codex-secondary-used-percent": "12.1",
            "x-codex-secondary-window-minutes": "10080",
            "x-codex-secondary-reset-at": "1749648000",
        }
    )
    mock_response.status_code = 200

    async def aiter_bytes():
        yield b'data: {"type":"response.completed"}\n\n'

    mock_response.aiter_bytes = aiter_bytes
    mock_response.aclose = AsyncMock()
    return mock_response


class TestStreamingPathUpdatesCodexRateLimits:
    @pytest.mark.asyncio
    async def test_streaming_path_updates_codex_rate_limits(self):
        """Verify that _stream_response calls update_from_headers (SSE path)."""
        proxy = _make_streaming_proxy()
        mock_response = _make_codex_streaming_response()
        proxy.http_client.build_request = MagicMock(return_value=MagicMock())
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        await proxy._stream_response(
            url="https://api.openai.com/v1/responses",
            headers={"authorization": "Bearer sk-test"},
            body={"model": "gpt-5.4-codex", "stream": True, "input": "hi"},
            provider="openai",
            model="gpt-5.4-codex",
            request_id="codex-stream-1",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        state = get_codex_rate_limit_state()
        assert state.latest is not None
        assert state.latest.primary is not None
        assert state.latest.primary.used_percent == 45.2
        assert state.latest.secondary is not None
        assert state.latest.secondary.used_percent == 12.1

    @pytest.mark.asyncio
    async def test_streaming_path_forwards_codex_headers_to_client(self):
        """x-codex-* headers must reach the client so the CLI display updates."""
        proxy = _make_streaming_proxy()
        mock_response = _make_codex_streaming_response()
        proxy.http_client.build_request = MagicMock(return_value=MagicMock())
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.openai.com/v1/responses",
            headers={"authorization": "Bearer sk-test"},
            body={"model": "gpt-5.4-codex", "stream": True, "input": "hi"},
            provider="openai",
            model="gpt-5.4-codex",
            request_id="codex-stream-2",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        assert result.headers.get("x-codex-primary-used-percent") == "45.2"
        assert result.headers.get("x-codex-secondary-used-percent") == "12.1"
