"""Tests for observed Anthropic cache TTL bucket metrics."""

from __future__ import annotations

import asyncio

import pytest

from headroom.proxy.cost import CostTracker, build_prefix_cache_stats
from headroom.proxy.prometheus_metrics import PrometheusMetrics


def test_prometheus_metrics_tracks_observed_ttl_buckets() -> None:
    metrics = PrometheusMetrics()

    asyncio.run(
        metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=20,
            tokens_saved=5,
            latency_ms=10.0,
            cache_read_tokens=40,
            cache_write_tokens=60,
            cache_write_5m_tokens=10,
            cache_write_1h_tokens=50,
        )
    )

    stats = metrics.cache_by_provider["anthropic"]
    assert stats["cache_write_5m_tokens"] == 10
    assert stats["cache_write_1h_tokens"] == 50
    assert stats["cache_write_5m_requests"] == 1
    assert stats["cache_write_1h_requests"] == 1


def test_cost_tracker_exposes_observed_ttl_buckets_per_model() -> None:
    tracker = CostTracker()
    tracker.record_tokens(
        "claude-opus-4-6",
        tokens_saved=10,
        tokens_sent=90,
        cache_read_tokens=40,
        cache_write_tokens=60,
        cache_write_5m_tokens=10,
        cache_write_1h_tokens=50,
        uncached_tokens=20,
    )

    stats = tracker.stats()
    assert stats["cache_write_5m_tokens"] == 10
    assert stats["cache_write_1h_tokens"] == 50
    assert stats["per_model"]["claude-opus-4-6"]["cache_write_5m_tokens"] == 10
    assert stats["per_model"]["claude-opus-4-6"]["cache_write_1h_tokens"] == 50


def test_prefix_cache_stats_include_observed_ttl_mix() -> None:
    metrics = PrometheusMetrics()
    provider_stats = metrics.cache_by_provider["anthropic"]
    provider_stats["requests"] = 2
    provider_stats["hit_requests"] = 1
    provider_stats["cache_read_tokens"] = 40
    provider_stats["cache_write_tokens"] = 60
    provider_stats["cache_write_5m_tokens"] = 15
    provider_stats["cache_write_1h_tokens"] = 45
    provider_stats["cache_write_5m_requests"] = 1
    provider_stats["cache_write_1h_requests"] = 1

    stats = build_prefix_cache_stats(metrics, None)
    anthropic = stats["by_provider"]["anthropic"]

    assert anthropic["observed_ttl_buckets"]["5m"]["tokens"] == 15
    assert anthropic["observed_ttl_buckets"]["1h"]["tokens"] == 45
    assert anthropic["observed_ttl_mix"]["5m_pct"] == 25.0
    assert anthropic["observed_ttl_mix"]["1h_pct"] == 75.0
    assert stats["totals"]["observed_ttl_buckets"]["5m"]["tokens"] == 15
    assert stats["totals"]["observed_ttl_buckets"]["1h"]["tokens"] == 45


def test_streaming_parser_extracts_anthropic_ttl_bucket_usage() -> None:
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    chunk = (
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":12,'
        b'"cache_read_input_tokens":3,"cache_creation_input_tokens":9,'
        b'"cache_creation":{"ephemeral_5m_input_tokens":4,"ephemeral_1h_input_tokens":5}}}}\n\n'
    )
    usage = proxy._parse_sse_usage(chunk, "anthropic")

    assert usage is not None
    assert usage["cache_creation_ephemeral_5m_input_tokens"] == 4
    assert usage["cache_creation_ephemeral_1h_input_tokens"] == 5


def test_stats_endpoint_reports_observed_ttl_buckets() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    proxy = app.state.proxy
    provider_stats = proxy.metrics.cache_by_provider["anthropic"]
    provider_stats["requests"] = 1
    provider_stats["hit_requests"] = 1
    provider_stats["cache_read_tokens"] = 30
    provider_stats["cache_write_tokens"] = 70
    provider_stats["cache_write_5m_tokens"] = 20
    provider_stats["cache_write_1h_tokens"] = 50
    provider_stats["cache_write_5m_requests"] = 1
    provider_stats["cache_write_1h_requests"] = 1

    with TestClient(app) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    prefix_cache = response.json()["prefix_cache"]
    anthropic = prefix_cache["by_provider"]["anthropic"]
    assert anthropic["observed_ttl_buckets"]["5m"]["tokens"] == 20
    assert anthropic["observed_ttl_buckets"]["1h"]["tokens"] == 50
    assert prefix_cache["totals"]["observed_ttl_mix"]["active_buckets"] == ["5m", "1h"]
