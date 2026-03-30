"""Tests for durable proxy savings history."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.savings_tracker import SavingsTracker
from headroom.proxy.server import ProxyConfig, create_app


def _record_request(client: TestClient, *, model: str, tokens_saved: int) -> None:
    proxy = client.app.state.proxy
    asyncio.run(
        proxy.metrics.record_request(
            provider="openai",
            model=model,
            input_tokens=120,
            output_tokens=24,
            tokens_saved=tokens_saved,
            latency_ms=15.0,
        )
    )


def test_savings_tracker_rollups_are_chart_friendly(tmp_path, monkeypatch):
    path = tmp_path / "proxy_savings.json"
    tracker = SavingsTracker(path=str(path), max_history_points=100, max_history_age_days=30)
    monkeypatch.setattr(
        "headroom.proxy.savings_tracker._estimate_compression_savings_usd",
        lambda model, tokens_saved: tokens_saved / 1000.0,
    )

    tracker.record_compression_savings(
        model="gpt-4o",
        tokens_saved=100,
        timestamp="2026-03-27T09:10:00Z",
    )
    tracker.record_compression_savings(
        model="gpt-4o",
        tokens_saved=50,
        timestamp="2026-03-27T09:40:00Z",
    )
    tracker.record_compression_savings(
        model="gpt-4o",
        tokens_saved=25,
        timestamp="2026-03-27T10:05:00Z",
    )
    tracker.record_compression_savings(
        model="gpt-4o",
        tokens_saved=10,
        timestamp="2026-03-28T08:00:00Z",
    )

    response = tracker.history_response()

    assert response["lifetime"]["tokens_saved"] == 185
    assert response["lifetime"]["compression_savings_usd"] == pytest.approx(0.185)
    assert len(response["history"]) == 4

    hourly = response["series"]["hourly"]
    assert [point["timestamp"] for point in hourly] == [
        "2026-03-27T09:00:00Z",
        "2026-03-27T10:00:00Z",
        "2026-03-28T08:00:00Z",
    ]
    assert hourly[0]["tokens_saved"] == 150
    assert hourly[0]["total_tokens_saved"] == 150
    assert hourly[1]["tokens_saved"] == 25
    assert hourly[1]["total_tokens_saved"] == 175
    assert hourly[2]["tokens_saved"] == 10
    assert hourly[2]["total_tokens_saved"] == 185

    daily = response["series"]["daily"]
    assert [point["timestamp"] for point in daily] == [
        "2026-03-27T00:00:00Z",
        "2026-03-28T00:00:00Z",
    ]
    assert daily[0]["tokens_saved"] == 175
    assert daily[0]["total_tokens_saved"] == 175
    assert daily[1]["tokens_saved"] == 10
    assert daily[1]["total_tokens_saved"] == 185


def test_stats_history_persists_across_restarts_and_stats_stays_compatible(tmp_path, monkeypatch):
    savings_path = tmp_path / "proxy_savings.json"
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(savings_path))

    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=False,
    )

    with TestClient(create_app(config)) as client:
        _record_request(client, model="gpt-4o", tokens_saved=40)

        stats = client.get("/stats")
        assert stats.status_code == 200
        stats_data = stats.json()
        assert "savings_history" in stats_data
        assert "persistent_savings" in stats_data
        assert all(len(point) == 2 for point in stats_data["savings_history"])
        assert stats_data["persistent_savings"]["lifetime"]["tokens_saved"] == 40
        assert stats_data["persistent_savings"]["storage_path"] == str(savings_path)

        history = client.get("/stats-history")
        assert history.status_code == 200
        history_data = history.json()
        assert history_data["schema_version"] == 1
        assert history_data["storage_path"] == str(savings_path)
        assert history_data["lifetime"]["tokens_saved"] == 40
        assert list(history_data["series"].keys()) == ["hourly", "daily"]

    with TestClient(create_app(config)) as client:
        history = client.get("/stats-history")
        assert history.status_code == 200
        assert history.json()["lifetime"]["tokens_saved"] == 40

        _record_request(client, model="gpt-4o", tokens_saved=15)

        updated = client.get("/stats-history").json()
        assert updated["lifetime"]["tokens_saved"] == 55
        assert len(updated["history"]) == 2

        persisted = json.loads(savings_path.read_text())
        assert persisted["lifetime"]["tokens_saved"] == 55


def test_malformed_savings_state_is_ignored_safely(tmp_path, monkeypatch):
    savings_path = tmp_path / "proxy_savings.json"
    savings_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(savings_path))

    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=False,
    )

    with TestClient(create_app(config)) as client:
        response = client.get("/stats-history")
        assert response.status_code == 200
        data = response.json()
        assert data["lifetime"]["tokens_saved"] == 0
        assert data["history"] == []


def test_dashboard_includes_history_toggle_and_endpoint(tmp_path, monkeypatch):
    savings_path = tmp_path / "proxy_savings.json"
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(savings_path))

    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=False,
    )

    with TestClient(create_app(config)) as client:
        response = client.get("/dashboard")
        assert response.status_code == 200
        html = response.text
        assert "Session" in html
        assert "Historical" in html
        assert "fetch('/stats-history')" in html
