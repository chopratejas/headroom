from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from headroom.dashboard import get_dashboard_html
from headroom.proxy import helpers as proxy_helpers


class _StatsStub:
    def __init__(self, calls: dict[str, int], key: str, payload: dict):
        self._calls = calls
        self._key = key
        self._payload = payload

    def get_stats(self) -> dict:
        self._calls[self._key] += 1
        return dict(self._payload)


class _ToinStub:
    def get_stats(self) -> dict:
        return {"patterns": 0}


@pytest.fixture(autouse=True)
def _reset_rtk_stats_cache() -> None:
    proxy_helpers._rtk_stats_cache.update({"expires_at": 0.0, "has_value": False, "value": None})


def test_get_rtk_stats_memoizes_subprocess_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    now = {"value": 100.0}
    calls = {"run": 0}

    def _fake_run(*args, **kwargs):
        calls["run"] += 1
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"summary": {"total_commands": 7, "total_saved": 1234}}),
        )

    monkeypatch.setattr(proxy_helpers.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rtk")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    first = proxy_helpers._get_rtk_stats()
    second = proxy_helpers._get_rtk_stats()

    assert first == second
    assert first == {
        "installed": True,
        "total_commands": 7,
        "tokens_saved": 1234,
        "avg_savings_pct": 0.0,
    }
    assert calls["run"] == 1

    now["value"] += proxy_helpers.RTK_STATS_CACHE_TTL_SECONDS + 0.1
    third = proxy_helpers._get_rtk_stats()

    assert third == first
    assert calls["run"] == 2


def test_stats_cached_query_reuses_short_ttl_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import headroom.proxy.server as server
    from headroom.proxy.server import ProxyConfig, create_app

    calls = {"store": 0, "telemetry": 0, "feedback": 0, "rtk": 0}
    now = {"value": 100.0}

    monkeypatch.setattr(server.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        server,
        "get_compression_store",
        lambda: _StatsStub(calls, "store", {"entry_count": 1, "max_entries": 100}),
    )
    monkeypatch.setattr(
        server,
        "get_telemetry_collector",
        lambda: _StatsStub(calls, "telemetry", {"enabled": True}),
    )
    monkeypatch.setattr(
        server,
        "get_compression_feedback",
        lambda: _StatsStub(calls, "feedback", {}),
    )

    def _fake_rtk_stats() -> dict[str, int | bool | float]:
        calls["rtk"] += 1
        return {
            "installed": True,
            "total_commands": 1,
            "tokens_saved": 5,
            "avg_savings_pct": 10.0,
        }

    monkeypatch.setattr(server, "_get_rtk_stats", _fake_rtk_stats)
    monkeypatch.setattr(server, "get_toin", lambda: _ToinStub())

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

    with TestClient(app) as client:
        first = client.get("/stats?cached=1")
        second = client.get("/stats?cached=1")
        now["value"] += 5.1
        third = client.get("/stats?cached=1")
        uncached = client.get("/stats")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert uncached.status_code == 200

    assert calls == {"store": 3, "telemetry": 3, "feedback": 3, "rtk": 3}
    assert first.json()["cli_filtering"]["tokens_saved"] == 5


def test_dashboard_uses_cached_stats_and_lazy_history_feed_polling() -> None:
    html = get_dashboard_html()

    assert "fetch('/stats?cached=1')" in html
    assert "@click=\"setViewMode('history')\"" in html
    assert '@click="toggleFeed()"' in html
    assert "this.viewMode === 'history'" in html
    assert "this.feedOpen" in html
