"""Tests for durable proxy savings history."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

import headroom.proxy.savings_tracker as savings_tracker_module
from fastapi.testclient import TestClient

from headroom.proxy.savings_tracker import HEADROOM_SAVINGS_PATH_ENV_VAR, SavingsTracker
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


def test_savings_tracker_helpers_normalize_inputs_and_paths(tmp_path, monkeypatch):
    override_path = tmp_path / "custom-savings.json"
    monkeypatch.setenv(HEADROOM_SAVINGS_PATH_ENV_VAR, str(override_path))
    assert savings_tracker_module.get_default_savings_storage_path() == str(override_path)

    monkeypatch.delenv(HEADROOM_SAVINGS_PATH_ENV_VAR, raising=False)
    default_path = savings_tracker_module.get_default_savings_storage_path()
    assert default_path.endswith(".headroom/proxy_savings.json")

    assert savings_tracker_module._parse_timestamp("") is None
    assert savings_tracker_module._parse_timestamp("not-a-timestamp") is None
    assert savings_tracker_module._parse_timestamp("2026-03-27T09:00:00") == datetime(
        2026, 3, 27, 9, 0, tzinfo=timezone.utc
    )

    assert savings_tracker_module._coerce_int("7") == 7
    assert savings_tracker_module._coerce_int(-5) == 0
    assert savings_tracker_module._coerce_float("0.25") == pytest.approx(0.25)
    assert savings_tracker_module._coerce_float(-0.25) == 0.0

    assert savings_tracker_module._normalize_history_entry(
        ["2026-03-27T09:00:00Z", "12", "0.5"]
    ) == {
        "timestamp": "2026-03-27T09:00:00Z",
        "total_tokens_saved": 12,
        "compression_savings_usd": 0.5,
    }
    assert savings_tracker_module._normalize_history_entry({"timestamp": "bad"}) is None
    assert savings_tracker_module._normalize_history_entry(object()) is None


def test_savings_tracker_sanitizes_legacy_state_and_applies_retention(tmp_path):
    path = tmp_path / "proxy_savings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "lifetime": {
                    "tokens_saved": 1,
                    "compression_savings_usd": 0.001,
                },
                "history": [
                    ["2026-03-24T08:00:00Z", 10, 0.01],
                    {
                        "timestamp": "2026-03-26T12:00:00Z",
                        "total_tokens_saved": 20,
                        "compression_savings_usd": 0.02,
                    },
                    {
                        "timestamp": "2026-03-27T09:00:00Z",
                        "total_tokens_saved": 30,
                        "compression_savings_usd": 0.03,
                    },
                    {"timestamp": "bad", "total_tokens_saved": 999},
                ],
            }
        ),
        encoding="utf-8",
    )

    tracker = SavingsTracker(path=str(path), max_history_points=1, max_history_age_days=2)
    snapshot = tracker.snapshot()

    assert snapshot["schema_version"] == 1
    assert snapshot["lifetime"] == {
        "tokens_saved": 30,
        "compression_savings_usd": pytest.approx(0.03),
    }
    assert snapshot["history"] == [
        {
            "timestamp": "2026-03-27T09:00:00Z",
            "total_tokens_saved": 30,
            "compression_savings_usd": 0.03,
        }
    ]
    assert snapshot["retention"] == {
        "max_history_points": 1,
        "max_history_age_days": 2,
    }


def test_non_dict_savings_state_resets_to_default(tmp_path):
    path = tmp_path / "proxy_savings.json"
    path.write_text("[]", encoding="utf-8")

    tracker = SavingsTracker(path=str(path))
    snapshot = tracker.snapshot()

    assert snapshot["lifetime"] == {
        "tokens_saved": 0,
        "compression_savings_usd": 0.0,
    }
    assert snapshot["history"] == []


def test_record_compression_savings_skips_empty_updates_and_normalizes_timestamps(
    tmp_path, monkeypatch
):
    path = tmp_path / "proxy_savings.json"
    tracker = SavingsTracker(path=str(path))
    monkeypatch.setattr(
        savings_tracker_module,
        "_estimate_compression_savings_usd",
        lambda model, tokens_saved: tokens_saved / 1000.0,
    )

    assert tracker.record_compression_savings(model="gpt-4o", tokens_saved=0) is False
    assert not path.exists()

    local_time = datetime(2026, 3, 27, 10, 0, tzinfo=timezone(timedelta(hours=2)))
    assert tracker.record_compression_savings(
        model="gpt-4o",
        tokens_saved=10,
        timestamp=local_time,
    )

    fallback_time = datetime(2026, 3, 27, 12, 34, tzinfo=timezone.utc)
    monkeypatch.setattr(savings_tracker_module, "_utc_now", lambda: fallback_time)
    assert tracker.record_compression_savings(
        model="gpt-4o",
        tokens_saved=5,
        timestamp="not-a-timestamp",
    )

    snapshot = tracker.snapshot()
    assert snapshot["history"] == [
        {
            "timestamp": "2026-03-27T08:00:00Z",
            "total_tokens_saved": 10,
            "compression_savings_usd": 0.01,
        },
        {
            "timestamp": "2026-03-27T12:34:00Z",
            "total_tokens_saved": 15,
            "compression_savings_usd": 0.015,
        },
    ]

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["lifetime"]["tokens_saved"] == 15
    assert persisted["history"][-1]["timestamp"] == "2026-03-27T12:34:00Z"


def test_litellm_resolution_and_savings_estimation_fallbacks(monkeypatch):
    def fake_cost_per_token(*, model, prompt_tokens, completion_tokens):
        if model in {"gpt-4o", "anthropic/claude-sonnet-4-6"}:
            return {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        raise RuntimeError("unknown model")

    fake_litellm = SimpleNamespace(
        cost_per_token=fake_cost_per_token,
        model_cost={
            "anthropic/claude-sonnet-4-6": {"input_cost_per_token": 0.002},
            "gpt-4o": {"input_cost_per_token": 0.001},
        },
    )
    monkeypatch.setattr(savings_tracker_module, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(savings_tracker_module, "litellm", fake_litellm)

    assert savings_tracker_module._resolve_litellm_model("gpt-4o") == "gpt-4o"
    assert (
        savings_tracker_module._resolve_litellm_model("claude-sonnet-4-6")
        == "anthropic/claude-sonnet-4-6"
    )
    assert savings_tracker_module._estimate_compression_savings_usd(
        "claude-sonnet-4-6", 100
    ) == pytest.approx(0.2)

    fake_litellm.model_cost = {}
    assert savings_tracker_module._estimate_compression_savings_usd("gpt-4o", 100) == 0.0

    monkeypatch.setattr(
        fake_litellm,
        "cost_per_token",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert savings_tracker_module._resolve_litellm_model("mystery-model") == "mystery-model"
    assert savings_tracker_module._estimate_compression_savings_usd("mystery-model", 100) == 0.0

    monkeypatch.setattr(savings_tracker_module, "LITELLM_AVAILABLE", False)
    assert savings_tracker_module._estimate_compression_savings_usd("gpt-4o", 100) == 0.0


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
