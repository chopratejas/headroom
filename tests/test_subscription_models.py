from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from headroom.subscription import models
from headroom.subscription.models import (
    ExtraUsage,
    HeadroomContribution,
    RateLimitWindow,
    SubscriptionSnapshot,
    SubscriptionState,
    WindowDiscrepancy,
    WindowTokens,
)


def _snapshot(index: int) -> SubscriptionSnapshot:
    return SubscriptionSnapshot(
        five_hour=RateLimitWindow(
            used=index,
            limit=100,
            utilization_pct=float(index),
            resets_at=datetime(2026, 4, 24, 12, tzinfo=timezone.utc) + timedelta(hours=index),
        ),
        seven_day=RateLimitWindow(used=index * 2, limit=200, utilization_pct=float(index)),
        token_prefix=f"token-{index}",
    )


def test_timestamp_and_safe_number_helpers() -> None:
    aware = models._parse_timestamp("2026-04-24T12:00:00Z")
    naive = models._parse_timestamp("2026-04-24T12:00:00")

    assert aware == datetime(2026, 4, 24, 12, tzinfo=timezone.utc)
    assert naive == datetime(2026, 4, 24, 12, tzinfo=timezone.utc)
    assert models._parse_timestamp("bad-value") is None
    assert models._parse_timestamp(123) is None
    assert models._safe_float("1.5") == 1.5
    assert models._safe_float("nan?") is None
    assert models._safe_int("7") == 7
    assert models._safe_int("oops") is None


def test_rate_limit_window_helpers(monkeypatch) -> None:
    now = datetime(2026, 4, 24, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(models, "_utc_now", lambda: now)
    window = RateLimitWindow.from_api_dict(
        {
            "used": "12",
            "limit": 50,
            "utilization": "24.444",
            "resets_at": "2026-04-24T12:05:00Z",
        }
    )

    assert window.seconds_to_reset(now=now) == 300.0
    assert window.to_dict() == {
        "used": 12,
        "limit": 50,
        "utilization_pct": 24.44,
        "resets_at": "2026-04-24T12:05:00Z",
        "seconds_to_reset": 300.0,
    }


def test_extra_usage_helpers() -> None:
    usage = ExtraUsage.from_api_dict(
        {
            "is_enabled": True,
            "monthly_limit": "1234",
            "used_credits": 321,
            "utilization": "15.678",
        }
    )

    assert usage.monthly_limit_usd == 12.34
    assert usage.used_credits_usd == 3.21
    assert usage.to_dict() == {
        "is_enabled": True,
        "monthly_limit_usd": 12.34,
        "used_credits_usd": 3.21,
        "utilization_pct": 15.68,
    }
    assert ExtraUsage().to_dict() == {
        "is_enabled": False,
        "monthly_limit_usd": None,
        "used_credits_usd": None,
        "utilization_pct": None,
    }


def test_subscription_snapshot_round_trip_with_optional_windows() -> None:
    snapshot = SubscriptionSnapshot.from_api_response(
        {
            "five_hour": {
                "used": 12,
                "limit": 100,
                "utilization": 12.3,
                "resets_at": "2026-04-24T15:00:00Z",
            },
            "seven_day": {
                "used": 30,
                "limit": 200,
                "utilization": 15.0,
                "resets_at": "2026-04-30T15:00:00Z",
            },
            "seven_day_opus": {
                "used": 5,
                "limit": 20,
                "utilization": 25.0,
                "resets_at": "2026-04-30T15:00:00Z",
            },
            "seven_day_sonnet": {
                "used": 10,
                "limit": 50,
                "utilization": 20.0,
                "resets_at": "2026-04-30T15:00:00Z",
            },
            "extra_usage": {
                "is_enabled": True,
                "monthly_limit": 5000,
                "used_credits": 1250,
                "utilization": 25.0,
            },
        },
        token="abcdefgh12345",
    )

    data = snapshot.to_dict()
    assert snapshot.token_prefix == "abcdefgh"
    assert data["five_hour"]["used"] == 12
    assert data["seven_day_opus"]["limit"] == 20
    assert data["seven_day_sonnet"]["utilization_pct"] == 20.0
    assert data["extra_usage"]["used_credits_usd"] == 12.5


def test_window_tokens_contribution_and_discrepancy_helpers() -> None:
    tokens = WindowTokens(
        input=10,
        output=5,
        cache_reads=3,
        cache_writes_5m=1,
        cache_writes_1h=2,
        cache_writes_total=3,
        by_model={"sonnet": {"input": 10}},
        weighted_token_equivalent=21.26,
    )
    contribution = HeadroomContribution(
        tokens_submitted=50,
        tokens_saved_compression=20,
        tokens_saved_rtk=5,
        tokens_saved_cache_reads=10,
        compression_savings_usd=1.23456,
        cache_savings_usd=0.33333,
    )
    discrepancy = WindowDiscrepancy(
        kind="surge_pricing",
        description="weighted usage exceeded expectation",
        severity="alert",
        expected_utilization_pct=30.5,
        actual_utilization_pct=66.6,
        delta_pct=36.1,
    )

    assert tokens.total_raw() == 21
    assert tokens.to_dict()["weighted_token_equivalent"] == 21.3
    assert HeadroomContribution().efficiency_pct() == 0.0
    assert contribution.total_saved() == 35
    assert contribution.total_savings_usd() == pytest.approx(1.56789)
    assert contribution.raw_without_headroom() == 75
    assert contribution.efficiency_pct() == 46.7
    assert contribution.to_dict()["savings_usd"]["total"] == 1.5679
    assert discrepancy.to_dict()["severity"] == "alert"


def test_subscription_state_trims_history_and_persisted_records(monkeypatch) -> None:
    now = datetime(2026, 4, 24, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(models, "_utc_now", lambda: now)
    state = SubscriptionState(
        window_tokens=WindowTokens(input=4),
        contribution=HeadroomContribution(tokens_submitted=2),
        last_active_at=now,
    )

    for index in range(105):
        state.add_snapshot(_snapshot(index))
    for index in range(25):
        state.add_discrepancy(WindowDiscrepancy(kind=f"d{index}"))
    state.mark_error("boom")

    assert len(state.history) == 100
    assert state.poll_count == 105
    assert len(state.discrepancies) == 20
    assert state.is_active(active_window_s=60.0) is True
    assert SubscriptionState().is_active() is False
    assert state.to_dict()["last_active_at"] == "2026-04-24T12:00:00Z"
    assert len(state.to_dict()["discrepancies"]) == 5
    assert len(state.to_persist_dict()["history"]) == 20
