from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import headroom.subscription as subscription_pkg
import headroom.subscription.tracker as tracker_module
from headroom.subscription.models import (
    HeadroomContribution,
    RateLimitWindow,
    SubscriptionSnapshot,
    WindowDiscrepancy,
    WindowTokens,
    _utc_now,
)
from headroom.subscription.tracker import SubscriptionTracker


def _make_snapshot(
    *, token_prefix: str = "token123", reset_offset_hours: int = 5
) -> SubscriptionSnapshot:
    return SubscriptionSnapshot(
        five_hour=RateLimitWindow(
            used=10,
            limit=100,
            utilization_pct=10.0,
            resets_at=_utc_now() + timedelta(hours=reset_offset_hours),
        ),
        seven_day=RateLimitWindow(used=20, limit=200, utilization_pct=10.0),
        token_prefix=token_prefix,
    )


def test_tracker_notify_active_update_and_basic_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker(enabled=False)

    assert tracker.is_available() is False
    assert tracker.latest_snapshot is None
    assert tracker.is_active() is False
    assert isinstance(tracker.get_stats(), dict)

    tracker.notify_active("")
    tracker.notify_active("Basic token")
    tracker.notify_active("Bearer sk-ant-api-key")
    assert tracker._current_token is None

    tracker.notify_active("Bearer oauth-token-123")
    assert tracker._current_token == "oauth-token-123"
    assert tracker._full_tokens["oauth-to"] == 1
    assert tracker.is_active() is True

    tracker.update_contribution(
        tokens_submitted=10,
        tokens_saved_compression=5,
        tokens_saved_rtk=-1,
        tokens_saved_cache_reads=3,
        compression_savings_usd=1.25,
        cache_savings_usd=-2.0,
    )
    contribution = tracker._state.contribution
    assert contribution.tokens_submitted == 10
    assert contribution.tokens_saved_compression == 5
    assert contribution.tokens_saved_rtk == 0
    assert contribution.tokens_saved_cache_reads == 3
    assert contribution.compression_savings_usd == 1.25
    assert contribution.cache_savings_usd == 0.0


@pytest.mark.asyncio
async def test_tracker_start_stop_and_rollover_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker(persist_path=tmp_path / "state.json")

    async def fake_poll_loop() -> None:
        assert tracker._stop_event is not None
        await tracker._stop_event.wait()

    tracker._poll_loop = fake_poll_loop  # type: ignore[method-assign]
    await tracker.start()
    first_task = tracker._poll_task
    assert first_task is not None

    await tracker.start()
    assert tracker._poll_task is first_task

    await tracker.stop()
    assert tracker._stop_event is not None and tracker._stop_event.is_set()
    assert tracker._persist_path.exists()

    tracker._state.history = [
        _make_snapshot(reset_offset_hours=5),
        _make_snapshot(reset_offset_hours=6),
    ]
    tracker._state.contribution = HeadroomContribution(tokens_submitted=99)
    tracker._maybe_reset_contribution(tracker._state.history[-1])
    assert tracker._state.contribution.tokens_submitted == 0


@pytest.mark.asyncio
async def test_maybe_poll_handles_inactive_and_none_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker()

    monkeypatch.setattr("headroom.subscription.client.read_cached_oauth_token", lambda: None)
    await tracker._maybe_poll()
    assert tracker._state.poll_count == 0

    monkeypatch.setattr(
        "headroom.subscription.client.read_cached_oauth_token", lambda: "cached-token"
    )

    async def fetch_none(token: str | None):
        return None

    tracker._client = SimpleNamespace(fetch=fetch_none)
    await tracker._maybe_poll()
    assert tracker._state.last_error == "fetch returned None"
    assert tracker._state.poll_errors == 1


@pytest.mark.asyncio
async def test_maybe_poll_success_updates_state_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker()
    tracker.notify_active("Bearer live-oauth-token")

    snapshot = _make_snapshot()
    discrepancies = [WindowDiscrepancy(kind="cache_miss", description="miss", severity="warning")]
    metrics_calls: list[dict] = []

    async def fetch_snapshot(token: str | None):
        assert token == "live-oauth-token"
        return snapshot

    tracker._client = SimpleNamespace(fetch=fetch_snapshot)
    monkeypatch.setattr(
        tracker_module, "_compute_window_tokens_for_snapshot", lambda snap: WindowTokens(input=7)
    )
    monkeypatch.setattr(tracker_module, "_detect_discrepancies", lambda snap, tokens: discrepancies)
    monkeypatch.setattr(
        tracker, "_persist_state", lambda: metrics_calls.append({"persisted": True})
    )
    monkeypatch.setitem(
        sys.modules,
        "headroom.observability.metrics",
        SimpleNamespace(
            get_otel_metrics=lambda: SimpleNamespace(
                record_subscription_window=lambda state: metrics_calls.append(state)
            )
        ),
    )

    await tracker._maybe_poll()
    assert tracker.latest_snapshot is snapshot
    assert tracker._state.window_tokens.input == 7
    assert tracker._state.discrepancies[-1].kind == "cache_miss"
    assert tracker._state.last_error is None
    assert tracker._state.poll_count == 1
    assert metrics_calls[0] == {"persisted": True}
    assert isinstance(metrics_calls[1], dict)


def test_persist_and_load_state_round_trip(tmp_path: Path) -> None:
    persist_path = tmp_path / "tracker-state.json"
    tracker = SubscriptionTracker(persist_path=persist_path)
    tracker.update_contribution(
        tokens_submitted=11,
        tokens_saved_compression=2,
        tokens_saved_rtk=3,
        tokens_saved_cache_reads=4,
        compression_savings_usd=1.5,
        cache_savings_usd=2.5,
    )
    tracker._state.poll_count = 7
    tracker._persist_state()

    loader = SubscriptionTracker(persist_path=persist_path)
    assert loader._state.contribution.tokens_submitted == 11
    assert loader._state.contribution.tokens_saved_compression == 2
    assert loader._state.contribution.tokens_saved_rtk == 3
    assert loader._state.contribution.tokens_saved_cache_reads == 4
    assert loader._state.contribution.compression_savings_usd == 1.5
    assert loader._state.contribution.cache_savings_usd == 2.5
    assert loader._state.poll_count == 7

    persist_path.write_text("{invalid", encoding="utf-8")
    broken = SubscriptionTracker(persist_path=persist_path)
    assert broken._state.poll_count == 0

    missing = SubscriptionTracker(persist_path=tmp_path / "missing.json")
    assert missing._state.poll_count == 0


@pytest.mark.asyncio
async def test_tracker_stop_cancels_task_when_wait_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker = SubscriptionTracker()
    tracker._stop_event = asyncio.Event()
    tracker._poll_task = asyncio.create_task(asyncio.sleep(60))

    async def fake_wait_for(awaitable, timeout):  # noqa: ANN001, ANN202
        raise asyncio.TimeoutError

    monkeypatch.setattr(tracker_module.asyncio, "wait_for", fake_wait_for)

    await tracker.stop()
    await asyncio.sleep(0)

    assert tracker._poll_task.cancelled() is True


def test_compute_window_tokens_for_snapshot_and_discrepancy_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _make_snapshot()
    fake_session_tracking = SimpleNamespace(
        compute_window_tokens=lambda start, end: WindowTokens(input=9)
    )
    monkeypatch.setitem(
        sys.modules, "headroom.subscription.session_tracking", fake_session_tracking
    )
    monkeypatch.setattr(subscription_pkg, "session_tracking", fake_session_tracking, raising=False)

    tokens = tracker_module._compute_window_tokens_for_snapshot(snapshot)
    assert tokens.input == 9

    snapshot.five_hour.limit = 100
    snapshot.five_hour.utilization_pct = 90.0
    discrepancies = tracker_module._detect_discrepancies(
        snapshot,
        WindowTokens(input=60_000, cache_reads=1_000, weighted_token_equivalent=55.0),
    )

    assert [item.kind for item in discrepancies] == ["surge_pricing", "cache_miss"]
    assert discrepancies[0].severity == "alert"


def test_compute_window_tokens_for_snapshot_handles_missing_reset_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _make_snapshot()
    snapshot.five_hour.resets_at = None
    assert tracker_module._compute_window_tokens_for_snapshot(snapshot) == WindowTokens()

    def boom(start: float, end: float) -> WindowTokens:
        raise RuntimeError("bad transcript")

    fake_session_tracking = SimpleNamespace(compute_window_tokens=boom)
    monkeypatch.setitem(
        sys.modules, "headroom.subscription.session_tracking", fake_session_tracking
    )
    monkeypatch.setattr(subscription_pkg, "session_tracking", fake_session_tracking, raising=False)
    snapshot.five_hour.resets_at = _utc_now() + timedelta(hours=5)

    assert tracker_module._compute_window_tokens_for_snapshot(snapshot) == WindowTokens()


def test_detect_discrepancies_returns_empty_for_normal_window() -> None:
    snapshot = _make_snapshot()
    snapshot.five_hour.limit = 100
    snapshot.five_hour.utilization_pct = 40.0

    assert (
        tracker_module._detect_discrepancies(
            snapshot,
            WindowTokens(input=1000, cache_reads=500, weighted_token_equivalent=39.5),
        )
        == []
    )


@pytest.mark.asyncio
async def test_configure_get_and_shutdown_tracker_singleton(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    tracker_module._tracker_instance = None

    first = tracker_module.configure_subscription_tracker(
        enabled=False, persist_path=tmp_path / "state-a.json"
    )
    second = tracker_module.configure_subscription_tracker(
        enabled=True, persist_path=tmp_path / "state-b.json"
    )
    assert tracker_module.get_subscription_tracker() is first
    assert second is first

    calls: list[str] = []

    async def fake_stop() -> None:
        calls.append("stopped")

    first.stop = fake_stop  # type: ignore[method-assign]
    await tracker_module.shutdown_subscription_tracker()

    assert calls == ["stopped"]
    assert tracker_module.get_subscription_tracker() is None
