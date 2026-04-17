"""Tests for the Anthropic subscription window tracking feature."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from headroom.subscription.models import (
    ExtraUsage,
    HeadroomContribution,
    RateLimitWindow,
    SubscriptionSnapshot,
    SubscriptionState,
    WindowDiscrepancy,
    WindowTokens,
)

# ---------------------------------------------------------------------------
# RateLimitWindow
# ---------------------------------------------------------------------------


class TestRateLimitWindow:
    def test_from_api_dict_full(self):
        now = datetime.now(timezone.utc)
        resets = (now + timedelta(hours=3)).isoformat()
        w = RateLimitWindow.from_api_dict(
            {"utilization": 42.5, "resets_at": resets, "used": 100, "limit": 235}
        )
        assert w.utilization_pct == pytest.approx(42.5)
        assert w.used == 100
        assert w.limit == 235
        assert w.resets_at is not None

    def test_from_api_dict_minimal(self):
        # Real API responses only include utilization + resets_at
        w = RateLimitWindow.from_api_dict({"utilization": 0.0, "resets_at": None})
        assert w.utilization_pct == 0.0
        assert w.resets_at is None
        assert w.used == 0
        assert w.limit == 0

    def test_seconds_to_reset(self):
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=2, minutes=30)
        w = RateLimitWindow(resets_at=future)
        secs = w.seconds_to_reset(now=now)
        assert secs is not None
        assert 9000 <= secs <= 9001

    def test_seconds_to_reset_none_when_no_reset(self):
        w = RateLimitWindow()
        assert w.seconds_to_reset() is None

    def test_to_dict_contains_expected_keys(self):
        w = RateLimitWindow(utilization_pct=55.0)
        d = w.to_dict()
        assert "utilization_pct" in d
        assert "resets_at" in d
        assert "seconds_to_reset" in d


# ---------------------------------------------------------------------------
# ExtraUsage
# ---------------------------------------------------------------------------


class TestExtraUsage:
    def test_from_api_dict_with_cents(self):
        eu = ExtraUsage.from_api_dict(
            {
                "is_enabled": True,
                "monthly_limit": 5000,  # $50.00
                "used_credits": 123,  # $1.23
                "utilization": 2.46,
            }
        )
        assert eu.is_enabled is True
        assert eu.monthly_limit_cents == 5000
        assert eu.used_credits_cents == 123
        assert eu.monthly_limit_usd == pytest.approx(50.0)
        assert eu.used_credits_usd == pytest.approx(1.23)
        assert eu.utilization_pct == pytest.approx(2.46)

    def test_from_api_dict_disabled(self):
        eu = ExtraUsage.from_api_dict({"is_enabled": False})
        assert eu.is_enabled is False
        assert eu.monthly_limit_cents is None
        assert eu.used_credits_cents is None
        assert eu.monthly_limit_usd is None

    def test_to_dict_converts_to_usd(self):
        eu = ExtraUsage(is_enabled=True, monthly_limit_cents=10000, used_credits_cents=499)
        d = eu.to_dict()
        assert d["monthly_limit_usd"] == pytest.approx(100.0, abs=0.01)
        assert d["used_credits_usd"] == pytest.approx(4.99, abs=0.001)


# ---------------------------------------------------------------------------
# SubscriptionSnapshot
# ---------------------------------------------------------------------------


class TestSubscriptionSnapshot:
    def test_from_api_response_all_fields(self):
        now = datetime.now(timezone.utc)
        data = {
            "five_hour": {"utilization": 30.0, "resets_at": (now + timedelta(hours=2)).isoformat()},
            "seven_day": {"utilization": 10.0, "resets_at": (now + timedelta(days=3)).isoformat()},
            "seven_day_opus": {"utilization": 5.0, "resets_at": None},
            "seven_day_sonnet": {"utilization": 8.0, "resets_at": None},
            "extra_usage": {
                "is_enabled": True,
                "monthly_limit": 5000,
                "used_credits": 250,
                "utilization": 5.0,
            },
        }
        snap = SubscriptionSnapshot.from_api_response(data, token="tok_test_abc")
        assert snap.five_hour.utilization_pct == pytest.approx(30.0)
        assert snap.seven_day.utilization_pct == pytest.approx(10.0)
        assert snap.seven_day_opus is not None
        assert snap.seven_day_sonnet is not None
        assert snap.extra_usage.is_enabled is True
        assert snap.extra_usage.used_credits_usd == pytest.approx(2.50)
        assert snap.token_prefix == "tok_test"

    def test_from_api_response_missing_optional_fields(self):
        snap = SubscriptionSnapshot.from_api_response(
            {"five_hour": {"utilization": 0.0, "resets_at": None}}
        )
        assert snap.seven_day_opus is None
        assert snap.seven_day_sonnet is None
        assert snap.extra_usage.is_enabled is False

    def test_to_dict_round_trip(self):
        snap = SubscriptionSnapshot.from_api_response(
            {
                "five_hour": {"utilization": 42.0, "resets_at": None},
                "seven_day": {"utilization": 15.0, "resets_at": None},
                "extra_usage": {"is_enabled": False},
            }
        )
        d = snap.to_dict()
        assert d["five_hour"]["utilization_pct"] == pytest.approx(42.0)
        assert d["seven_day"]["utilization_pct"] == pytest.approx(15.0)
        assert "seven_day_opus" not in d  # absent when None


# ---------------------------------------------------------------------------
# HeadroomContribution
# ---------------------------------------------------------------------------


class TestHeadroomContribution:
    def test_efficiency_pct_no_savings(self):
        c = HeadroomContribution(tokens_submitted=100)
        assert c.efficiency_pct() == 0.0

    def test_efficiency_pct_with_savings(self):
        c = HeadroomContribution(
            tokens_submitted=70,
            tokens_saved_compression=20,
            tokens_saved_rtk=10,
        )
        # raw_without_headroom = 70 + 20 + 10 = 100
        # total_saved = 30
        assert c.efficiency_pct() == pytest.approx(30.0)

    def test_total_savings_usd(self):
        c = HeadroomContribution(compression_savings_usd=1.5, cache_savings_usd=0.75)
        assert c.total_savings_usd() == pytest.approx(2.25)

    def test_to_dict_structure(self):
        c = HeadroomContribution(
            tokens_submitted=200,
            tokens_saved_compression=50,
            compression_savings_usd=0.10,
        )
        d = c.to_dict()
        assert d["tokens_submitted"] == 200
        assert d["tokens_saved"]["compression"] == 50
        assert d["savings_usd"]["compression"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# SubscriptionState
# ---------------------------------------------------------------------------


class TestSubscriptionState:
    def test_is_active_recent(self):
        state = SubscriptionState()
        state.last_active_at = datetime.now(timezone.utc)
        assert state.is_active(active_window_s=60.0) is True

    def test_is_active_stale(self):
        state = SubscriptionState()
        state.last_active_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        assert state.is_active(active_window_s=60.0) is False

    def test_is_active_none(self):
        state = SubscriptionState()
        assert state.is_active() is False

    def test_add_snapshot_caps_history(self):
        state = SubscriptionState()
        state._MAX_HISTORY = 3
        for _ in range(5):
            state.add_snapshot(SubscriptionSnapshot.from_api_response({}))
        assert len(state.history) == 3
        assert state.poll_count == 5

    def test_add_discrepancy_caps_list(self):
        state = SubscriptionState()
        state._MAX_DISCREPANCIES = 2
        for i in range(4):
            state.add_discrepancy(WindowDiscrepancy(kind=f"kind_{i}"))
        assert len(state.discrepancies) == 2


# ---------------------------------------------------------------------------
# SubscriptionTracker — unit tests with mocked client
# ---------------------------------------------------------------------------


class TestSubscriptionTrackerNotifyActive:
    def _make_tracker(self, tmp_path: Path):
        from headroom.subscription.tracker import SubscriptionTracker

        return SubscriptionTracker(
            poll_interval_s=30,
            active_window_s=60,
            persist_path=tmp_path / "state.json",
            client=MagicMock(),
        )

    def test_notify_active_oauth_token(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.notify_active("Bearer sk-ant-oat01-sometoken")
        assert t.is_active() is True
        assert t._current_token == "sk-ant-oat01-sometoken"

    def test_notify_active_ignores_api_key(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.notify_active("Bearer sk-ant-api03-key")
        assert t.is_active() is False
        assert t._current_token is None

    def test_notify_active_ignores_non_bearer(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.notify_active("x-api-key somevalue")
        assert t.is_active() is False

    def test_update_contribution(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.update_contribution(
            tokens_submitted=500,
            tokens_saved_compression=100,
            tokens_saved_cache_reads=50,
        )
        c = t._state.contribution
        assert c.tokens_submitted == 500
        assert c.tokens_saved_compression == 100
        assert c.tokens_saved_cache_reads == 50

    def test_update_contribution_accumulates(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.update_contribution(tokens_submitted=100)
        t.update_contribution(tokens_submitted=200)
        assert t._state.contribution.tokens_submitted == 300

    def test_state_dict_structure(self, tmp_path):
        t = self._make_tracker(tmp_path)
        d = t.state
        assert "latest" in d
        assert "contribution" in d
        assert "poll_count" in d
        assert "last_active_at" in d


# ---------------------------------------------------------------------------
# SubscriptionTracker — poll loop integration test
# ---------------------------------------------------------------------------


class TestSubscriptionTrackerPollLoop:
    @pytest.mark.asyncio
    async def test_poll_called_when_active(self, tmp_path):
        """When notify_active is called, the poll loop should fetch a snapshot."""
        from headroom.subscription.tracker import SubscriptionTracker

        mock_snapshot = SubscriptionSnapshot.from_api_response(
            {
                "five_hour": {"utilization": 25.0, "resets_at": None},
                "seven_day": {"utilization": 5.0, "resets_at": None},
            }
        )
        mock_client = MagicMock()
        mock_client.fetch = AsyncMock(return_value=mock_snapshot)

        tracker = SubscriptionTracker(
            poll_interval_s=1,
            active_window_s=60,
            persist_path=tmp_path / "state.json",
            client=mock_client,
        )

        # Mark active so the poll proceeds
        tracker.notify_active("Bearer sk-ant-oat01-testtoken")

        # Run poll once directly (bypasses loop timing)
        await tracker._maybe_poll()

        assert tracker.latest_snapshot is not None
        assert tracker.latest_snapshot.five_hour.utilization_pct == pytest.approx(25.0)
        mock_client.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_skipped_when_no_token_and_inactive(self, tmp_path):
        """When not active and no credentials file token, poll should skip."""
        from headroom.subscription.tracker import SubscriptionTracker

        mock_client = MagicMock()
        mock_client.fetch = AsyncMock(return_value=None)

        tracker = SubscriptionTracker(
            poll_interval_s=1,
            active_window_s=60,
            persist_path=tmp_path / "state.json",
            client=mock_client,
        )

        with patch("headroom.subscription.client.read_cached_oauth_token", return_value=None):
            await tracker._maybe_poll()

        mock_client.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_loop_does_not_leak_event_wait_tasks(self, tmp_path):
        """The poll loop must not accumulate idle Event.wait waiters.

        Regression for the ``asyncio.shield(event.wait())`` pattern:
        ``shield`` prevents the inner wait from being cancelled when
        ``wait_for`` times out, leaking one Task per poll interval.
        Observed as the "aged proxy degradation" in 2026-04-17 runtime
        analysis — over hours the accumulated idle waiters bog down the
        event-loop scheduler.
        """
        import asyncio

        from headroom.subscription.tracker import SubscriptionTracker

        mock_client = MagicMock()
        mock_client.fetch = AsyncMock(return_value=None)

        tracker = SubscriptionTracker(
            poll_interval_s=0.05,  # short so many iterations fit in the test window
            active_window_s=60,
            persist_path=tmp_path / "state.json",
            client=mock_client,
        )

        def _count_event_wait_tasks() -> int:
            return sum(
                1
                for t in asyncio.all_tasks()
                if (t.get_coro().__qualname__ if t.get_coro() else "") == "Event.wait"
            )

        baseline = _count_event_wait_tasks()
        await tracker.start()
        try:
            # Let the loop run through ~6 iterations. Each iteration would
            # previously leak one Event.wait waiter.
            await asyncio.sleep(0.3)
            peak = _count_event_wait_tasks()
        finally:
            await tracker.stop()

        # After stop the loop task has exited; anything above baseline + 1
        # (the single live wait that was in-flight before stop set the event)
        # indicates leaked waiters.
        await asyncio.sleep(0.05)  # let stop()'s awaited tasks settle
        residual = _count_event_wait_tasks()

        # Peak growth during polling should be at most one (the one currently
        # awaiting inside the loop); anything more is a leak.
        assert peak - baseline <= 1, (
            f"poll loop leaked Event.wait waiters: baseline={baseline} peak={peak}"
        )
        assert residual <= baseline, (
            f"poll loop left residual Event.wait waiters: baseline={baseline} residual={residual}"
        )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


class TestAnomalyDetection:
    def test_surge_pricing_detection(self):
        from headroom.subscription.tracker import _detect_discrepancies

        snap = SubscriptionSnapshot.from_api_response(
            {
                "five_hour": {"utilization": 80.0, "resets_at": None},
                "seven_day": {"utilization": 20.0, "resets_at": None},
            }
        )
        # Simulate API limit known; weighted tokens imply only 50% should be used
        snap.five_hour.limit = 1000
        window_tokens = WindowTokens(input=300, output=100, weighted_token_equivalent=500.0)

        discrepancies = _detect_discrepancies(snap, window_tokens)
        kinds = [d.kind for d in discrepancies]
        assert "surge_pricing" in kinds

    def test_no_surge_when_limit_unknown(self):
        from headroom.subscription.tracker import _detect_discrepancies

        snap = SubscriptionSnapshot.from_api_response(
            {"five_hour": {"utilization": 90.0, "resets_at": None}}
        )
        snap.five_hour.limit = 0  # Unknown
        window_tokens = WindowTokens(weighted_token_equivalent=500.0)

        discrepancies = _detect_discrepancies(snap, window_tokens)
        assert not any(d.kind == "surge_pricing" for d in discrepancies)

    def test_cache_miss_detection(self):
        from headroom.subscription.tracker import _detect_discrepancies

        snap = SubscriptionSnapshot.from_api_response(
            {"five_hour": {"utilization": 40.0, "resets_at": None}}
        )
        # High input, very few cache reads
        window_tokens = WindowTokens(input=100_000, cache_reads=500)

        discrepancies = _detect_discrepancies(snap, window_tokens)
        kinds = [d.kind for d in discrepancies]
        assert "cache_miss" in kinds

    def test_no_cache_miss_below_threshold(self):
        from headroom.subscription.tracker import _detect_discrepancies

        snap = SubscriptionSnapshot.from_api_response(
            {"five_hour": {"utilization": 30.0, "resets_at": None}}
        )
        # Low input total — doesn't trigger cache miss check
        window_tokens = WindowTokens(input=30_000, cache_reads=0)

        discrepancies = _detect_discrepancies(snap, window_tokens)
        assert not any(d.kind == "cache_miss" for d in discrepancies)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


class TestPersistence:
    @pytest.mark.asyncio
    async def test_persist_and_reload(self, tmp_path):
        from headroom.subscription.tracker import SubscriptionTracker

        persist_file = tmp_path / "sub_state.json"

        mock_client = MagicMock()
        snap = SubscriptionSnapshot.from_api_response(
            {"five_hour": {"utilization": 55.0, "resets_at": None}}
        )
        mock_client.fetch = AsyncMock(return_value=snap)

        t1 = SubscriptionTracker(
            persist_path=persist_file,
            client=mock_client,
        )
        t1.notify_active("Bearer sk-ant-oat01-tok")
        t1.update_contribution(tokens_submitted=1000, tokens_saved_compression=200)
        await t1._maybe_poll()
        t1._persist_state()

        assert persist_file.exists()

        # Load a new tracker from the same file
        t2 = SubscriptionTracker(
            persist_path=persist_file,
            client=MagicMock(),
        )
        assert t2._state.contribution.tokens_submitted == 1000
        assert t2._state.contribution.tokens_saved_compression == 200
