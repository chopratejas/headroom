"""Unit tests for headroom.subscription.codex_rate_limits."""

from __future__ import annotations

import time

from headroom.subscription.codex_rate_limits import (
    CodexRateLimitState,
    CodexRateLimitWindow,
    parse_codex_rate_limits,
)

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
