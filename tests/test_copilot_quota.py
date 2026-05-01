"""Unit tests for headroom.subscription.copilot_quota."""

from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace

import pytest

import headroom.subscription.copilot_quota as quota_module
from headroom.subscription.copilot_quota import (
    CopilotQuotaCategory,
    CopilotQuotaSnapshot,
    CopilotQuotaState,
    _CopilotQuotaTracker,
    discover_github_token,
    get_copilot_quota_tracker,
    parse_copilot_quota,
)

# ---------------------------------------------------------------------------
# CopilotQuotaCategory helpers
# ---------------------------------------------------------------------------


class TestCopilotQuotaCategory:
    def test_used_computed_from_entitlement_and_remaining(self):
        cat = CopilotQuotaCategory(name="chat", entitlement=300, remaining=120)
        assert cat.used == 180

    def test_used_percent_computed(self):
        cat = CopilotQuotaCategory(name="chat", entitlement=100, remaining=25)
        assert cat.used_percent == pytest.approx(75.0)

    def test_used_percent_from_percent_remaining(self):
        cat = CopilotQuotaCategory(name="completions", percent_remaining=40.0)
        assert cat.used_percent == pytest.approx(60.0)

    def test_unlimited_used_percent_is_zero(self):
        cat = CopilotQuotaCategory(name="premium_interactions", unlimited=True)
        assert cat.used_percent == 0.0

    def test_used_none_when_entitlement_missing(self):
        cat = CopilotQuotaCategory(name="chat", remaining=50)
        assert cat.used is None

    def test_to_dict_keys(self):
        cat = CopilotQuotaCategory(
            name="chat",
            entitlement=100,
            remaining=60,
            percent_remaining=60.0,
            overage_count=2,
            overage_permitted=True,
            unlimited=False,
            timestamp_utc="2025-01-01T00:00:00Z",
        )
        d = cat.to_dict()
        assert d["name"] == "chat"
        assert d["entitlement"] == 100
        assert d["remaining"] == 60
        assert d["used"] == 40
        assert d["used_percent"] == pytest.approx(40.0)
        assert d["overage_count"] == 2
        assert d["overage_permitted"] is True
        assert d["unlimited"] is False

    def test_used_percent_clipped_at_zero(self):
        # percent_remaining > 100 should not produce negative used_percent
        cat = CopilotQuotaCategory(name="chat", percent_remaining=110.0)
        assert cat.used_percent == pytest.approx(0.0)

    def test_used_percent_none_when_entitlement_is_zero(self):
        cat = CopilotQuotaCategory(name="chat", entitlement=0, remaining=0)
        assert cat.used_percent is None


# ---------------------------------------------------------------------------
# parse_copilot_quota
# ---------------------------------------------------------------------------


_SAMPLE_RESPONSE = {
    "login": "octocat",
    "copilot_plan": "individual",
    "access_type_sku": "copilot_for_individuals",
    "quota_reset_date_utc": "2025-02-01",
    "quota_snapshots": {
        "chat": {
            "entitlement": 50,
            "remaining": 30,
            "quota_remaining": 30,
            "percent_remaining": 60.0,
            "overage_count": 0,
            "overage_permitted": False,
            "unlimited": False,
            "timestamp_utc": "2025-01-15T10:00:00Z",
        },
        "completions": {
            "entitlement": 2000,
            "remaining": 1500,
            "percent_remaining": 75.0,
            "overage_count": 0,
            "overage_permitted": True,
            "unlimited": False,
            "timestamp_utc": "2025-01-15T10:00:00Z",
        },
        "premium_interactions": {
            "entitlement": 300,
            "remaining": 298,
            "percent_remaining": 99.3,
            "overage_count": 2,
            "overage_permitted": True,
            "unlimited": False,
            "timestamp_utc": "2025-01-15T10:00:00Z",
        },
    },
}


class TestParseCopilotQuota:
    def test_basic_fields(self):
        snap = parse_copilot_quota(_SAMPLE_RESPONSE)
        assert snap.login == "octocat"
        assert snap.copilot_plan == "individual"
        assert snap.access_type_sku == "copilot_for_individuals"
        assert snap.quota_reset_date_utc == "2025-02-01"

    def test_all_categories_parsed(self):
        snap = parse_copilot_quota(_SAMPLE_RESPONSE)
        assert set(snap.categories.keys()) == {"chat", "completions", "premium_interactions"}

    def test_chat_category(self):
        snap = parse_copilot_quota(_SAMPLE_RESPONSE)
        chat = snap.categories["chat"]
        assert chat.entitlement == 50
        assert chat.remaining == 30
        assert chat.percent_remaining == pytest.approx(60.0)
        assert chat.unlimited is False
        assert chat.overage_count == 0

    def test_premium_interactions_overage(self):
        snap = parse_copilot_quota(_SAMPLE_RESPONSE)
        prem = snap.categories["premium_interactions"]
        assert prem.overage_count == 2
        assert prem.overage_permitted is True

    def test_quota_remaining_alias(self):
        """quota_remaining should be used when remaining is absent."""
        data = {
            "quota_snapshots": {
                "chat": {
                    "entitlement": 100,
                    "quota_remaining": 75,
                }
            }
        }
        snap = parse_copilot_quota(data)
        assert snap.categories["chat"].remaining == 75

    def test_unlimited_category(self):
        data = {"quota_snapshots": {"completions": {"unlimited": True}}}
        snap = parse_copilot_quota(data)
        assert snap.categories["completions"].unlimited is True

    def test_empty_quota_snapshots(self):
        snap = parse_copilot_quota({"login": "ghost"})
        assert snap.login == "ghost"
        assert snap.categories == {}

    def test_quota_reset_date_fallback(self):
        data = {"quota_reset_date": "2025-03-01"}
        snap = parse_copilot_quota(data)
        assert snap.quota_reset_date_utc == "2025-03-01"

    def test_fetched_at_is_recent(self):
        before = time.time()
        snap = parse_copilot_quota({})
        after = time.time()
        assert before <= snap.fetched_at <= after

    def test_to_dict_structure(self):
        snap = parse_copilot_quota(_SAMPLE_RESPONSE)
        d = snap.to_dict()
        assert "login" in d
        assert "categories" in d
        assert "chat" in d["categories"]
        assert "used_percent" in d["categories"]["chat"]

    def test_missing_categories_skipped(self):
        data = {
            "quota_snapshots": {
                "chat": {"remaining": 10},
                # completions and premium_interactions absent
            }
        }
        snap = parse_copilot_quota(data)
        assert "chat" in snap.categories
        assert "completions" not in snap.categories
        assert "premium_interactions" not in snap.categories

    def test_free_plan(self):
        data = {"copilot_plan": "free", "quota_snapshots": {}}
        snap = parse_copilot_quota(data)
        assert snap.copilot_plan == "free"


# ---------------------------------------------------------------------------
# discover_github_token
# ---------------------------------------------------------------------------


class TestDiscoverGithubToken:
    def test_returns_none_when_no_env_vars(self, monkeypatch):
        for var in [
            "GITHUB_COPILOT_GITHUB_TOKEN",
            "GITHUB_TOKEN",
            "COPILOT_GITHUB_TOKEN",
            "GITHUB_COPILOT_API_TOKEN",
        ]:
            monkeypatch.delenv(var, raising=False)
        assert discover_github_token() is None

    def test_picks_up_github_token(self, monkeypatch):
        for var in [
            "GITHUB_COPILOT_GITHUB_TOKEN",
            "GITHUB_TOKEN",
            "COPILOT_GITHUB_TOKEN",
            "GITHUB_COPILOT_API_TOKEN",
        ]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_testtoken123")
        assert discover_github_token() == "ghp_testtoken123"

    def test_prefers_copilot_specific_token(self, monkeypatch):
        for var in [
            "GITHUB_COPILOT_GITHUB_TOKEN",
            "GITHUB_TOKEN",
            "COPILOT_GITHUB_TOKEN",
            "GITHUB_COPILOT_API_TOKEN",
        ]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GITHUB_COPILOT_GITHUB_TOKEN", "ghp_copilot_specific")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_generic")
        assert discover_github_token() == "ghp_copilot_specific"

    def test_falls_through_to_next_env_var(self, monkeypatch):
        for var in [
            "GITHUB_COPILOT_GITHUB_TOKEN",
            "GITHUB_TOKEN",
            "COPILOT_GITHUB_TOKEN",
            "GITHUB_COPILOT_API_TOKEN",
        ]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_copilot")
        assert discover_github_token() == "ghp_copilot"

    def test_ignores_empty_strings(self, monkeypatch):
        for var in [
            "GITHUB_COPILOT_GITHUB_TOKEN",
            "GITHUB_TOKEN",
            "COPILOT_GITHUB_TOKEN",
            "GITHUB_COPILOT_API_TOKEN",
        ]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GITHUB_COPILOT_GITHUB_TOKEN", "")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_valid")
        assert discover_github_token() == "ghp_valid"


# ---------------------------------------------------------------------------
# CopilotQuotaSnapshot.to_dict
# ---------------------------------------------------------------------------


class TestCopilotQuotaSnapshot:
    def test_to_dict_complete(self):
        snap = CopilotQuotaSnapshot(
            login="user1",
            copilot_plan="business",
            access_type_sku="copilot_enterprise",
            quota_reset_date_utc="2025-02-01",
        )
        snap.categories["chat"] = CopilotQuotaCategory(name="chat", entitlement=50, remaining=25)
        d = snap.to_dict()
        assert d["login"] == "user1"
        assert d["copilot_plan"] == "business"
        assert "chat" in d["categories"]
        assert d["categories"]["chat"]["entitlement"] == 50


# ---------------------------------------------------------------------------
# Poll-loop task-leak regression
# ---------------------------------------------------------------------------


class TestCopilotQuotaPollLoopLeak:
    @pytest.mark.asyncio
    async def test_poll_loop_does_not_leak_event_wait_tasks(self, monkeypatch):
        """Regression for the ``asyncio.shield(event.wait())`` pattern.

        Matches the equivalent guard in ``tests/test_subscription_tracker.py``:
        every poll interval the loop previously leaked one Event.wait
        waiter because ``asyncio.shield`` prevented ``wait_for`` from
        cancelling the inner wait on timeout.
        """
        import asyncio

        from headroom.subscription.copilot_quota import _CopilotQuotaTracker

        # No token configured → _maybe_poll returns immediately each cycle.
        for var in ("GITHUB_COPILOT_GITHUB_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        tracker = _CopilotQuotaTracker(poll_interval_s=0.05)

        def _count_event_wait() -> int:
            return sum(
                1
                for t in asyncio.all_tasks()
                if (t.get_coro().__qualname__ if t.get_coro() else "") == "Event.wait"
            )

        baseline = _count_event_wait()
        await tracker.start()
        try:
            await asyncio.sleep(0.3)  # ~6 poll cycles
            peak = _count_event_wait()
        finally:
            await tracker.stop()

        await asyncio.sleep(0.05)
        residual = _count_event_wait()

        assert peak - baseline <= 1, (
            f"CopilotQuotaTracker leaked Event.wait: baseline={baseline} peak={peak}"
        )
        assert residual <= baseline, (
            f"CopilotQuotaTracker left residual Event.wait: baseline={baseline} residual={residual}"
        )


def test_copilot_quota_state_to_dict_and_tracker_get_stats() -> None:
    state = CopilotQuotaState()
    tracker = _CopilotQuotaTracker()

    assert state.to_dict()["latest"] is None
    assert tracker.get_stats() is None

    state.latest = CopilotQuotaSnapshot(login="octocat")
    state.last_error = "boom"
    state.last_updated = 123.0
    tracker._state = state

    assert tracker.get_stats()["last_error"] == "boom"


@pytest.mark.asyncio
async def test_tracker_stop_cancels_when_wait_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _CopilotQuotaTracker()
    tracker._stop_event = asyncio.Event()
    tracker._task = asyncio.create_task(asyncio.sleep(60))

    async def fake_wait_for(awaitable, timeout):  # noqa: ANN001, ANN202
        raise asyncio.TimeoutError

    monkeypatch.setattr(quota_module.asyncio, "wait_for", fake_wait_for)

    await tracker.stop()
    await asyncio.sleep(0)

    assert tracker._task.cancelled() is True


def _install_fake_aiohttp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict | None = None,
    status: int = 200,
    ok: bool = True,
    exc: Exception | None = None,
) -> None:
    class FakeTimeout:
        def __init__(self, total: float) -> None:
            self.total = total

    class FakeResponse:
        def __init__(self) -> None:
            self.status = status
            self.ok = ok

        async def __aenter__(self):
            if exc is not None:
                raise exc
            return self

        async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        async def json(self):
            return payload or {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def get(self, url: str, headers: dict[str, str], timeout: FakeTimeout):
            assert url.endswith("/copilot_internal/user")
            assert headers["Authorization"].startswith("Bearer ")
            assert timeout.total == 10
            return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientTimeout=FakeTimeout, ClientSession=lambda: FakeSession()),
    )


@pytest.mark.asyncio
async def test_maybe_poll_covers_response_and_singleton_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _CopilotQuotaTracker()
    monkeypatch.setattr(quota_module, "discover_github_token", lambda: "ghp-test")

    _install_fake_aiohttp(monkeypatch, status=401)
    await tracker._maybe_poll()
    assert tracker._state.last_error == "unauthorized — check GITHUB_TOKEN"

    _install_fake_aiohttp(monkeypatch, status=404)
    await tracker._maybe_poll()
    assert tracker._state.last_error == "endpoint not found (non-Copilot account?)"

    _install_fake_aiohttp(monkeypatch, status=500, ok=False)
    await tracker._maybe_poll()
    assert tracker._state.last_error == "HTTP 500"

    _install_fake_aiohttp(monkeypatch, exc=RuntimeError("network down"))
    await tracker._maybe_poll()
    assert tracker._state.last_error == "network down"

    _install_fake_aiohttp(monkeypatch, payload={"quota_snapshots": {}})
    monkeypatch.setattr(
        quota_module,
        "parse_copilot_quota",
        lambda data: (_ for _ in ()).throw(ValueError("bad")),
    )
    await tracker._maybe_poll()
    assert tracker._state.last_error == "parse error: bad"

    monkeypatch.setattr(quota_module, "parse_copilot_quota", parse_copilot_quota)
    _install_fake_aiohttp(monkeypatch, payload=_SAMPLE_RESPONSE)
    await tracker._maybe_poll()
    assert tracker._state.latest is not None
    assert tracker._state.latest.login == "octocat"
    assert tracker._state.last_error is None
    assert tracker._state.last_updated is not None

    quota_module._singleton = None
    first = get_copilot_quota_tracker()
    second = get_copilot_quota_tracker()
    assert first is second
