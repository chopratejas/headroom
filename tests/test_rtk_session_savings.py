"""Session RTK savings must be the delta from the proxy-startup baseline.

Regression for the scope-mixing bug: the dashboard's *session* RTK number must
be computed from token deltas since the baseline pinned at proxy startup — NOT
from RTK's lifetime average (which dilutes a 62%-this-session rate down to an
18.5% all-time number). This exercises the real ``_get_context_tool_stats()``
plumbing rather than asserting the arithmetic in the abstract.
"""

from __future__ import annotations

import headroom.proxy.helpers as helpers


def _reset(monkeypatch):
    monkeypatch.delenv(helpers._RTK_GAIN_SCOPE_ENV, raising=False)
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "rtk")
    helpers._context_tool_stats_cache.update(
        {"expires_at": 0.0, "has_value": False, "tool": None, "value": None}
    )
    helpers._context_tool_session_baseline.update(
        {
            "initialized": False,
            "tool": None,
            "total_commands": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tokens_saved": 0,
            "total_time_ms": 0,
            "captured_at": 0.0,
        }
    )


def _bust_cache():
    helpers._context_tool_stats_cache.update(
        {"expires_at": 0.0, "has_value": False, "tool": None, "value": None}
    )


def test_session_savings_is_delta_not_lifetime_average(monkeypatch):
    _reset(monkeypatch)

    state: dict = {"summary": None}

    def fake_lifetime(tool):
        return helpers._context_tool_summary_payload(
            tool="rtk", installed=True, scope="global", summary=state["summary"]
        )

    monkeypatch.setattr(helpers, "_read_context_tool_lifetime_stats", fake_lifetime)

    # First poll pins the baseline to the current lifetime → session delta is 0,
    # but the lifetime number is preserved untouched.
    state["summary"] = {"total_input": 1000, "total_output": 400, "total_saved": 600}
    first = helpers._get_context_tool_stats()
    assert first is not None
    assert first["session"]["tokens_saved"] == 0
    assert first["lifetime"]["tokens_saved"] == 600

    # Lifetime advances (more RTK commands run this session); the session number
    # is the DELTA, not the 800 lifetime total.
    _bust_cache()
    state["summary"] = {"total_input": 1300, "total_output": 500, "total_saved": 800}
    second = helpers._get_context_tool_stats()
    assert second["session"]["tokens_saved"] == 200  # 800 - 600
    assert second["lifetime"]["tokens_saved"] == 800
    # Session % is derived from the delta (200 saved / 300 input delta), not the
    # lifetime-diluted average.
    assert second["session"]["savings_pct"] == round(200 / 300 * 100, 4)


# --- Failure semantics: a failed read is "no data", never a zero counter ---


def _fake_run_raises(*args, **kwargs):
    import subprocess

    raise subprocess.TimeoutExpired(cmd="rtk", timeout=5)


def test_rtk_reader_returns_none_on_timeout(monkeypatch):
    import headroom.rtk as rtk_mod

    monkeypatch.setattr(rtk_mod, "get_rtk_path", lambda: "/fake/rtk")
    monkeypatch.setattr(helpers, "run", _fake_run_raises)
    assert helpers._read_rtk_lifetime_stats() is None


def test_rtk_reader_returns_none_on_nonzero_exit(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    import headroom.rtk as rtk_mod

    monkeypatch.setattr(rtk_mod, "get_rtk_path", lambda: "/fake/rtk")
    monkeypatch.setattr(
        helpers,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with caplog.at_level(logging.WARNING):
        assert helpers._read_rtk_lifetime_stats() is None
    assert "rtk_stats_subprocess_failed" in caplog.text


def test_rtk_reader_returns_none_on_bad_json(monkeypatch):
    from types import SimpleNamespace

    import headroom.rtk as rtk_mod

    monkeypatch.setattr(rtk_mod, "get_rtk_path", lambda: "/fake/rtk")
    monkeypatch.setattr(
        helpers,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="not-json{", stderr=""),
    )
    assert helpers._read_rtk_lifetime_stats() is None


def test_rtk_reader_not_installed_keeps_zero_payload(monkeypatch):
    import headroom.rtk as rtk_mod

    monkeypatch.setattr(rtk_mod, "get_rtk_path", lambda: None)
    payload = helpers._read_rtk_lifetime_stats()
    assert payload is not None
    assert payload["installed"] is False
    assert payload["tokens_saved"] == 0


def test_lean_ctx_reader_returns_none_on_failure_and_logs(monkeypatch, caplog):
    import logging

    import headroom.lean_ctx as lean_mod

    monkeypatch.setattr(lean_mod, "get_lean_ctx_path", lambda: "/fake/lean-ctx")
    monkeypatch.setattr(helpers, "run", _fake_run_raises)
    with caplog.at_level(logging.WARNING):
        assert helpers._read_lean_ctx_lifetime_stats() is None
    assert "stats_subprocess_failed" in caplog.text


# --- Baseline guards: pin only from successful installed-tool reads ---


def _payload(saved: int, *, input_tokens: int = 1000, installed: bool = True):
    return helpers._context_tool_summary_payload(
        tool="rtk",
        installed=installed,
        scope="global",
        summary={"total_input": input_tokens, "total_output": 400, "total_saved": saved},
    )


def _stub_reads(monkeypatch, sequence):
    calls = {"n": 0}

    def fake(tool):
        idx = min(calls["n"], len(sequence) - 1)
        calls["n"] += 1
        item = sequence[idx]
        return item() if callable(item) else item

    monkeypatch.setattr(helpers, "_read_context_tool_lifetime_stats", fake)
    return calls


def test_transient_failure_does_not_repin_baseline_or_inflate_session(monkeypatch):
    """The headline regression, end to end through the real reader.

    A transient rtk subprocess failure between two identical successful reads
    must not re-pin the session baseline; today the reader converts the
    failure into a synthetic zero payload and recovery reports the tool's
    entire lifetime as session savings.
    """
    import json as json_mod
    from types import SimpleNamespace

    import headroom.rtk as rtk_mod

    _reset(monkeypatch)
    monkeypatch.setattr(rtk_mod, "get_rtk_path", lambda: "/fake/rtk")

    good = json_mod.dumps(
        {"summary": {"total_input": 1000, "total_output": 400, "total_saved": 600}}
    )
    behaviors = [
        lambda: SimpleNamespace(returncode=0, stdout=good, stderr=""),
        _fake_run_raises,
        lambda: SimpleNamespace(returncode=0, stdout=good, stderr=""),
    ]
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        behavior = behaviors[min(calls["n"], len(behaviors) - 1)]
        calls["n"] += 1
        return behavior()

    monkeypatch.setattr(helpers, "run", fake_run)

    first = helpers._get_context_tool_stats()
    assert first["session"]["tokens_saved"] == 0
    assert first["lifetime"]["tokens_saved"] == 600

    _bust_cache()
    helpers._get_context_tool_stats()
    # Baseline survives the failed poll untouched.
    assert helpers._context_tool_session_baseline["tokens_saved"] == 600

    _bust_cache()
    recovered = helpers._get_context_tool_stats()
    # Recovery must NOT report the full lifetime as session savings.
    assert recovered["session"]["tokens_saved"] == 0
    assert recovered["counter_reset_detected"] is False


def test_boot_fail_then_poll_fail_never_pins_zero_baseline(monkeypatch):
    import asyncio

    _reset(monkeypatch)
    _stub_reads(monkeypatch, [None, None, _payload(600)])

    asyncio.run(helpers.initialize_context_tool_session_baseline())
    assert helpers._context_tool_session_baseline["initialized"] is False

    _bust_cache()
    assert helpers._get_context_tool_stats() is None
    # Lazy-init must not have pinned zeros from the failed poll.
    assert helpers._context_tool_session_baseline["initialized"] is False

    _bust_cache()
    recovered = helpers._get_context_tool_stats()
    assert recovered["session"]["tokens_saved"] == 0
    assert recovered["lifetime"]["tokens_saved"] == 600


def test_stats_reset_with_failing_read_defers_to_next_success(monkeypatch):
    import asyncio

    _reset(monkeypatch)
    _stub_reads(monkeypatch, [_payload(600), None, _payload(650)])

    first = helpers._get_context_tool_stats()
    assert first["session"]["tokens_saved"] == 0

    # /stats/reset while rtk is down: old baseline dropped, pin deferred.
    asyncio.run(helpers.initialize_context_tool_session_baseline())
    assert helpers._context_tool_session_baseline["initialized"] is False

    _bust_cache()
    after = helpers._get_context_tool_stats()
    # First successful read after the deferred reset pins fresh: delta 0.
    assert after["session"]["tokens_saved"] == 0
    assert after["lifetime"]["tokens_saved"] == 650


def test_genuine_counter_reset_still_repins(monkeypatch):
    _reset(monkeypatch)
    _stub_reads(monkeypatch, [_payload(600), _payload(50)])

    helpers._get_context_tool_stats()
    _bust_cache()
    second = helpers._get_context_tool_stats()
    assert second["counter_reset_detected"] is True
    assert second["session"]["tokens_saved"] == 0
    assert second["lifetime"]["tokens_saved"] == 50


def test_not_installed_payload_does_not_repin_baseline(monkeypatch):
    _reset(monkeypatch)
    _stub_reads(
        monkeypatch,
        [_payload(600), _payload(0, input_tokens=0, installed=False), _payload(600)],
    )

    helpers._get_context_tool_stats()
    _bust_cache()
    absent = helpers._get_context_tool_stats()
    # Tool vanished at resolution time: honest zeros display, baseline intact.
    assert absent["installed"] is False
    assert helpers._context_tool_session_baseline["tokens_saved"] == 600

    _bust_cache()
    back = helpers._get_context_tool_stats()
    assert back["session"]["tokens_saved"] == 0
    assert back["counter_reset_detected"] is False


def test_tool_switch_with_failing_first_read_does_not_zero_pin(monkeypatch):
    _reset(monkeypatch)
    _stub_reads(monkeypatch, [_payload(600), None])

    helpers._get_context_tool_stats()
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
    _bust_cache()
    assert helpers._get_context_tool_stats() is None
    # Switching tools with a failing first read must not pin a zero baseline
    # for the new tool.
    assert helpers._context_tool_session_baseline.get("tool") != "lean-ctx"


def test_failed_poll_caches_none_for_ttl(monkeypatch):
    _reset(monkeypatch)
    calls = _stub_reads(monkeypatch, [None])

    assert helpers._get_context_tool_stats() is None
    assert helpers._get_context_tool_stats() is None
    # Second call inside the TTL is served from cache — no re-read storm.
    assert calls["n"] == 1
