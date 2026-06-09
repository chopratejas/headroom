"""Chunk 4.2b parity test — HeadroomEngine CCR request-side vs golden handler output.

For each CCR-ON golden fixture recorded from the handler, this test:
  1. Builds a real HeadroomEngine with CCRComponents wired to controlled stubs.
  2. Calls engine.on_request(ctx) — sync, no FastAPI involved.
  3. Asserts decision.body == fixture.outbound_bytes (byte-exact).

CCR steps covered:
  - Step 1: workspace resolution from headers_view + body
  - Step 2: marker scan + session-sticky tool injection (apply_session_sticky_ccr_tool)
  - Step 2b: system-instruction injection (injector.inject_into_system_message)
  - Step 3: compression tracking — structural test only (tests/engine/test_facade_ccr.py)
  - Step 4: proactive expansion — structural test only (tests/engine/test_facade_ccr.py)

Running
-------
  .venv/bin/python -m pytest tests/parity/test_engine_ccr_parity.py -v
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("fastapi")

from tests.parity.engine_request_recorder import (  # noqa: E402
    GoldenFixture,
    _FixedTracker,  # noqa: PLC2701
    _FreshCompressionCache,  # noqa: PLC2701
    load_ccr_golden_fixtures,
    seed_ccr_golden_fixtures,
)

# Seed CCR golden fixtures (idempotent — skips if file already exists)
seed_ccr_golden_fixtures()
_CCR_FIXTURES: list[GoldenFixture] = load_ccr_golden_fixtures()


# ---------------------------------------------------------------------------
# _ControlledCCRStore — same minimal store as test_engine_request_parity
# ---------------------------------------------------------------------------


@dataclass
class _ControlledCCRStore:
    """Minimal SessionTrackerStore stand-in for CCR parity tests."""

    tracker: _FixedTracker
    session_id: str = "engine-ccr-parity"
    fresh_caches: dict[str, _FreshCompressionCache] = field(default_factory=dict)

    def compute_session_id(self, request: Any, model: str, messages: Any) -> str:
        return self.session_id

    def get_or_create(self, session_id: str, provider: str) -> _FixedTracker:
        return self.tracker

    def get_fresh_cache(self, session_id: str) -> _FreshCompressionCache:
        if session_id not in self.fresh_caches:
            self.fresh_caches[session_id] = _FreshCompressionCache()
        return self.fresh_caches[session_id]


class _EmptyCompressionStore:
    """Stub compression store that always returns None for get_metadata.

    The CCR-ON parity fixtures don't need real stored compression entries —
    the marker scan detects hash strings in the message text, and the
    tool/system-instruction injection decisions are purely based on that
    scan result, not on store entries. Compression tracking (step 3) is
    covered by structural tests.
    """

    def get_metadata(self, hash_key: str) -> dict[str, Any] | None:
        return None


# ---------------------------------------------------------------------------
# Engine factory — builds a HeadroomEngine with CCRComponents for one fixture
# ---------------------------------------------------------------------------


def _build_ccr_engine_for_fixture(fix: GoldenFixture) -> Any:
    """Build a HeadroomEngine with CCRComponents matching the fixture config."""
    from headroom.engine.contract import Flavor, Provider
    from headroom.engine.facade import AnthropicComponents, CCRComponents, HeadroomEngine
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config_kwargs: dict[str, Any] = {
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_handle_responses": False,
        "image_optimize": False,
    }
    config_kwargs.update(fix.proxy_config)
    config = ProxyConfig(**config_kwargs)

    proxy = HeadroomProxy(config)

    tracker = _FixedTracker(frozen_count=fix.frozen_count)
    if fix.prev_original_messages:
        tracker._last_original_messages = list(fix.prev_original_messages)
    if fix.prev_forwarded_messages:
        tracker._last_forwarded_messages = list(fix.prev_forwarded_messages)

    controlled_store = _ControlledCCRStore(tracker=tracker)

    ac = AnthropicComponents(
        pipeline=proxy.anthropic_pipeline,
        provider=proxy.anthropic_provider,
        session_tracker_store=controlled_store,
        get_compression_cache=controlled_store.get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )

    # CCRComponents: no live context_tracker (tracking tested separately);
    # empty compression store (tool injection is marker-scan based, not store-based).
    ccr = CCRComponents(
        ccr_context_tracker=None,
        get_compression_store=_EmptyCompressionStore,
        session_turn_counters={},
    )

    engine = HeadroomEngine(
        pipelines={(Provider.ANTHROPIC, Flavor.MESSAGES): proxy.anthropic_pipeline},
        config=proxy.config,
        usage_reporter=None,
        salt=b"ccr-parity-test-salt",
        anthropic_components=ac,
        ccr_components=ccr,
    )
    return engine


# ---------------------------------------------------------------------------
# Parametrize over the 3 CCR fixtures
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "ccr_fixture" in metafunc.fixturenames:
        metafunc.parametrize(
            "ccr_fixture",
            _CCR_FIXTURES,
            ids=[f.name for f in _CCR_FIXTURES],
        )


# ---------------------------------------------------------------------------
# CCR parity test — byte-exact for all 3 cases
# ---------------------------------------------------------------------------


def test_engine_ccr_parity(ccr_fixture: GoldenFixture) -> None:
    """Engine with CCRComponents produces byte-identical output to the handler golden.

    All 3 CCR cases are byte-exact (tool injection and system-instruction
    injection are deterministic given a controlled session tracker with no
    prior CCR history and a fixed marker in the body).
    """
    fix = ccr_fixture
    from headroom.engine.contract import Flavor, Provider, RequestContext

    engine = _build_ccr_engine_for_fixture(fix)

    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view=fix.headers,
        raw_body=fix.inbound_bytes,
        session_key=f"ccr-parity-{fix.name}",
        request_id="",
    )

    decision = engine.on_request(ctx)
    got = decision.body
    expected = fix.outbound_bytes

    if got != expected:
        try:
            got_parsed = json.loads(got)
            exp_parsed = json.loads(expected)
            got_pretty = json.dumps(got_parsed, indent=2, sort_keys=True)
            exp_pretty = json.dumps(exp_parsed, indent=2, sort_keys=True)
        except Exception:
            got_pretty = repr(got[:300])
            exp_pretty = repr(expected[:300])

        pytest.fail(
            f"CCR fixture '{fix.name}': engine body differs from golden.\n"
            f"  proxy_config: {fix.proxy_config}\n"
            f"  frozen_count: {fix.frozen_count}\n"
            f"  notes: {fix.notes}\n"
            f"\n--- engine output ({len(got)} bytes) ---\n{got_pretty}\n"
            f"\n--- golden expected ({len(expected)} bytes) ---\n{exp_pretty}\n"
        )


# ---------------------------------------------------------------------------
# Guard: count check
# ---------------------------------------------------------------------------


def test_ccr_parity_coverage() -> None:
    """Exactly 3 CCR-ON byte-exact fixtures must be present."""
    assert len(_CCR_FIXTURES) == 3, (
        f"Expected 3 CCR byte-exact fixtures, got {len(_CCR_FIXTURES)}. "
        "Add new fixtures to _CCR_CORPUS in engine_request_recorder.py."
    )
