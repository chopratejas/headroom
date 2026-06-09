"""Chunk 4.2a parity test — HeadroomEngine vs golden handler output.

For each golden fixture recorded in Chunk 4.1, this test:
  1. Builds a *real* HeadroomEngine wired to the same compression components
     as the proxy server (same TransformPipeline, same AnthropicProvider, …).
  2. Seeds the engine's prefix-tracker store with the fixture's controlled
     state (same _FixedTracker the golden recorder uses).
  3. Builds a RequestContext from the fixture's ``inbound_b64`` bytes + headers.
  4. Calls ``engine.on_request(ctx)`` — sync, no FastAPI involved.
  5. Asserts ``decision.body == fix.outbound_bytes`` (byte-exact).

Fixtures that need CCR injection, memory injection, or proactive-expansion are
explicitly deferred below (``DEFERRED_FIXTURES``) with a diff-level explanation
of the missing piece.

Running
-------
  .venv/bin/python -m pytest tests/parity/test_engine_request_parity.py -v
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
    load_all_golden_fixtures,
    seed_all_golden_fixtures,
)

# ---------------------------------------------------------------------------
# Seed fixtures (idempotent — no-op if files already exist)
# ---------------------------------------------------------------------------

seed_all_golden_fixtures()
_ALL_FIXTURES: list[GoldenFixture] = load_all_golden_fixtures()

# ---------------------------------------------------------------------------
# Deferred fixtures — explain WHY they need 4.2b/4.2c
# ---------------------------------------------------------------------------

# None of the 18 fixtures in the current corpus exercise CCR-tool-injection,
# memory injection, or proactive-expansion (all are disabled in the recorder's
# _DEFAULT_CONFIG_KWARGS: ccr_inject_tool=False, ccr_handle_responses=False,
# ccr_context_tracking=False, image_optimize=False, memory disabled).
# Therefore the full 18-fixture corpus should pass byte-exact in Chunk 4.2a.
#
# If a future fixture exercises those features, add its name here with a note:
#
#   "fixture_name": "DEFERRED_4.2b: CCR tool injection needs CCRToolInjector
#                    + apply_session_sticky_ccr_tool; engine wires this in 4.2b."
#
DEFERRED_FIXTURES: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Engine factory — builds a real HeadroomEngine for one fixture
# ---------------------------------------------------------------------------


@dataclass
class _ControlledStore:
    """Minimal SessionTrackerStore stand-in for the engine.

    Returns a fixed _FixedTracker for every session and uses a stable
    deterministic session ID so no state leaks between cases.
    """

    tracker: _FixedTracker
    session_id: str = "engine-parity-golden"
    fresh_caches: dict[str, _FreshCompressionCache] = field(default_factory=dict)

    def compute_session_id(self, request: Any, model: str, messages: Any) -> str:
        return self.session_id

    def get_or_create(self, session_id: str, provider: str) -> _FixedTracker:
        return self.tracker

    def get_fresh_cache(self, session_id: str) -> _FreshCompressionCache:
        if session_id not in self.fresh_caches:
            self.fresh_caches[session_id] = _FreshCompressionCache()
        return self.fresh_caches[session_id]


def _build_engine_for_fixture(fix: GoldenFixture) -> Any:
    """Build a real HeadroomEngine wired to the same components as the proxy."""
    from headroom.engine.contract import Flavor, Provider
    from headroom.engine.facade import AnthropicComponents, HeadroomEngine
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    # Build a ProxyConfig matching the fixture's proxy_config.
    # Start from the recorder's default overrides so disabled features stay off.
    config_kwargs: dict[str, Any] = {
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "image_optimize": False,
    }
    config_kwargs.update(fix.proxy_config)
    config = ProxyConfig(**config_kwargs)

    # HeadroomProxy.__init__ builds the pipeline and provider.
    proxy = HeadroomProxy(config)

    # Seed tracker with fixture state — same as the golden recorder.
    tracker = _FixedTracker(frozen_count=fix.frozen_count)
    if fix.prev_original_messages:
        tracker._last_original_messages = list(fix.prev_original_messages)
    if fix.prev_forwarded_messages:
        tracker._last_forwarded_messages = list(fix.prev_forwarded_messages)

    controlled_store = _ControlledStore(tracker=tracker)

    ac = AnthropicComponents(
        pipeline=proxy.anthropic_pipeline,
        provider=proxy.anthropic_provider,
        session_tracker_store=controlled_store,
        get_compression_cache=controlled_store.get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )

    engine = HeadroomEngine(
        pipelines={(Provider.ANTHROPIC, Flavor.MESSAGES): proxy.anthropic_pipeline},
        config=proxy.config,
        usage_reporter=None,
        salt=b"parity-test-salt",
        anthropic_components=ac,
    )
    return engine


# ---------------------------------------------------------------------------
# Parametrize over all 18 fixtures
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "golden_fixture" in metafunc.fixturenames:
        metafunc.parametrize(
            "golden_fixture",
            _ALL_FIXTURES,
            ids=[f.name for f in _ALL_FIXTURES],
        )


# ---------------------------------------------------------------------------
# Main parity test
# ---------------------------------------------------------------------------


def test_engine_parity(golden_fixture: GoldenFixture) -> None:
    """Engine produces byte-identical outbound body to the recorded golden.

    For each fixture:
      - If it's in DEFERRED_FIXTURES: xfail with explanation.
      - If it's nondeterministic_flag=True: existence-check only (matches
        the golden oracle test's semantics).
      - Otherwise: byte-exact assertion.
    """
    fix = golden_fixture

    if fix.name in DEFERRED_FIXTURES:
        pytest.xfail(f"Fixture '{fix.name}' deferred to later chunk: {DEFERRED_FIXTURES[fix.name]}")

    from headroom.engine.contract import Flavor, Provider, RequestContext

    engine = _build_engine_for_fixture(fix)

    # Build RequestContext from inbound_bytes (NOT re-serialized fix.body)
    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view=fix.headers,
        raw_body=fix.inbound_bytes,
        session_key=f"parity-{fix.name}",
        request_id="",
    )

    decision = engine.on_request(ctx)
    got = decision.body

    if fix.nondeterministic_flag:
        assert got, (
            f"Fixture '{fix.name}' (nondeterministic_flag=True): engine produced "
            "empty output, expected at least some bytes."
        )
        return

    expected = fix.outbound_bytes
    if got != expected:
        # Produce a helpful diff
        try:
            got_parsed = json.loads(got)
            exp_parsed = json.loads(expected)
            got_pretty = json.dumps(got_parsed, indent=2, sort_keys=True)
            exp_pretty = json.dumps(exp_parsed, indent=2, sort_keys=True)
        except Exception:
            got_pretty = repr(got[:200])
            exp_pretty = repr(expected[:200])

        pytest.fail(
            f"Fixture '{fix.name}': engine body differs from golden.\n"
            f"  proxy_config: {fix.proxy_config}\n"
            f"  frozen_count: {fix.frozen_count}\n"
            f"  notes: {fix.notes}\n"
            f"\n--- engine output ({len(got)} bytes) ---\n{got_pretty}\n"
            f"\n--- golden expected ({len(expected)} bytes) ---\n{exp_pretty}\n"
        )


# ---------------------------------------------------------------------------
# Guard: deferred fixture names must match actual fixture names
# ---------------------------------------------------------------------------


def test_deferred_fixtures_are_valid_names() -> None:
    """All DEFERRED_FIXTURES names must correspond to real fixture files.

    Catches typos in DEFERRED_FIXTURES — a mistyped name would silently
    make a failing fixture appear as 'passing' (the xfail path is never hit).
    """
    known_names = {f.name for f in _ALL_FIXTURES}
    bad = set(DEFERRED_FIXTURES) - known_names
    assert not bad, (
        f"DEFERRED_FIXTURES contains names that don't match any fixture file: {sorted(bad)}. "
        "Fix the typo or remove the entry."
    )


# ---------------------------------------------------------------------------
# Coverage summary (always passes — for human review in -v output)
# ---------------------------------------------------------------------------


def test_parity_coverage_summary() -> None:
    """Print a coverage breakdown (always passes; for human review in -v output)."""
    by_mode: dict[str, list[str]] = {}
    for fix in _ALL_FIXTURES:
        pc = fix.proxy_config
        mode = pc.get("mode", "token")
        optimize = pc.get("optimize", True)
        key = f"{'passthrough' if not optimize else mode}"
        by_mode.setdefault(key, []).append(fix.name)

    total = len(_ALL_FIXTURES)
    deferred = len(DEFERRED_FIXTURES)
    byte_exact = total - deferred
    assert byte_exact >= 18 - deferred, (
        f"Expected at least {18 - deferred} byte-exact fixtures (got {byte_exact} / {total})"
    )
