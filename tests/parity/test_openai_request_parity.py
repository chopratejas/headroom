"""Chunk 5 + 5R parity test — HeadroomEngine vs OpenAI golden handler output.

For each golden fixture recorded in the OpenAI oracle:
  1. Builds a real HeadroomEngine wired to OpenAIComponents (same TransformPipeline,
     same OpenAIProvider as the proxy server).
  2. Seeds the engine's prefix-tracker store with the fixture's controlled state
     (same _FixedTracker the golden recorder uses).
  3. Builds a RequestContext from the fixture's ``inbound_b64`` bytes + headers,
     using ``Flavor.CHAT`` for /v1/chat/completions fixtures and
     ``Flavor.RESPONSES`` for /v1/responses fixtures.
  4. Calls ``engine.on_request(ctx)`` — sync, no FastAPI involved.
  5. Asserts ``decision.body == fix.outbound_bytes`` (byte-exact).

Scope
-----
All 22 fixtures are now byte-exact: 16 ``/v1/chat/completions`` + 6
``/v1/responses``.  The responses path is wired via
``_on_request_openai_responses`` (Chunk 5R) using the
``_ResponsesCompressor`` adapter which delegates to the real handler method.

Notes for OpenAI CCR/memory/shadow/flip
----------------------------------------
  - CCR on OpenAI is wired via ``CCRComponents`` in the same way as Anthropic —
    the engine already has the hook seam; add the responses orchestrator when needed.
  - Memory injection on OpenAI chat: the live handler uses
    ``append_text_to_latest_user_chat_message`` (not the Anthropic helper);
    wire via ``MemoryComponents`` in the same ``mc is not None`` block pattern.
  - Shadow / flip: the engine ``on`` path already accepts
    ``override_outbound_bytes`` via ``_retry_request`` — no engine change
    needed; just wire the flip flag in the async handler.

Running
-------
  .venv/bin/python -m pytest tests/parity/test_openai_request_parity.py -v
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("fastapi")

from tests.parity.openai_request_recorder import (  # noqa: E402
    OpenAIGoldenFixture,
    _FixedTracker,  # noqa: PLC2701
    _FreshCompressionCache,  # noqa: PLC2701
    load_all_openai_golden_fixtures,
    seed_all_openai_golden_fixtures,
)

# ---------------------------------------------------------------------------
# Seed fixtures (idempotent — no-op if files already exist)
# ---------------------------------------------------------------------------

seed_all_openai_golden_fixtures()
_ALL_FIXTURES: list[OpenAIGoldenFixture] = load_all_openai_golden_fixtures()

# ---------------------------------------------------------------------------
# /v1/responses fixtures — now fully wired (Chunk 5R)
# ---------------------------------------------------------------------------

# All 6 responses fixtures are byte-exact against the engine's
# ``_on_request_openai_responses`` path.  DEFERRED_RESPONSES_FIXTURES is kept
# as an empty set so the guard test below (``test_deferred_responses_fixtures_are_valid_names``)
# still passes without change.
DEFERRED_RESPONSES_FIXTURES: set[str] = set()

# ---------------------------------------------------------------------------
# Controlled session store — deterministic, isolated per fixture
# ---------------------------------------------------------------------------


@dataclass
class _ControlledOpenAIStore:
    """Minimal SessionTrackerStore stand-in for the engine.

    Returns a fixed _FixedTracker for every session and uses a stable
    deterministic session ID so no state leaks between cases.
    """

    tracker: _FixedTracker
    session_id: str = "engine-openai-parity-golden"
    fresh_caches: dict[str, _FreshCompressionCache] = field(default_factory=dict)

    def compute_session_id(self, request: Any, model: str, messages: Any) -> str:
        return self.session_id

    def get_or_create(self, session_id: str, provider: str) -> _FixedTracker:
        return self.tracker

    def get_fresh_cache(self, session_id: str) -> _FreshCompressionCache:
        if session_id not in self.fresh_caches:
            self.fresh_caches[session_id] = _FreshCompressionCache()
        return self.fresh_caches[session_id]


# ---------------------------------------------------------------------------
# Engine factory — builds a real HeadroomEngine for one fixture
# ---------------------------------------------------------------------------


def _build_openai_engine_for_fixture(fix: OpenAIGoldenFixture) -> Any:
    """Build a real HeadroomEngine wired to OpenAIComponents for ``fix``."""
    from headroom.engine.facade import HeadroomEngine, OpenAIComponents
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

    # HeadroomProxy.__init__ builds openai_pipeline and openai_provider.
    proxy = HeadroomProxy(config)

    # Seed tracker with fixture state — same as the golden recorder.
    tracker = _FixedTracker(frozen_count=fix.frozen_count)
    controlled_store = _ControlledOpenAIStore(tracker=tracker)

    oc = OpenAIComponents(
        pipeline=proxy.openai_pipeline,
        provider=proxy.openai_provider,
        session_tracker_store=controlled_store,
        get_compression_cache=controlled_store.get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )

    engine = HeadroomEngine(
        pipelines={},  # no fake-pipeline fallback needed; openai_components handles chat
        config=proxy.config,
        usage_reporter=None,
        salt=b"openai-parity-test-salt",
        openai_components=oc,
    )
    return engine


# ---------------------------------------------------------------------------
# Parametrize over all fixtures
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "openai_golden_fixture" in metafunc.fixturenames:
        metafunc.parametrize(
            "openai_golden_fixture",
            _ALL_FIXTURES,
            ids=[f.name for f in _ALL_FIXTURES],
        )


# ---------------------------------------------------------------------------
# Main parity test
# ---------------------------------------------------------------------------


def test_openai_engine_parity(openai_golden_fixture: OpenAIGoldenFixture) -> None:
    """Engine produces byte-identical outbound body to the recorded golden.

    For /v1/responses fixtures: drives the engine's
    ``_on_request_openai_responses`` path (Chunk 5R) — byte-exact.

    For nondeterministic chat fixtures: existence-check only.

    For all other chat fixtures: byte-exact assertion.
    """
    fix = openai_golden_fixture

    from headroom.engine.contract import Flavor, Provider, RequestContext

    engine = _build_openai_engine_for_fixture(fix)

    # Route to the correct flavor based on the fixture's endpoint.
    flavor = Flavor.RESPONSES if fix.endpoint == "/v1/responses" else Flavor.CHAT

    ctx = RequestContext(
        provider=Provider.OPENAI,
        flavor=flavor,
        headers_view=fix.headers,
        raw_body=fix.inbound_bytes,
        session_key=f"openai-parity-{fix.name}",
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
            got_pretty = json.dumps(got_parsed, indent=2)
            exp_pretty = json.dumps(exp_parsed, indent=2)
        except Exception:
            got_pretty = repr(got[:400])
            exp_pretty = repr(expected[:400])

        pytest.fail(
            f"Fixture '{fix.name}': engine body differs from golden.\n"
            f"  endpoint: {fix.endpoint}\n"
            f"  proxy_config: {fix.proxy_config}\n"
            f"  frozen_count: {fix.frozen_count}\n"
            f"  notes: {fix.notes}\n"
            f"\n--- engine output ({len(got)} bytes) ---\n{got_pretty}\n"
            f"\n--- golden expected ({len(expected)} bytes) ---\n{exp_pretty}\n"
        )


# ---------------------------------------------------------------------------
# Guard: deferred names match actual fixture names
# ---------------------------------------------------------------------------


def test_deferred_responses_fixtures_are_valid_names() -> None:
    """All DEFERRED_RESPONSES_FIXTURES names must correspond to real fixture files.

    Catches typos — a mistyped name would silently skip the xfail guard,
    causing the test to attempt to run against the wrong path.
    """
    known_names = {f.name for f in _ALL_FIXTURES}
    bad = DEFERRED_RESPONSES_FIXTURES - known_names
    assert not bad, (
        f"DEFERRED_RESPONSES_FIXTURES contains names that don't match any fixture "
        f"file: {sorted(bad)}.  Fix the typo or remove the entry."
    )


# ---------------------------------------------------------------------------
# Coverage summary (always passes — for human review in -v output)
# ---------------------------------------------------------------------------


def test_openai_engine_parity_coverage_summary() -> None:
    """Print a coverage breakdown (always passes; for human review in -v output)."""
    chat_fixtures = [f for f in _ALL_FIXTURES if f.endpoint == "/v1/chat/completions"]
    responses_fixtures = [f for f in _ALL_FIXTURES if f.endpoint == "/v1/responses"]
    byte_exact_chat = [f for f in chat_fixtures if not f.nondeterministic_flag]

    assert len(byte_exact_chat) >= 16, (
        f"Expected at least 16 byte-exact chat fixtures; found {len(byte_exact_chat)}"
    )
    assert len(responses_fixtures) >= 6, (
        f"Expected at least 6 /v1/responses fixtures; found {len(responses_fixtures)}"
    )
    assert len(DEFERRED_RESPONSES_FIXTURES) == 0, (
        "DEFERRED_RESPONSES_FIXTURES should be empty — all responses fixtures "
        "are now wired (Chunk 5R)"
    )
