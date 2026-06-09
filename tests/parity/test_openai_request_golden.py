"""Characterization golden test for Chunk 5 — OpenAI handler.

Parametrized over every fixture in
``tests/parity/fixtures/engine_request_golden_openai/``.
For each fixture:

1. Re-drive the current OpenAI proxy handler with the same controlled inputs.
2. Assert the intercepted ``outbound_bytes`` are byte-for-byte identical to
   the recorded ``outbound_b64``.

This test is the PARITY ORACLE for the upcoming OpenAI engine path
(Chunk 5 engine task): once the engine's ``/v1/chat/completions`` and
``/v1/responses`` paths take over, this same test (pointed at the new
code) must still pass 100%.

Fixture recording
-----------------
Fixtures are recorded by ``seed_all_openai_golden_fixtures()`` at the
bottom of this module; in CI they are committed and never auto-regenerated.
If you need to refresh a fixture (because the handler was *intentionally*
changed), run::

    HEADROOM_OPENAI_GOLDEN_OVERWRITE=1 python -m pytest \\
        tests/parity/test_openai_request_golden.py -v

Guards
------
* ``test_openai_golden_dir_has_fixtures`` — fails loudly if the directory
  is empty.
* ``test_openai_nondeterministic_cases_flagged`` — asserts the
  known-nondeterministic corpus cases are listed in
  ``DEFERRED_OPENAI_CASES``.
* Each parametrized case re-runs the handler twice and compares the two
  captured bodies (determinism self-check); failure here means the handler
  has a nondeterministic source that should be excluded.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")

from tests.parity.openai_request_recorder import (  # noqa: E402
    _FIXTURES_ROOT,  # noqa: PLC2701
    _OPENAI_CORPUS,  # noqa: PLC2701
    DEFERRED_OPENAI_CASES,
    OpenAIGoldenFixture,
    load_all_openai_golden_fixtures,
    replay_openai_golden_fixture,
    seed_all_openai_golden_fixtures,
)

# ---------------------------------------------------------------------------
# Auto-seed fixtures if directory is empty or HEADROOM_OPENAI_GOLDEN_OVERWRITE=1
# ---------------------------------------------------------------------------

_OVERWRITE = os.environ.get("HEADROOM_OPENAI_GOLDEN_OVERWRITE", "").strip() == "1"

# Seed at import time (collection phase) so parametrize can discover fixtures.
# Re-recording is skipped for existing files unless _OVERWRITE is set.
seed_all_openai_golden_fixtures(overwrite=_OVERWRITE)

# Load all recorded fixtures for parametrization.
_ALL_FIXTURES: list[OpenAIGoldenFixture] = load_all_openai_golden_fixtures()

# ---------------------------------------------------------------------------
# Parametrized characterization test
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize test_openai_golden_parity over all loaded fixtures."""
    if "openai_golden_fixture" in metafunc.fixturenames:
        metafunc.parametrize(
            "openai_golden_fixture",
            _ALL_FIXTURES,
            ids=[f.name for f in _ALL_FIXTURES],
        )


def test_openai_golden_parity(openai_golden_fixture: OpenAIGoldenFixture) -> None:
    """Re-drive the current OpenAI handler and assert byte-exact parity with the golden.

    For nondeterministic_flag=True cases: skip the byte-exact assert and
    only confirm the handler produces *some* output (existence check).  Any
    such case must be accompanied by a comment in openai_request_recorder.py
    explaining why it cannot be byte-exact.
    """
    fix = openai_golden_fixture

    # ── Run 1 ─────────────────────────────────────────────────────────────────
    run1 = replay_openai_golden_fixture(fix)

    if fix.nondeterministic_flag:
        # Existence check only — recorded for reference, not byte parity.
        assert run1, (
            f"Fixture '{fix.name}' (nondeterministic_flag=True): replay produced "
            "empty output, expected at least some bytes."
        )
        return

    # ── Determinism self-check (run 2) ─────────────────────────────────────────
    # Run the handler a second time with identical inputs and confirm both runs
    # agree.  Divergence indicates a nondeterministic source that must be
    # excluded from the byte-exact corpus.
    run2 = replay_openai_golden_fixture(fix)
    assert run1 == run2, (
        f"Fixture '{fix.name}': two consecutive replays produced DIFFERENT bytes "
        f"(run1={len(run1)} bytes, run2={len(run2)} bytes).  "
        "The handler has a nondeterministic source — this fixture MUST be moved "
        "to the deferred list with nondeterministic_flag=True.  "
        f"run1[:80]={run1[:80]!r}  run2[:80]={run2[:80]!r}"
    )

    # ── Byte-exact parity with recorded golden ─────────────────────────────────
    assert run1 == fix.outbound_bytes, (
        f"Fixture '{fix.name}': outbound bytes differ from golden.\n"
        f"  got      ({len(run1)} bytes): {run1[:120]!r}\n"
        f"  expected ({len(fix.outbound_bytes)} bytes): {fix.outbound_bytes[:120]!r}\n"
        f"  endpoint: {fix.endpoint}\n"
        f"  auth_mode: {fix.auth_mode}\n"
        f"  notes: {fix.notes}\n"
        f"  recorded_at: {fix.recorded_at}\n"
        "If the handler was intentionally changed, re-record with "
        "HEADROOM_OPENAI_GOLDEN_OVERWRITE=1."
    )


# ---------------------------------------------------------------------------
# Guard: fixtures directory must not be empty
# ---------------------------------------------------------------------------


def test_openai_golden_dir_has_fixtures() -> None:
    """Fail loudly if the fixture directory is empty.

    Prevents silent zero-test situations after an accidental directory delete.
    """
    assert _ALL_FIXTURES, (
        f"No engine_request_golden_openai fixtures found in {_FIXTURES_ROOT}.  "
        "Run seed_all_openai_golden_fixtures() or set "
        "HEADROOM_OPENAI_GOLDEN_OVERWRITE=1 to record them."
    )


# ---------------------------------------------------------------------------
# Guard: deferred cases are documented
# ---------------------------------------------------------------------------


def test_openai_nondeterministic_cases_flagged() -> None:
    """Assert that all nondeterministic_flag=True corpus cases appear in
    DEFERRED_OPENAI_CASES.

    This is a documentation invariant: anyone who adds a nondeterministic
    case must also list it in DEFERRED_OPENAI_CASES so reviewers can audit
    coverage gaps.
    """
    nondeterministic_in_corpus = {
        spec.name for spec in _OPENAI_CORPUS if spec.nondeterministic_flag
    }
    deferred_set = set(DEFERRED_OPENAI_CASES)
    undocumented = nondeterministic_in_corpus - deferred_set
    assert not undocumented, (
        f"Corpus cases have nondeterministic_flag=True but are not in "
        f"DEFERRED_OPENAI_CASES: {sorted(undocumented)}.  Add them to "
        "DEFERRED_OPENAI_CASES in openai_request_recorder.py."
    )


# ---------------------------------------------------------------------------
# Corpus completeness documentation (informational, always passes)
# ---------------------------------------------------------------------------


def test_openai_corpus_coverage_report() -> None:
    """Print a coverage report (always passes; for human review in -v output).

    Covered axes (byte-exact):

    /v1/chat/completions:
      - PAYG auth (Authorization: Bearer sk-*)
      - OAuth auth (Bearer sk-ant-oat-*)
      - Subscription auth (JWT Bearer + subscription UA)
      - bypass_header (x-headroom-bypass=true)
      - x-headroom-mode=passthrough
      - optimize=False (compression disabled)
      - cache mode (optimize=True, mode=cache)
      - token mode (optimize=True, mode=token)
      - frozen_count=0 (no prior frozen prefix)
      - frozen_count>0 (frozen prefix present)
      - tools present (unsorted → characterizes sort behavior)
      - tools absent
      - streaming=False (non-streaming)
      - streaming=True (streaming path)
      - large role:tool message (exercises ContentRouter → LogCompressor)
      - assistant with tool_calls (OpenAI-specific chat shape with null content)
      - unicode content (no \\uXXXX escaping)
      - numeric precision (float fields)
      - empty messages array

    /v1/responses:
      - PAYG passthrough (string input)
      - instructions field present (cache hot zone passthrough)
      - large function_call_output (exercises _compress_openai_responses_payload)
      - bypass header
      - streaming passthrough
      - reasoning items + reasoning field

    Deferred axes:
      - CCR proactive-expansion (ML nondeterminism)
      - Image compression (codec nondeterminism)
      - Memory injection (async I/O nondeterminism)
      - LLMLingua / IntelligentContext (ML nondeterminism)
      - WebSocket /v1/responses (different transport path)
    """
    covered_chat = [
        spec.name
        for spec in _OPENAI_CORPUS
        if spec.endpoint == "/v1/chat/completions" and not spec.nondeterministic_flag
    ]
    covered_responses = [
        spec.name
        for spec in _OPENAI_CORPUS
        if spec.endpoint == "/v1/responses" and not spec.nondeterministic_flag
    ]
    deferred = list(DEFERRED_OPENAI_CASES)

    assert len(covered_chat) >= 12, (
        f"Expected at least 12 byte-exact chat corpus cases; "
        f"found {len(covered_chat)}: {covered_chat}"
    )
    assert len(covered_responses) >= 4, (
        f"Expected at least 4 byte-exact responses corpus cases; "
        f"found {len(covered_responses)}: {covered_responses}"
    )
    assert len(deferred) >= 4, (
        f"Expected at least 4 deferred cases documented; found {len(deferred)}: {deferred}"
    )
