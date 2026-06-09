"""Engine-contract parity replay test (Chunk 3).

Loads every fixture under tests/parity/fixtures/engine_contract/*.json,
builds a RequestContext from the stored inputs, calls HeadroomEngine.on_request,
and asserts the resulting body is byte-for-byte identical to the golden
expected_body stored in the fixture.

This same fixture set will be replayed against the Rust engine in D2.
The loader (_load_fixture / _build_ctx) is therefore written to be
importable independently of HeadroomEngine so the D2 harness can reuse the
fixture-reading logic without pulling in Python-proxy internals.

Compression fixtures are deferred to Chunk 4 — see README in the fixtures dir.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from headroom.engine.contract import Flavor, Provider, RequestContext

# ---------------------------------------------------------------------------
# Fixture schema / loader — language-neutral, importable standalone
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "engine_contract"


@dataclass(frozen=True)
class EngineFixture:
    """Parsed representation of one engine_contract fixture file.

    Fields mirror the JSON schema documented in the fixtures/engine_contract/README.
    raw_body and expected_body are already decoded from base64 to bytes.
    """

    name: str
    provider: Provider
    flavor: Flavor
    auth_mode: str  # "payg" | "oauth" | "subscription" — informational
    headers: dict[str, str]
    session_key: str
    config_optimize: bool  # HeadroomConfig.optimize value used at recording time
    raw_body: bytes
    expected_body: bytes
    passthrough: bool
    passthrough_reason: str | None
    recorded_at: str
    extra: dict[str, Any] = field(default_factory=dict)


def _load_fixture(path: Path) -> EngineFixture:
    """Parse one engine_contract fixture file.

    Raises loudly on any malformed input — no silent fallbacks.
    """
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in fixture {path}: {exc}") from exc

    required = {
        "name",
        "provider",
        "flavor",
        "auth_mode",
        "headers",
        "session_key",
        "config_optimize",
        "raw_body_b64",
        "expected_body_b64",
        "passthrough",
        "recorded_at",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Fixture {path} missing required keys: {missing!r}")

    try:
        provider = Provider(data["provider"])
    except ValueError as exc:
        raise ValueError(f"Fixture {path}: unknown provider {data['provider']!r}") from exc

    try:
        flavor = Flavor(data["flavor"])
    except ValueError as exc:
        raise ValueError(f"Fixture {path}: unknown flavor {data['flavor']!r}") from exc

    try:
        raw_body = base64.b64decode(data["raw_body_b64"])
    except Exception as exc:
        raise ValueError(f"Fixture {path}: bad raw_body_b64: {exc}") from exc

    try:
        expected_body = base64.b64decode(data["expected_body_b64"])
    except Exception as exc:
        raise ValueError(f"Fixture {path}: bad expected_body_b64: {exc}") from exc

    known_keys = required | {"passthrough_reason"}
    extra = {k: v for k, v in data.items() if k not in known_keys}

    return EngineFixture(
        name=data["name"],
        provider=provider,
        flavor=flavor,
        auth_mode=data["auth_mode"],
        headers=dict(data["headers"]),
        session_key=str(data["session_key"]),
        config_optimize=bool(data["config_optimize"]),
        raw_body=raw_body,
        expected_body=expected_body,
        passthrough=bool(data["passthrough"]),
        passthrough_reason=data.get("passthrough_reason"),
        recorded_at=data["recorded_at"],
        extra=extra,
    )


def _build_ctx(fix: EngineFixture) -> RequestContext:
    """Construct a RequestContext from a parsed EngineFixture."""
    return RequestContext(
        provider=fix.provider,
        flavor=fix.flavor,
        headers_view=fix.headers,
        raw_body=fix.raw_body,
        session_key=fix.session_key,
    )


def _all_fixtures() -> list[EngineFixture]:
    """Load all *.json fixtures from the engine_contract directory.

    Returns an empty list (not an error) when the directory has no JSON files —
    the test is parametrized on the result, so an empty list means 0 tests,
    which will fail the suite via the `--collect-only` guard below.
    """
    paths = sorted(_FIXTURE_DIR.glob("*.json"))
    return [_load_fixture(p) for p in paths]


# ---------------------------------------------------------------------------
# Fake pipeline — never invoked on passthrough but required by the engine
# ---------------------------------------------------------------------------


class _NeverCalledPipeline:
    """Registered for (provider, flavor) pairs to satisfy the engine's
    KeyError guard. Raises loudly if actually invoked — passthrough
    fixtures must never reach the pipeline."""

    def apply(self, messages: list[Any], model: str, **kwargs: Any) -> Any:
        raise AssertionError(
            "_NeverCalledPipeline.apply was called — this indicates the engine "
            "incorrectly routed a passthrough fixture to the compression path."
        )


# ---------------------------------------------------------------------------
# Config stub
# ---------------------------------------------------------------------------


class _Config:
    def __init__(self, optimize: bool = True) -> None:
        self.optimize = optimize


# ---------------------------------------------------------------------------
# Parametrized replay test
# ---------------------------------------------------------------------------

_FIXTURES = _all_fixtures()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize test_replay_fixture over all loaded fixtures at collection time."""
    if "engine_fixture" in metafunc.fixturenames:
        metafunc.parametrize(
            "engine_fixture",
            _FIXTURES,
            ids=[f.name for f in _FIXTURES],
        )


def test_replay_fixture(engine_fixture: EngineFixture) -> None:
    """Replay one engine_contract fixture and assert byte-exact output."""
    from headroom.engine.facade import HeadroomEngine

    fix = engine_fixture
    key = (fix.provider, fix.flavor)

    # Every registered pipeline is the NeverCalled sentinel — passthrough
    # fixtures must not invoke it. Compression fixtures (Chunk 4) will
    # register a real pipeline instead.
    engine = HeadroomEngine(
        pipelines={key: _NeverCalledPipeline()},
        config=_Config(optimize=fix.config_optimize),
        usage_reporter=None,
        salt=b"parity-salt",
    )

    ctx = _build_ctx(fix)
    decision = engine.on_request(ctx)

    assert decision.body == fix.expected_body, (
        f"Fixture '{fix.name}': body mismatch.\n"
        f"  got      ({len(decision.body)} bytes): {decision.body[:120]!r}\n"
        f"  expected ({len(fix.expected_body)} bytes): {fix.expected_body[:120]!r}"
    )

    if fix.passthrough:
        assert decision.telemetry.compressed is False, (
            f"Fixture '{fix.name}': expected passthrough but telemetry.compressed=True"
        )


# ---------------------------------------------------------------------------
# Guard: fail if no fixtures found (catches accidental empty dir)
# ---------------------------------------------------------------------------


def test_fixture_dir_has_fixtures() -> None:
    """Ensure at least one engine_contract fixture exists.

    This test fails if the fixture directory is empty (e.g. after a bad merge
    that deleted them), making the missing-fixtures case loud rather than
    silently passing with zero parametrized cases.
    """
    assert _FIXTURES, (
        "No engine_contract fixtures found in "
        f"{_FIXTURE_DIR}. Run scripts/record_engine_fixtures.py to seed them."
    )
