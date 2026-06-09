"""Engine-contract fixture recorder (Chunk 3).

Runs HeadroomEngine.on_request for a given spec and writes the input→output
pair as a JSON fixture under tests/parity/fixtures/engine_contract/<name>.json.

Fixture format is documented in tests/parity/fixtures/engine_contract/README.
raw_body and expected_body are base64-encoded (RFC 4648 §4, no line wrapping)
so the format is reproducible byte-exactly from Rust in D2.

Usage — called by scripts/record_engine_fixtures.py:

    from tests.parity.engine_recorder import record_engine_fixture, PassthroughSpec
    record_engine_fixture(PassthroughSpec(...))
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from headroom.engine.contract import Flavor, Provider, RequestContext

# repo_root/tests/parity/fixtures/engine_contract
_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "engine_contract"


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassthroughSpec:
    """Input spec for a passthrough fixture.

    ``optimize`` and ``bypass_header`` together control which passthrough
    reason fires (bypass_header, compression_disabled, no_messages).
    """

    name: str
    provider: Provider
    flavor: Flavor
    headers: dict[str, str]
    raw_body: bytes
    session_key: str
    # If True: build engine with optimize=True so the decision is based on
    # the bypass header / empty messages rather than config.
    optimize: bool = True
    # Expected passthrough_reason — recorded for documentation; the engine
    # does not read this field back, but it lets readers audit fixture intent.
    expected_reason: str = "bypass_header"


# ---------------------------------------------------------------------------
# Fake pipeline — never invoked on passthrough
# ---------------------------------------------------------------------------


class _NeverCalledPipeline:
    def apply(self, messages: list[Any], model: str, **kwargs: Any) -> Any:
        raise AssertionError(
            "_NeverCalledPipeline.apply was invoked during fixture recording. "
            "Passthrough fixtures must not reach the compression path."
        )


class _Config:
    def __init__(self, optimize: bool) -> None:
        self.optimize = optimize


# ---------------------------------------------------------------------------
# Auth-mode inference (for the fixture's informational auth_mode field)
# ---------------------------------------------------------------------------


def _infer_auth_mode(headers: dict[str, str]) -> str:
    from headroom.proxy.auth_mode import classify_auth_mode

    return classify_auth_mode(headers).value


# ---------------------------------------------------------------------------
# Core recorder
# ---------------------------------------------------------------------------


def record_engine_fixture(
    spec: PassthroughSpec,
    *,
    root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Run HeadroomEngine.on_request for ``spec`` and write a fixture file.

    Parameters
    ----------
    spec:
        Input specification (provider, flavor, headers, body, etc.).
    root:
        Override the fixture output directory (mostly for unit tests).
    overwrite:
        If False (default) and the fixture file already exists, skip
        recording and return the existing path — idempotent for CI.

    Returns
    -------
    Path
        The path of the written (or already-existing) fixture file.

    Raises
    ------
    AssertionError
        If the engine's decision body does not match the expected passthrough
        body (raw_body) — guards against accidental compression on recording.
    """
    from headroom.engine.facade import HeadroomEngine

    out_dir = root or _FIXTURES_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}.json"

    if out_path.exists() and not overwrite:
        return out_path

    key = (spec.provider, spec.flavor)
    engine = HeadroomEngine(
        pipelines={key: _NeverCalledPipeline()},
        config=_Config(optimize=spec.optimize),
        usage_reporter=None,
        salt=b"parity-salt",
    )

    ctx = RequestContext(
        provider=spec.provider,
        flavor=spec.flavor,
        headers_view=spec.headers,
        raw_body=spec.raw_body,
        session_key=spec.session_key,
    )

    decision = engine.on_request(ctx)

    # Passthrough guard — expected_body must equal raw_body
    if spec.expected_reason in ("bypass_header", "compression_disabled", "no_messages"):
        assert decision.body == spec.raw_body, (
            f"record_engine_fixture: expected passthrough body == raw_body "
            f"for spec '{spec.name}', but got {len(decision.body)} != "
            f"{len(spec.raw_body)} bytes."
        )

    auth_mode = _infer_auth_mode(spec.headers)

    fixture: dict[str, Any] = {
        "name": spec.name,
        "provider": spec.provider.value,
        "flavor": spec.flavor.value,
        "auth_mode": auth_mode,
        "headers": dict(spec.headers),
        "session_key": spec.session_key,
        "config_optimize": spec.optimize,
        "raw_body_b64": base64.b64encode(spec.raw_body).decode(),
        "expected_body_b64": base64.b64encode(decision.body).decode(),
        "passthrough": not decision.telemetry.compressed,
        "passthrough_reason": spec.expected_reason if not decision.telemetry.compressed else None,
        "recorded_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }

    out_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    return out_path


__all__ = ["PassthroughSpec", "record_engine_fixture"]
