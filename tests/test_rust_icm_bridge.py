"""Smoke tests for the Rust-backed `IntelligentContextManager` PyO3 bridge.

These tests verify the FFI contract — construction, getter access on
`IcmConfig`, the `should_apply` gate, and the basic shape of
`apply()`'s return value. The Rust unit tests in
`crates/headroom-core/src/context/` cover the algorithm itself; this
file only checks that the Python surface stays wired correctly.

PR-D will add the higher-level Python tests against the new shim.
"""

from __future__ import annotations

import json

import pytest

# Hard import — if the wheel isn't built, fail loudly. Same pattern as
# the existing diff_compressor / smart_crusher tests.
from headroom._core import IcmConfig, IntelligentContextManager


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


@pytest.fixture
def manager() -> IntelligentContextManager:
    cfg = IcmConfig(keep_last_turns=1, output_buffer_tokens=0, ccr_on_drop=True)
    return IntelligentContextManager(cfg)


def test_icm_config_defaults_match_oss_intent():
    cfg = IcmConfig()
    assert cfg.enabled is True
    assert cfg.keep_system is True
    assert cfg.keep_last_turns == 2
    assert cfg.output_buffer_tokens == 4000
    assert cfg.ccr_on_drop is True


def test_icm_config_accepts_kwargs():
    cfg = IcmConfig(keep_last_turns=5, output_buffer_tokens=100, ccr_on_drop=False)
    assert cfg.keep_last_turns == 5
    assert cfg.output_buffer_tokens == 100
    assert cfg.ccr_on_drop is False


def test_should_apply_false_under_budget(manager):
    msgs = [_msg("user", "hello")]
    assert manager.should_apply(json.dumps(msgs), model_limit=128_000, output_buffer=4_000) is False


def test_should_apply_true_over_budget(manager):
    msgs = [_msg("user", "x" * 10_000)]
    assert manager.should_apply(json.dumps(msgs), model_limit=100, output_buffer=0) is True


def test_apply_under_budget_is_passthrough(manager):
    msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
    result = manager.apply(json.dumps(msgs), model_limit=128_000)
    assert result.tokens_before == result.tokens_after
    assert result.strategies_applied == []
    out = json.loads(result.messages_json)
    assert out == msgs


def test_apply_over_budget_drops_messages_and_emits_marker(manager):
    msgs = [_msg("user" if i % 2 == 0 else "assistant", f"msg {i} " * 30) for i in range(8)]
    msgs.append(_msg("user", "final"))
    initial_json = json.dumps(msgs)

    result = manager.apply(initial_json, model_limit=200, output_buffer=0)
    out = json.loads(result.messages_json)

    assert result.strategies_applied == ["drop_by_score"]
    assert result.tokens_after < result.tokens_before
    assert len(out) < len(msgs)
    # Final user message is protected by keep_last_turns=1.
    assert out[-1]["content"] == "final"
    # CCR marker emitted.
    assert any("ccr_retrieve" in m for m in result.markers_inserted)


def test_apply_invalid_json_raises(manager):
    # The bridge panics on bad JSON to match the SmartCrusher convention;
    # pyo3 surfaces panics as `pyo3.PanicException` (a subclass of
    # `BaseException` outside the normal Exception hierarchy). Accept any
    # raised exception — realistic callers always pass valid JSON.
    with pytest.raises(BaseException):  # noqa: B017,PT011
        manager.apply("not json", model_limit=1000)


def test_apply_respects_frozen_message_count(manager):
    msgs = [_msg("user", "FROZEN " * 50)]
    for i in range(15):
        msgs.append(_msg("user", f"filler {i} " * 20))
    initial_json = json.dumps(msgs)

    result = manager.apply(initial_json, model_limit=200, output_buffer=0, frozen_message_count=1)
    out = json.loads(result.messages_json)
    # The first message (frozen) must still be present.
    assert "FROZEN" in out[0]["content"]
