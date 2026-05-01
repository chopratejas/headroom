"""Smoke tests for `IntelligentContextManager` (Rust-backed Python shim).

The 2148-LOC test file that previously lived here covered the retired
Python implementation, including its COMPRESS_FIRST and SUMMARIZE
cascading strategies and many internal attributes (`.scorer`,
`._compression_store`, `._content_router`, etc.). The Rust port (PR-B)
ships only `DROP_BY_SCORE` in OSS and exposes none of those
implementation-detail attributes.

Coverage now lives in three places:

- **Algorithm** — `crates/headroom-core/src/context/` (50 Rust unit
  tests covering safety rails, candidate building, drop-by-score, and
  CCR-on-drop).
- **PyO3 boundary** — `tests/test_rust_icm_bridge.py` (8 tests
  covering construction, getters, `apply`, `should_apply`, frozen
  prefix protection).
- **Public Python shim** — this file. Tests the
  `Transform`-subclass contract that proxy + client + pipeline call
  sites depend on (constructor signature, `apply` returns a
  `TransformResult`, the `ContextStrategy` enum re-export, etc.).

For COMPRESS_FIRST / SUMMARIZE / memory-tier coverage — those belong
to the Enterprise edition; OSS tests don't cover them.
"""

from __future__ import annotations

import json
from typing import Any

from headroom.config import (
    IntelligentContextConfig,
    ScoringWeights,
    TransformResult,
)
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms.intelligent_context import (
    ContextStrategy,
    IntelligentContextManager,
    MessageScore,
)


def _msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def _tokenizer() -> EstimatingTokenCounter:
    return EstimatingTokenCounter()


# ─── Surface checks ─────────────────────────────────────────────────────────


def test_message_score_reexport_works() -> None:
    """Older callers import MessageScore from this module; preserve."""
    assert MessageScore is not None


def test_context_strategy_enum_values_preserved() -> None:
    """Older callers import ContextStrategy enum members; preserve all."""
    assert ContextStrategy.NONE.value == "none"
    assert ContextStrategy.COMPRESS_FIRST.value == "compress"
    assert ContextStrategy.SUMMARIZE.value == "summarize"
    assert ContextStrategy.DROP_BY_SCORE.value == "drop_scored"
    assert ContextStrategy.HYBRID.value == "hybrid"


# ─── Constructor ────────────────────────────────────────────────────────────


def test_init_with_defaults() -> None:
    m = IntelligentContextManager()
    assert m.config is not None
    assert m.config.enabled is True


def test_init_accepts_full_python_config() -> None:
    cfg = IntelligentContextConfig(
        enabled=True,
        keep_system=True,
        keep_last_turns=3,
        output_buffer_tokens=2000,
        scoring_weights=ScoringWeights(),
    )
    m = IntelligentContextManager(config=cfg)
    assert m.config.keep_last_turns == 3


def test_init_accepts_unused_legacy_args() -> None:
    """`toin`, `summarize_fn`, `observer` are accepted (back-compat)
    but currently unused by the Rust core."""
    m = IntelligentContextManager(
        config=IntelligentContextConfig(),
        toin=None,
        summarize_fn=None,
        observer=object(),  # opaque metric sink
    )
    # No crash; surface-level access works.
    assert m.config is not None


def test_init_logs_when_enterprise_features_requested(caplog) -> None:
    """When the config asks for SUMMARIZE / memory tiers (cut
    strategies), the shim should emit a one-time info-level log so
    operators see the behaviour change at proxy startup."""
    cfg = IntelligentContextConfig(summarization_enabled=True)
    with caplog.at_level("INFO", logger="headroom.transforms.intelligent_context"):
        IntelligentContextManager(config=cfg)
    assert any("DROP_BY_SCORE" in record.message for record in caplog.records), (
        "expected an info-level note about strategy availability"
    )


# ─── should_apply ────────────────────────────────────────────────────────────


def test_should_apply_false_when_disabled() -> None:
    cfg = IntelligentContextConfig(enabled=False)
    m = IntelligentContextManager(config=cfg)
    assert m.should_apply([_msg("user", "x" * 10_000)], _tokenizer(), model_limit=100) is False


def test_should_apply_false_under_budget() -> None:
    m = IntelligentContextManager()
    assert (
        m.should_apply(
            [_msg("user", "hello")],
            _tokenizer(),
            model_limit=128_000,
            output_buffer=4_000,
        )
        is False
    )


def test_should_apply_true_over_budget() -> None:
    m = IntelligentContextManager()
    huge = [_msg("user", "x" * 10_000)]
    assert m.should_apply(huge, _tokenizer(), model_limit=100, output_buffer=0) is True


# ─── apply ───────────────────────────────────────────────────────────────────


def test_apply_under_budget_is_passthrough() -> None:
    m = IntelligentContextManager()
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    result = m.apply(msgs, _tokenizer(), model_limit=128_000)
    assert isinstance(result, TransformResult)
    assert result.messages == msgs
    assert result.tokens_before == result.tokens_after
    assert result.transforms_applied == []


def test_apply_over_budget_returns_transform_result_with_drops() -> None:
    cfg = IntelligentContextConfig(keep_last_turns=1, output_buffer_tokens=0)
    m = IntelligentContextManager(config=cfg)
    msgs = [_msg("user" if i % 2 == 0 else "assistant", f"msg {i} " * 30) for i in range(8)]
    msgs.append(_msg("user", "final"))
    result = m.apply(msgs, _tokenizer(), model_limit=200, output_buffer=0)
    assert isinstance(result, TransformResult)
    assert result.tokens_after < result.tokens_before
    assert len(result.messages) < len(msgs)
    # Last user message protected by keep_last_turns.
    assert result.messages[-1]["content"] == "final"
    # CCR-on-drop default ON → marker emitted.
    assert any("ccr_retrieve" in m for m in result.markers_inserted)


def test_apply_does_not_mutate_input_messages() -> None:
    m = IntelligentContextManager()
    msgs = [_msg("user", "x" * 1000) for _ in range(5)]
    snapshot = json.dumps(msgs)
    m.apply(msgs, _tokenizer(), model_limit=50, output_buffer=0)
    # Caller's list still holds original content.
    assert json.dumps(msgs) == snapshot


def test_apply_protects_system_messages() -> None:
    cfg = IntelligentContextConfig(keep_last_turns=0, output_buffer_tokens=0)
    m = IntelligentContextManager(config=cfg)
    msgs: list[dict[str, Any]] = [_msg("system", "you are helpful")]
    msgs.extend(_msg("user", f"filler {i} " * 50) for i in range(15))
    result = m.apply(msgs, _tokenizer(), model_limit=100, output_buffer=0)
    # System message survives.
    assert any(msg.get("role") == "system" for msg in result.messages)


def test_apply_respects_frozen_message_count() -> None:
    cfg = IntelligentContextConfig(keep_last_turns=0, output_buffer_tokens=0)
    m = IntelligentContextManager(config=cfg)
    msgs: list[dict[str, Any]] = [_msg("user", "FROZEN PREFIX MARKER " * 30)]
    msgs.extend(_msg("user", f"filler {i} " * 30) for i in range(12))
    result = m.apply(
        msgs,
        _tokenizer(),
        model_limit=200,
        output_buffer=0,
        frozen_message_count=1,
    )
    # Frozen prefix message survives.
    assert "FROZEN PREFIX MARKER" in result.messages[0]["content"]


def test_apply_keeps_tool_pair_atomic() -> None:
    """OpenAI tool_call + tool response must drop together or not at all."""
    cfg = IntelligentContextConfig(keep_last_turns=1, output_buffer_tokens=0)
    m = IntelligentContextManager(config=cfg)
    msgs: list[dict[str, Any]] = [_msg("user", f"old turn {i} " * 30) for i in range(5)]
    msgs.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "tool result " * 50},
            _msg("user", "final"),
        ]
    )
    result = m.apply(msgs, _tokenizer(), model_limit=200, output_buffer=0)
    # If the assistant tool_call survives, its tool response must too.
    has_assistant_tc = any(
        msg.get("role") == "assistant" and msg.get("tool_calls") for msg in result.messages
    )
    has_tool_resp = any(
        msg.get("role") == "tool" and msg.get("tool_call_id") == "c1" for msg in result.messages
    )
    # Either both are present or both are absent — never one without the other.
    assert has_assistant_tc == has_tool_resp


# ─── Transform contract ─────────────────────────────────────────────────────


def test_subclasses_transform() -> None:
    """The proxy + client + pipeline pass `IntelligentContextManager`
    as a `Transform`. Verify the inheritance contract holds."""
    from headroom.transforms.base import Transform

    m = IntelligentContextManager()
    assert isinstance(m, Transform)
    assert m.name == "intelligent_context"
