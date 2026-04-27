"""Smart JSON array crusher — Rust-backed via PyO3.

The Python implementation has been retired (Stage 3c.1b, 2026-04-27).
All array compression now goes through `headroom._core.SmartCrusher`
(built from `crates/headroom-py`). Byte-equality of the two
implementations was verified against 17 recorded fixtures
(`tests/parity/fixtures/smart_crusher/`) before the Python source was
removed; the Rust crate has its own coverage in `crates/headroom-core/`
(388 unit tests + property tests).

This module retains the public surface — `SmartCrusherConfig`,
`CrushResult`, `SmartCrusher`, `smart_crush_tool_output` — so existing
call sites keep working unchanged. The dataclasses are still pure
Python because callers use `asdict()`, `__dict__`, and dataclass
matching on them. Only the `SmartCrusher` class delegates to Rust.

The `headroom._core` extension is a hard import: there is no Python
fallback. Build it locally with `scripts/build_rust_extension.sh`
(wraps `maturin develop`) or install a prebuilt wheel.

Stage 3c.1 deliberately keeps the optional subsystems (TOIN,
feedback, CCR marker injection, telemetry) disabled in the Rust port.
The shim accepts `relevance_config`, `scorer`, and `ccr_config`
constructor args for source compatibility but does not wire them
through — they re-attach in Stage 3c.2 when those subsystems land in
Rust. CCR marker injection in `_smart_crush_content` is a Stage 3c.2
follow-up; today the Rust port never emits CCR markers, so the
disabled-path behavior is byte-equal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import CCRConfig, TransformResult
from ..tokenizer import Tokenizer
from ..utils import compute_short_hash, create_tool_digest_marker, deep_copy_messages
from .base import Transform

logger = logging.getLogger(__name__)


# ─── Public dataclasses ───────────────────────────────────────────────────


@dataclass
class CrushResult:
    """Result from `SmartCrusher.crush()`.

    Used by `ContentRouter` when routing JSON arrays to `SmartCrusher`.
    """

    compressed: str
    original: str
    was_modified: bool
    strategy: str = "passthrough"


@dataclass
class SmartCrusherConfig:
    """Configuration for SmartCrusher.

    SCHEMA-PRESERVING: output contains only items from the original
    array. No wrappers, no generated text, no metadata keys.

    Field names + defaults match the Rust `SmartCrusherConfig` byte-for-
    byte; the shim copies these straight into the PyO3 constructor.
    """

    enabled: bool = True
    min_items_to_analyze: int = 5
    min_tokens_to_crush: int = 200
    variance_threshold: float = 2.0
    uniqueness_threshold: float = 0.1
    similarity_threshold: float = 0.8
    max_items_after_crush: int = 15
    preserve_change_points: bool = True
    factor_out_constants: bool = False
    include_summaries: bool = False
    use_feedback_hints: bool = True
    toin_confidence_threshold: float = 0.5
    dedup_identical_items: bool = True
    first_fraction: float = 0.3
    last_fraction: float = 0.15


# ─── Rust-backed SmartCrusher ─────────────────────────────────────────────


class SmartCrusher(Transform):
    """Rust-backed `SmartCrusher` (via PyO3 / `headroom._core`).

    Same `__init__` and method shapes as the retired Python class —
    drop-in replacement. The `crush()` and `_smart_crush_content()`
    methods delegate every byte to Rust; `apply()` keeps the
    Transform-protocol orchestration in Python (message walking,
    digest-marker insertion, token counting) since that's mostly glue
    around the per-message compression call.
    """

    name = "smart_crusher"

    def __init__(
        self,
        config: SmartCrusherConfig | None = None,
        relevance_config: Any = None,
        scorer: Any = None,
        ccr_config: CCRConfig | None = None,
    ):
        # Hard import — no Python fallback. If the wheel is missing the
        # caller must build it (scripts/build_rust_extension.sh) or
        # install a prebuilt one. Failing loudly is better than silent
        # degradation; see feedback memory `feedback_no_silent_fallbacks.md`.
        from headroom._core import (
            SmartCrusher as _RustSmartCrusher,
        )
        from headroom._core import (
            SmartCrusherConfig as _RustSmartCrusherConfig,
        )

        cfg = config or SmartCrusherConfig()
        self.config = cfg

        # CCR config is preserved on `self` for callers that read it
        # back (`headroom.proxy.server` does), but the Rust port doesn't
        # exercise it: Stage 3c.1 keeps CCR marker injection disabled
        # because the Rust port has no compression store. When Stage
        # 3c.2 lands the CCR port, this wires through to Rust.
        if ccr_config is None:
            self._ccr_config = CCRConfig(enabled=True, inject_retrieval_marker=False)
        else:
            self._ccr_config = ccr_config

        # `relevance_config` and `scorer` are accepted for source
        # compatibility but currently dropped — Stage 3c.1 ships with
        # the Rust default `HybridScorer`. Custom scorers re-attach in
        # Stage 3c.2 when the relevance crate gains a Python-bridged
        # constructor surface.
        if relevance_config is not None or scorer is not None:
            logger.debug(
                "SmartCrusher: relevance_config/scorer args are ignored in "
                "Stage 3c.1 (Rust port uses default HybridScorer). They "
                "will be wired through in Stage 3c.2."
            )

        # Build the Rust crusher with every field from the Python
        # config, plus the relevance_threshold default (0.3) — the
        # Python dataclass doesn't carry that field; it lives on
        # `RelevanceScorerConfig` instead.
        self._rust = _RustSmartCrusher(
            _RustSmartCrusherConfig(
                enabled=cfg.enabled,
                min_items_to_analyze=cfg.min_items_to_analyze,
                min_tokens_to_crush=cfg.min_tokens_to_crush,
                variance_threshold=cfg.variance_threshold,
                uniqueness_threshold=cfg.uniqueness_threshold,
                similarity_threshold=cfg.similarity_threshold,
                max_items_after_crush=cfg.max_items_after_crush,
                preserve_change_points=cfg.preserve_change_points,
                factor_out_constants=cfg.factor_out_constants,
                include_summaries=cfg.include_summaries,
                use_feedback_hints=cfg.use_feedback_hints,
                toin_confidence_threshold=cfg.toin_confidence_threshold,
                dedup_identical_items=cfg.dedup_identical_items,
                first_fraction=cfg.first_fraction,
                last_fraction=cfg.last_fraction,
                relevance_threshold=0.3,
            )
        )

    def crush(self, content: str, query: str = "", bias: float = 1.0) -> CrushResult:
        """Crush a single JSON content string.

        Mirrors the retired Python method. Returns a `CrushResult`
        dataclass so call sites that destructure with `asdict()` keep
        working.
        """
        r = self._rust.crush(content, query, bias)
        return CrushResult(
            compressed=r.compressed,
            original=r.original,
            was_modified=r.was_modified,
            strategy=r.strategy,
        )

    def _smart_crush_content(
        self,
        content: str,
        query_context: str = "",
        tool_name: str | None = None,
        bias: float = 1.0,
    ) -> tuple[str, bool, str]:
        """Apply smart crushing; return `(crushed, was_modified, info)`.

        Mirrors the retired Python method's tuple shape. `tool_name` is
        accepted for API compatibility and currently ignored — Stage
        3c.1 has no per-tool TOIN learning hook.
        """
        crushed, was_modified, info = self._rust.smart_crush_content(content, query_context, bias)
        return crushed, was_modified, info

    def _extract_context_from_messages(self, messages: list[dict[str, Any]]) -> str:
        """Build a query string from the last 5 user messages + recent
        assistant tool-call arguments. Used by `apply()` to derive the
        relevance context per-request.

        Pure Python because it walks the message envelope, not the
        compressed payload. The retired implementation lived inline on
        `SmartCrusher`; preserved here unchanged.
        """
        context_parts: list[str] = []
        user_message_count = 0

        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    context_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                context_parts.append(text)

                user_message_count += 1
                if user_message_count >= 5:
                    break

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        args = func.get("arguments", "")
                        if isinstance(args, str) and args:
                            context_parts.append(args)

        return " ".join(context_parts)

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Transform-protocol entry point. Walks every tool/tool_result
        message, applies SmartCrusher to large enough payloads, and
        replaces the message content with `<crushed>\\n<digest_marker>`.

        Pure orchestration — the per-message compression delegates to
        Rust via `_smart_crush_content`.
        """
        tokens_before = tokenizer.count_messages(messages)
        result_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        markers_inserted: list[str] = []
        warnings: list[str] = []

        query_context = self._extract_context_from_messages(result_messages)
        crushed_count = 0
        frozen_message_count = kwargs.get("frozen_message_count", 0)

        for msg_idx, msg in enumerate(result_messages):
            if msg_idx < frozen_message_count:
                continue

            # OpenAI-style: top-level role=tool with string content.
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str):
                    tokens = tokenizer.count_text(content)
                    if tokens > self.config.min_tokens_to_crush:
                        crushed, was_modified, info = self._smart_crush_content(
                            content, query_context
                        )
                        if was_modified:
                            marker = create_tool_digest_marker(compute_short_hash(content))
                            msg["content"] = crushed + "\n" + marker
                            crushed_count += 1
                            markers_inserted.append(marker)
                            if info:
                                transforms_applied.append(f"smart:{info}")

            # Anthropic-style: content is a list of blocks; each tool_result
            # block has a string content field of its own.
            content = msg.get("content")
            if isinstance(content, list):
                for i, block in enumerate(content):
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_content = block.get("content", "")
                    if not isinstance(tool_content, str):
                        continue
                    tokens = tokenizer.count_text(tool_content)
                    if tokens <= self.config.min_tokens_to_crush:
                        continue

                    crushed, was_modified, info = self._smart_crush_content(
                        tool_content, query_context
                    )
                    if was_modified:
                        marker = create_tool_digest_marker(compute_short_hash(tool_content))
                        content[i]["content"] = crushed + "\n" + marker
                        crushed_count += 1
                        markers_inserted.append(marker)
                        if info:
                            transforms_applied.append(f"smart:{info}")

        if crushed_count > 0:
            transforms_applied.insert(0, f"smart_crush:{crushed_count}")

        tokens_after = tokenizer.count_messages(result_messages)

        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
            warnings=warnings,
        )


# ─── Convenience function ─────────────────────────────────────────────────


def smart_crush_tool_output(
    content: str,
    config: SmartCrusherConfig | None = None,
    ccr_config: CCRConfig | None = None,
) -> tuple[str, bool, str]:
    """Compress a single tool output. Returns `(crushed, was_modified, info)`.

    Convenience wrapper that builds a one-shot `SmartCrusher` per call.
    `ccr_config` is accepted for source compatibility but currently
    not wired through (Stage 3c.1 keeps CCR marker injection
    disabled — Rust port has no compression store).
    """
    crusher = SmartCrusher(config=config, ccr_config=ccr_config)
    return crusher._smart_crush_content(content)
