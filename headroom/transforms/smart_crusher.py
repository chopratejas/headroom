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

# Functionality state (post-audit, 2026-04-28)

- **TOIN learning** — re-attached. `crush()` and `_smart_crush_content`
  call `toin.record_compression()` after a real compression (filtered on
  `strategy != "passthrough"` to ignore JSON re-canonicalization).
  The retired Python class did this inline; the bridge keeps the
  highest-traffic strategy fueling the learning loop.
- **CCR marker emission** — honored. `ccr_config.inject_retrieval_marker
  =False` flips the Rust crusher's `enable_ccr_marker` field; the
  lossy row-drop path then skips both the marker text and the store
  write. Scope: gates only the row-drop sentinel path. Opaque-string
  CCR substitutions (Stage 3c.2) still emit always — they have no
  Python equivalent.
- **Custom relevance scorer / scorer override** — still not wired.
  `relevance_config` and `scorer` constructor args are accepted for
  source compat but the Rust default `HybridScorer` is always used.
  The shim logs a WARNING when these are passed. See RUST_DEV.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ..config import CCRConfig, TransformResult
from ..tokenizer import Tokenizer
from ..utils import compute_short_hash, create_tool_digest_marker, deep_copy_messages
from .base import Transform

logger = logging.getLogger(__name__)


# ─── CCR sentinel ─────────────────────────────────────────────────────────
#
# When SmartCrusher's lossy path drops rows, it appends a sentinel object
# `{"_ccr_dropped": "<<ccr:HASH N_rows_offloaded>>"}` to the kept-items
# array. The LLM sees this in the prompt and can ask for the original via
# the CCR retrieval tool. Downstream consumers that iterate the array
# expecting a uniform schema (e.g. `for e in entries: e["level"]`) need
# to skip the sentinel — that's what `strip_ccr_sentinels` is for.

CCR_SENTINEL_KEY = "_ccr_dropped"


def is_ccr_sentinel(item: Any) -> bool:
    """True if `item` is a CCR-dropped sentinel object."""
    return isinstance(item, dict) and CCR_SENTINEL_KEY in item


def strip_ccr_sentinels(items: Any) -> Any:
    """Return `items` with any CCR-dropped sentinel objects filtered out.

    Pass this through any iteration over a compressed array's contents
    when your code expects a uniform-schema list of records. The sentinel
    carries a `<<ccr:HASH ...>>` marker for the LLM and shouldn't be
    confused for a record — it has only the `_ccr_dropped` key.

    Non-list inputs pass through unchanged so callers can wrap whatever
    `json.loads` returned without first checking the shape.
    """
    if not isinstance(items, list):
        return items
    return [x for x in items if not is_ccr_sentinel(x)]


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
        with_compaction: bool = True,
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
        self._with_compaction = with_compaction

        # CCR config is preserved on `self` for callers that read it
        # back (`headroom.proxy.server` does). The Rust crusher honors
        # `inject_retrieval_marker` end-to-end via its
        # `enable_ccr_marker` config field — when False, the lossy
        # row-drop path skips marker emission AND skips the CCR store
        # write (no point storing a payload nothing in the prompt can
        # reference).
        #
        # Scope: gates ONLY the row-drop sentinel path. Stage-3c.2
        # opaque-string CCR substitutions still emit always — they have
        # no Python equivalent and no production caller has asked for
        # them to be suppressed.
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
            logger.warning(
                "SmartCrusher: custom relevance_config/scorer args are "
                "currently ignored (Rust port uses default HybridScorer). "
                "Tracked as a known regression in RUST_DEV.md."
            )

        # Lazy TOIN handle. Loaded on first compression that has items
        # to learn from. Skipping import at __init__ keeps cold-start
        # fast for environments where telemetry is disabled.
        self._toin: Any = None
        self._toin_load_failed = False

        # Build the Rust crusher with every field from the Python
        # config, plus the relevance_threshold default (0.3) — the
        # Python dataclass doesn't carry that field; it lives on
        # `RelevanceScorerConfig` instead.
        rust_cfg = _RustSmartCrusherConfig(
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
            enable_ccr_marker=self._ccr_config.inject_retrieval_marker,
        )
        # Default: lossless-first compaction (PR4). Lossless wins for
        # cleanly tabular input where it saves ≥ 30% bytes; otherwise
        # falls through to the lossy path with CCR-Dropped retrieval
        # markers. Pass `with_compaction=False` to opt into the
        # pre-PR4 lossy-only path (used by retention-property tests
        # that depend on row-level item preservation).
        if with_compaction:
            self._rust = _RustSmartCrusher(rust_cfg)
        else:
            self._rust = _RustSmartCrusher.without_compaction(rust_cfg)

    def crush(self, content: str, query: str = "", bias: float = 1.0) -> CrushResult:
        """Crush a single JSON content string.

        Mirrors the retired Python method. Returns a `CrushResult`
        dataclass so call sites that destructure with `asdict()` keep
        working.
        """
        r = self._rust.crush(content, query, bias)
        # Re-attach the TOIN learning loop. The retired Python class
        # recorded compressions into TOIN inline; the Rust port doesn't
        # know about TOIN, and `ContentRouter._record_to_toin` skips
        # SmartCrusher on the assumption SmartCrusher records its own.
        # Bridging the gap here keeps JSON-array compressions fueling
        # the learning system.
        #
        # Filter on `was_modified AND strategy != "passthrough"`. The
        # Rust crusher sometimes flips `was_modified=True` from pure
        # JSON re-canonicalization (whitespace normalization) without
        # actually compressing — the strategy stays `"passthrough"` in
        # that case, and there's no learning value in recording it.
        if r.was_modified and r.strategy != "passthrough":
            self._record_to_toin(
                original=content,
                compressed=r.compressed,
                strategy=r.strategy,
                query_context=query,
                tool_name=None,
            )
        return CrushResult(
            compressed=r.compressed,
            original=r.original,
            was_modified=r.was_modified,
            strategy=r.strategy,
        )

    def crush_array_json(
        self,
        items_json: str,
        query: str = "",
        bias: float = 1.0,
    ) -> dict[str, Any]:
        """Crush a JSON array directly and surface the structured result.

        Returns a dict with `items` (kept rows as JSON), `ccr_hash` (12-char
        hash if rows were dropped), `dropped_summary` (the marker text),
        `strategy_info`, `compacted` (rendered bytes when lossless won),
        and `compaction_kind`.

        Used by tests and by the proxy's CCR retrieval flow when it needs
        the hash directly rather than parsing it out of a prompt marker.
        """
        result: dict[str, Any] = self._rust.crush_array_json(items_json, query, bias)
        return result

    def compact_document_json(self, doc_json: str) -> str:
        """Run the document walker on ``doc_json`` and return compacted JSON.

        Lossless walker pass over objects, arrays, and strings —
        tabular sub-arrays become CSV+schema strings, long opaque
        blobs become ``<<ccr:HASH,KIND,SIZE>>`` markers (originals
        stashed in this crusher's CCR store, so ``ccr_get`` resolves them).

        Use this when callers want pure document-shape compaction
        without per-array lossy crushing.
        """
        result: str = self._rust.compact_document_json(doc_json)
        return result

    def ccr_get(self, hash_key: str) -> str | None:
        """Look up an original payload by CCR hash from the Rust store.

        Returns the canonical-JSON serialization of the original
        `[item, item, ...]` array that the lossy path stashed before
        emitting `<<ccr:HASH ...>>`. Returns ``None`` if the hash is
        unknown, expired, or no store is configured.

        Used by the proxy's CCR retrieval tool to serve the dropped
        rows back to the LLM on demand.
        """
        result: str | None = self._rust.ccr_get(hash_key)
        return result

    def ccr_len(self) -> int:
        """Number of entries currently held by the Rust CCR store."""
        n: int = self._rust.ccr_len()
        return n

    def _smart_crush_content(
        self,
        content: str,
        query_context: str = "",
        tool_name: str | None = None,
        bias: float = 1.0,
    ) -> tuple[str, bool, str]:
        """Apply smart crushing; return `(crushed, was_modified, info)`.

        Mirrors the retired Python method's tuple shape. `tool_name` is
        threaded through to TOIN's per-tool learning records; if no
        tool name is available (e.g. the legacy pipeline doesn't have
        one in scope) the recording uses content-based signature only.
        """
        crushed, was_modified, info = self._rust.smart_crush_content(content, query_context, bias)
        # Same passthrough filter as `crush()` — re-canonicalization of
        # JSON whitespace can flip `was_modified=True` even when the
        # `info` field reports `passthrough` and no compression happened.
        if was_modified and info != "passthrough":
            self._record_to_toin(
                original=content,
                compressed=crushed,
                strategy=info or "smart_crusher",
                query_context=query_context,
                tool_name=tool_name,
            )
        return crushed, was_modified, info

    def _record_to_toin(
        self,
        original: str,
        compressed: str,
        strategy: str,
        query_context: str,
        tool_name: str | None,
    ) -> None:
        """Record a successful compression into TOIN's learning store.

        Replaces the inline TOIN call the retired Python SmartCrusher
        had at the end of its compression path. Best-effort: TOIN
        failures are logged at debug level and never bubble — the
        compression itself has already happened and is correct.

        Token estimates use the `len(json) // 4` rule the retired
        implementation used. The router doesn't pass a tokenizer down
        this far, and re-tokenizing here would dominate the recording
        cost. Rough estimates are fine for learning aggregates.
        """
        if self._toin_load_failed:
            return
        try:
            try:
                items = json.loads(original)
            except (json.JSONDecodeError, ValueError):
                # Not JSON — nothing structural for TOIN to learn from
                # at the array level. The Rust crusher only sets
                # `was_modified=True` on JSON-array inputs, so this
                # branch is rare; bail quietly.
                return
            if not isinstance(items, list):
                return

            from ..telemetry.models import ToolSignature

            signature = ToolSignature.from_items(items)
            original_tokens = max(1, len(original) // 4)
            compressed_tokens = max(1, len(compressed) // 4)

            if self._toin is None:
                from ..telemetry.toin import get_toin

                self._toin = get_toin()

            # Extract the kept-row count from the compressed payload
            # when possible. The lossy path emits a JSON array with a
            # `_ccr_dropped` sentinel suffix; the lossless path emits
            # CSV-schema or compact JSON. For the array case we get an
            # exact compressed_count; otherwise fall back to the rough
            # `original_count` (TOIN cares more about structural
            # signature than count precision).
            try:
                compressed_parsed = json.loads(compressed)
                compressed_count = (
                    len(strip_ccr_sentinels(compressed_parsed))
                    if isinstance(compressed_parsed, list)
                    else len(items)
                )
            except (json.JSONDecodeError, ValueError):
                compressed_count = len(items)

            self._toin.record_compression(
                tool_signature=signature,
                original_count=len(items),
                compressed_count=compressed_count,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                strategy=strategy,
                query_context=query_context if query_context else None,
                items=items[:5],  # Sample for field-level learning
            )
        except ImportError:
            # TOIN module not installed in this build — disable for
            # the lifetime of this crusher to avoid retry overhead.
            self._toin_load_failed = True
        except Exception as e:  # pragma: no cover - best effort
            logger.debug("SmartCrusher TOIN recording failed (non-fatal): %s", e)

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
    with_compaction: bool = True,
) -> tuple[str, bool, str]:
    """Compress a single tool output. Returns `(crushed, was_modified, info)`.

    Convenience wrapper that builds a one-shot `SmartCrusher` per call.
    Defaults to the PR4 lossless-first behavior; pass
    `with_compaction=False` to exercise the legacy lossy-only path
    (still useful for retention-property tests).
    """
    crusher = SmartCrusher(config=config, ccr_config=ccr_config, with_compaction=with_compaction)
    return crusher._smart_crush_content(content)
