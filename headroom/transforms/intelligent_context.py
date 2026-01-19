"""Intelligent context manager for Headroom SDK.

This module provides semantic-aware context management that extends
RollingWindow with importance-based scoring and TOIN integration.

Design principle: NO HARDCODED PATTERNS
All importance signals are derived from:
1. Computed metrics (recency, density, references)
2. TOIN-learned patterns (field_semantics, retrieval_rate)
3. Embedding similarity (optional)

Strategy Selection:
- NONE: Under budget, no action needed
- COMPRESS_FIRST: When <compress_threshold over budget, try deeper compression
  of tool outputs using ContentRouter before dropping messages
- DROP_BY_SCORE: When significantly over budget, drop lowest-scored messages
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..config import IntelligentContextConfig, TransformResult
from ..parser import find_tool_units
from ..tokenizer import Tokenizer
from ..utils import create_dropped_context_marker, deep_copy_messages
from .base import Transform
from .scoring import MessageScore, MessageScorer

if TYPE_CHECKING:
    from ..telemetry.toin import ToolIntelligenceNetwork
    from .content_router import ContentRouter

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Strategy for handling over-budget context."""

    NONE = "none"  # Under budget, do nothing
    COMPRESS_FIRST = "compress"  # Try deeper compression first
    DROP_BY_SCORE = "drop_scored"  # Drop lowest-scored messages
    HYBRID = "hybrid"  # Combination of strategies


class IntelligentContextManager(Transform):
    """
    Intelligent context management with semantic-aware scoring.

    This extends RollingWindow with:
    1. Multi-factor importance scoring
    2. TOIN integration for learned patterns
    3. Score-based dropping instead of position-based
    4. Strategy selection based on budget overage

    Safety guarantees preserved:
    - System messages never dropped (when keep_system=True)
    - Last N turns protected (configurable)
    - Tool call/response pairs kept atomic

    Drop order:
    1. Lowest-scored messages (excluding protected)
    2. Tool units with lowest scores
    3. Only as last resort: older messages by position
    """

    name = "intelligent_context"

    def __init__(
        self,
        config: IntelligentContextConfig | None = None,
        toin: ToolIntelligenceNetwork | None = None,
    ):
        """
        Initialize intelligent context manager.

        Args:
            config: Configuration for context management.
            toin: Optional TOIN instance for learned patterns.
        """
        from ..config import IntelligentContextConfig

        self.config = config or IntelligentContextConfig()
        self.toin = toin

        # Initialize scorer with TOIN if available
        self.scorer = MessageScorer(
            weights=self.config.scoring_weights,
            toin=toin,
            recency_decay_rate=self.config.recency_decay_rate,
        )

        # Lazy-loaded content router for COMPRESS_FIRST strategy
        self._content_router: ContentRouter | None = None

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if context management is needed."""
        if not self.config.enabled:
            return False

        model_limit = kwargs.get("model_limit", 128000)
        output_buffer = kwargs.get("output_buffer", self.config.output_buffer_tokens)

        current_tokens = tokenizer.count_messages(messages)
        available = model_limit - output_buffer

        return bool(current_tokens > available)

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """
        Apply intelligent context management.

        Args:
            messages: List of messages.
            tokenizer: Tokenizer for counting.
            **kwargs: Must include 'model_limit', optionally 'output_buffer'.

        Returns:
            TransformResult with managed messages.
        """
        model_limit = kwargs.get("model_limit", 128000)
        output_buffer = kwargs.get("output_buffer", self.config.output_buffer_tokens)
        available = model_limit - output_buffer

        tokens_before = tokenizer.count_messages(messages)
        result_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        markers_inserted: list[str] = []
        warnings: list[str] = []

        # Early exit if under budget
        current_tokens = tokens_before
        if current_tokens <= available:
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
                warnings=[],
            )

        # Determine strategy based on overage
        strategy = self._select_strategy(current_tokens, available)
        logger.debug(f"IntelligentContextManager: selected strategy {strategy.value}")

        # Get protected indices
        protected = self._get_protected_indices(result_messages)

        # ========== COMPRESS_FIRST STRATEGY ==========
        # Try to compress tool messages before dropping anything
        if strategy == ContextStrategy.COMPRESS_FIRST:
            result_messages, compress_transforms, tokens_saved = self._apply_compress_first(
                result_messages, tokenizer, protected
            )
            transforms_applied.extend(compress_transforms)

            # Recheck token count after compression
            current_tokens = tokenizer.count_messages(result_messages)

            # If now under budget, we're done!
            if current_tokens <= available:
                logger.info(
                    "IntelligentContextManager: COMPRESS_FIRST succeeded, "
                    "saved %d tokens: %d -> %d",
                    tokens_saved,
                    tokens_before,
                    current_tokens,
                )
                return TransformResult(
                    messages=result_messages,
                    tokens_before=tokens_before,
                    tokens_after=current_tokens,
                    transforms_applied=transforms_applied,
                    markers_inserted=markers_inserted,
                    warnings=warnings,
                )

            # Still over budget, fall through to DROP_BY_SCORE
            logger.debug(
                "IntelligentContextManager: COMPRESS_FIRST saved %d tokens but still "
                "over budget (%d > %d), proceeding to DROP_BY_SCORE",
                tokens_saved,
                current_tokens,
                available,
            )
            strategy = ContextStrategy.DROP_BY_SCORE
            # Need to recalculate protected indices after compression
            protected = self._get_protected_indices(result_messages)

        # ========== DROP_BY_SCORE STRATEGY ==========
        # Get tool units for atomic dropping
        tool_units = find_tool_units(result_messages)
        tool_unit_indices = self._get_tool_unit_indices(tool_units)

        # Score all messages
        if self.config.use_importance_scoring:
            scores = self.scorer.score_messages(
                result_messages,
                protected_indices=protected,
                tool_unit_indices=tool_unit_indices,
            )
        else:
            # Fallback to position-based scoring
            scores = self._position_based_scores(result_messages, protected, tool_unit_indices)

        # Build drop candidates sorted by score (lowest first)
        drop_candidates = self._build_scored_drop_candidates(
            result_messages, scores, protected, tool_units
        )

        # Drop until under budget
        indices_to_drop: set[int] = set()
        dropped_count = 0
        tool_units_dropped = 0

        for candidate in drop_candidates:
            if current_tokens <= available:
                break

            candidate_indices = candidate["indices"]

            # Skip if any are protected
            if any(idx in protected for idx in candidate_indices):
                continue

            # Skip if already dropped
            if any(idx in indices_to_drop for idx in candidate_indices):
                continue

            # Calculate tokens saved
            tokens_saved = sum(
                tokenizer.count_message(result_messages[idx])
                for idx in candidate_indices
                if idx < len(result_messages)
            )

            indices_to_drop.update(candidate_indices)
            current_tokens -= tokens_saved
            dropped_count += 1

            if candidate["type"] == "tool_unit":
                tool_units_dropped += 1

        # Remove dropped messages (reverse order)
        for idx in sorted(indices_to_drop, reverse=True):
            if idx < len(result_messages):
                del result_messages[idx]

        # Insert marker if we dropped anything
        if dropped_count > 0:
            logger.info(
                "IntelligentContextManager: dropped %d units (%d tool units) "
                "using strategy %s: %d -> %d tokens",
                dropped_count,
                tool_units_dropped,
                strategy.value,
                tokens_before,
                current_tokens,
            )

            marker = create_dropped_context_marker("intelligent_cap", dropped_count)
            markers_inserted.append(marker)

            # Insert marker after system messages
            insert_idx = 0
            for i, msg in enumerate(result_messages):
                if msg.get("role") != "system":
                    insert_idx = i
                    break
            else:
                insert_idx = len(result_messages)

            result_messages.insert(
                insert_idx,
                {"role": "user", "content": marker},
            )

            transforms_applied.append(f"intelligent_cap:{dropped_count}")

        tokens_after = tokenizer.count_messages(result_messages)

        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
            warnings=warnings,
        )

    def _select_strategy(self, current_tokens: int, available: int) -> ContextStrategy:
        """Select strategy based on how much over budget we are."""
        if current_tokens <= available:
            return ContextStrategy.NONE

        over_ratio = (current_tokens - available) / available

        if over_ratio < self.config.compress_threshold:
            return ContextStrategy.COMPRESS_FIRST

        return ContextStrategy.DROP_BY_SCORE

    def _get_content_router(self) -> ContentRouter | None:
        """Get or create content router for COMPRESS_FIRST strategy (lazy load)."""
        if self._content_router is None:
            try:
                from .content_router import ContentRouter, ContentRouterConfig

                # Configure for aggressive compression in COMPRESS_FIRST context
                router_config = ContentRouterConfig(
                    enable_code_aware=True,
                    enable_llmlingua=True,
                    enable_smart_crusher=True,
                    enable_search_compressor=True,
                    enable_log_compressor=True,
                    skip_user_messages=True,
                    protect_recent_code=0,  # Don't protect in COMPRESS_FIRST
                    protect_analysis_context=False,  # We're over budget
                    min_section_tokens=20,
                    ccr_enabled=True,
                )
                self._content_router = ContentRouter(config=router_config)
            except ImportError:
                logger.debug("ContentRouter not available for COMPRESS_FIRST")
        return self._content_router

    def _apply_compress_first(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        protected: set[int],
    ) -> tuple[list[dict[str, Any]], list[str], int]:
        """Apply deeper compression to tool messages using ContentRouter.

        This is the COMPRESS_FIRST strategy: try to compress tool outputs
        more aggressively before falling back to dropping messages.

        Args:
            messages: List of messages to compress.
            tokenizer: Tokenizer for counting.
            protected: Set of protected message indices.

        Returns:
            Tuple of (compressed_messages, transforms_applied, tokens_saved).
        """
        router = self._get_content_router()
        if router is None:
            return messages, [], 0

        compressed_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        total_tokens_saved = 0

        for i, msg in enumerate(compressed_messages):
            # Skip protected messages
            if i in protected:
                continue

            role = msg.get("role")
            content = msg.get("content")

            # Focus on tool messages (highest compression potential)
            if role == "tool" and isinstance(content, str) and len(content) > 100:
                try:
                    # Get source hint from tool call context
                    tool_call_id = msg.get("tool_call_id", "")
                    source_hint = self._get_tool_source_hint(messages, tool_call_id)

                    # Compress using ContentRouter
                    result = router.compress(
                        content,
                        source_hint=source_hint,
                        context="",  # No specific context in compress-first mode
                    )

                    # Check if compression was effective
                    if result.compression_ratio < 0.9:  # At least 10% savings
                        tokens_before = tokenizer.count_text(content)
                        tokens_after = tokenizer.count_text(result.compressed)
                        tokens_saved = tokens_before - tokens_after

                        if tokens_saved > 0:
                            compressed_messages[i] = {
                                **msg,
                                "content": result.compressed,
                            }
                            transforms_applied.append(
                                f"compress_first:{result.strategy_used.value}:{i}"
                            )
                            total_tokens_saved += tokens_saved
                            logger.debug(
                                "COMPRESS_FIRST: message %d compressed %d→%d tokens (%s)",
                                i,
                                tokens_before,
                                tokens_after,
                                result.strategy_used.value,
                            )
                except Exception as e:
                    logger.debug("COMPRESS_FIRST: compression failed for message %d: %s", i, e)
                    continue

            # Also try to compress assistant messages with tool results in content blocks
            elif role == "assistant" and isinstance(content, list):
                compressed_blocks, block_transforms, block_saved = self._compress_content_blocks(
                    content, router, tokenizer
                )
                if block_saved > 0:
                    compressed_messages[i] = {**msg, "content": compressed_blocks}
                    transforms_applied.extend(block_transforms)
                    total_tokens_saved += block_saved

        if total_tokens_saved > 0:
            logger.info(
                "COMPRESS_FIRST: saved %d tokens across %d compressions",
                total_tokens_saved,
                len(transforms_applied),
            )

        return compressed_messages, transforms_applied, total_tokens_saved

    def _get_tool_source_hint(self, messages: list[dict[str, Any]], tool_call_id: str) -> str:
        """Extract source hint from the tool call that produced this result.

        Args:
            messages: List of all messages.
            tool_call_id: The ID of the tool call to find.

        Returns:
            Source hint string (e.g., "tool:grep", "file:main.py").
        """
        if not tool_call_id:
            return ""

        # Find the assistant message with this tool call
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    if tc.get("id") == tool_call_id:
                        func = tc.get("function", {})
                        tool_name = func.get("name", "")

                        # Import here to avoid circular imports
                        try:
                            import json

                            from .content_router import generate_source_hint

                            args_str = func.get("arguments", "{}")
                            try:
                                args = (
                                    json.loads(args_str) if isinstance(args_str, str) else args_str
                                )
                            except json.JSONDecodeError:
                                args = {}

                            return generate_source_hint(tool_name, args)
                        except ImportError:
                            return f"tool:{tool_name}" if tool_name else ""
        return ""

    def _compress_content_blocks(
        self,
        content_blocks: list[Any],
        router: ContentRouter,
        tokenizer: Tokenizer,
    ) -> tuple[list[Any], list[str], int]:
        """Compress content blocks (Anthropic format) using ContentRouter.

        Args:
            content_blocks: List of content blocks.
            router: ContentRouter instance.
            tokenizer: Tokenizer for counting.

        Returns:
            Tuple of (compressed_blocks, transforms_applied, tokens_saved).
        """
        compressed_blocks: list[Any] = []
        transforms_applied: list[str] = []
        total_saved = 0

        for block in content_blocks:
            if not isinstance(block, dict):
                compressed_blocks.append(block)
                continue

            block_type = block.get("type")

            # Handle tool_result blocks
            if block_type == "tool_result":
                tool_content = block.get("content", "")

                if isinstance(tool_content, str) and len(tool_content) > 200:
                    try:
                        result = router.compress(tool_content, context="")

                        if result.compression_ratio < 0.9:
                            tokens_before = tokenizer.count_text(tool_content)
                            tokens_after = tokenizer.count_text(result.compressed)
                            saved = tokens_before - tokens_after

                            if saved > 0:
                                compressed_blocks.append(
                                    {
                                        **block,
                                        "content": result.compressed,
                                    }
                                )
                                transforms_applied.append(
                                    f"compress_first:block:{result.strategy_used.value}"
                                )
                                total_saved += saved
                                continue
                    except Exception:
                        pass

            compressed_blocks.append(block)

        return compressed_blocks, transforms_applied, total_saved

    def _get_protected_indices(self, messages: list[dict[str, Any]]) -> set[int]:
        """Get indices that should never be dropped."""
        protected: set[int] = set()

        # Protect system messages
        if self.config.keep_system:
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    protected.add(i)

        # Protect last N turns
        if self.config.keep_last_turns > 0:
            turns_seen = 0
            i = len(messages) - 1

            while i >= 0 and turns_seen < self.config.keep_last_turns:
                msg = messages[i]
                role = msg.get("role")
                protected.add(i)

                if role == "user":
                    turns_seen += 1

                i -= 1

            # Protect tool responses for protected assistant messages
            for i in list(protected):
                msg = messages[i]
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    tool_call_ids = {tc.get("id") for tc in msg.get("tool_calls", [])}
                    for j, other_msg in enumerate(messages):
                        if other_msg.get("role") == "tool":
                            if other_msg.get("tool_call_id") in tool_call_ids:
                                protected.add(j)

        return protected

    def _get_tool_unit_indices(self, tool_units: list[tuple[int, list[int]]]) -> set[int]:
        """Get all indices that are part of tool units."""
        indices: set[int] = set()
        for assistant_idx, response_indices in tool_units:
            indices.add(assistant_idx)
            indices.update(response_indices)
        return indices

    def _build_scored_drop_candidates(
        self,
        messages: list[dict[str, Any]],
        scores: list[MessageScore],
        protected: set[int],
        tool_units: list[tuple[int, list[int]]],
    ) -> list[dict[str, Any]]:
        """Build drop candidates sorted by importance score (lowest first)."""
        candidates: list[dict[str, Any]] = []

        # Track tool unit indices
        tool_unit_indices: set[int] = set()
        for assistant_idx, response_indices in tool_units:
            tool_unit_indices.add(assistant_idx)
            tool_unit_indices.update(response_indices)

        # Add tool units as atomic candidates
        for assistant_idx, response_indices in tool_units:
            if assistant_idx in protected:
                continue

            all_indices = [assistant_idx] + response_indices

            # Average score for the unit
            unit_scores = [scores[idx].total_score for idx in all_indices if idx < len(scores)]
            avg_score = sum(unit_scores) / len(unit_scores) if unit_scores else 0.5

            candidates.append(
                {
                    "type": "tool_unit",
                    "indices": all_indices,
                    "score": avg_score,
                    "position": assistant_idx,
                }
            )

        # Add non-tool messages
        i = 0
        while i < len(messages):
            if i in protected or i in tool_unit_indices:
                i += 1
                continue

            msg = messages[i]
            role = msg.get("role")

            if role in ("user", "assistant"):
                # Try to pair user+assistant
                if role == "user" and i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if (
                        next_msg.get("role") == "assistant"
                        and i + 1 not in tool_unit_indices
                        and i + 1 not in protected
                    ):
                        # Paired turn
                        pair_score = (scores[i].total_score + scores[i + 1].total_score) / 2
                        candidates.append(
                            {
                                "type": "turn",
                                "indices": [i, i + 1],
                                "score": pair_score,
                                "position": i,
                            }
                        )
                        i += 2
                        continue

                # Single message
                candidates.append(
                    {
                        "type": "single",
                        "indices": [i],
                        "score": scores[i].total_score,
                        "position": i,
                    }
                )

            i += 1

        # Sort by score (lowest first = drop first)
        candidates.sort(key=lambda c: (c["score"], c["position"]))

        return candidates

    def _position_based_scores(
        self,
        messages: list[dict[str, Any]],
        protected: set[int],
        tool_unit_indices: set[int],
    ) -> list[MessageScore]:
        """Fallback position-based scoring when importance scoring disabled."""
        scores = []
        total = len(messages)

        for i, _msg in enumerate(messages):
            # Simple position-based score: newer = higher
            position_score = i / max(1, total - 1) if total > 1 else 1.0

            scores.append(
                MessageScore(
                    message_index=i,
                    total_score=position_score,
                    recency_score=position_score,
                    is_protected=i in protected,
                    drop_safe=i not in tool_unit_indices,
                )
            )

        return scores
