"""Intelligent context manager for Headroom SDK.

This module provides semantic-aware context management that extends
RollingWindow with importance-based scoring and TOIN integration.

Design principle: NO HARDCODED PATTERNS
All importance signals are derived from:
1. Computed metrics (recency, density, references)
2. TOIN-learned patterns (field_semantics, retrieval_rate)
3. Embedding similarity (optional)
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

        # Get protected indices and tool units
        protected = self._get_protected_indices(result_messages)
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
