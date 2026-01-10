"""Rolling window transform for Headroom SDK."""

from __future__ import annotations

import logging
from typing import Any

from ..config import RollingWindowConfig, TransformResult
from ..parser import find_tool_units
from ..tokenizer import Tokenizer
from ..tokenizers import EstimatingTokenCounter
from ..utils import create_dropped_context_marker, deep_copy_messages
from .base import Transform

logger = logging.getLogger(__name__)


class RollingWindow(Transform):
    """
    Apply rolling window to keep messages within token budget.

    Drop order (deterministic):
    1. Oldest TOOL UNITS (assistant+tool_calls paired with tool responses)
    2. Oldest assistant+user pairs
    3. Oldest RAG blocks (if detectable)

    CRITICAL: Tool calls and tool results are atomic DROP UNITS.
    Never orphan a tool result.

    Never drops:
    - System prompt
    - Stable instructions
    - Last N conversational turns (configurable)
    """

    name = "rolling_window"

    def __init__(self, config: RollingWindowConfig | None = None):
        """
        Initialize rolling window.

        Args:
            config: Configuration for window behavior.
        """
        self.config = config or RollingWindowConfig()

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if token cap is exceeded."""
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
        Apply rolling window to messages.

        Args:
            messages: List of messages.
            tokenizer: Tokenizer for counting.
            **kwargs: Must include 'model_limit', optionally 'output_buffer'.

        Returns:
            TransformResult with windowed messages.
        """
        model_limit = kwargs.get("model_limit", 128000)
        output_buffer = kwargs.get("output_buffer", self.config.output_buffer_tokens)
        available = model_limit - output_buffer

        tokens_before = tokenizer.count_messages(messages)
        result_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        markers_inserted: list[str] = []
        warnings: list[str] = []

        dropped_count = 0
        tool_units_dropped = 0

        # If already under budget, no changes needed
        current_tokens = tokens_before
        if current_tokens <= available:
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
                warnings=[],
            )

        # Identify protected indices
        protected = self._get_protected_indices(result_messages)

        # Identify tool units
        tool_units = find_tool_units(result_messages)

        # Create drop candidates with priorities
        drop_candidates = self._build_drop_candidates(result_messages, protected, tool_units)

        # Drop until under budget
        indices_to_drop: set[int] = set()

        for candidate in drop_candidates:
            if current_tokens <= available:
                break

            # Get indices for this candidate
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

        # Remove dropped messages (in reverse order to preserve indices)
        for idx in sorted(indices_to_drop, reverse=True):
            if idx < len(result_messages):
                del result_messages[idx]

        # Insert marker if we dropped anything
        if dropped_count > 0:
            logger.info(
                "RollingWindow: dropped %d units (%d tool units) to fit budget: %d -> %d tokens",
                dropped_count,
                tool_units_dropped,
                tokens_before,
                current_tokens,
            )
            marker = create_dropped_context_marker("token_cap", dropped_count)
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
                {
                    "role": "user",
                    "content": marker,
                },
            )

            transforms_applied.append(f"window_cap:{dropped_count}")

        tokens_after = tokenizer.count_messages(result_messages)

        result = TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
            warnings=warnings,
        )

        return result

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
            # Count turns from end (user+assistant = 1 turn)
            turns_seen = 0
            i = len(messages) - 1

            while i >= 0 and turns_seen < self.config.keep_last_turns:
                msg = messages[i]
                role = msg.get("role")

                # Protect this message
                protected.add(i)

                # Count turns
                if role == "user":
                    turns_seen += 1

                i -= 1

            # Also protect any tool responses that belong to protected assistant messages
            for i in list(protected):
                msg = messages[i]
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Find and protect corresponding tool responses
                    tool_call_ids = {tc.get("id") for tc in msg.get("tool_calls", [])}
                    for j, other_msg in enumerate(messages):
                        if other_msg.get("role") == "tool":
                            if other_msg.get("tool_call_id") in tool_call_ids:
                                protected.add(j)

        return protected

    def _build_drop_candidates(
        self,
        messages: list[dict[str, Any]],
        protected: set[int],
        tool_units: list[tuple[int, list[int]]],
    ) -> list[dict[str, Any]]:
        """
        Build ordered list of drop candidates.

        Returns candidates in drop priority order (first to drop first).
        """
        candidates: list[dict[str, Any]] = []

        # Track which indices are part of tool units
        tool_unit_indices: set[int] = set()
        for assistant_idx, response_indices in tool_units:
            tool_unit_indices.add(assistant_idx)
            tool_unit_indices.update(response_indices)

        # Priority 1: Oldest tool units (all indices as atomic unit)
        for assistant_idx, response_indices in tool_units:
            if assistant_idx in protected:
                continue

            all_indices = [assistant_idx] + response_indices
            candidates.append(
                {
                    "type": "tool_unit",
                    "indices": all_indices,
                    "priority": 1,
                    "position": assistant_idx,  # For sorting by age
                }
            )

        # Priority 2: Oldest non-tool messages (user/assistant pairs)
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            if i in protected or i in tool_unit_indices:
                i += 1
                continue

            if role in ("user", "assistant"):
                # Try to find a pair
                if role == "user" and i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get("role") == "assistant" and i + 1 not in tool_unit_indices:
                        candidates.append(
                            {
                                "type": "turn",
                                "indices": [i, i + 1],
                                "priority": 2,
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
                        "priority": 2,
                        "position": i,
                    }
                )

            i += 1

        # Sort by priority, then by position (oldest first)
        candidates.sort(key=lambda c: (c["priority"], c["position"]))

        return candidates


def apply_rolling_window(
    messages: list[dict[str, Any]],
    model_limit: int,
    output_buffer: int = 4000,
    keep_last_turns: int = 2,
    config: RollingWindowConfig | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Convenience function to apply rolling window.

    Args:
        messages: List of messages.
        model_limit: Model's context limit.
        output_buffer: Tokens to reserve for output.
        keep_last_turns: Number of recent turns to protect.
        config: Optional configuration.

    Returns:
        Tuple of (windowed_messages, dropped_descriptions).
    """
    cfg = config or RollingWindowConfig()
    cfg.output_buffer_tokens = output_buffer
    cfg.keep_last_turns = keep_last_turns

    window = RollingWindow(cfg)
    tokenizer = Tokenizer(EstimatingTokenCounter())  # type: ignore[arg-type]

    result = window.apply(
        messages,
        tokenizer,
        model_limit=model_limit,
        output_buffer=output_buffer,
    )

    return result.messages, result.transforms_applied
