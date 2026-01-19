"""Progressive summarization for Headroom SDK.

This module provides anchored summarization that progressively summarizes
older messages while maintaining retrieval capability via CCR.

Design principles:
1. CALLBACK PATTERN: Summarization is done via a callback, not internal LLM calls
2. ANCHORED: Summaries track which message positions they represent
3. REVERSIBLE: Original content stored in CompressionStore for CCR retrieval
4. INCREMENTAL: Only summarize newly dropped spans, then merge

Usage:
    from headroom.transforms import ProgressiveSummarizer

    # With custom summarizer callback
    def my_summarizer(messages: list[dict], context: str) -> str:
        # Your summarization logic (LLM call, extractive, etc.)
        return "Summary of messages..."

    summarizer = ProgressiveSummarizer(
        summarize_fn=my_summarizer,
        max_summary_tokens=500,
    )

    result = summarizer.summarize_messages(messages, tokenizer, protected)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..cache.compression_store import CompressionStore
    from ..tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SummarizeFn(Protocol):
    """Protocol for summarization callback functions.

    The callback receives:
    - messages: List of messages to summarize
    - context: Optional context string (e.g., recent messages for relevance)

    Returns:
    - Summary string
    """

    def __call__(
        self,
        messages: list[dict[str, Any]],
        context: str = "",
    ) -> str: ...


@dataclass
class AnchoredSummary:
    """A summary anchored to specific message positions.

    Tracks which messages were summarized for:
    - Retrieval: Can reconstruct original messages via CCR
    - Merging: Can merge with adjacent summaries
    - Positioning: Know where in conversation this summary belongs
    """

    summary_text: str
    start_index: int  # First message index summarized
    end_index: int  # Last message index summarized (inclusive)
    original_message_count: int
    original_tokens: int
    summary_tokens: int
    cache_hash: str | None = None  # Hash for CCR retrieval
    tool_names: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def compression_ratio(self) -> float:
        """Ratio of summary tokens to original tokens (lower = more compression)."""
        if self.original_tokens == 0:
            return 1.0
        return self.summary_tokens / self.original_tokens

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved by summarization."""
        return max(0, self.original_tokens - self.summary_tokens)


@dataclass
class SummarizationResult:
    """Result of a summarization operation."""

    messages: list[dict[str, Any]]
    summaries_created: list[AnchoredSummary]
    tokens_before: int
    tokens_after: int
    transforms_applied: list[str]

    @property
    def tokens_saved(self) -> int:
        """Total tokens saved."""
        return max(0, self.tokens_before - self.tokens_after)


def extractive_summarizer(
    messages: list[dict[str, Any]],
    context: str = "",
    max_items_per_role: int = 2,
) -> str:
    """Default extractive summarizer (no LLM required).

    Creates a summary by extracting key content from messages:
    - First and last message of each role
    - Error indicators
    - Tool names and brief results

    This is a fallback when no LLM summarizer is provided.

    Args:
        messages: Messages to summarize.
        context: Optional context (unused in extractive mode).
        max_items_per_role: Max items to keep per role type.

    Returns:
        Extractive summary string.
    """
    if not messages:
        return "[No messages to summarize]"

    parts: list[str] = []
    parts.append(f"[Summary of {len(messages)} messages]")

    # Group by role
    by_role: dict[str, list[dict[str, Any]]] = {}
    for msg in messages:
        role = msg.get("role", "unknown")
        by_role.setdefault(role, []).append(msg)

    # Extract key content from each role
    for role, role_msgs in by_role.items():
        if role == "tool":
            # For tool messages, extract tool names and brief status
            tool_names = set()
            has_error = False
            for msg in role_msgs:
                content = msg.get("content", "")
                # Try to detect tool name from context
                tool_call_id = msg.get("tool_call_id", "")
                if tool_call_id:
                    tool_names.add(f"tool:{tool_call_id[:8]}")

                # Check for errors
                content_lower = content.lower() if isinstance(content, str) else ""
                if any(err in content_lower for err in ["error", "failed", "exception"]):
                    has_error = True

            status = "with errors" if has_error else "successful"
            parts.append(f"- {len(role_msgs)} tool outputs ({status})")

        elif role == "assistant":
            # Extract first and last assistant responses
            if len(role_msgs) == 1:
                content = role_msgs[0].get("content", "")
                if isinstance(content, str):
                    preview = content[:100] + "..." if len(content) > 100 else content
                    parts.append(f"- Assistant: {preview}")
            else:
                parts.append(f"- {len(role_msgs)} assistant messages")

        elif role == "user":
            # Count user messages
            parts.append(f"- {len(role_msgs)} user messages")

        elif role == "system":
            # Note system messages (shouldn't be summarized usually)
            parts.append(f"- {len(role_msgs)} system messages")

    return "\n".join(parts)


class ProgressiveSummarizer:
    """Progressive summarization with anchoring and CCR integration.

    This class implements the SUMMARIZE strategy for IntelligentContextManager:
    1. Identifies candidate messages (low-scored, non-protected)
    2. Groups consecutive messages for summarization
    3. Calls summarizer callback to create summaries
    4. Stores originals in CompressionStore for CCR retrieval
    5. Replaces messages with anchored summary message

    Key features:
    - Callback pattern: No LLM calls inside, summarization logic is external
    - Anchored: Summaries track original positions for context
    - Reversible: Originals cached for retrieval
    - Incremental: Can merge adjacent summaries
    """

    def __init__(
        self,
        summarize_fn: SummarizeFn | None = None,
        max_summary_tokens: int = 500,
        min_messages_to_summarize: int = 3,
        compression_store: CompressionStore | None = None,
        store_for_retrieval: bool = True,
    ):
        """Initialize the progressive summarizer.

        Args:
            summarize_fn: Callback function for summarization.
                If None, uses extractive_summarizer as fallback.
            max_summary_tokens: Target max tokens for each summary.
            min_messages_to_summarize: Minimum messages in a group to summarize.
            compression_store: Optional CompressionStore for CCR integration.
            store_for_retrieval: Whether to store originals for retrieval.
        """
        self.summarize_fn = summarize_fn or extractive_summarizer
        self.max_summary_tokens = max_summary_tokens
        self.min_messages_to_summarize = min_messages_to_summarize
        self._compression_store = compression_store
        self.store_for_retrieval = store_for_retrieval

    def _get_compression_store(self) -> CompressionStore | None:
        """Get or create compression store (lazy load)."""
        if self._compression_store is None and self.store_for_retrieval:
            try:
                from ..cache.compression_store import get_compression_store

                self._compression_store = get_compression_store()
            except ImportError:
                logger.debug("CompressionStore not available for CCR")
        return self._compression_store

    def summarize_messages(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        protected_indices: set[int],
        target_tokens: int | None = None,
        context_messages: list[dict[str, Any]] | None = None,
    ) -> SummarizationResult:
        """Summarize messages to reduce token count.

        Args:
            messages: List of messages to process.
            tokenizer: Tokenizer for counting.
            protected_indices: Indices that cannot be summarized.
            target_tokens: Target token count (optional, summarizes all candidates if None).
            context_messages: Recent messages for context in summarization.

        Returns:
            SummarizationResult with summarized messages.
        """
        from ..utils import deep_copy_messages

        tokens_before = tokenizer.count_messages(messages)
        result_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        summaries_created: list[AnchoredSummary] = []

        # Find candidate groups for summarization
        candidate_groups = self._find_summarization_candidates(result_messages, protected_indices)

        if not candidate_groups:
            logger.debug("ProgressiveSummarizer: no candidates for summarization")
            return SummarizationResult(
                messages=result_messages,
                summaries_created=[],
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
            )

        # Build context string from recent messages
        context_str = ""
        if context_messages:
            context_parts = []
            for msg in context_messages[-3:]:  # Last 3 messages for context
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    preview = content[:200] if len(content) > 200 else content
                    context_parts.append(f"{role}: {preview}")
            context_str = "\n".join(context_parts)

        # Process groups in reverse order (so indices stay valid)
        current_tokens = tokens_before

        for group in reversed(candidate_groups):
            # Check if we've reached target
            if target_tokens and current_tokens <= target_tokens:
                break

            start_idx, end_idx = group
            group_messages = result_messages[start_idx : end_idx + 1]

            # Skip if too few messages
            if len(group_messages) < self.min_messages_to_summarize:
                continue

            # Calculate group tokens
            group_tokens = sum(tokenizer.count_message(msg) for msg in group_messages)

            # Skip small groups
            if group_tokens < 100:
                continue

            # Create summary using callback
            try:
                summary_text = self.summarize_fn(group_messages, context_str)
            except Exception as e:
                logger.warning(
                    "ProgressiveSummarizer: summarization failed for group %d-%d: %s",
                    start_idx,
                    end_idx,
                    e,
                )
                continue

            summary_tokens = tokenizer.count_text(summary_text)

            # Only use summary if it saves tokens
            if summary_tokens >= group_tokens:
                logger.debug(
                    "ProgressiveSummarizer: summary not smaller (%d >= %d), skipping",
                    summary_tokens,
                    group_tokens,
                )
                continue

            # Store original for CCR retrieval
            cache_hash = None
            if self.store_for_retrieval:
                cache_hash = self._store_for_retrieval(
                    group_messages, summary_text, group_tokens, summary_tokens
                )

            # Extract tool names
            tool_names = []
            for msg in group_messages:
                if msg.get("role") == "tool":
                    tool_call_id = msg.get("tool_call_id", "")
                    if tool_call_id:
                        tool_names.append(tool_call_id[:8])

            # Create anchored summary
            anchored = AnchoredSummary(
                summary_text=summary_text,
                start_index=start_idx,
                end_index=end_idx,
                original_message_count=len(group_messages),
                original_tokens=group_tokens,
                summary_tokens=summary_tokens,
                cache_hash=cache_hash,
                tool_names=tool_names,
            )
            summaries_created.append(anchored)

            # Create summary message with retrieval marker
            summary_content = summary_text
            if cache_hash:
                summary_content += f"\n[Retrieve full content: hash={cache_hash}]"

            summary_message = {
                "role": "user",
                "content": summary_content,
            }

            # Replace group with summary message
            result_messages = (
                result_messages[:start_idx] + [summary_message] + result_messages[end_idx + 1 :]
            )

            # Update token count
            tokens_saved = group_tokens - summary_tokens
            current_tokens -= tokens_saved

            transforms_applied.append(f"summarize:{start_idx}-{end_idx}:{len(group_messages)}")

            logger.debug(
                "ProgressiveSummarizer: summarized %d messages (%d-%d), saved %d tokens (%d -> %d)",
                len(group_messages),
                start_idx,
                end_idx,
                tokens_saved,
                group_tokens,
                summary_tokens,
            )

            # Update protected indices for subsequent groups
            # (indices shift after replacement)
            shift = len(group_messages) - 1  # We replaced N messages with 1
            protected_indices = {idx - shift if idx > end_idx else idx for idx in protected_indices}

        tokens_after = tokenizer.count_messages(result_messages)

        if summaries_created:
            logger.info(
                "ProgressiveSummarizer: created %d summaries, saved %d tokens (%d -> %d)",
                len(summaries_created),
                tokens_before - tokens_after,
                tokens_before,
                tokens_after,
            )

        return SummarizationResult(
            messages=result_messages,
            summaries_created=summaries_created,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
        )

    def _find_summarization_candidates(
        self,
        messages: list[dict[str, Any]],
        protected: set[int],
    ) -> list[tuple[int, int]]:
        """Find groups of consecutive messages that can be summarized.

        Returns list of (start_index, end_index) tuples for candidate groups.
        Groups are consecutive non-protected messages.

        Args:
            messages: List of messages.
            protected: Set of protected indices.

        Returns:
            List of (start, end) tuples for candidate groups.
        """
        groups: list[tuple[int, int]] = []
        current_start: int | None = None

        for i, _msg in enumerate(messages):
            if i in protected:
                # End current group if exists
                if current_start is not None:
                    if i - 1 >= current_start:
                        groups.append((current_start, i - 1))
                    current_start = None
            else:
                # Start or continue group
                if current_start is None:
                    current_start = i

        # Handle final group
        if current_start is not None and len(messages) - 1 >= current_start:
            groups.append((current_start, len(messages) - 1))

        # Filter groups that are too small
        groups = [
            (start, end)
            for start, end in groups
            if end - start + 1 >= self.min_messages_to_summarize
        ]

        return groups

    def _store_for_retrieval(
        self,
        messages: list[dict[str, Any]],
        summary: str,
        original_tokens: int,
        summary_tokens: int,
    ) -> str | None:
        """Store original messages in CompressionStore for CCR retrieval.

        Args:
            messages: Original messages.
            summary: Summary text.
            original_tokens: Token count of originals.
            summary_tokens: Token count of summary.

        Returns:
            Cache hash for retrieval, or None if storage failed.
        """
        store = self._get_compression_store()
        if store is None:
            return None

        try:
            # Serialize messages for storage
            original_content = json.dumps(messages, ensure_ascii=False)

            # Generate hash
            content_hash = hashlib.sha256(original_content.encode()).hexdigest()[:24]

            # Store in compression store
            store.store(
                original=original_content,
                compressed=summary,
                original_tokens=original_tokens,
                compressed_tokens=summary_tokens,
                original_item_count=len(messages),
                compressed_item_count=1,
                tool_name="progressive_summarizer",
            )

            return content_hash

        except Exception as e:
            logger.debug("Failed to store for CCR retrieval: %s", e)
            return None
