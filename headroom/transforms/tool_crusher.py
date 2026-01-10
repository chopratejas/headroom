"""Tool output compression transform for Headroom SDK."""

from __future__ import annotations

import logging
from typing import Any

from ..config import ToolCrusherConfig, TransformResult
from ..tokenizer import Tokenizer
from ..utils import (
    compute_short_hash,
    create_tool_digest_marker,
    deep_copy_messages,
    safe_json_dumps,
    safe_json_loads,
)
from .base import Transform

logger = logging.getLogger(__name__)


class ToolCrusher(Transform):
    """
    Compress tool output to reduce token usage.

    This transform applies conservative compression:
    - Only compresses tool role messages > min_tokens
    - Preserves JSON structure (never removes keys)
    - Truncates arrays to max_items
    - Truncates long strings
    - Limits nesting depth

    Safety: If JSON parsing fails, content is returned unchanged.
    """

    name = "tool_crusher"

    def __init__(self, config: ToolCrusherConfig | None = None):
        """
        Initialize tool crusher.

        Args:
            config: Configuration for compression behavior.
        """
        self.config = config or ToolCrusherConfig()

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if any tool messages exceed threshold."""
        if not self.config.enabled:
            return False

        for msg in messages:
            # OpenAI style: role="tool"
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str):
                    tokens = tokenizer.count_text(content)
                    if tokens > self.config.min_tokens_to_crush:
                        return True

            # Anthropic style: role="user" with tool_result content blocks
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, str):
                            tokens = tokenizer.count_text(tool_content)
                            if tokens > self.config.min_tokens_to_crush:
                                return True

        return False

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """
        Apply tool crushing to messages.

        Args:
            messages: List of messages.
            tokenizer: Tokenizer for counting.
            **kwargs: May include 'tool_profiles' for per-tool config.

        Returns:
            TransformResult with crushed messages.
        """
        tool_profiles = kwargs.get("tool_profiles", self.config.tool_profiles)

        tokens_before = tokenizer.count_messages(messages)
        result_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        markers_inserted: list[str] = []
        warnings: list[str] = []

        crushed_count = 0

        for msg in result_messages:
            # OpenAI style: role="tool"
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue

                # Check token threshold
                tokens = tokenizer.count_text(content)
                if tokens <= self.config.min_tokens_to_crush:
                    continue

                # Get tool-specific profile if available
                tool_call_id = msg.get("tool_call_id", "")
                profile = self._get_profile(tool_call_id, tool_profiles)

                # Try to crush
                crushed, was_modified = self._crush_content(content, profile)

                if was_modified:
                    # Compute hash of original for marker
                    original_hash = compute_short_hash(content)
                    marker = create_tool_digest_marker(original_hash)

                    msg["content"] = crushed + "\n" + marker
                    crushed_count += 1
                    markers_inserted.append(marker)

            # Anthropic style: role="user" with tool_result content blocks
            content = msg.get("content")
            if isinstance(content, list):
                for i, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue

                    tool_content = block.get("content", "")
                    if not isinstance(tool_content, str):
                        continue

                    # Check token threshold
                    tokens = tokenizer.count_text(tool_content)
                    if tokens <= self.config.min_tokens_to_crush:
                        continue

                    # Get tool-specific profile if available
                    tool_use_id = block.get("tool_use_id", "")
                    profile = self._get_profile(tool_use_id, tool_profiles)

                    # Try to crush
                    crushed, was_modified = self._crush_content(tool_content, profile)

                    if was_modified:
                        # Compute hash of original for marker
                        original_hash = compute_short_hash(tool_content)
                        marker = create_tool_digest_marker(original_hash)

                        # Update the content block
                        content[i]["content"] = crushed + "\n" + marker
                        crushed_count += 1
                        markers_inserted.append(marker)

        if crushed_count > 0:
            transforms_applied.append(f"tool_crush:{crushed_count}")
            logger.info(
                "ToolCrusher: compressed %d tool outputs, %d -> %d tokens",
                crushed_count,
                tokens_before,
                tokenizer.count_messages(result_messages),
            )

        tokens_after = tokenizer.count_messages(result_messages)

        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
            warnings=warnings,
        )

    def _get_profile(
        self,
        tool_call_id: str,
        tool_profiles: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Get compression profile for a tool."""
        # Tool profiles are keyed by tool name, not call ID
        # For now, use default config
        # In a real implementation, you'd map call_id -> tool_name
        return {
            "max_array_items": self.config.max_array_items,
            "max_string_length": self.config.max_string_length,
            "max_depth": self.config.max_depth,
            "preserve_keys": self.config.preserve_keys,
        }

    def _crush_content(
        self,
        content: str,
        profile: dict[str, Any],
    ) -> tuple[str, bool]:
        """
        Crush content according to profile.

        Returns:
            Tuple of (crushed_content, was_modified).
            If parsing fails, returns (original_content, False).
        """
        # Try JSON parse
        parsed, success = safe_json_loads(content)
        if not success:
            # Safety: don't modify unparseable content
            return content, False

        # Apply crushing
        crushed = self._crush_value(
            parsed,
            depth=0,
            max_depth=profile.get("max_depth", 5),
            max_array_items=profile.get("max_array_items", 10),
            max_string_length=profile.get("max_string_length", 1000),
        )

        # Serialize back
        result = safe_json_dumps(crushed, indent=None)

        # Check if actually modified
        was_modified = result != content.strip()

        return result, was_modified

    def _crush_value(
        self,
        value: Any,
        depth: int,
        max_depth: int,
        max_array_items: int,
        max_string_length: int,
    ) -> Any:
        """Recursively crush a value."""
        if depth >= max_depth:
            # At max depth, summarize
            if isinstance(value, dict):
                return {"__headroom_depth_exceeded": len(value)}
            elif isinstance(value, list):
                return {"__headroom_depth_exceeded": len(value)}
            elif isinstance(value, str) and len(value) > max_string_length:
                return (
                    value[:max_string_length]
                    + f"...[truncated {len(value) - max_string_length} chars]"
                )
            return value

        if isinstance(value, dict):
            return {
                k: self._crush_value(
                    v,
                    depth + 1,
                    max_depth,
                    max_array_items,
                    max_string_length,
                )
                for k, v in value.items()
            }

        elif isinstance(value, list):
            if len(value) <= max_array_items:
                return [
                    self._crush_value(
                        item,
                        depth + 1,
                        max_depth,
                        max_array_items,
                        max_string_length,
                    )
                    for item in value
                ]
            else:
                # Truncate array
                truncated = [
                    self._crush_value(
                        item,
                        depth + 1,
                        max_depth,
                        max_array_items,
                        max_string_length,
                    )
                    for item in value[:max_array_items]
                ]
                truncated.append({"__headroom_truncated": len(value) - max_array_items})
                return truncated

        elif isinstance(value, str):
            if len(value) > max_string_length:
                return (
                    value[:max_string_length]
                    + f"...[truncated {len(value) - max_string_length} chars]"
                )
            return value

        else:
            # Numbers, bools, None - pass through
            return value


def crush_tool_output(
    content: str,
    config: ToolCrusherConfig | None = None,
) -> tuple[str, bool]:
    """
    Convenience function to crush a single tool output.

    Args:
        content: The tool output content.
        config: Optional configuration.

    Returns:
        Tuple of (crushed_content, was_modified).
    """
    cfg = config or ToolCrusherConfig()
    crusher = ToolCrusher(cfg)

    profile = {
        "max_array_items": cfg.max_array_items,
        "max_string_length": cfg.max_string_length,
        "max_depth": cfg.max_depth,
        "preserve_keys": cfg.preserve_keys,
    }

    return crusher._crush_content(content, profile)
