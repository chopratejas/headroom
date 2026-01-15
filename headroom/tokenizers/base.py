"""Base classes for tokenizer implementations.

Defines the TokenCounter protocol and BaseTokenizer class that all
tokenizer backends must implement.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting implementations.

    Any class implementing this protocol can be used with Headroom
    for token counting. This allows integration with various
    tokenizer backends (tiktoken, HuggingFace, custom, etc.).
    """

    def count_text(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Total token count including message overhead.
        """
        ...


class BaseTokenizer(ABC):
    """Abstract base class for tokenizer implementations.

    Provides common functionality for counting messages while
    requiring subclasses to implement text tokenization.
    """

    # Token overhead per message (role, formatting, etc.)
    # Override in subclasses for model-specific overhead
    MESSAGE_OVERHEAD = 4
    REPLY_OVERHEAD = 3  # Assistant reply start tokens

    @abstractmethod
    def count_text(self, text: str) -> int:
        """Count tokens in a text string. Must be implemented by subclasses."""
        pass

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a single message.

        Args:
            message: A message dict with 'role' and 'content'.

        Returns:
            Token count for this message.
        """
        return self.count_messages([message]) - self.REPLY_OVERHEAD

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of chat messages.

        Uses OpenAI-style message counting as the baseline, which
        works well for most models.

        Args:
            messages: List of message dicts.

        Returns:
            Total token count.
        """
        total = 0

        for message in messages:
            # Base message overhead
            total += self.MESSAGE_OVERHEAD

            # Count role
            role = message.get("role", "")
            total += self.count_text(role)

            # Count content
            content = message.get("content")
            if content is not None:
                if isinstance(content, str):
                    total += self.count_text(content)
                elif isinstance(content, list):
                    # Multi-part content (images, tool results, etc.)
                    total += self._count_content_parts(content)

            # Count tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                total += self._count_tool_calls(tool_calls)

            # Count function call (legacy)
            function_call = message.get("function_call")
            if function_call:
                total += self._count_function_call(function_call)

            # Count name field
            name = message.get("name")
            if name:
                total += self.count_text(name)
                total += 1  # Name field overhead

        # Reply start overhead
        total += self.REPLY_OVERHEAD

        return total

    def _count_content_parts(self, parts: list[Any]) -> int:
        """Count tokens in multi-part content."""
        total = 0
        for part in parts:
            if isinstance(part, dict):
                part_type = part.get("type", "")

                if part_type == "text":
                    total += self.count_text(part.get("text", ""))
                elif part_type == "image_url":
                    # Images have fixed token cost (varies by model)
                    total += 85  # Base image token count
                elif part_type == "tool_result":
                    content = part.get("content", "")
                    if isinstance(content, str):
                        total += self.count_text(content)
                    else:
                        total += self.count_text(json.dumps(content))
                elif part_type == "tool_use":
                    total += self.count_text(part.get("name", ""))
                    total += self.count_text(json.dumps(part.get("input", {})))
                else:
                    # Unknown type - estimate from JSON
                    total += self.count_text(json.dumps(part))
            elif isinstance(part, str):
                total += self.count_text(part)

        return total

    def _count_tool_calls(self, tool_calls: list[dict[str, Any]]) -> int:
        """Count tokens in tool calls."""
        total = 0
        for call in tool_calls:
            total += 4  # Tool call overhead

            if "function" in call:
                func = call["function"]
                total += self.count_text(func.get("name", ""))
                total += self.count_text(func.get("arguments", ""))

            if "id" in call:
                total += self.count_text(call["id"])

        return total

    def _count_function_call(self, function_call: dict[str, Any]) -> int:
        """Count tokens in legacy function call."""
        total = 4  # Function call overhead
        total += self.count_text(function_call.get("name", ""))
        total += self.count_text(function_call.get("arguments", ""))
        return total

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Optional method - not all backends support encoding.
        Default implementation raises NotImplementedError.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.

        Raises:
            NotImplementedError: If encoding is not supported.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support encoding")

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Optional method - not all backends support decoding.
        Default implementation raises NotImplementedError.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.

        Raises:
            NotImplementedError: If decoding is not supported.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support decoding")
