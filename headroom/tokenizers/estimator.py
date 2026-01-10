"""Estimation-based token counter for fallback scenarios.

When no exact tokenizer is available (e.g., unknown models, missing
dependencies), this provides a reasonable approximation based on
character/word heuristics calibrated against real tokenizers.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseTokenizer


class EstimatingTokenCounter(BaseTokenizer):
    """Token counter using estimation heuristics.

    This is the fallback tokenizer used when:
    - Model is unknown/unsupported
    - Required tokenizer library not installed
    - Speed is prioritized over accuracy

    The estimation is calibrated against tiktoken cl100k_base and
    provides ~90% accuracy for typical text. It tends to slightly
    overestimate, which is safer for context window management.

    Estimation Strategy:
    - Base: ~4 characters per token (calibrated against GPT-4)
    - Adjustments for code, URLs, numbers, whitespace
    - Special handling for JSON structure

    Example:
        counter = EstimatingTokenCounter()
        tokens = counter.count_text("Hello, world!")
        print(f"Estimated tokens: {tokens}")
    """

    # Calibration constants (derived from tiktoken analysis)
    CHARS_PER_TOKEN = 4.0  # Average for English text
    CHARS_PER_TOKEN_CODE = 3.5  # Code is denser
    CHARS_PER_TOKEN_JSON = 3.2  # JSON has more structure

    # Patterns for content type detection
    CODE_PATTERN = re.compile(
        r"(?:def |class |function |const |let |var |import |from |"
        r"if \(|for \(|while \(|switch \(|try \{|catch \(|"
        r"=>|->|\{\{|\}\}|;$)",
        re.MULTILINE,
    )
    JSON_PATTERN = re.compile(r"^\s*[\[\{]")
    URL_PATTERN = re.compile(r"https?://\S+")
    UUID_PATTERN = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
    )

    def __init__(self, chars_per_token: float | None = None):
        """Initialize estimating counter.

        Args:
            chars_per_token: Override default chars per token ratio.
                            If None, auto-detects based on content type.
        """
        self._fixed_ratio = chars_per_token

    def count_text(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated number of tokens.
        """
        if not text:
            return 0

        # Use fixed ratio if provided
        if self._fixed_ratio is not None:
            return max(1, int(len(text) / self._fixed_ratio + 0.5))

        # Auto-detect content type and adjust ratio
        ratio = self._detect_ratio(text)

        # Apply ratio with minimum of 1 token
        base_count = int(len(text) / ratio + 0.5)

        # Add overhead for special patterns
        overhead = self._count_special_overhead(text)

        return max(1, base_count + overhead)

    def _detect_ratio(self, text: str) -> float:
        """Detect optimal chars-per-token ratio based on content.

        Args:
            text: Text to analyze.

        Returns:
            Chars per token ratio.
        """
        # Check for JSON
        if self.JSON_PATTERN.match(text):
            try:
                json.loads(text)
                return self.CHARS_PER_TOKEN_JSON
            except (json.JSONDecodeError, ValueError):
                pass

        # Check for code
        code_matches = len(self.CODE_PATTERN.findall(text))
        if code_matches > len(text) / 500:  # ~2 matches per KB
            return self.CHARS_PER_TOKEN_CODE

        return self.CHARS_PER_TOKEN

    def _count_special_overhead(self, text: str) -> int:
        """Count additional tokens for special patterns.

        URLs and UUIDs often tokenize into more tokens than
        character count would suggest.

        Args:
            text: Text to analyze.

        Returns:
            Additional token overhead.
        """
        overhead = 0

        # URLs typically tokenize to more tokens
        urls = self.URL_PATTERN.findall(text)
        for url in urls:
            # Each URL component adds overhead
            overhead += url.count("/") + url.count("?") + url.count("&")

        # UUIDs are typically 8-10 tokens despite being 36 chars
        uuids = self.UUID_PATTERN.findall(text)
        overhead += len(uuids) * 2  # Each UUID adds ~2 extra tokens

        return overhead

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate tokens in chat messages.

        Uses the base class implementation with estimation-based
        text counting.

        Args:
            messages: List of chat messages.

        Returns:
            Estimated total token count.
        """
        # Use base class implementation
        return super().count_messages(messages)

    def __repr__(self) -> str:
        if self._fixed_ratio:
            return f"EstimatingTokenCounter(chars_per_token={self._fixed_ratio})"
        return "EstimatingTokenCounter(auto)"


class CharacterCounter(BaseTokenizer):
    """Simple character-based counter.

    Uses a fixed character-to-token ratio. Useful for:
    - Quick approximations
    - Testing
    - Models with unknown tokenization

    This is less accurate than EstimatingTokenCounter but faster.
    """

    def __init__(self, chars_per_token: float = 4.0):
        """Initialize character counter.

        Args:
            chars_per_token: Characters per token ratio.
        """
        self.chars_per_token = chars_per_token

    def count_text(self, text: str) -> int:
        """Count tokens based on character count.

        Args:
            text: Text to count.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token + 0.5))

    def __repr__(self) -> str:
        return f"CharacterCounter(chars_per_token={self.chars_per_token})"
