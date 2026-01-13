"""Generic text compressor for plain text content.

This module provides a fallback compressor for plain text that doesn't match
any specialized format (search results, logs, code, diffs). Uses line-based
sampling with anchor preservation.

Compression Strategy:
1. Identify anchor lines (contain context keywords)
2. Keep first N and last M lines
3. Sample from middle based on line importance
4. Add summary of omitted content
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TextCompressorConfig:
    """Configuration for text compression."""

    # Line limits
    keep_first_lines: int = 10
    keep_last_lines: int = 10
    max_total_lines: int = 50

    # Sampling
    sample_every_n_lines: int = 10

    # Anchor detection
    anchor_keywords: list[str] = field(default_factory=list)
    boost_pattern_lines: bool = True

    # CCR integration
    enable_ccr: bool = True
    min_lines_for_ccr: int = 100


class TextCompressor:
    """Compresses generic plain text.

    Example:
        >>> compressor = TextCompressor()
        >>> result = compressor.compress(large_text, context="find errors")
        >>> print(result.compressed)
    """

    # Patterns that indicate important lines
    _IMPORTANT_PATTERNS = [
        re.compile(r"\b(error|exception|fail|warning)\b", re.IGNORECASE),
        re.compile(r"\b(important|note|todo|fixme)\b", re.IGNORECASE),
        re.compile(r"^#+\s"),  # Markdown headers
        re.compile(r"^\*\*"),  # Bold text
        re.compile(r"^>\s"),  # Quotes
    ]

    def __init__(self, config: TextCompressorConfig | None = None):
        """Initialize text compressor.

        Args:
            config: Compression configuration.
        """
        self.config = config or TextCompressorConfig()

    def compress(self, content: str, context: str = "") -> TextCompressionResult:
        """Compress text content.

        Args:
            content: Raw text content.
            context: User query context for anchor detection.

        Returns:
            TextCompressionResult with compressed output.
        """
        lines = content.split("\n")

        if len(lines) <= self.config.max_total_lines:
            return TextCompressionResult(
                compressed=content,
                original=content,
                original_line_count=len(lines),
                compressed_line_count=len(lines),
                compression_ratio=1.0,
            )

        # Score lines by importance
        scored_lines = self._score_lines(lines, context)

        # Select lines
        selected = self._select_lines(scored_lines, lines)

        # Format output
        compressed = self._format_output(selected, len(lines))

        ratio = len(compressed) / max(len(content), 1)

        # Store in CCR if significant compression
        cache_key = None
        if self.config.enable_ccr and len(lines) >= self.config.min_lines_for_ccr and ratio < 0.7:
            cache_key = self._store_in_ccr(content, compressed, len(lines))
            if cache_key:
                compressed += f"\n[{len(lines)} lines compressed. hash={cache_key}]"

        return TextCompressionResult(
            compressed=compressed,
            original=content,
            original_line_count=len(lines),
            compressed_line_count=len(selected),
            compression_ratio=ratio,
            cache_key=cache_key,
        )

    def _score_lines(self, lines: list[str], context: str) -> list[tuple[int, str, float]]:
        """Score lines by importance."""
        context_lower = context.lower()
        context_words = set(context_lower.split()) if context else set()
        anchor_keywords = {k.lower() for k in self.config.anchor_keywords}

        scored: list[tuple[int, str, float]] = []

        for i, line in enumerate(lines):
            score = 0.0
            line_lower = line.lower()

            # Boost if contains context words
            for word in context_words:
                if len(word) > 2 and word in line_lower:
                    score += 0.3

            # Boost if contains anchor keywords
            for keyword in anchor_keywords:
                if keyword in line_lower:
                    score += 0.4

            # Boost if matches important patterns
            if self.config.boost_pattern_lines:
                for pattern in self._IMPORTANT_PATTERNS:
                    if pattern.search(line):
                        score += 0.2
                        break

            # Small boost for non-empty lines
            if line.strip():
                score += 0.1

            scored.append((i, line, min(1.0, score)))

        return scored

    def _select_lines(
        self, scored_lines: list[tuple[int, str, float]], original_lines: list[str]
    ) -> list[tuple[int, str]]:
        """Select lines to keep."""
        total = len(scored_lines)
        selected_indices: set[int] = set()

        # Always keep first N lines
        for i in range(min(self.config.keep_first_lines, total)):
            selected_indices.add(i)

        # Always keep last M lines
        for i in range(max(0, total - self.config.keep_last_lines), total):
            selected_indices.add(i)

        # Add high-scoring lines
        high_score_lines = [
            (idx, line, score)
            for idx, line, score in scored_lines
            if score >= 0.3 and idx not in selected_indices
        ]
        high_score_lines.sort(key=lambda x: x[2], reverse=True)

        remaining_slots = self.config.max_total_lines - len(selected_indices)
        for idx, _line, _score in high_score_lines[:remaining_slots]:
            selected_indices.add(idx)
            remaining_slots -= 1
            if remaining_slots <= 0:
                break

        # Sample from remaining middle lines
        if remaining_slots > 0:
            middle_start = self.config.keep_first_lines
            middle_end = total - self.config.keep_last_lines

            for i in range(middle_start, middle_end, self.config.sample_every_n_lines):
                if i not in selected_indices:
                    selected_indices.add(i)
                    remaining_slots -= 1
                    if remaining_slots <= 0:
                        break

        # Sort by line number and return
        selected = sorted(selected_indices)
        return [(i, original_lines[i]) for i in selected]

    def _format_output(self, selected: list[tuple[int, str]], total_lines: int) -> str:
        """Format selected lines with ellipsis markers."""
        if not selected:
            return f"[{total_lines} lines omitted]"

        output_lines: list[str] = []
        prev_idx = -1

        for idx, line in selected:
            # Add ellipsis if there's a gap
            if prev_idx >= 0 and idx - prev_idx > 1:
                gap = idx - prev_idx - 1
                output_lines.append(f"[... {gap} lines omitted ...]")

            output_lines.append(line)
            prev_idx = idx

        # Add trailing ellipsis if needed
        if selected and selected[-1][0] < total_lines - 1:
            gap = total_lines - selected[-1][0] - 1
            output_lines.append(f"[... {gap} lines omitted ...]")

        return "\n".join(output_lines)

    def _store_in_ccr(self, original: str, compressed: str, original_count: int) -> str | None:
        """Store original in CCR for later retrieval."""
        try:
            from ..cache.compression_store import get_compression_store

            store = get_compression_store()
            return store.store(
                original_content=original,
                compressed_content=compressed,
                content_type="plain_text",
                original_item_count=original_count,
            )
        except ImportError:
            return None
        except Exception:
            return None


@dataclass
class TextCompressionResult:
    """Result of text compression."""

    compressed: str
    original: str
    original_line_count: int
    compressed_line_count: int
    compression_ratio: float
    cache_key: str | None = None

    @property
    def tokens_saved_estimate(self) -> int:
        """Estimate tokens saved."""
        chars_saved = len(self.original) - len(self.compressed)
        return max(0, chars_saved // 4)

    @property
    def lines_omitted(self) -> int:
        """Number of lines omitted."""
        return self.original_line_count - self.compressed_line_count
