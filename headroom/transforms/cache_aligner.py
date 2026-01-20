"""Cache alignment transform for Headroom SDK.

Phase 1 Enhancement: Integrates DynamicContentDetector for comprehensive
dynamic content detection beyond just dates.

Detection capabilities include:
- UUIDs, API keys, JWT tokens
- Unix timestamps, request/trace IDs
- Hex hashes (MD5, SHA1, SHA256)
- Version numbers, structural patterns
- High-entropy strings (random-looking IDs)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..cache.dynamic_detector import (
    DetectorConfig,
    DynamicContentDetector,
)
from ..config import CacheAlignerConfig, CachePrefixMetrics, TransformResult
from ..tokenizer import Tokenizer
from ..tokenizers import EstimatingTokenCounter
from ..utils import compute_short_hash, deep_copy_messages
from .base import Transform

logger = logging.getLogger(__name__)


class CacheAligner(Transform):
    """
    Align messages for optimal cache hits.

    This transform:
    1. Extracts dynamic content from system prompt (dates, UUIDs, tokens, etc.)
    2. Normalizes whitespace for consistent hashing
    3. Computes a stable prefix hash

    The goal is to make the prefix byte-identical across requests
    so that LLM provider caching can be effective.

    Phase 1 Enhancement: Now uses DynamicContentDetector for comprehensive
    detection of 15+ dynamic content patterns including:
    - UUIDs, API keys, JWT tokens
    - Unix timestamps, request/trace IDs
    - Hex hashes (MD5, SHA1, SHA256)
    - Version numbers, structural patterns
    - High-entropy strings
    """

    name = "cache_aligner"

    def __init__(self, config: CacheAlignerConfig | None = None):
        """
        Initialize cache aligner.

        Args:
            config: Configuration for alignment behavior.
        """
        self.config = config or CacheAlignerConfig()

        # Initialize dynamic content detector if enabled
        self._dynamic_detector: DynamicContentDetector | None = None
        if self.config.use_dynamic_detector:
            self._init_dynamic_detector()

        # Legacy: compiled regex patterns (used when use_dynamic_detector=False)
        self._compiled_patterns: list[re.Pattern[str]] = []
        if not self.config.use_dynamic_detector:
            self._compile_patterns()

        # Track previous hash for cache hit detection
        self._previous_prefix_hash: str | None = None

    def _init_dynamic_detector(self) -> None:
        """Initialize the DynamicContentDetector with configured tiers."""
        # Build detector config
        detector_config = DetectorConfig(
            tiers=list(self.config.detection_tiers),
            entropy_threshold=self.config.entropy_threshold,
        )

        # Add extra dynamic labels if configured
        if self.config.extra_dynamic_labels:
            detector_config.dynamic_labels = (
                detector_config.dynamic_labels + self.config.extra_dynamic_labels
            )

        self._dynamic_detector = DynamicContentDetector(detector_config)

        # Log available tiers
        available = self._dynamic_detector.available_tiers
        logger.info(
            "CacheAligner: DynamicContentDetector initialized with tiers: %s",
            available,
        )

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency (legacy mode)."""
        self._compiled_patterns = [re.compile(pattern) for pattern in self.config.date_patterns]

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if alignment is needed."""
        if not self.config.enabled:
            return False

        # Check if system prompt contains dynamic content
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    if self._has_dynamic_content(content):
                        return True

        return False

    def _has_dynamic_content(self, content: str) -> bool:
        """Check if content has any dynamic patterns."""
        if self.config.use_dynamic_detector and self._dynamic_detector:
            # Use DynamicContentDetector for comprehensive detection
            result = self._dynamic_detector.detect(content)
            return len(result.spans) > 0
        else:
            # Legacy: use compiled date patterns only
            for pattern in self._compiled_patterns:
                if pattern.search(content):
                    return True
            return False

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """
        Apply cache alignment to messages.

        Args:
            messages: List of messages.
            tokenizer: Tokenizer for counting.
            **kwargs: Additional arguments.

        Returns:
            TransformResult with aligned messages.
        """
        tokens_before = tokenizer.count_messages(messages)
        result_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        warnings: list[str] = []

        extracted_dynamic: list[str] = []
        detection_stats: dict[str, int] = {}  # Track what was detected

        # Process system messages
        for msg in result_messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Extract dynamic content (dates, UUIDs, tokens, etc.)
                    new_content, extracted, stats = self._extract_dynamic_content(content)

                    if extracted:
                        extracted_dynamic.extend(extracted)
                        msg["content"] = new_content
                        # Accumulate detection stats
                        for category, count in stats.items():
                            detection_stats[category] = detection_stats.get(category, 0) + count

        # Normalize whitespace if configured
        if self.config.normalize_whitespace:
            for msg in result_messages:
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = self._normalize_whitespace(content)

        # Compute stable prefix content and hash BEFORE reinserting dates
        # This ensures the hash is based on the static content only
        stable_prefix_content = self._get_stable_prefix_content(result_messages)
        stable_hash = compute_short_hash(stable_prefix_content)

        # Compute cache metrics
        prefix_bytes = len(stable_prefix_content.encode("utf-8"))
        prefix_tokens_est = tokenizer.count_text(stable_prefix_content)
        prefix_changed = (
            self._previous_prefix_hash is not None and self._previous_prefix_hash != stable_hash
        )
        previous_hash = self._previous_prefix_hash

        # Update tracking for next request
        self._previous_prefix_hash = stable_hash

        cache_metrics = CachePrefixMetrics(
            stable_prefix_bytes=prefix_bytes,
            stable_prefix_tokens_est=prefix_tokens_est,
            stable_prefix_hash=stable_hash,
            prefix_changed=prefix_changed,
            previous_hash=previous_hash,
        )

        # If we extracted dynamic content, add it as dynamic context
        if extracted_dynamic:
            # Insert dynamic content as a context note after system messages
            # Strategy: add as a context note after system messages
            self._reinsert_dynamic_content(result_messages, extracted_dynamic)
            transforms_applied.append("cache_align")

            # Log what was detected
            if detection_stats:
                stats_str = ", ".join(
                    f"{cat}={cnt}" for cat, cnt in sorted(detection_stats.items())
                )
                logger.info(
                    "CacheAligner: extracted %d dynamic patterns (%s)",
                    len(extracted_dynamic),
                    stats_str,
                )
            else:
                logger.debug(
                    "CacheAligner: extracted %d dynamic patterns for cache alignment",
                    len(extracted_dynamic),
                )

        # Log cache hit/miss
        if prefix_changed:
            logger.debug(
                "CacheAligner: prefix changed (likely cache miss), hash: %s -> %s",
                previous_hash,
                stable_hash,
            )
        else:
            logger.debug("CacheAligner: prefix stable, hash: %s", stable_hash)

        tokens_after = tokenizer.count_messages(result_messages)

        result = TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            warnings=warnings,
            cache_metrics=cache_metrics,
        )

        # Store hash in flags for access by caller (backwards compatibility)
        result.markers_inserted.append(f"stable_prefix_hash:{stable_hash}")

        return result

    def _extract_dynamic_content(self, content: str) -> tuple[str, list[str], dict[str, int]]:
        """
        Extract dynamic content from text.

        Uses DynamicContentDetector when enabled, otherwise falls back
        to legacy date pattern matching.

        Returns:
            Tuple of (content_without_dynamic, list_of_extracted, category_counts).
        """
        if self.config.use_dynamic_detector and self._dynamic_detector:
            return self._extract_with_detector(content)
        else:
            # Legacy mode: extract dates only
            result, extracted = self._extract_dates_legacy(content)
            stats = {"date": len(extracted)} if extracted else {}
            return result, extracted, stats

    def _extract_with_detector(self, content: str) -> tuple[str, list[str], dict[str, int]]:
        """
        Extract dynamic content using DynamicContentDetector.

        Returns:
            Tuple of (static_content, extracted_values, category_counts).
        """
        if not self._dynamic_detector:
            return content, [], {}

        result = self._dynamic_detector.detect(content)

        if not result.spans:
            return content, [], {}

        # Count by category
        category_counts: dict[str, int] = {}
        extracted: list[str] = []

        for span in result.spans:
            category = span.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            extracted.append(span.text)

        # Use the static content from the detector
        static_content = self._cleanup_empty_lines(result.static_content)

        # Log detection details at debug level
        logger.debug(
            "DynamicContentDetector found %d spans in %.2fms: %s",
            len(result.spans),
            result.processing_time_ms,
            category_counts,
        )

        return static_content, extracted, category_counts

    def _extract_dates_legacy(self, content: str) -> tuple[str, list[str]]:
        """
        Extract date patterns from content (legacy mode).

        Returns:
            Tuple of (content_without_dates, list_of_extracted_dates).
        """
        extracted: list[str] = []
        result = content

        for pattern in self._compiled_patterns:
            matches = pattern.findall(result)
            extracted.extend(matches)
            result = pattern.sub("", result)

        # Clean up any resulting empty lines
        if extracted:
            result = self._cleanup_empty_lines(result)

        return result, extracted

    def _extract_dates(self, content: str) -> tuple[str, list[str]]:
        """
        Extract date patterns from content.

        DEPRECATED: Use _extract_dynamic_content instead.
        Kept for backward compatibility.

        Returns:
            Tuple of (content_without_dates, list_of_extracted_dates).
        """
        result, extracted, _ = self._extract_dynamic_content(content)
        return result, extracted

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace for consistent hashing."""
        # Normalize line endings
        result = content.replace("\r\n", "\n").replace("\r", "\n")

        # Trim trailing whitespace from lines
        lines = result.split("\n")
        lines = [line.rstrip() for line in lines]

        # Collapse multiple blank lines if configured
        if self.config.collapse_blank_lines:
            new_lines: list[str] = []
            prev_blank = False
            for line in lines:
                is_blank = not line.strip()
                if is_blank and prev_blank:
                    continue
                new_lines.append(line)
                prev_blank = is_blank
            lines = new_lines

        return "\n".join(lines)

    def _cleanup_empty_lines(self, content: str) -> str:
        """Remove empty lines that result from date extraction."""
        lines = content.split("\n")
        # Remove lines that are now empty after pattern removal
        lines = [line for line in lines if line.strip() or line == ""]

        # Collapse multiple consecutive empty lines
        new_lines: list[str] = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            new_lines.append(line)
            prev_empty = is_empty

        return "\n".join(new_lines).strip()

    def _reinsert_dynamic_content(
        self,
        messages: list[dict[str, Any]],
        dynamic_values: list[str],
    ) -> None:
        """
        Reinsert extracted dynamic content as dynamic context.

        Strategy: Append to the end of system message with a clear separator.
        The separator marks where static (cacheable) content ends and
        dynamic content begins.

        Note: The stable prefix hash is computed BEFORE this method is called,
        so the hash is based on static content only.
        """
        if not dynamic_values:
            return

        # Format dynamic content as a note
        # Use newlines for multiple items to improve readability
        if len(dynamic_values) <= 3:
            dynamic_note = ", ".join(dynamic_values)
        else:
            dynamic_note = "\n".join(f"- {v}" for v in dynamic_values)

        separator = self.config.dynamic_tail_separator

        # Find last system message and append dynamic content
        for msg in reversed(messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Use separator to clearly mark dynamic content
                    msg["content"] = content.strip() + separator + dynamic_note
                break

    def _reinsert_dates(
        self,
        messages: list[dict[str, Any]],
        dates: list[str],
    ) -> None:
        """
        Reinsert extracted dates as dynamic context.

        DEPRECATED: Use _reinsert_dynamic_content instead.
        Kept for backward compatibility.
        """
        self._reinsert_dynamic_content(messages, dates)

    def _get_stable_prefix_content(self, messages: list[dict[str, Any]]) -> str:
        """Get the stable prefix content (static portion of system messages).

        Only includes content BEFORE the dynamic_tail_separator in each
        system message. This ensures the content is stable across different
        dates/dynamic content.
        """
        prefix_parts: list[str] = []
        separator = self.config.dynamic_tail_separator

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Only include content BEFORE the dynamic separator
                    if separator in content:
                        content = content.split(separator)[0]
                    prefix_parts.append(content.strip())
            else:
                # Stop at first non-system message
                break

        return "\n---\n".join(prefix_parts)

    def _compute_stable_prefix_hash(self, messages: list[dict[str, Any]]) -> str:
        """Compute hash of the stable prefix portion.

        Only includes content BEFORE the dynamic_tail_separator in each
        system message. This ensures the hash is stable across different
        dates/dynamic content.
        """
        prefix_content = self._get_stable_prefix_content(messages)
        return compute_short_hash(prefix_content)

    def get_alignment_score(self, messages: list[dict[str, Any]]) -> float:
        """
        Compute cache alignment score (0-100).

        Higher score means better cache alignment potential.
        Uses DynamicContentDetector when enabled for comprehensive detection.
        """
        score = 100.0

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Penalize for each dynamic pattern found
                    if self.config.use_dynamic_detector and self._dynamic_detector:
                        # Use comprehensive detector
                        result = self._dynamic_detector.detect(content)
                        score -= len(result.spans) * 10
                    else:
                        # Legacy: use compiled date patterns
                        for pattern in self._compiled_patterns:
                            matches = pattern.findall(content)
                            score -= len(matches) * 10

                    # Penalize for inconsistent whitespace
                    if "\r" in content:
                        score -= 5
                    if "  " in content:  # Double spaces
                        score -= 2
                    if "\n\n\n" in content:  # Triple newlines
                        score -= 2

        return max(0.0, min(100.0, score))


def align_for_cache(
    messages: list[dict[str, Any]],
    config: CacheAlignerConfig | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Convenience function to align messages for cache.

    Args:
        messages: List of messages.
        config: Optional configuration.

    Returns:
        Tuple of (aligned_messages, stable_prefix_hash).
    """
    cfg = config or CacheAlignerConfig()
    aligner = CacheAligner(cfg)
    tokenizer = Tokenizer(EstimatingTokenCounter())  # type: ignore[arg-type]

    result = aligner.apply(messages, tokenizer)

    # Extract hash from markers
    stable_hash = ""
    for marker in result.markers_inserted:
        if marker.startswith("stable_prefix_hash:"):
            stable_hash = marker.split(":", 1)[1]
            break

    return result.messages, stable_hash
