"""Content router for intelligent compression strategy selection.

This module provides the ContentRouter, which analyzes content and routes it
to the optimal compressor. It handles mixed content by splitting, routing
each section to the appropriate compressor, and reassembling.

Supported Compressors:
- CodeAwareCompressor: Source code (AST-preserving)
- SmartCrusher: JSON arrays
- SearchCompressor: grep/ripgrep results
- LogCompressor: Build/test output
- LLMLinguaCompressor: Plain text (ML-based)
- TextCompressor: Plain text (heuristic-based)

Routing Strategy:
1. Use source hint if available (highest confidence)
2. Check for mixed content (split and route sections)
3. Detect content type (JSON, code, search, logs, text)
4. Route to appropriate compressor
5. Reassemble and return with routing metadata

Usage:
    >>> from headroom.transforms import ContentRouter
    >>> router = ContentRouter()
    >>> result = router.compress(content)  # Auto-routes to best compressor
    >>> print(result.strategy_used)
    >>> print(result.routing_log)

Pipeline Usage:
    >>> pipeline = TransformPipeline([
    ...     ContentRouter(),   # Handles all content types
    ...     RollingWindow(),   # Final size constraint
    ... ])
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import DEFAULT_EXCLUDE_TOOLS, TransformResult
from ..tokenizer import Tokenizer
from .base import Transform
from .content_detector import ContentType, detect_content_type

logger = logging.getLogger(__name__)


def _create_content_signature(
    content_type: str,
    content: str,
    language: str | None = None,
) -> Any:
    """Create a ToolSignature for non-JSON content types.

    This allows TOIN to track compression patterns for code, search results,
    logs, and text - not just JSON arrays.

    Args:
        content_type: The type of content (e.g., "code_aware", "search", "log", "text").
        content: The content being compressed (for structural hints).
        language: Optional language hint for code.

    Returns:
        A ToolSignature for TOIN tracking.
    """
    try:
        from ..telemetry.models import ToolSignature

        # Create a deterministic structure hash based on content type
        # This groups similar content types together for pattern learning
        if language:
            hash_input = f"content:{content_type}:{language}"
        else:
            hash_input = f"content:{content_type}"

        # Add a structural hint from the content (first 100 chars, hashed)
        # This helps differentiate tool outputs of the same type
        content_sample = content[:100] if content else ""
        structure_hint = hashlib.sha256(content_sample.encode()).hexdigest()[:8]
        hash_input = f"{hash_input}:{structure_hint}"

        structure_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:24]

        return ToolSignature(
            structure_hash=structure_hash,
            field_count=0,  # Not applicable for non-JSON
            has_nested_objects=False,
            has_arrays=False,
            max_depth=0,
        )
    except ImportError:
        return None


class CompressionStrategy(Enum):
    """Available compression strategies."""

    CODE_AWARE = "code_aware"
    SMART_CRUSHER = "smart_crusher"
    SEARCH = "search"
    LOG = "log"
    LLMLINGUA = "llmlingua"
    TEXT = "text"
    DIFF = "diff"
    HTML = "html"
    MIXED = "mixed"
    PASSTHROUGH = "passthrough"


@dataclass
class RoutingDecision:
    """Record of a single routing decision."""

    content_type: ContentType
    strategy: CompressionStrategy
    original_tokens: int
    compressed_tokens: int
    confidence: float = 1.0
    section_index: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


@dataclass
class ContentSection:
    """A typed section of content."""

    content: str
    content_type: ContentType
    language: str | None = None
    start_line: int = 0
    end_line: int = 0
    is_code_fence: bool = False


@dataclass
class RouterCompressionResult:
    """Result from ContentRouter with routing metadata.

    Attributes:
        compressed: The compressed content.
        original: Original content before compression.
        strategy_used: Primary strategy used for compression.
        routing_log: List of routing decisions made.
        sections_processed: Number of content sections processed.
    """

    compressed: str
    original: str
    strategy_used: CompressionStrategy
    routing_log: list[RoutingDecision] = field(default_factory=list)
    sections_processed: int = 1

    @property
    def total_original_tokens(self) -> int:
        """Total tokens before compression."""
        return sum(r.original_tokens for r in self.routing_log)

    @property
    def total_compressed_tokens(self) -> int:
        """Total tokens after compression."""
        return sum(r.compressed_tokens for r in self.routing_log)

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        if self.total_original_tokens == 0:
            return 1.0
        return self.total_compressed_tokens / self.total_original_tokens

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return max(0, self.total_original_tokens - self.total_compressed_tokens)

    @property
    def savings_percentage(self) -> float:
        """Percentage of tokens saved."""
        if self.total_original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.total_original_tokens) * 100

    def summary(self) -> str:
        """Human-readable routing summary."""
        if self.strategy_used == CompressionStrategy.MIXED:
            strategies = {r.strategy.value for r in self.routing_log}
            return (
                f"Mixed content: {self.sections_processed} sections, "
                f"routed to {strategies}. "
                f"{self.total_original_tokens:,}→{self.total_compressed_tokens:,} tokens "
                f"({self.savings_percentage:.0f}% saved)"
            )
        else:
            return (
                f"Pure {self.strategy_used.value}: "
                f"{self.total_original_tokens:,}→{self.total_compressed_tokens:,} tokens "
                f"({self.savings_percentage:.0f}% saved)"
            )


@dataclass
class ContentRouterConfig:
    """Configuration for intelligent content routing.

    Attributes:
        enable_code_aware: Enable AST-based code compression.
        enable_llmlingua: Enable ML-based text compression.
        enable_smart_crusher: Enable JSON array compression.
        enable_search_compressor: Enable search result compression.
        enable_log_compressor: Enable build/test log compression.
        enable_image_optimizer: Enable image token optimization.
        prefer_code_aware_for_code: Use CodeAware over LLMLingua for code.
        mixed_content_threshold: Min distinct types to consider "mixed".
        min_section_tokens: Minimum tokens for a section to compress.
        fallback_strategy: Strategy when no compressor matches.
        skip_user_messages: Never compress user messages (they're the subject).
        skip_recent_messages: Don't compress last N messages (likely the subject).
        protect_analysis_context: Detect "analyze/review" intent, skip compression.
    """

    # Enable/disable specific compressors
    enable_code_aware: bool = True
    enable_llmlingua: bool = True
    enable_smart_crusher: bool = True
    enable_search_compressor: bool = True
    enable_log_compressor: bool = True
    enable_html_extractor: bool = True  # HTML content extraction
    enable_image_optimizer: bool = True  # Image token optimization

    # Routing preferences
    prefer_code_aware_for_code: bool = True
    mixed_content_threshold: int = 2  # Min types to consider mixed
    min_section_tokens: int = 20  # Min tokens to compress a section

    # Fallback
    fallback_strategy: CompressionStrategy = CompressionStrategy.PASSTHROUGH

    # Protection: Don't compress content that's likely the subject of analysis
    skip_user_messages: bool = True  # User messages contain what they want analyzed
    protect_recent_code: int = 4  # Don't compress CODE in last N messages (0 = disabled)
    protect_analysis_context: bool = True  # Detect "analyze/review" intent, protect code

    # CCR (Compress-Cache-Retrieve) settings for SmartCrusher
    ccr_enabled: bool = True  # Enable CCR marker injection for reversible compression
    ccr_inject_marker: bool = True  # Add retrieval markers to compressed content

    # Tools to exclude from compression (output passed through unmodified)
    # Set to None to use DEFAULT_EXCLUDE_TOOLS, or provide custom set
    exclude_tools: set[str] | None = None


# Patterns for detecting mixed content
_CODE_FENCE_PATTERN = re.compile(r"^```(\w*)\s*$", re.MULTILINE)
_JSON_BLOCK_START = re.compile(r"^\s*[\[{]", re.MULTILINE)
_SEARCH_RESULT_PATTERN = re.compile(r"^\S+:\d+:", re.MULTILINE)


def is_mixed_content(content: str) -> bool:
    """Detect if content contains multiple distinct types.

    Args:
        content: Content to analyze.

    Returns:
        True if content appears to be mixed (multiple types).
    """
    indicators = {
        "has_code_fences": bool(_CODE_FENCE_PATTERN.search(content)),
        "has_json_blocks": bool(_JSON_BLOCK_START.search(content)),
        "has_prose": len(re.findall(r"[A-Z][a-z]+\s+\w+\s+\w+", content)) > 5,
        "has_search_results": bool(_SEARCH_RESULT_PATTERN.search(content)),
    }

    # Mixed if 2+ indicators are true
    return sum(indicators.values()) >= 2


def split_into_sections(content: str) -> list[ContentSection]:
    """Parse mixed content into typed sections.

    Args:
        content: Mixed content to split.

    Returns:
        List of ContentSection objects.
    """
    sections: list[ContentSection] = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code fence: ```language
        if match := _CODE_FENCE_PATTERN.match(line):
            language = match.group(1) or "unknown"
            code_lines = []
            start_line = i
            i += 1

            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1

            sections.append(
                ContentSection(
                    content="\n".join(code_lines),
                    content_type=ContentType.SOURCE_CODE,
                    language=language,
                    start_line=start_line,
                    end_line=i,
                    is_code_fence=True,
                )
            )
            i += 1  # Skip closing ```
            continue

        # JSON block
        if line.strip().startswith(("[", "{")):
            json_content, end_i = _extract_json_block(lines, i)
            if json_content:
                sections.append(
                    ContentSection(
                        content=json_content,
                        content_type=ContentType.JSON_ARRAY,
                        start_line=i,
                        end_line=end_i,
                    )
                )
                i = end_i + 1
                continue

        # Search result lines
        if _SEARCH_RESULT_PATTERN.match(line):
            search_lines = []
            start_line = i
            while i < len(lines) and _SEARCH_RESULT_PATTERN.match(lines[i]):
                search_lines.append(lines[i])
                i += 1
            sections.append(
                ContentSection(
                    content="\n".join(search_lines),
                    content_type=ContentType.SEARCH_RESULTS,
                    start_line=start_line,
                    end_line=i - 1,
                )
            )
            continue

        # Collect text until next special section
        text_lines = [line]
        start_line = i
        i += 1

        while i < len(lines):
            next_line = lines[i]
            # Stop if we hit a special section
            if (
                _CODE_FENCE_PATTERN.match(next_line)
                or next_line.strip().startswith(("[", "{"))
                or _SEARCH_RESULT_PATTERN.match(next_line)
            ):
                break
            text_lines.append(next_line)
            i += 1

        # Only add non-empty text sections
        text_content = "\n".join(text_lines)
        if text_content.strip():
            sections.append(
                ContentSection(
                    content=text_content,
                    content_type=ContentType.PLAIN_TEXT,
                    start_line=start_line,
                    end_line=i - 1,
                )
            )

    return sections


def _extract_json_block(lines: list[str], start: int) -> tuple[str | None, int]:
    """Extract a complete JSON block from lines.

    Args:
        lines: All lines of content.
        start: Starting line index.

    Returns:
        Tuple of (json_content, end_line_index) or (None, start) if invalid.
    """
    bracket_count = 0
    brace_count = 0
    json_lines = []

    for i in range(start, len(lines)):
        line = lines[i]
        json_lines.append(line)

        bracket_count += line.count("[") - line.count("]")
        brace_count += line.count("{") - line.count("}")

        if bracket_count <= 0 and brace_count <= 0 and json_lines:
            return "\n".join(json_lines), i

    # Didn't find complete JSON
    return None, start


class ContentRouter(Transform):
    """Intelligent router that selects optimal compression strategy.

    ContentRouter is the recommended entry point for Headroom's compression.
    It analyzes content and routes it to the most appropriate compressor,
    handling mixed content by splitting and reassembling.

    Key Features:
    - Automatic content type detection
    - Source hint support for high-confidence routing
    - Mixed content handling (split → route → reassemble)
    - Graceful fallback when compressors unavailable
    - Rich routing metadata for debugging

    Example:
        >>> router = ContentRouter()
        >>>
        >>> # Automatically uses CodeAwareCompressor
        >>> result = router.compress(python_code)
        >>> print(result.strategy_used)  # CompressionStrategy.CODE_AWARE
        >>>
        >>> # Automatically uses SmartCrusher
        >>> result = router.compress(json_array)
        >>> print(result.strategy_used)  # CompressionStrategy.SMART_CRUSHER
        >>>
        >>> # Splits and routes each section
        >>> result = router.compress(readme_with_code)
        >>> print(result.strategy_used)  # CompressionStrategy.MIXED

    Pipeline Integration:
        >>> pipeline = TransformPipeline([
        ...     ContentRouter(),   # Handles ALL content types
        ...     RollingWindow(),   # Final size constraint
        ... ])
    """

    name: str = "content_router"

    def __init__(self, config: ContentRouterConfig | None = None):
        """Initialize content router.

        Args:
            config: Router configuration. Uses defaults if None.
        """
        self.config = config or ContentRouterConfig()

        # Lazy-loaded compressors
        self._code_compressor: Any = None
        self._smart_crusher: Any = None
        self._search_compressor: Any = None
        self._log_compressor: Any = None
        self._diff_compressor: Any = None
        self._html_extractor: Any = None
        self._llmlingua: Any = None
        self._text_compressor: Any = None
        self._image_optimizer: Any = None

        # TOIN integration for cross-strategy learning
        self._toin: Any = None

    def _record_to_toin(
        self,
        strategy: CompressionStrategy,
        content: str,
        compressed: str,
        original_tokens: int,
        compressed_tokens: int,
        language: str | None = None,
        context: str = "",
    ) -> None:
        """Record compression to TOIN for cross-user learning.

        This allows TOIN to track compression patterns for ALL content types,
        not just JSON arrays. When the LLM retrieves original content via CCR,
        TOIN learns which compressions users need to expand.

        Args:
            strategy: The compression strategy used.
            content: Original content (for signature generation).
            compressed: Compressed content.
            original_tokens: Token count before compression.
            compressed_tokens: Token count after compression.
            language: Optional language hint for code.
            context: Query context for pattern learning.
        """
        # Skip SmartCrusher - it handles its own TOIN recording
        if strategy == CompressionStrategy.SMART_CRUSHER:
            return

        # Skip if no actual compression happened
        if original_tokens <= compressed_tokens:
            return

        try:
            # Lazy load TOIN
            if self._toin is None:
                from ..telemetry.toin import get_toin

                self._toin = get_toin()

            # Create a content-type signature
            signature = _create_content_signature(
                content_type=strategy.value,
                content=content,
                language=language,
            )

            if signature is None:
                return

            # Record the compression
            self._toin.record_compression(
                tool_signature=signature,
                original_count=1,  # Single content block
                compressed_count=1,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                strategy=strategy.value,
                query_context=context if context else None,
            )

            logger.debug(
                "TOIN: Recorded %s compression: %d → %d tokens",
                strategy.value,
                original_tokens,
                compressed_tokens,
            )

        except Exception as e:
            # TOIN recording should never break compression
            logger.debug("TOIN recording failed (non-fatal): %s", e)

    def compress(
        self,
        content: str,
        context: str = "",
    ) -> RouterCompressionResult:
        """Compress content using optimal strategy based on content detection.

        Args:
            content: Content to compress.
            context: Optional context for relevance-aware compression.

        Returns:
            RouterCompressionResult with compressed content and routing metadata.
        """
        if not content or not content.strip():
            return RouterCompressionResult(
                compressed=content,
                original=content,
                strategy_used=CompressionStrategy.PASSTHROUGH,
                routing_log=[],
            )

        # Determine strategy from content analysis
        strategy = self._determine_strategy(content)

        if strategy == CompressionStrategy.MIXED:
            return self._compress_mixed(content, context)
        else:
            return self._compress_pure(content, strategy, context)

    def _determine_strategy(self, content: str) -> CompressionStrategy:
        """Determine the compression strategy from content analysis.

        Args:
            content: Content to analyze.

        Returns:
            Selected compression strategy.
        """
        # 1. Check for mixed content
        if is_mixed_content(content):
            return CompressionStrategy.MIXED

        # 2. Detect content type from content itself
        detection = detect_content_type(content)
        return self._strategy_from_detection(detection)

    def _strategy_from_detection(self, detection: Any) -> CompressionStrategy:
        """Get strategy from content detection result.

        Args:
            detection: Result from detect_content_type.

        Returns:
            Selected strategy.
        """
        mapping = {
            ContentType.SOURCE_CODE: CompressionStrategy.CODE_AWARE,
            ContentType.JSON_ARRAY: CompressionStrategy.SMART_CRUSHER,
            ContentType.SEARCH_RESULTS: CompressionStrategy.SEARCH,
            ContentType.BUILD_OUTPUT: CompressionStrategy.LOG,
            ContentType.GIT_DIFF: CompressionStrategy.DIFF,
            ContentType.HTML: CompressionStrategy.HTML,
            ContentType.PLAIN_TEXT: CompressionStrategy.TEXT,
        }

        strategy = mapping.get(detection.content_type, self.config.fallback_strategy)

        # Override: prefer CodeAware for code if configured
        if (
            strategy == CompressionStrategy.CODE_AWARE
            and not self.config.prefer_code_aware_for_code
        ):
            strategy = CompressionStrategy.LLMLINGUA

        return strategy

    def _compress_mixed(
        self,
        content: str,
        context: str,
    ) -> RouterCompressionResult:
        """Compress mixed content by splitting and routing sections.

        Args:
            content: Mixed content to compress.
            context: User context for relevance.

        Returns:
            RouterCompressionResult with reassembled content.
        """
        sections = split_into_sections(content)

        if not sections:
            return RouterCompressionResult(
                compressed=content,
                original=content,
                strategy_used=CompressionStrategy.PASSTHROUGH,
            )

        compressed_sections: list[str] = []
        routing_log: list[RoutingDecision] = []

        for i, section in enumerate(sections):
            # Get strategy for this section
            strategy = self._strategy_from_detection_type(section.content_type)

            # Compress section
            original_tokens = len(section.content.split())
            compressed_content, compressed_tokens = self._apply_strategy_to_content(
                section.content, strategy, context, section.language
            )

            # Preserve code fence markers
            if section.is_code_fence and section.language:
                compressed_content = f"```{section.language}\n{compressed_content}\n```"

            compressed_sections.append(compressed_content)
            routing_log.append(
                RoutingDecision(
                    content_type=section.content_type,
                    strategy=strategy,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    section_index=i,
                )
            )

        return RouterCompressionResult(
            compressed="\n\n".join(compressed_sections),
            original=content,
            strategy_used=CompressionStrategy.MIXED,
            routing_log=routing_log,
            sections_processed=len(sections),
        )

    def _compress_pure(
        self,
        content: str,
        strategy: CompressionStrategy,
        context: str,
    ) -> RouterCompressionResult:
        """Compress pure (non-mixed) content.

        Args:
            content: Content to compress.
            strategy: Selected strategy.
            context: User context.

        Returns:
            RouterCompressionResult.
        """
        original_tokens = len(content.split())

        compressed, compressed_tokens = self._apply_strategy_to_content(content, strategy, context)

        return RouterCompressionResult(
            compressed=compressed,
            original=content,
            strategy_used=strategy,
            routing_log=[
                RoutingDecision(
                    content_type=self._content_type_from_strategy(strategy),
                    strategy=strategy,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                )
            ],
        )

    def _apply_strategy_to_content(
        self,
        content: str,
        strategy: CompressionStrategy,
        context: str,
        language: str | None = None,
    ) -> tuple[str, int]:
        """Apply a compression strategy to content.

        Args:
            content: Content to compress.
            strategy: Strategy to use.
            context: User context.
            language: Language hint for code.

        Returns:
            Tuple of (compressed_content, compressed_token_count).
        """
        # Track original tokens for TOIN recording
        original_tokens = len(content.split())
        compressed: str | None = None
        compressed_tokens: int | None = None

        try:
            if strategy == CompressionStrategy.CODE_AWARE:
                if self.config.enable_code_aware:
                    compressor = self._get_code_compressor()
                    if compressor:
                        result = compressor.compress(content, language=language, context=context)
                        compressed, compressed_tokens = result.compressed, result.compressed_tokens
                if compressed is None:
                    # Fallback to LLMLingua
                    compressed, compressed_tokens = self._try_llmlingua(content, context)
                    strategy = CompressionStrategy.LLMLINGUA  # Update for TOIN

            elif strategy == CompressionStrategy.SMART_CRUSHER:
                # SmartCrusher handles its own TOIN recording
                if self.config.enable_smart_crusher:
                    crusher = self._get_smart_crusher()
                    if crusher:
                        result = crusher.crush(content, query=context)
                        return result.compressed, len(result.compressed.split())

            elif strategy == CompressionStrategy.SEARCH:
                if self.config.enable_search_compressor:
                    compressor = self._get_search_compressor()
                    if compressor:
                        result = compressor.compress(content, context=context)
                        compressed, compressed_tokens = (
                            result.compressed,
                            len(result.compressed.split()),
                        )

            elif strategy == CompressionStrategy.LOG:
                if self.config.enable_log_compressor:
                    compressor = self._get_log_compressor()
                    if compressor:
                        result = compressor.compress(content)
                        compressed, compressed_tokens = (
                            result.compressed,
                            result.compressed_line_count,
                        )

            elif strategy == CompressionStrategy.DIFF:
                compressor = self._get_diff_compressor()
                if compressor:
                    result = compressor.compress(content, context=context)
                    compressed, compressed_tokens = (
                        result.compressed,
                        result.compressed_line_count,
                    )

            elif strategy == CompressionStrategy.HTML:
                if self.config.enable_html_extractor:
                    extractor = self._get_html_extractor()
                    if extractor:
                        result = extractor.extract(content)
                        compressed = result.extracted
                        # Estimate tokens from extracted text (simple word count)
                        compressed_tokens = len(compressed.split()) if compressed else 0

            elif strategy == CompressionStrategy.LLMLINGUA:
                compressed, compressed_tokens = self._try_llmlingua(content, context)

            elif strategy == CompressionStrategy.TEXT:
                # Prefer LLMLingua for text if available (ML-based, better compression)
                # Falls back to heuristic TextCompressor if LLMLingua unavailable
                compressed, compressed_tokens = self._try_llmlingua(content, context)

        except Exception as e:
            logger.warning("Compression with %s failed: %s", strategy.value, e)

        # If compression succeeded, record to TOIN
        if compressed is not None and compressed_tokens is not None:
            self._record_to_toin(
                strategy=strategy,
                content=content,
                compressed=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                language=language,
                context=context,
            )
            return compressed, compressed_tokens

        # Fallback: return unchanged
        return content, original_tokens

    def _try_llmlingua(self, content: str, context: str) -> tuple[str, int]:
        """Try LLMLingua compression with fallback.

        Args:
            content: Content to compress.
            context: User context.

        Returns:
            Tuple of (compressed, token_count).
        """
        if self.config.enable_llmlingua:
            compressor = self._get_llmlingua()
            if compressor:
                try:
                    result = compressor.compress(content, context=context)
                    return result.compressed, result.compressed_tokens
                except Exception as e:
                    logger.debug("LLMLingua failed: %s", e)

        # Fallback to text compressor
        compressor = self._get_text_compressor()
        if compressor:
            result = compressor.compress(content, context=context)
            return result.compressed, result.compressed_line_count

        return content, len(content.split())

    def _strategy_from_detection_type(self, content_type: ContentType) -> CompressionStrategy:
        """Get strategy from ContentType enum."""
        mapping = {
            ContentType.SOURCE_CODE: CompressionStrategy.CODE_AWARE,
            ContentType.JSON_ARRAY: CompressionStrategy.SMART_CRUSHER,
            ContentType.SEARCH_RESULTS: CompressionStrategy.SEARCH,
            ContentType.BUILD_OUTPUT: CompressionStrategy.LOG,
            ContentType.GIT_DIFF: CompressionStrategy.DIFF,
            ContentType.HTML: CompressionStrategy.HTML,
            ContentType.PLAIN_TEXT: CompressionStrategy.TEXT,
        }
        return mapping.get(content_type, self.config.fallback_strategy)

    def _content_type_from_strategy(self, strategy: CompressionStrategy) -> ContentType:
        """Get ContentType from strategy."""
        mapping = {
            CompressionStrategy.CODE_AWARE: ContentType.SOURCE_CODE,
            CompressionStrategy.SMART_CRUSHER: ContentType.JSON_ARRAY,
            CompressionStrategy.SEARCH: ContentType.SEARCH_RESULTS,
            CompressionStrategy.LOG: ContentType.BUILD_OUTPUT,
            CompressionStrategy.DIFF: ContentType.GIT_DIFF,
            CompressionStrategy.HTML: ContentType.HTML,
            CompressionStrategy.TEXT: ContentType.PLAIN_TEXT,
            CompressionStrategy.LLMLINGUA: ContentType.PLAIN_TEXT,
            CompressionStrategy.PASSTHROUGH: ContentType.PLAIN_TEXT,
        }
        return mapping.get(strategy, ContentType.PLAIN_TEXT)

    # Lazy compressor getters

    def _get_code_compressor(self) -> Any:
        """Get CodeAwareCompressor (lazy load)."""
        if self._code_compressor is None:
            try:
                from .code_compressor import CodeAwareCompressor, _check_tree_sitter_available

                if _check_tree_sitter_available():
                    self._code_compressor = CodeAwareCompressor()
                else:
                    logger.debug("tree-sitter not available")
            except ImportError:
                logger.debug("CodeAwareCompressor not available")
        return self._code_compressor

    def _get_smart_crusher(self) -> Any:
        """Get SmartCrusher (lazy load) with CCR config."""
        if self._smart_crusher is None:
            try:
                from ..config import CCRConfig
                from .smart_crusher import SmartCrusher

                # Pass CCR config for marker injection
                ccr_config = CCRConfig(
                    enabled=self.config.ccr_enabled,
                    inject_retrieval_marker=self.config.ccr_inject_marker,
                )
                self._smart_crusher = SmartCrusher(ccr_config=ccr_config)
            except ImportError:
                logger.debug("SmartCrusher not available")
        return self._smart_crusher

    def _get_search_compressor(self) -> Any:
        """Get SearchCompressor (lazy load)."""
        if self._search_compressor is None:
            try:
                from .search_compressor import SearchCompressor

                self._search_compressor = SearchCompressor()
            except ImportError:
                logger.debug("SearchCompressor not available")
        return self._search_compressor

    def _get_log_compressor(self) -> Any:
        """Get LogCompressor (lazy load)."""
        if self._log_compressor is None:
            try:
                from .log_compressor import LogCompressor

                self._log_compressor = LogCompressor()
            except ImportError:
                logger.debug("LogCompressor not available")
        return self._log_compressor

    def _get_diff_compressor(self) -> Any:
        """Get DiffCompressor (lazy load)."""
        if self._diff_compressor is None:
            try:
                from .diff_compressor import DiffCompressor

                self._diff_compressor = DiffCompressor()
            except ImportError:
                logger.debug("DiffCompressor not available")
        return self._diff_compressor

    def _get_html_extractor(self) -> Any:
        """Get HTMLExtractor (lazy load)."""
        if self._html_extractor is None:
            try:
                from .html_extractor import HTMLExtractor

                self._html_extractor = HTMLExtractor()
            except ImportError:
                logger.debug("HTMLExtractor not available (install trafilatura)")
        return self._html_extractor

    def _get_llmlingua(self) -> Any:
        """Get LLMLinguaCompressor (lazy load)."""
        if self._llmlingua is None:
            try:
                from .llmlingua_compressor import (
                    LLMLinguaCompressor,
                    _check_llmlingua_available,
                )

                if _check_llmlingua_available():
                    self._llmlingua = LLMLinguaCompressor()
            except ImportError:
                logger.debug("LLMLinguaCompressor not available")
        return self._llmlingua

    def _get_text_compressor(self) -> Any:
        """Get TextCompressor (lazy load)."""
        if self._text_compressor is None:
            try:
                from .text_compressor import TextCompressor

                self._text_compressor = TextCompressor()
            except ImportError:
                logger.debug("TextCompressor not available")
        return self._text_compressor

    def _get_image_optimizer(self) -> Any:
        """Get ImageCompressor (lazy load).

        The ImageCompressor handles image token compression using:
        - Trained MiniLM classifier from HuggingFace (chopratejas/technique-router)
        - SigLIP for image analysis
        - Provider-specific compression (OpenAI detail, Anthropic/Google resize)
        """
        if self._image_optimizer is None:
            try:
                from ..image import ImageCompressor

                self._image_optimizer = ImageCompressor()
            except ImportError:
                logger.debug("ImageCompressor not available")
        return self._image_optimizer

    def optimize_images_in_messages(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        provider: str = "openai",
        user_query: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Optimize images in messages.

        This is a convenience method for image optimization that can be called
        directly or as part of the transform pipeline.

        Uses ImageCompressor with trained MiniLM router from HuggingFace
        (chopratejas/technique-router) + SigLIP for image analysis.

        Args:
            messages: Messages potentially containing images.
            tokenizer: Tokenizer for token counting (unused, kept for API compat).
            provider: LLM provider (openai, anthropic, google).
            user_query: User query for task intent detection (unused, auto-extracted).

        Returns:
            Tuple of (optimized_messages, metrics).
        """
        if not self.config.enable_image_optimizer:
            return messages, {"images_optimized": 0, "tokens_saved": 0}

        compressor = self._get_image_optimizer()
        if compressor is None:
            return messages, {"images_optimized": 0, "tokens_saved": 0}

        # Check if there are images to compress
        if not compressor.has_images(messages):
            return messages, {"images_optimized": 0, "tokens_saved": 0}

        # Compress images (query is auto-extracted from messages)
        optimized = compressor.compress(messages, provider=provider)

        # Get metrics from last compression
        result = compressor.last_result
        if result:
            metrics = {
                "images_optimized": result.compressed_tokens < result.original_tokens,
                "tokens_before": result.original_tokens,
                "tokens_after": result.compressed_tokens,
                "tokens_saved": result.original_tokens - result.compressed_tokens,
                "technique": result.technique.value,
                "confidence": result.confidence,
            }
        else:
            metrics = {"images_optimized": 0, "tokens_saved": 0}

        return optimized, metrics

    # Transform interface

    def _build_tool_name_map(self, messages: list[dict[str, Any]]) -> dict[str, str]:
        """Build mapping from tool_call_id to tool_name.

        Scans assistant messages to find tool calls and extract their names.
        Supports both OpenAI and Anthropic message formats.
        """
        mapping: dict[str, str] = {}

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            # OpenAI format: tool_calls array
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    name = tc.get("function", {}).get("name", "")
                    if tc_id and name:
                        mapping[tc_id] = name

            # Anthropic format: content blocks with type=tool_use
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tc_id = block.get("id", "")
                        name = block.get("name", "")
                        if tc_id and name:
                            mapping[tc_id] = name

        return mapping

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Apply intelligent routing to messages.

        Args:
            messages: Messages to transform.
            tokenizer: Tokenizer for counting.
            **kwargs: Additional arguments (context).

        Returns:
            TransformResult with routed and compressed messages.
        """
        tokens_before = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
        context = kwargs.get("context", "")

        # Build tool name map for exclusion checking
        tool_name_map = self._build_tool_name_map(messages)

        # Compute excluded tool IDs based on config
        exclude_tools = (
            self.config.exclude_tools
            if self.config.exclude_tools is not None
            else DEFAULT_EXCLUDE_TOOLS
        )
        excluded_tool_ids = {
            tool_id for tool_id, name in tool_name_map.items() if name in exclude_tools
        }

        transformed_messages: list[dict[str, Any]] = []
        transforms_applied: list[str] = []
        warnings: list[str] = []

        # Check for analysis intent in the most recent user message
        analysis_intent = False
        if self.config.protect_analysis_context:
            analysis_intent = self._detect_analysis_intent(messages)

        num_messages = len(messages)

        for i, message in enumerate(messages):
            role = message.get("role", "")
            content = message.get("content", "")

            # Handle list content (Anthropic format with content blocks)
            if isinstance(content, list):
                transformed_message = self._process_content_blocks(
                    message, content, context, transforms_applied, excluded_tool_ids
                )
                transformed_messages.append(transformed_message)
                continue

            # Skip non-string content (other types)
            if not isinstance(content, str):
                transformed_messages.append(message)
                continue

            # Skip OpenAI-style tool messages for excluded tools
            if role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                if tool_call_id in excluded_tool_ids:
                    transformed_messages.append(message)
                    transforms_applied.append("router:excluded:tool")
                    continue

            # Protection 1: Never compress user messages
            if self.config.skip_user_messages and role == "user":
                transformed_messages.append(message)
                transforms_applied.append("router:protected:user_message")
                continue

            if not content or len(content.split()) < 50:
                # Skip small content
                transformed_messages.append(message)
                continue

            # Detect content type for protection decisions
            detection = detect_content_type(content)
            is_code = detection.content_type == ContentType.SOURCE_CODE

            # Protection 2: Don't compress recent CODE
            messages_from_end = num_messages - i
            if (
                self.config.protect_recent_code > 0
                and messages_from_end <= self.config.protect_recent_code
                and is_code
            ):
                transformed_messages.append(message)
                transforms_applied.append("router:protected:recent_code")
                continue

            # Protection 3: Don't compress CODE when analysis intent detected
            if analysis_intent and is_code:
                transformed_messages.append(message)
                transforms_applied.append("router:protected:analysis_context")
                continue

            # Route and compress based on content detection
            result = self.compress(content, context=context)

            if result.compression_ratio < 0.9:
                transformed_messages.append({**message, "content": result.compressed})
                transforms_applied.append(
                    f"router:{result.strategy_used.value}:{result.compression_ratio:.2f}"
                )
            else:
                transformed_messages.append(message)

        tokens_after = sum(
            tokenizer.count_text(str(m.get("content", ""))) for m in transformed_messages
        )

        return TransformResult(
            messages=transformed_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied if transforms_applied else ["router:noop"],
            warnings=warnings,
        )

    def _process_content_blocks(
        self,
        message: dict[str, Any],
        content_blocks: list[Any],
        context: str,
        transforms_applied: list[str],
        excluded_tool_ids: set[str],
    ) -> dict[str, Any]:
        """Process content blocks (Anthropic format) for tool_result compression.

        Handles tool_result blocks by compressing their string content using
        the appropriate strategy (typically SmartCrusher for JSON arrays).

        Args:
            message: The original message.
            content_blocks: List of content blocks.
            context: Context for compression.
            transforms_applied: List to append transform names to.

        Returns:
            Transformed message with compressed content blocks.
        """
        new_blocks = []
        any_compressed = False

        for block in content_blocks:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue

            block_type = block.get("type")

            # Handle tool_result blocks
            if block_type == "tool_result":
                # Check if tool is excluded from compression
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id in excluded_tool_ids:
                    new_blocks.append(block)
                    transforms_applied.append("router:excluded:tool")
                    continue

                tool_content = block.get("content", "")

                # Only process string content
                if isinstance(tool_content, str) and len(tool_content) > 500:
                    # Compress using content detection (will auto-detect JSON arrays, etc.)
                    result = self.compress(tool_content, context=context)
                    if result.compression_ratio < 0.9:
                        new_blocks.append({**block, "content": result.compressed})
                        transforms_applied.append(
                            f"router:tool_result:{result.strategy_used.value}"
                        )
                        any_compressed = True
                        continue

            # Keep block unchanged
            new_blocks.append(block)

        if any_compressed:
            return {**message, "content": new_blocks}
        return message

    def _detect_analysis_intent(self, messages: list[dict[str, Any]]) -> bool:
        """Detect if user wants to analyze/review code.

        Looks at the most recent user message for analysis keywords.

        Args:
            messages: Conversation messages.

        Returns:
            True if analysis intent detected.
        """
        # Analysis keywords that suggest user wants full code details
        analysis_keywords = {
            "analyze",
            "analyse",
            "review",
            "audit",
            "inspect",
            "security",
            "vulnerability",
            "bug",
            "issue",
            "problem",
            "explain",
            "understand",
            "how does",
            "what does",
            "debug",
            "fix",
            "error",
            "wrong",
            "broken",
            "refactor",
            "improve",
            "optimize",
            "clean up",
        }

        # Find most recent user message
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    content_lower = content.lower()
                    for keyword in analysis_keywords:
                        if keyword in content_lower:
                            return True
                break

        return False

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if routing should be applied.

        Always returns True - the router handles all content types.
        """
        return True


def route_and_compress(
    content: str,
    context: str = "",
) -> str:
    """Convenience function for one-off routing and compression.

    Args:
        content: Content to compress.
        context: Optional context for relevance-aware compression.

    Returns:
        Compressed content.

    Example:
        >>> compressed = route_and_compress(mixed_content)
    """
    router = ContentRouter()
    result = router.compress(content, context=context)
    return result.compressed
