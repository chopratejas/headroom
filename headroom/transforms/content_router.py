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

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer
from .base import Transform
from .content_detector import ContentType, detect_content_type

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Available compression strategies."""

    CODE_AWARE = "code_aware"
    SMART_CRUSHER = "smart_crusher"
    SEARCH = "search"
    LOG = "log"
    LLMLINGUA = "llmlingua"
    TEXT = "text"
    DIFF = "diff"
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


def generate_source_hint(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Generate a source hint from tool metadata.

    This enables higher-confidence routing decisions.

    Args:
        tool_name: Name of the tool that produced the output.
        tool_input: Input parameters to the tool.

    Returns:
        Source hint string (e.g., "file:auth.py", "tool:grep").
    """
    # File read operations
    if tool_name in ("Read", "read_file", "cat", "ReadFile"):
        file_path = tool_input.get("file_path", tool_input.get("path", ""))
        if file_path:
            return f"file:{file_path}"

    # Search operations
    if tool_name in ("Grep", "grep", "ripgrep", "rg", "search", "Search"):
        return "tool:grep"

    # Glob operations
    if tool_name in ("Glob", "glob", "find"):
        return "tool:glob"

    # Build/test operations
    if tool_name == "Bash":
        command = str(tool_input.get("command", ""))
        if any(cmd in command for cmd in ["pytest", "npm test", "cargo test", "go test"]):
            return "tool:pytest"
        if any(cmd in command for cmd in ["npm run build", "cargo build", "make"]):
            return "tool:build"
        if "git diff" in command:
            return "tool:git-diff"
        if "git log" in command:
            return "tool:git-log"

    # Web fetch
    if tool_name in ("WebFetch", "fetch", "curl", "WebSearch"):
        return "tool:web"

    return ""


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
        self._llmlingua: Any = None
        self._text_compressor: Any = None

    def compress(
        self,
        content: str,
        source_hint: str | None = None,
        context: str = "",
    ) -> RouterCompressionResult:
        """Compress content using optimal strategy.

        Args:
            content: Content to compress.
            source_hint: Optional hint about content source.
                Examples: "file:auth.py", "tool:grep", "tool:pytest"
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

        # Determine strategy
        strategy = self._determine_strategy(content, source_hint)

        if strategy == CompressionStrategy.MIXED:
            return self._compress_mixed(content, context)
        else:
            return self._compress_pure(content, strategy, context)

    def _determine_strategy(
        self,
        content: str,
        source_hint: str | None,
    ) -> CompressionStrategy:
        """Determine the compression strategy.

        Args:
            content: Content to analyze.
            source_hint: Optional source hint.

        Returns:
            Selected compression strategy.
        """
        # 1. Source hint takes priority
        if source_hint:
            strategy = self._strategy_from_hint(source_hint)
            if strategy:
                return strategy

        # 2. Check for mixed content
        if is_mixed_content(content):
            return CompressionStrategy.MIXED

        # 3. Detect content type
        detection = detect_content_type(content)
        return self._strategy_from_detection(detection)

    def _strategy_from_hint(self, hint: str) -> CompressionStrategy | None:
        """Get strategy from source hint.

        Args:
            hint: Source hint string.

        Returns:
            Strategy if determinable, None otherwise.
        """
        hint_lower = hint.lower()

        # File hints
        if hint_lower.startswith("file:"):
            file_path = hint_lower[5:]
            if file_path.endswith((".py", ".pyw")):
                return CompressionStrategy.CODE_AWARE
            if file_path.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs")):
                return CompressionStrategy.CODE_AWARE
            if file_path.endswith((".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp")):
                return CompressionStrategy.CODE_AWARE
            if file_path.endswith(".json"):
                return CompressionStrategy.SMART_CRUSHER
            if file_path.endswith((".md", ".txt", ".rst")):
                return CompressionStrategy.TEXT
            if file_path.endswith((".log", ".out")):
                return CompressionStrategy.LOG

        # Tool hints
        if hint_lower.startswith("tool:"):
            tool = hint_lower[5:]
            if tool in ("grep", "rg", "ripgrep", "ag", "search"):
                return CompressionStrategy.SEARCH
            if tool in ("pytest", "jest", "cargo-test", "go-test", "npm-test"):
                return CompressionStrategy.LOG
            if tool in ("build", "make", "cargo-build", "npm-build"):
                return CompressionStrategy.LOG
            if tool in ("git-diff", "diff"):
                return CompressionStrategy.DIFF

        return None

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
        try:
            if strategy == CompressionStrategy.CODE_AWARE:
                if self.config.enable_code_aware:
                    compressor = self._get_code_compressor()
                    if compressor:
                        result = compressor.compress(content, language=language, context=context)
                        return result.compressed, result.compressed_tokens
                # Fallback to LLMLingua
                return self._try_llmlingua(content, context)

            elif strategy == CompressionStrategy.SMART_CRUSHER:
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
                        return result.compressed, len(result.compressed.split())

            elif strategy == CompressionStrategy.LOG:
                if self.config.enable_log_compressor:
                    compressor = self._get_log_compressor()
                    if compressor:
                        result = compressor.compress(content)
                        return result.compressed, result.compressed_line_count

            elif strategy == CompressionStrategy.LLMLINGUA:
                return self._try_llmlingua(content, context)

            elif strategy == CompressionStrategy.TEXT:
                compressor = self._get_text_compressor()
                if compressor:
                    result = compressor.compress(content, context=context)
                    return result.compressed, result.compressed_line_count

        except Exception as e:
            logger.warning("Compression with %s failed: %s", strategy.value, e)

        # Fallback: return unchanged
        return content, len(content.split())

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

    # Transform interface

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
            **kwargs: Additional arguments (context, source_hints).

        Returns:
            TransformResult with routed and compressed messages.
        """
        tokens_before = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
        context = kwargs.get("context", "")
        source_hints = kwargs.get("source_hints", {})  # message_id -> hint

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
                    message, content, context, transforms_applied
                )
                transformed_messages.append(transformed_message)
                continue

            # Skip non-string content (other types)
            if not isinstance(content, str):
                transformed_messages.append(message)
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

            # Get source hint if available
            source_hint = source_hints.get(i) or source_hints.get(str(i))

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

            # Route and compress
            result = self.compress(content, source_hint=source_hint, context=context)

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
        import json

        new_blocks = []
        any_compressed = False

        for block in content_blocks:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue

            block_type = block.get("type")

            # Handle tool_result blocks
            if block_type == "tool_result":
                tool_content = block.get("content", "")

                # Only process string content
                if isinstance(tool_content, str) and len(tool_content) > 500:
                    # Try to detect if it's JSON array data (SmartCrusher target)
                    try:
                        parsed = json.loads(tool_content)
                        if isinstance(parsed, list) and len(parsed) > 10:
                            # Route to SmartCrusher for arrays
                            result = self.compress(
                                tool_content,
                                source_hint="json_array",
                                context=context,
                            )
                            if result.compression_ratio < 0.9:
                                new_blocks.append(
                                    {
                                        **block,
                                        "content": result.compressed,
                                    }
                                )
                                transforms_applied.append(
                                    f"router:tool_result:{result.strategy_used.value}"
                                )
                                any_compressed = True
                                continue
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON, try general compression
                        pass

                    # Try general compression for large non-JSON content
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
    source_hint: str | None = None,
    context: str = "",
) -> str:
    """Convenience function for one-off routing and compression.

    Args:
        content: Content to compress.
        source_hint: Optional source hint.
        context: Optional context.

    Returns:
        Compressed content.

    Example:
        >>> compressed = route_and_compress(mixed_content)
    """
    router = ContentRouter()
    result = router.compress(content, source_hint=source_hint, context=context)
    return result.compressed
