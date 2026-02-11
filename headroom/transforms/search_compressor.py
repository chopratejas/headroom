"""Search results compressor for grep/ripgrep output.

This module compresses search results (grep, ripgrep, ag) which are one of
the most common outputs in coding tasks. Typical compression: 5-10x.

Input Format (grep -n style):
    src/utils.py:42:def process_data(items):
    src/utils.py:43:    \"\"\"Process items with validation.\"\"\"
    src/models.py:15:class DataProcessor:

Compression Strategy:
1. Parse into {file: [(line, content), ...]} structure
2. Group by file
3. For each file: keep first match, last match, context-relevant matches
4. Deduplicate near-identical lines
5. Add summary: [... and N more matches in file.py]

Integrates with CCR for reversible compression.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SearchMatch:
    """A single search match."""

    file: str
    line_number: int
    content: str
    score: float = 0.0  # Relevance score


@dataclass
class FileMatches:
    """All matches in a single file."""

    file: str
    matches: list[SearchMatch] = field(default_factory=list)

    @property
    def first(self) -> SearchMatch | None:
        return self.matches[0] if self.matches else None

    @property
    def last(self) -> SearchMatch | None:
        return self.matches[-1] if self.matches else None


@dataclass
class SearchCompressorConfig:
    """Configuration for search result compression."""

    # Per-file limits
    max_matches_per_file: int = 5
    always_keep_first: bool = True
    always_keep_last: bool = True

    # Global limits
    max_total_matches: int = 30
    max_files: int = 15

    # Context matching
    context_keywords: list[str] = field(default_factory=list)
    boost_errors: bool = True

    # CCR integration
    enable_ccr: bool = True
    min_matches_for_ccr: int = 10


class SearchCompressor:
    """Compresses grep/ripgrep search results.

    Example:
        >>> compressor = SearchCompressor()
        >>> result = compressor.compress(search_output, context="find error handlers")
        >>> print(result.compressed)  # Reduced output with summary
    """

    # Pattern to parse grep-style output: file:line:content
    _GREP_PATTERN = re.compile(r"^([^:]+):(\d+):(.*)$")

    # Pattern for ripgrep with context (file-line-content or file:line:content)
    _RG_CONTEXT_PATTERN = re.compile(r"^([^:-]+)[:-](\d+)[:-](.*)$")

    # Error/important patterns to prioritize
    _PRIORITY_PATTERNS = [
        re.compile(r"\b(error|exception|fail|fatal)\b", re.IGNORECASE),
        re.compile(r"\b(warn|warning)\b", re.IGNORECASE),
        re.compile(r"\b(todo|fixme|hack|xxx)\b", re.IGNORECASE),
    ]

    def __init__(self, config: SearchCompressorConfig | None = None):
        """Initialize search compressor.

        Args:
            config: Compression configuration.
        """
        self.config = config or SearchCompressorConfig()

    def compress(
        self,
        content: str,
        context: str = "",
        bias: float = 1.0,
    ) -> SearchCompressionResult:
        """Compress search results.

        Args:
            content: Raw grep/ripgrep output.
            context: User query context for relevance scoring.
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).

        Returns:
            SearchCompressionResult with compressed output and metadata.
        """
        # Parse search results
        file_matches = self._parse_search_results(content)

        if not file_matches:
            return SearchCompressionResult(
                compressed=content,
                original=content,
                original_match_count=0,
                compressed_match_count=0,
                files_affected=0,
                compression_ratio=1.0,
            )

        # Count original matches
        original_count = sum(len(fm.matches) for fm in file_matches.values())

        # Score matches by relevance
        self._score_matches(file_matches, context)

        # Select top matches per file (with adaptive sizing)
        selected = self._select_matches(file_matches, bias=bias)

        # Format compressed output
        compressed, summaries = self._format_output(selected, file_matches)

        # Count compressed matches
        compressed_count = sum(len(fm.matches) for fm in selected.values())

        # Calculate compression ratio
        ratio = len(compressed) / max(len(content), 1)

        # Store in CCR if significant compression
        cache_key = None
        if (
            self.config.enable_ccr
            and original_count >= self.config.min_matches_for_ccr
            and ratio < 0.8
        ):
            cache_key = self._store_in_ccr(content, compressed, original_count)
            if cache_key:
                # Use consistent CCR marker format for CCRToolInjector detection
                compressed += f"\n[{original_count} matches compressed to {compressed_count}. Retrieve more: hash={cache_key}]"

        return SearchCompressionResult(
            compressed=compressed,
            original=content,
            original_match_count=original_count,
            compressed_match_count=compressed_count,
            files_affected=len(file_matches),
            compression_ratio=ratio,
            cache_key=cache_key,
            summaries=summaries,
        )

    def _parse_search_results(self, content: str) -> dict[str, FileMatches]:
        """Parse grep-style output into structured data."""
        file_matches: dict[str, FileMatches] = {}

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Try grep pattern first
            match = self._GREP_PATTERN.match(line)
            if not match:
                match = self._RG_CONTEXT_PATTERN.match(line)

            if match:
                file_path, line_num, match_content = match.groups()

                if file_path not in file_matches:
                    file_matches[file_path] = FileMatches(file=file_path)

                file_matches[file_path].matches.append(
                    SearchMatch(
                        file=file_path,
                        line_number=int(line_num),
                        content=match_content,
                    )
                )

        return file_matches

    def _score_matches(
        self,
        file_matches: dict[str, FileMatches],
        context: str,
    ) -> None:
        """Score matches by relevance to context."""
        context_lower = context.lower()
        context_words = set(context_lower.split())

        for fm in file_matches.values():
            for match in fm.matches:
                score = 0.0
                content_lower = match.content.lower()

                # Score by context word overlap
                for word in context_words:
                    if len(word) > 2 and word in content_lower:
                        score += 0.3

                # Boost error/warning patterns
                if self.config.boost_errors:
                    for i, pattern in enumerate(self._PRIORITY_PATTERNS):
                        if pattern.search(match.content):
                            score += 0.5 - (i * 0.1)  # Higher boost for errors

                # Boost for keyword matches
                for keyword in self.config.context_keywords:
                    if keyword.lower() in content_lower:
                        score += 0.4

                match.score = min(1.0, score)

    def _select_matches(
        self,
        file_matches: dict[str, FileMatches],
        bias: float = 1.0,
    ) -> dict[str, FileMatches]:
        """Select top matches per file and globally using adaptive sizing."""
        from .adaptive_sizer import compute_optimal_k

        selected: dict[str, FileMatches] = {}

        # Sort files by total match score (highest first)
        sorted_files = sorted(
            file_matches.items(),
            key=lambda x: sum(m.score for m in x[1].matches),
            reverse=True,
        )

        # Limit number of files
        sorted_files = sorted_files[: self.config.max_files]

        # Compute adaptive total matches limit
        all_match_strings = [
            f"{file_path}:{m.line_number}:{m.content}"
            for file_path, fm in sorted_files
            for m in fm.matches
        ]
        adaptive_total = compute_optimal_k(
            all_match_strings,
            bias=bias,
            min_k=5,
            max_k=self.config.max_total_matches,
        )

        total_selected = 0
        for file_path, fm in sorted_files:
            if total_selected >= adaptive_total:
                break

            # Sort matches by score
            sorted_matches = sorted(fm.matches, key=lambda m: m.score, reverse=True)

            # Select matches for this file
            file_selected: list[SearchMatch] = []
            remaining_slots = min(
                self.config.max_matches_per_file,
                adaptive_total - total_selected,
            )

            # Always include first and last if configured
            if self.config.always_keep_first and fm.first:
                file_selected.append(fm.first)
                remaining_slots -= 1

            if (
                self.config.always_keep_last
                and fm.last
                and fm.last != fm.first
                and remaining_slots > 0
            ):
                file_selected.append(fm.last)
                remaining_slots -= 1

            # Fill remaining slots with highest-scoring matches
            for match in sorted_matches:
                if remaining_slots <= 0:
                    break
                if match not in file_selected:
                    file_selected.append(match)
                    remaining_slots -= 1

            # Sort by line number for output
            file_selected.sort(key=lambda m: m.line_number)

            selected[file_path] = FileMatches(file=file_path, matches=file_selected)
            total_selected += len(file_selected)

        return selected

    def _format_output(
        self,
        selected: dict[str, FileMatches],
        original: dict[str, FileMatches],
    ) -> tuple[str, dict[str, str]]:
        """Format selected matches back to grep-style output."""
        lines: list[str] = []
        summaries: dict[str, str] = {}

        for file_path, fm in sorted(selected.items()):
            for match in fm.matches:
                lines.append(f"{match.file}:{match.line_number}:{match.content}")

            # Add summary if matches were omitted
            original_fm = original.get(file_path)
            if original_fm and len(original_fm.matches) > len(fm.matches):
                omitted = len(original_fm.matches) - len(fm.matches)
                summary = f"[... and {omitted} more matches in {file_path}]"
                lines.append(summary)
                summaries[file_path] = summary

        return "\n".join(lines), summaries

    def _store_in_ccr(
        self,
        original: str,
        compressed: str,
        original_count: int,
    ) -> str | None:
        """Store original in CCR for later retrieval."""
        try:
            from ..cache.compression_store import get_compression_store

            store = get_compression_store()
            return store.store(
                original,
                compressed,
                original_item_count=original_count,
            )
        except ImportError:
            # CCR not available
            return None
        except Exception:
            # Silently fail CCR storage
            return None


@dataclass
class SearchCompressionResult:
    """Result of search result compression."""

    compressed: str
    original: str
    original_match_count: int
    compressed_match_count: int
    files_affected: int
    compression_ratio: float
    cache_key: str | None = None
    summaries: dict[str, str] = field(default_factory=dict)

    @property
    def tokens_saved_estimate(self) -> int:
        """Estimate tokens saved (rough: 1 token per 4 chars)."""
        chars_saved = len(self.original) - len(self.compressed)
        return max(0, chars_saved // 4)

    @property
    def matches_omitted(self) -> int:
        """Number of matches omitted."""
        return self.original_match_count - self.compressed_match_count
