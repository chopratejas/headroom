"""Log/build output compressor for test and compiler output.

This module compresses build and test output which can be 10,000+ lines
with only 5-10 actual errors. Typical compression: 10-50x.

Supported formats:
- pytest output
- npm/yarn output
- cargo/rustc output
- make/gcc output
- generic log format (ERROR, WARN, INFO)

Compression Strategy:
1. Detect log format (pytest, npm, cargo, etc.)
2. Extract all ERROR/FAIL lines with context
3. Extract first stack trace completely
4. Deduplicate repeated warnings
5. Summarize: [247 INFO lines, 12 WARN lines omitted]

Key Patterns to Preserve:
- First error (often root cause)
- Last error (sometimes the real failure)
- Stack traces
- Exit codes
- Test summary lines
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class LogFormat(Enum):
    """Detected log format."""

    PYTEST = "pytest"
    NPM = "npm"
    CARGO = "cargo"
    MAKE = "make"
    JEST = "jest"
    GENERIC = "generic"


class LogLevel(Enum):
    """Log level for categorization."""

    ERROR = "error"
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"
    DEBUG = "debug"
    TRACE = "trace"
    UNKNOWN = "unknown"


@dataclass(eq=False)
class LogLine:
    """A single log line with metadata."""

    line_number: int
    content: str
    level: LogLevel = LogLevel.UNKNOWN
    is_stack_trace: bool = False
    is_summary: bool = False
    score: float = 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogLine):
            return NotImplemented
        return self.line_number == other.line_number

    def __hash__(self) -> int:
        return hash(self.line_number)


@dataclass
class LogCompressorConfig:
    """Configuration for log compression."""

    # Error handling
    max_errors: int = 10
    error_context_lines: int = 3
    keep_first_error: bool = True
    keep_last_error: bool = True

    # Stack trace handling
    max_stack_traces: int = 3
    stack_trace_max_lines: int = 20

    # Warning handling
    max_warnings: int = 5
    dedupe_warnings: bool = True

    # Summary handling
    keep_summary_lines: bool = True

    # Global limits
    max_total_lines: int = 100

    # CCR integration
    enable_ccr: bool = True
    min_lines_for_ccr: int = 50


class LogCompressor:
    """Compresses build/test log output.

    Example:
        >>> compressor = LogCompressor()
        >>> result = compressor.compress(pytest_output)
        >>> print(result.compressed)  # Just errors + summary
    """

    # Format detection patterns
    _FORMAT_PATTERNS = {
        LogFormat.PYTEST: [
            re.compile(r"^={3,} (FAILURES|ERRORS|test session|short test summary)"),
            re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+\["),
            re.compile(r"^collected \d+ items?"),
        ],
        LogFormat.NPM: [
            re.compile(r"^npm (ERR!|WARN|info|http)"),
            re.compile(r"^(>|added|removed) .+ packages?"),
        ],
        LogFormat.CARGO: [
            re.compile(r"^\s*(Compiling|Finished|Running|error\[E\d+\])"),
            re.compile(r"^warning: .+"),
        ],
        LogFormat.JEST: [
            re.compile(r"^(PASS|FAIL)\s+.+\.test\.(js|ts)"),
            re.compile(r"^Test Suites:"),
        ],
        LogFormat.MAKE: [
            re.compile(r"^make(\[\d+\])?: "),
            re.compile(r"^(gcc|g\+\+|clang).*-o "),
        ],
    }

    # Level detection patterns
    _LEVEL_PATTERNS = {
        LogLevel.ERROR: re.compile(r"\b(ERROR|error|Error|FATAL|fatal|Fatal|CRITICAL|critical)\b"),
        LogLevel.FAIL: re.compile(r"\b(FAIL|FAILED|fail|failed|Fail|Failed)\b"),
        LogLevel.WARN: re.compile(r"\b(WARN|WARNING|warn|warning|Warn|Warning)\b"),
        LogLevel.INFO: re.compile(r"\b(INFO|info|Info)\b"),
        LogLevel.DEBUG: re.compile(r"\b(DEBUG|debug|Debug)\b"),
        LogLevel.TRACE: re.compile(r"\b(TRACE|trace|Trace)\b"),
    }

    # Stack trace patterns
    _STACK_TRACE_PATTERNS = [
        re.compile(r"^\s*Traceback \(most recent call last\)"),
        re.compile(r'^\s*File ".+", line \d+'),
        re.compile(r"^\s*at .+\(.+:\d+:\d+\)"),  # JS stack trace
        re.compile(r"^\s+at [\w.$]+\("),  # Java stack trace
        re.compile(r"^\s*--> .+:\d+:\d+"),  # Rust error
        re.compile(r"^\s*\d+:\s+0x[0-9a-f]+"),  # Go stack trace
    ]

    # Summary line patterns
    _SUMMARY_PATTERNS = [
        re.compile(r"^={3,}"),  # pytest separators
        re.compile(r"^-{3,}"),
        re.compile(r"^\d+ (passed|failed|skipped|error|warning)"),
        re.compile(r"^(Tests?|Suites?):?\s+\d+"),
        re.compile(r"^(TOTAL|Total|Summary)"),
        re.compile(r"^(Build|Compile|Test).*(succeeded|failed|complete)"),
    ]

    def __init__(self, config: LogCompressorConfig | None = None):
        """Initialize log compressor.

        Args:
            config: Compression configuration.
        """
        self.config = config or LogCompressorConfig()

    def compress(self, content: str, context: str = "", bias: float = 1.0) -> LogCompressionResult:
        """Compress log output.

        Args:
            content: Raw log output.
            context: User query context (unused for now).
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).

        Returns:
            LogCompressionResult with compressed output and metadata.
        """
        lines = content.split("\n")

        if len(lines) < self.config.min_lines_for_ccr:
            return LogCompressionResult(
                compressed=content,
                original=content,
                original_line_count=len(lines),
                compressed_line_count=len(lines),
                format_detected=LogFormat.GENERIC,
                compression_ratio=1.0,
            )

        # Detect format
        log_format = self._detect_format(lines)

        # Parse and categorize lines
        log_lines = self._parse_lines(lines)

        # Select important lines (with adaptive sizing)
        selected = self._select_lines(log_lines, bias=bias)

        # Format output with summaries
        compressed, stats = self._format_output(selected, log_lines)

        ratio = len(compressed) / max(len(content), 1)

        # Store in CCR if significant compression
        cache_key = None
        if self.config.enable_ccr and ratio < 0.5:
            cache_key = self._store_in_ccr(content, compressed, len(lines))
            if cache_key:
                # Use consistent CCR marker format for CCRToolInjector detection
                compressed += f"\n[{len(lines)} lines compressed to {len(selected)}. Retrieve more: hash={cache_key}]"

        return LogCompressionResult(
            compressed=compressed,
            original=content,
            original_line_count=len(lines),
            compressed_line_count=len(selected),
            format_detected=log_format,
            compression_ratio=ratio,
            cache_key=cache_key,
            stats=stats,
        )

    def _detect_format(self, lines: list[str]) -> LogFormat:
        """Detect the log format."""
        sample = lines[:100]  # Check first 100 lines

        format_scores: dict[LogFormat, int] = {}
        for log_format, patterns in self._FORMAT_PATTERNS.items():
            score = 0
            for line in sample:
                for pattern in patterns:
                    if pattern.search(line):
                        score += 1
                        break
            if score > 0:
                format_scores[log_format] = score

        if not format_scores:
            return LogFormat.GENERIC

        return max(format_scores, key=lambda k: format_scores[k])

    def _parse_lines(self, lines: list[str]) -> list[LogLine]:
        """Parse lines and categorize by level."""
        log_lines: list[LogLine] = []
        in_stack_trace = False
        stack_trace_lines = 0

        for i, line in enumerate(lines):
            log_line = LogLine(line_number=i, content=line)

            # Detect level
            for level, pattern in self._LEVEL_PATTERNS.items():
                if pattern.search(line):
                    log_line.level = level
                    break

            # Detect stack trace
            for pattern in self._STACK_TRACE_PATTERNS:
                if pattern.search(line):
                    in_stack_trace = True
                    stack_trace_lines = 0
                    break

            if in_stack_trace:
                log_line.is_stack_trace = True
                stack_trace_lines += 1
                # End stack trace after max lines or empty line
                if stack_trace_lines > self.config.stack_trace_max_lines or not line.strip():
                    in_stack_trace = False

            # Detect summary lines
            for pattern in self._SUMMARY_PATTERNS:
                if pattern.search(line):
                    log_line.is_summary = True
                    break

            # Score line by importance
            log_line.score = self._score_line(log_line)

            log_lines.append(log_line)

        return log_lines

    def _score_line(self, log_line: LogLine) -> float:
        """Score a line by importance."""
        score = 0.0

        # Level-based scoring
        level_scores = {
            LogLevel.ERROR: 1.0,
            LogLevel.FAIL: 1.0,
            LogLevel.WARN: 0.5,
            LogLevel.INFO: 0.1,
            LogLevel.DEBUG: 0.05,
            LogLevel.TRACE: 0.02,
            LogLevel.UNKNOWN: 0.1,
        }
        score += level_scores.get(log_line.level, 0.1)

        # Boost stack traces
        if log_line.is_stack_trace:
            score += 0.3

        # Boost summary lines
        if log_line.is_summary:
            score += 0.4

        return min(1.0, score)

    def _select_lines(self, log_lines: list[LogLine], bias: float = 1.0) -> list[LogLine]:
        """Select important lines to keep using adaptive sizing."""
        from .adaptive_sizer import compute_optimal_k

        selected: list[LogLine] = []

        # Compute adaptive line budget from the FULL log output
        # This determines "how many lines of value exist" across the entire log
        all_line_strings = [line.content for line in log_lines]
        adaptive_max = compute_optimal_k(
            all_line_strings,
            bias=bias,
            min_k=10,
            max_k=self.config.max_total_lines,
        )

        # Group by category
        errors: list[LogLine] = []
        fails: list[LogLine] = []
        warnings: list[LogLine] = []
        stack_traces: list[list[LogLine]] = []
        summaries: list[LogLine] = []
        current_stack: list[LogLine] = []

        for log_line in log_lines:
            if log_line.level == LogLevel.ERROR:
                errors.append(log_line)
            elif log_line.level == LogLevel.FAIL:
                fails.append(log_line)
            elif log_line.level == LogLevel.WARN:
                warnings.append(log_line)

            if log_line.is_stack_trace:
                current_stack.append(log_line)
            elif current_stack:
                stack_traces.append(current_stack)
                current_stack = []

            if log_line.is_summary:
                summaries.append(log_line)

        if current_stack:
            stack_traces.append(current_stack)

        # Select errors (first, last, highest scoring)
        if errors:
            selected_errors = self._select_with_first_last(errors, self.config.max_errors)
            selected.extend(selected_errors)

        # Select fails
        if fails:
            selected_fails = self._select_with_first_last(fails, self.config.max_errors)
            selected.extend(selected_fails)

        # Select warnings (dedupe if configured)
        if warnings:
            if self.config.dedupe_warnings:
                warnings = self._dedupe_similar(warnings)
            selected.extend(warnings[: self.config.max_warnings])

        # Select stack traces
        for stack in stack_traces[: self.config.max_stack_traces]:
            selected.extend(stack[: self.config.stack_trace_max_lines])

        # Always include summary lines
        if self.config.keep_summary_lines:
            selected.extend(summaries)

        # Add context lines around errors
        selected = self._add_context(log_lines, selected)

        # Sort by line number and dedupe
        selected = sorted(set(selected), key=lambda x: x.line_number)

        # Apply adaptive line limit
        if len(selected) > adaptive_max:
            # Keep most important lines
            selected = sorted(selected, key=lambda x: x.score, reverse=True)
            selected = selected[:adaptive_max]
            selected = sorted(selected, key=lambda x: x.line_number)

        return selected

    def _select_with_first_last(self, lines: list[LogLine], max_count: int) -> list[LogLine]:
        """Select lines keeping first and last."""
        if len(lines) <= max_count:
            return lines

        selected: list[LogLine] = []

        if self.config.keep_first_error and lines:
            selected.append(lines[0])

        if self.config.keep_last_error and lines and lines[-1] not in selected:
            selected.append(lines[-1])

        # Fill remaining with highest scoring
        remaining = max_count - len(selected)
        if remaining > 0:
            candidates = [line for line in lines if line not in selected]
            candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
            selected.extend(candidates[:remaining])

        return selected

    def _dedupe_similar(self, lines: list[LogLine]) -> list[LogLine]:
        """Remove duplicate/similar lines."""
        seen_patterns: set[str] = set()
        deduped: list[LogLine] = []

        for line in lines:
            # Normalize: remove numbers, paths for comparison
            normalized = re.sub(r"\d+", "N", line.content)
            normalized = re.sub(r"/[\w/]+/", "/PATH/", normalized)
            normalized = re.sub(r"0x[0-9a-f]+", "ADDR", normalized)

            if normalized not in seen_patterns:
                seen_patterns.add(normalized)
                deduped.append(line)

        return deduped

    def _add_context(self, all_lines: list[LogLine], selected: list[LogLine]) -> list[LogLine]:
        """Add context lines around selected lines."""
        selected_indices = {line.line_number for line in selected}
        context_indices: set[int] = set()

        for idx in selected_indices:
            # Add lines before
            for i in range(max(0, idx - self.config.error_context_lines), idx):
                context_indices.add(i)
            # Add lines after
            for i in range(
                idx + 1,
                min(len(all_lines), idx + self.config.error_context_lines + 1),
            ):
                context_indices.add(i)

        # Add context lines to selected
        for idx in context_indices:
            if idx not in selected_indices and idx < len(all_lines):
                selected.append(all_lines[idx])

        return selected

    def _format_output(
        self, selected: list[LogLine], all_lines: list[LogLine]
    ) -> tuple[str, dict[str, int]]:
        """Format selected lines with summary stats."""
        # Count categories
        stats: dict[str, int] = {
            "errors": sum(1 for line in all_lines if line.level == LogLevel.ERROR),
            "fails": sum(1 for line in all_lines if line.level == LogLevel.FAIL),
            "warnings": sum(1 for line in all_lines if line.level == LogLevel.WARN),
            "info": sum(1 for line in all_lines if line.level == LogLevel.INFO),
            "total": len(all_lines),
            "selected": len(selected),
        }

        # Build output
        output_lines = [line.content for line in selected]

        # Add summary of omitted lines
        omitted = len(all_lines) - len(selected)
        if omitted > 0:
            summary_parts = []
            for level_name, count in [
                ("ERROR", stats["errors"]),
                ("FAIL", stats["fails"]),
                ("WARN", stats["warnings"]),
                ("INFO", stats["info"]),
            ]:
                if count > 0:
                    summary_parts.append(f"{count} {level_name}")

            if summary_parts:
                summary = f"[{omitted} lines omitted: {', '.join(summary_parts)}]"
                output_lines.append(summary)

        return "\n".join(output_lines), stats

    def _store_in_ccr(self, original: str, compressed: str, original_count: int) -> str | None:
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
            return None
        except Exception:
            return None


@dataclass
class LogCompressionResult:
    """Result of log compression."""

    compressed: str
    original: str
    original_line_count: int
    compressed_line_count: int
    format_detected: LogFormat
    compression_ratio: float
    cache_key: str | None = None
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def tokens_saved_estimate(self) -> int:
        """Estimate tokens saved (rough: 1 token per 4 chars)."""
        chars_saved = len(self.original) - len(self.compressed)
        return max(0, chars_saved // 4)

    @property
    def lines_omitted(self) -> int:
        """Number of lines omitted."""
        return self.original_line_count - self.compressed_line_count
