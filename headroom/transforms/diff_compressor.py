"""Git diff output compressor for unified diff format.

This module compresses git diff output which can be very verbose with
many context lines. Typical compression: 3-10x.

Supported formats:
- Unified diff format (git diff, diff -u)
- Combined diff format (merge conflicts)

Compression Strategy:
1. Parse unified diff format into file sections and hunks
2. Always keep file headers (diff --git, ---, +++)
3. Always keep ALL actual changes (+/- lines)
4. Reduce context lines (` ` prefix) to configurable max
5. If too many hunks, keep first N and summarize rest
6. Add summary at end

Key Patterns to Preserve:
- All additions (+)
- All deletions (-)
- Hunk headers (@@ ... @@)
- File headers (diff --git, ---, +++)
- Context around changes (limited)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DiffHunk:
    """A single hunk within a diff file."""

    header: str  # @@ -start,count +start,count @@ optional function
    lines: list[str]  # All lines in the hunk
    additions: int = 0
    deletions: int = 0
    context_lines: int = 0
    score: float = 0.0  # Relevance score for context-aware compression

    @property
    def change_count(self) -> int:
        """Total number of actual changes (additions + deletions)."""
        return self.additions + self.deletions


@dataclass
class DiffFile:
    """A single file's diff."""

    header: str  # diff --git a/... b/...
    old_file: str  # --- a/...
    new_file: str  # +++ b/...
    hunks: list[DiffHunk] = field(default_factory=list)
    is_binary: bool = False
    is_new_file: bool = False
    is_deleted_file: bool = False
    is_renamed: bool = False

    @property
    def total_additions(self) -> int:
        return sum(h.additions for h in self.hunks)

    @property
    def total_deletions(self) -> int:
        return sum(h.deletions for h in self.hunks)


@dataclass
class DiffCompressorConfig:
    """Configuration for diff compression."""

    # Context line limits
    max_context_lines: int = 2  # Reduce from default 3 lines before/after changes

    # Hunk limits
    max_hunks_per_file: int = 10

    # File limits
    max_files: int = 20

    # Change preservation
    always_keep_additions: bool = True  # Always keep + lines
    always_keep_deletions: bool = True  # Always keep - lines

    # CCR integration
    enable_ccr: bool = True
    min_lines_for_ccr: int = 50


@dataclass
class DiffCompressionResult:
    """Result of diff compression."""

    compressed: str
    original_line_count: int
    compressed_line_count: int
    files_affected: int
    additions: int
    deletions: int
    hunks_kept: int
    hunks_removed: int
    cache_key: str | None = None

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed to original (lower is better compression)."""
        if self.original_line_count == 0:
            return 1.0
        return self.compressed_line_count / self.original_line_count

    @property
    def tokens_saved_estimate(self) -> int:
        """Estimate tokens saved (rough: 1 token per 4 chars)."""
        # Use line counts as proxy for chars
        lines_saved = self.original_line_count - self.compressed_line_count
        # Estimate ~40 chars per line average for diffs
        chars_saved = lines_saved * 40
        return max(0, chars_saved // 4)


class DiffCompressor:
    """Compresses git diff output.

    Example:
        >>> compressor = DiffCompressor()
        >>> result = compressor.compress(git_diff_output)
        >>> print(result.compressed)  # Reduced diff with summary
    """

    # Pattern for diff --git header
    _DIFF_GIT_PATTERN = re.compile(r"^diff --git a/(.+) b/(.+)$")

    # Pattern for --- a/file or --- /dev/null
    _OLD_FILE_PATTERN = re.compile(r"^--- (a/(.+)|/dev/null)$")

    # Pattern for +++ b/file or +++ /dev/null
    _NEW_FILE_PATTERN = re.compile(r"^\+\+\+ (b/(.+)|/dev/null)$")

    # Pattern for hunk header @@ -start,count +start,count @@ optional context
    _HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")

    # Pattern for binary file indication
    _BINARY_PATTERN = re.compile(r"^Binary files .+ differ$")

    # Patterns for new/deleted file mode
    _NEW_FILE_MODE_PATTERN = re.compile(r"^new file mode")
    _DELETED_FILE_MODE_PATTERN = re.compile(r"^deleted file mode")
    _RENAME_PATTERN = re.compile(r"^(rename|similarity|copy) ")

    # Priority patterns for context-aware hunk selection
    _PRIORITY_PATTERNS = [
        re.compile(r"\b(error|exception|fail|bug|fix)\b", re.IGNORECASE),
        re.compile(r"\b(todo|fixme|hack|xxx)\b", re.IGNORECASE),
        re.compile(r"\b(security|auth|password|secret|token)\b", re.IGNORECASE),
    ]

    def __init__(self, config: DiffCompressorConfig | None = None):
        """Initialize diff compressor.

        Args:
            config: Compression configuration.
        """
        self.config = config or DiffCompressorConfig()

    def compress(self, content: str, context: str = "") -> DiffCompressionResult:
        """Compress diff output.

        Args:
            content: Raw git diff output.
            context: User query context for relevance scoring.

        Returns:
            DiffCompressionResult with compressed output and metadata.
        """
        lines = content.split("\n")
        original_line_count = len(lines)

        if original_line_count < self.config.min_lines_for_ccr:
            return DiffCompressionResult(
                compressed=content,
                original_line_count=original_line_count,
                compressed_line_count=original_line_count,
                files_affected=0,
                additions=0,
                deletions=0,
                hunks_kept=0,
                hunks_removed=0,
            )

        # Parse diff into structured format
        diff_files = self._parse_diff(lines)

        if not diff_files:
            return DiffCompressionResult(
                compressed=content,
                original_line_count=original_line_count,
                compressed_line_count=original_line_count,
                files_affected=0,
                additions=0,
                deletions=0,
                hunks_kept=0,
                hunks_removed=0,
            )

        # Score hunks by relevance
        self._score_hunks(diff_files, context)

        # Compress each file's hunks
        compressed_files, stats = self._compress_files(diff_files)

        # Format output
        compressed_output = self._format_output(compressed_files, stats)
        compressed_line_count = len(compressed_output.split("\n"))

        # Store in CCR if significant compression
        cache_key = None
        if self.config.enable_ccr and compressed_line_count < original_line_count * 0.8:
            cache_key = self._store_in_ccr(content, compressed_output, original_line_count)
            if cache_key:
                compressed_output += f"\n[{original_line_count} lines compressed to {compressed_line_count}. Retrieve full diff: hash={cache_key}]"

        return DiffCompressionResult(
            compressed=compressed_output,
            original_line_count=original_line_count,
            compressed_line_count=compressed_line_count,
            files_affected=stats["files_affected"],
            additions=stats["total_additions"],
            deletions=stats["total_deletions"],
            hunks_kept=stats["hunks_kept"],
            hunks_removed=stats["hunks_removed"],
            cache_key=cache_key,
        )

    def _parse_diff(self, lines: list[str]) -> list[DiffFile]:
        """Parse diff content into structured format.

        Args:
            lines: Lines of diff content.

        Returns:
            List of DiffFile objects.
        """
        diff_files: list[DiffFile] = []
        current_file: DiffFile | None = None
        current_hunk: DiffHunk | None = None
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for diff --git header (new file section)
            if self._DIFF_GIT_PATTERN.match(line):
                # Save previous hunk and file
                if current_hunk and current_file:
                    current_file.hunks.append(current_hunk)
                if current_file:
                    diff_files.append(current_file)

                current_file = DiffFile(
                    header=line,
                    old_file="",
                    new_file="",
                )
                current_hunk = None
                i += 1
                continue

            # Check for file mode indicators
            if current_file:
                if self._NEW_FILE_MODE_PATTERN.match(line):
                    current_file.is_new_file = True
                elif self._DELETED_FILE_MODE_PATTERN.match(line):
                    current_file.is_deleted_file = True
                elif self._RENAME_PATTERN.match(line):
                    current_file.is_renamed = True
                elif self._BINARY_PATTERN.match(line):
                    current_file.is_binary = True

            # Check for --- a/file
            if self._OLD_FILE_PATTERN.match(line):
                if current_file:
                    current_file.old_file = line
                i += 1
                continue

            # Check for +++ b/file
            if self._NEW_FILE_PATTERN.match(line):
                if current_file:
                    current_file.new_file = line
                i += 1
                continue

            # Check for hunk header
            if self._HUNK_HEADER_PATTERN.match(line):
                # Save previous hunk
                if current_hunk and current_file:
                    current_file.hunks.append(current_hunk)

                current_hunk = DiffHunk(
                    header=line,
                    lines=[],
                )
                i += 1
                continue

            # Process hunk content lines
            if current_hunk is not None:
                if line.startswith("+") and not line.startswith("+++"):
                    current_hunk.additions += 1
                    current_hunk.lines.append(line)
                elif line.startswith("-") and not line.startswith("---"):
                    current_hunk.deletions += 1
                    current_hunk.lines.append(line)
                elif line.startswith(" ") or line == "":
                    current_hunk.context_lines += 1
                    current_hunk.lines.append(line)
                else:
                    # Other line (e.g., "\ No newline at end of file")
                    current_hunk.lines.append(line)

            i += 1

        # Save final hunk and file
        if current_hunk and current_file:
            current_file.hunks.append(current_hunk)
        if current_file:
            diff_files.append(current_file)

        return diff_files

    def _score_hunks(self, diff_files: list[DiffFile], context: str) -> None:
        """Score hunks by relevance to context.

        Args:
            diff_files: Parsed diff files.
            context: User query context.
        """
        context_lower = context.lower()
        context_words = set(context_lower.split()) if context else set()

        for diff_file in diff_files:
            for hunk in diff_file.hunks:
                score = 0.0

                # Base score from change count (more changes = more important)
                score += min(0.3, hunk.change_count * 0.03)

                hunk_content = "\n".join(hunk.lines).lower()

                # Score by context word overlap
                for word in context_words:
                    if len(word) > 2 and word in hunk_content:
                        score += 0.2

                # Boost for priority patterns
                for pattern in self._PRIORITY_PATTERNS:
                    if pattern.search(hunk_content):
                        score += 0.3
                        break

                hunk.score = min(1.0, score)

    def _compress_files(self, diff_files: list[DiffFile]) -> tuple[list[DiffFile], dict[str, int]]:
        """Compress hunks in each file.

        Args:
            diff_files: Parsed diff files.

        Returns:
            Tuple of (compressed files, stats dict).
        """
        stats = {
            "files_affected": 0,
            "total_additions": 0,
            "total_deletions": 0,
            "hunks_kept": 0,
            "hunks_removed": 0,
        }

        # Limit files if too many
        if len(diff_files) > self.config.max_files:
            # Sort by total changes (most changes first)
            diff_files = sorted(
                diff_files,
                key=lambda f: f.total_additions + f.total_deletions,
                reverse=True,
            )
            diff_files = diff_files[: self.config.max_files]

        compressed_files: list[DiffFile] = []

        for diff_file in diff_files:
            stats["files_affected"] += 1
            stats["total_additions"] += diff_file.total_additions
            stats["total_deletions"] += diff_file.total_deletions

            # Compress hunks within file
            compressed_hunks = self._compress_hunks(diff_file.hunks)

            stats["hunks_kept"] += len(compressed_hunks)
            stats["hunks_removed"] += len(diff_file.hunks) - len(compressed_hunks)

            # Create compressed file with reduced context in hunks
            new_file = DiffFile(
                header=diff_file.header,
                old_file=diff_file.old_file,
                new_file=diff_file.new_file,
                hunks=compressed_hunks,
                is_binary=diff_file.is_binary,
                is_new_file=diff_file.is_new_file,
                is_deleted_file=diff_file.is_deleted_file,
                is_renamed=diff_file.is_renamed,
            )
            compressed_files.append(new_file)

        return compressed_files, stats

    def _compress_hunks(self, hunks: list[DiffHunk]) -> list[DiffHunk]:
        """Compress hunks by reducing context and limiting count.

        Args:
            hunks: List of hunks to compress.

        Returns:
            Compressed list of hunks.
        """
        if not hunks:
            return []

        # Sort by score if we need to limit
        if len(hunks) > self.config.max_hunks_per_file:
            # Keep first and last hunks (often important)
            first_hunk = hunks[0]
            last_hunk = hunks[-1] if len(hunks) > 1 else None

            # Sort middle hunks by score
            middle_hunks = sorted(
                hunks[1:-1] if last_hunk else [], key=lambda h: h.score, reverse=True
            )

            # Take top scoring middle hunks
            remaining_slots = (
                self.config.max_hunks_per_file - 2
                if last_hunk
                else self.config.max_hunks_per_file - 1
            )
            selected_middle = middle_hunks[:remaining_slots]

            # Rebuild list in original order by re-sorting by appearance
            selected = [first_hunk] + selected_middle
            if last_hunk:
                selected.append(last_hunk)

            # Sort back to original order (using header line numbers as proxy)
            hunks = sorted(selected, key=lambda h: self._extract_line_number(h.header))

        # Reduce context in each hunk
        compressed_hunks = []
        for hunk in hunks:
            compressed_hunk = self._reduce_context(hunk)
            compressed_hunks.append(compressed_hunk)

        return compressed_hunks

    def _extract_line_number(self, header: str) -> int:
        """Extract starting line number from hunk header for sorting."""
        match = self._HUNK_HEADER_PATTERN.match(header)
        if match:
            return int(match.group(1))
        return 0

    def _reduce_context(self, hunk: DiffHunk) -> DiffHunk:
        """Reduce context lines while preserving all changes.

        Args:
            hunk: Hunk to reduce context in.

        Returns:
            New hunk with reduced context.
        """
        max_context = self.config.max_context_lines

        # Identify change positions
        change_positions: list[int] = []
        for i, line in enumerate(hunk.lines):
            if line.startswith("+") or line.startswith("-"):
                change_positions.append(i)

        if not change_positions:
            # No changes, just context - keep minimal
            return DiffHunk(
                header=hunk.header,
                lines=hunk.lines[:max_context] if hunk.lines else [],
                additions=0,
                deletions=0,
                context_lines=min(len(hunk.lines), max_context),
                score=hunk.score,
            )

        # Determine which lines to keep
        keep_indices: set[int] = set()

        for pos in change_positions:
            # Always keep the change line
            keep_indices.add(pos)

            # Keep context before
            for i in range(max(0, pos - max_context), pos):
                keep_indices.add(i)

            # Keep context after
            for i in range(pos + 1, min(len(hunk.lines), pos + max_context + 1)):
                keep_indices.add(i)

        # Build new lines list
        new_lines: list[str] = []
        additions = 0
        deletions = 0
        context_lines = 0

        for i in sorted(keep_indices):
            line = hunk.lines[i]
            new_lines.append(line)
            if line.startswith("+"):
                additions += 1
            elif line.startswith("-"):
                deletions += 1
            else:
                context_lines += 1

        return DiffHunk(
            header=hunk.header,
            lines=new_lines,
            additions=additions,
            deletions=deletions,
            context_lines=context_lines,
            score=hunk.score,
        )

    def _format_output(self, diff_files: list[DiffFile], stats: dict[str, int]) -> str:
        """Format compressed diff files back to unified diff format.

        Args:
            diff_files: Compressed diff files.
            stats: Compression statistics.

        Returns:
            Formatted diff string.
        """
        output_lines: list[str] = []

        for diff_file in diff_files:
            # File header
            output_lines.append(diff_file.header)

            # File mode indicators if present
            if diff_file.is_new_file:
                output_lines.append("new file mode 100644")
            elif diff_file.is_deleted_file:
                output_lines.append("deleted file mode 100644")

            if diff_file.is_binary:
                output_lines.append("Binary files differ")
                continue

            # Old/new file markers
            if diff_file.old_file:
                output_lines.append(diff_file.old_file)
            if diff_file.new_file:
                output_lines.append(diff_file.new_file)

            # Hunks
            for hunk in diff_file.hunks:
                output_lines.append(hunk.header)
                output_lines.extend(hunk.lines)

        # Add summary
        if stats["hunks_removed"] > 0 or stats["files_affected"] > 0:
            summary_parts = [
                f"{stats['files_affected']} files changed",
                f"+{stats['total_additions']} -{stats['total_deletions']} lines",
            ]
            if stats["hunks_removed"] > 0:
                summary_parts.append(f"{stats['hunks_removed']} hunks omitted")
            output_lines.append(f"[{', '.join(summary_parts)}]")

        return "\n".join(output_lines)

    def _store_in_ccr(self, original: str, compressed: str, original_count: int) -> str | None:
        """Store original in CCR for later retrieval.

        Args:
            original: Original diff content.
            compressed: Compressed diff content.
            original_count: Original line count.

        Returns:
            Cache key if stored, None otherwise.
        """
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
