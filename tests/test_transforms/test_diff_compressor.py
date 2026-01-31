"""Comprehensive tests for diff_compressor.py.

Tests cover:
1. Parsing of unified diff format
2. Context line reduction
3. Hunk selection and limiting
4. Compression ratios
5. Edge cases
"""

from headroom.transforms.diff_compressor import (
    DiffCompressionResult,
    DiffCompressor,
    DiffCompressorConfig,
    DiffFile,
    DiffHunk,
)


class TestDiffParsing:
    """Tests for parsing unified diff format."""

    def test_parse_simple_diff(self):
        """Simple single-file diff is parsed correctly."""
        content = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -10,6 +10,7 @@ def main():
     print("hello")
+    print("world")
     return 0
"""
        compressor = DiffCompressor()
        diff_files = compressor._parse_diff(content.split("\n"))

        assert len(diff_files) == 1
        assert diff_files[0].header == "diff --git a/src/main.py b/src/main.py"
        assert diff_files[0].old_file == "--- a/src/main.py"
        assert diff_files[0].new_file == "+++ b/src/main.py"
        assert len(diff_files[0].hunks) == 1
        assert diff_files[0].hunks[0].additions == 1
        assert diff_files[0].hunks[0].deletions == 0

    def test_parse_multi_file_diff(self):
        """Multi-file diff is parsed into separate DiffFile objects."""
        content = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,4 @@
 line1
+added line
 line2
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -5,4 +5,3 @@
 keep
-removed
 keep2
"""
        compressor = DiffCompressor()
        diff_files = compressor._parse_diff(content.split("\n"))

        assert len(diff_files) == 2
        assert "file1.py" in diff_files[0].header
        assert "file2.py" in diff_files[1].header
        assert diff_files[0].hunks[0].additions == 1
        assert diff_files[0].hunks[0].deletions == 0
        assert diff_files[1].hunks[0].additions == 0
        assert diff_files[1].hunks[0].deletions == 1

    def test_parse_multi_hunk_file(self):
        """File with multiple hunks is parsed correctly."""
        content = """diff --git a/src/utils.py b/src/utils.py
--- a/src/utils.py
+++ b/src/utils.py
@@ -10,4 +10,5 @@ def helper():
     pass
+    # added comment
     return True
@@ -50,3 +51,4 @@ def other():
     x = 1
+    y = 2
     return x
"""
        compressor = DiffCompressor()
        diff_files = compressor._parse_diff(content.split("\n"))

        assert len(diff_files) == 1
        assert len(diff_files[0].hunks) == 2
        assert diff_files[0].total_additions == 2

    def test_parse_new_file(self):
        """New file diff is detected."""
        content = """diff --git a/newfile.py b/newfile.py
new file mode 100644
--- /dev/null
+++ b/newfile.py
@@ -0,0 +1,3 @@
+def new_func():
+    pass
+    return None
"""
        compressor = DiffCompressor()
        diff_files = compressor._parse_diff(content.split("\n"))

        assert len(diff_files) == 1
        assert diff_files[0].is_new_file is True
        assert diff_files[0].hunks[0].additions == 3

    def test_parse_deleted_file(self):
        """Deleted file diff is detected."""
        content = """diff --git a/oldfile.py b/oldfile.py
deleted file mode 100644
--- a/oldfile.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def old_func():
-    pass
"""
        compressor = DiffCompressor()
        diff_files = compressor._parse_diff(content.split("\n"))

        assert len(diff_files) == 1
        assert diff_files[0].is_deleted_file is True
        assert diff_files[0].hunks[0].deletions == 2

    def test_parse_binary_file(self):
        """Binary file diff is detected."""
        content = """diff --git a/image.png b/image.png
Binary files a/image.png and b/image.png differ
"""
        compressor = DiffCompressor()
        diff_files = compressor._parse_diff(content.split("\n"))

        assert len(diff_files) == 1
        assert diff_files[0].is_binary is True


class TestContextReduction:
    """Tests for context line reduction."""

    def test_reduce_context_lines(self):
        """Context lines are reduced to configured maximum."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,10 +1,11 @@
 context1
 context2
 context3
 context4
+added
 context5
 context6
 context7
 context8
"""
        # Default max_context_lines is 2
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_context_lines=2,
                min_lines_for_ccr=5,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should keep 2 context before and 2 after the +added line
        # Plus the added line itself
        lines = result.compressed.split("\n")
        context_count = sum(1 for line in lines if line.startswith(" "))

        # At most 4 context lines (2 before + 2 after)
        assert context_count <= 4

    def test_preserve_all_changes(self):
        """All addition and deletion lines are preserved."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,10 +1,10 @@
 ctx1
 ctx2
-removed1
+added1
 ctx3
 ctx4
-removed2
+added2
 ctx5
 ctx6
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                min_lines_for_ccr=5,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        assert "-removed1" in result.compressed
        assert "-removed2" in result.compressed
        assert "+added1" in result.compressed
        assert "+added2" in result.compressed


class TestHunkSelection:
    """Tests for hunk selection when limiting."""

    def test_max_hunks_per_file(self):
        """Hunks are limited to max_hunks_per_file."""
        # Create a diff with many hunks
        hunks = []
        for i in range(20):
            hunks.append(f"""@@ -{i * 10},3 +{i * 10},4 @@
 context
+added_{i}
 more
""")

        content = f"""diff --git a/bigfile.py b/bigfile.py
--- a/bigfile.py
+++ b/bigfile.py
{"".join(hunks)}"""

        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_hunks_per_file=5,
                min_lines_for_ccr=10,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should have at most 5 hunks
        hunk_count = result.compressed.count("@@")
        # Each hunk has one @@ header (we count full hunk headers)
        assert hunk_count <= 10  # Each hunk header appears twice @@...@@

    def test_keeps_first_and_last_hunk(self):
        """First and last hunks are preserved when limiting."""
        hunks = []
        for i in range(10):
            hunks.append(f"""@@ -{i * 10},3 +{i * 10},4 @@
 context
+added_{i}
 more
""")

        content = f"""diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
{"".join(hunks)}"""

        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_hunks_per_file=3,
                min_lines_for_ccr=10,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # First hunk (added_0) should be present
        assert "+added_0" in result.compressed
        # Last hunk (added_9) should be present
        assert "+added_9" in result.compressed


class TestFileSelection:
    """Tests for file selection when limiting."""

    def test_max_files(self):
        """Files are limited to max_files."""
        # Create diff with many files
        files = []
        for i in range(30):
            files.append(f"""diff --git a/file{i}.py b/file{i}.py
--- a/file{i}.py
+++ b/file{i}.py
@@ -1,2 +1,3 @@
 ctx
+added
 ctx2
""")

        content = "\n".join(files)

        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_files=10,
                min_lines_for_ccr=20,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Count diff --git headers
        file_count = result.compressed.count("diff --git")
        assert file_count <= 10


class TestCompressionResult:
    """Tests for DiffCompressionResult properties."""

    def test_compression_ratio_calculation(self):
        """Compression ratio is calculated correctly."""
        result = DiffCompressionResult(
            compressed="a\nb\nc",
            original_line_count=100,
            compressed_line_count=10,
            files_affected=2,
            additions=5,
            deletions=3,
            hunks_kept=2,
            hunks_removed=5,
        )

        assert result.compression_ratio == 0.1

    def test_tokens_saved_estimate(self):
        """Token savings estimation works correctly."""
        result = DiffCompressionResult(
            compressed="short",
            original_line_count=100,
            compressed_line_count=10,
            files_affected=1,
            additions=10,
            deletions=5,
            hunks_kept=1,
            hunks_removed=0,
        )

        # 90 lines saved * 40 chars/line / 4 chars/token = 900 tokens
        assert result.tokens_saved_estimate == 900


class TestHunkScoring:
    """Tests for context-aware hunk scoring."""

    def test_score_by_context_keywords(self):
        """Hunks containing context keywords get higher scores."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 normal context
+normal change
 more context
@@ -10,3 +11,4 @@
 error handling
+fix the bug here
 return result
"""
        compressor = DiffCompressor()
        diff_files = compressor._parse_diff(content.split("\n"))
        compressor._score_hunks(diff_files, "fix error bug")

        # Second hunk should have higher score (contains "fix" and "bug")
        assert len(diff_files[0].hunks) == 2
        assert diff_files[0].hunks[1].score > diff_files[0].hunks[0].score

    def test_score_priority_patterns(self):
        """Hunks with priority patterns (error, security) score higher."""
        compressor = DiffCompressor()

        hunk_normal = DiffHunk(
            header="@@ -1,1 +1,2 @@",
            lines=["+normal change"],
            additions=1,
        )
        hunk_error = DiffHunk(
            header="@@ -10,1 +10,2 @@",
            lines=["+fix critical error"],
            additions=1,
        )

        diff_file = DiffFile(
            header="diff --git a/f.py b/f.py",
            old_file="--- a/f.py",
            new_file="+++ b/f.py",
            hunks=[hunk_normal, hunk_error],
        )

        compressor._score_hunks([diff_file], "")

        assert hunk_error.score > hunk_normal.score


class TestSmallDiffPassthrough:
    """Tests for small diff passthrough behavior."""

    def test_small_diff_unchanged(self):
        """Diffs smaller than threshold pass through unchanged."""
        content = """diff --git a/small.py b/small.py
--- a/small.py
+++ b/small.py
@@ -1,2 +1,3 @@
 line1
+added
 line2
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                min_lines_for_ccr=100,  # High threshold
            )
        )
        result = compressor.compress(content)

        # Should be unchanged
        assert result.compressed == content
        assert result.compression_ratio == 1.0


class TestOutputFormatting:
    """Tests for output formatting."""

    def test_summary_line_added(self):
        """Summary line is added at end of compressed diff."""
        # Large diff that will be compressed
        hunks = []
        for i in range(15):
            hunks.append(f"""@@ -{i * 10},5 +{i * 10},6 @@
 ctx1
 ctx2
+added_{i}
 ctx3
 ctx4
""")

        content = f"""diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
{"".join(hunks)}"""

        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_hunks_per_file=5,
                min_lines_for_ccr=10,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should have summary at end
        assert "files changed" in result.compressed
        assert "hunks omitted" in result.compressed

    def test_preserves_diff_format(self):
        """Output preserves valid unified diff format."""
        content = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def test():
+    # new comment
     pass
     return True
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                min_lines_for_ccr=5,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should have all standard diff markers
        assert "diff --git" in result.compressed
        assert "---" in result.compressed
        assert "+++" in result.compressed
        assert "@@" in result.compressed


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_input(self):
        """Empty input is handled gracefully."""
        compressor = DiffCompressor()
        result = compressor.compress("")

        assert result.compressed == ""
        assert result.compression_ratio == 1.0

    def test_non_diff_input(self):
        """Non-diff input passes through unchanged."""
        content = "This is not a diff\nJust regular text"
        compressor = DiffCompressor()
        result = compressor.compress(content)

        # Should pass through (no diff --git found)
        assert result.compressed == content

    def test_unicode_content(self):
        """Unicode characters in diff are handled."""
        content = """diff --git a/i18n.py b/i18n.py
--- a/i18n.py
+++ b/i18n.py
@@ -1,2 +1,3 @@
 msg = "hello"
+msg_ja = "こんにちは"
 return msg
"""
        compressor = DiffCompressor()
        result = compressor.compress(content)

        assert "こんにちは" in result.compressed

    def test_no_newline_at_eof(self):
        """Handles 'No newline at end of file' indicator."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,2 +1,2 @@
 line1
-line2
\\ No newline at end of file
+line2_modified
\\ No newline at end of file
"""
        compressor = DiffCompressor()
        result = compressor.compress(content)

        # Should not crash and preserve the indicator
        assert "No newline" in result.compressed or "-line2" in result.compressed

    def test_empty_hunks(self):
        """Files with no actual hunks are handled."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
"""
        compressor = DiffCompressor()
        result = compressor.compress(content)

        # Should not crash
        assert result.compressed is not None


class TestDiffHunkDataclass:
    """Tests for DiffHunk dataclass."""

    def test_change_count_property(self):
        """change_count returns sum of additions and deletions."""
        hunk = DiffHunk(
            header="@@ -1,5 +1,6 @@",
            lines=["+a", "+b", "-c", " ctx"],
            additions=2,
            deletions=1,
        )
        assert hunk.change_count == 3

    def test_default_values(self):
        """DiffHunk default values are correct."""
        hunk = DiffHunk(header="@@", lines=[])
        assert hunk.additions == 0
        assert hunk.deletions == 0
        assert hunk.context_lines == 0
        assert hunk.score == 0.0


class TestDiffFileDataclass:
    """Tests for DiffFile dataclass."""

    def test_total_additions_property(self):
        """total_additions sums across all hunks."""
        hunk1 = DiffHunk(header="@@", lines=[], additions=3)
        hunk2 = DiffHunk(header="@@", lines=[], additions=5)
        diff_file = DiffFile(
            header="diff --git",
            old_file="---",
            new_file="+++",
            hunks=[hunk1, hunk2],
        )
        assert diff_file.total_additions == 8

    def test_total_deletions_property(self):
        """total_deletions sums across all hunks."""
        hunk1 = DiffHunk(header="@@", lines=[], deletions=2)
        hunk2 = DiffHunk(header="@@", lines=[], deletions=4)
        diff_file = DiffFile(
            header="diff --git",
            old_file="---",
            new_file="+++",
            hunks=[hunk1, hunk2],
        )
        assert diff_file.total_deletions == 6


class TestConfigOptions:
    """Tests for configuration options."""

    def test_max_context_lines_config(self):
        """max_context_lines configuration controls context reduction."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,10 +1,11 @@
 c1
 c2
 c3
 c4
 c5
+added
 c6
 c7
 c8
 c9
 c10
"""
        # With max_context_lines=1
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_context_lines=1,
                min_lines_for_ccr=5,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Count context lines (lines starting with space)
        context_count = sum(1 for line in result.compressed.split("\n") if line.startswith(" "))

        # Should have at most 2 context lines (1 before + 1 after)
        assert context_count <= 2

    def test_always_keep_additions_default(self):
        """Additions are always kept by default."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,5 @@
 ctx
+add1
+add2
 ctx
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                always_keep_additions=True,
                min_lines_for_ccr=2,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        assert "+add1" in result.compressed
        assert "+add2" in result.compressed

    def test_always_keep_deletions_default(self):
        """Deletions are always kept by default."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,5 +1,3 @@
 ctx
-del1
-del2
 ctx
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                always_keep_deletions=True,
                min_lines_for_ccr=2,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        assert "-del1" in result.compressed
        assert "-del2" in result.compressed
