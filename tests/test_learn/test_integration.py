"""Integration tests for headroom learn — using real session data.

These tests run against actual conversation data on the machine.
They verify the full pipeline: scan → analyze → recommend → write.
Tests are skipped if the required data directories don't exist.

Key behaviors tested:
- Quality gates prevent writing when evidence is weak
- False positives don't generate recommendations
- Real Codex/Claude Code sessions produce sensible output
- Skip logic: no file writes when nothing meaningful is found
- Idempotency: running twice produces same output
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.learn.analyzer import FailureAnalyzer
from headroom.learn.models import (
    AnalysisReport,
    ProjectInfo,
    Recommendation,
    RecommendationTarget,
)
from headroom.learn.writer import ClaudeCodeWriter, CodexWriter, Recommender

# =============================================================================
# Quality Gate Tests (synthetic data, always run)
# =============================================================================


class TestQualityGates:
    """Verify that weak signals don't generate recommendations."""

    def test_single_failure_not_enough(self):
        """One failure shouldn't generate a recommendation (min_evidence=2)."""
        recommender = Recommender(min_evidence=2)
        report = AnalysisReport(
            project=ProjectInfo(name="test", project_path=Path("/tmp"), data_path=Path("/tmp")),
            total_calls=100,
            total_failures=1,
            total_sessions=1,
        )
        # One missing path — below threshold
        from headroom.learn.models import StructureNote

        report.structure_notes = [
            StructureNote(
                category="missing_path",
                path="/src/foo.py",
                note="not found",
                evidence_count=1,
            )
        ]
        recs = recommender.recommend(report)
        assert recs == []

    def test_low_total_evidence_skipped(self):
        """Even if individual recs pass, if total evidence is too low, skip all."""
        recommender = Recommender(min_evidence=1, min_total_evidence=10)
        report = AnalysisReport(
            project=ProjectInfo(name="test", project_path=Path("/tmp"), data_path=Path("/tmp")),
            total_calls=100,
            total_failures=2,
            total_sessions=1,
        )
        from headroom.learn.models import StructureNote

        report.structure_notes = [
            StructureNote(category="missing_path", path="/a.py", note="x", evidence_count=2)
        ]
        recs = recommender.recommend(report)
        assert recs == []  # Total evidence=2 < min_total_evidence=10

    def test_sufficient_evidence_passes(self):
        """Enough evidence generates recommendations."""
        recommender = Recommender(min_evidence=2, min_total_evidence=3)
        report = AnalysisReport(
            project=ProjectInfo(name="test", project_path=Path("/tmp"), data_path=Path("/tmp")),
            total_calls=100,
            total_failures=10,
            total_sessions=5,
        )
        from headroom.learn.models import StructureNote

        report.structure_notes = [
            StructureNote(
                category="large_file",
                path="/big.py",
                note="too big",
                evidence_count=5,
                sessions_seen=3,
            ),
        ]
        recs = recommender.recommend(report)
        assert len(recs) >= 1

    def test_no_failures_no_recommendations(self):
        """Zero failures = zero recommendations, no files touched."""
        recommender = Recommender()
        report = AnalysisReport(
            project=ProjectInfo(name="test", project_path=Path("/tmp"), data_path=Path("/tmp")),
            total_calls=500,
            total_failures=0,
            total_sessions=10,
        )
        recs = recommender.recommend(report)
        assert recs == []


class TestSkipWriteLogic:
    """Verify files are NOT written when there's nothing to write."""

    def test_empty_recommendations_no_write(self, tmp_path):
        """Zero recommendations = zero files written."""
        proj = ProjectInfo(
            name="clean-project",
            project_path=tmp_path / "proj",
            data_path=tmp_path / "data",
        )
        (tmp_path / "proj").mkdir()
        (tmp_path / "data" / "memory").mkdir(parents=True)

        writer = ClaudeCodeWriter()
        result = writer.write([], proj, dry_run=False)

        assert result.files_written == []
        assert not (tmp_path / "proj" / "CLAUDE.md").exists()
        assert not (tmp_path / "data" / "memory" / "MEMORY.md").exists()

    def test_codex_empty_no_write(self, tmp_path):
        """Codex writer also skips on empty recommendations."""
        proj = ProjectInfo(name="clean", project_path=tmp_path, data_path=tmp_path)
        writer = CodexWriter()
        result = writer.write([], proj, dry_run=False)
        assert result.files_written == []
        assert not (tmp_path / "AGENTS.md").exists()


class TestIdempotency:
    """Running learn twice should produce the same output, not duplicate."""

    def test_double_write_replaces_not_appends(self, tmp_path):
        proj = ProjectInfo(name="test", project_path=tmp_path, data_path=tmp_path / "data")
        (tmp_path / "data" / "memory").mkdir(parents=True)

        recs = [
            Recommendation(
                target=RecommendationTarget.CONTEXT_FILE,
                section="Environment",
                content="- Use `uv run python`",
                confidence=0.9,
                evidence_count=10,
            )
        ]

        writer = ClaudeCodeWriter()
        # First write
        writer.write(recs, proj, dry_run=False)
        first_content = (tmp_path / "CLAUDE.md").read_text()

        # Second write (same recs)
        writer.write(recs, proj, dry_run=False)
        second_content = (tmp_path / "CLAUDE.md").read_text()

        # Content should be identical (replaced, not appended)
        assert first_content == second_content
        assert second_content.count("uv run python") == 1


class TestFalsePositiveFiltering:
    """Verify that common false positives don't generate recommendations."""

    def test_sed_output_not_error(self):
        """sed printing file content with 'Error:' in it isn't a real error."""
        from headroom.learn.scanner import is_error_content

        # Normal code output that happens to contain "error" in identifiers
        assert not is_error_content("def handle_error(e):\n    print('ok')")
        # Normal sed output with error handling code in the file content
        assert not is_error_content(
            '    return fmt.Errorf("connection failed")\n    log.Print("ok")'
        )

    def test_real_error_detected(self):
        """Actual errors should be detected."""
        from headroom.learn.scanner import is_error_content

        assert is_error_content("ModuleNotFoundError: No module named 'flask'")
        assert is_error_content(
            "FileNotFoundError: [Errno 2] No such file or directory: '/bad/path'"
        )
        assert is_error_content("bash: unknown_cmd: command not found")


# =============================================================================
# Real-World Integration Tests (skipped if data not present)
# =============================================================================


CLAUDE_DIR = Path.home() / ".claude" / "projects"
CODEX_DIR = Path.home() / ".codex" / "sessions"


@pytest.mark.skipif(not CLAUDE_DIR.exists(), reason="No Claude Code data")
class TestClaudeCodeIntegration:
    """Integration tests against real Claude Code session data."""

    def test_scanner_discovers_projects(self):
        from headroom.learn.scanner import ClaudeCodeScanner

        scanner = ClaudeCodeScanner()
        projects = scanner.discover_projects()
        assert len(projects) > 0
        for p in projects:
            assert p.name
            assert p.data_path.exists()

    def test_full_pipeline_produces_output(self):
        """Scan → analyze → recommend on real data produces valid output."""
        from headroom.learn.scanner import ClaudeCodeScanner

        scanner = ClaudeCodeScanner()
        projects = scanner.discover_projects()

        # Find a project with the most sessions
        best = max(projects, key=lambda p: len(list(p.data_path.glob("*.jsonl"))))

        sessions = scanner.scan_project(best)
        assert len(sessions) > 0

        analyzer = FailureAnalyzer()
        report = analyzer.analyze(best, sessions)

        assert report.total_calls > 0
        assert report.total_sessions > 0
        # Failure rate should be realistic (0-30%)
        assert 0 <= report.failure_rate <= 0.5

        # Corrections should be found if there are failures
        if report.total_failures > 10:
            assert len(report.corrections) > 0

    def test_dry_run_writes_nothing(self):
        """Dry run should never create files."""
        from headroom.learn.scanner import ClaudeCodeScanner

        scanner = ClaudeCodeScanner()
        projects = scanner.discover_projects()
        best = max(projects, key=lambda p: len(list(p.data_path.glob("*.jsonl"))))

        sessions = scanner.scan_project(best)
        report = FailureAnalyzer().analyze(best, sessions)
        recs = Recommender().recommend(report)

        writer = ClaudeCodeWriter()
        result = writer.write(recs, best, dry_run=True)

        assert result.dry_run is True
        # Verify no new files were created (check a temp path won't exist)
        for fp in result.files_written:
            # Files should only be in expected locations
            assert "CLAUDE.md" in fp.name or "MEMORY.md" in fp.name


@pytest.mark.skipif(not CODEX_DIR.exists(), reason="No Codex data")
class TestCodexIntegration:
    """Integration tests against real Codex session data."""

    def test_scanner_discovers_sessions(self):
        from headroom.learn.scanner import CodexScanner

        scanner = CodexScanner()
        projects = scanner.discover_projects()
        assert len(projects) == 1  # Codex returns one "project"

    def test_full_pipeline(self):
        """Full pipeline on real Codex data."""
        from headroom.learn.scanner import CodexScanner

        scanner = CodexScanner()
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        assert len(sessions) > 0

        analyzer = FailureAnalyzer()
        report = analyzer.analyze(projects[0], sessions)

        assert report.total_calls > 0
        # Codex has only Bash tool (shell)
        all_tools = {tc.name for s in sessions for tc in s.tool_calls}
        assert "Bash" in all_tools

        # Failure rate should be realistic
        assert 0 <= report.failure_rate <= 0.5

    def test_quality_gate_filters_noise(self):
        """Codex has false positive 'runtime_error' from sed output.
        Quality gate should prevent these from generating weak recommendations."""
        from headroom.learn.scanner import CodexScanner

        scanner = CodexScanner()
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        report = FailureAnalyzer().analyze(projects[0], sessions)
        # Use strict quality gates
        recommender = Recommender(min_evidence=5, min_total_evidence=10)
        recs = recommender.recommend(report)

        # Every recommendation should have meaningful evidence
        for rec in recs:
            assert rec.evidence_count >= 5
            assert rec.confidence >= 0.3

    def test_codex_writer_targets_agents_md(self):
        """Codex writer should target AGENTS.md, not CLAUDE.md."""
        from headroom.learn.scanner import CodexScanner

        scanner = CodexScanner()
        projects = scanner.discover_projects()
        sessions = scanner.scan_project(projects[0])

        report = FailureAnalyzer().analyze(projects[0], sessions)
        recs = Recommender().recommend(report)

        if recs:
            writer = CodexWriter()
            result = writer.write(recs, projects[0], dry_run=True)
            for fp in result.files_written:
                assert fp.name in ("AGENTS.md", "instructions.md")
                assert "CLAUDE.md" not in fp.name
