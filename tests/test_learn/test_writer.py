"""Tests for recommendation writer — marker-based file updates."""

from pathlib import Path

from headroom.learn.models import ProjectInfo, Recommendation, RecommendationTarget
from headroom.learn.writer import _MARKER_END, _MARKER_START, ClaudeCodeWriter


def _project(tmp_path: Path) -> ProjectInfo:
    proj_dir = tmp_path / "myproject"
    proj_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    memory_dir = data_dir / "memory"
    memory_dir.mkdir()
    return ProjectInfo(
        name="myproject",
        project_path=proj_dir,
        data_path=data_dir,
    )


def _rec(target: RecommendationTarget, section: str, content: str) -> Recommendation:
    return Recommendation(
        target=target, section=section, content=content, confidence=0.8, evidence_count=5
    )


class TestClaudeCodeWriter:
    def test_dry_run_does_not_write(self, tmp_path):
        proj = _project(tmp_path)
        writer = ClaudeCodeWriter()
        recs = [_rec(RecommendationTarget.CONTEXT_FILE, "Environment", "- Use uv")]

        result = writer.write(recs, proj, dry_run=True)

        assert result.dry_run is True
        assert len(result.files_written) == 1
        # File should NOT exist (dry run)
        claude_md = proj.project_path / "CLAUDE.md"
        assert not claude_md.exists()

    def test_apply_writes_claude_md(self, tmp_path):
        proj = _project(tmp_path)
        writer = ClaudeCodeWriter()
        recs = [_rec(RecommendationTarget.CONTEXT_FILE, "Environment", "- Use `uv run python`")]

        result = writer.write(recs, proj, dry_run=False)

        assert result.dry_run is False
        claude_md = proj.project_path / "CLAUDE.md"
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "uv run python" in content
        assert _MARKER_START in content
        assert _MARKER_END in content

    def test_apply_writes_memory_md(self, tmp_path):
        proj = _project(tmp_path)
        writer = ClaudeCodeWriter()
        recs = [_rec(RecommendationTarget.MEMORY_FILE, "Retry Prevention", "- Don't retry globs")]

        writer.write(recs, proj, dry_run=False)

        memory_md = proj.data_path / "memory" / "MEMORY.md"
        assert memory_md.exists()
        assert "Don't retry globs" in memory_md.read_text()

    def test_preserves_existing_claude_md_content(self, tmp_path):
        proj = _project(tmp_path)
        claude_md = proj.project_path / "CLAUDE.md"
        claude_md.write_text("# My Project\n\nExisting instructions here.\n")

        writer = ClaudeCodeWriter()
        recs = [_rec(RecommendationTarget.CONTEXT_FILE, "Environment", "- Use uv")]
        writer.write(recs, proj, dry_run=False)

        content = claude_md.read_text()
        assert "My Project" in content
        assert "Existing instructions here" in content
        assert "Use uv" in content

    def test_replaces_existing_headroom_section(self, tmp_path):
        proj = _project(tmp_path)
        claude_md = proj.project_path / "CLAUDE.md"
        old_section = (
            f"# My Project\n\n{_MARKER_START}\n## Old Patterns\nold stuff\n{_MARKER_END}\n"
        )
        claude_md.write_text(old_section)

        writer = ClaudeCodeWriter()
        recs = [_rec(RecommendationTarget.CONTEXT_FILE, "Environment", "- New stuff")]
        writer.write(recs, proj, dry_run=False)

        content = claude_md.read_text()
        assert "old stuff" not in content
        assert "New stuff" in content
        assert "My Project" in content
        # Should have exactly one marker pair
        assert content.count(_MARKER_START) == 1
        assert content.count(_MARKER_END) == 1

    def test_appends_to_existing_memory_md(self, tmp_path):
        proj = _project(tmp_path)
        memory_md = proj.data_path / "memory" / "MEMORY.md"
        memory_md.write_text("# Existing Memory\n\nSome facts.\n")

        writer = ClaudeCodeWriter()
        recs = [_rec(RecommendationTarget.MEMORY_FILE, "Retry Prevention", "- New pattern")]
        writer.write(recs, proj, dry_run=False)

        content = memory_md.read_text()
        assert "Existing Memory" in content
        assert "Some facts" in content
        assert "New pattern" in content
