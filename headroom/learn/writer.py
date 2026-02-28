"""Context writers — write learned patterns to agent-specific context files.

Writers take Recommendations and write them to the appropriate context
injection mechanism for each agent system (CLAUDE.md, .cursorrules, etc.).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    AnalysisReport,
    CommandPattern,
    EnvironmentFact,
    ProjectInfo,
    Recommendation,
    RecommendationTarget,
    RetryPattern,
    StructureNote,
)

# Marker delimiters for Headroom-managed sections
_MARKER_START = "<!-- headroom:learn:start -->"
_MARKER_END = "<!-- headroom:learn:end -->"
_MARKER_PATTERN = re.compile(
    re.escape(_MARKER_START) + r".*?" + re.escape(_MARKER_END),
    re.DOTALL,
)


# =============================================================================
# Recommender: AnalysisReport → Recommendations
# =============================================================================


class Recommender:
    """Converts an AnalysisReport into concrete markdown recommendations."""

    def recommend(self, report: AnalysisReport) -> list[Recommendation]:
        recommendations: list[Recommendation] = []

        # Environment facts → CONTEXT_FILE (CLAUDE.md)
        if report.environment_facts:
            content = self._format_environment(report.environment_facts)
            recommendations.append(
                Recommendation(
                    target=RecommendationTarget.CONTEXT_FILE,
                    section="Environment",
                    content=content,
                    confidence=min(
                        1.0, sum(f.evidence_count for f in report.environment_facts) / 10
                    ),
                    evidence_count=sum(f.evidence_count for f in report.environment_facts),
                )
            )

        # Large files → CONTEXT_FILE
        large_files = [n for n in report.structure_notes if n.category == "large_file"]
        if large_files:
            content = self._format_large_files(large_files)
            recommendations.append(
                Recommendation(
                    target=RecommendationTarget.CONTEXT_FILE,
                    section="Known Large Files",
                    content=content,
                    confidence=min(1.0, sum(n.evidence_count for n in large_files) / 5),
                    evidence_count=sum(n.evidence_count for n in large_files),
                )
            )

        # Path corrections → CONTEXT_FILE (these are stable project structure facts)
        path_corrections = [n for n in report.structure_notes if n.category == "path_correction"]
        if path_corrections:
            content = self._format_path_corrections(path_corrections)
            recommendations.append(
                Recommendation(
                    target=RecommendationTarget.CONTEXT_FILE,
                    section="File Path Corrections",
                    content=content,
                    confidence=0.9,
                    evidence_count=sum(n.evidence_count for n in path_corrections),
                )
            )

        # Search scope corrections → CONTEXT_FILE
        scope_corrections = [n for n in report.structure_notes if n.category == "search_scope"]
        if scope_corrections:
            content = self._format_scope_corrections(scope_corrections)
            recommendations.append(
                Recommendation(
                    target=RecommendationTarget.CONTEXT_FILE,
                    section="Search Scope",
                    content=content,
                    confidence=0.8,
                    evidence_count=sum(n.evidence_count for n in scope_corrections),
                )
            )

        # Command patterns → CONTEXT_FILE (stable project-level facts)
        if report.command_patterns:
            content = self._format_command_patterns(report.command_patterns)
            recommendations.append(
                Recommendation(
                    target=RecommendationTarget.CONTEXT_FILE,
                    section="Command Patterns",
                    content=content,
                    confidence=0.9,
                    evidence_count=sum(p.evidence_count for p in report.command_patterns),
                )
            )

        # Missing paths (no correction found) → MEMORY_FILE
        missing_paths = [n for n in report.structure_notes if n.category == "missing_path"]
        if missing_paths:
            content = self._format_missing_paths(missing_paths)
            recommendations.append(
                Recommendation(
                    target=RecommendationTarget.MEMORY_FILE,
                    section="Known Missing Paths",
                    content=content,
                    confidence=0.6,
                    evidence_count=sum(n.evidence_count for n in missing_paths),
                )
            )

        # Retry patterns (with specific suggestions) → MEMORY_FILE
        if report.retry_patterns:
            content = self._format_retry_patterns(report.retry_patterns)
            recommendations.append(
                Recommendation(
                    target=RecommendationTarget.MEMORY_FILE,
                    section="Retry Prevention",
                    content=content,
                    confidence=0.7,
                    evidence_count=sum(p.evidence_count for p in report.retry_patterns),
                )
            )

        # Permission issues → MEMORY_FILE
        if report.permission_issues:
            content = self._format_permissions(report.permission_issues)
            recommendations.append(
                Recommendation(
                    target=RecommendationTarget.MEMORY_FILE,
                    section="Permission Notes",
                    content=content,
                    confidence=0.5,
                    evidence_count=len(report.permission_issues),
                )
            )

        return recommendations

    def _format_environment(self, facts: list[EnvironmentFact]) -> str:
        lines = []
        for fact in facts:
            wrong = ", ".join(f"`{w}`" for w in fact.wrong_commands[:3])
            lines.append(
                f"- **{fact.category.title()}**: use `{fact.correct_command}` "
                f"(not {wrong} — {fact.evidence_count} failures observed)"
            )
        return "\n".join(lines)

    def _format_large_files(self, notes: list[StructureNote]) -> str:
        lines = ["Always use `offset` and `limit` parameters with Read for these files:"]
        for note in sorted(notes, key=lambda n: -n.evidence_count):
            lines.append(f"- `{note.path}` ({note.note})")
        return "\n".join(lines)

    def _format_path_corrections(self, notes: list[StructureNote]) -> str:
        lines = ["These file paths are commonly guessed wrong. Use the correct paths:"]
        for note in sorted(notes, key=lambda n: -n.evidence_count):
            lines.append(f"- `{note.path}` → actually at `{note.correct_path}`")
        return "\n".join(lines)

    def _format_scope_corrections(self, notes: list[StructureNote]) -> str:
        lines = ["When searching, use these scopes (broader paths work, narrow ones fail):"]
        for note in sorted(notes, key=lambda n: -n.evidence_count):
            lines.append(f"- Don't search `{note.path}` → use `{note.correct_path}` instead")
        return "\n".join(lines)

    def _format_command_patterns(self, patterns: list[CommandPattern]) -> str:
        lines = []
        for p in sorted(patterns, key=lambda p: -p.evidence_count):
            lines.append(f"- **{p.category}**: {p.explanation}")
            lines.append(f"  - Wrong: {p.wrong_pattern}")
            lines.append(f"  - Correct: {p.correct_pattern}")
        return "\n".join(lines)

    def _format_missing_paths(self, notes: list[StructureNote]) -> str:
        lines = []
        for note in sorted(notes, key=lambda n: -n.evidence_count):
            lines.append(f"- `{note.path}` — {note.note}")
        return "\n".join(lines)

    def _format_retry_patterns(self, patterns: list[RetryPattern]) -> str:
        lines = []
        for p in sorted(patterns, key=lambda p: -p.evidence_count):
            lines.append(f"- {p.description}")
            lines.append(f"  → {p.suggestion}")
        return "\n".join(lines)

    def _format_permissions(self, issues: list[str]) -> str:
        lines = []
        for issue in issues:
            lines.append(f"- {issue}")
        return "\n".join(lines)


# =============================================================================
# Abstract Writer
# =============================================================================


class ContextWriter(ABC):
    """Base class for writing recommendations to context/memory files."""

    @abstractmethod
    def write(
        self,
        recommendations: list[Recommendation],
        project: ProjectInfo,
        dry_run: bool = True,
    ) -> WriteResult: ...


# =============================================================================
# Write Result
# =============================================================================


class WriteResult:
    """Result of a write operation."""

    def __init__(self) -> None:
        self.files_written: list[Path] = []
        self.content_by_file: dict[Path, str] = {}
        self.dry_run: bool = True

    def add(self, path: Path, content: str) -> None:
        self.files_written.append(path)
        self.content_by_file[path] = content


# =============================================================================
# Claude Code Writer
# =============================================================================


class ClaudeCodeWriter(ContextWriter):
    """Writes learned patterns to CLAUDE.md and MEMORY.md for Claude Code."""

    def write(
        self,
        recommendations: list[Recommendation],
        project: ProjectInfo,
        dry_run: bool = True,
    ) -> WriteResult:
        result = WriteResult()
        result.dry_run = dry_run

        # Group recommendations by target
        context_recs = [r for r in recommendations if r.target == RecommendationTarget.CONTEXT_FILE]
        memory_recs = [r for r in recommendations if r.target == RecommendationTarget.MEMORY_FILE]

        # Generate CLAUDE.md content
        if context_recs:
            claude_md_path = self._resolve_context_path(project)
            section_content = self._build_section(context_recs)
            full_content = self._merge_into_file(claude_md_path, section_content)
            result.add(claude_md_path, full_content)

            if not dry_run:
                claude_md_path.parent.mkdir(parents=True, exist_ok=True)
                claude_md_path.write_text(full_content)

        # Generate MEMORY.md content
        if memory_recs:
            memory_path = self._resolve_memory_path(project)
            section_content = self._build_section(memory_recs)
            full_content = self._merge_into_file(memory_path, section_content)
            result.add(memory_path, full_content)

            if not dry_run:
                memory_path.parent.mkdir(parents=True, exist_ok=True)
                memory_path.write_text(full_content)

        return result

    def _resolve_context_path(self, project: ProjectInfo) -> Path:
        """Resolve path for CLAUDE.md."""
        if project.context_file:
            return project.context_file
        return project.project_path / "CLAUDE.md"

    def _resolve_memory_path(self, project: ProjectInfo) -> Path:
        """Resolve path for MEMORY.md."""
        if project.memory_file:
            return project.memory_file
        return project.data_path / "memory" / "MEMORY.md"

    def _build_section(self, recommendations: list[Recommendation]) -> str:
        """Build the marker-delimited section content."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lines = [
            _MARKER_START,
            "## Headroom Learned Patterns",
            f"*Auto-generated by `headroom learn` on {now} — do not edit manually*",
            "",
        ]

        for rec in recommendations:
            lines.append(f"### {rec.section}")
            lines.append(rec.content)
            lines.append("")

        lines.append(_MARKER_END)
        return "\n".join(lines)

    def _merge_into_file(self, file_path: Path, section: str) -> str:
        """Merge the section into an existing file, replacing any prior section."""
        if file_path.exists():
            existing = file_path.read_text()
            # Replace existing headroom section
            if _MARKER_START in existing:
                return _MARKER_PATTERN.sub(section, existing)
            # Append to end
            return existing.rstrip() + "\n\n" + section + "\n"
        else:
            return section + "\n"
