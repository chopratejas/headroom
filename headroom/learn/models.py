"""Data models for Headroom Learn — tool-agnostic abstractions.

These models normalize tool call data from ANY agent system (Claude Code, Cursor,
Codex, custom agents) into a common format that analyzers can work with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# =============================================================================
# Error Classification
# =============================================================================


class ErrorCategory(str, Enum):
    """Classified error categories for tool call failures."""

    FILE_NOT_FOUND = "file_not_found"
    MODULE_NOT_FOUND = "module_not_found"
    COMMAND_NOT_FOUND = "command_not_found"
    PERMISSION_DENIED = "permission_denied"
    FILE_TOO_LARGE = "file_too_large"
    IS_DIRECTORY = "is_directory"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    NO_MATCHES = "no_matches"  # Grep/Glob found nothing
    USER_REJECTED = "user_rejected"
    SIBLING_ERROR = "sibling_error"  # Cascade from parallel call failure
    EXIT_CODE = "exit_code"
    CONNECTION_ERROR = "connection_error"
    BUILD_FAILURE = "build_failure"
    UNKNOWN = "unknown"


# =============================================================================
# Core Data Models (Tool-Agnostic)
# =============================================================================


@dataclass
class ToolCall:
    """A single tool call and its result — normalized from any agent system.

    This is the fundamental unit of analysis. Scanners produce these,
    analyzers consume them.
    """

    name: str  # Tool name ("Bash", "Read", "file_search", etc.)
    tool_call_id: str  # Unique ID linking call to result
    input_data: dict  # Tool input parameters
    output: str  # Result content (may be error message)
    is_error: bool  # Whether the call failed
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    msg_index: int = 0  # Position in conversation
    output_bytes: int = 0  # Size of output

    @property
    def input_summary(self) -> str:
        """Short summary of tool input for display."""
        if self.name in ("Bash", "bash"):
            cmd: str = self.input_data.get("command", "")
            return cmd[:100] + "..." if len(cmd) > 100 else cmd
        if self.name in ("Read", "read"):
            return str(self.input_data.get("file_path", "?"))
        if self.name in ("Grep", "grep"):
            return str(self.input_data.get("pattern", "?"))
        if self.name in ("Glob", "glob"):
            return str(self.input_data.get("pattern", "?"))
        if self.name in ("Edit", "edit", "Write", "write"):
            return str(self.input_data.get("file_path", "?"))
        return str(self.input_data)[:80]


@dataclass
class SessionData:
    """Normalized data from a single conversation session."""

    session_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: datetime | None = None

    @property
    def failure_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.is_error)

    @property
    def failure_rate(self) -> float:
        if not self.tool_calls:
            return 0.0
        return self.failure_count / len(self.tool_calls)


@dataclass
class ProjectInfo:
    """Information about a project discovered by a scanner."""

    name: str  # Human-readable project name
    project_path: Path  # Actual project directory
    data_path: Path  # Where conversation logs are stored
    context_file: Path | None = None  # CLAUDE.md / .cursorrules / AGENTS.md
    memory_file: Path | None = None  # MEMORY.md or equivalent


# =============================================================================
# Analysis Output Models
# =============================================================================


class RecommendationTarget(str, Enum):
    """Where a recommendation should be written."""

    CONTEXT_FILE = "context_file"  # CLAUDE.md, .cursorrules, AGENTS.md
    MEMORY_FILE = "memory_file"  # MEMORY.md or equivalent


@dataclass
class EnvironmentFact:
    """A learned fact about the project's runtime environment."""

    category: str  # "python", "build_tool", "test_runner", "linter"
    correct_command: str  # What works: "uv run python"
    wrong_commands: list[str] = field(default_factory=list)  # What fails: ["python3"]
    evidence_count: int = 0  # How many failures support this
    sessions_seen: int = 0  # Across how many sessions


@dataclass
class StructureNote:
    """A learned fact about the project's file structure."""

    category: str  # "large_file", "missing_path", "path_correction", "search_scope"
    path: str  # The file path in question
    note: str  # Human-readable note
    correct_path: str = ""  # If corrected, what the actual path is
    evidence_count: int = 0
    sessions_seen: int = 0


@dataclass
class Correction:
    """A failure→success pair: what failed and what worked instead.

    This is the core learning primitive. By comparing the failed input to
    the successful input, we extract specific actionable knowledge.
    """

    tool_name: str
    failed_input: dict  # The input that failed
    success_input: dict  # The input that succeeded
    error_category: ErrorCategory
    session_id: str

    @property
    def failed_summary(self) -> str:
        if self.tool_name in ("Read", "read"):
            return str(self.failed_input.get("file_path", "?"))
        if self.tool_name in ("Bash", "bash"):
            return str(self.failed_input.get("command", "?"))[:100]
        if self.tool_name in ("Grep", "grep"):
            path = str(self.failed_input.get("path", ""))
            pattern = str(self.failed_input.get("pattern", ""))
            return f"pattern={pattern[:40]} path={path}"
        return str(self.failed_input)[:80]

    @property
    def success_summary(self) -> str:
        if self.tool_name in ("Read", "read"):
            return str(self.success_input.get("file_path", "?"))
        if self.tool_name in ("Bash", "bash"):
            return str(self.success_input.get("command", "?"))[:100]
        if self.tool_name in ("Grep", "grep"):
            path = str(self.success_input.get("path", ""))
            pattern = str(self.success_input.get("pattern", ""))
            return f"pattern={pattern[:40]} path={path}"
        return str(self.success_input)[:80]


@dataclass
class CommandPattern:
    """A learned pattern about how commands should be run in this project."""

    category: str  # "gradle", "python", "test", "build", "lint"
    wrong_pattern: str  # What fails (e.g., "cd /path && ./gradlew")
    correct_pattern: str  # What works (e.g., "../gradlew from axion/")
    explanation: str  # Why (e.g., "user rejects cd-based gradle, use relative path")
    evidence_count: int = 0
    sessions_seen: int = 0


@dataclass
class RetryPattern:
    """A pattern of stubborn retries that should be prevented."""

    tool_name: str
    error_category: ErrorCategory
    description: str  # What keeps failing
    max_retries_seen: int  # Worst case observed
    suggestion: str  # What to do instead (SPECIFIC, from success correlation)
    evidence_count: int = 0


@dataclass
class Recommendation:
    """A concrete recommendation to write to a context/memory file."""

    target: RecommendationTarget
    section: str  # Section heading (e.g., "Environment", "Known Large Files")
    content: str  # Markdown content for the section
    confidence: float = 0.0  # 0-1, based on evidence strength
    evidence_count: int = 0  # Number of failures supporting this


@dataclass
class AnalysisReport:
    """Complete output of failure analysis for a project."""

    project: ProjectInfo
    total_calls: int = 0
    total_failures: int = 0
    total_sessions: int = 0
    waste_bytes: int = 0

    environment_facts: list[EnvironmentFact] = field(default_factory=list)
    structure_notes: list[StructureNote] = field(default_factory=list)
    retry_patterns: list[RetryPattern] = field(default_factory=list)
    command_patterns: list[CommandPattern] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    permission_issues: list[str] = field(default_factory=list)
    cross_session_patterns: list[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        if not self.total_calls:
            return 0.0
        return self.total_failures / self.total_calls
