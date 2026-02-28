"""Conversation scanners — read tool call logs from different agent systems.

Scanners normalize conversation data into ToolCall sequences that analyzers
can process regardless of the source system.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from .models import (
    ErrorCategory,
    ProjectInfo,
    SessionData,
    ToolCall,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Error Classification
# =============================================================================

# Patterns checked in order — first match wins
_ERROR_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    (
        re.compile(r"No such file or directory|ENOENT|FileNotFoundError|does not exist", re.I),
        ErrorCategory.FILE_NOT_FOUND,
    ),
    (
        re.compile(r"ModuleNotFoundError|ImportError|No module named", re.I),
        ErrorCategory.MODULE_NOT_FOUND,
    ),
    (re.compile(r"command not found", re.I), ErrorCategory.COMMAND_NOT_FOUND),
    (
        re.compile(r"Permission denied|EACCES|EPERM|auto-denied", re.I),
        ErrorCategory.PERMISSION_DENIED,
    ),
    (
        re.compile(r"file is too large|too many lines|exceeds.*limit", re.I),
        ErrorCategory.FILE_TOO_LARGE,
    ),
    (re.compile(r"EISDIR|Is a directory", re.I), ErrorCategory.IS_DIRECTORY),
    (re.compile(r"SyntaxError|IndentationError", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"Traceback \(most recent|Exception:|Error:", re.I), ErrorCategory.RUNTIME_ERROR),
    (re.compile(r"timed? ?out|TimeoutError|deadline exceeded", re.I), ErrorCategory.TIMEOUT),
    (re.compile(r"No (?:matches|files|results) found|0 matches", re.I), ErrorCategory.NO_MATCHES),
    (
        re.compile(r"user.*reject|user.*denied|declined|didn't want to proceed", re.I),
        ErrorCategory.USER_REJECTED,
    ),
    (re.compile(r"[Ss]ibling tool call errored", re.I), ErrorCategory.SIBLING_ERROR),
    (re.compile(r"exit code|non-zero|exited with", re.I), ErrorCategory.EXIT_CODE),
    (
        re.compile(r"ConnectionError|ConnectionRefused|ECONNREFUSED|network", re.I),
        ErrorCategory.CONNECTION_ERROR,
    ),
    (
        re.compile(r"BUILD FAILED|compilation error|compile error", re.I),
        ErrorCategory.BUILD_FAILURE,
    ),
]


def classify_error(content: str) -> ErrorCategory:
    """Classify an error message into a category."""
    for pattern, category in _ERROR_PATTERNS:
        if pattern.search(content[:2000]):  # Only check first 2KB
            return category
    return ErrorCategory.UNKNOWN


def is_error_content(content: str) -> bool:
    """Heuristic: does this tool result look like an error?"""
    if not content or len(content) < 10:
        return False
    # Check for common error indicators in first 1KB
    snippet = content[:1000]
    indicators = [
        "Error:",
        "error:",
        "ENOENT",
        "No such file",
        "command not found",
        "Permission denied",
        "ModuleNotFoundError",
        "Traceback (most recent",
        "FAILED",
        "EISDIR",
        "auto-denied",
        "Sibling tool call errored",
        "timed out",
        "exit code",
        "FileNotFoundError",
    ]
    return any(ind in snippet for ind in indicators)


# =============================================================================
# Abstract Scanner
# =============================================================================


class ConversationScanner(ABC):
    """Base class for scanning conversation logs from any agent system.

    Subclasses implement log format parsing for specific tools (Claude Code,
    Cursor, Codex, etc.) and produce normalized ToolCall sequences.
    """

    @abstractmethod
    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects with conversation data."""
        ...

    @abstractmethod
    def scan_project(self, project: ProjectInfo) -> list[SessionData]:
        """Scan all sessions for a project, returning normalized tool calls."""
        ...


# =============================================================================
# Claude Code Scanner
# =============================================================================


class ClaudeCodeScanner(ConversationScanner):
    """Reads Claude Code conversation logs from ~/.claude/projects/.

    Claude Code stores conversations as JSONL files with these line types:
    - type="assistant": message.content[] has tool_use blocks (name, input, id)
    - type="user": message.content[] has tool_result blocks (tool_use_id, content)
    """

    def __init__(self, claude_dir: Path | None = None):
        self.claude_dir = claude_dir or Path.home() / ".claude"
        self.projects_dir = self.claude_dir / "projects"

    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects under ~/.claude/projects/."""
        if not self.projects_dir.exists():
            return []

        projects = []
        for entry in sorted(self.projects_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            # Decode project path from escaped directory name
            # e.g., "-Users-tchopra-claude-projects-headroom" → "/Users/tchopra/claude-projects/headroom"
            project_path = Path("/" + entry.name.replace("-", "/", entry.name.count("-")))

            # Try smarter decoding: split on segments that look like path components
            # The escaping replaces / with - but also - in names stays as -
            # Heuristic: try the decoded path, if it exists use it
            decoded = _decode_project_path(entry.name)
            if decoded:
                project_path = decoded

            # Derive human-readable name
            name = project_path.name if project_path != Path("/") else entry.name

            # Check for CLAUDE.md in actual project directory
            context_file = None
            if project_path.exists():
                claude_md = project_path / "CLAUDE.md"
                if claude_md.exists():
                    context_file = claude_md

            # Check for MEMORY.md
            memory_dir = entry / "memory"
            memory_file = memory_dir / "MEMORY.md" if memory_dir.exists() else None
            if memory_file and not memory_file.exists():
                memory_file = None

            # Only include projects with JSONL files
            jsonl_files = list(entry.glob("*.jsonl"))
            if not jsonl_files:
                continue

            projects.append(
                ProjectInfo(
                    name=name,
                    project_path=project_path,
                    data_path=entry,
                    context_file=context_file,
                    memory_file=memory_file,
                )
            )

        return projects

    def scan_project(self, project: ProjectInfo) -> list[SessionData]:
        """Scan all conversation JSONL files for a project."""
        sessions = []

        # Find all JSONL files (main conversations, not subagent files)
        jsonl_files = sorted(project.data_path.glob("*.jsonl"))

        for jsonl_path in jsonl_files:
            session = self._scan_session(jsonl_path)
            if session and session.tool_calls:
                sessions.append(session)

        return sessions

    def _scan_session(self, jsonl_path: Path) -> SessionData | None:
        """Scan a single JSONL conversation file."""
        session_id = jsonl_path.stem
        tool_uses: dict[str, tuple[str, dict]] = {}  # tc_id → (tool_name, input)
        tool_calls: list[ToolCall] = []
        msg_index = 0

        try:
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg_index += 1
                    line_type = d.get("type", "")

                    if line_type == "assistant":
                        self._extract_tool_uses(d, tool_uses)
                    elif line_type == "user":
                        self._extract_tool_results(d, tool_uses, tool_calls, msg_index)

        except (OSError, UnicodeDecodeError) as e:
            logger.debug("Failed to read %s: %s", jsonl_path, e)
            return None

        return SessionData(session_id=session_id, tool_calls=tool_calls)

    def _extract_tool_uses(self, d: dict, tool_uses: dict[str, tuple[str, dict]]) -> None:
        """Extract tool_use blocks from an assistant message."""
        msg = d.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tc_id = block.get("id", "")
            name = block.get("name", "")
            inp = block.get("input", {})
            if tc_id and name:
                tool_uses[tc_id] = (name, inp if isinstance(inp, dict) else {})

    def _extract_tool_results(
        self,
        d: dict,
        tool_uses: dict[str, tuple[str, dict]],
        tool_calls: list[ToolCall],
        msg_index: int,
    ) -> None:
        """Extract tool_result blocks from a user message and match to tool_uses."""
        msg = d.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue

            tc_id = block.get("tool_use_id", "")
            result_content = block.get("content", "")
            if not isinstance(result_content, str):
                result_content = str(result_content)

            # Match to tool_use
            if tc_id not in tool_uses:
                continue

            name, inp = tool_uses[tc_id]

            # Determine if error
            explicit_error = block.get("is_error", False)
            detected_error = is_error_content(result_content)
            is_err = explicit_error or detected_error

            error_cat = classify_error(result_content) if is_err else ErrorCategory.UNKNOWN

            tool_calls.append(
                ToolCall(
                    name=name,
                    tool_call_id=tc_id,
                    input_data=inp,
                    output=result_content,
                    is_error=is_err,
                    error_category=error_cat,
                    msg_index=msg_index,
                    output_bytes=len(result_content.encode("utf-8")),
                )
            )


def _decode_project_path(escaped_name: str) -> Path | None:
    """Decode a Claude Code escaped project path.

    Claude Code escapes paths by replacing / with -.
    e.g., "-Users-tchopra-claude-projects-headroom"
    → "/Users/tchopra/claude-projects/headroom"

    Since - is ambiguous (path separator vs literal hyphen), we try
    progressively and check which decoded path actually exists.
    """
    if not escaped_name.startswith("-"):
        return None

    # Simple approach: replace all - with / and check if path exists
    simple = Path("/" + escaped_name[1:].replace("-", "/"))
    if simple.exists():
        return simple

    # Try common patterns: /Users/username/...
    parts = escaped_name[1:].split("-")
    if len(parts) < 3:
        return None

    # Build path greedily: try joining with / and check existence
    # Start with /Users/username (first 2 components are almost always correct)
    if parts[0] == "Users" and len(parts) > 2:
        base = Path(f"/{parts[0]}/{parts[1]}")
        remaining = parts[2:]
        return _greedy_path_decode(base, remaining)

    return None


def _greedy_path_decode(base: Path, parts: list[str]) -> Path | None:
    """Greedily decode remaining path parts, trying - as / first."""
    if not parts:
        return base if base.exists() else None

    # Try using / (this part is a directory component)
    slash_path = base / parts[0]
    result = _greedy_path_decode(slash_path, parts[1:])
    if result:
        return result

    # Try joining with - (this part has a literal hyphen)
    if len(parts) > 1:
        hyphen_name = f"{parts[0]}-{parts[1]}"
        hyphen_path = base / hyphen_name
        result = _greedy_path_decode(hyphen_path, parts[2:])
        if result:
            return result

    # If we've exhausted parts, check if current path exists
    return base if base.exists() else None
