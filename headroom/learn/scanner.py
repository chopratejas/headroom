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
    SessionEvent,
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

            # Decode project path from escaped directory name. Fall back to a
            # simple slash replacement for display if we can't recover the real
            # on-disk path from the filesystem.
            project_path = _decode_project_path(entry.name) or Path(
                "/" + entry.name[1:].replace("-", "/")
            )

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
        events: list[SessionEvent] = []
        total_input_tokens = 0
        total_output_tokens = 0
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
                    ts = d.get("timestamp", None)

                    if line_type == "assistant":
                        self._extract_tool_uses(d, tool_uses)
                        # Extract token usage
                        usage = d.get("message", {}).get("usage", {})
                        total_input_tokens += usage.get("input_tokens", 0)
                        total_input_tokens += usage.get("cache_read_input_tokens", 0)
                        total_input_tokens += usage.get("cache_creation_input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)
                    elif line_type == "user":
                        self._extract_tool_results(d, tool_uses, tool_calls, events, msg_index, ts)
                        self._extract_user_events(d, events, msg_index, ts)

        except (OSError, UnicodeDecodeError) as e:
            logger.debug("Failed to read %s: %s", jsonl_path, e)
            return None

        # Also wrap tool_calls as events for unified access
        for tc in tool_calls:
            if not any(e.type == "tool_call" and e.tool_call is tc for e in events):
                events.append(SessionEvent(type="tool_call", msg_index=tc.msg_index, tool_call=tc))
        events.sort(key=lambda e: e.msg_index)

        return SessionData(
            session_id=session_id,
            tool_calls=tool_calls,
            events=events,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

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
        events: list[SessionEvent],
        msg_index: int,
        timestamp: str | None = None,
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

            tc = ToolCall(
                name=name,
                tool_call_id=tc_id,
                input_data=inp,
                output=result_content,
                is_error=is_err,
                error_category=error_cat,
                msg_index=msg_index,
                output_bytes=len(result_content.encode("utf-8")),
            )
            tool_calls.append(tc)
            events.append(
                SessionEvent(
                    type="tool_call", msg_index=msg_index, timestamp=timestamp, tool_call=tc
                )
            )

            # Extract subagent summary from toolUseResult metadata
            if name in ("Agent", "agent"):
                tool_result_meta = d.get("toolUseResult", {})
                if isinstance(tool_result_meta, dict):
                    events.append(
                        SessionEvent(
                            type="agent_summary",
                            msg_index=msg_index,
                            timestamp=timestamp,
                            agent_id=tool_result_meta.get("agentId", ""),
                            agent_tool_count=tool_result_meta.get("totalToolUseCount", 0),
                            agent_tokens=tool_result_meta.get("totalTokens", 0),
                            agent_duration_ms=tool_result_meta.get("totalDurationMs", 0),
                            agent_prompt=tool_result_meta.get("prompt", "")[:200],
                        )
                    )

    def _extract_user_events(
        self,
        d: dict,
        events: list[SessionEvent],
        msg_index: int,
        timestamp: str | None = None,
    ) -> None:
        """Extract user text messages and interruptions from a user line."""
        msg = d.get("message", {})
        content = msg.get("content", "")

        # Human text messages have content as a string, not a list
        if isinstance(content, str) and content.strip():
            events.append(
                SessionEvent(
                    type="user_message",
                    msg_index=msg_index,
                    timestamp=timestamp,
                    text=content[:500],
                )
            )
            return

        # Check for interruptions in list-format content
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if "[Request interrupted by user" in text:
                        events.append(
                            SessionEvent(
                                type="interruption",
                                msg_index=msg_index,
                                timestamp=timestamp,
                                text=text[:200],
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
    """Greedily decode remaining path parts using real child directories.

    Claude's project directory escaping is ambiguous: both ``/`` and literal
    punctuation such as ``-`` or ``.`` may show up as ``-`` in the escaped
    directory name. Instead of guessing every separator combination, walk the
    actual filesystem and match each child directory against the remaining
    escaped tokens.
    """
    if not parts:
        return base if base.exists() else None

    if not base.exists() or not base.is_dir():
        return None

    try:
        children = sorted(child for child in base.iterdir() if child.is_dir())
    except OSError:
        return None

    for child in children:
        for tokenization in _component_tokenizations(child.name):
            n_tokens = len(tokenization)
            if parts[:n_tokens] != tokenization:
                continue

            result = _greedy_path_decode(child, parts[n_tokens:])
            if result:
                return result

    return None


def _component_tokenizations(component: str) -> list[list[str]]:
    """Return possible escaped token sequences for a real path component."""
    tokenizations: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(tokens: list[str]) -> None:
        key = tuple(tokens)
        if tokens and key not in seen:
            seen.add(key)
            tokenizations.append(tokens)

    add([component])

    for separator in ("-", ".", None):
        if separator is None:
            tokens = [token for token in re.split(r"[-.]", component) if token]
        else:
            tokens = [token for token in component.split(separator) if token]
        add(tokens)

    # Claude's flattened encoding can turn a leading "." in hidden directory
    # names into an empty token followed by the remaining component tokens.
    if component.startswith(".") and len(component) > 1:
        hidden_component = component[1:]
        add(["", hidden_component])
        for separator in ("-", ".", None):
            if separator is None:
                tokens = [token for token in re.split(r"[-.]", hidden_component) if token]
            else:
                tokens = [token for token in hidden_component.split(separator) if token]
            add(["", *tokens])

    return tokenizations


# =============================================================================
# Codex Scanner (OpenAI Codex CLI)
# =============================================================================


class CodexScanner(ConversationScanner):
    """Reads OpenAI Codex CLI session logs from ~/.codex/sessions/.

    Codex stores sessions as JSON files with:
    - session.id, session.timestamp, session.instructions
    - items[]: array of message/function_call/function_call_output/reasoning objects

    function_call items have: name, call_id, arguments (JSON string)
    function_call_output items have: call_id, output (string or JSON string)
    """

    def __init__(self, codex_dir: Path | None = None):
        self.codex_dir = codex_dir or Path.home() / ".codex"
        self.sessions_dir = self.codex_dir / "sessions"

    def _iter_session_files(self, root: Path | None = None) -> list[Path]:
        """Return all known Codex session files, including nested rollouts."""
        search_root = root or self.sessions_dir
        session_files = list(search_root.rglob("*.json")) + list(search_root.rglob("*.jsonl"))
        return sorted(path for path in session_files if path.is_file())

    def discover_projects(self) -> list[ProjectInfo]:
        """Codex doesn't organize by project — return a single 'codex' project.

        Codex sessions aren't scoped to projects. We treat all sessions as one
        project and use the cwd from session data (if available) to group later.
        """
        if not self.sessions_dir.exists():
            return []

        session_files = self._iter_session_files()
        if not session_files:
            return []

        # Check for global AGENTS.md
        agents_md = self.codex_dir / "AGENTS.md"
        instructions_md = self.codex_dir / "instructions.md"

        return [
            ProjectInfo(
                name="codex",
                project_path=Path.cwd(),  # Codex doesn't track project paths
                data_path=self.sessions_dir,
                context_file=agents_md if agents_md.exists() else None,
                memory_file=instructions_md if instructions_md.exists() else None,
            )
        ]

    def scan_project(self, project: ProjectInfo) -> list[SessionData]:
        """Scan all Codex session JSON files."""
        sessions = []
        for json_path in self._iter_session_files(project.data_path):
            session = self._scan_session(json_path)
            if session and session.tool_calls:
                sessions.append(session)
        return sessions

    def _scan_session(self, json_path: Path) -> SessionData | None:
        """Parse a single Codex session file."""
        if json_path.suffix == ".jsonl":
            return self._scan_jsonl_session(json_path)
        return self._scan_json_session(json_path)

    def _scan_json_session(self, json_path: Path) -> SessionData | None:
        """Parse a single Codex session file."""
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Failed to read Codex session %s: %s", json_path, e)
            return None

        session_info = data.get("session", {})
        session_id = session_info.get("id", json_path.stem)
        items = data.get("items", [])

        if not items:
            return None

        # Build call_id → (name, input) map from function_call items
        func_calls: dict[str, tuple[str, dict]] = {}
        tool_calls: list[ToolCall] = []
        msg_index = 0

        for item in items:
            msg_index += 1
            item_type = item.get("type", "")

            if item_type == "function_call":
                call_id = item.get("call_id", "")
                name = item.get("name", "")
                # Parse arguments (JSON string or list)
                raw_args = item.get("arguments", "")
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    parsed = raw_args
                else:
                    parsed = {"raw": str(raw_args)}

                # Codex uses "shell" as the tool name with command in args
                # Normalize: extract command for consistency with other scanners
                if name == "shell" and "command" in parsed:
                    cmd = parsed["command"]
                    if isinstance(cmd, list):
                        # Codex passes commands as ["bash", "-lc", "actual command"]
                        parsed["command"] = cmd[-1] if cmd else ""
                    name = "Bash"  # Normalize to match Claude Code tool names

                if call_id:
                    func_calls[call_id] = (name, parsed)

            elif item_type == "function_call_output":
                call_id = item.get("call_id", "")
                output_raw = item.get("output", "")

                if call_id not in func_calls:
                    continue

                name, inp = func_calls[call_id]

                # Output may be JSON string with "output" field
                if isinstance(output_raw, str):
                    try:
                        parsed_out = json.loads(output_raw)
                        if isinstance(parsed_out, dict) and "output" in parsed_out:
                            result_content = str(parsed_out["output"])
                        else:
                            result_content = output_raw
                    except (json.JSONDecodeError, TypeError):
                        result_content = output_raw
                else:
                    result_content = str(output_raw)

                is_err = is_error_content(result_content)
                error_cat = classify_error(result_content) if is_err else ErrorCategory.UNKNOWN

                tool_calls.append(
                    ToolCall(
                        name=name,
                        tool_call_id=call_id,
                        input_data=inp,
                        output=result_content,
                        is_error=is_err,
                        error_category=error_cat,
                        msg_index=msg_index,
                        output_bytes=len(result_content.encode("utf-8")),
                    )
                )

        return SessionData(session_id=session_id, tool_calls=tool_calls)

    def _scan_jsonl_session(self, jsonl_path: Path) -> SessionData | None:
        """Parse a modern Codex rollout session stored as JSONL."""
        session_id = jsonl_path.stem
        func_calls: dict[str, tuple[str, dict]] = {}
        tool_calls: list[ToolCall] = []
        msg_index = 0

        try:
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if entry.get("type") == "session_meta":
                        payload = entry.get("payload", {})
                        if isinstance(payload, dict):
                            session_id = payload.get("id", session_id)
                        continue

                    if entry.get("type") != "response_item":
                        continue

                    payload = entry.get("payload", {})
                    if not isinstance(payload, dict):
                        continue

                    msg_index += 1
                    item_type = payload.get("type", "")

                    if item_type in ("function_call", "custom_tool_call"):
                        call_id = payload.get("call_id", "")
                        name = payload.get("name", "")
                        parsed = self._parse_codex_arguments(payload)
                        name, parsed = self._normalize_codex_tool(name, parsed)
                        if call_id and name:
                            func_calls[call_id] = (name, parsed)
                        continue

                    if item_type not in ("function_call_output", "custom_tool_call_output"):
                        continue

                    call_id = payload.get("call_id", "")
                    if call_id not in func_calls:
                        continue

                    name, inp = func_calls[call_id]
                    result_content = self._parse_codex_output(payload.get("output", ""))
                    is_err = is_error_content(result_content)
                    error_cat = classify_error(result_content) if is_err else ErrorCategory.UNKNOWN

                    tool_calls.append(
                        ToolCall(
                            name=name,
                            tool_call_id=call_id,
                            input_data=inp,
                            output=result_content,
                            is_error=is_err,
                            error_category=error_cat,
                            msg_index=msg_index,
                            output_bytes=len(result_content.encode("utf-8")),
                        )
                    )

        except OSError as e:
            logger.debug("Failed to read Codex session %s: %s", jsonl_path, e)
            return None

        if not tool_calls:
            return None

        return SessionData(session_id=session_id, tool_calls=tool_calls)

    def _parse_codex_arguments(self, payload: dict) -> dict:
        """Parse arguments for either legacy or rollout Codex tool calls."""
        raw_args = payload.get("arguments", payload.get("input", ""))
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                return parsed if isinstance(parsed, dict) else {"raw": raw_args}
            except (json.JSONDecodeError, TypeError):
                return {"raw": raw_args}
        if isinstance(raw_args, dict):
            return raw_args
        return {"raw": str(raw_args)}

    def _normalize_codex_tool(self, name: str, parsed: dict) -> tuple[str, dict]:
        """Normalize modern Codex tool names to the cross-agent schema."""
        if name == "shell" and "command" in parsed:
            cmd = parsed["command"]
            if isinstance(cmd, list):
                parsed["command"] = cmd[-1] if cmd else ""
            return "Bash", parsed

        if name == "exec_command" and "cmd" in parsed:
            parsed = dict(parsed)
            parsed["command"] = parsed.get("cmd", "")
            return "Bash", parsed

        return name, parsed

    def _parse_codex_output(self, output_raw: object) -> str:
        """Parse tool output from Codex rollout records."""
        if isinstance(output_raw, str):
            try:
                parsed_out = json.loads(output_raw)
            except (json.JSONDecodeError, TypeError):
                return output_raw

            if isinstance(parsed_out, dict):
                if "output" in parsed_out:
                    return str(parsed_out["output"])
                return json.dumps(parsed_out)
            return output_raw

        return str(output_raw)
