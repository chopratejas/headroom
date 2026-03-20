"""Live Traffic Pattern Learner — extracts memories from proxy traffic.

Hooks into the proxy request/response pipeline to learn patterns without
any LLM calls. Rule-based extraction from traffic the proxy already sees:
- Error → Recovery patterns (tool fails → next success teaches right approach)
- Environment facts (commands that work/fail, paths, tool availability)
- Preference signals (repeated patterns, corrections)
- Architectural decisions (file references, dependency choices)

Usage:
    learner = TrafficLearner(memory_backend)
    await learner.on_request(messages, agent_type="claude")
    await learner.on_response(response, messages, agent_type="claude")

The learner is designed to be zero-config and zero-latency: it processes
patterns in the background and never blocks the proxy pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from headroom.memory.backends.local import LocalBackend

logger = logging.getLogger(__name__)


# =============================================================================
# Pattern Categories
# =============================================================================


class PatternCategory(str, Enum):
    """Categories of patterns extracted from traffic."""

    ERROR_RECOVERY = "error_recovery"  # Tool failed → next call succeeded
    ENVIRONMENT = "environment"  # Working commands, paths, tool availability
    PREFERENCE = "preference"  # Repeated choices, corrections
    ARCHITECTURE = "architecture"  # File structure, dependencies, conventions


class AgentType(str, Enum):
    """Supported coding agent types."""

    CLAUDE = "claude"
    CURSOR = "cursor"
    CODEX = "codex"
    AIDER = "aider"
    GEMINI = "gemini"
    UNKNOWN = "unknown"


# =============================================================================
# Extracted Pattern Model
# =============================================================================


@dataclass
class ExtractedPattern:
    """A pattern extracted from proxy traffic."""

    category: PatternCategory
    content: str  # Human-readable memory content
    importance: float  # 0.0 - 1.0
    evidence_count: int = 1  # How many times this pattern was observed
    entity_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


# =============================================================================
# Error Classification (reused from learn/scanner.py patterns)
# =============================================================================

_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"No such file or directory|ENOENT|FileNotFoundError|does not exist", re.I),
        "file_not_found",
    ),
    (re.compile(r"ModuleNotFoundError|ImportError|No module named", re.I), "module_not_found"),
    (re.compile(r"command not found", re.I), "command_not_found"),
    (re.compile(r"Permission denied|EACCES|EPERM|auto-denied", re.I), "permission_denied"),
    (re.compile(r"file is too large|too many lines|exceeds.*limit", re.I), "file_too_large"),
    (re.compile(r"SyntaxError|IndentationError", re.I), "syntax_error"),
    (re.compile(r"Traceback \(most recent|Exception:|Error:", re.I), "runtime_error"),
    (re.compile(r"timed? ?out|TimeoutError|deadline exceeded", re.I), "timeout"),
    (re.compile(r"exit code|non-zero|exited with", re.I), "exit_code"),
    (re.compile(r"BUILD FAILED|compilation error|compile error", re.I), "build_failure"),
]


def _classify_error(content: str) -> str | None:
    """Classify error content. Returns category or None if not an error."""
    snippet = content[:2000]
    for pattern, category in _ERROR_PATTERNS:
        if pattern.search(snippet):
            return category
    return None


def _is_error(content: str) -> bool:
    """Quick check if tool output looks like an error."""
    if not content or len(content) < 10:
        return False
    return _classify_error(content) is not None


# =============================================================================
# Tool Call Extractors
# =============================================================================

# Extract command from Bash tool calls
_COMMAND_RE = re.compile(r"^(?:source\s+\S+\s*&&\s*)?(.+)", re.I)

# Extract file paths
_FILE_PATH_RE = re.compile(r"(?:/[\w./-]+(?:\.\w+)?)")

# Extract package/module names from errors
_MODULE_RE = re.compile(r"No module named ['\"]?(\w[\w.]*)['\"]?")
_COMMAND_NF_RE = re.compile(r"(\w[\w-]*): command not found")


# =============================================================================
# Traffic Learner
# =============================================================================


class TrafficLearner:
    """Extracts learnable patterns from live proxy traffic.

    Operates entirely on rule-based heuristics — no LLM calls.
    Designed to be called from the proxy request/response path
    with minimal overhead (async, non-blocking).
    """

    def __init__(
        self,
        backend: LocalBackend | None = None,
        user_id: str = "default",
        max_history: int = 20,
        dedup_window: int = 100,
        min_evidence: int = 2,
    ) -> None:
        """Initialize the traffic learner.

        Args:
            backend: Memory backend to save patterns to. If None, patterns
                are accumulated but not persisted until a backend is set.
            user_id: Default user ID for saved memories.
            max_history: Number of recent tool calls to keep for pattern matching.
            dedup_window: Number of recent pattern hashes to track for dedup.
            min_evidence: Minimum times a pattern must be seen before saving.
        """
        self._backend = backend
        self._user_id = user_id
        self._max_history = max_history
        self._min_evidence = min_evidence

        # Recent tool call history for error→recovery matching
        self._tool_history: list[dict[str, Any]] = []

        # Pattern accumulator: hash → (pattern, count)
        self._pattern_counts: dict[str, tuple[ExtractedPattern, int]] = {}

        # Dedup: hashes of patterns already saved to DB
        self._saved_hashes: set[str] = set()
        self._dedup_window = dedup_window

        # Stats
        self._patterns_extracted = 0
        self._patterns_saved = 0
        self._requests_processed = 0

        # Background save queue
        self._save_queue: asyncio.Queue[ExtractedPattern] = asyncio.Queue(maxsize=100)
        self._save_task: asyncio.Task[None] | None = None

    # =========================================================================
    # Public API
    # =========================================================================

    def set_backend(self, backend: LocalBackend) -> None:
        """Set or update the memory backend."""
        self._backend = backend

    async def start(self) -> None:
        """Start the background save worker."""
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._save_worker())

    async def stop(self) -> None:
        """Stop the background save worker."""
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

    async def on_tool_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        is_error: bool,
        agent_type: str = "unknown",
    ) -> None:
        """Process a tool call result from proxy traffic.

        Called by the proxy after each tool_result block is processed.
        Non-blocking — patterns are queued for async persistence.

        Args:
            tool_name: Name of the tool (Bash, Read, Grep, etc.)
            tool_input: Tool input parameters
            tool_output: Tool output content
            is_error: Whether the tool call failed
            agent_type: Which agent is being proxied
        """
        self._requests_processed += 1

        entry = {
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output[:2000],  # Cap for memory
            "is_error": is_error,
            "error_category": _classify_error(tool_output) if is_error else None,
            "timestamp": time.time(),
            "agent_type": agent_type,
        }

        # Check for error→recovery pattern BEFORE adding to history
        if not is_error and self._tool_history:
            patterns = self._extract_error_recovery(entry)
            for pattern in patterns:
                await self._accumulate(pattern)

        # Extract environment patterns
        env_patterns = self._extract_environment(entry)
        for pattern in env_patterns:
            await self._accumulate(pattern)

        # Add to history (bounded)
        self._tool_history.append(entry)
        if len(self._tool_history) > self._max_history:
            self._tool_history.pop(0)

    async def on_messages(
        self,
        messages: list[dict[str, Any]],
        agent_type: str = "unknown",
    ) -> None:
        """Process message content for preference/architecture patterns.

        Called with the messages array from a proxy request.
        Extracts patterns from user corrections, assistant decisions, etc.

        Args:
            messages: The messages array from the API request
            agent_type: Which agent is being proxied
        """
        for msg in messages[-3:]:  # Only look at recent messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                content = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if not content:
                continue

            if role == "user":
                patterns = self._extract_preferences(content)
                for pattern in patterns:
                    await self._accumulate(pattern)

    def get_stats(self) -> dict[str, Any]:
        """Get learner statistics."""
        return {
            "requests_processed": self._requests_processed,
            "patterns_extracted": self._patterns_extracted,
            "patterns_saved": self._patterns_saved,
            "pending_patterns": len(self._pattern_counts),
            "history_size": len(self._tool_history),
        }

    # =========================================================================
    # Pattern Extraction
    # =========================================================================

    def _extract_error_recovery(self, success_entry: dict[str, Any]) -> list[ExtractedPattern]:
        """Extract error→recovery patterns.

        Looks backward in history for recent errors, then checks if the
        current successful call is a recovery (same tool, different params).
        """
        patterns: list[ExtractedPattern] = []
        tool_name = success_entry["tool_name"]

        # Look at recent history for matching errors
        for i in range(len(self._tool_history) - 1, max(-1, len(self._tool_history) - 6), -1):
            prev = self._tool_history[i]
            if not prev["is_error"]:
                continue

            # Same tool type — likely a retry with corrected params
            if prev["tool_name"] == tool_name:
                pattern = self._build_recovery_pattern(prev, success_entry)
                if pattern:
                    patterns.append(pattern)
                break  # Only match the most recent error

            # Bash → Bash with different command (common for env issues)
            if prev["tool_name"] == "Bash" and tool_name == "Bash":
                pattern = self._build_command_recovery(prev, success_entry)
                if pattern:
                    patterns.append(pattern)
                break

        return patterns

    def _build_recovery_pattern(
        self,
        error_entry: dict[str, Any],
        success_entry: dict[str, Any],
    ) -> ExtractedPattern | None:
        """Build a recovery pattern from an error→success pair."""
        tool = error_entry["tool_name"]
        error_cat = error_entry.get("error_category", "unknown")

        if tool == "Bash":
            return self._build_command_recovery(error_entry, success_entry)
        elif tool == "Read":
            error_path = error_entry["input"].get("file_path", "")
            success_path = success_entry["input"].get("file_path", "")
            if error_path and success_path and error_path != success_path:
                content = (
                    f"File `{error_path}` does not exist. The correct path is `{success_path}`."
                )
                return ExtractedPattern(
                    category=PatternCategory.ERROR_RECOVERY,
                    content=content,
                    importance=0.7,
                    entity_refs=[success_path],
                    metadata={"error_category": error_cat},
                )
        elif tool in ("Grep", "Glob"):
            error_pattern = error_entry["input"].get("pattern", "")
            success_pattern = success_entry["input"].get("pattern", "")
            if error_pattern != success_pattern:
                content = (
                    f"Search pattern `{error_pattern}` found no results. "
                    f"Use `{success_pattern}` instead."
                )
                return ExtractedPattern(
                    category=PatternCategory.ERROR_RECOVERY,
                    content=content,
                    importance=0.5,
                )
        return None

    def _build_command_recovery(
        self,
        error_entry: dict[str, Any],
        success_entry: dict[str, Any],
    ) -> ExtractedPattern | None:
        """Build a command recovery pattern from Bash error→success."""
        failed_cmd = error_entry["input"].get("command", "")
        success_cmd = success_entry["input"].get("command", "")
        error_cat = error_entry.get("error_category", "unknown")

        if not failed_cmd or not success_cmd or failed_cmd == success_cmd:
            return None

        # Determine importance based on error category
        importance = 0.7
        if error_cat == "command_not_found":
            importance = 0.85  # Environment setup is high-value
        elif error_cat == "module_not_found":
            importance = 0.8

        # Truncate long commands
        failed_short = failed_cmd[:200]
        success_short = success_cmd[:200]

        content = f"Command `{failed_short}` fails ({error_cat}). Use `{success_short}` instead."

        # Extract entity references
        entities: list[str] = []
        module_match = _MODULE_RE.search(error_entry["output"])
        if module_match:
            entities.append(module_match.group(1))
        cmd_match = _COMMAND_NF_RE.search(error_entry["output"])
        if cmd_match:
            entities.append(cmd_match.group(1))

        return ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content=content,
            importance=importance,
            entity_refs=entities,
            metadata={"error_category": error_cat, "failed_cmd": failed_short},
        )

    def _extract_environment(self, entry: dict[str, Any]) -> list[ExtractedPattern]:
        """Extract environment facts from tool calls."""
        patterns: list[ExtractedPattern] = []

        if entry["tool_name"] != "Bash":
            return patterns

        cmd = entry["input"].get("command", "")
        output = entry["output"]

        # Successful commands reveal working environment patterns
        if not entry["is_error"]:
            # Python/venv activation patterns
            if "activate" in cmd and "source" in cmd:
                # Extract the venv path
                venv_match = re.search(r"source\s+(\S+/activate)", cmd)
                if venv_match:
                    venv_path = venv_match.group(1)
                    patterns.append(
                        ExtractedPattern(
                            category=PatternCategory.ENVIRONMENT,
                            content=f"Python virtual environment: `source {venv_path}` before running Python tools.",
                            importance=0.8,
                            entity_refs=[venv_path],
                            metadata={"type": "venv_activation"},
                        )
                    )

            # Detect working test commands
            if "pytest" in cmd and "PASSED" in output:
                patterns.append(
                    ExtractedPattern(
                        category=PatternCategory.ENVIRONMENT,
                        content=f"Working test command: `{cmd[:200]}`",
                        importance=0.6,
                        metadata={"type": "test_command"},
                    )
                )

        return patterns

    def _extract_preferences(self, user_text: str) -> list[ExtractedPattern]:
        """Extract preference signals from user messages.

        Looks for correction patterns: "no", "don't", "instead", "use X not Y".
        """
        patterns: list[ExtractedPattern] = []

        # Negative corrections: "don't X", "stop X", "no, X"
        correction_res = [
            re.compile(r"(?:don'?t|do not|stop|never|avoid)\s+(.{10,100})", re.I),
            re.compile(r"(?:no,?\s+)(?:use|try|do)\s+(.{10,100})", re.I),
            re.compile(r"instead(?:,?\s+)(.{10,80})", re.I),
        ]

        for regex in correction_res:
            match = regex.search(user_text[:500])
            if match:
                correction = match.group(1).strip().rstrip(".")
                patterns.append(
                    ExtractedPattern(
                        category=PatternCategory.PREFERENCE,
                        content=f"User preference: {correction}",
                        importance=0.75,
                        metadata={"type": "correction", "source_text": user_text[:200]},
                    )
                )
                break  # One preference per message

        return patterns

    # =========================================================================
    # Pattern Accumulation & Persistence
    # =========================================================================

    async def _accumulate(self, pattern: ExtractedPattern) -> None:
        """Accumulate a pattern, saving when evidence threshold is met."""
        self._patterns_extracted += 1
        h = pattern.content_hash

        # Already saved — skip
        if h in self._saved_hashes:
            return

        # Accumulate evidence
        if h in self._pattern_counts:
            existing, count = self._pattern_counts[h]
            count += 1
            self._pattern_counts[h] = (existing, count)
        else:
            self._pattern_counts[h] = (pattern, 1)
            return  # First sighting — wait for more evidence

        # Check if evidence threshold met
        _, count = self._pattern_counts[h]
        if count >= self._min_evidence:
            # Ready to save
            del self._pattern_counts[h]
            self._saved_hashes.add(h)
            # Trim saved hashes to prevent unbounded growth
            if len(self._saved_hashes) > self._dedup_window:
                # Remove oldest (arbitrary, set is unordered, but prevents growth)
                self._saved_hashes.pop()

            try:
                self._save_queue.put_nowait(pattern)
            except asyncio.QueueFull:
                logger.debug("Traffic learner save queue full, dropping pattern")

    async def _save_worker(self) -> None:
        """Background worker that persists patterns to memory backend."""
        while True:
            try:
                pattern = await self._save_queue.get()
                if self._backend is None:
                    continue

                await self._backend.save_memory(
                    content=pattern.content,
                    user_id=self._user_id,
                    metadata={
                        "source": "traffic_learner",
                        "category": pattern.category.value,
                        "evidence_count": pattern.evidence_count,
                        **pattern.metadata,
                    },
                )
                self._patterns_saved += 1
                logger.debug(f"Traffic learner saved pattern: {pattern.content[:80]}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Traffic learner save failed: {e}")

    # =========================================================================
    # Convenience: Extract from Anthropic messages format
    # =========================================================================

    def extract_tool_results_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract tool_result blocks from Anthropic-format messages.

        Useful for processing the messages array to find tool calls and
        their results for pattern extraction.

        Returns list of dicts with: tool_name, input, output, is_error
        """
        results: list[dict[str, Any]] = []

        # Build tool_use_id → tool_use mapping
        tool_uses: dict[str, dict[str, Any]] = {}
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_uses[block.get("id", "")] = block

        # Find tool_results and match with tool_uses
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue

                tool_use_id = block.get("tool_use_id", "")
                tool_use = tool_uses.get(tool_use_id, {})

                # Extract output text
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = " ".join(
                        b.get("text", "")
                        for b in result_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )

                results.append(
                    {
                        "tool_name": tool_use.get("name", "unknown"),
                        "input": tool_use.get("input", {}),
                        "output": str(result_content),
                        "is_error": block.get("is_error", False) or _is_error(str(result_content)),
                    }
                )

        return results
