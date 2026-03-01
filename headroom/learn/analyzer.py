"""Failure analyzers with success correlation.

The core insight: don't just catalog failures — find what SUCCEEDED after
each failure. The diff between failed input and successful input is the
actual learning.

All analyzers work on normalized ToolCall sequences. They are tool-agnostic:
same analysis works for Claude Code, Cursor, Codex, or any agent with tool calls.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

from .models import (
    AnalysisReport,
    CommandPattern,
    Correction,
    EnvironmentFact,
    ErrorCategory,
    ProjectInfo,
    RetryPattern,
    SessionData,
    StructureNote,
    ToolCall,
)

# How many messages ahead to look for a success after a failure
_CORRECTION_WINDOW = 10


class FailureAnalyzer:
    """Runs all analyzers on tool call data and produces an AnalysisReport."""

    def analyze(self, project: ProjectInfo, sessions: list[SessionData]) -> AnalysisReport:
        all_calls = [tc for s in sessions for tc in s.tool_calls]
        failed_calls = [tc for tc in all_calls if tc.is_error]

        report = AnalysisReport(
            project=project,
            total_calls=len(all_calls),
            total_failures=len(failed_calls),
            total_sessions=len(sessions),
            waste_bytes=sum(tc.output_bytes for tc in failed_calls),
        )

        # Phase 1: Extract failure→success corrections (the core learning)
        report.corrections = _extract_corrections(sessions)

        # Phase 2: Analyze specific dimensions using corrections + raw data
        report.environment_facts = _analyze_environment(sessions)
        report.structure_notes = _analyze_structure(sessions, report.corrections)
        report.command_patterns = _analyze_commands(sessions, report.corrections)
        report.retry_patterns = _analyze_retries(sessions, report.corrections)
        report.permission_issues = _analyze_permissions(sessions)
        report.cross_session_patterns = _analyze_cross_session(sessions)

        return report


# =============================================================================
# Success Correlation: The Core Learning Primitive
# =============================================================================


def _extract_corrections(sessions: list[SessionData]) -> list[Correction]:
    """For each failure, find the next success of the same tool type.

    The pair (failed_input, success_input) is a Correction — the model
    learned something and corrected its approach.
    """
    corrections: list[Correction] = []

    for session in sessions:
        calls = session.tool_calls
        for i, tc in enumerate(calls):
            if not tc.is_error:
                continue
            # Skip sibling errors (cascades, not real failures)
            if tc.error_category == ErrorCategory.SIBLING_ERROR:
                continue
            # Look forward for a success of the same tool
            for j in range(i + 1, min(i + _CORRECTION_WINDOW, len(calls))):
                candidate = calls[j]
                if candidate.name != tc.name or candidate.is_error:
                    continue
                # Found a success — is the input meaningfully different?
                if candidate.input_data == tc.input_data:
                    continue  # Exact same input succeeded (transient error)
                corrections.append(
                    Correction(
                        tool_name=tc.name,
                        failed_input=tc.input_data,
                        success_input=candidate.input_data,
                        error_category=tc.error_category,
                        session_id=session.session_id,
                    )
                )
                break

    return corrections


# =============================================================================
# Environment Analyzer (uses corrections for python/build tool detection)
# =============================================================================


def _analyze_environment(sessions: list[SessionData]) -> list[EnvironmentFact]:
    """Detect which runtime commands work vs fail."""
    facts: list[EnvironmentFact] = []

    python_failures: Counter[str] = Counter()
    python_successes: Counter[str] = Counter()
    python_sessions: dict[str, set[str]] = defaultdict(set)

    for session in sessions:
        for tc in session.tool_calls:
            if tc.name not in ("Bash", "bash"):
                continue
            cmd = tc.input_data.get("command", "")
            if not cmd:
                continue

            if tc.is_error and tc.error_category == ErrorCategory.MODULE_NOT_FOUND:
                prefix = _extract_python_command(cmd)
                if prefix:
                    python_failures[prefix] += 1
                    python_sessions[prefix].add(session.session_id)
            elif not tc.is_error:
                prefix = _extract_python_command(cmd)
                if prefix:
                    python_successes[prefix] += 1

    if python_failures:
        wrong = sorted(python_failures.keys(), key=lambda x: -python_failures[x])
        correct = None
        for cmd, _count in python_successes.most_common():
            if cmd not in python_failures or python_successes[cmd] > python_failures[cmd] * 2:
                correct = cmd
                break
        if correct and wrong:
            total_evidence = sum(python_failures[w] for w in wrong)
            total_sessions = len(set().union(*(python_sessions[w] for w in wrong)))
            facts.append(
                EnvironmentFact(
                    category="python",
                    correct_command=correct,
                    wrong_commands=wrong[:5],
                    evidence_count=total_evidence,
                    sessions_seen=total_sessions,
                )
            )

    return facts


# =============================================================================
# Structure Analyzer (uses corrections to learn correct paths)
# =============================================================================


def _analyze_structure(
    sessions: list[SessionData], corrections: list[Correction]
) -> list[StructureNote]:
    """Find file structure issues and learn correct paths from corrections."""
    notes: list[StructureNote] = []

    # 1. Path corrections: wrong path → correct path (from success correlation)
    path_corrections: dict[str, Counter[str]] = defaultdict(Counter)
    for c in corrections:
        if c.error_category != ErrorCategory.FILE_NOT_FOUND:
            continue
        # Extract file paths from Read tool or from Bash commands (Codex uses shell for everything)
        if c.tool_name in ("Read", "read"):
            failed_path = c.failed_input.get("file_path", "")
            success_path = c.success_input.get("file_path", "")
        elif c.tool_name in ("Bash", "bash"):
            failed_path = _extract_path_from_command(c.failed_input.get("command", ""))
            success_path = _extract_path_from_command(c.success_input.get("command", ""))
        else:
            continue
        if failed_path and success_path and failed_path != success_path:
            path_corrections[failed_path][success_path] += 1

    for wrong_path, correct_paths in path_corrections.items():
        best_correct, count = correct_paths.most_common(1)[0]
        # Make paths relative to project for readability
        wrong_short = _shorten_path(wrong_path)
        correct_short = _shorten_path(best_correct)
        notes.append(
            StructureNote(
                category="path_correction",
                path=wrong_short,
                correct_path=correct_short,
                note=f"Not at `{wrong_short}` → actually at `{correct_short}`",
                evidence_count=count,
            )
        )

    # 2. Grep scope corrections: narrow path → broader path worked
    scope_corrections: dict[str, Counter[str]] = defaultdict(Counter)
    for c in corrections:
        if c.tool_name not in ("Grep", "grep"):
            continue
        failed_path = c.failed_input.get("path", "")
        success_path = c.success_input.get("path", "")
        if failed_path and success_path and failed_path != success_path:
            scope_corrections[_shorten_path(failed_path)][_shorten_path(success_path)] += 1

    for wrong_scope, correct_scopes in scope_corrections.items():
        best_scope, count = correct_scopes.most_common(1)[0]
        notes.append(
            StructureNote(
                category="search_scope",
                path=wrong_scope,
                correct_path=best_scope,
                note=f"Grep fails at `{wrong_scope}` → use `{best_scope}` instead",
                evidence_count=count,
            )
        )

    # 3. Large files (from raw failures, no correction needed)
    large_files: Counter[str] = Counter()
    large_sessions: dict[str, set[str]] = defaultdict(set)
    for session in sessions:
        for tc in session.tool_calls:
            if (
                tc.name in ("Read", "read")
                and tc.is_error
                and tc.error_category == ErrorCategory.FILE_TOO_LARGE
            ):
                path = tc.input_data.get("file_path", "")
                if path:
                    short = _shorten_path(path)
                    large_files[short] += 1
                    large_sessions[short].add(session.session_id)

    for path, count in large_files.most_common(10):
        if count < 2:
            break
        notes.append(
            StructureNote(
                category="large_file",
                path=path,
                note=f"Too large for full read — always use offset/limit ({count} failures, {len(large_sessions[path])} sessions)",
                evidence_count=count,
                sessions_seen=len(large_sessions[path]),
            )
        )

    # 4. Persistent missing paths (no correction found — file truly doesn't exist)
    missing_no_correction: Counter[str] = Counter()
    missing_sessions: dict[str, set[str]] = defaultdict(set)
    corrected_paths = set(path_corrections.keys())
    for session in sessions:
        for tc in session.tool_calls:
            if (
                tc.name in ("Read", "read")
                and tc.is_error
                and tc.error_category == ErrorCategory.FILE_NOT_FOUND
            ):
                path = tc.input_data.get("file_path", "")
                if path and path not in corrected_paths:
                    missing_no_correction[path] += 1
                    missing_sessions[path].add(session.session_id)

    for path, count in missing_no_correction.most_common(10):
        if count < 2:
            break
        short = _shorten_path(path)
        notes.append(
            StructureNote(
                category="missing_path",
                path=short,
                note=f"Does not exist ({count} attempts, {len(missing_sessions[path])} sessions)",
                evidence_count=count,
                sessions_seen=len(missing_sessions[path]),
            )
        )

    return notes


# =============================================================================
# Command Pattern Analyzer (uses corrections to learn command patterns)
# =============================================================================


def _analyze_commands(
    sessions: list[SessionData], corrections: list[Correction]
) -> list[CommandPattern]:
    """Learn specific command patterns from Bash failure→success corrections."""
    patterns: list[CommandPattern] = []

    # Analyze Bash corrections
    bash_corrections = [c for c in corrections if c.tool_name in ("Bash", "bash")]

    # Group by error category to find patterns
    by_category: dict[ErrorCategory, list[Correction]] = defaultdict(list)
    for c in bash_corrections:
        by_category[c.error_category].append(c)

    # User-rejected commands: model should suggest, not execute
    rejected = by_category.get(ErrorCategory.USER_REJECTED, [])
    if rejected:
        # Find the most commonly rejected command patterns
        rejected_cmds: Counter[str] = Counter()
        for c in rejected:
            cmd = c.failed_input.get("command", "")
            base = _extract_command_signature(cmd)
            if base:
                rejected_cmds[base] += 1

        for cmd_sig, count in rejected_cmds.most_common(5):
            if count < 2:
                break
            patterns.append(
                CommandPattern(
                    category="user_prefers_manual",
                    wrong_pattern=f"Executing: {cmd_sig}",
                    correct_pattern="Show the command to the user and let them run it",
                    explanation=f"User rejected this command {count} times — they prefer to run it themselves",
                    evidence_count=count,
                    sessions_seen=len(
                        {
                            c.session_id
                            for c in rejected
                            if _extract_command_signature(c.failed_input.get("command", ""))
                            == cmd_sig
                        }
                    ),
                )
            )

    # Build failures: learn what command form works
    build_fails = by_category.get(ErrorCategory.BUILD_FAILURE, [])
    for c in build_fails:
        failed_cmd = c.failed_input.get("command", "")
        success_cmd = c.success_input.get("command", "")
        if failed_cmd and success_cmd:
            patterns.append(
                CommandPattern(
                    category="build",
                    wrong_pattern=_extract_command_signature(failed_cmd),
                    correct_pattern=_extract_command_signature(success_cmd),
                    explanation="Build failed with first form, succeeded with second",
                    evidence_count=1,
                )
            )

    # Module not found: learn correct python invocation
    module_fails = by_category.get(ErrorCategory.MODULE_NOT_FOUND, [])
    if module_fails:
        wrong_pythons: Counter[str] = Counter()
        correct_pythons: Counter[str] = Counter()
        for c in module_fails:
            wp = _extract_python_command(c.failed_input.get("command", ""))
            cp = _extract_python_command(c.success_input.get("command", ""))
            if wp:
                wrong_pythons[wp] += 1
            if cp:
                correct_pythons[cp] += 1

        if wrong_pythons and correct_pythons:
            wrong = wrong_pythons.most_common(1)[0][0]
            correct = correct_pythons.most_common(1)[0][0]
            if wrong != correct:
                patterns.append(
                    CommandPattern(
                        category="python_runtime",
                        wrong_pattern=f"`{wrong}` (modules not available)",
                        correct_pattern=f"`{correct}` (has project dependencies)",
                        explanation=f"Using `{wrong}` causes ModuleNotFoundError — use `{correct}` which has the project's venv",
                        evidence_count=sum(wrong_pythons.values()),
                    )
                )

    # Deduplicate patterns
    seen = set()
    unique = []
    for p in patterns:
        key = (p.category, p.wrong_pattern[:50])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# =============================================================================
# Retry Analyzer (uses corrections to provide specific suggestions)
# =============================================================================


def _analyze_retries(
    sessions: list[SessionData], corrections: list[Correction]
) -> list[RetryPattern]:
    """Find stubborn retries with specific fix suggestions from corrections."""
    patterns: list[RetryPattern] = []

    # Build a correction lookup: (tool, error_category) → list of corrections
    correction_lookup: dict[tuple[str, str], list[Correction]] = defaultdict(list)
    for c in corrections:
        correction_lookup[(c.tool_name, c.error_category.value)].append(c)

    # Find retry streaks
    pattern_counter: Counter[tuple[str, str, str]] = Counter()
    max_retries: dict[tuple[str, str, str], int] = {}

    for session in sessions:
        streak: dict[str, list[ToolCall]] = defaultdict(list)
        for tc in session.tool_calls:
            key = f"{tc.name}:{tc.error_category.value}"
            if tc.is_error:
                streak[key].append(tc)
            else:
                if len(streak.get(key, [])) >= 3:
                    calls = streak[key]
                    pk = (tc.name, calls[0].error_category.value, calls[0].input_summary[:50])
                    pattern_counter[pk] += 1
                    max_retries[pk] = max(max_retries.get(pk, 0), len(calls))
                streak[key] = []
        for _key, calls in streak.items():
            if len(calls) >= 3:
                pk = (calls[0].name, calls[0].error_category.value, calls[0].input_summary[:50])
                pattern_counter[pk] += 1
                max_retries[pk] = max(max_retries.get(pk, 0), len(calls))

    for (tool, err_cat, input_key), count in pattern_counter.most_common(10):
        max_r = max_retries.get((tool, err_cat, input_key), 3)

        # Try to get a SPECIFIC suggestion from corrections
        relevant_corrections = correction_lookup.get((tool, err_cat), [])
        suggestion = _build_specific_suggestion(tool, err_cat, relevant_corrections)

        patterns.append(
            RetryPattern(
                tool_name=tool,
                error_category=ErrorCategory(err_cat),
                description=f"{tool} failing with {err_cat}: {input_key}",
                max_retries_seen=max_r,
                suggestion=suggestion,
                evidence_count=count,
            )
        )

    return patterns


def _build_specific_suggestion(
    tool: str, error_category: str, corrections: list[Correction]
) -> str:
    """Build a specific suggestion from actual corrections, not generic advice."""
    if not corrections:
        # No corrections available — use tool+error specific defaults
        return _default_suggestion(tool, error_category)

    # Summarize what corrections tell us
    if tool in ("Read", "read") and error_category == "file_not_found":
        examples = []
        for c in corrections[:3]:
            wrong = _shorten_path(c.failed_input.get("file_path", ""))
            right = _shorten_path(c.success_input.get("file_path", ""))
            if wrong and right:
                examples.append(f"`{wrong}` → `{right}`")
        if examples:
            return "Use Glob to discover actual path. Known corrections: " + "; ".join(examples)

    if tool in ("Grep", "grep"):
        # Summarize scope corrections
        scopes = set()
        for c in corrections[:5]:
            right_path = c.success_input.get("path", "")
            if right_path:
                scopes.add(_shorten_path(right_path))
        if scopes:
            return f"Scope searches to: {', '.join(sorted(scopes)[:3])}"

    if tool in ("Bash", "bash") and error_category == "user_rejected":
        return "User prefers to run this command themselves. Show the command, don't execute it."

    if tool in ("Bash", "bash") and error_category == "module_not_found":
        correct_cmds = set()
        for c in corrections[:5]:
            prefix = _extract_python_command(c.success_input.get("command", ""))
            if prefix:
                correct_cmds.add(prefix)
        if correct_cmds:
            return f"Use {' or '.join(sorted(correct_cmds))} (has project dependencies)"

    # Fallback: show one correction example
    c = corrections[0]
    return f"What worked: {c.success_summary[:80]}"


def _default_suggestion(tool: str, error_category: str) -> str:
    """Fallback when no corrections are available."""
    defaults = {
        (
            "Glob",
            "no_matches",
        ): "Broaden pattern to **/*.ext or use ls to explore directory structure",
        ("Grep", "no_matches"): "Try case-insensitive (-i) or broaden search scope",
        ("Grep", "timeout"): "Scope Grep to a specific subdirectory — the full repo is too large",
        ("Read", "file_not_found"): "Use Glob to discover the file path before Read",
        ("Read", "file_too_large"): "Use offset/limit parameters for this file",
        ("Bash", "module_not_found"): "Use the project's virtualenv Python",
        ("Bash", "permission_denied"): "Do not retry — try a different approach",
        ("Bash", "command_not_found"): "Verify tool is installed: which <tool>",
        ("Bash", "user_rejected"): "User does not want this command executed. Show it instead.",
        ("Edit", "unknown"): "If old_string has multiple matches, add more surrounding context",
    }
    return defaults.get((tool, error_category), "Try an alternative approach after 2 failures")


# =============================================================================
# Permission Analyzer
# =============================================================================


def _analyze_permissions(sessions: list[SessionData]) -> list[str]:
    """Find commands repeatedly denied — with specific advice."""
    denied: Counter[str] = Counter()
    denied_cmds: dict[str, str] = {}  # key → full command example

    for session in sessions:
        for tc in session.tool_calls:
            if not tc.is_error:
                continue
            if tc.error_category not in (
                ErrorCategory.PERMISSION_DENIED,
                ErrorCategory.USER_REJECTED,
            ):
                continue
            sig = _extract_command_signature(
                tc.input_data.get("command", "")
                if tc.name in ("Bash", "bash")
                else tc.input_summary
            )
            key = f"{tc.name}: {sig}"
            denied[key] += 1
            if key not in denied_cmds:
                denied_cmds[key] = tc.input_summary[:80]

    results = []
    for key, count in denied.most_common(10):
        if count < 3:
            break
        results.append(
            f"{key} — denied {count} times. Show the command to the user instead of executing it."
        )
    return results


# =============================================================================
# Cross-Session Analyzer
# =============================================================================


def _analyze_cross_session(sessions: list[SessionData]) -> list[str]:
    """Find failure patterns that repeat across 3+ sessions."""
    pattern_sessions: dict[str, set[str]] = defaultdict(set)

    for session in sessions:
        for tc in session.tool_calls:
            if not tc.is_error or tc.error_category == ErrorCategory.SIBLING_ERROR:
                continue
            key = f"{tc.name}|{tc.error_category.value}|{tc.input_summary[:60]}"
            pattern_sessions[key].add(session.session_id)

    cross_session = []
    for key, session_ids in sorted(pattern_sessions.items(), key=lambda x: -len(x[1])):
        if len(session_ids) < 3:
            continue
        parts = key.split("|", 2)
        tool, err, inp = parts[0], parts[1], parts[2] if len(parts) > 2 else "?"
        cross_session.append(f"{tool} {err}: {inp} (across {len(session_ids)} sessions)")
        if len(cross_session) >= 15:
            break

    return cross_session


# =============================================================================
# Helpers
# =============================================================================


_PYTHON_CMD_RE = re.compile(
    r"^((?:source\s+\S+\s*&&\s*)?(?:\.venv/bin/)?(?:python3?|uv run python|uv run|/opt/nflx/python))"
    r"(?:\s|$|-)"  # Must be followed by space, end, or dash (python3 -c, python -)
)


def _extract_python_command(cmd: str) -> str | None:
    """Extract the python invocation prefix from a command."""
    cmd = cmd.strip()
    if "&&" in cmd:
        for part in cmd.split("&&"):
            result = _extract_python_command(part.strip())
            if result:
                return result
        return None
    m = _PYTHON_CMD_RE.match(cmd)
    return m.group(1) if m else None


def _extract_command_signature(cmd: str) -> str:
    """Extract a normalizable command signature (first ~60 chars, no args)."""
    cmd = cmd.strip()
    # Truncate at first newline
    if "\n" in cmd:
        cmd = cmd.split("\n")[0]
    return cmd[:60]


def _extract_path_from_command(cmd: str) -> str:
    """Extract a file path from a shell command (cat, sed, head, etc.)."""
    cmd = cmd.strip()
    # Look for path-like tokens (containing / and ending in a file extension)
    path_re = re.compile(r"""(?:^|[\s'"])(/[^\s'"]+\.\w+)""")
    match = path_re.search(cmd)
    if match:
        return match.group(1)
    # Also try: last token that looks like a path
    tokens = cmd.split()
    for token in reversed(tokens):
        if "/" in token and not token.startswith("-"):
            return token.strip("'\"")
    return ""


def _shorten_path(path: str) -> str:
    """Make a path relative to home for readability."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path
