"""Tests for failure analyzers — generic tool call pattern recognition."""

from pathlib import Path

from headroom.learn.analyzer import FailureAnalyzer
from headroom.learn.models import (
    ErrorCategory,
    ProjectInfo,
    SessionData,
    ToolCall,
)


def _project() -> ProjectInfo:
    return ProjectInfo(
        name="test-project",
        project_path=Path("/tmp/test-project"),
        data_path=Path("/tmp/test-data"),
    )


def _tc(
    name: str = "Bash",
    input_data: dict | None = None,
    output: str = "ok",
    is_error: bool = False,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN,
    msg_index: int = 0,
) -> ToolCall:
    return ToolCall(
        name=name,
        tool_call_id=f"tc_{msg_index}",
        input_data=input_data or {},
        output=output,
        is_error=is_error,
        error_category=error_category,
        msg_index=msg_index,
        output_bytes=len(output),
    )


class TestAnalyzerBasics:
    def test_empty_sessions(self):
        analyzer = FailureAnalyzer()
        report = analyzer.analyze(_project(), [])
        assert report.total_calls == 0
        assert report.total_failures == 0
        assert report.failure_rate == 0.0

    def test_no_failures(self):
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[_tc(msg_index=i) for i in range(10)],
            )
        ]
        report = analyzer.analyze(_project(), sessions)
        assert report.total_calls == 10
        assert report.total_failures == 0

    def test_basic_failure_counting(self):
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(msg_index=0),
                    _tc(msg_index=1, is_error=True, output="Error: something broke"),
                    _tc(msg_index=2),
                ],
            )
        ]
        report = analyzer.analyze(_project(), sessions)
        assert report.total_calls == 3
        assert report.total_failures == 1


class TestEnvironmentAnalyzer:
    def test_detects_wrong_python(self):
        """Module not found with python3 + successes with uv run → learn correct command."""
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    # Failures with python3
                    _tc(
                        name="Bash",
                        input_data={"command": "python3 -c 'import mylib'"},
                        output="ModuleNotFoundError: No module named 'mylib'",
                        is_error=True,
                        error_category=ErrorCategory.MODULE_NOT_FOUND,
                        msg_index=0,
                    ),
                    _tc(
                        name="Bash",
                        input_data={"command": "python3 -c 'import mylib'"},
                        output="ModuleNotFoundError",
                        is_error=True,
                        error_category=ErrorCategory.MODULE_NOT_FOUND,
                        msg_index=1,
                    ),
                    # Success with uv run
                    _tc(
                        name="Bash",
                        input_data={"command": "uv run python -c 'import mylib'"},
                        output="ok",
                        msg_index=2,
                    ),
                    _tc(
                        name="Bash",
                        input_data={"command": "uv run python -c 'import mylib'"},
                        output="ok",
                        msg_index=3,
                    ),
                    _tc(
                        name="Bash",
                        input_data={"command": "uv run python -c 'import mylib'"},
                        output="ok",
                        msg_index=4,
                    ),
                ],
            )
        ]
        report = analyzer.analyze(_project(), sessions)
        assert len(report.environment_facts) >= 1
        fact = report.environment_facts[0]
        assert fact.category == "python"
        assert "uv run" in fact.correct_command
        assert "python3" in fact.wrong_commands


class TestStructureAnalyzer:
    def test_detects_missing_paths(self):
        """Files that repeatedly fail Read → learned as missing."""
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(
                        name="Read",
                        input_data={"file_path": "/src/missing.py"},
                        output="No such file",
                        is_error=True,
                        error_category=ErrorCategory.FILE_NOT_FOUND,
                        msg_index=0,
                    ),
                ],
            ),
            SessionData(
                session_id="s2",
                tool_calls=[
                    _tc(
                        name="Read",
                        input_data={"file_path": "/src/missing.py"},
                        output="No such file",
                        is_error=True,
                        error_category=ErrorCategory.FILE_NOT_FOUND,
                        msg_index=0,
                    ),
                ],
            ),
        ]
        report = analyzer.analyze(_project(), sessions)
        missing = [n for n in report.structure_notes if n.category == "missing_path"]
        assert len(missing) >= 1
        assert "/src/missing.py" in missing[0].path

    def test_detects_large_files(self):
        """Files that repeatedly trigger too-large errors → learned."""
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(
                        name="Read",
                        input_data={"file_path": "/src/huge.py"},
                        output="file is too large",
                        is_error=True,
                        error_category=ErrorCategory.FILE_TOO_LARGE,
                        msg_index=0,
                    ),
                    _tc(
                        name="Read",
                        input_data={"file_path": "/src/huge.py"},
                        output="file is too large",
                        is_error=True,
                        error_category=ErrorCategory.FILE_TOO_LARGE,
                        msg_index=1,
                    ),
                ],
            ),
        ]
        report = analyzer.analyze(_project(), sessions)
        large = [n for n in report.structure_notes if n.category == "large_file"]
        assert len(large) >= 1
        assert "/src/huge.py" in large[0].path

    def test_single_occurrence_not_reported(self):
        """A single file_not_found shouldn't be reported (might be transient)."""
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(
                        name="Read",
                        input_data={"file_path": "/src/one_time.py"},
                        output="No such file",
                        is_error=True,
                        error_category=ErrorCategory.FILE_NOT_FOUND,
                        msg_index=0,
                    ),
                ],
            )
        ]
        report = analyzer.analyze(_project(), sessions)
        missing = [n for n in report.structure_notes if n.category == "missing_path"]
        assert len(missing) == 0


class TestRetryAnalyzer:
    def test_detects_stubborn_retries(self):
        """Same tool failing 3+ times in a row → retry pattern."""
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(
                        name="Bash",
                        input_data={"command": "mkdir -p /x"},
                        output="auto-denied",
                        is_error=True,
                        error_category=ErrorCategory.PERMISSION_DENIED,
                        msg_index=i,
                    )
                    for i in range(5)
                ],
            )
        ]
        report = analyzer.analyze(_project(), sessions)
        assert len(report.retry_patterns) >= 1
        pattern = report.retry_patterns[0]
        assert pattern.tool_name == "Bash"
        assert pattern.max_retries_seen >= 5

    def test_two_failures_not_stubborn(self):
        """Only 2 failures shouldn't trigger a retry pattern (threshold is 3)."""
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(
                        name="Glob",
                        output="No matches",
                        is_error=True,
                        error_category=ErrorCategory.NO_MATCHES,
                        msg_index=0,
                    ),
                    _tc(
                        name="Glob",
                        output="No matches",
                        is_error=True,
                        error_category=ErrorCategory.NO_MATCHES,
                        msg_index=1,
                    ),
                    _tc(name="Glob", output="found.py", msg_index=2),  # Success breaks streak
                ],
            )
        ]
        report = analyzer.analyze(_project(), sessions)
        assert len(report.retry_patterns) == 0


class TestCrossSessionAnalyzer:
    def test_cross_session_pattern(self):
        """Same failure in 3+ sessions → cross-session pattern."""
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id=f"s{i}",
                tool_calls=[
                    _tc(
                        name="Read",
                        input_data={"file_path": "/docs/RESEARCH.md"},
                        output="No such file",
                        is_error=True,
                        error_category=ErrorCategory.FILE_NOT_FOUND,
                        msg_index=0,
                    ),
                ],
            )
            for i in range(4)
        ]
        report = analyzer.analyze(_project(), sessions)
        assert len(report.cross_session_patterns) >= 1
        assert any("RESEARCH.md" in p for p in report.cross_session_patterns)

    def test_two_sessions_not_enough(self):
        """Only 2 sessions shouldn't trigger cross-session (threshold is 3)."""
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id=f"s{i}",
                tool_calls=[
                    _tc(
                        name="Read",
                        input_data={"file_path": "/rare.py"},
                        output="No such file",
                        is_error=True,
                        error_category=ErrorCategory.FILE_NOT_FOUND,
                        msg_index=0,
                    ),
                ],
            )
            for i in range(2)
        ]
        report = analyzer.analyze(_project(), sessions)
        assert len(report.cross_session_patterns) == 0


class TestPermissionAnalyzer:
    def test_detects_repeated_denials(self):
        """Commands denied 3+ times → permission note."""
        analyzer = FailureAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(
                        name="Bash",
                        input_data={"command": "mkdir -p /x"},
                        output="auto-denied",
                        is_error=True,
                        error_category=ErrorCategory.PERMISSION_DENIED,
                        msg_index=i,
                    )
                    for i in range(4)
                ],
            )
        ]
        report = analyzer.analyze(_project(), sessions)
        assert len(report.permission_issues) >= 1
