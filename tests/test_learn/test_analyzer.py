"""Tests for session analyzer — digest builder and LLM-based analysis."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from headroom.learn.analyzer import (
    SessionAnalyzer,
    _build_digest,
    _detect_default_model,
    _parse_llm_response,
)
from headroom.learn.models import (
    AnalysisResult,
    ErrorCategory,
    ProjectInfo,
    RecommendationTarget,
    SessionData,
    SessionEvent,
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
    output_bytes: int = 0,
) -> ToolCall:
    return ToolCall(
        name=name,
        tool_call_id=f"tc_{msg_index}",
        input_data=input_data or {},
        output=output,
        is_error=is_error,
        error_category=error_category,
        msg_index=msg_index,
        output_bytes=output_bytes or len(output),
    )


# =============================================================================
# Digest Builder Tests
# =============================================================================


class TestDigestBuilder:
    def test_includes_project_info(self):
        project = _project()
        sessions = [SessionData(session_id="s1", tool_calls=[_tc()])]
        digest = _build_digest(project, sessions)
        assert "test-project" in digest
        assert "/tmp/test-project" in digest

    def test_includes_session_stats(self):
        sessions = [
            SessionData(
                session_id="abc123",
                tool_calls=[_tc(msg_index=0), _tc(msg_index=1, is_error=True, output="Error!")],
            )
        ]
        digest = _build_digest(_project(), sessions)
        assert "abc123" in digest
        assert "2 calls" in digest
        assert "1 failure" in digest

    def test_includes_tool_call_details(self):
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(
                        name="Read",
                        input_data={"file_path": "/src/foo.py"},
                        output="contents",
                        msg_index=0,
                    ),
                    _tc(
                        name="Bash",
                        input_data={"command": "python3 run.py"},
                        output="ModuleNotFoundError",
                        is_error=True,
                        error_category=ErrorCategory.MODULE_NOT_FOUND,
                        msg_index=1,
                    ),
                ],
            )
        ]
        digest = _build_digest(_project(), sessions)
        assert "/src/foo.py" in digest
        assert "python3 run.py" in digest
        assert "ERROR" in digest
        assert "ModuleNotFoundError" in digest

    def test_includes_user_messages(self):
        tc = _tc(msg_index=0)
        events = [
            SessionEvent(type="tool_call", msg_index=0, tool_call=tc),
            SessionEvent(type="user_message", msg_index=1, text="Use uv run instead"),
        ]
        sessions = [SessionData(session_id="s1", tool_calls=[tc], events=events)]
        digest = _build_digest(_project(), sessions)
        assert "USER:" in digest
        assert "Use uv run instead" in digest

    def test_includes_subagent_summaries(self):
        events = [
            SessionEvent(
                type="agent_summary",
                msg_index=0,
                agent_tool_count=150,
                agent_tokens=60000,
                agent_prompt="Explore all test files",
            ),
        ]
        sessions = [SessionData(session_id="s1", events=events)]
        digest = _build_digest(_project(), sessions)
        assert "SUBAGENT" in digest
        assert "150 tool calls" in digest
        assert "Explore all test files" in digest

    def test_includes_interruptions(self):
        events = [
            SessionEvent(
                type="interruption",
                msg_index=0,
                text="[Request interrupted by user]",
            ),
        ]
        sessions = [SessionData(session_id="s1", events=events)]
        digest = _build_digest(_project(), sessions)
        assert "INTERRUPTED" in digest

    def test_empty_sessions(self):
        digest = _build_digest(_project(), [])
        assert "0 sessions" in digest or "test-project" in digest


# =============================================================================
# LLM Response Parser Tests
# =============================================================================


class TestLLMResponseParser:
    def test_parses_context_file_rules(self):
        raw = {
            "context_file_rules": [
                {
                    "section": "Environment",
                    "content": "- Use `uv run python` instead of `python3`",
                    "estimated_tokens_saved": 800,
                    "evidence_count": 5,
                }
            ],
            "memory_file_rules": [],
        }
        recs = _parse_llm_response(raw)
        assert len(recs) == 1
        assert recs[0].target == RecommendationTarget.CONTEXT_FILE
        assert recs[0].section == "Environment"
        assert "uv run python" in recs[0].content
        assert recs[0].estimated_tokens_saved == 800
        assert recs[0].evidence_count == 5

    def test_parses_memory_file_rules(self):
        raw = {
            "context_file_rules": [],
            "memory_file_rules": [
                {
                    "section": "User Preferences",
                    "content": "- Do not auto-execute curl commands",
                    "estimated_tokens_saved": 500,
                    "evidence_count": 3,
                }
            ],
        }
        recs = _parse_llm_response(raw)
        assert len(recs) == 1
        assert recs[0].target == RecommendationTarget.MEMORY_FILE
        assert "curl" in recs[0].content

    def test_sorts_by_token_savings(self):
        raw = {
            "context_file_rules": [
                {
                    "section": "Paths",
                    "content": "- Use correct paths",
                    "estimated_tokens_saved": 200,
                    "evidence_count": 2,
                },
                {
                    "section": "Environment",
                    "content": "- Use uv",
                    "estimated_tokens_saved": 1000,
                    "evidence_count": 5,
                },
            ],
            "memory_file_rules": [],
        }
        recs = _parse_llm_response(raw)
        assert recs[0].estimated_tokens_saved == 1000
        assert recs[1].estimated_tokens_saved == 200

    def test_handles_missing_fields(self):
        raw = {
            "context_file_rules": [
                {"section": "Env", "content": "- stuff"},
                {"section": "", "content": ""},  # should be skipped
                {"not_a_real_field": True},  # should be skipped
            ],
            "memory_file_rules": [],
        }
        recs = _parse_llm_response(raw)
        assert len(recs) == 1

    def test_handles_empty_response(self):
        recs = _parse_llm_response({})
        assert recs == []

    def test_handles_non_dict_entries(self):
        raw = {"context_file_rules": ["not a dict", 42], "memory_file_rules": []}
        recs = _parse_llm_response(raw)
        assert recs == []


# =============================================================================
# Full Analyzer Integration Tests (mocked LLM)
# =============================================================================


class TestSessionAnalyzer:
    def test_empty_sessions_no_llm_call(self):
        """No failures + no events → no LLM call, empty result."""
        analyzer = SessionAnalyzer()
        result = analyzer.analyze(_project(), [])
        assert result.total_calls == 0
        assert result.total_failures == 0
        assert result.recommendations == []

    @patch("headroom.learn.analyzer._call_llm")
    def test_calls_llm_with_digest(self, mock_call_llm: MagicMock):
        mock_call_llm.return_value = {
            "context_file_rules": [
                {
                    "section": "Environment",
                    "content": "- Use uv run python",
                    "estimated_tokens_saved": 800,
                    "evidence_count": 3,
                }
            ],
            "memory_file_rules": [],
        }

        analyzer = SessionAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[
                    _tc(msg_index=0, is_error=True, output="ModuleNotFoundError"),
                    _tc(msg_index=1),
                ],
            )
        ]
        result = analyzer.analyze(_project(), sessions)

        mock_call_llm.assert_called_once()
        assert result.total_calls == 2
        assert result.total_failures == 1
        assert len(result.recommendations) == 1
        assert "uv run python" in result.recommendations[0].content

    @patch("headroom.learn.analyzer._call_llm")
    def test_handles_llm_failure_gracefully(self, mock_call_llm: MagicMock):
        mock_call_llm.side_effect = RuntimeError("API key not set")

        analyzer = SessionAnalyzer()
        sessions = [
            SessionData(
                session_id="s1",
                tool_calls=[_tc(msg_index=0, is_error=True, output="error")],
            )
        ]
        result = analyzer.analyze(_project(), sessions)

        # Stats should still work, just no recommendations
        assert result.total_calls == 1
        assert result.total_failures == 1
        assert result.recommendations == []

    @patch("headroom.learn.analyzer._call_llm")
    def test_passes_events_to_digest(self, mock_call_llm: MagicMock):
        """User messages and subagent events should appear in the digest."""
        mock_call_llm.return_value = {"context_file_rules": [], "memory_file_rules": []}

        tc = _tc(msg_index=0, is_error=True, output="error")
        events = [
            SessionEvent(type="tool_call", msg_index=0, tool_call=tc),
            SessionEvent(type="user_message", msg_index=1, text="use venv python"),
        ]
        sessions = [SessionData(session_id="s1", tool_calls=[tc], events=events)]

        analyzer = SessionAnalyzer()
        analyzer.analyze(_project(), sessions)

        # Check that the digest passed to the LLM includes user message
        call_args = mock_call_llm.call_args
        digest = call_args[0][0]  # first positional arg
        assert "use venv python" in digest


# =============================================================================
# Model Auto-Detection
# =============================================================================


class TestDetectDefaultModel:
    def test_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert _detect_default_model() == "claude-sonnet-4-6"

    def test_openai_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert _detect_default_model() == "gpt-4o"

    def test_gemini_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        assert _detect_default_model() == "gemini/gemini-2.0-flash"

    def test_anthropic_preferred_over_openai(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert _detect_default_model() == "claude-sonnet-4-6"

    def test_no_keys_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        import pytest

        with pytest.raises(RuntimeError, match="No LLM API key found"):
            _detect_default_model()


# =============================================================================
# Legacy Compatibility
# =============================================================================


class TestFailureAnalyzerCompat:
    @patch("headroom.learn.analyzer._call_llm")
    def test_legacy_alias_works(self, mock_call_llm: MagicMock):
        from headroom.learn.analyzer import FailureAnalyzer

        mock_call_llm.return_value = {"context_file_rules": [], "memory_file_rules": []}

        analyzer = FailureAnalyzer()
        result = analyzer.analyze(_project(), [])
        assert isinstance(result, AnalysisResult)
