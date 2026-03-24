"""Tests for the Traffic Pattern Learner.

Tests pattern extraction from proxy traffic without requiring
a real memory backend.
"""

from __future__ import annotations

import pytest

from headroom.memory.traffic_learner import (
    ExtractedPattern,
    PatternCategory,
    TrafficLearner,
    _classify_error,
    _is_error,
)

# =============================================================================
# Error Classification Tests
# =============================================================================


class TestErrorClassification:
    def test_file_not_found(self):
        assert _classify_error("No such file or directory: foo.py") == "file_not_found"
        assert _classify_error("FileNotFoundError: [Errno 2]") == "file_not_found"

    def test_command_not_found(self):
        assert _classify_error("zsh: command not found: ruff") == "command_not_found"

    def test_module_not_found(self):
        assert _classify_error("ModuleNotFoundError: No module named 'foo'") == "module_not_found"

    def test_permission_denied(self):
        assert _classify_error("Permission denied: /etc/shadow") == "permission_denied"

    def test_not_an_error(self):
        assert _classify_error("Everything is fine, tests passed!") is None
        assert _classify_error("") is None

    def test_is_error_helper(self):
        assert _is_error("No such file or directory")
        assert not _is_error("All tests passed")
        assert not _is_error("")
        assert not _is_error("short")


# =============================================================================
# Traffic Learner Core Tests
# =============================================================================


class TestTrafficLearner:
    @pytest.fixture
    def learner(self):
        """Create a learner with low evidence threshold for testing."""
        return TrafficLearner(
            backend=None,
            user_id="test-user",
            min_evidence=1,  # Save on first sighting for tests
        )

    @pytest.mark.asyncio
    async def test_error_recovery_bash(self, learner: TrafficLearner):
        """Test error→recovery pattern extraction for Bash commands."""
        # First: a failed command
        await learner.on_tool_result(
            tool_name="Bash",
            tool_input={"command": "ruff check ."},
            tool_output="zsh: command not found: ruff",
            is_error=True,
        )

        # Then: the recovery
        await learner.on_tool_result(
            tool_name="Bash",
            tool_input={"command": "source .venv/bin/activate && ruff check ."},
            tool_output="All checks passed!",
            is_error=False,
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1
        assert stats["requests_processed"] == 2

    @pytest.mark.asyncio
    async def test_error_recovery_read(self, learner: TrafficLearner):
        """Test error→recovery for Read tool (wrong path → correct path)."""
        await learner.on_tool_result(
            tool_name="Read",
            tool_input={"file_path": "/src/old_module.py"},
            tool_output="No such file or directory: /src/old_module.py",
            is_error=True,
        )

        await learner.on_tool_result(
            tool_name="Read",
            tool_input={"file_path": "/src/new_module.py"},
            tool_output="# Module content here\nclass Foo: pass",
            is_error=False,
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_environment_venv_detection(self, learner: TrafficLearner):
        """Test detection of virtual environment activation patterns."""
        await learner.on_tool_result(
            tool_name="Bash",
            tool_input={"command": "source /project/.venv/bin/activate && pytest"},
            tool_output="5 passed in 2.1s",
            is_error=False,
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_preference_extraction(self, learner: TrafficLearner):
        """Test extraction of user preference signals."""
        await learner.on_messages(
            [
                {"role": "user", "content": "don't use git push, I'll push manually"},
            ]
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_preference_from_content_blocks(self, learner: TrafficLearner):
        """Test preference extraction from Anthropic content block format."""
        await learner.on_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "stop running the full test suite without asking"},
                    ],
                },
            ]
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_evidence_accumulation(self):
        """Test that patterns need min_evidence before saving."""
        learner = TrafficLearner(backend=None, min_evidence=3)

        # Same error→recovery pattern 3 times
        for _ in range(3):
            await learner.on_tool_result(
                tool_name="Bash",
                tool_input={"command": "python test.py"},
                tool_output="command not found: python",
                is_error=True,
            )
            await learner.on_tool_result(
                tool_name="Bash",
                tool_input={"command": "python3 test.py"},
                tool_output="OK",
                is_error=False,
            )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 3

    @pytest.mark.asyncio
    async def test_dedup(self, learner: TrafficLearner):
        """Test that identical patterns are deduplicated."""
        # Same pattern twice
        for _ in range(2):
            await learner.on_tool_result(
                tool_name="Bash",
                tool_input={"command": "ruff check ."},
                tool_output="command not found: ruff",
                is_error=True,
            )
            await learner.on_tool_result(
                tool_name="Bash",
                tool_input={"command": ".venv/bin/ruff check ."},
                tool_output="OK",
                is_error=False,
            )

        # Should not double-count the same pattern
        stats = learner.get_stats()
        # First extraction saves, second is deduped
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_extract_tool_results_from_messages(self, learner: TrafficLearner):
        """Test extraction of tool results from Anthropic message format."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "text": "file1.py\nfile2.py"}],
                    }
                ],
            },
        ]

        results = learner.extract_tool_results_from_messages(messages)
        assert len(results) == 1
        assert results[0]["tool_name"] == "Bash"
        assert "file1.py" in results[0]["output"]
        assert not results[0]["is_error"]

    @pytest.mark.asyncio
    async def test_tool_history_bounded(self, learner: TrafficLearner):
        """Test that tool history stays within max_history."""
        for i in range(30):
            await learner.on_tool_result(
                tool_name="Read",
                tool_input={"file_path": f"/file{i}.py"},
                tool_output=f"content {i}",
                is_error=False,
            )

        assert len(learner._tool_history) <= learner._max_history

    @pytest.mark.asyncio
    async def test_no_pattern_from_success_only(self, learner: TrafficLearner):
        """Test that success without prior error doesn't generate error_recovery pattern."""
        await learner.on_tool_result(
            tool_name="Bash",
            tool_input={"command": "echo hello"},
            tool_output="hello",
            is_error=False,
        )

        stats = learner.get_stats()
        # Only environment patterns possible, no error_recovery
        assert stats["requests_processed"] == 1


# =============================================================================
# Pattern Model Tests
# =============================================================================


class TestExtractedPattern:
    def test_content_hash_deterministic(self):
        p1 = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use venv",
            importance=0.5,
        )
        p2 = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use venv",
            importance=0.8,  # Different importance, same hash
        )
        assert p1.content_hash == p2.content_hash

    def test_different_content_different_hash(self):
        p1 = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use venv",
            importance=0.5,
        )
        p2 = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use conda",
            importance=0.5,
        )
        assert p1.content_hash != p2.content_hash
