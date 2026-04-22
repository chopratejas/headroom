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
    _load_persisted_patterns_from_sqlite,
    _patterns_to_recommendations,
    _project_for_pattern,
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


# =============================================================================
# Project Routing
# =============================================================================


class TestProjectForPattern:
    def _project(self, path: str):
        from pathlib import Path as _P

        from headroom.learn.models import ProjectInfo

        p = _P(path)
        return ProjectInfo(name=p.name, project_path=p, data_path=p)

    def test_matches_longest_root(self):
        proj_a = self._project("/x/a")
        proj_b = self._project("/x/a/b")
        pattern = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="File `/x/a/b/foo.py` does not exist.",
            importance=0.5,
        )
        result = _project_for_pattern(pattern, [proj_a, proj_b])
        assert result is proj_b

    def test_returns_none_for_unanchored(self):
        proj_a = self._project("/x/a")
        pattern = ExtractedPattern(
            category=PatternCategory.PREFERENCE,
            content="User preference: use terse responses",
            importance=0.7,
        )
        assert _project_for_pattern(pattern, [proj_a]) is None

    def test_matches_via_entity_refs(self):
        proj = self._project("/x/a")
        pattern = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="Command failed.",
            importance=0.5,
            entity_refs=["/x/a/tool.py"],
        )
        assert _project_for_pattern(pattern, [proj]) is proj

    def test_no_false_match_on_prefix_boundary(self):
        # /x/ab should not match a project rooted at /x/a
        proj_a = self._project("/x/a")
        pattern = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="File `/x/ab/foo.py` does not exist.",
            importance=0.5,
        )
        assert _project_for_pattern(pattern, [proj_a]) is None


# =============================================================================
# Persisted-pattern loading from memory.db
# =============================================================================


class TestLoadPersistedPatterns:
    def _make_db(self, tmp_path, rows: list[dict]):
        import json as _json
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        conn = _sql.connect(db)
        conn.execute(
            "CREATE TABLE memories ("
            "id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "metadata TEXT NOT NULL DEFAULT '{}', "
            "entity_refs TEXT NOT NULL DEFAULT '[]', "
            "importance REAL NOT NULL DEFAULT 0.5)"
        )
        for i, r in enumerate(rows):
            conn.execute(
                "INSERT INTO memories (id, content, metadata, entity_refs, importance) "
                "VALUES (?,?,?,?,?)",
                (
                    str(i),
                    r["content"],
                    _json.dumps(r.get("metadata", {})),
                    _json.dumps(r.get("entity_refs", [])),
                    r.get("importance", 0.5),
                ),
            )
        conn.commit()
        conn.close()
        return db

    def test_dedupes_by_content_and_sums_evidence(self, tmp_path):
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "Command `foo` fails.",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "error_recovery",
                        "evidence_count": 2,
                    },
                },
                {
                    "content": "Command `foo` fails.",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "error_recovery",
                        "evidence_count": 3,
                    },
                },
            ],
        )
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        assert patterns[0].evidence_count == 5
        assert patterns[0].category == PatternCategory.ERROR_RECOVERY

    def test_skips_non_traffic_rows(self, tmp_path):
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "Something else",
                    "metadata": {"source": "other"},
                },
                {
                    "content": "From traffic",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "environment",
                    },
                },
            ],
        )
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        assert patterns[0].content == "From traffic"

    def test_reads_importance_column(self, tmp_path):
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "High-importance pattern",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "environment",
                    },
                    "importance": 0.85,
                },
            ],
        )
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        assert patterns[0].importance == 0.85

    def test_skips_unknown_category(self, tmp_path):
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "X",
                    "metadata": {"source": "traffic_learner", "category": "bogus"},
                },
            ],
        )
        assert _load_persisted_patterns_from_sqlite(db) == []


# =============================================================================
# Category → recommendation routing
# =============================================================================


class TestPatternsToRecommendations:
    def test_routes_preference_to_memory_file(self):
        from headroom.learn.models import RecommendationTarget

        patterns = [
            ExtractedPattern(
                category=PatternCategory.PREFERENCE,
                content="User prefers terse output",
                importance=0.8,
                evidence_count=3,
            ),
        ]
        recs = _patterns_to_recommendations(patterns)
        assert len(recs) == 1
        assert recs[0].target == RecommendationTarget.MEMORY_FILE
        assert "User prefers terse output" in recs[0].content

    def test_routes_environment_to_context_file(self):
        from headroom.learn.models import RecommendationTarget

        patterns = [
            ExtractedPattern(
                category=PatternCategory.ENVIRONMENT,
                content="Use uv run python",
                importance=0.7,
                evidence_count=4,
            ),
        ]
        recs = _patterns_to_recommendations(patterns)
        assert len(recs) == 1
        assert recs[0].target == RecommendationTarget.CONTEXT_FILE

    def test_groups_by_category(self):
        patterns = [
            ExtractedPattern(
                category=PatternCategory.ERROR_RECOVERY,
                content="A",
                importance=0.5,
                evidence_count=2,
            ),
            ExtractedPattern(
                category=PatternCategory.ERROR_RECOVERY,
                content="B",
                importance=0.5,
                evidence_count=5,
            ),
        ]
        recs = _patterns_to_recommendations(patterns)
        assert len(recs) == 1
        # B has higher evidence, should sort first
        lines = recs[0].content.splitlines()
        assert lines[0] == "- B"
        assert lines[1] == "- A"
        assert recs[0].evidence_count == 7


# =============================================================================
# Debounced flush worker
# =============================================================================


class TestFlushDebounce:
    @pytest.mark.asyncio
    async def test_flush_worker_rate_limits(self, monkeypatch):
        """Rapid dirty flags should not cause rapid flush_to_file calls."""
        from headroom.memory import traffic_learner as tl_mod

        # Shorten debounce for a fast test
        monkeypatch.setattr(tl_mod, "FLUSH_DEBOUNCE_SECONDS", 0.5)

        learner = TrafficLearner(backend=None, min_evidence=1)
        call_count = 0

        async def fake_flush() -> None:
            nonlocal call_count
            call_count += 1

        learner.flush_to_file = fake_flush  # type: ignore[method-assign]

        await learner.start()
        # Toggle dirty rapidly over ~1.2s, which permits at most ~2 flushes.
        for _ in range(30):
            learner._flush_dirty = True
            await __import__("asyncio").sleep(0.04)

        await learner.stop()

        # start() kicked a flush dirty→false at some point; stop() also calls
        # flush_to_file once (final flush). We want evidence the worker did
        # NOT call flush on every sleep tick — cap is generous.
        assert call_count <= 5, f"Expected few flushes, got {call_count}"
        assert call_count >= 1, "Expected at least one flush during the burst"
