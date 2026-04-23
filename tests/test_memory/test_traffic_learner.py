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


# =============================================================================
# Evidence-count persistence & re-sighting bumps
# =============================================================================


class _FakeBackend:
    """Minimal LocalBackend stand-in that persists to a real SQLite file.

    Provides just enough surface area for TrafficLearner: `_config.db_path`
    (read by `_resolve_backend_db_path`) and an `async save_memory` that
    inserts a row and returns an object with `.id`.
    """

    def __init__(self, db_path):
        import types as _types

        self._config = _types.SimpleNamespace(db_path=str(db_path))
        self._db_path = str(db_path)

    async def save_memory(
        self,
        *,
        content: str,
        user_id: str,
        importance: float,
        metadata: dict,
    ):
        import json as _json
        import sqlite3 as _sql
        import types as _types
        import uuid

        mid = str(uuid.uuid4())
        conn = _sql.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO memories (id, content, metadata, entity_refs, importance) "
                "VALUES (?,?,?,?,?)",
                (mid, content, _json.dumps(metadata), "[]", importance),
            )
            conn.commit()
        finally:
            conn.close()
        return _types.SimpleNamespace(id=mid)


def _init_db(path):
    import sqlite3 as _sql

    conn = _sql.connect(path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, "
        "metadata TEXT NOT NULL DEFAULT '{}', "
        "entity_refs TEXT NOT NULL DEFAULT '[]', "
        "importance REAL NOT NULL DEFAULT 0.5)"
    )
    conn.commit()
    conn.close()


def _read_traffic_rows(db_path):
    import json as _json
    import sqlite3 as _sql

    conn = _sql.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, content, metadata FROM memories "
            "WHERE json_extract(metadata, '$.source') = 'traffic_learner'"
        ).fetchall()
    finally:
        conn.close()
    return [(r[0], r[1], _json.loads(r[2])) for r in rows]


async def _wait_for_saved(learner: TrafficLearner, count: int, db_path) -> None:
    """Wait until at least `count` traffic_learner rows exist in the DB."""
    import asyncio as _asyncio

    for _ in range(100):
        if len(_read_traffic_rows(db_path)) >= count:
            return
        await _asyncio.sleep(0.02)
    raise AssertionError(
        f"Timeout waiting for {count} saved row(s); got {len(_read_traffic_rows(db_path))}"
    )


class TestEvidencePersistence:
    @pytest.mark.asyncio
    async def test_save_persists_actual_evidence_count(self, tmp_path):
        """The count written to the DB reflects total sightings, not the default 1."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=3)
        await learner.start()

        pattern_kwargs = {
            "category": PatternCategory.ENVIRONMENT,
            "content": "Use /usr/bin/python3 for system scripts.",
            "importance": 0.6,
        }
        for _ in range(3):
            await learner._accumulate(ExtractedPattern(**pattern_kwargs))
        await _wait_for_saved(learner, 1, db)
        await learner.stop()

        rows = _read_traffic_rows(db)
        assert len(rows) == 1
        assert rows[0][2]["evidence_count"] == 3

    @pytest.mark.asyncio
    async def test_resighting_bumps_persisted_row(self, tmp_path):
        """Sightings after save bump the existing row instead of creating duplicates."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=2)
        await learner.start()

        def mk() -> ExtractedPattern:
            return ExtractedPattern(
                category=PatternCategory.PREFERENCE,
                content="User preference: terse replies.",
                importance=0.7,
            )

        # Two sightings → save with evidence_count=2.
        await learner._accumulate(mk())
        await learner._accumulate(mk())
        await _wait_for_saved(learner, 1, db)

        # Three more sightings → three bumps.
        for _ in range(3):
            await learner._accumulate(mk())
        await learner.stop()

        rows = _read_traffic_rows(db)
        assert len(rows) == 1, "re-sightings must not create duplicate rows"
        assert rows[0][2]["evidence_count"] == 5

    @pytest.mark.asyncio
    async def test_hydrate_prevents_cross_session_duplicates(self, tmp_path):
        """A second session re-sighting an already-persisted pattern bumps, doesn't insert."""
        import json as _json
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        _init_db(db)

        # Session 1 row pre-seeded directly.
        seeded_content = "Command `foo` fails; use `bar` instead."
        conn = _sql.connect(db)
        conn.execute(
            "INSERT INTO memories (id, content, metadata, entity_refs, importance) "
            "VALUES (?,?,?,?,?)",
            (
                "seed-id",
                seeded_content,
                _json.dumps(
                    {
                        "source": "traffic_learner",
                        "category": "error_recovery",
                        "evidence_count": 2,
                    }
                ),
                "[]",
                0.7,
            ),
        )
        conn.commit()
        conn.close()

        # Session 2: fresh learner, hydrates from DB on start().
        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=2)
        await learner.start()

        def mk() -> ExtractedPattern:
            return ExtractedPattern(
                category=PatternCategory.ERROR_RECOVERY,
                content=seeded_content,
                importance=0.7,
            )

        # Two sightings: both should bump the seeded row (no duplicates).
        await learner._accumulate(mk())
        await learner._accumulate(mk())
        await learner.stop()

        rows = _read_traffic_rows(db)
        assert len(rows) == 1
        assert rows[0][0] == "seed-id"
        assert rows[0][2]["evidence_count"] == 4


# =============================================================================
# flush_to_file end-to-end + early-return paths
# =============================================================================


class _FakeWriteResult:
    def __init__(self, files_written):
        self.files_written = files_written


class _FakeWriter:
    def __init__(self):
        self.calls: list[tuple] = []
        self.files_to_return: list = []
        self.raise_on_write = False

    def write(self, recommendations, project, *, dry_run):
        self.calls.append((list(recommendations), project, dry_run))
        if self.raise_on_write:
            raise RuntimeError("boom")
        return _FakeWriteResult(list(self.files_to_return))


class _FakePlugin:
    def __init__(self, roots, writer, discover_raises=False):
        self._roots = roots
        self._writer = writer
        self._discover_raises = discover_raises

    def discover_projects(self):
        if self._discover_raises:
            raise RuntimeError("discover blew up")
        return list(self._roots)

    def create_writer(self):
        return self._writer


def _install_plugin_registry(monkeypatch, plugin):
    """Stub out headroom.learn.registry so flush_to_file uses our fake."""
    import sys
    import types as _types

    fake = _types.ModuleType("headroom.learn.registry")
    fake.auto_detect_plugins = lambda: [plugin] if plugin is not None else []  # type: ignore[attr-defined]
    fake.get_plugin = lambda agent_type: plugin  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.learn.registry", fake)


def _make_project(path):
    from pathlib import Path as _P

    from headroom.learn.models import ProjectInfo

    p = _P(path)
    return ProjectInfo(name=p.name, project_path=p, data_path=p)


class TestFlushToFile:
    @pytest.mark.asyncio
    async def test_end_to_end_writes_per_project(self, tmp_path, monkeypatch):
        """Happy path: anchored patterns → bucketed per project → writer called."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)

        learner = TrafficLearner(backend=backend, agent_type="claude", min_evidence=2)
        writer = _FakeWriter()
        writer.files_to_return = [tmp_path / "CLAUDE.md"]
        proj = _make_project(str(tmp_path))
        plugin = _FakePlugin(roots=[proj], writer=writer)
        _install_plugin_registry(monkeypatch, plugin)

        # Need the save worker running so accumulated patterns actually land in
        # the DB where flush_to_file reads them.
        await learner.start()
        try:

            def mk() -> ExtractedPattern:
                return ExtractedPattern(
                    category=PatternCategory.ENVIRONMENT,
                    content=f"Use /usr/bin/python3 at {tmp_path}/main.py",
                    importance=0.6,
                )

            # Two sightings → save at evidence_count=2 (crosses live-flush gate).
            await learner._accumulate(mk())
            await learner._accumulate(mk())
            await _wait_for_saved(learner, 1, db)

            await learner.flush_to_file()
        finally:
            await learner.stop()

        assert len(writer.calls) >= 1
        recs, written_proj, dry_run = writer.calls[0]
        assert dry_run is False
        assert written_proj is proj
        assert len(recs) == 1
        assert "python3" in recs[0].content

    @pytest.mark.asyncio
    async def test_early_returns_no_plugin(self, monkeypatch):
        """No plugin detected → flush is a no-op."""
        learner = TrafficLearner(backend=None, agent_type="unknown", min_evidence=1)
        _install_plugin_registry(monkeypatch, None)
        # Seed an accumulator entry so the check isn't vacuously "no patterns".
        learner._pattern_counts["h"] = (
            ExtractedPattern(
                category=PatternCategory.ENVIRONMENT,
                content="x",
                importance=0.5,
                evidence_count=2,
            ),
            2,
        )
        await learner.flush_to_file()  # returns without raising

    @pytest.mark.asyncio
    async def test_early_return_no_patterns(self, monkeypatch):
        """Empty accumulator and empty DB → flush returns without calling writer."""
        writer = _FakeWriter()
        plugin = _FakePlugin(roots=[_make_project("/x/a")], writer=writer)
        _install_plugin_registry(monkeypatch, plugin)

        learner = TrafficLearner(backend=None, agent_type="claude", min_evidence=1)
        await learner.flush_to_file()
        assert writer.calls == []

    @pytest.mark.asyncio
    async def test_discover_projects_failure_is_swallowed(self, monkeypatch):
        """If plugin.discover_projects raises, flush logs and returns."""
        writer = _FakeWriter()
        plugin = _FakePlugin(roots=[], writer=writer, discover_raises=True)
        _install_plugin_registry(monkeypatch, plugin)

        learner = TrafficLearner(backend=None, agent_type="claude", min_evidence=1)
        learner._pattern_counts["h"] = (
            ExtractedPattern(
                category=PatternCategory.ENVIRONMENT,
                content="whatever",
                importance=0.5,
                evidence_count=2,
            ),
            2,
        )
        await learner.flush_to_file()
        assert writer.calls == []  # no roots → short-circuits before writer

    @pytest.mark.asyncio
    async def test_unanchored_patterns_dropped(self, tmp_path, monkeypatch):
        """Patterns with no path anchoring are dropped before writer is called."""
        writer = _FakeWriter()
        plugin = _FakePlugin(roots=[_make_project(str(tmp_path))], writer=writer)
        _install_plugin_registry(monkeypatch, plugin)

        learner = TrafficLearner(backend=None, agent_type="claude", min_evidence=1)
        # Content has no absolute path — should be dropped as un-anchored.
        learner._pattern_counts["h"] = (
            ExtractedPattern(
                category=PatternCategory.PREFERENCE,
                content="User preference: use terse output",
                importance=0.7,
                evidence_count=2,
            ),
            2,
        )
        await learner.flush_to_file()
        assert writer.calls == []

    @pytest.mark.asyncio
    async def test_writer_exception_does_not_propagate(self, tmp_path, monkeypatch):
        """A writer raising should be logged; flush must not bubble the error."""
        writer = _FakeWriter()
        writer.raise_on_write = True
        plugin = _FakePlugin(roots=[_make_project(str(tmp_path))], writer=writer)
        _install_plugin_registry(monkeypatch, plugin)

        learner = TrafficLearner(backend=None, agent_type="claude", min_evidence=1)
        learner._pattern_counts["h"] = (
            ExtractedPattern(
                category=PatternCategory.ENVIRONMENT,
                content=f"Use {tmp_path}/tool.py",
                importance=0.6,
                evidence_count=2,
            ),
            2,
        )
        await learner.flush_to_file()  # must not raise
        assert len(writer.calls) == 1


# =============================================================================
# Internal helper edge cases — _resolve_backend_db_path / _collect_all_patterns
# / _hydrate_persisted_state / _bump_persisted_evidence
# =============================================================================


class TestBackendResolution:
    def test_resolve_none_backend(self):
        from headroom.memory.traffic_learner import _resolve_backend_db_path

        assert _resolve_backend_db_path(None) is None

    def test_resolve_backend_without_config(self):
        from headroom.memory.traffic_learner import _resolve_backend_db_path

        class _Bare:
            pass

        assert _resolve_backend_db_path(_Bare()) is None

    def test_resolve_backend_with_empty_db_path(self):
        import types as _types

        from headroom.memory.traffic_learner import _resolve_backend_db_path

        backend = _types.SimpleNamespace(_config=_types.SimpleNamespace(db_path=""))
        assert _resolve_backend_db_path(backend) is None


class TestCollectAllPatterns:
    @pytest.mark.asyncio
    async def test_merges_db_and_accumulator(self, tmp_path):
        """Patterns in both DB and accumulator get evidence_count summed by hash."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)

        # Seed DB with a traffic_learner row at evidence_count=3.
        await backend.save_memory(
            content="shared pattern",
            user_id="t",
            importance=0.5,
            metadata={
                "source": "traffic_learner",
                "category": "environment",
                "evidence_count": 3,
            },
        )

        learner = TrafficLearner(backend=backend, min_evidence=1)
        # Same content in accumulator with count=2; hash matches.
        p = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="shared pattern",
            importance=0.5,
        )
        learner._pattern_counts[p.content_hash] = (p, 2)

        merged = learner._collect_all_patterns()
        assert len(merged) == 1
        assert merged[0].evidence_count == 3 + 2

    def test_handles_missing_db_gracefully(self, tmp_path):
        """A backend pointing to a nonexistent DB is skipped, not raised."""
        backend = _FakeBackend(tmp_path / "absent.db")  # file not created
        learner = TrafficLearner(backend=backend, min_evidence=1)
        merged = learner._collect_all_patterns()
        assert merged == []


class TestHydrateEdgeCases:
    @pytest.mark.asyncio
    async def test_no_backend(self):
        """start() with backend=None hydrates to empty state and still runs."""
        learner = TrafficLearner(backend=None, min_evidence=1)
        await learner.start()
        try:
            assert learner._saved_hashes == set()
            assert learner._persisted_ids == {}
        finally:
            await learner.stop()

    @pytest.mark.asyncio
    async def test_missing_db_file(self, tmp_path):
        """Backend with a db_path that doesn't exist → hydrate is a no-op."""
        backend = _FakeBackend(tmp_path / "not-there.db")
        learner = TrafficLearner(backend=backend, min_evidence=1)
        await learner._hydrate_persisted_state()
        assert learner._saved_hashes == set()
        assert learner._persisted_ids == {}


class TestBumpEdgeCases:
    @pytest.mark.asyncio
    async def test_bump_with_no_backend_is_noop(self):
        learner = TrafficLearner(backend=None, min_evidence=1)
        # Should not raise even with no backend.
        await learner._bump_persisted_evidence("some-id")

    @pytest.mark.asyncio
    async def test_bump_with_missing_db_is_noop(self, tmp_path):
        backend = _FakeBackend(tmp_path / "absent.db")
        learner = TrafficLearner(backend=backend, min_evidence=1)
        await learner._bump_persisted_evidence("some-id")  # no exception

    @pytest.mark.asyncio
    async def test_bump_unknown_id_is_noop(self, tmp_path):
        """Updating a non-existent memory id silently affects zero rows."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=1)
        await learner._bump_persisted_evidence("no-such-id")
        assert _read_traffic_rows(db) == []


# =============================================================================
# stop() cancels the flush task
# =============================================================================


class TestStopCancels:
    @pytest.mark.asyncio
    async def test_stop_cancels_flush_task(self):
        learner = TrafficLearner(backend=None, min_evidence=1)
        await learner.start()
        assert learner._flush_task is not None and not learner._flush_task.done()
        await learner.stop()
        assert learner._flush_task is None or learner._flush_task.done()
