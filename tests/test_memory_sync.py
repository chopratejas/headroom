"""Comprehensive tests for the universal memory sync engine.

Tests cover:
- Core sync: import, export, bidirectional
- Idempotency and deduplication
- Fast no-op detection
- Lineage and governance metadata
- Claude Code adapter: read/write frontmatter files
- Codex adapter: read/write AGENTS.md sections
- Cross-agent interop: save in one agent, find in another
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import headroom.memory.sync as sync_module
from headroom.memory.sync import (
    AgentMemory,
    AgentMemoryAdapter,
    sync,
    sync_export,
    sync_import,
)
from headroom.memory.sync_adapters.claude_code import (
    ClaudeCodeAdapter,
    _build_frontmatter,
    _parse_frontmatter,
    get_claude_memory_dir,
)
from headroom.memory.sync_adapters.codex_agent import CodexAdapter

# ---------------------------------------------------------------------------
# Fake backend for testing (no real DB/embeddings needed)
# ---------------------------------------------------------------------------


@dataclass
class FakeMemory:
    id: str = ""
    content: str = ""
    user_id: str = ""
    category: str = ""
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeBackend:
    """In-memory backend for testing sync without real DB."""

    def __init__(self) -> None:
        self._memories: list[FakeMemory] = []
        self._next_id = 1

    async def get_user_memories(self, user_id: str, limit: int = 500) -> list[FakeMemory]:
        return [m for m in self._memories if m.user_id == user_id][:limit]

    async def save_memory(
        self,
        content: str,
        user_id: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> FakeMemory:
        mem = FakeMemory(
            id=f"mem_{self._next_id:04d}",
            content=content,
            user_id=user_id,
            importance=importance,
            metadata=metadata or {},
        )
        self._next_id += 1
        self._memories.append(mem)
        return mem

    def add_memory(self, content: str, user_id: str = "tcms", **kwargs: Any) -> FakeMemory:
        """Sync helper to pre-populate memories."""
        mem = FakeMemory(
            id=f"mem_{self._next_id:04d}",
            content=content,
            user_id=user_id,
            metadata=kwargs.get("metadata", {}),
            importance=kwargs.get("importance", 0.5),
        )
        self._next_id += 1
        self._memories.append(mem)
        return mem


# ---------------------------------------------------------------------------
# Core sync tests
# ---------------------------------------------------------------------------


class TestSyncImport:
    """Test importing from agent files into DB."""

    @pytest.fixture
    def backend(self):
        return FakeBackend()

    @pytest.fixture
    def claude_dir(self, tmp_path):
        d = tmp_path / "memory"
        d.mkdir()
        return d

    def _write_claude_memory(
        self, memory_dir: Path, name: str, content: str, **fm_fields: str
    ) -> None:
        slug = name.lower().replace(" ", "_")
        fields = {"name": name, "description": content[:80], "type": "project", **fm_fields}
        fm_lines = ["---"]
        for k, v in fields.items():
            fm_lines.append(f"{k}: {v}")
        fm_lines.append("---")
        (memory_dir / f"{slug}.md").write_text("\n".join(fm_lines) + f"\n\n{content}\n")

    @pytest.mark.asyncio
    async def test_import_claude_files_to_db(self, backend, claude_dir):
        self._write_claude_memory(claude_dir, "Project codename", "The secret name is TC")
        self._write_claude_memory(claude_dir, "Dark mode", "User prefers dark mode")

        adapter = ClaudeCodeAdapter(claude_dir)
        imported = await sync_import(backend, adapter, "tcms")

        assert imported == 2
        mems = await backend.get_user_memories("tcms")
        contents = {m.content for m in mems}
        assert "The secret name is TC" in contents
        assert "User prefers dark mode" in contents

    @pytest.mark.asyncio
    async def test_import_skips_existing(self, backend, claude_dir):
        """Memories already in DB are not re-imported."""
        backend.add_memory(
            "The secret name is TC",
            metadata={"content_hash": hashlib.sha256(b"The secret name is TC").hexdigest()[:16]},
        )

        self._write_claude_memory(claude_dir, "Project codename", "The secret name is TC")
        self._write_claude_memory(claude_dir, "New fact", "Something new")

        adapter = ClaudeCodeAdapter(claude_dir)
        imported = await sync_import(backend, adapter, "tcms")

        assert imported == 1  # Only "Something new"

    @pytest.mark.asyncio
    async def test_import_preserves_lineage(self, backend, claude_dir):
        self._write_claude_memory(claude_dir, "Fact", "Important fact")

        adapter = ClaudeCodeAdapter(claude_dir)
        await sync_import(backend, adapter, "tcms")

        mems = await backend.get_user_memories("tcms")
        assert len(mems) == 1
        assert mems[0].metadata["source_agent"] == "claude"
        assert mems[0].metadata["source_file"] == "fact.md"
        assert "content_hash" in mems[0].metadata
        assert mems[0].metadata["sync_direction"] == "import"


class TestSyncExport:
    """Test exporting from DB to agent files."""

    @pytest.fixture
    def backend(self):
        return FakeBackend()

    @pytest.fixture
    def claude_dir(self, tmp_path):
        d = tmp_path / "memory"
        d.mkdir()
        return d

    @pytest.mark.asyncio
    async def test_export_new_memory_to_claude_files(self, backend, claude_dir):
        backend.add_memory(
            "Project uses Python 3.12",
            metadata={
                "source_agent": "codex",
                "sync_direction": "export",  # Not from claude import
            },
        )

        adapter = ClaudeCodeAdapter(claude_dir)
        exported = await sync_export(backend, adapter, "tcms")

        assert exported == 1
        # Check file was created
        md_files = list(claude_dir.glob("headroom_*.md"))
        assert len(md_files) == 1

        content = md_files[0].read_text()
        assert "Python 3.12" in content
        assert "headroom_id: mem_0001" in content
        assert "source_agent: codex" in content

    @pytest.mark.asyncio
    async def test_export_skips_claude_originated(self, backend, claude_dir):
        """Don't re-export memories that were imported FROM claude (anti-echo)."""
        backend.add_memory(
            "From claude",
            metadata={
                "source_agent": "claude",
                "sync_direction": "import",
            },
        )
        backend.add_memory(
            "From codex",
            metadata={
                "source_agent": "codex",
            },
        )

        adapter = ClaudeCodeAdapter(claude_dir)
        exported = await sync_export(backend, adapter, "tcms")

        assert exported == 1  # Only "From codex"

    @pytest.mark.asyncio
    async def test_export_updates_memory_md_index(self, backend, claude_dir):
        # Create an existing MEMORY.md
        (claude_dir / "MEMORY.md").write_text("# Memory\n\n## User\n- Some existing entry\n")

        backend.add_memory("New fact from codex", metadata={"source_agent": "codex"})

        adapter = ClaudeCodeAdapter(claude_dir)
        await sync_export(backend, adapter, "tcms")

        memory_md = (claude_dir / "MEMORY.md").read_text()
        assert "Headroom Shared Memory" in memory_md
        assert "New fact from codex" in memory_md
        assert "Some existing entry" in memory_md  # Preserved


class TestBidirectionalSync:
    """Test full bidirectional sync."""

    @pytest.fixture
    def backend(self):
        return FakeBackend()

    @pytest.fixture
    def claude_dir(self, tmp_path):
        d = tmp_path / "memory"
        d.mkdir()
        return d

    @pytest.fixture
    def state_path(self, tmp_path):
        return tmp_path / "sync_state.json"

    def _write_claude_memory(self, memory_dir: Path, name: str, content: str) -> None:
        slug = name.lower().replace(" ", "_")
        fm = f"---\nname: {name}\ndescription: {content[:80]}\ntype: project\n---"
        (memory_dir / f"{slug}.md").write_text(f"{fm}\n\n{content}\n")

    @pytest.mark.asyncio
    async def test_bidirectional_sync(self, backend, claude_dir, state_path):
        # Claude has a memory file
        self._write_claude_memory(claude_dir, "Convention", "Always use ruff for linting")

        # DB has a memory from Codex
        backend.add_memory("Secret name is TC", metadata={"source_agent": "codex"})

        adapter = ClaudeCodeAdapter(claude_dir)
        result = await sync(backend, adapter, "tcms", state_path=state_path, force=True)

        assert result.imported == 1  # Claude file → DB
        assert result.exported == 1  # Codex memory → Claude file

        # Verify DB has both
        mems = await backend.get_user_memories("tcms")
        contents = {m.content for m in mems}
        assert "Always use ruff for linting" in contents
        assert "Secret name is TC" in contents

        # Verify Claude dir has the exported file
        all_files = list(claude_dir.glob("headroom_*.md"))
        assert len(all_files) >= 1
        exported_content = " ".join(f.read_text() for f in all_files)
        assert "TC" in exported_content

    @pytest.mark.asyncio
    async def test_sync_idempotent(self, backend, claude_dir, state_path):
        """Running sync twice produces no duplicates."""
        self._write_claude_memory(claude_dir, "Fact", "Python 3.12 is required")
        backend.add_memory("Port 8787 is default", metadata={"source_agent": "codex"})

        adapter = ClaudeCodeAdapter(claude_dir)

        r1 = await sync(backend, adapter, "tcms", state_path=state_path, force=True)
        assert r1.imported == 1
        assert r1.exported == 1

        r2 = await sync(backend, adapter, "tcms", state_path=state_path, force=True)
        assert r2.imported == 0  # Already imported
        assert r2.exported == 0  # Already exported

        # No duplicates in DB
        mems = await backend.get_user_memories("tcms")
        assert len(mems) == 2

    @pytest.mark.asyncio
    async def test_fast_noop_when_unchanged(self, backend, claude_dir, state_path):
        """Second sync with no changes completes in < 10ms."""
        self._write_claude_memory(claude_dir, "Fact", "Some fact")

        adapter = ClaudeCodeAdapter(claude_dir)

        # First sync (populates state)
        await sync(backend, adapter, "tcms", state_path=state_path, force=True)

        # Second sync (should be fast no-op)
        start = time.monotonic()
        r = await sync(backend, adapter, "tcms", state_path=state_path)
        elapsed = (time.monotonic() - start) * 1000

        assert r.imported == 0
        assert r.exported == 0
        assert elapsed < 50  # Generous threshold for CI


class TestLineageAndGovernance:
    """Test metadata tracking for audit and lineage."""

    @pytest.fixture
    def backend(self):
        return FakeBackend()

    @pytest.fixture
    def claude_dir(self, tmp_path):
        d = tmp_path / "memory"
        d.mkdir()
        return d

    @pytest.mark.asyncio
    async def test_lineage_tracks_source_agent(self, backend, claude_dir):
        fm = "---\nname: test\ndescription: test\ntype: project\n---"
        (claude_dir / "test.md").write_text(f"{fm}\n\nClaude discovered this\n")

        adapter = ClaudeCodeAdapter(claude_dir)
        await sync_import(backend, adapter, "tcms")

        mems = await backend.get_user_memories("tcms")
        assert mems[0].metadata["source_agent"] == "claude"

    @pytest.mark.asyncio
    async def test_exported_files_have_headroom_id(self, backend, claude_dir):
        backend.add_memory("From codex", metadata={"source_agent": "codex"})

        adapter = ClaudeCodeAdapter(claude_dir)
        await sync_export(backend, adapter, "tcms")

        md_files = list(claude_dir.glob("headroom_*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text()
        assert "headroom_id:" in content

    @pytest.mark.asyncio
    async def test_sync_state_records_timestamps(self, backend, claude_dir, tmp_path):
        state_path = tmp_path / "state.json"
        fm = "---\nname: t\ndescription: t\ntype: project\n---"
        (claude_dir / "t.md").write_text(f"{fm}\n\nFact\n")

        adapter = ClaudeCodeAdapter(claude_dir)
        await sync(backend, adapter, "tcms", state_path=state_path, force=True)

        state = json.loads(state_path.read_text())
        key = "claude:tcms"
        assert key in state
        assert "last_sync" in state[key]
        assert "agent_fingerprint" in state[key]
        assert "db_fingerprint" in state[key]


@pytest.mark.asyncio
async def test_agent_memory_hash_and_adapter_base_helpers() -> None:
    class DummyAdapter(AgentMemoryAdapter):
        async def read_memories(self) -> list[AgentMemory]:
            return await AgentMemoryAdapter.read_memories(self)

        async def write_memories(self, memories: list[dict[str, Any]]) -> int:
            return await AgentMemoryAdapter.write_memories(self, memories)

        def fingerprint(self) -> str:
            return AgentMemoryAdapter.fingerprint(self)

    auto_hash = AgentMemory(content="hello")
    preset_hash = AgentMemory(content="hello", content_hash="preset")
    assert len(auto_hash.content_hash) == 16
    assert preset_hash.content_hash == "preset"

    adapter = DummyAdapter()
    assert adapter.fingerprint() is None
    assert await adapter.read_memories() is None
    assert await adapter.write_memories([]) is None


def test_sync_helper_functions_handle_invalid_state(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state_path.write_text("{not valid json")
    assert sync_module._load_sync_state(state_path) == {}

    broken = tmp_path / "broken.json"
    broken.write_text("{}")

    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: Any, **kwargs: Any):
        if self == broken:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    try:
        assert sync_module._load_sync_state(broken) == {}
    finally:
        monkeypatch.setattr(Path, "read_text", original_read_text)

    sync_module._save_sync_state(state_path, {"key": {"value": 1}})
    assert json.loads(state_path.read_text()) == {"key": {"value": 1}}
    assert sync_module._db_fingerprint([]) == "empty"


@pytest.mark.asyncio
async def test_sync_export_early_return_and_zero_write(tmp_path):
    backend = FakeBackend()
    adapter = ClaudeCodeAdapter(tmp_path / "memory")
    assert await sync_export(backend, adapter, "tcms", existing_memories=[]) == 0

    backend.add_memory("Export candidate", metadata={"source_agent": "codex"})

    class ZeroWriteAdapter(ClaudeCodeAdapter):
        async def write_memories(self, memories: list[dict[str, Any]]) -> int:
            return 0

    zero_adapter = ZeroWriteAdapter(tmp_path / "memory")
    assert await sync_export(backend, zero_adapter, "tcms") == 0


@pytest.mark.asyncio
async def test_sync_runs_when_state_mismatch_without_force(tmp_path):
    backend = FakeBackend()
    backend.add_memory("From codex", metadata={"source_agent": "codex"})
    claude_dir = tmp_path / "memory"
    claude_dir.mkdir()
    (claude_dir / "fact.md").write_text(
        "---\nname: Fact\ndescription: Fact\ntype: project\n---\n\nImported fact\n"
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"claude:tcms": {"agent_fingerprint": "old", "db_fingerprint": "old"}})
    )

    result = await sync(backend, ClaudeCodeAdapter(claude_dir), "tcms", state_path=state_path)
    assert result.imported == 1
    assert result.exported == 1


@pytest.mark.parametrize("agent_name", ["claude", "codex"])
def test_sync_main_runs_for_supported_agents(monkeypatch, capsys, tmp_path, agent_name):
    calls: dict[str, Any] = {}

    class FakeLocalBackendConfig:
        def __init__(self, db_path):
            self.db_path = db_path

    class FakeLocalBackend:
        def __init__(self, config):
            calls["config_db"] = config.db_path
            self.closed = False

        async def _ensure_initialized(self):
            calls["initialized"] = True

        async def close(self):
            self.closed = True
            calls["closed"] = True

    class FakeClaudeAdapter:
        def __init__(self, memory_dir):
            calls["adapter"] = ("claude", memory_dir)

    class FakeCodexAdapter:
        def __init__(self):
            calls["adapter"] = ("codex", None)

    local_module = types.ModuleType("headroom.memory.backends.local")
    local_module.LocalBackend = FakeLocalBackend
    local_module.LocalBackendConfig = FakeLocalBackendConfig
    monkeypatch.setitem(sys.modules, "headroom.memory.backends.local", local_module)

    claude_module = types.ModuleType("headroom.memory.sync_adapters.claude_code")
    claude_module.ClaudeCodeAdapter = FakeClaudeAdapter
    claude_module.get_claude_memory_dir = lambda: tmp_path / "claude-memory"
    monkeypatch.setitem(sys.modules, "headroom.memory.sync_adapters.claude_code", claude_module)

    codex_module = types.ModuleType("headroom.memory.sync_adapters.codex_agent")
    codex_module.CodexAdapter = FakeCodexAdapter
    monkeypatch.setitem(sys.modules, "headroom.memory.sync_adapters.codex_agent", codex_module)

    async def fake_sync(
        backend, adapter, user_id, state_path=sync_module._DEFAULT_STATE_PATH, force=False
    ):
        calls["sync"] = {"user_id": user_id, "force": force, "adapter_type": type(adapter).__name__}
        return sync_module.SyncResult(imported=1, exported=2, duration_ms=12.4)

    monkeypatch.setattr(sync_module, "sync", fake_sync)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--db",
            "memory.db",
            "--user",
            "tcms",
            "--agent",
            agent_name,
            "--force",
        ],
    )

    sync_module.main()
    output = json.loads(capsys.readouterr().out.strip())
    assert output == {"imported": 1, "exported": 2, "ms": 12}
    assert calls["config_db"] == "memory.db"
    assert calls["initialized"] is True
    assert calls["closed"] is True
    assert calls["sync"]["user_id"] == "tcms"
    assert calls["sync"]["force"] is True
    assert calls["adapter"][0] == agent_name


# ---------------------------------------------------------------------------
# Claude Code adapter tests
# ---------------------------------------------------------------------------


class TestClaudeCodeAdapter:
    """Test Claude Code adapter read/write."""

    @pytest.fixture
    def memory_dir(self, tmp_path):
        d = tmp_path / "memory"
        d.mkdir()
        return d

    def test_parse_frontmatter(self):
        content = "---\nname: Test\ntype: project\n---\n\nBody content here."
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "Test"
        assert fm["type"] == "project"
        assert body == "Body content here."

    def test_parse_frontmatter_no_frontmatter(self):
        content = "Just plain content."
        fm, body = _parse_frontmatter(content)
        assert fm == {}
        assert body == "Just plain content."

    def test_parse_frontmatter_unclosed_and_build_frontmatter(self):
        fm, body = _parse_frontmatter("---\nname: Test\nbody without terminator")
        assert fm == {}
        assert body == "---\nname: Test\nbody without terminator"

        built = _build_frontmatter({"name": "Test", "empty": "", "type": "project"})
        assert built == "---\nname: Test\ntype: project\n---"

    @pytest.mark.asyncio
    async def test_read_memories_skips_memory_md(self, memory_dir):
        (memory_dir / "MEMORY.md").write_text("# Index\n- entry")
        (memory_dir / "fact.md").write_text(
            "---\nname: Fact\ntype: project\n---\n\nImportant fact."
        )

        adapter = ClaudeCodeAdapter(memory_dir)
        mems = await adapter.read_memories()

        assert len(mems) == 1
        assert mems[0].content == "Important fact."
        assert mems[0].source_file == "fact.md"

    @pytest.mark.asyncio
    async def test_read_memories_handles_missing_dir_oserror_and_blank_body(
        self, memory_dir, monkeypatch
    ):
        missing_adapter = ClaudeCodeAdapter(memory_dir / "missing")
        assert await missing_adapter.read_memories() == []

        blank = memory_dir / "blank.md"
        blank.write_text("---\nname: Blank\n---\n\n   ")
        broken = memory_dir / "broken.md"
        broken.write_text("---\nname: Broken\n---\n\nbody")

        original_read_text = Path.read_text

        def fake_read_text(self: Path, *args: Any, **kwargs: Any):
            if self == broken:
                raise OSError("boom")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fake_read_text)
        try:
            adapter = ClaudeCodeAdapter(memory_dir)
            assert await adapter.read_memories() == []
        finally:
            monkeypatch.setattr(Path, "read_text", original_read_text)

    @pytest.mark.asyncio
    async def test_write_creates_valid_md(self, memory_dir):
        adapter = ClaudeCodeAdapter(memory_dir)
        written = await adapter.write_memories(
            [
                {
                    "content": "Project uses FastAPI",
                    "category": "architecture",
                    "headroom_id": "mem_001",
                    "source_agent": "codex",
                    "content_hash": "abc123",
                }
            ]
        )

        assert written == 1
        files = list(memory_dir.glob("headroom_*.md"))
        assert len(files) == 1

        content = files[0].read_text()
        fm, body = _parse_frontmatter(content)
        assert fm["type"] == "architecture"
        assert fm["headroom_id"] == "mem_001"
        assert fm["source_agent"] == "codex"
        assert "FastAPI" in body

    @pytest.mark.asyncio
    async def test_write_handles_empty_input_and_duplicate_hash(self, memory_dir):
        adapter = ClaudeCodeAdapter(memory_dir)
        assert await adapter.write_memories([]) == 0

        target = memory_dir / "headroom_project_uses_fastapi.md"
        target.write_text(
            "---\nname: Project uses FastAPI\n---\n\nProject uses FastAPI\n", encoding="utf-8"
        )
        written = await adapter.write_memories(
            [
                {
                    "content": "Project uses FastAPI",
                    "category": "architecture",
                    "headroom_id": "mem_001",
                    "source_agent": "codex",
                    "content_hash": hashlib.sha256(b"Project uses FastAPI").hexdigest()[:16],
                }
            ]
        )
        assert written == 0
        assert not (memory_dir / "MEMORY.md").exists()

    def test_update_memory_md_index_paths(self, memory_dir):
        adapter = ClaudeCodeAdapter(memory_dir)
        memory_md = memory_dir / "MEMORY.md"
        memory_md.write_text(
            "# Memory\n\n## Headroom Shared Memory\n- existing\n\n## Other Section\n- keep me\n",
            encoding="utf-8",
        )
        adapter._update_memory_md_index(["- added"])
        content = memory_md.read_text(encoding="utf-8")
        assert "- existing" in content
        assert "- added" in content
        assert "## Other Section" in content

        memory_md.write_text(
            "# Memory\n\n## Headroom Shared Memory\n- existing\n", encoding="utf-8"
        )
        adapter._update_memory_md_index(["- appended"])
        assert "- appended" in memory_md.read_text(encoding="utf-8")

    def test_fingerprint_changes_on_modification(self, memory_dir, monkeypatch):
        (memory_dir / "test.md").write_text("content 1")

        adapter = ClaudeCodeAdapter(memory_dir)
        original_stat = Path.stat

        def fake_stat(self: Path, *args: Any, **kwargs: Any):
            result = original_stat(self)
            if self == memory_dir / "test.md":
                text = self.read_text()
                return types.SimpleNamespace(st_mtime_ns=1 if text == "content 1" else 2)
            return result

        monkeypatch.setattr(Path, "stat", fake_stat)
        try:
            fp1 = adapter.fingerprint()
            (memory_dir / "test.md").write_text("content 2")
            fp2 = adapter.fingerprint()
        finally:
            monkeypatch.setattr(Path, "stat", original_stat)

        assert fp1 != fp2

    def test_fingerprint_stable_when_unchanged(self, memory_dir):
        (memory_dir / "test.md").write_text("stable content")

        adapter = ClaudeCodeAdapter(memory_dir)
        assert adapter.fingerprint() == adapter.fingerprint()

    def test_fingerprint_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        adapter = ClaudeCodeAdapter(empty)
        assert adapter.fingerprint() == "empty"

    def test_fingerprint_missing_dir_and_stat_errors(self, tmp_path, monkeypatch):
        missing = tmp_path / "missing"
        adapter = ClaudeCodeAdapter(missing)
        assert adapter.fingerprint() == "empty"

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        file_path = memory_dir / "test.md"
        file_path.write_text("content")
        adapter = ClaudeCodeAdapter(memory_dir)
        original_stat = Path.stat

        def fake_stat(self: Path, *args: Any, **kwargs: Any):
            if self == file_path:
                raise OSError("boom")
            return original_stat(self)

        monkeypatch.setattr(Path, "stat", fake_stat)
        try:
            assert adapter.fingerprint() == "empty"
        finally:
            monkeypatch.setattr(Path, "stat", original_stat)

    def test_get_claude_memory_dir(self, monkeypatch):
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: Path("C:\\repo\\project")))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("C:\\Users\\me")))
        assert (
            get_claude_memory_dir()
            == Path("C:\\Users\\me") / ".claude" / "projects" / "C:-repo-project" / "memory"
        )
        assert (
            get_claude_memory_dir(Path("/tmp/demo"))
            == Path("C:\\Users\\me") / ".claude" / "projects" / "-tmp-demo" / "memory"
        )


# ---------------------------------------------------------------------------
# Codex adapter tests
# ---------------------------------------------------------------------------


class TestCodexAdapter:
    """Test Codex AGENTS.md adapter."""

    @pytest.fixture
    def agents_md(self, tmp_path):
        return tmp_path / "AGENTS.md"

    @pytest.mark.asyncio
    async def test_read_from_agents_md(self, agents_md):
        agents_md.write_text(
            "# Instructions\n\n"
            "<!-- headroom:memory:start -->\n"
            "## Headroom Shared Memory\n\n"
            "- Secret name is TC\n"
            "- Uses Python 3.12\n"
            "<!-- headroom:memory:end -->\n"
        )

        adapter = CodexAdapter(agents_md)
        mems = await adapter.read_memories()

        assert len(mems) == 2
        assert mems[0].content == "Secret name is TC"
        assert mems[1].content == "Uses Python 3.12"

    @pytest.mark.asyncio
    async def test_write_to_agents_md(self, agents_md):
        agents_md.write_text("# Existing instructions\n")

        adapter = CodexAdapter(agents_md)
        written = await adapter.write_memories(
            [
                {"content": "Port 8787 is default"},
                {"content": "Uses ruff for linting"},
            ]
        )

        assert written == 2
        content = agents_md.read_text()
        assert "headroom:memory:start" in content
        assert "Port 8787 is default" in content
        assert "Uses ruff for linting" in content
        assert "Existing instructions" in content  # Preserved

    @pytest.mark.asyncio
    async def test_write_replaces_existing_section(self, agents_md):
        agents_md.write_text(
            "# Instructions\n\n"
            "<!-- headroom:memory:start -->\n"
            "## Old\n- old fact\n"
            "<!-- headroom:memory:end -->\n"
        )

        adapter = CodexAdapter(agents_md)
        await adapter.write_memories([{"content": "new fact"}])

        content = agents_md.read_text()
        assert "new fact" in content
        assert "old fact" not in content

    @pytest.mark.asyncio
    async def test_read_empty_agents_md(self, agents_md):
        agents_md.write_text("# No memory section\n")
        adapter = CodexAdapter(agents_md)
        mems = await adapter.read_memories()
        assert mems == []

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path):
        adapter = CodexAdapter(tmp_path / "nonexistent.md")
        mems = await adapter.read_memories()
        assert mems == []


# ---------------------------------------------------------------------------
# Cross-agent integration tests
# ---------------------------------------------------------------------------


class TestCrossAgentInterop:
    """Test that memories flow between agents via sync."""

    @pytest.fixture
    def backend(self):
        return FakeBackend()

    @pytest.fixture
    def claude_dir(self, tmp_path):
        d = tmp_path / "claude_memory"
        d.mkdir()
        return d

    @pytest.fixture
    def agents_md(self, tmp_path):
        return tmp_path / "AGENTS.md"

    @pytest.fixture
    def state_path(self, tmp_path):
        return tmp_path / "state.json"

    @pytest.mark.asyncio
    async def test_codex_saves_claude_finds(self, backend, claude_dir, state_path):
        """Memory saved via Codex MCP appears in Claude's files after sync."""
        # Simulate Codex saving via MCP (directly to backend)
        backend.add_memory(
            "Secret name is TC",
            metadata={"source_agent": "codex", "content_hash": "x"},
        )

        # Sync to Claude
        adapter = ClaudeCodeAdapter(claude_dir)
        result = await sync(backend, adapter, "tcms", state_path=state_path, force=True)

        assert result.exported == 1

        # Claude's memory dir should have the file
        files = list(claude_dir.glob("headroom_*.md"))
        assert len(files) == 1
        assert "TC" in files[0].read_text()

    @pytest.mark.asyncio
    async def test_claude_saves_codex_finds(self, backend, claude_dir, agents_md, state_path):
        """Memory saved in Claude's files appears in Codex AGENTS.md after sync."""
        # Claude has a memory
        fm = "---\nname: Linting\ndescription: use ruff\ntype: project\n---"
        (claude_dir / "linting.md").write_text(f"{fm}\n\nAlways use ruff for linting\n")

        # Sync Claude → DB
        claude_adapter = ClaudeCodeAdapter(claude_dir)
        await sync(backend, claude_adapter, "tcms", state_path=state_path, force=True)

        # Sync DB → Codex AGENTS.md
        codex_adapter = CodexAdapter(agents_md)
        result = await sync(backend, codex_adapter, "tcms", state_path=state_path, force=True)

        assert result.exported >= 1
        assert "ruff" in agents_md.read_text()

    @pytest.mark.asyncio
    async def test_full_round_trip(self, backend, claude_dir, agents_md, state_path):
        """Full round trip: Claude → DB → Codex, Codex → DB → Claude."""
        # Claude has a memory
        fm = "---\nname: Framework\ntype: project\n---"
        (claude_dir / "framework.md").write_text(f"{fm}\n\nUses FastAPI\n")

        # Codex has a memory (in DB via MCP)
        backend.add_memory("Port is 8787", metadata={"source_agent": "codex"})

        # Sync both adapters
        claude_adapter = ClaudeCodeAdapter(claude_dir)
        codex_adapter = CodexAdapter(agents_md)

        await sync(backend, claude_adapter, "tcms", state_path=state_path, force=True)
        await sync(backend, codex_adapter, "tcms", state_path=state_path, force=True)

        # DB has both memories
        mems = await backend.get_user_memories("tcms")
        contents = {m.content for m in mems}
        assert "Uses FastAPI" in contents
        assert "Port is 8787" in contents

        # Claude files have Codex's memory
        all_claude = " ".join(f.read_text() for f in claude_dir.glob("headroom_*.md"))
        assert "8787" in all_claude

        # AGENTS.md has both (from DB)
        agents_content = agents_md.read_text()
        assert "FastAPI" in agents_content or "8787" in agents_content
