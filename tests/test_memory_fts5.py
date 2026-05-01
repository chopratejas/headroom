from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


def _load_fts5(monkeypatch):
    @dataclass
    class Memory:
        id: str
        content: str
        user_id: str
        session_id: str | None = None

    @dataclass
    class TextFilter:
        query: str
        user_id: str | None = None
        session_id: str | None = None
        limit: int = 100

    @dataclass
    class TextSearchResult:
        memory: Memory
        score: float
        rank: int

    models_module = types.ModuleType("headroom.memory.models")
    models_module.Memory = Memory
    ports_module = types.ModuleType("headroom.memory.ports")
    ports_module.TextFilter = TextFilter
    ports_module.TextSearchResult = TextSearchResult
    monkeypatch.setitem(sys.modules, "headroom.memory.models", models_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.ports", ports_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "adapters" / "fts5.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.adapters.fts5", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.adapters.fts5", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.adapters.fts5"] = module
    spec.loader.exec_module(module)
    return module, Memory, TextFilter


@pytest.mark.asyncio
async def test_fts5_index_search_update_and_async_helpers(monkeypatch, tmp_path: Path) -> None:
    fts5, Memory, TextFilter = _load_fts5(monkeypatch)
    db_path = tmp_path / "memory.db"
    index = fts5.FTS5TextIndex(db_path)

    assert index.db_path == db_path
    assert index.count() == 0
    assert index._sanitize_fts_query('hello "world"!!!') == '"hello" OR "world"'
    assert index._sanitize_fts_query("!!!") == ""
    assert index.search("!!!") == []

    index.index_raw("m1", "Alice likes Python", {"user_id": "u1", "session_id": "s1"})
    index.index("m1", "Alice likes Python and SQL", {"user_id": "u1", "session_id": "s1"})
    index.index_batch(
        ["m2", "m3"],
        ["Bob uses Python", "Carol prefers Rust"],
        [{"user_id": "u1", "session_id": "s2"}, {"user_id": "u2", "session_id": "s3"}],
    )

    with pytest.raises(ValueError, match="same length"):
        index.index_batch(["m1"], ["one", "two"])

    with pytest.raises(ValueError, match="must match memory_ids"):
        index.index_batch(["m1", "m2"], ["one", "two"], metadata=[{}])

    results = index.search("Python", k=5)
    assert {result.memory_id for result in results} == {"m1", "m2"}
    assert all(result.score >= 0.0 for result in results)
    assert results[0].metadata["category"] == ""

    filtered = index.search(
        "Python", filter=TextFilter(query="Python", user_id="u1", session_id="s1")
    )
    assert [result.memory_id for result in filtered] == ["m1"]

    assert await index.index_batch_memories([]) == 0
    assert (
        await index.index_batch_memories(
            [
                Memory(id="m4", content="Dave uses Go", user_id="u1", session_id="s4"),
                Memory(id="m5", content="Erin tests SQLite", user_id="u3"),
            ]
        )
        == 2
    )
    await index.index_memory(Memory(id="m6", content="Frank uses Python", user_id="u4"))

    search_results = await index.search_memories(TextFilter(query="Python", limit=3))
    assert {result.memory.id for result in search_results} == {"m1", "m2", "m6"}
    assert search_results[0].rank == 1
    assert {result.memory.id: result.memory.user_id for result in search_results} == {
        "m1": "u1",
        "m2": "u1",
        "m6": "u4",
    }

    assert await index.update_content("m2", "Bob now uses TypeScript") is True
    assert index.search("TypeScript")[0].memory_id == "m2"
    assert await index.update_content("missing", "ignored") is False

    assert index.delete("m3") is True
    assert index.delete("missing") is False
    assert await index.remove("m4") is True
    assert await index.remove_batch(["m5", "missing", "m6"]) == 2

    assert index.count() == 2
    index.clear()
    assert index.count() == 0
