from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest


def _load_core_module(monkeypatch):
    class ScopeLevel(Enum):
        USER = "user"
        SESSION = "session"
        AGENT = "agent"
        TURN = "turn"

    @dataclass
    class Memory:
        content: str
        user_id: str
        session_id: str | None = None
        agent_id: str | None = None
        turn_id: str | None = None
        importance: float = 0.5
        entity_refs: list[str] = field(default_factory=list)
        metadata: dict = field(default_factory=dict)
        embedding: list[float] | None = None
        promoted_from: str | None = None
        promotion_chain: list[str] = field(default_factory=list)
        id: str = field(default_factory=lambda: f"m-{Memory._counter()}")
        supersedes: str | None = None
        superseded_by: str | None = None
        valid_until: datetime | None = None

        _next_id = 0

        @classmethod
        def _counter(cls):
            cls._next_id += 1
            return cls._next_id

        @property
        def scope_level(self):
            if self.turn_id:
                return ScopeLevel.TURN
            if self.agent_id:
                return ScopeLevel.AGENT
            if self.session_id:
                return ScopeLevel.SESSION
            return ScopeLevel.USER

    @dataclass
    class MemoryConfig:
        auto_bubble: bool = False
        bubble_threshold: float = 0.8

    @dataclass
    class MemoryFilter:
        user_id: str | None = None
        session_id: str | None = None
        agent_id: str | None = None
        turn_id: str | None = None
        include_superseded: bool = False
        limit: int | None = None
        scope_levels: list[ScopeLevel] | None = None

    @dataclass
    class TextFilter:
        query: str
        user_id: str | None = None
        session_id: str | None = None
        limit: int = 100

    @dataclass
    class VectorFilter:
        query_vector: list[float]
        top_k: int
        min_similarity: float
        user_id: str | None = None
        session_id: str | None = None
        agent_id: str | None = None
        scope_levels: list[ScopeLevel] | None = None
        include_superseded: bool = False

    @dataclass
    class VectorSearchResult:
        memory: Memory
        similarity: float
        rank: int = 1

    @dataclass
    class TextSearchResult:
        memory: Memory
        score: float

    config_module = types.ModuleType("headroom.memory.config")
    config_module.MemoryConfig = MemoryConfig

    factory_module = types.ModuleType("headroom.memory.factory")

    async def create_memory_system(config=None):
        return factory_module._components

    factory_module.create_memory_system = create_memory_system
    factory_module._components = None

    models_module = types.ModuleType("headroom.memory.models")
    models_module.Memory = Memory
    models_module.ScopeLevel = ScopeLevel

    ports_module = types.ModuleType("headroom.memory.ports")
    ports_module.MemoryFilter = MemoryFilter
    ports_module.TextFilter = TextFilter
    ports_module.VectorFilter = VectorFilter
    ports_module.TextSearchResult = TextSearchResult
    ports_module.VectorSearchResult = VectorSearchResult

    monkeypatch.setitem(sys.modules, "headroom.memory.config", config_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.factory", factory_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.models", models_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.ports", ports_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "core.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.core", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.core", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.core"] = module
    spec.loader.exec_module(module)
    return (
        module,
        factory_module,
        Memory,
        MemoryConfig,
        MemoryFilter,
        ScopeLevel,
        VectorSearchResult,
        TextSearchResult,
    )


class FakeStore:
    def __init__(self):
        self.memories = {}
        self.saved = []
        self.deleted = []
        self.closed = False
        self.query_results = []
        self.history_results = []
        self.count_value = 0
        self.clear_count = 0

    async def save(self, memory):
        self.memories[memory.id] = memory
        self.saved.append(memory.id)

    async def save_batch(self, memories):
        for memory in memories:
            self.memories[memory.id] = memory
        self.saved.extend(m.id for m in memories)

    async def get(self, memory_id):
        return self.memories.get(memory_id)

    async def query(self, filt):
        self.last_query = filt
        return self.query_results

    async def count(self, filt):
        self.last_count_filter = filt
        return self.count_value

    async def supersede(self, old_memory_id, new_memory, supersede_time):
        old = self.memories[old_memory_id]
        old.superseded_by = new_memory.id
        old.valid_until = supersede_time or datetime(2024, 1, 1)
        new_memory.supersedes = old_memory_id
        self.memories[new_memory.id] = new_memory
        return new_memory

    async def get_history(self, memory_id, include_future):
        self.last_history = (memory_id, include_future)
        return self.history_results

    async def delete(self, memory_id):
        self.deleted.append(memory_id)
        return self.memories.pop(memory_id, None) is not None

    async def clear_scope(self, user_id, session_id, agent_id, turn_id):
        self.last_clear = (user_id, session_id, agent_id, turn_id)
        return self.clear_count

    async def close(self):
        self.closed = True


class FakeVectorIndex:
    def __init__(self):
        self.indexed = []
        self.removed = []
        self.search_results = []
        self.closed = False

    async def index(self, memory):
        self.indexed.append(memory.id)

    async def index_batch(self, memories):
        self.indexed.extend(m.id for m in memories)

    async def search(self, filt):
        self.last_filter = filt
        return self.search_results

    async def remove(self, memory_id):
        self.removed.append(memory_id)

    async def remove_batch(self, memory_ids):
        self.removed.extend(memory_ids)

    async def close(self):
        self.closed = True


class FakeTextIndex:
    def __init__(self):
        self.indexed = []
        self.removed = []
        self.search_results = []
        self.closed = False

    async def search(self, filt):
        self.last_filter = filt
        return self.search_results

    async def index_memory(self, memory):
        self.indexed.append(memory.id)

    async def remove(self, memory_id):
        self.removed.append(memory_id)

    async def close(self):
        self.closed = True


class FakeEmbedder:
    def __init__(self):
        self.closed = False
        self.calls = []

    async def embed(self, text):
        self.calls.append(("embed", text))
        return [float(len(text))]

    async def embed_batch(self, texts):
        self.calls.append(("embed_batch", list(texts)))
        return [[float(len(t))] for t in texts]

    async def close(self):
        self.closed = True


class FakeCache:
    def __init__(self):
        self.items = {}
        self.invalidated = []
        self.scopes = []
        self.closed = False

    async def get(self, memory_id):
        return self.items.get(memory_id)

    async def put(self, memory):
        self.items[memory.id] = memory

    async def put_batch(self, memories):
        for memory in memories:
            self.items[memory.id] = memory

    async def invalidate(self, memory_id):
        self.invalidated.append(memory_id)
        self.items.pop(memory_id, None)

    async def invalidate_scope(self, user_id, session_id, agent_id):
        self.scopes.append((user_id, session_id, agent_id))

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_hierarchical_memory_full_flow(monkeypatch) -> None:
    (
        core,
        factory,
        Memory,
        MemoryConfig,
        MemoryFilter,
        ScopeLevel,
        VectorSearchResult,
        TextSearchResult,
    ) = _load_core_module(monkeypatch)
    store = FakeStore()
    vector = FakeVectorIndex()
    text = FakeTextIndex()
    embedder = FakeEmbedder()
    cache = FakeCache()
    factory._components = (store, vector, text, embedder, cache)

    created = await core.HierarchicalMemory.create(MemoryConfig(auto_bubble=False))
    assert created.store is store
    assert created.vector_index is vector
    assert created.text_index is text
    assert created.embedder is embedder
    assert created.cache is cache

    system = core.HierarchicalMemory(
        store, vector, text, embedder, cache, MemoryConfig(auto_bubble=False)
    )
    added = await system.add(
        content="hello world",
        user_id="u1",
        session_id="s1",
        importance=0.3,
        entity_refs=["Python"],
        metadata={"a": 1},
    )
    assert added.embedding == [11.0]
    assert added.id in store.memories
    assert vector.indexed == [added.id]
    assert text.indexed == [added.id]
    assert cache.items[added.id] is added

    no_embed = await system.add("short", "u1", auto_embed=False, auto_bubble=False)
    assert no_embed.embedding is None

    batch = await system.add_batch(
        [
            {"content": "one", "user_id": "u1"},
            {"content": "two", "user_id": "u1", "entity_refs": ["A"]},
        ]
    )
    assert len(batch) == 2
    assert ("embed_batch", ["one", "two"]) in embedder.calls
    assert batch[0].id in cache.items

    cache.items[added.id] = added
    assert await system.get(added.id) is added
    await cache.invalidate(added.id)
    assert await system.get(added.id) is added

    store.query_results = [added]
    assert await system.query(MemoryFilter(user_id="u1")) == [added]
    store.count_value = 7
    assert await system.count(MemoryFilter(user_id="u1")) == 7

    vector.search_results = [VectorSearchResult(memory=added, similarity=0.8)]
    results = await system.search(
        "hello",
        user_id="u1",
        session_id="s1",
        agent_id="a1",
        top_k=3,
        min_similarity=0.2,
        scope_levels=[ScopeLevel.USER],
        include_superseded=True,
    )
    assert results[0].memory is added
    assert vector.last_filter.user_id == "u1"
    assert vector.last_filter.include_superseded is True

    text.search_results = [TextSearchResult(memory=added, score=1.5)]
    text_results = await system.text_search("hello", user_id="u1", session_id="s1", limit=2)
    assert text_results[0].score == 1.5
    assert text.last_filter.session_id == "s1"

    updated = await system.update(
        added.id, content="changed", importance=0.9, entity_refs=["JS"], metadata={"b": 2}
    )
    assert updated.content == "changed"
    assert updated.importance == 0.9
    assert updated.metadata["b"] == 2
    assert added.id in cache.invalidated
    assert vector.indexed[-1] == added.id

    unchanged = await system.update(no_embed.id, importance=0.6, re_embed=False)
    assert unchanged.importance == 0.6
    assert await system.update("missing") is None

    history_seed = Memory(
        content="old", user_id="u1", session_id="s1", entity_refs=["X"], metadata={"m": 1}
    )
    store.memories[history_seed.id] = history_seed
    superseded = await system.supersede(history_seed.id, "new text")
    assert superseded.supersedes == history_seed.id
    assert superseded.id in vector.indexed
    with pytest.raises(ValueError):
        await system.supersede("missing", "new")
    store.history_results = [history_seed, superseded]
    assert await system.get_history(history_seed.id, include_future=True) == [
        history_seed,
        superseded,
    ]

    assert await system.delete("missing") is False
    store.memories[added.id] = added
    assert await system.delete(added.id) is True
    assert added.id in vector.removed
    assert added.id in text.removed

    store.query_results = []
    assert await system.clear_scope("u1") == 0
    mem_a = Memory(content="a", user_id="u1")
    mem_b = Memory(content="b", user_id="u1")
    store.query_results = [mem_a, mem_b]
    store.clear_count = 2
    assert await system.clear_scope("u1", session_id="s1", agent_id="a1", turn_id="t1") == 2
    assert vector.removed[-2:] == [mem_a.id, mem_b.id]
    assert cache.scopes[-1] == ("u1", "s1", "a1")

    remembered = await system.remember("remember me", "u1", session_id="s2", importance=0.7)
    assert remembered.content == "remember me"
    vector.search_results = [VectorSearchResult(memory=remembered, similarity=0.9)]
    assert await system.recall("remember", "u1") == [remembered]

    store.query_results = [remembered]
    user_only = await system.get_user_memories("u1", include_sessions=False)
    assert user_only == [remembered]
    assert store.last_query.scope_levels == [ScopeLevel.USER]
    session_memories = await system.get_session_memories("u1", "s2")
    assert session_memories == [remembered]
    assert store.last_query.session_id == "s2"

    assert system._scope_level_value(ScopeLevel.USER) == 0
    assert system._scope_level_value(ScopeLevel.TURN) == 3

    await system.close()
    assert embedder.closed is True
    assert store.closed is True
    assert vector.closed is True
    assert text.closed is True
    assert cache.closed is True


@pytest.mark.asyncio
async def test_hierarchical_memory_bubbling_and_context_manager(monkeypatch) -> None:
    (
        core,
        _factory,
        Memory,
        MemoryConfig,
        _MemoryFilter,
        ScopeLevel,
        _VectorSearchResult,
        _TextSearchResult,
    ) = _load_core_module(monkeypatch)
    store = FakeStore()
    vector = FakeVectorIndex()
    text = FakeTextIndex()
    embedder = FakeEmbedder()
    config = MemoryConfig(auto_bubble=True, bubble_threshold=0.8)
    system = core.HierarchicalMemory(store, vector, text, embedder, None, config)

    low = Memory(content="low", user_id="u1", session_id="s1", importance=0.5, embedding=[1.0])
    await system._maybe_bubble(low)
    assert store.saved == []

    top = Memory(content="top", user_id="u1", importance=0.95, embedding=[2.0])
    await system._maybe_bubble(top)
    assert store.saved == []

    candidate = Memory(
        content="important",
        user_id="u1",
        session_id="s1",
        agent_id="a1",
        turn_id="t1",
        importance=0.95,
        entity_refs=["Py"],
        metadata={"m": 1},
        embedding=[3.0],
    )
    await system._maybe_bubble(candidate)
    bubbled_id = store.saved[-1]
    bubbled = store.memories[bubbled_id]
    assert bubbled.user_id == "u1"
    assert bubbled.session_id is None
    assert bubbled.promoted_from == candidate.id
    assert bubbled.promotion_chain[-1] == candidate.id
    assert vector.indexed[-1] == bubbled.id
    assert text.indexed[-1] == bubbled.id

    async with core.HierarchicalMemory(store, vector, text, embedder, None, config) as managed:
        assert managed.config is config
