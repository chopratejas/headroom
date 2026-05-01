from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_mem0_modules(monkeypatch):
    @dataclass
    class Memory:
        id: str
        content: str
        user_id: str
        session_id: str | None = None
        agent_id: str | None = None
        turn_id: str | None = None
        created_at: datetime = field(
            default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
        )
        valid_from: datetime = field(
            default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
        )
        valid_until: datetime | None = None
        importance: float = 0.5
        supersedes: str | None = None
        superseded_by: str | None = None
        promoted_from: str | None = None
        promotion_chain: list[str] = field(default_factory=list)
        access_count: int = 0
        last_accessed: datetime | None = None
        entity_refs: list[str] = field(default_factory=list)
        embedding: object | None = None
        metadata: dict = field(default_factory=dict)

    @dataclass
    class MemoryFilter:
        user_id: str | None = None
        session_id: str | None = None
        agent_id: str | None = None
        turn_id: str | None = None
        min_importance: float | None = None
        max_importance: float | None = None
        created_after: datetime | None = None
        created_before: datetime | None = None
        include_superseded: bool = True
        entity_refs: list[str] = field(default_factory=list)
        order_by: str = "created_at"
        order_desc: bool = False
        offset: int = 0
        limit: int | None = None

    @dataclass
    class VectorFilter:
        user_id: str | None = None
        session_id: str | None = None
        agent_id: str | None = None

    @dataclass
    class VectorSearchResult:
        memory: Memory
        similarity: float
        rank: int

    @dataclass
    class MemorySearchResult:
        memory: Memory
        score: float
        related_entities: list[str] = field(default_factory=list)
        related_memories: list[str] = field(default_factory=list)

    models_module = types.ModuleType("headroom.memory.models")
    models_module.Memory = Memory
    ports_module = types.ModuleType("headroom.memory.ports")
    ports_module.MemoryFilter = MemoryFilter
    ports_module.VectorFilter = VectorFilter
    ports_module.VectorSearchResult = VectorSearchResult
    ports_module.MemorySearchResult = MemorySearchResult

    backends_pkg = types.ModuleType("headroom.memory.backends")
    backends_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.memory.backends", backends_pkg)
    monkeypatch.setitem(sys.modules, "headroom.memory.models", models_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.ports", ports_module)

    repo_root = Path(__file__).resolve().parents[1]
    mem0_path = repo_root / "headroom" / "memory" / "backends" / "mem0.py"
    adapter_path = repo_root / "headroom" / "memory" / "backends" / "mem0_system_adapter.py"

    monkeypatch.delitem(sys.modules, "headroom.memory.backends.mem0", raising=False)
    mem0_spec = importlib.util.spec_from_file_location("headroom.memory.backends.mem0", mem0_path)
    assert mem0_spec and mem0_spec.loader
    mem0_module = importlib.util.module_from_spec(mem0_spec)
    sys.modules["headroom.memory.backends.mem0"] = mem0_module
    mem0_spec.loader.exec_module(mem0_module)

    monkeypatch.delitem(sys.modules, "headroom.memory.backends.mem0_system_adapter", raising=False)
    adapter_spec = importlib.util.spec_from_file_location(
        "headroom.memory.backends.mem0_system_adapter", adapter_path
    )
    assert adapter_spec and adapter_spec.loader
    adapter_module = importlib.util.module_from_spec(adapter_spec)
    sys.modules["headroom.memory.backends.mem0_system_adapter"] = adapter_module
    adapter_spec.loader.exec_module(adapter_module)

    return (
        mem0_module,
        adapter_module,
        Memory,
        MemoryFilter,
        VectorFilter,
        VectorSearchResult,
        MemorySearchResult,
    )


@pytest.mark.asyncio
async def test_mem0_backend_core_paths(monkeypatch) -> None:
    mem0, adapter, Memory, MemoryFilter, VectorFilter, _VSR, _MSR = _load_mem0_modules(monkeypatch)

    backend = mem0.Mem0Backend(mem0.Mem0Config(mode="cloud", api_key="key"))

    class FakeMem0Memory:
        def __init__(self, api_key=None):
            self.api_key = api_key

        @classmethod
        def from_config(cls, config):
            return FakeClient(config=config)

    class FakeClient:
        def __init__(self, config=None):
            self.config = config
            self.add_calls = []
            self.search_result = []
            self.get_result = None
            self.get_all_result = []
            self.raise_update = False
            self.raise_delete = False

        def add(self, content, **kwargs):
            self.add_calls.append((content, kwargs))
            return {"results": [{"id": "mem-added"}]}

        def search(self, **kwargs):
            return self.search_result

        def update(self, **kwargs):
            if self.raise_update:
                raise RuntimeError("update failed")
            self.updated = kwargs

        def delete(self, **kwargs):
            if self.raise_delete:
                raise RuntimeError("delete failed")

        def get(self, **kwargs):
            result = self.get_result
            if isinstance(result, Exception):
                raise result
            return result

        def get_all(self, **kwargs):
            result = self.get_all_result
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(Memory=FakeMem0Memory))

    client = await backend._ensure_client()
    assert client.api_key == "key"
    assert await backend._ensure_client() is client

    local_backend = mem0.Mem0Backend(mem0.Mem0Config(mode="local", enable_graph=True))
    local_client = FakeClient()
    monkeypatch.setitem(
        sys.modules,
        "mem0",
        types.SimpleNamespace(
            Memory=type(
                "Mem0Factory",
                (),
                {
                    "from_config": classmethod(lambda cls, config: local_client),
                    "__call__": lambda self, **kwargs: None,
                },
            )
        ),
    )
    await local_backend._ensure_client()
    assert local_backend._initialized is True

    memory = Memory(
        id="m1",
        content="hello",
        user_id="user",
        session_id="sess",
        agent_id="agent",
        turn_id="turn",
        importance=0.9,
        supersedes="old",
        superseded_by="new",
        promoted_from="src",
        promotion_chain=["a"],
        access_count=2,
        last_accessed=datetime(2024, 1, 2),
        entity_refs=["python"],
        metadata={"k": "v"},
    )
    metadata = local_backend._build_mem0_metadata(memory)
    assert metadata["custom_metadata"] == {"k": "v"}

    restored = local_backend._mem0_result_to_memory(
        {
            "id": "mem0-id",
            "memory": "stored",
            "metadata": {
                "headroom_id": "m1",
                "user_id": "user",
                "importance": 0.8,
                "created_at": "2024-01-01T00:00:00",
                "valid_from": "2024-01-01T00:00:00",
                "valid_until": "2024-01-02T00:00:00",
                "last_accessed": "2024-01-03T00:00:00",
                "entity_refs": ["python"],
                "custom_metadata": {"x": 1},
            },
        }
    )
    assert restored.id == "m1"
    assert restored.metadata == {"x": 1}
    assert local_backend._build_mem0_filters(
        VectorFilter(user_id="u", session_id="s", agent_id="a")
    ) == {
        "user_id": "u",
        "session_id": "s",
        "agent_id": "a",
    }

    local_backend._client = local_client
    assert await local_backend.save_memory(memory) == "mem-added"
    local_client.add = lambda content, **kwargs: [{"id": "list-id"}]
    assert await local_backend.save_memory(memory) == "list-id"
    local_client.add = lambda content, **kwargs: {}
    assert await local_backend.save_memory(memory) == "m1"
    assert await local_backend.save_memory_batch([memory, memory]) == ["m1", "m1"]

    local_client.search_result = [
        {"id": "a", "memory": "foo", "metadata": {"user_id": "user"}, "score": 0.9},
        {"id": "b", "memory": "bar", "metadata": {"user_id": "user"}, "similarity": 0.5},
    ]
    search = await local_backend.search_memories(
        "q", user_id="user", filter=VectorFilter(agent_id="agent"), limit=2
    )
    assert len(search) == 2
    assert search[0].rank == 1

    assert await local_backend.update_memory("id", content="new", metadata={"x": 1}) is True
    local_client.raise_update = True
    assert await local_backend.update_memory("id", content="new") is False
    local_client.raise_update = False
    assert await local_backend.delete_memory("id") is True
    local_client.raise_delete = True
    assert await local_backend.delete_memory("id") is False
    local_client.raise_delete = False
    local_backend.delete_memory = lambda memory_id: asyncio.sleep(0, result=(memory_id != "bad"))
    assert await local_backend.delete_memory_batch(["ok", "bad", "ok2"]) == 2

    local_client.get_result = {"id": "x", "memory": "one", "metadata": {"user_id": "u"}}
    assert (await local_backend.get_memory("x")).content == "one"
    local_client.get_result = [{"id": "x", "memory": "two", "metadata": {"user_id": "u"}}]
    assert (await local_backend.get_memory("x")).content == "two"
    local_client.get_result = None
    assert await local_backend.get_memory("x") is None
    local_client.get_result = RuntimeError("boom")
    assert await local_backend.get_memory("x") is None

    local_client.get_all_result = [
        {"id": "x", "memory": "one", "metadata": {"user_id": "u"}},
        {"id": "y", "memory": "two", "metadata": {"user_id": "u"}},
    ]
    assert len(await local_backend.get_all_memories("u")) == 2
    local_client.get_all_result = RuntimeError("boom")
    assert await local_backend.get_all_memories("u") == []

    query_backend = mem0.Mem0Backend()
    query_backend._client = FakeClient()
    query_backend._initialized = True
    query_backend._client.get_all_result = [
        {
            "id": "1",
            "memory": "alpha",
            "metadata": {
                "user_id": "u",
                "session_id": "s1",
                "agent_id": "a1",
                "turn_id": "t1",
                "importance": 0.8,
                "created_at": "2024-01-02T00:00:00",
                "valid_from": "2024-01-02T00:00:00",
                "entity_refs": ["python"],
            },
        },
        {
            "id": "2",
            "memory": "beta",
            "metadata": {
                "user_id": "u",
                "session_id": "s2",
                "importance": 0.2,
                "created_at": "2024-01-01T00:00:00",
                "valid_from": "2024-01-01T00:00:00",
                "valid_until": "2024-01-03T00:00:00",
                "entity_refs": ["java"],
            },
        },
    ]
    filtered = await query_backend.query(
        MemoryFilter(user_id="u", session_id="s1", entity_refs=["python"], include_superseded=False)
    )
    assert [m.id for m in filtered] == ["1"]
    assert await query_backend.query(MemoryFilter()) == []
    assert query_backend.supports_graph() is True
    assert query_backend.supports_vector_search() is True

    related_backend = mem0.Mem0Backend()
    related_backend.get_memory = lambda memory_id: asyncio.sleep(
        0, result=Memory(id="1", content="query", user_id="u")
    )
    related_backend.search_memories = lambda **kwargs: asyncio.sleep(
        0,
        result=[
            SimpleNamespace(
                memory=Memory(id="1", content="query", user_id="u"), similarity=1.0, rank=1
            ),
            SimpleNamespace(
                memory=Memory(id="2", content="related", user_id="u"), similarity=0.9, rank=2
            ),
        ],
    )
    assert [m.id for m in await related_backend.get_related_memories("1")] == ["2"]
    related_backend.get_memory = lambda memory_id: asyncio.sleep(0, result=None)
    assert await related_backend.get_related_memories("1") == []
    await related_backend.close()
    assert related_backend._client is None


@pytest.mark.asyncio
async def test_mem0_system_adapter_paths(monkeypatch) -> None:
    mem0, adapter, Memory, _MemoryFilter, _VectorFilter, _VSR, MemorySearchResult = (
        _load_mem0_modules(monkeypatch)
    )

    class FakeBackend:
        def __init__(self, config=None):
            self.config = config
            self.client = SimpleNamespace(
                add=lambda content, **kwargs: {"results": [{"id": "mem0-id", "memory": "stored"}]},
                get=lambda memory_id: {"id": memory_id, "user_id": "user", "memory": "existing"},
                update=lambda **kwargs: None,
                delete=lambda **kwargs: None,
            )
            self.vector_results = []

        async def _ensure_client(self):
            return self.client

        async def search_memories(self, **kwargs):
            return self.vector_results

        async def close(self):
            self.closed = True

    monkeypatch.setattr(adapter, "Mem0Backend", FakeBackend)
    monkeypatch.setattr(adapter, "_utcnow", lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    system = adapter.Mem0SystemAdapter(adapter.Mem0Config(enable_graph=False))

    saved = await system.save_memory("fact", "user", 0.8, entities=["python"], metadata={"x": 1})
    assert saved.id == "mem0-id"
    assert saved.content == "stored"

    system._backend.client.add = lambda content, **kwargs: {}
    fallback = await system.save_memory("fact", "user", 0.8)
    assert fallback.metadata["_mem0_status"] == "not_extracted"

    system._backend.vector_results = [
        SimpleNamespace(
            memory=Memory(
                id="1",
                content="python fact",
                user_id="user",
                entity_refs=["python"],
                session_id="s1",
            ),
            similarity=0.9,
            rank=1,
        ),
        SimpleNamespace(
            memory=Memory(
                id="2", content="java fact", user_id="user", entity_refs=["java"], session_id="s2"
            ),
            similarity=0.5,
            rank=2,
        ),
    ]
    results = await system.search_memories(
        "python", "user", entities=["python"], include_related=True, session_id="s1"
    )
    assert len(results) == 1
    assert isinstance(results[0], MemorySearchResult)

    updated = await system.update_memory("id", "new", reason="fix", user_id="user")
    assert updated.content == "existing"
    system._backend.client.get = lambda memory_id: {
        "id": memory_id,
        "user_id": "other",
        "memory": "existing",
    }
    with pytest.raises(ValueError):
        await system.update_memory("id", "new", user_id="user")
    system._backend.client.get = lambda memory_id: None
    with pytest.raises(ValueError):
        await system.update_memory("id", "new")
    system._backend.client.get = lambda memory_id: {
        "id": memory_id,
        "user_id": "user",
        "memory": "existing",
    }
    system._backend.client.update = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad"))
    with pytest.raises(ValueError):
        await system.update_memory("id", "new")

    system._backend.client.get = lambda memory_id: (
        {"id": memory_id, "user_id": "other"} if memory_id == "id" else None
    )
    assert await system.delete_memory("id", user_id="user") is False
    system._backend.client.get = lambda memory_id: {"id": memory_id, "user_id": "user"}
    assert await system.delete_memory("id", user_id="user") is True
    system._backend.client.delete = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad"))
    assert await system.delete_memory("id") is False

    system._backend.client.get = lambda memory_id: {
        "id": memory_id,
        "memory": "stored",
        "user_id": "user",
        "metadata": {"a": 1},
    }
    got = await system.get_memory("id")
    assert got.metadata == {"a": 1}
    system._backend.client.get = lambda memory_id: None
    assert await system.get_memory("id") is None

    assert system.supports_graph is False
    assert system.supports_vector_search is True
    await system.close()
    assert system._backend.closed is True
