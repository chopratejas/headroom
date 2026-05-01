from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_local_backend(monkeypatch):
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
        entity_refs: list[str] = field(default_factory=list)
        importance: float = 0.5
        metadata: dict = field(default_factory=dict)

    @dataclass
    class MemorySearchResult:
        memory: Memory
        score: float
        related_entities: list[str]
        related_memories: list[str]

    @dataclass
    class TextFilter:
        query: str
        user_id: str
        limit: int

    @dataclass
    class Entity:
        id: str
        user_id: str
        name: str
        entity_type: str
        metadata: dict = field(default_factory=dict)

    @dataclass
    class Relationship:
        id: str
        user_id: str
        source_id: str
        target_id: str
        relation_type: str
        metadata: dict = field(default_factory=dict)

    @dataclass
    class Subgraph:
        entities: list
        relationships: list
        root_entity_ids: list[str]

    graph_models_module = types.ModuleType("headroom.memory.adapters.graph_models")
    graph_models_module.Entity = Entity
    graph_models_module.Relationship = Relationship
    graph_models_module.Subgraph = Subgraph

    models_module = types.ModuleType("headroom.memory.models")
    models_module.Memory = Memory
    ports_module = types.ModuleType("headroom.memory.ports")
    ports_module.MemorySearchResult = MemorySearchResult
    ports_module.TextFilter = TextFilter

    defaults = SimpleNamespace(sentence_transformer="mini", sentence_transformer_dim=384)
    config_module = types.ModuleType("headroom.models.config")
    config_module.ML_MODEL_DEFAULTS = defaults

    class EmbedderBackend:
        LOCAL = "LOCAL"
        ONNX = "ONNX"
        OPENAI = "OPENAI"
        OLLAMA = "OLLAMA"

    memory_config_module = types.ModuleType("headroom.memory.config")
    memory_config_module.EmbedderBackend = EmbedderBackend

    class FakeHierarchy:
        created_configs = []

        def __init__(self):
            self.add_calls = []
            self.search_results = []
            self.memories = {}
            self.text_results = []
            self.closed = False
            self.text_index = SimpleNamespace(search_memories=self._text_search)

        async def _text_search(self, text_filter):
            self.last_text_filter = text_filter
            return self.text_results

        @classmethod
        async def create(cls, config):
            cls.created_configs.append(config)
            inst = cls()
            cls.last_instance = inst
            return inst

        async def add(self, **kwargs):
            self.add_calls.append(kwargs)
            memory = Memory(
                id=f"mem-{len(self.add_calls)}",
                content=kwargs["content"],
                user_id=kwargs["user_id"],
                session_id=kwargs.get("session_id"),
                agent_id=kwargs.get("agent_id"),
                turn_id=kwargs.get("turn_id"),
                entity_refs=list(kwargs.get("entity_refs") or []),
                importance=kwargs.get("importance", 0.5),
                metadata=kwargs.get("metadata") or {},
            )
            self.memories[memory.id] = memory
            return memory

        async def search(self, **kwargs):
            self.last_search_kwargs = kwargs
            return self.search_results

        async def get(self, memory_id):
            return self.memories.get(memory_id)

        async def supersede(self, old_memory_id, new_content):
            return Memory(id=f"{old_memory_id}-new", content=new_content, user_id="user")

        async def delete(self, memory_id):
            return self.memories.pop(memory_id, None) is not None

        async def get_user_memories(self, user_id, limit):
            return [m for m in self.memories.values() if m.user_id == user_id][:limit]

        async def clear_scope(self, user_id):
            removed = [mid for mid, mem in self.memories.items() if mem.user_id == user_id]
            for mid in removed:
                self.memories.pop(mid, None)
            return len(removed)

        async def close(self):
            self.closed = True

    class MemoryConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    memory_module = types.ModuleType("headroom.memory")
    memory_module.HierarchicalMemory = FakeHierarchy
    memory_module.MemoryConfig = MemoryConfig

    class FakeGraphStore:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.entities = {}
            self.relationships = []
            self.deleted_entities = []
            self.cleared_users = []
            self.subgraph = Subgraph(entities=[], relationships=[], root_entity_ids=[])

        async def get_entity_by_name(self, user_id, name):
            return self.entities.get((user_id, name.lower()))

        async def add_entity(self, entity):
            self.entities[(entity.user_id, entity.name.lower())] = entity

        async def add_relationship(self, relationship):
            self.relationships.append(relationship)

        async def query_subgraph(self, entity_ids, max_hops):
            self.last_query = (entity_ids, max_hops)
            return self.subgraph

        async def get_entities_for_user(self, user_id):
            return [entity for (uid, _), entity in self.entities.items() if uid == user_id]

        async def delete_entity(self, entity_id):
            self.deleted_entities.append(entity_id)

        async def clear_user(self, user_id):
            self.cleared_users.append(user_id)

    sqlite_graph_module = types.ModuleType("headroom.memory.adapters.sqlite_graph")
    sqlite_graph_module.SQLiteGraphStore = FakeGraphStore
    graph_module = types.ModuleType("headroom.memory.adapters.graph")
    graph_module.InMemoryGraphStore = FakeGraphStore

    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.graph_models", graph_models_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.models", models_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.ports", ports_module)
    monkeypatch.setitem(sys.modules, "headroom.models.config", config_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.config", memory_config_module)
    monkeypatch.setitem(sys.modules, "headroom.memory", memory_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.sqlite_graph", sqlite_graph_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.graph", graph_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "backends" / "local.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.backends.local", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.backends.local", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.backends.local"] = module
    spec.loader.exec_module(module)
    return (
        module,
        FakeHierarchy,
        FakeGraphStore,
        Memory,
        MemorySearchResult,
        Entity,
        Relationship,
        Subgraph,
    )


@pytest.mark.asyncio
async def test_local_backend_initialization_save_and_search(monkeypatch) -> None:
    (
        local,
        FakeHierarchy,
        FakeGraphStore,
        Memory,
        MemorySearchResult,
        Entity,
        Relationship,
        Subgraph,
    ) = _load_local_backend(monkeypatch)

    backend = local.LocalBackend(local.LocalBackendConfig(db_path="mem.db", graph_persist=True))
    await backend._ensure_initialized()
    assert isinstance(backend._graph, FakeGraphStore)
    assert backend._graph.kwargs["db_path"].endswith("mem_graph.db")
    assert FakeHierarchy.created_configs[-1].embedder_backend == "LOCAL"

    existing = Entity(id="e-existing", user_id="user", name="Alice", entity_type="person")
    backend._graph.entities[("user", "alice")] = existing
    saved = await backend.save_memory(
        content="Alice works at Acme",
        user_id="user",
        importance=0.9,
        entities=["Alice"],
        relationships=[{"source": "Alice", "target": "Acme", "type": "works_at"}],
        metadata={"source": "manual"},
        session_id="sess",
        agent_id="agent",
        turn_id="turn",
        facts=["Alice works at Acme", "Alice likes Python"],
        extracted_entities=[{"entity": "Acme", "entity_type": "org"}],
        extracted_relationships=[
            {"source": "Acme", "relationship": "employs", "destination": "Alice"}
        ],
    )
    assert saved.content == "Alice works at Acme"
    assert len(backend._hierarchical_memory.add_calls) == 2
    assert len(backend._graph.relationships) == 2

    vector_primary = SimpleNamespace(memory=saved, similarity=0.9)
    duplicate = SimpleNamespace(memory=saved, similarity=0.7)
    other = Memory(
        id="mem-related",
        content="Acme hires Bob",
        user_id="user",
        entity_refs=["Acme"],
        session_id="sess",
    )
    backend._hierarchical_memory.memories[other.id] = other
    backend._hierarchical_memory.search_results = [vector_primary, duplicate]
    backend._graph.subgraph = Subgraph(
        entities=[
            Entity(
                id="e2",
                user_id="user",
                name="Acme",
                entity_type="org",
                metadata={"source_memory_id": "mem-related"},
            )
        ],
        relationships=[
            Relationship(
                id="r1",
                user_id="user",
                source_id="e-existing",
                target_id="e2",
                relation_type="works_at",
                metadata={"source_memory_id": "mem-other-session"},
            )
        ],
        root_entity_ids=["e-existing"],
    )
    backend._hierarchical_memory.memories["mem-other-session"] = Memory(
        id="mem-other-session",
        content="hidden",
        user_id="user",
        entity_refs=["Acme"],
        session_id="other",
    )
    results = await backend.search_memories(
        query="Alice",
        user_id="user",
        top_k=5,
        entities=["Acme"],
        include_related=True,
        min_similarity=0.2,
        session_id="sess",
    )
    assert [r.memory.id for r in results] == ["mem-1", "mem-related"]

    no_related = await backend.search_memories("Alice", "user", include_related=False)
    assert len(no_related) == 1

    memless = local.LocalBackend(local.LocalBackendConfig(graph_persist=False))
    await memless._ensure_initialized()
    assert isinstance(memless._graph, FakeGraphStore)


@pytest.mark.asyncio
async def test_local_backend_update_delete_helpers_and_hybrid(monkeypatch) -> None:
    (
        local,
        FakeHierarchy,
        FakeGraphStore,
        Memory,
        MemorySearchResult,
        Entity,
        _Relationship,
        Subgraph,
    ) = _load_local_backend(monkeypatch)

    backend = local.LocalBackend(local.LocalBackendConfig(graph_persist=False))
    await backend._ensure_initialized()
    hm = backend._hierarchical_memory
    graph = backend._graph
    assert hm is not None and graph is not None

    mem = Memory(id="mem-1", content="alpha", user_id="user", entity_refs=["Alice"])
    hm.memories[mem.id] = mem
    graph.entities[("user", "alice")] = Entity(
        id="ent-1",
        user_id="user",
        name="Alice",
        entity_type="person",
        metadata={"source_memory_id": "mem-1"},
    )

    updated = await backend.update_memory("mem-1", "beta", reason="fix", user_id="user")
    assert updated.id == "mem-1-new"

    assert await backend.delete_memory("missing") is False
    assert await backend.delete_memory("mem-1") is True
    assert graph.deleted_entities == ["ent-1"]

    hm.memories["mem-2"] = Memory(id="mem-2", content="lookup", user_id="user")
    assert (await backend.get_memory("mem-2")).content == "lookup"
    assert backend.supports_graph is True
    assert backend.supports_vector_search is True
    assert backend.supports_text_search is True
    assert await backend.get_graph() is graph

    graph.entities[("user", "alice")] = Entity(
        id="ent-a", user_id="user", name="Alice", entity_type="person"
    )
    graph.subgraph = Subgraph(
        entities=[graph.entities[("user", "alice")]], relationships=[], root_entity_ids=["ent-a"]
    )
    assert (await backend.query_subgraph(["Alice"], "user")).root_entity_ids == ["ent-a"]
    empty_subgraph = await backend.query_subgraph(["Unknown"], "user")
    assert empty_subgraph.root_entity_ids == []

    hm.memories["mem-3"] = Memory(id="mem-3", content="u1", user_id="user")
    hm.memories["mem-4"] = Memory(id="mem-4", content="u2", user_id="other")
    assert [m.id for m in await backend.get_user_memories("user", limit=10)] == ["mem-2", "mem-3"]
    assert await backend.clear_user("user") == 2
    assert graph.cleared_users == ["user"]

    text_memory = Memory(id="text-1", content="text", user_id="user", entity_refs=["A"])
    hm.text_results = [SimpleNamespace(memory=text_memory, score=6.0)]
    text_results = await backend.text_search("needle", "user", limit=3)
    assert isinstance(text_results[0], MemorySearchResult)
    assert hm.last_text_filter.query == "needle"

    vector_memory = Memory(id="vec-1", content="vec", user_id="user")
    shared_memory = Memory(id="shared", content="both", user_id="user")
    monkeypatch.setattr(
        backend,
        "search_memories",
        lambda **kwargs: pytest.helpers.awaitable(
            [
                MemorySearchResult(
                    memory=vector_memory, score=0.8, related_entities=[], related_memories=[]
                ),
                MemorySearchResult(
                    memory=shared_memory, score=0.2, related_entities=["x"], related_memories=[]
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        backend,
        "text_search",
        lambda **kwargs: pytest.helpers.awaitable(
            [
                MemorySearchResult(
                    memory=shared_memory, score=4.0, related_entities=["x"], related_memories=[]
                ),
                MemorySearchResult(
                    memory=text_memory, score=2.0, related_entities=["A"], related_memories=[]
                ),
            ]
        ),
    )
    hybrid = await backend.hybrid_search("q", "user", top_k=3, vector_weight=0.25, text_weight=0.75)
    assert [r.memory.id for r in hybrid] == ["shared", "text-1", "vec-1"]

    await backend.close()
    assert hm.closed is True
    assert backend._initialized is False


@pytest.fixture(autouse=True)
def _awaitable_helper():
    class Helpers:
        @staticmethod
        async def awaitable(value):
            return value

    pytest.helpers = Helpers()
    yield
