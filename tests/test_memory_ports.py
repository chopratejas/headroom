from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest


def _load_ports(monkeypatch):
    class ScopeLevel(Enum):
        USER = "user"
        SESSION = "session"
        AGENT = "agent"

    @dataclass
    class Memory:
        id: str
        content: str
        importance: float = 0.5
        entity_refs: list[str] = field(default_factory=list)
        created_at: datetime = field(default_factory=lambda: datetime(2024, 1, 2, 3, 4, 5))

    models_module = types.ModuleType("headroom.memory.models")
    models_module.Memory = Memory
    models_module.ScopeLevel = ScopeLevel
    monkeypatch.setitem(sys.modules, "headroom.memory.models", models_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "ports.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.ports", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.ports", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.ports"] = module
    spec.loader.exec_module(module)
    return module, Memory, ScopeLevel


def test_memory_port_dataclasses_and_results(monkeypatch) -> None:
    ports, Memory, ScopeLevel = _load_ports(monkeypatch)
    memory = Memory(id="mem-1", content="Hello", importance=0.8, entity_refs=["alice", "proj"])

    memory_filter = ports.MemoryFilter(user_id="u1", scope_levels=[ScopeLevel.USER], limit=5)
    vector_filter = ports.VectorFilter(query_text="hello", user_id="u1")
    text_filter = ports.TextFilter(query="hello", session_id="s1")
    assert memory_filter.offset == 0
    assert memory_filter.order_by == "created_at"
    assert memory_filter.metadata_filters == {}
    assert vector_filter.top_k == 10
    assert vector_filter.min_similarity == 0.0
    assert vector_filter.metadata_filters == {}
    assert text_filter.match_mode == "contains"
    assert text_filter.case_sensitive is False
    assert text_filter.metadata_filters == {}

    higher_similarity = ports.VectorSearchResult(memory=memory, similarity=0.9, rank=1)
    lower_similarity = ports.VectorSearchResult(memory=memory, similarity=0.4, rank=2)
    assert sorted([lower_similarity, higher_similarity]) == [higher_similarity, lower_similarity]

    higher_score = ports.TextSearchResult(memory=memory, score=0.9, rank=1)
    lower_score = ports.TextSearchResult(memory=memory, score=0.4, rank=2)
    assert sorted([lower_score, higher_score]) == [higher_score, lower_score]

    search_result = ports.MemorySearchResult(
        memory=memory,
        score=0.7,
        related_entities=["alice"],
        related_memories=["mem-2"],
    )
    assert search_result.to_dict() == {
        "memory_id": "mem-1",
        "content": "Hello",
        "importance": 0.8,
        "entities": ["alice", "proj"],
        "created_at": "2024-01-02T03:04:05",
        "score": 0.7,
        "related_entities": ["alice"],
        "related_memories": ["mem-2"],
    }


def test_subgraph_to_context_formats_entities_relationships_and_fallbacks(monkeypatch) -> None:
    ports, _, _ = _load_ports(monkeypatch)

    empty = ports.Subgraph()
    assert empty.to_context() == ""

    source = ports.Entity(id="e1", name="Alice", entity_type="person", metadata={"role": "lead"})
    target = ports.Entity(id="e2", name="Project X", entity_type="project")
    rel1 = ports.Relationship(
        source_entity_id="e1",
        target_entity_id="e2",
        relation_type="owns",
        weight=0.5,
    )
    rel2 = ports.Relationship(
        source_entity_id="missing-source",
        target_entity_id="missing-target",
        relation_type="references",
    )

    context = ports.Subgraph(entities=[source, target], relationships=[rel1, rel2]).to_context()
    assert "Entities:" in context
    assert "  - Alice (person) [role=lead]" in context
    assert "  - Project X (project)" in context
    assert "Relationships:" in context
    assert "  - Alice --[owns]--> Project X (weight=0.5)" in context
    assert "  - missing-source --[references]--> missing-target" in context
    assert context.count("Relationships:") == 1


@pytest.mark.asyncio
async def test_runtime_checkable_memory_protocols(monkeypatch) -> None:
    ports, Memory, _ = _load_ports(monkeypatch)
    memory = Memory(id="mem-1", content="Hello")

    class FakeMemoryStore:
        async def save(self, memory): ...
        async def save_batch(self, memories): ...
        async def get(self, memory_id):
            return memory

        async def get_batch(self, memory_ids):
            return [memory]

        async def delete(self, memory_id):
            return True

        async def delete_batch(self, memory_ids):
            return len(memory_ids)

        async def query(self, filter):
            return [memory]

        async def count(self, filter):
            return 1

        async def supersede(self, old_memory_id, new_memory, supersede_time=None):
            return new_memory

        async def get_history(self, memory_id, include_future=False):
            return [memory]

        async def clear_scope(self, user_id, session_id=None, agent_id=None, turn_id=None):
            return 0

    class FakeVectorIndex:
        async def index(self, memory): ...
        async def index_batch(self, memories):
            return len(memories)

        async def remove(self, memory_id):
            return True

        async def remove_batch(self, memory_ids):
            return len(memory_ids)

        async def search(self, filter):
            return []

        async def update_embedding(self, memory_id, embedding):
            return True

        @property
        def dimension(self):
            return 3

        @property
        def size(self):
            return 1

    class FakeTextIndex:
        async def index(self, memory): ...
        async def index_batch(self, memories):
            return len(memories)

        async def remove(self, memory_id):
            return True

        async def remove_batch(self, memory_ids):
            return len(memory_ids)

        async def search(self, filter):
            return []

        async def update_content(self, memory_id, content):
            return True

    class FakeEmbedder:
        async def embed(self, text):
            return [1.0, 2.0]

        async def embed_batch(self, texts):
            return [[1.0, 2.0] for _ in texts]

        @property
        def dimension(self):
            return 2

        @property
        def model_name(self):
            return "fake"

        @property
        def max_tokens(self):
            return 1000

    class FakeMemoryCache:
        async def get(self, memory_id):
            return memory

        async def get_batch(self, memory_ids):
            return {memory.id: memory}

        async def put(self, memory, ttl_seconds=None): ...
        async def put_batch(self, memories, ttl_seconds=None): ...
        async def invalidate(self, memory_id):
            return True

        async def invalidate_batch(self, memory_ids):
            return len(memory_ids)

        async def invalidate_scope(self, user_id, session_id=None, agent_id=None):
            return 0

        async def clear(self): ...

        @property
        def size(self):
            return 1

        @property
        def max_size(self):
            return 10

    class FakeGraphStore:
        async def add_entity(self, entity): ...
        async def add_relationship(self, relationship): ...
        async def get_entity(self, entity_id):
            return None

        async def get_entity_by_name(self, name, user_id, entity_type=None):
            return None

        async def get_relationships(self, entity_id, relation_types=None, direction="both"):
            return []

        async def query_subgraph(self, entity_ids, hops=1, relation_types=None):
            return ports.Subgraph()

        async def find_path(self, source_entity_id, target_entity_id, max_hops=3):
            return None

        async def delete_entity(self, entity_id):
            return True

        async def delete_relationship(self, relationship_id):
            return True

        async def clear_user(self, user_id):
            return 0

    assert isinstance(FakeMemoryStore(), ports.MemoryStore)
    assert isinstance(FakeVectorIndex(), ports.VectorIndex)
    assert isinstance(FakeTextIndex(), ports.TextIndex)
    assert isinstance(FakeEmbedder(), ports.Embedder)
    assert isinstance(FakeMemoryCache(), ports.MemoryCache)
    assert isinstance(FakeGraphStore(), ports.GraphStore)


@pytest.mark.asyncio
async def test_protocol_stub_methods_are_executable(monkeypatch) -> None:
    ports, Memory, ScopeLevel = _load_ports(monkeypatch)
    memory = Memory(id="mem-1", content="Hello")
    memory_filter = ports.MemoryFilter(user_id="u1", scope_levels=[ScopeLevel.USER])
    vector_filter = ports.VectorFilter(query_text="hello")
    text_filter = ports.TextFilter(query="hello")
    entity = ports.Entity(id="e1", name="Alice")
    relationship = ports.Relationship(source_entity_id="e1", target_entity_id="e2")

    assert await ports.MemoryStore.save(object(), memory) is None
    assert await ports.MemoryStore.save_batch(object(), [memory]) is None
    assert await ports.MemoryStore.get(object(), "mem-1") is None
    assert await ports.MemoryStore.get_batch(object(), ["mem-1"]) is None
    assert await ports.MemoryStore.delete(object(), "mem-1") is None
    assert await ports.MemoryStore.delete_batch(object(), ["mem-1"]) is None
    assert await ports.MemoryStore.query(object(), memory_filter) is None
    assert await ports.MemoryStore.count(object(), memory_filter) is None
    assert await ports.MemoryStore.supersede(object(), "mem-1", memory) is None
    assert await ports.MemoryStore.get_history(object(), "mem-1") is None
    assert await ports.MemoryStore.clear_scope(object(), "u1") is None

    assert await ports.VectorIndex.index(object(), memory) is None
    assert await ports.VectorIndex.index_batch(object(), [memory]) is None
    assert await ports.VectorIndex.remove(object(), "mem-1") is None
    assert await ports.VectorIndex.remove_batch(object(), ["mem-1"]) is None
    assert await ports.VectorIndex.search(object(), vector_filter) is None
    assert await ports.VectorIndex.update_embedding(object(), "mem-1", [1.0, 2.0]) is None
    assert ports.VectorIndex.dimension.fget(object()) is None
    assert ports.VectorIndex.size.fget(object()) is None

    assert await ports.TextIndex.index(object(), memory) is None
    assert await ports.TextIndex.index_batch(object(), [memory]) is None
    assert await ports.TextIndex.remove(object(), "mem-1") is None
    assert await ports.TextIndex.remove_batch(object(), ["mem-1"]) is None
    assert await ports.TextIndex.search(object(), text_filter) is None
    assert await ports.TextIndex.update_content(object(), "mem-1", "updated") is None

    assert await ports.Embedder.embed(object(), "hello") is None
    assert await ports.Embedder.embed_batch(object(), ["hello"]) is None
    assert ports.Embedder.dimension.fget(object()) is None
    assert ports.Embedder.model_name.fget(object()) is None
    assert ports.Embedder.max_tokens.fget(object()) is None

    assert await ports.MemoryCache.get(object(), "mem-1") is None
    assert await ports.MemoryCache.get_batch(object(), ["mem-1"]) is None
    assert await ports.MemoryCache.put(object(), memory) is None
    assert await ports.MemoryCache.put_batch(object(), [memory]) is None
    assert await ports.MemoryCache.invalidate(object(), "mem-1") is None
    assert await ports.MemoryCache.invalidate_batch(object(), ["mem-1"]) is None
    assert await ports.MemoryCache.invalidate_scope(object(), "u1") is None
    assert await ports.MemoryCache.clear(object()) is None
    assert ports.MemoryCache.size.fget(object()) is None
    assert ports.MemoryCache.max_size.fget(object()) is None

    assert await ports.GraphStore.add_entity(object(), entity) is None
    assert await ports.GraphStore.add_relationship(object(), relationship) is None
    assert await ports.GraphStore.get_entity(object(), "e1") is None
    assert await ports.GraphStore.get_entity_by_name(object(), "Alice", "u1") is None
    assert await ports.GraphStore.get_relationships(object(), "e1") is None
    assert await ports.GraphStore.query_subgraph(object(), ["e1"]) is None
    assert await ports.GraphStore.find_path(object(), "e1", "e2") is None
    assert await ports.GraphStore.delete_entity(object(), "e1") is None
    assert await ports.GraphStore.delete_relationship(object(), "r1") is None
    assert await ports.GraphStore.clear_user(object(), "u1") is None
