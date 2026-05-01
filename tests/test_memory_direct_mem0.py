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


def _load_direct_mem0(monkeypatch):
    @dataclass
    class Memory:
        id: str
        content: str
        user_id: str
        session_id: str | None = None
        importance: float = 0.5
        entity_refs: list[str] = field(default_factory=list)
        metadata: dict = field(default_factory=dict)
        created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
        valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @dataclass
    class MemorySearchResult:
        memory: Memory
        score: float
        related_entities: list[str] = field(default_factory=list)
        related_memories: list[str] = field(default_factory=list)

    models_module = types.ModuleType("headroom.memory.models")
    models_module.Memory = Memory
    ports_module = types.ModuleType("headroom.memory.ports")
    ports_module.MemorySearchResult = MemorySearchResult
    monkeypatch.setitem(sys.modules, "headroom.memory.models", models_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.ports", ports_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "backends" / "direct_mem0.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.backends.direct_mem0", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.backends.direct_mem0", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.backends.direct_mem0"] = module
    spec.loader.exec_module(module)
    return module, Memory, MemorySearchResult


@pytest.mark.asyncio
async def test_direct_mem0_initialization_and_direct_paths(monkeypatch) -> None:
    direct_mem0, Memory, _MemorySearchResult = _load_direct_mem0(monkeypatch)

    class FakeOpenAI:
        def __init__(self):
            self.embeddings = SimpleNamespace(
                create=lambda input, model: SimpleNamespace(
                    data=[SimpleNamespace(embedding=[float(len(input))])]
                )
            )

    class FakeQdrantClient:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.upserts = []

        def upsert(self, collection_name, points):
            self.upserts.append((collection_name, points))

    class FakeSession:
        def __init__(self, log):
            self.log = log

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, cypher, params):
            self.log.append((cypher, params))

    class FakeNeo4jDriver:
        def __init__(self):
            self.log = []
            self.closed = False

        def session(self):
            return FakeSession(self.log)

        def close(self):
            self.closed = True

    fake_neo4j_driver = FakeNeo4jDriver()

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri, auth):
            return fake_neo4j_driver

    class FakeMem0Client:
        def __init__(self):
            self.add_result = {"results": [{"id": "fallback-id", "memory": "stored"}]}
            self.search_result = []

        def add(self, content, **kwargs):
            return self.add_result

        def search(self, **kwargs):
            return self.search_result

        def update(self, **kwargs):
            self.updated = kwargs

        def delete(self, **kwargs):
            self.deleted = kwargs

        def get(self, **kwargs):
            return {
                "id": kwargs["memory_id"],
                "memory": "got",
                "user_id": "u",
                "metadata": {"a": 1},
            }

    fake_mem0_client = FakeMem0Client()

    class FakeMem0Memory:
        @classmethod
        def from_config(cls, config):
            FakeMem0Memory.config = config
            return fake_mem0_client

    @dataclass
    class PointStruct:
        id: str
        vector: list[float]
        payload: dict

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(
        sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=FakeQdrantClient)
    )
    monkeypatch.setitem(
        sys.modules, "qdrant_client.models", types.SimpleNamespace(PointStruct=PointStruct)
    )
    monkeypatch.setitem(
        sys.modules, "neo4j", types.SimpleNamespace(GraphDatabase=FakeGraphDatabase)
    )
    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(Memory=FakeMem0Memory))
    monkeypatch.setattr(
        direct_mem0,
        "_utcnow",
        lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    adapter = direct_mem0.DirectMem0Adapter(direct_mem0.Mem0Config(enable_graph=True))
    await adapter._ensure_initialized()
    assert adapter._initialized is True
    assert adapter._qdrant_client.host == "localhost"
    assert "graph_store" in FakeMem0Memory.config
    assert adapter._embed("abc") == [3.0]
    assert len(adapter._generate_id("fact", "u1")) == 32

    ids = await adapter._write_facts_to_qdrant(["alpha", "beta"], "u1", 0.9, {"x": 1})
    assert len(ids) == 2
    assert adapter._qdrant_client.upserts[0][0] == adapter._config.collection_name

    await adapter._write_graph_to_neo4j(
        [{"entity": "Alice", "entity_type": "Person"}],
        [{"source": "Alice", "relationship": "WORKS AT", "destination": "Acme Inc"}],
        "u1",
    )
    assert "works_at" in fake_neo4j_driver.log[0][0]
    adapter._neo4j_driver = None
    await adapter._write_graph_to_neo4j([], [], "u1")

    adapter._neo4j_driver = fake_neo4j_driver
    direct = await adapter._save_memory_internal(
        content="raw",
        user_id="u1",
        importance=0.8,
        entities=["Alice"],
        session_id="s1",
        metadata={"m": 1},
        facts=["fact1", "fact2"],
        extracted_entities=[{"entity": "Alice", "entity_type": "person"}],
        extracted_relationships=[
            {"source": "Alice", "relationship": "knows", "destination": "Bob"}
        ],
    )
    assert direct.metadata["_direct_write"] is True
    assert direct.metadata["_fact_count"] == 2

    direct_no_facts = await adapter._save_memory_internal(
        content="content-only",
        user_id="u1",
        importance=0.8,
        metadata={"z": 1},
        extracted_entities=[{"entity": "Alice", "entity_type": "person"}],
    )
    assert direct_no_facts.content == "content-only"

    fake_mem0_client.add_result = {"results": [{"id": "fallback-id", "memory": "stored"}]}
    fallback = await adapter._save_memory_internal(
        "raw", "u1", 0.4, entities=["Alice"], metadata={"x": 1}
    )
    assert fallback.id == "fallback-id"
    fake_mem0_client.add_result = {}
    not_extracted = await adapter._save_memory_internal("raw", "u1", 0.4)
    assert not_extracted.metadata["_mem0_status"] == "not_extracted"

    await adapter.close()
    assert fake_neo4j_driver.closed is True


@pytest.mark.asyncio
async def test_direct_mem0_tasks_search_and_crud(monkeypatch) -> None:
    direct_mem0, Memory, MemorySearchResult = _load_direct_mem0(monkeypatch)
    monkeypatch.setattr(
        direct_mem0,
        "_utcnow",
        lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    adapter = direct_mem0.DirectMem0Adapter(
        direct_mem0.Mem0Config(async_writes=True, enable_graph=False)
    )
    adapter._initialized = True
    adapter._mem0_client = SimpleNamespace()

    async def fake_internal(**kwargs):
        await asyncio.sleep(0)
        return Memory(id="saved", content=kwargs["content"], user_id=kwargs["user_id"])

    monkeypatch.setattr(adapter, "_save_memory_internal", fake_internal)
    bg = await adapter.save_memory("content", "u1", 0.7, facts=["fact1"], background=True)
    task_id = bg.metadata["_task_id"]
    assert bg.metadata["_status"] == "processing"
    assert adapter.get_task_status(task_id)["status"] in {"processing", "completed"}
    waited = await adapter.wait_for_task(task_id, timeout=1)
    assert waited["status"] == "completed"
    assert adapter.get_pending_tasks() == []
    assert adapter.get_task_status("missing")["status"] == "not_found"

    adapter._background_tasks["pending"] = asyncio.create_task(asyncio.sleep(0.2, result="ok"))
    assert (await adapter.wait_for_task("pending", timeout=0.01))["status"] == "timeout"

    async def _explode():
        raise RuntimeError("boom")

    adapter._background_tasks = {
        "ok": asyncio.create_task(asyncio.sleep(0, result="x")),
        "bad": asyncio.create_task(_explode()),
    }
    summary = await adapter.flush_pending(timeout=1)
    assert summary["completed"] >= 1
    assert summary["failed"] >= 1
    empty_summary = await direct_mem0.DirectMem0Adapter().flush_pending()
    assert empty_summary["completed"] == 0

    adapter._background_tasks = {"slow": asyncio.create_task(asyncio.sleep(0.2))}
    flush_timeout = await adapter.flush_pending(timeout=0.01)
    assert flush_timeout["pending"] >= 1

    search_client = SimpleNamespace(
        search=lambda **kwargs: {
            "results": [
                {
                    "id": "1",
                    "metadata": {"user_id": "u1", "entity_refs": ["Python"], "session_id": "s1"},
                    "memory": "Python facts",
                    "score": 0.8,
                },
                {
                    "id": "2",
                    "metadata": {"user_id": "u1", "entity_refs": ["Rust"], "session_id": "s2"},
                    "content": "Rust notes",
                    "similarity": 0.5,
                },
            ]
        },
        update=lambda **kwargs: None,
        delete=lambda **kwargs: None,
        get=lambda **kwargs: {
            "id": kwargs["memory_id"],
            "memory": "stored",
            "user_id": "u1",
            "metadata": {"a": 1},
        },
    )
    adapter2 = direct_mem0.DirectMem0Adapter()
    adapter2._initialized = True
    adapter2._mem0_client = search_client
    results = await adapter2.search_memories(
        "python", "u1", entities=["Python"], include_related=True, top_k=5, session_id="s1"
    )
    assert isinstance(results[0], MemorySearchResult)
    assert results[0].memory.content == "Python facts"
    assert await adapter2.search_memories("python", "u1", entities=["Go"], session_id="s1") == []

    updated = await adapter2.update_memory("m1", "new", reason="why", user_id="u1")
    assert updated.metadata["update_reason"] == "why"
    failing_update_client = SimpleNamespace(
        update=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    adapter2._mem0_client = failing_update_client
    with pytest.raises(ValueError):
        await adapter2.update_memory("m1", "new")

    adapter2._mem0_client = search_client
    assert await adapter2.delete_memory("m1") is True
    adapter2._mem0_client = SimpleNamespace(
        delete=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    assert await adapter2.delete_memory("m1") is False

    adapter2._mem0_client = search_client
    got = await adapter2.get_memory("m1")
    assert got.metadata == {"a": 1}
    adapter2._mem0_client = SimpleNamespace(get=lambda **kwargs: None)
    assert await adapter2.get_memory("m1") is None
    adapter2._mem0_client = SimpleNamespace(
        get=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    assert await adapter2.get_memory("m1") is None

    assert adapter2.supports_graph is True
    assert adapter2.supports_vector_search is True
