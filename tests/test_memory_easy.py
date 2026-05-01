from __future__ import annotations

import builtins
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_easy_module(monkeypatch):
    paths_module = types.ModuleType("headroom.paths")
    paths_module.memory_db_path = lambda: Path("workspace") / "memory.db"
    monkeypatch.setitem(sys.modules, "headroom.paths", paths_module)
    monkeypatch.setitem(sys.modules, "headroom", types.SimpleNamespace(paths=paths_module))

    class FakeLocalBackendConfig:
        def __init__(self, db_path):
            self.db_path = db_path

    class FakeLocalBackend:
        def __init__(self, config):
            self.config = config
            self.closed = False
            self.saved = []

        async def save_memory(self, **kwargs):
            self.saved.append(kwargs)
            return SimpleNamespace(id="saved-id")

        async def search_memories(self, **kwargs):
            return [
                SimpleNamespace(
                    memory=SimpleNamespace(content="python", id="m1", metadata={"a": 1}),
                    score=0.9,
                )
            ]

        async def delete_memory(self, memory_id):
            return memory_id == "ok"

        async def clear_user(self, user_id):
            return 3

        async def close(self):
            self.closed = True

    local_module = types.ModuleType("headroom.memory.backends.local")
    local_module.LocalBackend = FakeLocalBackend
    local_module.LocalBackendConfig = FakeLocalBackendConfig
    monkeypatch.setitem(sys.modules, "headroom.memory.backends.local", local_module)

    class FakeMem0Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeDirectMem0Adapter(FakeLocalBackend):
        pass

    direct_mem0_module = types.ModuleType("headroom.memory.backends.direct_mem0")
    direct_mem0_module.DirectMem0Adapter = FakeDirectMem0Adapter
    direct_mem0_module.Mem0Config = FakeMem0Config
    monkeypatch.setitem(sys.modules, "headroom.memory.backends.direct_mem0", direct_mem0_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "easy.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.easy", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.easy", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.easy"] = module
    spec.loader.exec_module(module)
    return module, FakeLocalBackend, FakeDirectMem0Adapter, FakeMem0Config


@pytest.mark.asyncio
async def test_easy_memory_local_backend_paths(monkeypatch) -> None:
    easy, FakeLocalBackend, _FakeDirectMem0Adapter, _FakeMem0Config = _load_easy_module(monkeypatch)
    mem = easy.Memory()
    assert mem.backend_type == "local"
    assert "local" in repr(mem)

    assert mem._db_path == Path("workspace") / "memory.db"
    saved_id = await mem.save(
        "Alice likes Python",
        user_id="alice",
        facts=["Alice likes Python"],
        entities=[{"entity": "Python", "entity_type": "language"}],
        relationships=[{"source": "Alice", "relationship": "likes", "destination": "Python"}],
        metadata={"x": 1},
    )
    assert saved_id == "saved-id"
    assert isinstance(mem._backend, FakeLocalBackend)
    assert mem._backend.saved[0]["extracted_entities"][0]["entity"] == "Python"

    results = await mem.search("language", user_id="alice", top_k=5, include_graph=False)
    assert results[0].content == "python"
    assert results[0].metadata == {"a": 1}
    assert await mem.delete("ok") is True
    assert await mem.delete("missing") is False
    assert await mem.clear("alice") == 3
    await mem.close()
    assert mem._initialized is False


@pytest.mark.asyncio
async def test_easy_memory_direct_mem0_and_errors(monkeypatch) -> None:
    easy, _FakeLocalBackend, FakeDirectMem0Adapter, _FakeMem0Config = _load_easy_module(monkeypatch)
    mem = easy.Memory(backend="qdrant-neo4j", db_path="custom.db", qdrant_host="q", neo4j_uri="neo")
    await mem._ensure_initialized()
    assert isinstance(mem._backend, FakeDirectMem0Adapter)

    async def _close():
        return None

    mem._backend = SimpleNamespace(close=_close)
    assert await mem.clear("alice") == 0

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "headroom.memory.backends.direct_mem0":
            raise ImportError("missing direct mem0")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    broken = easy.Memory(backend="qdrant-neo4j")
    with pytest.raises(ImportError):
        await broken._ensure_initialized()

    invalid = easy.Memory(backend="bad")
    with pytest.raises(ValueError):
        await invalid._ensure_initialized()
