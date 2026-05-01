from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


def _load_cache_module(monkeypatch):
    @dataclass
    class Memory:
        id: str
        user_id: str
        session_id: str | None = None
        agent_id: str | None = None

    models_module = types.ModuleType("headroom.memory.models")
    models_module.Memory = Memory
    monkeypatch.setitem(sys.modules, "headroom.memory.models", models_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "adapters" / "cache.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.adapters.cache", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.adapters.cache", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.adapters.cache"] = module
    spec.loader.exec_module(module)
    return module, Memory


@pytest.mark.asyncio
async def test_lru_memory_cache_full_behavior(monkeypatch) -> None:
    cache_module, Memory = _load_cache_module(monkeypatch)

    with pytest.raises(ValueError):
        cache_module.LRUMemoryCache(0)

    cache = cache_module.LRUMemoryCache(max_size=2)
    a = Memory(id="a", user_id="u1", session_id="s1", agent_id="g1")
    b = Memory(id="b", user_id="u1", session_id="s1", agent_id="g2")
    c = Memory(id="c", user_id="u2", session_id="s2", agent_id="g3")

    assert await cache.get("missing") is None
    await cache.put(a, ttl_seconds=5)
    await cache.put(b)
    assert cache.keys() == ["a", "b"]
    assert await cache.get("a") is a
    assert cache.keys() == ["b", "a"]

    batch = await cache.get_batch(["a", "missing", "b"])
    assert batch == {"a": a, "b": b}
    assert cache.keys() == ["a", "b"]

    await cache.put(Memory(id="b", user_id="u1", session_id="s1", agent_id="g2"))
    assert cache.keys() == ["a", "b"]
    await cache.put(c)
    assert cache.keys() == ["b", "c"]
    assert cache.contains("a") is False

    await cache.put_batch([a, b], ttl_seconds=3)
    assert cache.keys() == ["a", "b"]
    await cache.put_batch([c])
    assert cache.keys() == ["b", "c"]

    assert await cache.invalidate("b") is True
    assert await cache.invalidate("missing") is False

    wide_cache = cache_module.LRUMemoryCache(max_size=5)
    await wide_cache.put_batch([a, b, c])
    assert await wide_cache.invalidate_batch(["a", "missing", "c"]) == 2

    await wide_cache.put_batch([a, b, c])
    removed_user = await wide_cache.invalidate_scope("u1")
    assert removed_user == 2
    await wide_cache.put_batch([a, b, c])
    removed_session = await wide_cache.invalidate_scope("u1", session_id="s1")
    assert removed_session == 2
    await wide_cache.put_batch([a, b, c])
    removed_agent = await wide_cache.invalidate_scope("u1", session_id="s1", agent_id="g2")
    assert removed_agent == 1

    assert cache.size == 1
    assert cache.max_size == 2
    assert wide_cache.contains("c") is True
    assert wide_cache.keys() == ["a", "c"]
    assert wide_cache.stats() == {"size": 2, "max_size": 5, "utilization": 40.0}

    await cache.clear()
    assert cache.size == 0
    assert cache.keys() == []
