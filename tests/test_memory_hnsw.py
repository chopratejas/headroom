from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest


class _FakeIndex:
    saved_indices: dict[str, dict[str, Any]] = {}

    def __init__(self, space: str, dim: int) -> None:
        self.space = space
        self.dim = dim
        self.max_elements: int | None = None
        self.ef_construction: int | None = None
        self.m: int | None = None
        self.ef_search: int | None = None
        self.items: dict[int, Any] = {}
        self.deleted: set[int] = set()
        self.loaded_from: str | None = None

    def init_index(self, max_elements: int, ef_construction: int, M: int) -> None:
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.m = M

    def set_ef(self, ef_search: int) -> None:
        self.ef_search = ef_search

    def add_items(self, embeddings: Any, ids: Any) -> None:
        import numpy as np

        for hnsw_id, embedding in zip(ids.tolist(), embeddings, strict=False):
            self.items[int(hnsw_id)] = np.asarray(embedding, dtype=np.float32)

    def mark_deleted(self, hnsw_id: int) -> None:
        self.deleted.add(hnsw_id)

    def knn_query(self, query_vector: Any, k: int) -> tuple[Any, Any]:
        import numpy as np

        query = np.asarray(query_vector[0], dtype=np.float32)
        scored: list[tuple[int, float]] = []
        for hnsw_id, embedding in self.items.items():
            if hnsw_id in self.deleted:
                continue
            denom = float(np.linalg.norm(query) * np.linalg.norm(embedding))
            similarity = 0.0 if denom == 0 else float(np.dot(query, embedding) / denom)
            scored.append((hnsw_id, similarity))
        scored.sort(key=lambda item: item[1], reverse=True)
        trimmed = scored[:k]
        labels = np.array([[item[0] for item in trimmed]], dtype=np.int64)
        distances = np.array([[1.0 - item[1] for item in trimmed]], dtype=np.float32)
        return labels, distances

    def resize_index(self, new_max_elements: int) -> None:
        self.max_elements = new_max_elements

    def save_index(self, path: str) -> None:
        Path(path).write_text("fake-index")
        self.saved_indices[path] = {
            "items": {str(hid): embedding.tolist() for hid, embedding in self.items.items()},
            "deleted": sorted(self.deleted),
            "dim": self.dim,
            "max_elements": self.max_elements,
            "ef_construction": self.ef_construction,
            "m": self.m,
            "ef_search": self.ef_search,
        }

    def load_index(self, path: str, max_elements: int) -> None:
        import numpy as np

        self.loaded_from = path
        self.max_elements = max_elements
        state = self.saved_indices[path]
        self.items = {
            int(hid): np.asarray(embedding, dtype=np.float32)
            for hid, embedding in state["items"].items()
        }
        self.deleted = set(state["deleted"])
        self.ef_construction = state["ef_construction"]
        self.m = state["m"]
        self.ef_search = state["ef_search"]


def _deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    import headroom.memory.adapters.hnsw as hnsw
    from headroom.memory.adapters.hnsw import HNSWVectorIndex, IndexedMemoryMetadata
    from headroom.memory.models import Memory, ScopeLevel
    from headroom.memory.ports import VectorFilter

    return hnsw, HNSWVectorIndex, IndexedMemoryMetadata, Memory, ScopeLevel, VectorFilter


def _memory(
    memory_id: str,
    embedding: list[float] | None,
    *,
    user_id: str = "user-1",
    session_id: str | None = None,
    agent_id: str | None = None,
    valid_until: datetime | None = None,
    entity_refs: list[str] | None = None,
    importance: float = 0.5,
    created_at: datetime | None = None,
) -> Any:
    import numpy as np

    _, _, _, Memory, _, _ = _deps()
    return Memory(
        id=memory_id,
        content=f"content-{memory_id}",
        user_id=user_id,
        session_id=session_id,
        agent_id=agent_id,
        valid_until=valid_until,
        entity_refs=entity_refs or [],
        created_at=created_at or datetime(2024, 1, 1, tzinfo=timezone.utc),
        importance=importance,
        embedding=np.array(embedding, dtype=np.float32) if embedding is not None else None,
        metadata={"tag": memory_id},
    )


def _install_fake_hnsw(monkeypatch: pytest.MonkeyPatch) -> None:
    hnsw, _, _, _, _, _ = _deps()
    _FakeIndex.saved_indices.clear()
    monkeypatch.setattr(hnsw, "_check_hnswlib_available", lambda: True)
    monkeypatch.setattr(hnsw, "hnswlib", type("FakeHnswLib", (), {"Index": _FakeIndex}))


def test_indexed_memory_metadata_and_availability_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    hnsw, _, IndexedMemoryMetadata, _, _, _ = _deps()
    import numpy as np

    memory = _memory(
        "mem-a",
        [1.0, 0.0],
        session_id="session-1",
        agent_id="agent-1",
        valid_until=datetime(2024, 1, 2, tzinfo=timezone.utc),
        entity_refs=["entity-a"],
        importance=0.9,
    )
    metadata = IndexedMemoryMetadata.from_memory(memory)
    assert metadata.to_dict()["valid_until"] == "2024-01-02T00:00:00+00:00"

    round_tripped = IndexedMemoryMetadata.from_dict(metadata.to_dict())
    restored = round_tripped.to_memory(embedding=np.array([1.0, 0.0], dtype=np.float32))
    assert restored.id == "mem-a"
    assert restored.session_id == "session-1"
    assert restored.agent_id == "agent-1"
    assert restored.metadata == {"tag": "mem-a"}

    monkeypatch.setattr(hnsw, "HNSW_AVAILABLE", None)
    monkeypatch.setattr(hnsw, "hnswlib", None)

    class _Result:
        returncode = 0
        stderr = b""

    import builtins
    import subprocess

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "hnswlib":
            return type("ImportedHnsw", (), {"Index": _FakeIndex})
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Result())
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        assert hnsw._check_hnswlib_available() is True
        assert hnsw._check_hnswlib_available() is True
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)

    monkeypatch.setattr(hnsw, "HNSW_AVAILABLE", None)

    def raise_timeout(*args: Any, **kwargs: Any):
        raise subprocess.TimeoutExpired(cmd="python", timeout=10)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    assert hnsw._check_hnswlib_available() is False


@pytest.mark.asyncio
async def test_hnsw_index_lifecycle_search_filters_and_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, HNSWVectorIndex, _, _, ScopeLevel, VectorFilter = _deps()
    import numpy as np

    _install_fake_hnsw(monkeypatch)

    with pytest.raises(ValueError, match="save_path"):
        HNSWVectorIndex(auto_save=True)

    index = HNSWVectorIndex(
        dimension=2,
        max_elements=2,
        max_entries=2,
        eviction_batch_size=1,
        auto_save=True,
        save_path=tmp_path / "autosave-index",
    )

    assert index.dimension == 2
    assert index.size == 0

    with pytest.raises(ValueError, match="no embedding"):
        await index.index(_memory("missing", None))

    with pytest.raises(ValueError, match="Embedding dimension 3"):
        await index.index(_memory("wrong-dim", [1.0, 2.0, 3.0]))

    low = _memory(
        "low",
        [1.0, 0.0],
        user_id="alice",
        session_id="session-a",
        importance=0.1,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        entity_refs=["project-a"],
    )
    high = _memory(
        "high",
        [0.9, 0.1],
        user_id="alice",
        agent_id="agent-a",
        importance=0.9,
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        entity_refs=["project-a", "shared"],
    )
    newer = _memory(
        "newer",
        [0.0, 1.0],
        user_id="bob",
        session_id="session-b",
        valid_until=datetime(2024, 1, 3, tzinfo=timezone.utc),
        importance=0.7,
        created_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        entity_refs=["project-b"],
    )

    await index.index(low)
    await index.index(high)
    assert index.size == 2

    await index.index(newer)
    assert "low" not in index._metadata
    assert index.stats()["eviction_count"] == 1
    assert index.size == 2

    high_updated = _memory(
        "high",
        [0.8, 0.2],
        user_id="alice",
        agent_id="agent-a",
        importance=0.95,
        created_at=datetime(2024, 1, 4, tzinfo=timezone.utc),
        entity_refs=["project-a", "shared"],
    )
    await index.index(high_updated)
    assert index._metadata["high"].importance == 0.95

    assert await index.update_embedding("missing", np.array([1.0, 0.0], dtype=np.float32)) is False
    with pytest.raises(ValueError, match="Embedding dimension 3"):
        await index.update_embedding("high", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert await index.update_embedding("high", np.array([0.2, 0.8], dtype=np.float32)) is True

    with pytest.raises(ValueError, match="query_text provided"):
        await index.search(VectorFilter(query_text="hello"))
    with pytest.raises(ValueError, match="Either query_vector or query_text"):
        await index.search(VectorFilter())
    with pytest.raises(ValueError, match="Query vector dimension 3"):
        await index.search(VectorFilter(query_vector=np.array([1.0, 2.0, 3.0], dtype=np.float32)))

    results = await index.search(
        VectorFilter(
            query_vector=np.array([1.0, 0.0], dtype=np.float32),
            top_k=2,
            min_similarity=0.1,
            user_id="alice",
            entity_refs=["shared"],
            scope_levels=[ScopeLevel.AGENT],
        )
    )
    assert [result.memory.id for result in results] == ["high"]
    assert results[0].rank == 1
    assert results[0].memory.embedding.tolist() == [0.20000000298023224, 0.800000011920929]

    assert (
        await index.search(
            VectorFilter(
                query_vector=np.array([0.0, 1.0], dtype=np.float32),
                valid_at=datetime(2024, 1, 4, tzinfo=timezone.utc),
                user_id="bob",
                include_superseded=True,
            )
        )
        == []
    )
    current_only = await index.search(
        VectorFilter(
            query_vector=np.array([0.0, 1.0], dtype=np.float32),
            include_superseded=True,
            user_id="bob",
        )
    )
    assert [result.memory.id for result in current_only] == ["newer"]

    assert await index.remove("missing") is False
    assert await index.remove("newer") is True
    assert await index.remove_batch(["high", "missing"]) == 1
    assert index.size == 0

    indexed = await index.index_batch(
        [
            _memory("batch-a", [1.0, 0.0], user_id="alice"),
            _memory("batch-b", [0.0, 1.0], user_id="alice", session_id="session-z"),
            _memory("bad", None),
        ]
    )
    assert indexed == 2
    assert index.size == 2
    assert (
        await index.index_batch([_memory("still-bad", None), _memory("wrong", [1.0, 2.0, 3.0])])
        == 0
    )


def test_hnsw_persistence_clear_stats_and_memory_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, HNSWVectorIndex, IndexedMemoryMetadata, _, _, _ = _deps()
    import numpy as np

    _install_fake_hnsw(monkeypatch)
    index = HNSWVectorIndex(dimension=2, max_elements=3, max_entries=4, eviction_batch_size=2)

    index._memory_to_hnsw = {"a": 0, "b": 1}
    index._hnsw_to_memory = {0: "a", 1: "b"}
    index._next_hnsw_id = 2
    index._eviction_count = 3
    index._metadata = {
        "a": IndexedMemoryMetadata.from_memory(_memory("a", [1.0, 0.0], entity_refs=["x"])),
        "b": IndexedMemoryMetadata.from_memory(_memory("b", [0.0, 1.0], session_id="s1")),
    }
    index._embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
    }
    index._index.add_items(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), np.array([0, 1]))

    base_path = tmp_path / "memory-index"
    index.save_index(base_path)
    meta_payload = json.loads(base_path.with_suffix(".meta").read_text())
    assert meta_payload["dimension"] == 2
    assert meta_payload["eviction_count"] == 3
    assert meta_payload["memory_to_hnsw"] == {"a": 0, "b": 1}

    loaded = HNSWVectorIndex(dimension=2, max_elements=1)
    loaded.load_index(base_path)
    assert loaded.size == 2
    assert loaded._hnsw_to_memory == {0: "a", 1: "b"}
    assert loaded._metadata["a"].memory_id == "a"
    assert loaded._embeddings["b"].tolist() == [0.0, 1.0]

    with pytest.raises(FileNotFoundError, match="HNSW index not found"):
        loaded.load_index(tmp_path / "missing-index")

    wrong_dimension = HNSWVectorIndex(dimension=3)
    with pytest.raises(ValueError, match="Saved index dimension 2"):
        wrong_dimension.load_index(base_path)

    stats = loaded.stats()
    assert stats["size"] == 2
    assert stats["entry_utilization"] == 50.0
    assert stats["utilization"] > 0

    memory_stats = loaded.get_memory_stats()
    assert memory_stats.name == "vector_index"
    assert memory_stats.entry_count == 2
    assert memory_stats.evictions == 3
    assert memory_stats.budget_bytes is not None
    assert memory_stats.size_bytes > 0

    loaded.set_ef_search(77)
    assert loaded.stats()["ef_search"] == 77

    loaded.clear()
    assert loaded.size == 0
    assert loaded.stats()["eviction_count"] == 0


def test_hnsw_passes_filter_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    _, HNSWVectorIndex, IndexedMemoryMetadata, _, ScopeLevel, VectorFilter = _deps()
    _install_fake_hnsw(monkeypatch)
    index = HNSWVectorIndex(dimension=2)
    metadata = IndexedMemoryMetadata(
        memory_id="mem-1",
        user_id="user-1",
        session_id="session-1",
        agent_id=None,
        valid_until=datetime(2024, 1, 5, tzinfo=timezone.utc),
        entity_refs=["entity-a", "entity-b"],
        content="content",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        importance=0.5,
        metadata={"custom": "value"},
    )

    assert (
        index._passes_filter(metadata, VectorFilter(user_id="user-1", include_superseded=True))
        is True
    )
    assert index._passes_filter(metadata, VectorFilter(user_id="other")) is False
    assert (
        index._passes_filter(
            metadata, VectorFilter(session_id="session-1", include_superseded=True)
        )
        is True
    )
    assert index._passes_filter(metadata, VectorFilter(agent_id="agent-1")) is False
    assert (
        index._passes_filter(
            metadata,
            VectorFilter(scope_levels=[ScopeLevel.SESSION], include_superseded=True),
        )
        is True
    )
    assert index._passes_filter(metadata, VectorFilter(scope_levels=[ScopeLevel.AGENT])) is False
    assert (
        index._passes_filter(
            metadata,
            VectorFilter(
                valid_at=datetime(2024, 1, 4, tzinfo=timezone.utc), include_superseded=True
            ),
        )
        is True
    )
    assert (
        index._passes_filter(
            metadata,
            VectorFilter(
                valid_at=datetime(2024, 1, 6, tzinfo=timezone.utc), include_superseded=True
            ),
        )
        is False
    )
    assert index._passes_filter(metadata, VectorFilter(include_superseded=False)) is False
    assert index._passes_filter(metadata, VectorFilter(entity_refs=["missing"])) is False
    assert (
        index._passes_filter(
            metadata,
            VectorFilter(entity_refs=["entity-b"], include_superseded=True),
        )
        is True
    )
