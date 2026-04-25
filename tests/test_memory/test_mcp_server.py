from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

pytest.importorskip("mcp")

from headroom.memory.mcp_server import _warm_up_backend
from headroom.memory.models import Memory


@pytest.mark.asyncio
async def test_warm_up_backend_batches_embedding_and_indexing() -> None:
    """Warm-up should batch missing embeddings and vector indexing."""
    warmup_embedding = np.ones(384, dtype=np.float32)
    batch_embeddings = [
        np.full(384, 2.0, dtype=np.float32),
        np.full(384, 3.0, dtype=np.float32),
    ]

    embedder = SimpleNamespace(
        embed=AsyncMock(return_value=warmup_embedding),
        embed_batch=AsyncMock(return_value=batch_embeddings),
    )
    store = SimpleNamespace(save_batch=AsyncMock())
    vector_index = SimpleNamespace(index_batch=AsyncMock(return_value=3))

    memory_without_embedding_a = Memory(content="First", user_id="alice")
    memory_with_embedding = Memory(
        content="Second",
        user_id="alice",
        embedding=np.full(384, 5.0, dtype=np.float32),
    )
    memory_without_embedding_b = Memory(content="Third", user_id="alice")
    memories = [
        memory_without_embedding_a,
        memory_with_embedding,
        memory_without_embedding_b,
    ]

    backend = SimpleNamespace(
        _ensure_initialized=AsyncMock(),
        _hierarchical_memory=SimpleNamespace(
            _embedder=embedder,
            _store=store,
            _vector_index=vector_index,
        ),
        get_user_memories=AsyncMock(return_value=memories),
    )

    await _warm_up_backend(backend, "alice")

    backend._ensure_initialized.assert_awaited_once()
    backend.get_user_memories.assert_awaited_once_with("alice", limit=500)
    embedder.embed.assert_awaited_once_with("warmup")
    embedder.embed_batch.assert_awaited_once_with(["First", "Third"])
    store.save_batch.assert_awaited_once_with(
        [memory_without_embedding_a, memory_without_embedding_b]
    )
    vector_index.index_batch.assert_awaited_once_with(memories)
    assert np.array_equal(memory_without_embedding_a.embedding, batch_embeddings[0])
    assert np.array_equal(memory_without_embedding_b.embedding, batch_embeddings[1])
