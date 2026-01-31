"""Isolated HNSW tests - copy of relevant parts from test_hierarchical.py."""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile
from pathlib import Path

import numpy as np
import pytest

from headroom.memory.models import Memory
from headroom.memory.ports import VectorFilter

# Check if hnswlib is available (use lazy check to avoid SIGILL on incompatible CPUs)
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not installed")
class TestHNSWVectorIndex:
    """Tests for HNSWVectorIndex."""

    @pytest.fixture
    def vector_index(self, temp_db_path):
        """Create an HNSW vector index for testing."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        return HNSWVectorIndex(dimension=384, save_path=temp_db_path.with_suffix(".hnsw"))

    @pytest.mark.asyncio
    async def test_index_and_search(self, vector_index):
        """Test indexing and searching vectors."""
        print("\n[TEST] Starting test_index_and_search")

        # Create memories with random embeddings
        np.random.seed(42)
        memories = []
        for i in range(10):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Test content {i}",
                user_id="alice",
                embedding=embedding,
            )
            memories.append(memory)
        print(f"[TEST] Created {len(memories)} memories")

        # Index all memories
        print("[TEST] Indexing...")
        for memory in memories:
            await vector_index.index(memory)
        print("[TEST] All indexed!")

        # Search with first memory's embedding
        filter = VectorFilter(
            query_vector=memories[0].embedding,
            top_k=3,
            user_id="alice",
        )
        print("[TEST] Searching...")
        results = await vector_index.search(filter)
        print(f"[TEST] Found {len(results)} results")

        assert len(results) == 3
        assert results[0].memory.id == memories[0].id
        assert results[0].similarity > 0.99
        print("[TEST] PASSED!")
