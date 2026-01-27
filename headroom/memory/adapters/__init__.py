"""Memory adapters for Headroom's hierarchical memory system.

This module provides concrete implementations of the memory system's ports:
- SQLiteMemoryStore: SQLite-based memory persistence
- FTS5TextIndex: SQLite FTS5 full-text search index
- HNSWVectorIndex: HNSW-based vector index using hnswlib (optional)
- LRUMemoryCache: Thread-safe LRU cache for hot memories
- InMemoryGraphStore: In-memory graph store for knowledge graphs
- LocalEmbedder: sentence-transformers embedding (local, optional)
- OpenAIEmbedder: OpenAI API embedding (cloud, optional)
- OllamaEmbedder: Ollama API embedding (local server, optional)

Note: Some adapters require optional dependencies. Import errors are
deferred until the adapter is actually used.
"""

# Core adapters (no external dependencies beyond sqlite3)
from headroom.memory.adapters.cache import LRUMemoryCache
from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.adapters.graph import InMemoryGraphStore
from headroom.memory.adapters.sqlite import SQLiteMemoryStore

# Check for optional dependencies availability
try:
    from headroom.memory.adapters.hnsw import HNSW_AVAILABLE
except ImportError:
    HNSW_AVAILABLE = False

# Lazy imports for optional adapters
_HNSWVectorIndex = None
_LocalEmbedder = None
_OpenAIEmbedder = None
_OllamaEmbedder = None


def __getattr__(name: str) -> type:
    """Lazy import for optional adapters."""
    global _HNSWVectorIndex, _LocalEmbedder, _OpenAIEmbedder, _OllamaEmbedder

    if name == "HNSWVectorIndex":
        if _HNSWVectorIndex is None:
            from headroom.memory.adapters.hnsw import HNSWVectorIndex

            _HNSWVectorIndex = HNSWVectorIndex
        return _HNSWVectorIndex

    if name == "LocalEmbedder":
        if _LocalEmbedder is None:
            from headroom.memory.adapters.embedders import LocalEmbedder

            _LocalEmbedder = LocalEmbedder
        return _LocalEmbedder

    if name == "OpenAIEmbedder":
        if _OpenAIEmbedder is None:
            from headroom.memory.adapters.embedders import OpenAIEmbedder

            _OpenAIEmbedder = OpenAIEmbedder
        return _OpenAIEmbedder

    if name == "OllamaEmbedder":
        if _OllamaEmbedder is None:
            from headroom.memory.adapters.embedders import OllamaEmbedder

            _OllamaEmbedder = OllamaEmbedder
        return _OllamaEmbedder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core adapters (always available)
    "FTS5TextIndex",
    "InMemoryGraphStore",
    "LRUMemoryCache",
    "SQLiteMemoryStore",
    # Optional adapters (lazy-loaded)
    "HNSWVectorIndex",
    "LocalEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    # Availability flags
    "HNSW_AVAILABLE",
]
