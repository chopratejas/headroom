"""SQLite vector index for Headroom Memory using sqlite-vec.

Provides vector similarity search backed by SQLite, offering:
- True CRUD operations (real deletes, not marks)
- Bounded memory via SQLite page cache
- Persistent storage by default
- Native integration with FTS5 for hybrid search
- No external dependencies beyond sqlite-vec

Note: sqlite-vec is an optional dependency. Install with:
    pip install sqlite-vec

Requirements:
- Python built with loadable extension support
- SQLite 3.41+ recommended (works with older versions)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

import numpy as np

from ..models import Memory, ScopeLevel
from ..ports import VectorFilter, VectorSearchResult

if TYPE_CHECKING:
    from ..tracker import ComponentStats

logger = logging.getLogger(__name__)

# sqlite-vec availability check
_SQLITE_VEC_AVAILABLE: bool | None = None
_sqlite_vec_module: Any = None


def _check_sqlite_vec_available() -> bool:
    """Check if sqlite-vec is available and can be loaded.

    Returns:
        True if sqlite-vec is available and working.
    """
    global _SQLITE_VEC_AVAILABLE, _sqlite_vec_module

    if _SQLITE_VEC_AVAILABLE is not None:
        return _SQLITE_VEC_AVAILABLE

    try:
        import sqlite_vec

        # Test that we can actually load the extension
        test_conn = sqlite3.connect(":memory:")
        test_conn.enable_load_extension(True)
        sqlite_vec.load(test_conn)
        test_conn.enable_load_extension(False)

        # Verify it works
        version = test_conn.execute("SELECT vec_version()").fetchone()[0]
        test_conn.close()

        _sqlite_vec_module = sqlite_vec
        _SQLITE_VEC_AVAILABLE = True
        logger.debug(f"sqlite-vec available, version: {version}")

    except ImportError:
        _SQLITE_VEC_AVAILABLE = False
        logger.debug("sqlite-vec not installed")
    except AttributeError:
        # enable_load_extension not available (Python not built with extension support)
        _SQLITE_VEC_AVAILABLE = False
        logger.debug("Python sqlite3 does not support loadable extensions")
    except Exception as e:
        _SQLITE_VEC_AVAILABLE = False
        logger.debug(f"sqlite-vec check failed: {e}")

    return _SQLITE_VEC_AVAILABLE


def is_sqlite_vec_available() -> bool:
    """Public function to check sqlite-vec availability."""
    return _check_sqlite_vec_available()


@dataclass
class VectorMetadata:
    """Metadata stored alongside vectors for filtering and reconstruction."""

    memory_id: str
    user_id: str
    session_id: str | None
    agent_id: str | None
    valid_until: datetime | None
    entity_refs: list[str]
    content: str
    created_at: datetime
    importance: float
    metadata: dict[str, Any] | None = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "memory_id": self.memory_id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "agent_id": self.agent_id,
                "valid_until": self.valid_until.isoformat() if self.valid_until else None,
                "entity_refs": self.entity_refs,
                "content": self.content,
                "created_at": self.created_at.isoformat(),
                "importance": self.importance,
                "metadata": self.metadata or {},
            }
        )

    @classmethod
    def from_json(cls, data: str) -> VectorMetadata:
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(
            memory_id=d["memory_id"],
            user_id=d["user_id"],
            session_id=d.get("session_id"),
            agent_id=d.get("agent_id"),
            valid_until=(
                datetime.fromisoformat(d["valid_until"]) if d.get("valid_until") else None
            ),
            entity_refs=d.get("entity_refs", []),
            content=d["content"],
            created_at=datetime.fromisoformat(d["created_at"]),
            importance=d.get("importance", 0.5),
            metadata=d.get("metadata"),
        )

    @classmethod
    def from_memory(cls, memory: Memory) -> VectorMetadata:
        """Create metadata from a Memory object."""
        return cls(
            memory_id=memory.id,
            user_id=memory.user_id,
            session_id=memory.session_id,
            agent_id=memory.agent_id,
            valid_until=memory.valid_until,
            entity_refs=memory.entity_refs.copy(),
            content=memory.content,
            created_at=memory.created_at,
            importance=memory.importance,
            metadata=memory.metadata.copy() if memory.metadata else None,
        )

    def to_memory(self, embedding: np.ndarray | None = None) -> Memory:
        """Reconstruct a Memory object from metadata."""
        return Memory(
            id=self.memory_id,
            content=self.content,
            user_id=self.user_id,
            session_id=self.session_id,
            agent_id=self.agent_id,
            valid_until=self.valid_until,
            entity_refs=self.entity_refs.copy(),
            created_at=self.created_at,
            importance=self.importance,
            embedding=embedding,
            metadata=self.metadata.copy() if self.metadata else {},
        )


class SQLiteVectorIndex:
    """SQLite-based vector index using sqlite-vec extension.

    Features:
    - Cosine similarity search via sqlite-vec
    - True CRUD operations (real deletes, not marks)
    - Bounded memory via SQLite page cache
    - Persistent storage by default
    - Post-filtering by user_id, session_id, agent_id, entity_refs
    - Thread-safe operations

    Usage:
        index = SQLiteVectorIndex(dimension=384, db_path="vectors.db")
        await index.index(memory_with_embedding)
        results = await index.search(VectorFilter(
            query_vector=query_embedding,
            top_k=10,
            user_id="alice"
        ))

    Note: sqlite-vec uses brute-force search which is fast enough for
    most use cases (up to ~1M vectors). For larger datasets, consider
    using quantization or a dedicated vector database.
    """

    def __init__(
        self,
        dimension: int = 384,
        db_path: str | Path = "vectors.db",
        page_cache_size_kb: int = 8192,
    ) -> None:
        """Initialize the SQLite vector index.

        Args:
            dimension: Embedding dimension. Default 384 for MiniLM.
            db_path: Path to SQLite database file.
            page_cache_size_kb: SQLite page cache size in KB. Default 8MB.

        Raises:
            ImportError: If sqlite-vec is not available.
        """
        if not _check_sqlite_vec_available():
            raise ImportError(
                "sqlite-vec is required for SQLiteVectorIndex. "
                "Install with: pip install sqlite-vec\n"
                "Note: Requires Python built with loadable extension support. "
                "On macOS, use Homebrew Python: brew install python"
            )

        self._dimension = dimension
        self._db_path = Path(db_path)
        self._page_cache_size_kb = page_cache_size_kb
        self._lock = RLock()

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with sqlite-vec loaded."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        conn.enable_load_extension(True)
        _sqlite_vec_module.load(conn)
        conn.enable_load_extension(False)

        # Configure page cache
        if self._page_cache_size_kb > 0:
            conn.execute(f"PRAGMA cache_size = -{self._page_cache_size_kb}")

        return conn

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_conn() as conn:
            # Create virtual table for vectors with cosine distance
            # sqlite-vec uses float[N] syntax for dimension
            # distance_metric=cosine gives distance in [0, 2] range
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
                USING vec0(embedding float[{self._dimension}] distance_metric=cosine)
            """)

            # Create metadata table (linked by rowid)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vec_metadata (
                    rowid INTEGER PRIMARY KEY,
                    memory_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    agent_id TEXT,
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    valid_until TEXT,
                    entity_refs TEXT NOT NULL DEFAULT '[]',
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
            """)

            # Create indexes for filtering
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vec_meta_memory_id ON vec_metadata(memory_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_meta_user_id ON vec_metadata(user_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vec_meta_session ON vec_metadata(session_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_meta_agent ON vec_metadata(agent_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vec_meta_importance ON vec_metadata(importance)"
            )

            conn.commit()

    @staticmethod
    def _serialize_f32(vector: np.ndarray) -> bytes:
        """Serialize numpy array to compact float32 bytes."""
        return struct.pack(f"{len(vector)}f", *vector.astype(np.float32))

    @staticmethod
    def _deserialize_f32(data: bytes, dimension: int) -> np.ndarray:
        """Deserialize bytes to numpy array."""
        return np.array(struct.unpack(f"{dimension}f", data), dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of vectors indexed."""
        with self._lock:
            with self._get_conn() as conn:
                result = conn.execute("SELECT COUNT(*) FROM vec_metadata").fetchone()[0]
                return int(result)

    async def index(self, memory: Memory) -> None:
        """Index a memory's embedding.

        Args:
            memory: The memory to index (must have embedding).

        Raises:
            ValueError: If memory has no embedding or wrong dimension.
        """
        if memory.embedding is None:
            raise ValueError(f"Memory {memory.id} has no embedding")

        embedding = np.asarray(memory.embedding, dtype=np.float32)
        if embedding.shape[0] != self._dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} does not match "
                f"index dimension {self._dimension}"
            )

        metadata = VectorMetadata.from_memory(memory)

        with self._lock:
            with self._get_conn() as conn:
                # Check if already exists
                existing = conn.execute(
                    "SELECT rowid FROM vec_metadata WHERE memory_id = ?",
                    (memory.id,),
                ).fetchone()

                if existing:
                    # Update existing entry
                    rowid = existing[0]

                    # Update vector
                    conn.execute(
                        "UPDATE vec_embeddings SET embedding = ? WHERE rowid = ?",
                        (self._serialize_f32(embedding), rowid),
                    )

                    # Update metadata
                    conn.execute(
                        """
                        UPDATE vec_metadata SET
                            user_id = ?, session_id = ?, agent_id = ?,
                            importance = ?, created_at = ?, valid_until = ?,
                            entity_refs = ?, content = ?, metadata_json = ?
                        WHERE rowid = ?
                        """,
                        (
                            metadata.user_id,
                            metadata.session_id,
                            metadata.agent_id,
                            metadata.importance,
                            metadata.created_at.isoformat(),
                            metadata.valid_until.isoformat() if metadata.valid_until else None,
                            json.dumps(metadata.entity_refs),
                            metadata.content,
                            json.dumps(metadata.metadata or {}),
                            rowid,
                        ),
                    )
                else:
                    # Insert new entry
                    # First insert metadata to get rowid
                    cursor = conn.execute(
                        """
                        INSERT INTO vec_metadata (
                            memory_id, user_id, session_id, agent_id,
                            importance, created_at, valid_until,
                            entity_refs, content, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            memory.id,
                            metadata.user_id,
                            metadata.session_id,
                            metadata.agent_id,
                            metadata.importance,
                            metadata.created_at.isoformat(),
                            metadata.valid_until.isoformat() if metadata.valid_until else None,
                            json.dumps(metadata.entity_refs),
                            metadata.content,
                            json.dumps(metadata.metadata or {}),
                        ),
                    )
                    rowid = cursor.lastrowid

                    # Insert vector with matching rowid
                    conn.execute(
                        "INSERT INTO vec_embeddings (rowid, embedding) VALUES (?, ?)",
                        (rowid, self._serialize_f32(embedding)),
                    )

                conn.commit()

    async def index_batch(self, memories: list[Memory]) -> int:
        """Index multiple memories.

        Args:
            memories: List of memories to index.

        Returns:
            Number of memories indexed.
        """
        indexed = 0
        for memory in memories:
            if memory.embedding is not None:
                try:
                    await self.index(memory)
                    indexed += 1
                except ValueError:
                    pass
        return indexed

    async def remove(self, memory_id: str) -> bool:
        """Remove a memory from the index.

        Args:
            memory_id: The memory ID to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            with self._get_conn() as conn:
                # Get rowid
                row = conn.execute(
                    "SELECT rowid FROM vec_metadata WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()

                if row is None:
                    return False

                rowid = row[0]

                # Delete from both tables
                conn.execute("DELETE FROM vec_embeddings WHERE rowid = ?", (rowid,))
                conn.execute("DELETE FROM vec_metadata WHERE rowid = ?", (rowid,))
                conn.commit()

                return True

    async def remove_batch(self, memory_ids: list[str]) -> int:
        """Remove multiple memories from the index.

        Args:
            memory_ids: List of memory IDs to remove.

        Returns:
            Number removed.
        """
        removed = 0
        for memory_id in memory_ids:
            if await self.remove(memory_id):
                removed += 1
        return removed

    async def search(self, filter: VectorFilter) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            filter: Search filter with query vector and constraints.

        Returns:
            List of search results sorted by similarity (descending).
        """
        if filter.query_vector is None:
            if filter.query_text is not None:
                raise ValueError(
                    "query_text provided but SQLiteVectorIndex does not embed text. "
                    "Provide query_vector directly or use an Embedder first."
                )
            raise ValueError("query_vector must be provided")

        query_vector = np.asarray(filter.query_vector, dtype=np.float32)
        if query_vector.shape[0] != self._dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[0]} does not match "
                f"index dimension {self._dimension}"
            )

        with self._lock:
            with self._get_conn() as conn:
                # sqlite-vec returns distance (lower = more similar for L2)
                # For cosine, we need to convert: similarity = 1 - distance
                # But sqlite-vec's cosine distance is already 1 - cosine_similarity
                # So similarity = 1 - distance

                # Get more results than needed for post-filtering
                k_with_buffer = filter.top_k * 10

                # Query sqlite-vec for nearest neighbors
                # sqlite-vec requires 'k = ?' constraint
                # Use subquery to get KNN results first, then join with metadata
                rows = conn.execute(
                    """
                    SELECT
                        knn.rowid,
                        knn.distance,
                        m.memory_id,
                        m.user_id,
                        m.session_id,
                        m.agent_id,
                        m.importance,
                        m.created_at,
                        m.valid_until,
                        m.entity_refs,
                        m.content,
                        m.metadata_json
                    FROM (
                        SELECT rowid, distance
                        FROM vec_embeddings
                        WHERE embedding MATCH ?
                          AND k = ?
                    ) knn
                    JOIN vec_metadata m ON knn.rowid = m.rowid
                    ORDER BY knn.distance
                    """,
                    (self._serialize_f32(query_vector), k_with_buffer),
                ).fetchall()

                results: list[VectorSearchResult] = []

                for row in rows:
                    # Convert distance to similarity
                    # sqlite-vec cosine distance = 1 - cosine_similarity
                    similarity = 1.0 - row["distance"]

                    if similarity < filter.min_similarity:
                        continue

                    # Build metadata for filtering
                    meta = VectorMetadata(
                        memory_id=row["memory_id"],
                        user_id=row["user_id"],
                        session_id=row["session_id"],
                        agent_id=row["agent_id"],
                        importance=row["importance"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        valid_until=(
                            datetime.fromisoformat(row["valid_until"])
                            if row["valid_until"]
                            else None
                        ),
                        entity_refs=json.loads(row["entity_refs"]),
                        content=row["content"],
                        metadata=json.loads(row["metadata_json"]),
                    )

                    # Apply filters
                    if not self._passes_filter(meta, filter):
                        continue

                    # Create Memory from metadata
                    memory = meta.to_memory()

                    results.append(
                        VectorSearchResult(
                            memory=memory,
                            similarity=float(similarity),
                            rank=0,
                        )
                    )

                    if len(results) >= filter.top_k:
                        break

                # Assign ranks
                for i, result in enumerate(results):
                    result.rank = i + 1

                return results

    def _passes_filter(self, meta: VectorMetadata, filter: VectorFilter) -> bool:
        """Check if metadata passes filter constraints."""
        if filter.user_id is not None and meta.user_id != filter.user_id:
            return False

        if filter.session_id is not None and meta.session_id != filter.session_id:
            return False

        if filter.agent_id is not None and meta.agent_id != filter.agent_id:
            return False

        if filter.scope_levels is not None:
            if meta.agent_id is not None:
                memory_scope = ScopeLevel.AGENT
            elif meta.session_id is not None:
                memory_scope = ScopeLevel.SESSION
            else:
                memory_scope = ScopeLevel.USER

            if memory_scope not in filter.scope_levels:
                return False

        if filter.valid_at is not None:
            if meta.valid_until is not None and filter.valid_at > meta.valid_until:
                return False

        if not filter.include_superseded:
            if meta.valid_until is not None:
                return False

        if filter.entity_refs is not None and len(filter.entity_refs) > 0:
            if not any(ref in meta.entity_refs for ref in filter.entity_refs):
                return False

        return True

    async def get_embedding(self, memory_id: str) -> np.ndarray | None:
        """Get the stored embedding for a memory.

        Args:
            memory_id: The memory ID.

        Returns:
            The embedding array, or None if not found.
        """
        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    """
                    SELECT v.embedding
                    FROM vec_metadata m
                    JOIN vec_embeddings v ON m.rowid = v.rowid
                    WHERE m.memory_id = ?
                    """,
                    (memory_id,),
                ).fetchone()

                if row is None:
                    return None

                return self._deserialize_f32(row[0], self._dimension)

    def clear(self) -> None:
        """Clear all entries from the index."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM vec_embeddings")
                conn.execute("DELETE FROM vec_metadata")
                conn.commit()

    def stats(self) -> dict[str, Any]:
        """Get index statistics."""
        with self._lock:
            with self._get_conn() as conn:
                count = conn.execute("SELECT COUNT(*) FROM vec_metadata").fetchone()[0]
                users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM vec_metadata").fetchone()[
                    0
                ]

                db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

                return {
                    "size": count,
                    "dimension": self._dimension,
                    "users": users,
                    "db_path": str(self._db_path),
                    "db_size_bytes": db_size,
                    "page_cache_size_kb": self._page_cache_size_kb,
                }

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for MemoryTracker."""
        import sys

        from ..tracker import ComponentStats

        with self._lock:
            with self._get_conn() as conn:
                count = conn.execute("SELECT COUNT(*) FROM vec_metadata").fetchone()[0]

            # SQLite manages memory via page cache
            python_overhead = sys.getsizeof(self) + sys.getsizeof(self._lock)
            page_cache_bytes = self._page_cache_size_kb * 1024

            return ComponentStats(
                name="sqlite_vector_index",
                entry_count=count,
                size_bytes=python_overhead + page_cache_bytes,
                budget_bytes=page_cache_bytes,
                hits=0,
                misses=0,
                evictions=0,  # SQLite handles eviction internally
            )

    def vacuum(self) -> None:
        """Reclaim unused space in the database."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("VACUUM")

    def close(self) -> None:
        """Close the index (cleanup)."""
        pass  # Connection-per-request pattern, nothing to close
