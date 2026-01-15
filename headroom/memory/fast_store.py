"""Fast embedding-based memory store.

Sub-100ms write and read latency by:
1. NO LLM extraction - just embed and store
2. Vector similarity search - not keyword matching
3. Optional local embeddings for sub-10ms latency

This replaces the slow LLM-based extraction approach.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryChunk:
    """A memory chunk with text and embedding."""

    id: str = field(default_factory=lambda: str(uuid4()))
    text: str = ""
    role: str = "user"  # "user" or "assistant"
    embedding: np.ndarray | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "role": self.role,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryChunk:
        """Create from dictionary."""
        embedding = None
        if data.get("embedding"):
            embedding = np.array(data["embedding"], dtype=np.float32)
        return cls(
            id=data["id"],
            text=data["text"],
            role=data.get("role", "user"),
            embedding=embedding,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


# Type aliases for embedding functions
EmbedFn = Callable[[str], np.ndarray]
BatchEmbedFn = Callable[[list[str]], list[np.ndarray]]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and multiple vectors."""
    # Normalize query
    query_norm = query / (np.linalg.norm(query) + 1e-9)
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vectors_norm = vectors / norms
    # Dot product - cast to ndarray to satisfy mypy
    result: np.ndarray = np.dot(vectors_norm, query_norm)
    return result


class FastMemoryStore:
    """Fast embedding-based memory store.

    Features:
    - Sub-100ms write latency (no LLM, just embedding)
    - Sub-50ms read latency (vector similarity search)
    - Pluggable embedding functions (local or API)
    - SQLite storage with in-memory vector cache

    Usage:
        store = FastMemoryStore(db_path, embed_fn=my_embed_fn)
        store.add("user_123", "I prefer Python", role="user")
        results = store.search("user_123", "programming language", top_k=5)
    """

    def __init__(
        self,
        db_path: str | Path,
        embed_fn: EmbedFn | None = None,
        embedding_dim: int = 1536,  # OpenAI default
    ):
        """Initialize the store.

        Args:
            db_path: Path to SQLite database
            embed_fn: Function to embed text (if None, must call set_embed_fn later)
            embedding_dim: Dimension of embeddings
        """
        self.db_path = Path(db_path)
        self.embed_fn = embed_fn
        self.embedding_dim = embedding_dim

        # In-memory vector cache for fast similarity search
        self._vector_cache: dict[
            str, dict[str, np.ndarray]
        ] = {}  # user_id -> {chunk_id -> embedding}
        self._chunk_cache: dict[str, dict[str, MemoryChunk]] = {}  # user_id -> {chunk_id -> chunk}

        self._init_db()
        self._load_cache()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    embedding BLOB,
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_user_id
                ON memory_chunks(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_timestamp
                ON memory_chunks(user_id, timestamp DESC)
            """)
            conn.commit()

    def _load_cache(self) -> None:
        """Load all embeddings into memory for fast search."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT id, user_id, text, role, embedding, timestamp, metadata
                FROM memory_chunks
                WHERE embedding IS NOT NULL
            """)

            for row in cursor:
                chunk_id, user_id, text, role, embedding_blob, timestamp, metadata = row

                if user_id not in self._vector_cache:
                    self._vector_cache[user_id] = {}
                    self._chunk_cache[user_id] = {}

                # Deserialize embedding
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)

                self._vector_cache[user_id][chunk_id] = embedding

                chunk = MemoryChunk(
                    id=chunk_id,
                    text=text,
                    role=role,
                    embedding=embedding,
                    timestamp=datetime.fromisoformat(timestamp),
                    metadata=json.loads(metadata) if metadata else {},
                )
                self._chunk_cache[user_id][chunk_id] = chunk

        logger.debug(f"Loaded {sum(len(v) for v in self._vector_cache.values())} chunks into cache")

    def set_embed_fn(self, embed_fn: EmbedFn) -> None:
        """Set the embedding function."""
        self.embed_fn = embed_fn

    def add(
        self,
        user_id: str,
        text: str,
        role: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryChunk:
        """Add a memory chunk.

        This is the FAST path - just embed and store, no LLM extraction.
        Typical latency: <50ms with API embeddings, <10ms with local.

        Args:
            user_id: User/entity identifier
            text: Text to store
            role: "user" or "assistant"
            metadata: Optional metadata

        Returns:
            The created MemoryChunk
        """
        if not self.embed_fn:
            raise ValueError("No embedding function set. Call set_embed_fn() first.")

        start_time = time.perf_counter()

        # Embed the text
        embedding = self.embed_fn(text)
        embed_time = time.perf_counter() - start_time

        # Create chunk
        chunk = MemoryChunk(
            text=text,
            role=role,
            embedding=embedding,
            metadata=metadata or {},
        )

        # Store in SQLite
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO memory_chunks (id, user_id, text, role, embedding, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    user_id,
                    chunk.text,
                    chunk.role,
                    embedding.astype(np.float32).tobytes(),
                    chunk.timestamp.isoformat(),
                    json.dumps(chunk.metadata),
                ),
            )
            conn.commit()

        # Update cache
        if user_id not in self._vector_cache:
            self._vector_cache[user_id] = {}
            self._chunk_cache[user_id] = {}

        self._vector_cache[user_id][chunk.id] = embedding
        self._chunk_cache[user_id][chunk.id] = chunk

        total_time = time.perf_counter() - start_time
        logger.debug(f"Added chunk in {total_time * 1000:.1f}ms (embed: {embed_time * 1000:.1f}ms)")

        return chunk

    def add_turn(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryChunk, MemoryChunk]:
        """Add a conversation turn (user message + assistant response).

        Convenience method that stores both parts of a turn.

        Args:
            user_id: User/entity identifier
            user_message: The user's message
            assistant_response: The assistant's response
            metadata: Optional metadata for both chunks

        Returns:
            Tuple of (user_chunk, assistant_chunk)
        """
        user_chunk = self.add(user_id, user_message, role="user", metadata=metadata)
        assistant_chunk = self.add(user_id, assistant_response, role="assistant", metadata=metadata)
        return user_chunk, assistant_chunk

    def add_turn_batched(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        batch_embed_fn: BatchEmbedFn,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryChunk, MemoryChunk]:
        """Add a conversation turn using BATCHED embedding (single API call).

        This is the FASTEST path - embeds both messages in ONE API call.
        Typical latency: 50-100ms total vs 200-400ms with individual calls.

        Args:
            user_id: User/entity identifier
            user_message: The user's message
            assistant_response: The assistant's response
            batch_embed_fn: Batch embedding function
            metadata: Optional metadata for both chunks

        Returns:
            Tuple of (user_chunk, assistant_chunk)
        """
        start_time = time.perf_counter()

        # Embed BOTH messages in ONE API call
        embeddings = batch_embed_fn([user_message, assistant_response])
        embed_time = time.perf_counter() - start_time

        # Create chunks
        user_chunk = MemoryChunk(
            text=user_message,
            role="user",
            embedding=embeddings[0],
            metadata=metadata or {},
        )
        assistant_chunk = MemoryChunk(
            text=assistant_response,
            role="assistant",
            embedding=embeddings[1],
            metadata=metadata or {},
        )

        # Store in SQLite (batch insert)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executemany(
                """
                INSERT INTO memory_chunks (id, user_id, text, role, embedding, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        user_chunk.id,
                        user_id,
                        user_chunk.text,
                        user_chunk.role,
                        embeddings[0].astype(np.float32).tobytes(),
                        user_chunk.timestamp.isoformat(),
                        json.dumps(user_chunk.metadata),
                    ),
                    (
                        assistant_chunk.id,
                        user_id,
                        assistant_chunk.text,
                        assistant_chunk.role,
                        embeddings[1].astype(np.float32).tobytes(),
                        assistant_chunk.timestamp.isoformat(),
                        json.dumps(assistant_chunk.metadata),
                    ),
                ],
            )
            conn.commit()

        # Update cache
        if user_id not in self._vector_cache:
            self._vector_cache[user_id] = {}
            self._chunk_cache[user_id] = {}

        self._vector_cache[user_id][user_chunk.id] = embeddings[0]
        self._vector_cache[user_id][assistant_chunk.id] = embeddings[1]
        self._chunk_cache[user_id][user_chunk.id] = user_chunk
        self._chunk_cache[user_id][assistant_chunk.id] = assistant_chunk

        total_time = time.perf_counter() - start_time
        logger.debug(
            f"Added turn (batched) in {total_time * 1000:.1f}ms (embed: {embed_time * 1000:.1f}ms)"
        )

        return user_chunk, assistant_chunk

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        role_filter: str | None = None,
    ) -> list[tuple[MemoryChunk, float]]:
        """Search for relevant memory chunks.

        Uses vector similarity search for semantic matching.
        Typical latency: <50ms with API embeddings, <10ms with local.

        Args:
            user_id: User/entity identifier
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold
            role_filter: Optional filter by role ("user" or "assistant")

        Returns:
            List of (chunk, similarity_score) tuples, sorted by relevance
        """
        if not self.embed_fn:
            raise ValueError("No embedding function set. Call set_embed_fn() first.")

        start_time = time.perf_counter()

        # Check if user has any memories
        if user_id not in self._vector_cache or not self._vector_cache[user_id]:
            return []

        # Embed query
        query_embedding = self.embed_fn(query)
        embed_time = time.perf_counter() - start_time

        # Get user's vectors
        chunk_ids = list(self._vector_cache[user_id].keys())
        vectors = np.array([self._vector_cache[user_id][cid] for cid in chunk_ids])

        # Compute similarities
        similarities = cosine_similarity_batch(query_embedding, vectors)
        search_time = time.perf_counter() - start_time - embed_time

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        # Collect results
        results = []
        for idx in sorted_indices:
            chunk_id = chunk_ids[idx]
            similarity = float(similarities[idx])

            if similarity < min_similarity:
                break

            chunk = self._chunk_cache[user_id][chunk_id]

            # Apply role filter
            if role_filter and chunk.role != role_filter:
                continue

            results.append((chunk, similarity))

            if len(results) >= top_k:
                break

        total_time = time.perf_counter() - start_time
        logger.debug(
            f"Search completed in {total_time * 1000:.1f}ms "
            f"(embed: {embed_time * 1000:.1f}ms, search: {search_time * 1000:.1f}ms)"
        )

        return results

    def get_recent(
        self,
        user_id: str,
        limit: int = 10,
        role_filter: str | None = None,
    ) -> list[MemoryChunk]:
        """Get recent memory chunks.

        Args:
            user_id: User/entity identifier
            limit: Maximum number of chunks to return
            role_filter: Optional filter by role

        Returns:
            List of chunks, sorted by timestamp (newest first)
        """
        if user_id not in self._chunk_cache:
            return []

        chunks = list(self._chunk_cache[user_id].values())

        # Apply role filter
        if role_filter:
            chunks = [c for c in chunks if c.role == role_filter]

        # Sort by timestamp
        chunks.sort(key=lambda c: c.timestamp, reverse=True)

        return chunks[:limit]

    def get_all(self, user_id: str) -> list[MemoryChunk]:
        """Get all memory chunks for a user."""
        if user_id not in self._chunk_cache:
            return []
        return list(self._chunk_cache[user_id].values())

    def delete(self, user_id: str, chunk_id: str) -> bool:
        """Delete a specific chunk."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM memory_chunks WHERE id = ? AND user_id = ?",
                (chunk_id, user_id),
            )
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted and user_id in self._vector_cache:
            self._vector_cache[user_id].pop(chunk_id, None)
            self._chunk_cache[user_id].pop(chunk_id, None)

        return deleted

    def clear(self, user_id: str) -> int:
        """Clear all memories for a user."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM memory_chunks WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            count = cursor.rowcount

        self._vector_cache.pop(user_id, None)
        self._chunk_cache.pop(user_id, None)

        return count

    def stats(self, user_id: str) -> dict[str, Any]:
        """Get statistics for a user."""
        chunks = self.get_all(user_id)
        return {
            "total": len(chunks),
            "user_messages": sum(1 for c in chunks if c.role == "user"),
            "assistant_messages": sum(1 for c in chunks if c.role == "assistant"),
        }


# =============================================================================
# Embedding Functions
# =============================================================================


def create_openai_embed_fn(
    client: Any,
    model: str = "text-embedding-3-small",
) -> EmbedFn:
    """Create an embedding function using OpenAI API.

    Typical latency: 30-100ms per call.

    Args:
        client: OpenAI client
        model: Embedding model to use

    Returns:
        Embedding function
    """

    def embed(text: str) -> np.ndarray:
        response = client.embeddings.create(
            model=model,
            input=text,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    return embed


def create_openai_batch_embed_fn(
    client: Any,
    model: str = "text-embedding-3-small",
) -> BatchEmbedFn:
    """Create a BATCH embedding function using OpenAI API.

    Much faster than individual calls - single API round trip for multiple texts.
    Typical latency: 50-200ms for 10 texts vs 500-2000ms for 10 individual calls.

    Args:
        client: OpenAI client
        model: Embedding model to use

    Returns:
        Batch embedding function
    """

    def embed_batch(texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        response = client.embeddings.create(
            model=model,
            input=texts,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [np.array(d.embedding, dtype=np.float32) for d in sorted_data]

    return embed_batch


def create_local_embed_fn(
    model_name: str = "all-MiniLM-L6-v2",
) -> EmbedFn:
    """Create an embedding function using local sentence-transformers.

    Typical latency: 5-20ms per call (after model load).

    Args:
        model_name: Sentence-transformers model name

    Returns:
        Embedding function
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        ) from None

    model = SentenceTransformer(model_name)

    def embed(text: str) -> np.ndarray:
        return model.encode(text, convert_to_numpy=True).astype(np.float32)

    return embed
