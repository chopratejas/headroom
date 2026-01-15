"""SQLite + FTS5 memory storage for Headroom Memory.

Simple, fast, local-first storage with full-text search.
No external dependencies - just SQLite (built into Python).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class Memory:
    """A single memory entry."""

    content: str
    category: Literal["preference", "fact", "context"] = "fact"
    importance: float = 0.5
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class PendingExtraction:
    """A conversation pending memory extraction."""

    user_id: str
    query: str
    response: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: Literal["pending", "processing", "done", "failed"] = "pending"


class SQLiteMemoryStore:
    """SQLite + FTS5 storage for memories.

    Features:
    - Full-text search via FTS5
    - User isolation (each user_id has separate memories)
    - Pending extractions for crash recovery
    - Thread-safe with connection per call

    Usage:
        store = SQLiteMemoryStore("./memory.db")
        store.save("alice", Memory(content="Prefers Python"))
        results = store.search("alice", "python")
    """

    def __init__(self, db_path: str | Path = "headroom_memory.db"):
        """Initialize the store.

        Args:
            db_path: Path to SQLite database file. Created if doesn't exist.
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new connection (thread-safe pattern)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            # Main memories table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact',
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)

            # FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='rowid'
                )
            """)

            # Triggers to keep FTS in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)

            # Index for user_id filtering
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user_id
                ON memories(user_id)
            """)

            # Pending extractions table (for crash recovery)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_extractions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pending_status
                ON pending_extractions(status)
            """)

            conn.commit()

    def save(self, user_id: str, memory: Memory) -> None:
        """Save a memory for a user.

        Args:
            user_id: User identifier for isolation
            memory: Memory to save
        """
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, user_id, content, category, importance, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    user_id,
                    memory.content,
                    memory.category,
                    memory.importance,
                    memory.created_at.isoformat(),
                    json.dumps(memory.metadata),
                ),
            )
            conn.commit()

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[Memory]:
        """Search memories using FTS5 full-text search.

        Args:
            user_id: User identifier for isolation
            query: Search query (auto-escaped, or use raw FTS5 syntax with prefix '_raw:')
            top_k: Maximum number of results

        Returns:
            List of matching memories, ranked by relevance
        """
        # Sanitize query for FTS5 (escape special characters unless raw mode)
        if query.startswith("_raw:"):
            fts_query = query[5:]  # Use raw FTS5 syntax
        else:
            fts_query = self._sanitize_fts_query(query)

        if not fts_query.strip():
            return []

        with self._get_conn() as conn:
            # Use FTS5 MATCH with BM25 ranking, filtered by user_id
            cursor = conn.execute(
                """
                SELECT m.*, bm25(memories_fts) as rank
                FROM memories m
                JOIN memories_fts ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ? AND m.user_id = ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, user_id, top_k),
            )

            results = []
            for row in cursor:
                results.append(
                    Memory(
                        id=row["id"],
                        content=row["content"],
                        category=row["category"],
                        importance=row["importance"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        metadata=json.loads(row["metadata"]),
                    )
                )
            return results

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize a query for FTS5.

        Escapes special characters and converts to prefix search for better matching.

        Args:
            query: Raw user query

        Returns:
            FTS5-safe query string
        """
        # FTS5 special characters that need escaping
        # We use a simple approach: extract words and use OR between them
        import re

        # Extract alphanumeric words
        words = re.findall(r"\w+", query)

        if not words:
            return ""

        # Use OR between words with prefix matching for flexibility
        # This allows "What language" to match "Python" memories when searching
        # by using prefix matching (word*)
        escaped_words = []
        for word in words:
            # Quote each word to handle any remaining special chars
            escaped_words.append(f'"{word}"')

        return " OR ".join(escaped_words)

    def get_all(self, user_id: str) -> list[Memory]:
        """Get all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            All memories for the user, ordered by creation time (newest first)
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            )

            return [
                Memory(
                    id=row["id"],
                    content=row["content"],
                    category=row["category"],
                    importance=row["importance"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    metadata=json.loads(row["metadata"]),
                )
                for row in cursor
            ]

    def delete(self, user_id: str, memory_id: str) -> bool:
        """Delete a specific memory.

        Args:
            user_id: User identifier
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear(self, user_id: str) -> int:
        """Delete all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of memories deleted
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return cursor.rowcount

    def stats(self, user_id: str) -> dict:
        """Get memory statistics for a user.

        Args:
            user_id: User identifier

        Returns:
            Dict with count, categories breakdown, etc.
        """
        with self._get_conn() as conn:
            # Total count
            total = conn.execute(
                "SELECT COUNT(*) as count FROM memories WHERE user_id = ?",
                (user_id,),
            ).fetchone()["count"]

            # Category breakdown
            categories = {}
            for row in conn.execute(
                """
                SELECT category, COUNT(*) as count
                FROM memories WHERE user_id = ?
                GROUP BY category
                """,
                (user_id,),
            ):
                categories[row["category"]] = row["count"]

            return {
                "total": total,
                "categories": categories,
            }

    # --- Pending Extractions (for crash recovery) ---

    def queue_extraction(self, pending: PendingExtraction) -> None:
        """Queue a conversation for memory extraction.

        Args:
            pending: The pending extraction to queue
        """
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO pending_extractions (id, user_id, query, response, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pending.id,
                    pending.user_id,
                    pending.query,
                    pending.response,
                    pending.created_at.isoformat(),
                    pending.status,
                ),
            )
            conn.commit()

    def get_pending_extractions(
        self, limit: int = 10, status: str = "pending"
    ) -> list[PendingExtraction]:
        """Get pending extractions for processing.

        Args:
            limit: Maximum number to return
            status: Filter by status

        Returns:
            List of pending extractions
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM pending_extractions
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (status, limit),
            )

            return [
                PendingExtraction(
                    id=row["id"],
                    user_id=row["user_id"],
                    query=row["query"],
                    response=row["response"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    status=row["status"],
                )
                for row in cursor
            ]

    def update_extraction_status(self, extraction_id: str, status: str) -> None:
        """Update the status of a pending extraction.

        Args:
            extraction_id: ID of the extraction
            status: New status
        """
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE pending_extractions SET status = ? WHERE id = ?",
                (status, extraction_id),
            )
            conn.commit()

    def delete_extraction(self, extraction_id: str) -> None:
        """Delete a completed extraction.

        Args:
            extraction_id: ID of the extraction to delete
        """
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM pending_extractions WHERE id = ?",
                (extraction_id,),
            )
            conn.commit()
