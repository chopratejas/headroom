"""Tests for SQLite memory store.

100% REAL SQLite - no mocks! These tests use actual SQLite
databases in temp directories for realistic testing.
"""

from __future__ import annotations

from headroom.memory.store import Memory, PendingExtraction, SQLiteMemoryStore


class TestMemorySaveAndSearch:
    """Test basic save and search operations."""

    def test_save_and_search_finds_match(self, temp_db):
        """Real SQLite, real FTS5, real queries."""
        store = SQLiteMemoryStore(temp_db)

        store.save("alice", Memory(content="User prefers Python", category="preference"))

        results = store.search("alice", "python", top_k=5)

        assert len(results) == 1
        assert "Python" in results[0].content

    def test_search_no_results_for_unrelated_query(self, temp_db):
        """Search returns empty when no matches."""
        store = SQLiteMemoryStore(temp_db)

        store.save("alice", Memory(content="User prefers Python"))

        results = store.search("alice", "javascript", top_k=5)

        assert len(results) == 0

    def test_search_respects_top_k_limit(self, temp_db):
        """Search returns at most top_k results."""
        store = SQLiteMemoryStore(temp_db)

        for i in range(10):
            store.save("alice", Memory(content=f"Python fact number {i}"))

        results = store.search("alice", "python", top_k=3)

        assert len(results) == 3

    def test_save_preserves_all_fields(self, temp_db):
        """All memory fields are preserved through save/search."""
        store = SQLiteMemoryStore(temp_db)

        original = Memory(
            content="User prefers vim",
            category="preference",
            importance=0.9,
            metadata={"source": "chat"},
        )
        store.save("alice", original)

        results = store.search("alice", "vim", top_k=1)

        assert len(results) == 1
        retrieved = results[0]
        assert retrieved.content == original.content
        assert retrieved.category == original.category
        assert retrieved.importance == original.importance
        assert retrieved.metadata == original.metadata


class TestUserIsolation:
    """Test that memories are isolated by user_id."""

    def test_users_have_separate_memories(self, temp_db):
        """Memories are isolated by user_id."""
        store = SQLiteMemoryStore(temp_db)

        store.save("alice", Memory(content="Likes Python programming"))
        store.save("bob", Memory(content="Likes JavaScript programming"))

        # Search for content that's actually in the memories
        alice_results = store.search("alice", "Python", top_k=10)
        bob_results = store.search("bob", "JavaScript", top_k=10)

        # Each user should only see their own memories
        assert len(alice_results) == 1
        assert "Python" in alice_results[0].content

        assert len(bob_results) == 1
        assert "JavaScript" in bob_results[0].content

    def test_get_all_returns_only_user_memories(self, temp_db):
        """get_all only returns memories for the specified user."""
        store = SQLiteMemoryStore(temp_db)

        store.save("alice", Memory(content="Alice memory 1"))
        store.save("alice", Memory(content="Alice memory 2"))
        store.save("bob", Memory(content="Bob memory"))

        alice_memories = store.get_all("alice")
        bob_memories = store.get_all("bob")

        assert len(alice_memories) == 2
        assert len(bob_memories) == 1


class TestMemoryDeletion:
    """Test deletion operations."""

    def test_delete_specific_memory(self, temp_db):
        """Delete removes a specific memory."""
        store = SQLiteMemoryStore(temp_db)

        mem = Memory(content="To be deleted")
        store.save("alice", mem)

        result = store.delete("alice", mem.id)

        assert result is True
        assert len(store.get_all("alice")) == 0

    def test_delete_nonexistent_returns_false(self, temp_db):
        """Delete returns False for nonexistent memory."""
        store = SQLiteMemoryStore(temp_db)

        result = store.delete("alice", "nonexistent-id")

        assert result is False

    def test_clear_removes_all_user_memories(self, temp_db):
        """Clear removes all memories for a user."""
        store = SQLiteMemoryStore(temp_db)

        for i in range(5):
            store.save("alice", Memory(content=f"Memory {i}"))
        store.save("bob", Memory(content="Bob's memory"))

        count = store.clear("alice")

        assert count == 5
        assert len(store.get_all("alice")) == 0
        assert len(store.get_all("bob")) == 1  # Bob's memory untouched


class TestMemoryStats:
    """Test statistics operations."""

    def test_stats_counts_memories(self, temp_db):
        """Stats returns correct count."""
        store = SQLiteMemoryStore(temp_db)

        store.save("alice", Memory(content="Mem 1", category="preference"))
        store.save("alice", Memory(content="Mem 2", category="fact"))
        store.save("alice", Memory(content="Mem 3", category="fact"))

        stats = store.stats("alice")

        assert stats["total"] == 3
        assert stats["categories"]["preference"] == 1
        assert stats["categories"]["fact"] == 2

    def test_stats_empty_user(self, temp_db):
        """Stats for user with no memories."""
        store = SQLiteMemoryStore(temp_db)

        stats = store.stats("alice")

        assert stats["total"] == 0
        assert stats["categories"] == {}


class TestPendingExtractions:
    """Test pending extraction queue for crash recovery."""

    def test_queue_and_retrieve_pending(self, temp_db):
        """Queue and retrieve pending extractions."""
        store = SQLiteMemoryStore(temp_db)

        pending = PendingExtraction(
            user_id="alice",
            query="What's your favorite language?",
            response="I prefer Python for data science.",
        )
        store.queue_extraction(pending)

        retrieved = store.get_pending_extractions(limit=10)

        assert len(retrieved) == 1
        assert retrieved[0].user_id == "alice"
        assert retrieved[0].query == pending.query
        assert retrieved[0].response == pending.response
        assert retrieved[0].status == "pending"

    def test_update_extraction_status(self, temp_db):
        """Update status of pending extraction."""
        store = SQLiteMemoryStore(temp_db)

        pending = PendingExtraction(user_id="alice", query="Q", response="R")
        store.queue_extraction(pending)

        store.update_extraction_status(pending.id, "processing")

        # Should not appear in pending list anymore
        pending_list = store.get_pending_extractions(status="pending")
        processing_list = store.get_pending_extractions(status="processing")

        assert len(pending_list) == 0
        assert len(processing_list) == 1

    def test_delete_extraction(self, temp_db):
        """Delete completed extraction."""
        store = SQLiteMemoryStore(temp_db)

        pending = PendingExtraction(user_id="alice", query="Q", response="R")
        store.queue_extraction(pending)

        store.delete_extraction(pending.id)

        assert len(store.get_pending_extractions(limit=10)) == 0

    def test_pending_fifo_order(self, temp_db):
        """Pending extractions returned in FIFO order."""
        store = SQLiteMemoryStore(temp_db)

        for i in range(5):
            store.queue_extraction(
                PendingExtraction(user_id="alice", query=f"Q{i}", response=f"R{i}")
            )

        retrieved = store.get_pending_extractions(limit=3)

        assert len(retrieved) == 3
        assert retrieved[0].query == "Q0"
        assert retrieved[1].query == "Q1"
        assert retrieved[2].query == "Q2"


class TestFTS5Features:
    """Test FTS5 full-text search features."""

    def test_phrase_search(self, temp_db):
        """FTS5 supports phrase search with _raw: prefix."""
        store = SQLiteMemoryStore(temp_db)

        store.save("alice", Memory(content="User prefers dark mode"))
        store.save("alice", Memory(content="User is in dark times"))

        # Exact phrase match using raw FTS5 syntax
        results = store.search("alice", '_raw:"dark mode"', top_k=5)

        assert len(results) == 1
        assert "dark mode" in results[0].content

    def test_prefix_search(self, temp_db):
        """FTS5 supports prefix search with _raw: prefix."""
        store = SQLiteMemoryStore(temp_db)

        store.save("alice", Memory(content="User loves Python programming"))
        store.save("alice", Memory(content="User loves JavaScript"))

        # Prefix search with * using raw FTS5 syntax
        results = store.search("alice", "_raw:Pyth*", top_k=5)

        assert len(results) == 1
        assert "Python" in results[0].content

    def test_boolean_and(self, temp_db):
        """FTS5 supports boolean AND with _raw: prefix."""
        store = SQLiteMemoryStore(temp_db)

        store.save("alice", Memory(content="User prefers Python"))
        store.save("alice", Memory(content="User prefers dark mode"))
        store.save("alice", Memory(content="User prefers Python and dark mode"))

        # Boolean AND using raw FTS5 syntax
        results = store.search("alice", "_raw:Python AND dark", top_k=5)

        assert len(results) == 1
        assert "Python" in results[0].content
        assert "dark" in results[0].content
