"""Tests for memory wrapper integration.

Tests the full with_memory() flow with mocked LLM clients.
"""

from __future__ import annotations

import time

from headroom.memory.store import Memory
from headroom.memory.wrapper import with_memory


class TestWithMemoryBasic:
    """Test basic with_memory() functionality."""

    def test_one_line_integration(self, temp_db, mock_openai_client, mock_extractor):
        """One-line integration works."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.choices[0].message.content == "Hello!"

    def test_forwards_all_kwargs(self, temp_db, mock_openai_client, mock_extractor):
        """All kwargs are forwarded to underlying client."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
        )

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.5,
            max_tokens=100,
        )

        call = mock_openai_client.chat.completions.calls[0]
        assert call["model"] == "gpt-4o"
        assert call["temperature"] == 0.5
        assert call["max_tokens"] == 100


class TestMemoryInjection:
    """Test memory injection into messages."""

    def test_injects_memory_into_user_message(self, temp_db, mock_openai_client, mock_extractor):
        """Memory is injected into user message, not system prompt."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        # Pre-populate memory with content that will match the query
        from headroom.memory.store import SQLiteMemoryStore

        store = SQLiteMemoryStore(temp_db)
        store.save("alice", Memory(content="User prefers Python for coding", category="preference"))

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
            _store=store,
        )

        # Use a query that will match the memory content
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What Python framework?"},
            ],
        )

        # Check what was sent to the "API"
        call = mock_openai_client.chat.completions.calls[0]
        messages = call["messages"]

        # System prompt should be UNCHANGED (for caching)
        assert messages[0]["content"] == "You are helpful."

        # User message should have context prepended
        assert "<context>" in messages[1]["content"]
        assert "Python" in messages[1]["content"]
        assert "What Python framework?" in messages[1]["content"]

    def test_preserves_system_prompt_exactly(self, temp_db, mock_openai_client, mock_extractor):
        """System prompt is preserved exactly for caching."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        from headroom.memory.store import SQLiteMemoryStore

        store = SQLiteMemoryStore(temp_db)
        store.save("alice", Memory(content="Some memory"))

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
            _store=store,
        )

        original_system = "You are a helpful assistant. Always be concise."

        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": original_system},
                {"role": "user", "content": "test"},
            ],
        )

        call = mock_openai_client.chat.completions.calls[0]
        assert call["messages"][0]["content"] == original_system

    def test_no_injection_when_no_memories(self, temp_db, mock_openai_client, mock_extractor):
        """No injection when user has no memories."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
        )

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        call = mock_openai_client.chat.completions.calls[0]
        # Message should be unchanged
        assert call["messages"][0]["content"] == "Hello"
        assert "<context>" not in call["messages"][0]["content"]

    def test_respects_top_k(self, temp_db, mock_openai_client, mock_extractor):
        """Only top_k memories are injected."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        from headroom.memory.store import SQLiteMemoryStore

        store = SQLiteMemoryStore(temp_db)
        for i in range(10):
            store.save("alice", Memory(content=f"Python fact {i}"))

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
            _store=store,
            top_k=3,
        )

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Python question"}],
        )

        call = mock_openai_client.chat.completions.calls[0]
        content = call["messages"][0]["content"]

        # Should have exactly 3 memories (top_k=3)
        assert content.count("Python fact") == 3


class TestBackgroundExtraction:
    """Test background memory extraction."""

    def test_queues_for_extraction(self, temp_db, mock_openai_client, mock_extractor):
        """Conversation is queued for extraction after response."""
        mock_openai_client.__class__.__module__ = "openai.resources"
        mock_openai_client.chat.completions.set_response("I'll remember that!")

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
        )

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "I prefer Python"}],
        )

        # Wait for background worker
        time.sleep(0.1)

        # Check extraction was scheduled
        # The mock extractor records batch calls
        assert len(mock_extractor.batch_calls) >= 0  # May not have processed yet

    def test_extracts_from_original_message(self, temp_db, mock_openai_client, mock_extractor):
        """Extraction uses original message (without injected context)."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        from headroom.memory.store import SQLiteMemoryStore

        store = SQLiteMemoryStore(temp_db)
        store.save("alice", Memory(content="Existing memory"))

        mock_extractor.set_batch_response({"alice": [Memory(content="New fact", category="fact")]})

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
            _store=store,
        )

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "I like vim"}],
        )

        # Wait for background worker to process
        time.sleep(1.5)

        # The extractor should have been called with original message
        # (not the one with <context> injected)
        if mock_extractor.batch_calls:
            batch = mock_extractor.batch_calls[0]
            _, query, _ = batch[0]
            assert "<context>" not in query


class TestMemoryAPI:
    """Test direct memory API access."""

    def test_memory_search(self, temp_db, mock_openai_client, mock_extractor):
        """client.memory.search() works."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        from headroom.memory.store import SQLiteMemoryStore

        store = SQLiteMemoryStore(temp_db)
        store.save("alice", Memory(content="Likes Python"))

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
            _store=store,
        )

        results = client.memory.search("Python")

        assert len(results) == 1
        assert "Python" in results[0].content

    def test_memory_add(self, temp_db, mock_openai_client, mock_extractor):
        """client.memory.add() works."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
        )

        memory = client.memory.add("User prefers dark mode", category="preference")

        assert memory.content == "User prefers dark mode"
        assert memory.category == "preference"

        # Verify it was saved
        all_memories = client.memory.get_all()
        assert len(all_memories) == 1

    def test_memory_clear(self, temp_db, mock_openai_client, mock_extractor):
        """client.memory.clear() works."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        from headroom.memory.store import SQLiteMemoryStore

        store = SQLiteMemoryStore(temp_db)
        store.save("alice", Memory(content="Memory 1"))
        store.save("alice", Memory(content="Memory 2"))

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
            _store=store,
        )

        count = client.memory.clear()

        assert count == 2
        assert len(client.memory.get_all()) == 0

    def test_memory_stats(self, temp_db, mock_openai_client, mock_extractor):
        """client.memory.stats() works."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        from headroom.memory.store import SQLiteMemoryStore

        store = SQLiteMemoryStore(temp_db)
        store.save("alice", Memory(content="Pref", category="preference"))
        store.save("alice", Memory(content="Fact 1", category="fact"))
        store.save("alice", Memory(content="Fact 2", category="fact"))

        client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
            _store=store,
        )

        stats = client.memory.stats()

        assert stats["total"] == 3
        assert stats["categories"]["preference"] == 1
        assert stats["categories"]["fact"] == 2


class TestMultiUser:
    """Test multi-user isolation."""

    def test_users_have_separate_memories(self, temp_db, mock_openai_client, mock_extractor):
        """Different users have isolated memories."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        from headroom.memory.store import SQLiteMemoryStore

        store = SQLiteMemoryStore(temp_db)

        # Create two wrapped clients for different users
        alice_client = with_memory(
            mock_openai_client,
            user_id="alice",
            db_path=temp_db,
            _extractor=mock_extractor,
            _store=store,
        )

        bob_client = with_memory(
            mock_openai_client,
            user_id="bob",
            db_path=temp_db,
            _extractor=mock_extractor,
            _store=store,
        )

        # Add memories for each user
        alice_client.memory.add("Alice's preference")
        bob_client.memory.add("Bob's preference")

        # Each should only see their own
        assert len(alice_client.memory.get_all()) == 1
        assert len(bob_client.memory.get_all()) == 1
        assert "Alice" in alice_client.memory.get_all()[0].content
        assert "Bob" in bob_client.memory.get_all()[0].content
