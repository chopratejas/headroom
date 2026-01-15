"""Memory wrapper - the main API for Headroom Memory.

One-line integration:
    from headroom import with_memory
    client = with_memory(OpenAI(), user_id="alice")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from headroom.memory.extractor import MemoryExtractor
from headroom.memory.store import Memory, SQLiteMemoryStore
from headroom.memory.worker import ExtractionWorker


class MemoryWrapper:
    """Wraps an LLM client to add automatic memory.

    Intercepts chat completions to:
    1. BEFORE: Inject relevant memories into user message
    2. AFTER: Queue conversation for background memory extraction

    The system prompt is left unchanged to preserve prompt caching.

    Usage:
        client = MemoryWrapper(OpenAI(), user_id="alice")
        response = client.chat.completions.create(...)
    """

    def __init__(
        self,
        client: Any,
        user_id: str,
        db_path: str | Path = "headroom_memory.db",
        extraction_model: str | None = None,
        top_k: int = 5,
        _extractor: Any = None,  # For testing - inject mock
        _store: SQLiteMemoryStore | None = None,  # For testing
    ):
        """Initialize the memory wrapper.

        Args:
            client: LLM client (OpenAI, Anthropic, etc.)
            user_id: User identifier for memory isolation
            db_path: Path to SQLite database
            extraction_model: Override extraction model (auto-detect if None)
            top_k: Number of memories to inject
            _extractor: Override extractor (for testing)
            _store: Override store (for testing)
        """
        self._client = client
        self._user_id = user_id
        self._top_k = top_k

        # Initialize store
        self._store = _store or SQLiteMemoryStore(db_path)

        # Initialize extractor
        self._extractor = _extractor or MemoryExtractor(client, model=extraction_model)

        # Initialize background worker with shorter wait for responsiveness
        self._worker = ExtractionWorker(
            store=self._store,
            extractor=self._extractor,
            max_wait_seconds=5.0,  # Process partial batches after 5s
        )
        self._worker.start()

        # Create wrapped chat interface
        self.chat = _WrappedChat(self)

    def flush_extractions(self, timeout: float = 60.0) -> bool:
        """Force immediate processing of all queued extractions.

        Useful for testing or when you need to ensure memories are saved.

        Args:
            timeout: Max time to wait in seconds

        Returns:
            True if all extractions completed, False if timed out
        """
        return self._worker.flush(timeout=timeout)

    @property
    def memory(self) -> _MemoryAPI:
        """Direct access to memory operations."""
        return _MemoryAPI(self._store, self._user_id)

    def _inject_memories(self, messages: list[dict]) -> list[dict]:
        """Inject relevant memories into messages.

        Memories are prepended to the FIRST user message to preserve
        system prompt caching.

        Args:
            messages: Original messages list

        Returns:
            New messages list with memories injected
        """
        # Find the last user message
        user_content = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        if not user_content:
            return messages

        # Search for relevant memories
        memories = self._store.search(
            self._user_id,
            str(user_content),
            top_k=self._top_k,
        )

        if not memories:
            return messages

        # Build context block
        context_lines = ["<context>"]
        for mem in memories:
            context_lines.append(f"- {mem.content}")
        context_lines.append("</context>")
        context_block = "\n".join(context_lines)

        # Find the first user message and prepend context
        new_messages = copy.deepcopy(messages)
        for msg in new_messages:
            if msg.get("role") == "user":
                original = msg.get("content", "")
                msg["content"] = f"{context_block}\n\n{original}"
                break

        return new_messages

    def _queue_extraction(self, query: str, response: str) -> None:
        """Queue conversation for background memory extraction.

        Args:
            query: User's message
            response: Assistant's response
        """
        self._worker.schedule(self._user_id, query, response)


class _WrappedChat:
    """Wrapped chat interface that intercepts completions."""

    def __init__(self, wrapper: MemoryWrapper):
        self._wrapper = wrapper
        self.completions = _WrappedCompletions(wrapper)


class _WrappedCompletions:
    """Wrapped completions that add memory to requests."""

    def __init__(self, wrapper: MemoryWrapper):
        self._wrapper = wrapper

    def create(self, **kwargs: Any) -> Any:
        """Create a chat completion with memory injection.

        This intercepts the request to:
        1. Inject relevant memories into user message
        2. Forward to the real client
        3. Queue response for background extraction

        All kwargs are passed through to the underlying client.
        """
        messages = kwargs.get("messages", [])

        # 1. Inject memories into user message
        enhanced_messages = self._wrapper._inject_memories(messages)
        kwargs["messages"] = enhanced_messages

        # 2. Forward to real client
        response = self._wrapper._client.chat.completions.create(**kwargs)

        # 3. Queue for extraction (non-blocking)
        self._extract_and_queue(messages, response)

        return response

    def _extract_and_queue(self, original_messages: list[dict], response: Any) -> None:
        """Extract query and response, queue for extraction."""
        # Get the last user message (without context injection)
        user_query = None
        for msg in reversed(original_messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query:
            return

        # Get assistant response
        try:
            assistant_response = response.choices[0].message.content
        except (AttributeError, IndexError):
            return

        if assistant_response:
            self._wrapper._queue_extraction(user_query, assistant_response)


class _MemoryAPI:
    """Direct API for memory operations."""

    def __init__(self, store: SQLiteMemoryStore, user_id: str):
        self._store = store
        self._user_id = user_id

    def search(self, query: str, top_k: int = 5) -> list[Memory]:
        """Search memories.

        Args:
            query: Search query
            top_k: Max results

        Returns:
            Matching memories
        """
        return self._store.search(self._user_id, query, top_k)

    def add(
        self,
        content: str,
        category: str = "fact",
        importance: float = 0.5,
    ) -> Memory:
        """Manually add a memory.

        Args:
            content: Memory content
            category: preference, fact, or context
            importance: 0.0-1.0

        Returns:
            The created memory
        """
        memory = Memory(
            content=content,
            category=category,  # type: ignore
            importance=importance,
        )
        self._store.save(self._user_id, memory)
        return memory

    def get_all(self) -> list[Memory]:
        """Get all memories for this user."""
        return self._store.get_all(self._user_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        return self._store.delete(self._user_id, memory_id)

    def clear(self) -> int:
        """Clear all memories for this user."""
        return self._store.clear(self._user_id)

    def stats(self) -> dict:
        """Get memory statistics."""
        return self._store.stats(self._user_id)


def with_memory(
    client: Any,
    user_id: str,
    db_path: str | Path = "headroom_memory.db",
    extraction_model: str | None = None,
    top_k: int = 5,
    **kwargs: Any,
) -> MemoryWrapper:
    """Wrap an LLM client to add automatic memory.

    One-line integration for adding persistent memory to any LLM client.

    Args:
        client: LLM client (OpenAI, Anthropic, Mistral, Groq, etc.)
        user_id: User identifier for memory isolation
        db_path: Path to SQLite database (default: headroom_memory.db)
        extraction_model: Override extraction model (auto-detects by default)
        top_k: Number of memories to inject per request (default: 5)
        **kwargs: Additional arguments passed to MemoryWrapper

    Returns:
        Wrapped client with automatic memory

    Example:
        from openai import OpenAI
        from headroom import with_memory

        client = with_memory(OpenAI(), user_id="alice")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "I prefer Python"}]
        )
        # Memory automatically extracted in background

        # Later...
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What language should I use?"}]
        )
        # Memory about Python preference automatically injected!
    """
    return MemoryWrapper(
        client=client,
        user_id=user_id,
        db_path=db_path,
        extraction_model=extraction_model,
        top_k=top_k,
        **kwargs,
    )
