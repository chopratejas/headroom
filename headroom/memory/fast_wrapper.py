"""Fast Memory Wrapper - Zero-latency inline extraction + semantic retrieval.

This is the ultimate memory solution:
1. ZERO extra latency - memories extracted as part of LLM response (Letta-style)
2. Semantic retrieval - vector similarity for intelligent memory lookup
3. Local embeddings - sub-50ms retrieval, no API calls needed

Usage:
    from headroom.memory import with_fast_memory

    client = with_fast_memory(OpenAI(), user_id="alice")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I prefer Python"}]
    )
    # Memory extracted INLINE - zero extra latency!
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from headroom.memory.fast_store import (
    FastMemoryStore,
    MemoryChunk,
    create_local_embed_fn,
    create_openai_embed_fn,
)
from headroom.memory.inline_extractor import (
    inject_memory_instruction,
    parse_response_with_memory,
)


class FastMemoryWrapper:
    """Wraps an LLM client with zero-latency inline memory extraction.

    Architecture:
    1. BEFORE: Inject relevant memories into user message (semantic search)
    2. DURING: Memory instruction is in system prompt
    3. AFTER: Parse memory block from response, store extracted memories

    All memory operations happen as part of the normal LLM flow - no extra calls!
    """

    def __init__(
        self,
        client: Any,
        user_id: str,
        db_path: str | Path = "headroom_fast_memory.db",
        top_k: int = 5,
        use_local_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        _store: FastMemoryStore | None = None,
    ):
        """Initialize the fast memory wrapper.

        Args:
            client: OpenAI-compatible LLM client
            user_id: User identifier for memory isolation
            db_path: Path to SQLite database
            top_k: Number of memories to inject
            use_local_embeddings: Use local model (fast) or OpenAI API
            embedding_model: Model name for local embeddings
            _store: Override store (for testing)
        """
        self._client = client
        self._user_id = user_id
        self._top_k = top_k

        # Initialize store with appropriate embedding function
        if _store:
            self._store = _store
        elif use_local_embeddings:
            embed_fn = create_local_embed_fn(embedding_model)
            # MiniLM-L6-v2 produces 384-dim embeddings
            self._store = FastMemoryStore(db_path, embed_fn=embed_fn, embedding_dim=384)
        else:
            embed_fn = create_openai_embed_fn(client)
            self._store = FastMemoryStore(db_path, embed_fn=embed_fn)

        # Create wrapped chat interface
        self.chat = _FastWrappedChat(self)

    @property
    def memory(self) -> _FastMemoryAPI:
        """Direct access to memory operations."""
        return _FastMemoryAPI(self._store, self._user_id)

    def _inject_memories(self, messages: list[dict]) -> list[dict]:
        """Inject relevant memories into user message.

        Uses semantic search (vector similarity) to find relevant memories.
        Injects into FIRST user message to preserve system prompt caching.

        Args:
            messages: Original messages list

        Returns:
            New messages with memories injected
        """
        # Find the last user message for search context
        user_content = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        if not user_content:
            return messages

        # Semantic search for relevant memories
        results = self._store.search(self._user_id, str(user_content), top_k=self._top_k)

        if not results:
            return messages

        # Build context block
        context_lines = ["<context>"]
        for chunk, _score in results:
            context_lines.append(f"- {chunk.text}")
        context_lines.append("</context>")
        context_block = "\n".join(context_lines)

        # Inject into first user message
        new_messages = copy.deepcopy(messages)
        for msg in new_messages:
            if msg.get("role") == "user":
                original = msg.get("content", "")
                msg["content"] = f"{context_block}\n\n{original}"
                break

        return new_messages

    def _store_memories(self, memories: list[dict[str, Any]]) -> None:
        """Store extracted memories.

        Args:
            memories: List of memory dicts from inline extraction
        """
        for mem in memories:
            content = mem.get("content", "")
            category = mem.get("category", "fact")
            if content:
                self._store.add(
                    self._user_id,
                    content,
                    role="memory",
                    metadata={"category": category, "source": "inline_extraction"},
                )


class _FastWrappedChat:
    """Wrapped chat interface."""

    def __init__(self, wrapper: FastMemoryWrapper):
        self._wrapper = wrapper
        self.completions = _FastWrappedCompletions(wrapper)


class _FastWrappedCompletions:
    """Wrapped completions with inline memory extraction."""

    def __init__(self, wrapper: FastMemoryWrapper):
        self._wrapper = wrapper

    def create(self, **kwargs: Any) -> Any:
        """Create chat completion with inline memory extraction.

        Flow:
        1. Search for relevant memories (semantic)
        2. Inject memories into user message
        3. Add memory instruction to system prompt
        4. Forward to LLM
        5. Parse response to extract memories
        6. Store extracted memories
        7. Return clean response (without memory block)
        """
        messages = kwargs.get("messages", [])

        # 1. Inject relevant memories into user message
        enhanced_messages = self._wrapper._inject_memories(messages)

        # 2. Add memory extraction instruction to system prompt
        enhanced_messages = inject_memory_instruction(enhanced_messages, short=True)
        kwargs["messages"] = enhanced_messages

        # 3. Forward to LLM
        response = self._wrapper._client.chat.completions.create(**kwargs)

        # 4. Parse response and extract memories
        raw_content = response.choices[0].message.content
        parsed = parse_response_with_memory(raw_content)

        # 5. Store extracted memories
        if parsed.memories:
            self._wrapper._store_memories(parsed.memories)

        # 6. Return clean response (modify in place)
        response.choices[0].message.content = parsed.content

        return response


class _FastMemoryAPI:
    """Direct API for memory operations."""

    def __init__(self, store: FastMemoryStore, user_id: str):
        self._store = store
        self._user_id = user_id

    def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryChunk, float]]:
        """Semantic search for memories.

        Args:
            query: Search query
            top_k: Max results

        Returns:
            List of (memory, similarity_score) tuples
        """
        return self._store.search(self._user_id, query, top_k)

    def add(self, content: str, category: str = "fact") -> MemoryChunk:
        """Manually add a memory.

        Args:
            content: Memory content
            category: preference, fact, or context

        Returns:
            The created memory chunk
        """
        return self._store.add(
            self._user_id,
            content,
            role="memory",
            metadata={"category": category, "source": "manual"},
        )

    def get_all(self) -> list[MemoryChunk]:
        """Get all memories for this user."""
        return self._store.get_all(self._user_id)

    def clear(self) -> int:
        """Clear all memories for this user."""
        return self._store.clear(self._user_id)

    def stats(self) -> dict:
        """Get memory statistics."""
        return self._store.stats(self._user_id)


def with_fast_memory(
    client: Any,
    user_id: str,
    db_path: str | Path = "headroom_fast_memory.db",
    top_k: int = 5,
    use_local_embeddings: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
    **kwargs: Any,
) -> FastMemoryWrapper:
    """Wrap an LLM client with zero-latency inline memory extraction.

    This is the fastest memory solution:
    1. ZERO extra LLM calls - memories extracted inline as part of response
    2. Sub-50ms retrieval - local embeddings, no API calls
    3. Semantic search - finds conceptually related memories

    Args:
        client: OpenAI-compatible LLM client
        user_id: User identifier for memory isolation
        db_path: Path to SQLite database
        top_k: Number of memories to inject per request
        use_local_embeddings: Use local model (True) or OpenAI API (False)
        embedding_model: Model name for local embeddings
        **kwargs: Additional arguments

    Returns:
        Wrapped client with automatic memory

    Example:
        from openai import OpenAI
        from headroom.memory import with_fast_memory

        client = with_fast_memory(OpenAI(), user_id="alice")

        # First conversation - memory extracted INLINE
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "I prefer Python for backend work"}]
        )

        # Later - memories automatically retrieved
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What language should I use?"}]
        )
        # User sees: "Based on your preference for Python..."
    """
    return FastMemoryWrapper(
        client=client,
        user_id=user_id,
        db_path=db_path,
        top_k=top_k,
        use_local_embeddings=use_local_embeddings,
        embedding_model=embedding_model,
        **kwargs,
    )
