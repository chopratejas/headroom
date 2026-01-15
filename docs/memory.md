# Memory

**Persistent memory for LLM applications.** Enable your AI to remember across conversations without carrying full history.

## Why Memory?

LLMs have two fundamental limitations:
1. **Context windows overflow** - Too much history, need to truncate
2. **No persistence** - Every conversation starts from zero

Memory solves both: **extract key facts, persist them, inject when relevant.**

This is *temporal compression* - instead of carrying 10,000 tokens of conversation history, carry 100 tokens of extracted memories.

---

## Quick Start

### Zero-Latency Memory (Recommended)

```python
from openai import OpenAI
from headroom.memory import with_fast_memory

# One line - that's it
client = with_fast_memory(OpenAI(), user_id="alice")

# Use exactly like normal
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "I prefer Python for backend work"}]
)
# Memory extracted INLINE - zero extra latency

# Later, in a new conversation...
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What language should I use?"}]
)
# → Response uses the Python preference from memory
```

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    with_fast_memory()                        │
│                                                              │
│   1. INJECT: Search memories → prepend to user message      │
│   2. INSTRUCT: Add memory extraction instruction            │
│   3. CALL: Forward to LLM                                   │
│   4. PARSE: Extract <memory> block from response            │
│   5. STORE: Save memories with embeddings                   │
│   6. RETURN: Clean response (without memory block)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Memory extraction happens *inline* as part of the LLM response. No extra API calls, no extra latency.

---

## Two Approaches

### 1. Fast Memory (Inline Extraction)

```python
from headroom.memory import with_fast_memory

client = with_fast_memory(
    OpenAI(),
    user_id="alice",
    db_path="memory.db",           # SQLite storage
    top_k=5,                       # Memories to inject
    use_local_embeddings=True,    # Local model (fast) vs OpenAI API
)
```

**Characteristics:**
- Zero extra latency (extraction is part of response)
- ~100 extra output tokens per response
- Smart extraction (LLM decides what's important)
- Semantic retrieval (vector similarity)

### 2. Background Memory (Separate Extraction)

```python
from headroom.memory import with_memory

client = with_memory(
    OpenAI(),
    user_id="alice",
    db_path="memory.db",
)
```

**Characteristics:**
- Non-blocking (extraction happens in background worker)
- Separate LLM call for extraction
- Good when you don't want to modify responses

---

## Memory API

Both wrappers provide a `.memory` API for direct access:

```python
client = with_fast_memory(OpenAI(), user_id="alice")

# Search memories
results = client.memory.search("python preferences", top_k=5)
for memory, score in results:
    print(f"{score:.2f}: {memory.text}")

# Add manual memory
client.memory.add("User is a senior engineer", category="fact")

# Get all memories
all_memories = client.memory.get_all()

# Clear memories
client.memory.clear()

# Get stats
stats = client.memory.stats()
print(f"Total memories: {stats['total_chunks']}")
```

---

## Memory Categories

Memories are categorized for better organization:

| Category | Description | Examples |
|----------|-------------|----------|
| `preference` | Likes, dislikes, preferred approaches | "Prefers Python", "Likes async/await" |
| `fact` | Identity, role, constraints | "Works at fintech startup", "Senior engineer" |
| `context` | Current goals, ongoing tasks | "Migrating to microservices", "Working on auth" |

---

## Configuration

### Storage

```python
# SQLite (default, local)
client = with_fast_memory(OpenAI(), user_id="alice", db_path="memory.db")

# Custom path
client = with_fast_memory(OpenAI(), user_id="alice", db_path="/data/memories.db")
```

### Embeddings

```python
# Local embeddings (recommended - fast, free)
client = with_fast_memory(
    OpenAI(),
    user_id="alice",
    use_local_embeddings=True,
    embedding_model="all-MiniLM-L6-v2",  # 384 dimensions
)

# OpenAI embeddings (higher quality, costs money)
client = with_fast_memory(
    OpenAI(),
    user_id="alice",
    use_local_embeddings=False,  # Uses text-embedding-3-small
)
```

### Retrieval

```python
# Number of memories to inject
client = with_fast_memory(
    OpenAI(),
    user_id="alice",
    top_k=10,  # Inject up to 10 relevant memories
)
```

---

## Multi-User Isolation

Memories are isolated by `user_id`:

```python
# Alice's memories
alice_client = with_fast_memory(OpenAI(), user_id="alice")

# Bob's memories (completely separate)
bob_client = with_fast_memory(OpenAI(), user_id="bob")

# Agent memories
agent_client = with_fast_memory(OpenAI(), user_id="agent-researcher")
```

---

## How Memory Enables Compression

Memory is *temporal compression*. Instead of carrying full conversation history:

```
WITHOUT MEMORY:
Context = Turn 1 + Turn 2 + ... + Turn 50 = 10,000 tokens

WITH MEMORY:
Context = 5 relevant memories = 100 tokens
Compression ratio: 100x
```

This lets you use aggressive rolling window truncation while preserving important facts.

```python
from headroom.memory import with_fast_memory
from headroom.transforms import RollingWindowTransform

# Memory + aggressive truncation = best of both worlds
client = with_fast_memory(OpenAI(), user_id="alice")
transform = RollingWindowTransform(max_tokens=4000)

# Old messages get truncated, but key facts live in memory
messages = transform.apply(very_long_conversation)
response = client.chat.completions.create(model="gpt-4o", messages=messages)
```

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Memory injection | <50ms | Local embeddings + vector search |
| Memory extraction | +50-100ms | Part of LLM response (inline) |
| Memory storage | <10ms | SQLite write + cache update |

**Overhead**: ~100 extra output tokens per response for the `<memory>` block.

---

## Providers

Memory works with any OpenAI-compatible client:

```python
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq

# OpenAI
client = with_fast_memory(OpenAI(), user_id="alice")

# Anthropic (via OpenAI-compatible wrapper)
client = with_fast_memory(OpenAI(base_url="..."), user_id="alice")

# Groq
client = with_fast_memory(Groq(), user_id="alice")

# Any OpenAI-compatible client
client = with_fast_memory(YourClient(), user_id="alice")
```

---

## Example: Multi-Turn Conversation

```python
from openai import OpenAI
from headroom.memory import with_fast_memory

client = with_fast_memory(OpenAI(), user_id="developer_jane")

# Conversation 1: User shares context
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "I'm a Python developer at a fintech startup. We use PostgreSQL."
    }]
)
# Memories extracted: "Python developer", "fintech startup", "uses PostgreSQL"

# Conversation 2 (new session): User asks question
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "What database should I use for my new project?"
    }]
)
# Response references PostgreSQL preference from memory
print(response.choices[0].message.content)
# → "Given your experience with PostgreSQL at your fintech company..."
```

---

## Troubleshooting

### Memories not being extracted

1. Check if the conversation has memory-worthy content (not just greetings)
2. Verify the LLM is following the memory instruction
3. Check logs for parsing errors

### Memories not being retrieved

1. Verify `user_id` matches between sessions
2. Check if memories exist: `client.memory.get_all()`
3. Try a more specific search query

### High latency

1. Switch to local embeddings: `use_local_embeddings=True`
2. Reduce `top_k` for fewer memories to retrieve
3. Check database size and consider pruning old memories

---

## Best Practices

1. **Use consistent `user_id`** - Same ID across sessions for continuity
2. **Start with local embeddings** - Faster, free, good enough for most cases
3. **Combine with rolling window** - Memory + truncation = aggressive compression
4. **Monitor memory growth** - Periodically review and prune if needed
5. **Use categories** - Helps with debugging and selective retrieval
