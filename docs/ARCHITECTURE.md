# Headroom SDK: A Complete Explanation

## What Problem Does Headroom Solve?

When you use AI models like GPT-4 or Claude, you pay for **tokens** - the pieces of text you send (input) and receive (output). The problem is:

1. **Tool outputs are HUGE**: When an AI agent calls tools (search, database queries, APIs), the responses are often massive JSON blobs with thousands of tokens
2. **Most of that data is REDUNDANT**: 60 metric data points showing `cpu: 45%` repeated, or 50 log entries with the same error message
3. **You're paying for waste**: Every token costs money and adds latency
4. **Context windows fill up**: Models have limits (128K tokens), and bloated tool outputs eat into your available space

**Headroom creates "headroom"** - it intelligently compresses your input tokens so you have more room (and budget) for what matters.

---

## How Headroom Works: The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR APPLICATION                          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HEADROOM CLIENT                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   ANALYZE   │→ │  TRANSFORM  │→ │    CALL     │             │
│  │  (Parser)   │  │  (Pipeline) │  │   (API)     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│   Count tokens     Apply compressions   Send to OpenAI/Claude  │
│   Detect waste     Preserve meaning     Log metrics            │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OPENAI / ANTHROPIC API                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Core Components (In Simple Terms)

### 1. HeadroomClient (`client.py`) - The Wrapper

This is what you interact with. It wraps your existing OpenAI or Anthropic client:

```python
# Before (normal OpenAI)
client = OpenAI(api_key="...")
response = client.chat.completions.create(model="gpt-4o", messages=[...])

# After (with Headroom)
base = OpenAI(api_key="...")
client = HeadroomClient(original_client=base, provider=OpenAIProvider())
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

**What it does:**
- Intercepts your API calls
- Runs messages through the transform pipeline
- Calls the real API with optimized messages
- Logs metrics to a database
- Returns the response unchanged

**Two modes:**
- `audit`: Just observe and log (no changes)
- `optimize`: Apply transforms to reduce tokens

---

### 2. Providers (`providers/`) - Model-Specific Knowledge

Different AI providers have different rules:

```python
class OpenAIProvider:
    # Knows GPT-4o has 128K context
    # Knows how to count tokens (tiktoken)
    # Knows pricing ($2.50 per million input tokens)

class AnthropicProvider:
    # Knows Claude has 200K context
    # Uses different tokenization (~4 chars per token)
    # Different pricing structure
```

**Why this matters:** Token counting is model-specific. GPT-4 uses different tokenization than Claude. Headroom needs accurate counts to know how much to compress.

---

### 3. Parser (`parser.py`) - Understanding Your Messages

Before optimizing, Headroom needs to understand what's in your messages:

```python
messages = [
    {"role": "system", "content": "You are helpful..."},
    {"role": "user", "content": "Search for X"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "{huge JSON}"},
]

# Parser breaks this into "blocks":
blocks = [
    Block(kind="system", tokens=50, ...),
    Block(kind="user", tokens=10, ...),
    Block(kind="tool_call", tokens=20, ...),
    Block(kind="tool_result", tokens=5000, ...),  # ← This is the problem!
]
```

**It also detects waste signals:**
- Large JSON blobs (>500 tokens)
- HTML tags and comments
- Base64 encoded data
- Excessive whitespace

---

### 4. Transforms (`transforms/`) - The Compression Magic

This is where the real work happens. Headroom has 4 transforms that run in sequence:

#### Transform 1: Cache Aligner

**Problem:** LLM providers cache your prompts, but only if they're byte-identical. If your system prompt has today's date, every day is a cache miss.

```python
# Before:
"You are helpful. Current Date: 2024-12-15"  # Changes daily = no cache

# After:
"You are helpful."  # Static = cacheable
"[Context: Current Date: 2024-12-15]"  # Dynamic part moved to end
```

**How it works:**
1. Find date patterns in system prompt
2. Extract them
3. Move to end of message
4. Now the PREFIX is stable → cache hits!

---

#### Transform 2: Tool Crusher (Naive) - DISABLED BY DEFAULT

This was our first approach - simple but limited:

```python
# Before: 60 items
[{"ts": 1, "cpu": 45}, {"ts": 2, "cpu": 45}, ..., {"ts": 60, "cpu": 95}]

# After: First 10 items only
[{"ts": 1, "cpu": 45}, ..., {"ts": 10, "cpu": 45}, {"__truncated": 50}]
```

**Problem:** If the important data (CPU spike) is at position 45, it gets thrown away!

---

#### Transform 3: Smart Crusher (NEW DEFAULT)

This is the intelligent approach using **statistical analysis**:

```python
# Analyzes the data first:
analysis = {
    "ts": {"type": "sequential", "unique_ratio": 1.0},
    "host": {"type": "constant", "value": "prod-1"},  # ← Same everywhere!
    "cpu": {"variance": 892, "change_points": [45]},  # ← Spike detected!
}

# Smart compression:
{
    "__headroom_constants": {"host": "prod-1"},  # Factor out
    "__headroom_summary": "items 0-44: cpu stable at ~45",  # Summarize boring part
    "data": [
        {"ts": 45, "cpu": 92},  # Keep the spike!
        {"ts": 46, "cpu": 95},
        ...
    ]
}
```

**Strategies it uses:**
1. **TIME_SERIES**: Detect variance spikes, keep change points
2. **CLUSTER**: Group similar log messages, keep 1-2 per cluster
3. **TOP_N**: For search results, keep highest scored
4. **SMART_SAMPLE**: Statistical sampling with constant extraction

---

#### Transform 4: Rolling Window

**Problem:** Even after compression, you might exceed the model's context limit.

```python
# Model limit: 128K tokens
# Your messages: 150K tokens
# Need to drop 22K tokens

# Rolling Window drops OLDEST messages first:
# - Keeps system prompt (always)
# - Keeps last 2 turns (always)
# - Drops old tool calls + their responses as atomic units
```

**Safety rule:** If we drop a tool CALL, we MUST drop its RESPONSE too (or vice versa). Otherwise the model sees orphaned data.

---

### 5. Storage (`storage/`) - Metrics Database

Every request is logged:

```sql
CREATE TABLE requests (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    model TEXT,
    mode TEXT,  -- audit or optimize
    tokens_input_before INTEGER,  -- Before Headroom
    tokens_input_after INTEGER,   -- After Headroom
    tokens_saved INTEGER,         -- The win!
    transforms_applied TEXT,      -- What we did
    ...
);
```

This lets you:
- See how much you're saving
- Generate reports
- Track trends over time

---

## The Data Flow (Step by Step)

Let's trace a real request:

### Step 1: You call the API
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an SRE. Date: 2024-12-15"},
        {"role": "user", "content": "Check the metrics"},
        {"role": "assistant", "tool_calls": [...]},
        {"role": "tool", "content": "{60 metric points...}"},  # 5000 tokens!
        {"role": "user", "content": "What's wrong?"},
    ],
    headroom_mode="optimize",
)
```

### Step 2: HeadroomClient intercepts
```python
# In client.py:
def _create(self, messages, ...):
    # 1. Parse messages into blocks
    blocks, breakdown, waste = parse_messages(messages, tokenizer)
    # breakdown = {"system": 50, "user": 20, "tool_result": 5000, ...}

    # 2. Count original tokens
    tokens_before = 5100
```

### Step 3: Transform Pipeline runs
```python
# In pipeline.py:
def apply(self, messages, ...):
    # Transform 1: Cache Aligner
    # - Extracts "Date: 2024-12-15" from system prompt
    # - Moves to end

    # Transform 2: Smart Crusher
    # - Analyzes 60 metric points
    # - Detects CPU spike at point 45
    # - Compresses to 17 points (preserving spike)
    # - Factors out constant "host" field

    # Transform 3: Rolling Window
    # - Checks if we're under limit (we are)
    # - No drops needed

    return TransformResult(
        messages=optimized,
        tokens_before=5100,
        tokens_after=1200,  # 76% reduction!
        transforms=["cache_align", "smart_crush:1"]
    )
```

### Step 4: Call real API
```python
# In client.py:
response = self._original.chat.completions.create(
    model="gpt-4o",
    messages=optimized_messages,  # Only 1200 tokens now!
)
```

### Step 5: Log metrics and return
```python
# Save to database
metrics = RequestMetrics(
    tokens_input_before=5100,
    tokens_input_after=1200,
    tokens_saved=3900,  # 76%!
    ...
)
storage.save(metrics)

return response  # Unchanged from API
```

---

## The Smart Crusher Deep Dive

This is the most sophisticated part. Here's how it analyzes data:

### Field Analysis
```python
def analyze_field(key, items):
    values = [item[key] for item in items]

    return {
        "unique_ratio": len(set(values)) / len(values),
        # 0.0 = all same (constant)
        # 1.0 = all different (unique IDs)

        "variance": statistics.variance(values),  # For numbers
        # Low = stable
        # High = changing

        "change_points": detect_spikes(values),
        # Indices where value jumps significantly
    }
```

### Pattern Detection
```python
def detect_pattern(field_stats):
    # Has timestamp + numeric variance? → TIME_SERIES
    if has_timestamp and has_numeric_variance:
        return "time_series"

    # Has message field + level field? → LOGS
    if has_message_field and has_level_field:
        return "logs"

    # Has score/rank field? → SEARCH_RESULTS
    if has_score_field:
        return "search_results"

    return "generic"
```

### Compression Strategy
```python
def compress(items, analysis):
    if analysis.pattern == "time_series":
        # Keep points around change points
        # Summarize stable regions
        return time_series_compress(items, analysis.change_points)

    elif analysis.pattern == "logs":
        # Cluster similar messages
        # Keep 1-2 per cluster
        return cluster_compress(items, analysis.clusters)

    elif analysis.pattern == "search_results":
        # Sort by score
        # Keep top N
        return top_n_compress(items, analysis.score_field)
```

---

## CCR Architecture: Compress-Cache-Retrieve

### The Key Insight

> "Prefer raw > Compaction > Summarization only when compaction no longer yields enough space. Compaction (Reversible) strips out information that is redundant because it exists in the environment—if the agent needs to read the data later, it can use a tool to retrieve it." — Phil Schmid, Context Engineering

**The problem with traditional compression:** If we guess wrong about what's important, we've permanently lost data. The LLM might need something we threw away.

**CCR's solution:** Make compression **reversible**. When SmartCrusher compresses, the original data is cached. If the LLM needs more, it can retrieve instantly.

```
┌──────────────────────────────────────────────────────────────────┐
│  TOOL OUTPUT (1000 items)                                         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  HEADROOM CCR LAYER                                               │
│                                                                   │
│  1. COMPRESS: Keep 20 items (errors, anomalies, relevant)        │
│  2. CACHE: Store full 1000 items in fast local cache             │
│  3. INJECT: Add retrieval capability to LLM context              │
│                                                                   │
│  "20 items shown. Use /v1/retrieve?hash=xxx for more."           │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  LLM PROCESSING                                                   │
│                                                                   │
│  Option A: LLM solves task with 20 items → Done                  │
│  Option B: LLM needs more → retrieves via API                    │
│            → We fetch from cache → Return instantly              │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  FEEDBACK LOOP                                                    │
│                                                                   │
│  Track: What did the LLM retrieve? What queries?                 │
│  Learn: "For this tool, keep items matching common queries"      │
│  Improve: Next compression uses learned patterns                 │
└──────────────────────────────────────────────────────────────────┘
```

---

### CCR Phase 1: Compression Store

**Location:** `headroom/cache/compression_store.py`

When SmartCrusher compresses, the original content is stored for on-demand retrieval:

```python
@dataclass
class CompressionEntry:
    hash: str                    # 16-char SHA256 for retrieval
    original_content: str        # Full JSON before compression
    compressed_content: str      # Compressed JSON
    original_item_count: int
    compressed_item_count: int
    tool_name: str | None        # For feedback tracking
    created_at: float
    ttl: int = 300               # 5 minute default
```

**Features:**
- Thread-safe in-memory storage
- TTL-based expiration (default 5 minutes)
- LRU-style eviction when capacity reached
- Built-in BM25 search within cached content

**Usage:**
```python
store = get_compression_store()

# Store compressed content
hash_key = store.store(
    original=original_json,
    compressed=compressed_json,
    original_item_count=1000,
    compressed_item_count=20,
    tool_name="search_api",
)

# Retrieve later
entry = store.retrieve(hash_key)

# Or search within cached content
results = store.search(hash_key, "user query")
```

---

### CCR Phase 2: Retrieval API

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/retrieve` | POST | Retrieve original content by hash |
| `/v1/retrieve?query=X` | POST | Search within cached content |

**Retrieval Request:**
```json
{
    "hash": "abc123def456...",
    "query": "find errors"     // Optional: search within
}
```

**Response (full retrieval):**
```json
{
    "hash": "abc123def456...",
    "original_content": "[{...}, {...}, ...]",
    "original_item_count": 1000,
    "tool_name": "search_api"
}
```

**Response (search):**
```json
{
    "hash": "abc123def456...",
    "query": "find errors",
    "results": [{...}, {...}, ...],
    "count": 15
}
```

---

### CCR Phase 3: Tool Injection

When compression happens, Headroom injects retrieval instructions into the LLM context.

**Method A: System Message Injection**
```
## Compressed Context Available
The following tool outputs have been compressed. If you need more detail,
call the retrieve_compressed tool with the hash.

Available: hash=abc123 (1000→20 items from search_api)
```

**Method B: MCP Tool Registration (Hybrid)**
When running as MCP server, Headroom exposes retrieval as a tool:

```json
{
    "name": "headroom_retrieve",
    "description": "Retrieve more items from compressed tool output",
    "inputSchema": {
        "type": "object",
        "properties": {
            "hash": {"type": "string"},
            "query": {"type": "string"}
        }
    }
}
```

**Marker Injection:**
Compressed content includes retrieval markers:
```json
{
    "__headroom_compressed": true,
    "__headroom_hash": "abc123def456",
    "__headroom_stats": {
        "original_items": 1000,
        "kept_items": 20,
        "errors_preserved": 5
    },
    "data": [...]
}
```

---

### CCR Phase 4: Feedback Loop

**Location:** `headroom/cache/compression_feedback.py`

The feedback system learns from retrieval patterns to improve future compression.

**Tracked Patterns per Tool:**
```python
@dataclass
class ToolPattern:
    tool_name: str
    total_compressions: int      # Times we compressed this tool
    total_retrievals: int        # Times LLM asked for more
    full_retrievals: int         # Retrieved everything
    search_retrievals: int       # Used search query
    common_queries: dict[str, int]   # Query frequency
    queried_fields: dict[str, int]   # Fields mentioned in queries
```

**Key Metrics:**
- **Retrieval Rate**: `total_retrievals / total_compressions`
  - High (>50%) → Compressing too aggressively
  - Low (<20%) → Compression is effective
- **Full Retrieval Rate**: `full_retrievals / total_retrievals`
  - High (>80%) → Data is unique, consider skipping compression

**Compression Hints:**
```python
@dataclass
class CompressionHints:
    max_items: int = 15          # Target item count
    suggested_items: int | None  # Calculated optimal
    skip_compression: bool       # Don't compress at all
    preserve_fields: list[str]   # Always keep these fields
    aggressiveness: float        # 0.0 = aggressive, 1.0 = conservative
    reason: str                  # Explanation
```

**Feedback-Driven Adjustment:**
```python
# In SmartCrusher._crush_array()
if self.config.use_feedback_hints and tool_name:
    feedback = get_compression_feedback()
    hints = feedback.get_compression_hints(tool_name)

    if hints.skip_compression:
        return items, f"skip:feedback({hints.reason})", None

    if hints.suggested_items is not None:
        self.config.max_items_after_crush = hints.suggested_items
```

**Feedback Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/feedback` | GET | Get all learned patterns |
| `/v1/feedback/{tool_name}` | GET | Get hints for specific tool |

**Example Response:**
```json
{
    "total_compressions": 150,
    "total_retrievals": 23,
    "global_retrieval_rate": 0.15,
    "tools_tracked": 5,
    "tool_patterns": {
        "search_api": {
            "compressions": 50,
            "retrievals": 5,
            "retrieval_rate": 0.10,
            "full_rate": 0.20,
            "search_rate": 0.80,
            "common_queries": ["status:error", "level:critical"],
            "queried_fields": ["status", "level", "message"]
        }
    }
}
```

---

### Why CCR is a Moat

1. **Reversible**: No permanent information loss. Worst case = retrieve everything.
2. **Transparent**: LLM knows it can ask for more data.
3. **Feedback Loop**: Learn from actual needs, not guesses.
4. **Network Effect**: Retrieval patterns across users improve compression for everyone.
5. **Zero-Risk**: If compression fails, instant fallback to original data.

---

## File Structure Explained

```
headroom/
├── __init__.py          # Public exports
├── client.py            # HeadroomClient - the main wrapper
├── config.py            # All configuration dataclasses
├── parser.py            # Message → Block decomposition
├── tokenizer.py         # Token counting abstraction
├── utils.py             # Hashing, markers, helpers
│
├── providers/
│   ├── base.py          # Provider/TokenCounter protocols
│   ├── openai.py        # OpenAI-specific (tiktoken)
│   └── anthropic.py     # Anthropic-specific
│
├── transforms/
│   ├── base.py          # Transform protocol
│   ├── pipeline.py      # Orchestrates all transforms
│   ├── cache_aligner.py # Date extraction for caching
│   ├── tool_crusher.py  # Naive compression (disabled)
│   ├── smart_crusher.py # Statistical compression (default)
│   └── rolling_window.py # Token limit enforcement
│
├── cache/               # CCR Architecture
│   ├── compression_store.py    # Phase 1: Store original content
│   ├── compression_feedback.py # Phase 4: Learn from retrievals
│   ├── anthropic.py     # Anthropic cache optimizer
│   ├── openai.py        # OpenAI cache optimizer
│   ├── google.py        # Google cache optimizer
│   └── dynamic_detector.py # Dynamic content detection
│
├── relevance/           # Relevance scoring for compression
│   ├── bm25.py          # BM25 keyword scorer
│   ├── embedding.py     # Semantic embedding scorer
│   └── hybrid.py        # Adaptive fusion scorer
│
├── storage/
│   ├── base.py          # Storage protocol
│   ├── sqlite.py        # SQLite implementation
│   └── jsonl.py         # JSON Lines implementation
│
├── proxy/
│   └── server.py        # Production HTTP proxy (CCR endpoints)
│
└── reporting/
    └── generator.py     # HTML report generation
```

---

## Key Design Decisions

### 1. Provider-Agnostic
Works with ANY OpenAI-compatible API:
- OpenAI
- Azure OpenAI
- Anthropic
- Groq
- Together
- Local models (Ollama)

### 2. Deterministic Transforms
No LLM calls for compression. Everything is:
- Statistical analysis
- Pattern matching
- Rule-based

This means:
- Predictable results
- Fast (<10ms overhead)
- No added API costs

### 3. Safety First
- Never modify user/assistant TEXT content
- Tool call + response are atomic (drop both or neither)
- Parse failures = no-op (return unchanged)
- Audit mode for testing before optimizing

### 4. Smart by Default
- SmartCrusher enabled (statistical analysis)
- ToolCrusher disabled (naive rules)
- Conservative settings that preserve important data

---

## What Makes This Different?

### vs. Summarization (LLM-based compression)
| Headroom | Summarization |
|----------|---------------|
| Deterministic | Non-deterministic |
| ~10ms overhead | ~2-5 seconds overhead |
| No extra API cost | Costs money to summarize |
| Preserves structure | Loses structure |
| Can't hallucinate | Can hallucinate |

### vs. Simple Truncation
| Headroom | Truncation |
|----------|------------|
| Keeps important data | Loses end of data |
| Statistical analysis | No analysis |
| Detects spikes | Misses spikes |
| Factors out constants | Keeps redundancy |

---

## The Numbers (From Our Tests)

Real-world SRE incident investigation:
- **5 tool calls**: Metrics, logs, status, deployments, runbook
- **Original**: 22,048 tokens
- **After SmartCrusher**: 2,190 tokens
- **Reduction**: 90%
- **Quality Score**: 5.0/5 (no information loss)

The model could still:
- Identify the CPU spike (preserved by change point detection)
- Reference specific error rates (kept in compressed data)
- Provide correct remediation commands

---

## Summary

**Headroom is a Context Budget Controller that:**

1. **Wraps** your existing LLM client
2. **Analyzes** your messages to find waste
3. **Compresses** tool outputs intelligently (not blindly)
4. **Preserves** important information (spikes, anomalies, unique data)
5. **Logs** everything for observability
6. **Saves** 70-90% of tokens on tool-heavy workloads

**The key insight:** Most tool output redundancy is **statistical** (repeated values, constant fields, similar messages). By analyzing the data first, we can compress intelligently without losing the information that matters.
