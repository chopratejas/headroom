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
├── storage/
│   ├── base.py          # Storage protocol
│   ├── sqlite.py        # SQLite implementation
│   └── jsonl.py         # JSON Lines implementation
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
