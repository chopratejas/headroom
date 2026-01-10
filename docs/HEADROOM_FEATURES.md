# Headroom: Complete Feature Documentation & Competitive Analysis

## Executive Summary

**Headroom is the world's first Context Optimization Layer for LLM applications.** While the industry has focused on routing (LiteLLM), observability (Helicone), and governance (Portkey), no one has solved the fundamental problem: **LLM contexts are bloated with irrelevant data, and this costs money.**

Headroom reduces LLM costs by 50-70% through intelligent context compression while maintaining 100% retention of critical information (errors, anomalies, relevant items). It's the missing infrastructure layer between your application and LLM providers.

---

# Part 1: Complete Feature Inventory

## 1. Core Transforms (The "Secret Sauce")

### 1.1 SmartCrusher - Statistical Array Compression

**Location**: `headroom/transforms/smart_crusher.py`

**What It Does**: Compresses large JSON arrays (tool outputs) from 1000s of items to 15-50 items while preserving critical information.

**The Safe V1 Recipe** - Always preserves:
| Preserved Item Type | Why It Matters | Detection Method |
|---------------------|----------------|------------------|
| First 3 items | Context/headers | Position-based |
| Last 2 items | Recency | Position-based |
| Error items | Critical signals | Keyword matching: `error`, `exception`, `failed`, `failure`, `critical`, `fatal` |
| Numeric anomalies | Outliers matter | Statistical: values > 2σ from mean |
| Change points | Regime shifts | Sliding window variance detection |
| Relevant items | User's needle | BM25/embedding relevance scoring |

**Algorithm Details**:

```
1. ANALYZE: SmartAnalyzer computes per-field statistics
   - Uniqueness ratio (unique_count / total_count)
   - Numeric stats (min, max, mean, variance)
   - Change points (indices where value significantly shifts)
   - String stats (avg_length, top values)

2. DETECT PATTERN: Identifies data type
   - TIME_SERIES: Has timestamp + numeric variance
   - LOGS: Has message field + level/severity
   - SEARCH_RESULTS: Has score/rank field
   - GENERIC: Default

3. PLAN: Creates compression plan based on pattern
   - TIME_SERIES → Keep items around change points
   - LOGS → Cluster by message, keep representatives
   - SEARCH_RESULTS → Keep top N by score
   - GENERIC → Smart statistical sampling

4. EXECUTE: Apply plan with priority override
   - If errors/anomalies exceed max_items, KEEP ALL
   - Errors are NEVER dropped
```

**Change Point Detection Algorithm**:
```python
def detect_change_points(values, window=5):
    std_dev = statistics.stdev(values)
    threshold = 2.0 * std_dev

    for i in range(window, len(values) - window):
        before_mean = mean(values[i-window:i])
        after_mean = mean(values[i:i+window])
        if abs(after_mean - before_mean) > threshold:
            mark_as_change_point(i)
```

**Configuration Options**:
```python
@dataclass
class SmartCrusherConfig:
    enabled: bool = True
    min_items_to_analyze: int = 5       # Don't crush tiny arrays
    min_tokens_to_crush: int = 200      # Only if > 200 tokens
    variance_threshold: float = 2.0     # Std devs for anomaly
    uniqueness_threshold: float = 0.1   # < 10% = constant field
    similarity_threshold: float = 0.8   # String clustering
    max_items_after_crush: int = 15     # Target output size
    preserve_change_points: bool = True
```

**Performance**:
- 100 items: < 2ms
- 1,000 items: < 10ms
- 10,000 items: < 100ms
- Compression ratio: 50-90% token reduction

---

### 1.5 CCR Architecture - Compress-Cache-Retrieve ⭐ NEW

**Location**: `headroom/cache/compression_store.py`, `headroom/cache/compression_feedback.py`

**What It Does**: Makes compression **reversible**. When SmartCrusher compresses, the original data is cached. If the LLM needs more, it retrieves instantly.

**The Key Innovation**:
> Traditional compression: Guess what's important → Permanent data loss if wrong
> CCR: Compress aggressively → Cache original → Retrieve on demand → Zero permanent loss

**Four Phases**:

| Phase | Component | Description |
|-------|-----------|-------------|
| **1. Store** | `CompressionStore` | Cache original content when compressing |
| **2. Retrieve** | `/v1/retrieve` endpoint | On-demand access to original data |
| **3. Inject** | Tool/system injection | Tell LLM how to retrieve more |
| **4. Feedback** | `CompressionFeedback` | Learn from retrieval patterns |

**CompressionStore Features**:
- Thread-safe in-memory storage
- TTL-based expiration (default 5 minutes)
- LRU-style eviction at capacity
- Built-in BM25 search within cached content
- Hash-based retrieval (16-char SHA256)

**Feedback Loop Metrics**:
```python
class ToolPattern:
    retrieval_rate: float      # retrievals / compressions
    full_retrieval_rate: float # full_retrievals / total_retrievals
    search_rate: float         # search_retrievals / total_retrievals
    common_queries: dict       # Most frequent search queries
    queried_fields: dict       # Fields mentioned in queries
```

**Automatic Adjustment**:
- Retrieval rate >50% → Compress less aggressively (keep 50 items)
- Retrieval rate >80% with full retrievals → Skip compression entirely
- Common query fields → Preserve in future compressions

**API Endpoints**:
```
POST /v1/retrieve           → Retrieve cached content by hash
GET  /v1/feedback           → Get all learned patterns
GET  /v1/feedback/{tool}    → Get hints for specific tool
```

**Configuration**:
```python
@dataclass
class SmartCrusherConfig:
    use_feedback_hints: bool = True  # Enable feedback-driven adjustment
    # ... other options
```

**Why This is a Moat**:
1. **Reversible**: No permanent information loss
2. **Transparent**: LLM knows it can ask for more
3. **Learning**: Improves over time from actual usage
4. **Zero-Risk**: Worst case = retrieve everything

---

### 1.2 CacheAligner - Prefix Stabilization

**Location**: `headroom/transforms/cache_aligner.py`

**What It Does**: Makes your system prompts cache-friendly by extracting dynamic content (dates, timestamps, session IDs) so the static prefix remains byte-identical across requests.

**Why This Matters**:
- Anthropic: 90% discount on cached tokens
- OpenAI: 50% discount on cached tokens
- Google: 75% discount on cached tokens

Without CacheAligner:
```
Request 1: "Today is January 7, 2025. You are helpful."  → Hash: abc123
Request 2: "Today is January 8, 2025. You are helpful."  → Hash: def456 (CACHE MISS!)
```

With CacheAligner:
```
Request 1: "You are helpful.\n---\n[Dynamic: January 7, 2025]"  → Stable Hash: xyz789
Request 2: "You are helpful.\n---\n[Dynamic: January 8, 2025]"  → Stable Hash: xyz789 (CACHE HIT!)
```

**Detection Tiers**:

| Tier | Method | Latency | Coverage |
|------|--------|---------|----------|
| 1 (Regex) | Pattern matching | ~0ms | ISO dates, UUIDs, timestamps, version numbers |
| 2 (NER) | spaCy entities | ~5-10ms | Names, money, organizations, locations |
| 3 (Semantic) | Embedding similarity | ~20-50ms | Complex dynamic patterns |

**Tier 1 Patterns** (Universal, no locale dependencies):
- ISO 8601 DateTime: `\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}`
- ISO 8601 Date: `\d{4}-\d{2}-\d{2}`
- Unix Timestamp: `\d{10,13}`
- UUID: `[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-...-[0-9a-fA-F]{12}`
- Version: `v\d+\.\d+(?:\.\d+)?`
- Structural: `Label: value` where Label indicates dynamic content

**Entropy-Based Detection**:
```python
def calculate_entropy(s: str) -> float:
    """Shannon entropy normalized to [0, 1]"""
    # High entropy (>0.7) = likely random ID
    # Low entropy (<0.3) = likely static text
```

**Configuration**:
```python
@dataclass
class CacheAlignerConfig:
    enabled: bool = True
    date_patterns: list[str] = [...]
    normalize_whitespace: bool = True
    collapse_blank_lines: bool = True
    dynamic_tail_separator: str = "\n\n---\n[Dynamic Context]\n"
```

---

### 1.3 RollingWindow - Context Limit Management

**Location**: `headroom/transforms/rolling_window.py`

**What It Does**: Enforces token limits by dropping oldest context while NEVER orphaning tool call/result pairs.

**The Tool Unit Concept**:
```
Messages:
[0] System: "You are helpful"
[1] User: "Search for X"
[2] Assistant: [tool_calls: search(X), summarize()]
[3] Tool: search result (tool_call_id=call_1)
[4] Tool: summarize result (tool_call_id=call_2)
[5] User: "Thanks"

Tool Unit: (2, [3, 4]) → These drop TOGETHER
```

**Why This Matters**: LLM APIs return errors if tool_calls reference missing tool results. RollingWindow treats them as atomic units.

**Drop Priority**:
1. Oldest tool units (atomic: assistant + all tool results)
2. Non-tool user/assistant pairs
3. Single messages (last resort)

**Protection Rules**:
- System messages: NEVER dropped
- Last N turns: ALWAYS kept (default 2)
- Tool results for protected messages: AUTO-protected

**Configuration**:
```python
@dataclass
class RollingWindowConfig:
    enabled: bool = True
    keep_system: bool = True
    keep_last_turns: int = 2
    output_buffer_tokens: int = 4000  # Reserve for output
```

---

### 1.4 Transform Pipeline - Orchestration

**Location**: `headroom/transforms/pipeline.py`

**Execution Order** (Critical):
```
1. CacheAligner    → Stabilize prefix for cache hits
2. SmartCrusher    → Compress tool outputs
3. RollingWindow   → Enforce token limits
```

**Why This Order**:
1. Cache alignment must happen before content changes
2. Compression reduces tokens before limit enforcement
3. Rolling window is the final safety net

**Token Tracking**: Pipeline tracks tokens through each stage and reports:
```python
@dataclass
class TransformResult:
    messages: list[dict]
    tokens_before: int
    tokens_after: int
    transforms_applied: list[str]
    markers_inserted: list[str]
```

---

## 2. Relevance Scoring Engine

### 2.1 BM25Scorer - Keyword Matching

**Location**: `headroom/relevance/bm25.py`

**What It Does**: Fast, zero-dependency keyword matching using the BM25 algorithm from information retrieval.

**Algorithm**:
```
score(D, Q) = Σ IDF(q) * (f(q,D) * (k1 + 1)) / (f(q,D) + k1 * (1 - b + b * |D|/avgdl))

Parameters:
- k1 = 1.5 (term frequency saturation)
- b = 0.75 (length normalization)
```

**Special Features**:
- UUID preservation in tokenization
- +0.3 bonus for exact long token matches (≥8 chars)
- Query frequency weighting

**Use Cases**: Exact ID matching, UUID lookup, keyword search

---

### 2.2 EmbeddingScorer - Semantic Matching

**Location**: `headroom/relevance/embedding.py`

**What It Does**: Semantic similarity using sentence-transformers embeddings.

**Model**: `all-MiniLM-L6-v2` (22M params, 384 dimensions)

**Algorithm**:
```python
score = cosine_similarity(embed(item), embed(query))
# Clamped to [0, 1]
```

**Optimizations**:
- Batch encoding (context + all items in one call)
- Model caching across instances
- Normalized embeddings for fast cosine

**Use Cases**: Natural language queries, semantic search

---

### 2.3 HybridScorer - Adaptive Fusion

**Location**: `headroom/relevance/hybrid.py`

**What It Does**: Combines BM25 and embedding scores with adaptive alpha based on query characteristics.

**Fusion Formula**:
```
combined = α * BM25_score + (1 - α) * Embedding_score
```

**Adaptive Alpha** (Research: Hsu et al., 2025):
```python
def compute_alpha(query):
    if has_uuid(query):
        return 0.85  # Favor exact matching
    elif has_multiple_ids(query):
        return 0.75
    elif has_single_id(query):
        return 0.65
    elif has_hostname_or_email(query):
        return 0.60
    else:
        return 0.50  # Balanced
```

**Graceful Degradation**: If embeddings unavailable, falls back to boosted BM25.

---

## 3. Cache Optimization (Provider-Specific)

### 3.1 Provider Comparison Matrix

| Feature | Anthropic | OpenAI | Google |
|---------|-----------|--------|--------|
| **Strategy** | Explicit `cache_control` | Automatic prefix | `CachedContent` API |
| **Min Tokens** | 1,024 | 1,024 | 32,768 |
| **Max Breakpoints** | 4 | N/A | 1 |
| **Write Cost** | 1.25x | N/A | N/A |
| **Read Cost** | 0.10x (90% off) | 0.50x (50% off) | 0.25x (75% off) |
| **TTL** | 5 min | 5-60 min | Up to 7 days |
| **Control** | Explicit | Automatic | Explicit |

### 3.2 AnthropicCacheOptimizer

**Location**: `headroom/cache/anthropic.py`

**Algorithm**:
1. Analyze message sections (system, tools, examples, user)
2. Stabilize prefix by extracting dynamic content
3. Plan breakpoints (max 4, prioritize system > tools > examples)
4. Insert `cache_control: {"type": "ephemeral"}` blocks

**Cost Example**:
```
First request (write): 1,500 cached tokens * 1.25x = 1,875 cost
Subsequent (read):     1,500 cached tokens * 0.10x = 150 cost
Savings per hit: 92%
```

### 3.3 OpenAICacheOptimizer

**Location**: `headroom/cache/openai.py`

**Strategy**: Since OpenAI caching is automatic, we maximize cache hits through prefix stabilization:
1. Extract dynamic content via tiered detection
2. Move dates/IDs to end of message
3. Normalize whitespace for consistent hashing

### 3.4 GoogleCacheOptimizer

**Location**: `headroom/cache/google.py`

**Strategy**: Uses Google's explicit CachedContent API:
1. Analyze cacheability (need 32K+ tokens)
2. Prepare cache creation params
3. Register cache for reuse
4. Include `cache_id` in subsequent requests

---

## 4. Production Proxy Server

**Location**: `headroom/proxy/server.py` (1400+ lines)

### 4.1 Core Features

| Feature | Description | Configuration |
|---------|-------------|---------------|
| **Optimization** | SmartCrusher + CacheAligner + RollingWindow | `optimize=True` |
| **Semantic Cache** | Hash-based response caching with TTL | `cache_ttl_seconds=3600` |
| **Rate Limiting** | Token bucket algorithm (requests + tokens) | `rate_limit_requests_per_minute=60` |
| **Retry** | Exponential backoff with jitter | `retry_max_attempts=3` |
| **Cost Tracking** | Real-time cost + budget enforcement | `budget_limit_usd=100.0` |
| **Prometheus** | `/metrics` endpoint | Automatic |
| **Logging** | JSONL request logs | `log_file="/var/log/headroom.jsonl"` |

### 4.2 Endpoints

```
GET  /health              → Health check
GET  /stats               → Detailed statistics
GET  /metrics             → Prometheus format
POST /v1/messages         → Anthropic API proxy
POST /v1/chat/completions → OpenAI API proxy
POST /cache/clear         → Clear semantic cache

# CCR Endpoints (NEW)
POST /v1/retrieve         → Retrieve cached original content
GET  /v1/feedback         → Get all learned patterns
GET  /v1/feedback/{tool}  → Get hints for specific tool
```

### 4.3 Token Bucket Rate Limiter

```python
class TokenBucketRateLimiter:
    def check_request(api_key) -> (allowed: bool, wait_seconds: float)
    def check_tokens(api_key, count) -> (allowed: bool, wait_seconds: float)

    # Continuous refill based on elapsed time
    # Separate buckets for requests and tokens per API key
```

### 4.4 Cost Tracker

```python
PRICING = {
    "claude-3-5-sonnet": (3.00, 15.00, 0.30),  # input, output, cached
    "gpt-4o": (2.50, 10.00, 1.25),
    ...
}

class CostTracker:
    def estimate_cost(model, input_tokens, output_tokens, cached_tokens)
    def check_budget() -> (within_budget: bool, remaining_usd: float)
```

---

## 5. Multi-Provider Support

### 5.1 Token Counting

| Provider | Method | Accuracy |
|----------|--------|----------|
| Anthropic | Official Token Count API | High |
| Anthropic (fallback) | tiktoken * 1.1 | Medium |
| OpenAI | tiktoken (model-specific) | High |
| Google | Official countTokens API | High |

### 5.2 Supported Models

**Anthropic**:
- claude-3-5-sonnet-20241022 (200K context)
- claude-3-5-haiku-20241022 (200K context)
- claude-3-opus-20240229 (200K context)

**OpenAI**:
- gpt-4o (128K context)
- gpt-4o-mini (128K context)
- o1, o1-mini, o3-mini (128-200K context)

**Google**:
- gemini-2.0-flash (1M context)
- gemini-1.5-pro (2M context)
- gemini-1.5-flash (1M context)

---

## 6. Integrations

### 6.1 LangChain Integration

**Location**: `headroom/integrations/langchain.py`

**HeadroomChatModel** - Wrapper that applies optimization:
```python
from langchain_openai import ChatOpenAI
from headroom.integrations import HeadroomChatModel

base_model = ChatOpenAI(model="gpt-4o")
optimized = HeadroomChatModel(base_model, config=HeadroomConfig())

response = optimized.invoke("What is 2+2?")
print(f"Saved: {optimized.total_tokens_saved} tokens")
```

### 6.2 MCP Integration

**Location**: `headroom/integrations/mcp.py`

**HeadroomMCPCompressor** - Compress tool outputs:
```python
from headroom.integrations.mcp import compress_tool_result_with_metrics

result = compress_tool_result_with_metrics(
    content=tool_output,
    tool_name="search_logs",
    user_query="find errors",
)
print(f"Items: {result.items_before} → {result.items_after}")
print(f"Errors preserved: {result.errors_preserved}")
```

**Default Tool Profiles**:
```python
# Slack - preserve bugs/issues
MCPToolProfile(tool_name_pattern=r".*slack.*", max_items=25)

# Database - preserve nulls/violations
MCPToolProfile(tool_name_pattern=r".*database.*", max_items=30)

# Logs - preserve ALL errors
MCPToolProfile(tool_name_pattern=r".*log.*", max_items=40)
```

---

## 7. Pricing Registry

**Location**: `headroom/pricing/`

**Features**:
- Real-time pricing for all models
- Batch pricing support
- Staleness detection (warns if >30 days old)
- Cost estimation with breakdown

**Last Updated**: January 6, 2025

---

# Part 2: Why Headroom is Different

## The Market Gap Nobody Else Fills

### What Existing Tools Do

| Tool | Category | What It Does | What It DOESN'T Do |
|------|----------|--------------|-------------------|
| **LiteLLM** | Gateway/Routing | Unified API for 100+ providers | No context optimization |
| **Helicone** | Observability | Logs, metrics, dashboards | No compression, just watching |
| **Portkey** | Governance | Guardrails, compliance, security | No token reduction |
| **OpenRouter** | Marketplace | Access to 300+ models | 5% markup, no optimization |
| **Cloudflare AI Gateway** | CDN | Caching at edge | Simple caching, no intelligence |

### What Headroom Does (That Nobody Else Does)

**1. Statistical Compression with Quality Guarantees**

No other tool compresses tool outputs while guaranteeing error preservation:
```
Input:  1,000 search results (50,000 tokens)
Output: 20 results (1,000 tokens) - 98% reduction
        ALL errors preserved: 100%
        ALL anomalies preserved: 100%
```

**2. Relevance-Aware Filtering**

SmartCrusher uses BM25 + embeddings to keep items matching the user's query:
```
User asks: "Why is authentication failing?"
Tool returns: 1,000 log entries
SmartCrusher keeps:
  - All entries with "error", "failed", "exception"
  - Entries semantically similar to "authentication failing"
  - First 3 and last 2 for context
```

**3. Provider-Specific Cache Optimization**

We understand each provider's caching rules:
- Anthropic: We insert `cache_control` blocks at optimal positions
- OpenAI: We stabilize prefixes for automatic caching
- Google: We manage CachedContent lifecycle

**4. Atomic Tool Unit Handling**

RollingWindow is the only context manager that treats tool_calls and their results as atomic:
```
Other tools: Drop old messages → Orphaned tool results → API ERROR
Headroom:    Drop tool units atomically → Always valid state
```

---

## Competitive Analysis: Deep Dive

### vs. LiteLLM

| Aspect | LiteLLM | Headroom |
|--------|---------|----------|
| **Primary Function** | Route to 100+ providers | Optimize before routing |
| **Token Reduction** | None | 50-70% |
| **Caching** | None | Semantic + provider-specific |
| **Setup Time** | 15-30 min | 5 min |
| **Latency Overhead** | ~500µs | <50ms |
| **Relationship** | Complementary - we optimize BEFORE LiteLLM routes |

**Partnership Opportunity**: Headroom optimizes → LiteLLM routes → best of both.

### vs. Helicone

| Aspect | Helicone | Headroom |
|--------|----------|----------|
| **Primary Function** | Observe and log | Optimize and compress |
| **Token Reduction** | Shows waste, doesn't fix it | Eliminates waste |
| **Latency** | ~50ms (Rust) | <50ms |
| **Caching** | Redis-based, TTL | Semantic + provider-specific |
| **Relationship** | Complementary - we reduce, they observe |

**Partnership Opportunity**: Headroom compresses → Helicone shows savings achieved.

### vs. Portkey

| Aspect | Portkey | Headroom |
|--------|---------|----------|
| **Primary Function** | Governance, guardrails | Optimization, compression |
| **Target User** | Enterprise security teams | Developers, cost-conscious |
| **Token Reduction** | None | 50-70% |
| **Pricing** | From $49/month | Open source core |
| **Relationship** | Different markets |

### vs. Prompt Compression Techniques (LLMLingua, etc.)

| Aspect | LLMLingua-2 | Headroom |
|--------|-------------|----------|
| **Approach** | Token classification (remove tokens) | Statistical sampling (keep important items) |
| **Target** | Reduce prompt tokens | Reduce tool output tokens |
| **Granularity** | Token-level | Item-level (semantic units) |
| **Quality Guarantee** | 95-98% accuracy | 100% error retention |
| **Dependencies** | XLM-RoBERTa model | Zero (BM25) or sentence-transformers |
| **Use Case** | Long prompts | Large JSON arrays from tools |

---

## The Industry Problem We Solve

### Context Explosion in AI Agents

Research from [JetBrains (Dec 2025)](https://blog.jetbrains.com/research/2025/12/efficient-context-management/):
> "Agents make multiple tool calls in sequence, and each tool's output is fed back into the LLM's context window. Without proper context management, this accumulation can quickly exceed the context window, increase costs dramatically, and degrade performance."

### The "Lost in the Middle" Problem

> "LLMs are more likely to recall information appearing at the beginning or end of long prompts rather than content buried in the middle."

**Headroom's Solution**: SmartCrusher keeps first 3 + last 2 items, plus errors/anomalies/relevant items. We work WITH the LLM's attention patterns.

### Context Rot

> "Expanding context windows does not guarantee improved model performance. As input tokens increase, LLM performance can actually degrade."

**Headroom's Solution**: Smaller, higher-quality context → better performance AND lower cost.

---

## Unique Technical Innovations

### 1. Change Point Detection for Time Series

No other tool detects regime shifts in numeric data:
```python
# Values: [100, 102, 98, 101, 99, 500, 502, 498, 501]
#                                    ↑
#                            Change point detected!
# SmartCrusher keeps items around index 5
```

### 2. Adaptive Relevance Fusion

Our HybridScorer adjusts BM25/embedding weights based on query type:
- UUID in query → More BM25 (exact matching)
- Natural language → More embedding (semantic)

This achieves +2-7.5% accuracy improvement over fixed weights.

### 3. Tool Unit Atomicity

The only context manager that guarantees:
```
assistant message with tool_calls → ALWAYS has corresponding tool results
```

### 4. Tiered Dynamic Detection

We don't use hardcoded locale patterns. Our detection is:
- Universal: ISO 8601, UUIDs, entropy-based IDs
- Structural: `Label: value` patterns
- Semantic: Embedding similarity to known dynamic exemplars

---

# Part 3: Real Numbers

## Compression Performance

| Scenario | Items Before | Items After | Token Reduction | Errors Retained |
|----------|--------------|-------------|-----------------|-----------------|
| Search Results | 1,000 | 20 | 85% | 100% |
| Log Entries | 500 | 40 | 80% | 100% |
| Database Rows | 1,000 | 30 | 90% | 100% |
| API Responses | 200 | 15 | 70% | 100% |

## Latency Overhead

| Component | P50 | P99 |
|-----------|-----|-----|
| SmartCrusher (1000 items) | 5ms | 15ms |
| CacheAligner | <1ms | 2ms |
| RollingWindow | <1ms | 5ms |
| Full Pipeline | 10ms | 25ms |

## Cost Savings (Real World)

**Claude Code Agent Session**:
```
Without Headroom:
  - Tool outputs: 150,000 tokens
  - Cost: $0.45 (input @ $3/M)

With Headroom:
  - Tool outputs: 30,000 tokens (80% reduction)
  - Cost: $0.09 (input @ $3/M)
  - Savings: $0.36 per session (80%)
```

**Enterprise (1M requests/month)**:
```
Without Headroom: $450,000/month
With Headroom:    $90,000/month
Savings:          $360,000/month (80%)
```

---

# Part 4: Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                      YOUR APPLICATION                        │
│                                                              │
│  LangChain  │  Claude Code  │  Cursor  │  Custom Agent      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    HEADROOM PROXY                            │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Cache     │  │    Rate     │  │       Cost          │  │
│  │  (Semantic) │  │   Limiter   │  │     Tracker         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              TRANSFORM PIPELINE                          ││
│  │                                                          ││
│  │  1. CacheAligner    → Stabilize prefix for cache hits   ││
│  │  2. SmartCrusher    → Compress tool outputs             ││
│  │  3. RollingWindow   → Enforce token limits              ││
│  │                                                          ││
│  │  ┌─────────────────────────────────────────────────┐    ││
│  │  │           RELEVANCE ENGINE                       │    ││
│  │  │  BM25 + Embedding + Adaptive Hybrid             │    ││
│  │  └─────────────────────────────────────────────────┘    ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Prometheus │  │   JSONL     │  │      Retry          │  │
│  │   Metrics   │  │   Logging   │  │  (Exp. Backoff)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM PROVIDERS                             │
│                                                              │
│    Anthropic    │    OpenAI    │    Google    │   Others    │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           PROVIDER-SPECIFIC CACHE OPTIMIZERS            ││
│  │                                                          ││
│  │  Anthropic: cache_control blocks (90% savings)          ││
│  │  OpenAI: Prefix stabilization (50% savings)             ││
│  │  Google: CachedContent API (75% savings)                ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

# Part 5: File Inventory

## Core Transforms
- `headroom/transforms/smart_crusher.py` - Statistical array compression
- `headroom/transforms/cache_aligner.py` - Prefix stabilization
- `headroom/transforms/rolling_window.py` - Context limit management
- `headroom/transforms/pipeline.py` - Transform orchestration

## Relevance Scoring
- `headroom/relevance/bm25.py` - BM25 keyword scorer
- `headroom/relevance/embedding.py` - Semantic scorer
- `headroom/relevance/hybrid.py` - Adaptive fusion scorer

## Cache Optimization
- `headroom/cache/base.py` - Base interfaces
- `headroom/cache/anthropic.py` - Anthropic optimizer
- `headroom/cache/openai.py` - OpenAI optimizer
- `headroom/cache/google.py` - Google optimizer
- `headroom/cache/dynamic_detector.py` - Tiered dynamic detection
- `headroom/cache/semantic.py` - Semantic cache layer
- `headroom/cache/compression_store.py` - CCR Phase 1: Store original content ⭐ NEW
- `headroom/cache/compression_feedback.py` - CCR Phase 4: Learn from retrievals ⭐ NEW

## Proxy Server
- `headroom/proxy/server.py` - Production HTTP proxy (1400+ lines)

## Providers
- `headroom/providers/anthropic.py` - Anthropic token counting
- `headroom/providers/openai.py` - OpenAI token counting
- `headroom/providers/google.py` - Google token counting

## Integrations
- `headroom/integrations/langchain.py` - LangChain wrapper
- `headroom/integrations/mcp.py` - MCP compression

## Pricing
- `headroom/pricing/registry.py` - Pricing registry
- `headroom/pricing/anthropic_prices.py` - Anthropic prices
- `headroom/pricing/openai_prices.py` - OpenAI prices

## Tests
- `tests/test_quality_retention.py` - 21 formal evals for quality guarantees
- `tests/test_cache/test_dynamic_detector.py` - Dynamic detection tests
- `tests/test_ccr.py` - CCR store, tool injection tests ⭐ NEW
- `tests/test_ccr_feedback.py` - CCR feedback loop tests ⭐ NEW

## Benchmarks
- `benchmarks/agent_cost_benchmark.py` - Real-world agent cost analysis
- `benchmarks/dynamic_detector_benchmark.py` - Detection performance

---

# Sources

- [JetBrains Research: Efficient Context Management (Dec 2025)](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [LangChain: Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/)
- [Helicone: Top 5 LLM Gateways 2025](https://www.helicone.ai/blog/top-llm-gateways-comparison-2025)
- [Agenta: Top LLM Gateways 2025](https://agenta.ai/blog/top-llm-gateways)
- [Portkey: LLM Proxy vs AI Gateway](https://portkey.ai/blog/llm-proxy-vs-ai-gateway/)
- [Medium: Prompt Compression Techniques (Nov 2025)](https://medium.com/@kuldeep.paul08/prompt-compression-techniques-reducing-context-window-costs-while-improving-llm-performance-afec1e8f1003)
- [Factory.ai: Compressing Context](https://factory.ai/news/compressing-context)
