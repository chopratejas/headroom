<p align="center">
  <h1 align="center">Headroom</h1>
  <p align="center">
    <strong>The Context Optimization Layer for LLM Applications</strong>
  </p>
  <p align="center">
    Cut your LLM costs by 50-90% without losing accuracy
  </p>
</p>

<p align="center">
  <a href="https://github.com/chopratejas/headroom/actions/workflows/ci.yml">
    <img src="https://github.com/chopratejas/headroom/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/headroom-ai/">
    <img src="https://img.shields.io/pypi/v/headroom-ai.svg" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/headroom-ai/">
    <img src="https://img.shields.io/pypi/pyversions/headroom-ai.svg" alt="Python">
  </a>
  <a href="https://github.com/chopratejas/headroom/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

---

## Why Headroom?

AI coding agents and tool-using applications generate **massive contexts**:

- Tool outputs with 1000s of search results, log entries, API responses
- Long conversation histories that hit token limits
- System prompts with dynamic dates that break provider caching

**Result**: You pay for tokens you don't need, and cache hits are rare.

Headroom is a **smart compression layer** that sits between your app and LLM providers:

| Transform | What It Does | Savings |
|-----------|--------------|---------|
| **SmartCrusher** | Compresses tool outputs statistically (keeps errors, anomalies, relevant items) | 70-90% |
| **CacheAligner** | Stabilizes prefixes so provider caching works | Up to 10x |
| **RollingWindow** | Manages context within limits without breaking tool calls | Prevents failures |

**Zero accuracy loss** - we keep what matters: errors, anomalies, relevant items.

---

## 5-Minute Quickstart

### Option 1: Proxy Server (Recommended)

Works with **any** OpenAI-compatible client without code changes:

```bash
# Install
pip install "headroom-ai[proxy]"

# Start the proxy
headroom proxy --port 8787

# Verify it's running
curl http://localhost:8787/health
# Expected: {"status": "healthy", ...}
```

**Use with your tools:**

```bash
# Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude

# Cursor / Continue / any OpenAI client
OPENAI_BASE_URL=http://localhost:8787/v1 your-app

# Python OpenAI SDK
export OPENAI_BASE_URL=http://localhost:8787/v1
python your_script.py
```

### Option 2: Python SDK

Wrap your existing client for fine-grained control:

```bash
pip install headroom-ai openai
```

```python
from headroom import HeadroomClient, OpenAIProvider
from openai import OpenAI

# Create wrapped client
client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    default_mode="optimize",  # or "audit" to observe only
)

# Use exactly like the original client
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello!"},
    ],
)

print(response.choices[0].message.content)

# Check what happened
stats = client.get_stats()
print(f"Tokens saved this session: {stats['session']['tokens_saved_total']}")
```

**With tool outputs (where real savings happen):**

```python
import json

# Conversation with large tool output
messages = [
    {"role": "user", "content": "Search for Python tutorials"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "python"}'},
        }],
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": json.dumps({
            "results": [{"title": f"Tutorial {i}", "score": 100-i} for i in range(500)]
        }),
    },
    {"role": "user", "content": "What are the top 3?"},
]

# Headroom compresses 500 results to ~15, keeping highest-scoring items
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(f"Tokens saved: {client.get_stats()['session']['tokens_saved_total']}")
# Typical output: "Tokens saved: 3500"
```

### Option 3: LangChain Integration (Coming Soon)

```python
# Coming soon - use proxy server for now
# OPENAI_BASE_URL=http://localhost:8787/v1 python your_langchain_app.py
```

---

## Verify It's Working

### Check Proxy Stats

```bash
curl http://localhost:8787/stats
```

```json
{
  "requests": {"total": 42, "cached": 5, "rate_limited": 0, "failed": 0},
  "tokens": {"input": 50000, "output": 8000, "saved": 12500, "savings_percent": 25.0},
  "cost": {"total_cost_usd": 0.15, "total_savings_usd": 0.04},
  "cache": {"entries": 10, "total_hits": 5}
}
```

### Check SDK Stats

```python
# Quick session stats (no database query)
stats = client.get_stats()
print(stats)
# {
#   "session": {"requests_total": 10, "tokens_saved_total": 5000, ...},
#   "config": {"mode": "optimize", "provider": "openai", ...},
#   "transforms": {"smart_crusher_enabled": True, ...}
# }

# Validate setup is correct
result = client.validate_setup()
if not result["valid"]:
    print("Setup issues:", result)
```

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now you'll see:
# INFO:headroom.transforms.pipeline:Pipeline complete: 45000 -> 4500 tokens (saved 40500, 90.0% reduction)
# INFO:headroom.transforms.smart_crusher:SmartCrusher applied top_n strategy: kept 15 of 1000 items
```

---

## Installation

```bash
# Core only (minimal dependencies: tiktoken, pydantic)
pip install headroom-ai

# With semantic relevance scoring (adds sentence-transformers)
pip install "headroom-ai[relevance]"

# With proxy server (adds fastapi, uvicorn)
pip install "headroom-ai[proxy]"

# With HTML reports (adds jinja2)
pip install "headroom-ai[reports]"

# Everything
pip install "headroom-ai[all]"
```

**Requirements**: Python 3.10+

---

## Configuration

### SDK Configuration

```python
from headroom import HeadroomClient, OpenAIProvider
from openai import OpenAI

# Full configuration example
client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    default_mode="optimize",              # "audit" (observe only) or "optimize" (apply transforms)
    enable_cache_optimizer=True,          # Enable provider-specific cache optimization
    enable_semantic_cache=False,          # Enable query-level semantic caching
    model_context_limits={                # Override default context limits
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
    },
    # store_url defaults to temp directory; override with absolute path if needed:
    # store_url="sqlite:////absolute/path/to/headroom.db",
)
```

### Proxy Configuration

```bash
# Via command line
headroom proxy \
  --port 8787 \
  --budget 10.00 \
  --log-file headroom.jsonl

# Disable optimization (passthrough mode)
headroom proxy --no-optimize

# Disable semantic caching
headroom proxy --no-cache

# See all options
headroom proxy --help
```

### Per-Request Overrides

```python
# Override mode for specific requests
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    headroom_mode="audit",              # Just observe, don't optimize
    headroom_output_buffer_tokens=8000, # Reserve more for output
    headroom_keep_turns=5,              # Keep last 5 turns
)
```

---

## Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `audit` | Observes and logs, no modifications | Production monitoring, baseline measurement |
| `optimize` | Applies safe, deterministic transforms | Production optimization |
| `simulate` | Returns plan without API call | Testing, cost estimation |

```python
# Simulate to see what would happen
plan = client.chat.completions.simulate(
    model="gpt-4o",
    messages=large_conversation,
)
print(f"Would save {plan.tokens_saved} tokens")
print(f"Transforms: {plan.transforms}")
print(f"Estimated savings: {plan.estimated_savings}")
```

---

## Error Handling

Headroom provides explicit exceptions for debugging:

```python
from headroom import (
    HeadroomClient,
    HeadroomError,        # Base class - catch all Headroom errors
    ConfigurationError,   # Invalid configuration
    ProviderError,        # Provider issues (unknown model, etc.)
    StorageError,         # Database/storage failures
    CompressionError,     # Compression failures (rare - we fail safe)
    ValidationError,      # Setup validation failures
)

try:
    client = HeadroomClient(...)
    response = client.chat.completions.create(...)
except ConfigurationError as e:
    print(f"Config issue: {e}")
    print(f"Details: {e.details}")  # Additional context
except StorageError as e:
    print(f"Storage issue: {e}")
    # Headroom continues to work, just without metrics persistence
except HeadroomError as e:
    print(f"Headroom error: {e}")
```

**Safety guarantee**: If compression fails, the original content passes through unchanged. Your LLM calls never fail due to Headroom.

---

## How It Works

### SmartCrusher: Statistical Compression

```python
# Before: 50KB tool response with 1000 items
{"results": [{"id": 1, "status": "ok", ...}, ... 1000 items ...]}

# After: ~2KB with important items preserved
# Headroom keeps:
# - First 3 items (context)
# - Last 2 items (recency)
# - All error items (status != "ok")
# - Statistical anomalies (values > 2 std dev from mean)
# - Items matching user's query (BM25/embedding similarity)
```

### CacheAligner: Prefix Stabilization

```python
# Before: Cache miss every day due to changing date
"You are helpful. Today is January 7, 2025."

# After: Stable prefix (cache hit!) + dynamic context moved to end
"You are helpful."
# Dynamic content: "Current date: January 7, 2025"
```

### RollingWindow: Context Management

```python
# When context exceeds limit:
# 1. Drop oldest tool outputs first (as atomic units with their calls)
# 2. Drop oldest conversation turns
# 3. NEVER drop: system prompt, last N turns, orphaned tool responses
```

---

## Metrics & Monitoring

### Prometheus Metrics (Proxy)

```bash
curl http://localhost:8787/metrics
```

```
# HELP headroom_requests_total Total requests processed
headroom_requests_total{mode="optimize"} 1234

# HELP headroom_tokens_saved_total Total tokens saved
headroom_tokens_saved_total 5678900

# HELP headroom_compression_ratio Compression ratio histogram
headroom_compression_ratio_bucket{le="0.5"} 890
```

### Query Stored Metrics (SDK)

```python
from datetime import datetime, timedelta

# Get recent metrics
metrics = client.get_metrics(
    start_time=datetime.utcnow() - timedelta(hours=1),
    limit=100,
)

for m in metrics:
    print(f"{m.timestamp}: {m.tokens_input_before} -> {m.tokens_input_after}")

# Get summary statistics
summary = client.get_summary()
print(f"Total requests: {summary['total_requests']}")
print(f"Total tokens saved: {summary['total_tokens_saved']}")
```

---

## Troubleshooting

### "Proxy won't start"

```bash
# Check if port is in use
lsof -i :8787

# Try a different port
headroom proxy --port 8788

# Check logs
headroom proxy --log-level debug
```

### "No token savings"

```python
# 1. Verify mode is "optimize"
stats = client.get_stats()
print(stats["config"]["mode"])  # Should be "optimize"

# 2. Check if transforms are enabled
print(stats["transforms"])  # smart_crusher_enabled should be True

# 3. Enable logging to see what's happening
import logging
logging.basicConfig(level=logging.DEBUG)

# 4. Use simulate to see what WOULD happen
plan = client.chat.completions.simulate(model="gpt-4o", messages=msgs)
print(f"Transforms that would apply: {plan.transforms}")
```

### "High latency"

```python
# Headroom adds ~1-5ms overhead. If you see more:

# 1. Check if embedding scorer is enabled (slower but better relevance)
# Switch to BM25 for faster scoring:
config.smart_crusher.relevance.tier = "bm25"

# 2. Disable transforms you don't need
config.cache_aligner.enabled = False  # If you don't need cache alignment

# 3. Increase min_tokens_to_crush to skip small payloads
config.smart_crusher.min_tokens_to_crush = 500
```

### "Compression too aggressive"

```python
# Keep more items
config.smart_crusher.max_items_after_crush = 50  # Default is 15

# Or disable compression for specific tools
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    headroom_tool_profiles={
        "important_tool": {"skip_compression": True}
    }
)
```

---

## Supported Providers

| Provider | Token Counting | Cache Optimization | Status |
|----------|----------------|-------------------|--------|
| OpenAI | tiktoken (exact) | Automatic prefix caching | Full |
| Anthropic | Official API | cache_control blocks | Full |
| Google | Official API | Context caching | Full |
| Cohere | Official API | - | Full |
| Mistral | Official tokenizer | - | Full |
| LiteLLM | Via underlying provider | - | Full |

---

## Safety Guarantees

Headroom follows strict safety rules:

1. **Never removes human content** - User/assistant messages are never compressed
2. **Never breaks tool ordering** - Tool calls and responses stay paired as atomic units
3. **Parse failures are no-ops** - Malformed content passes through unchanged
4. **Preserves recency** - Last N turns are always kept
5. **Errors surface, don't hide** - Explicit exceptions with context

---

## Performance

| Scenario | Before | After | Savings | Overhead |
|----------|--------|-------|---------|----------|
| Search results (1000 items) | 45,000 tokens | 4,500 tokens | 90% | ~2ms |
| Log analysis (500 entries) | 22,000 tokens | 3,300 tokens | 85% | ~1ms |
| API response (nested JSON) | 15,000 tokens | 2,250 tokens | 85% | ~1ms |
| Long conversation (50 turns) | 80,000 tokens | 32,000 tokens | 60% | ~3ms |

---

## Documentation

- **[Quickstart Guide](docs/quickstart.md)** - Complete working examples
- **[Proxy Documentation](docs/proxy.md)** - Production deployment
- **[Transform Reference](docs/transforms.md)** - How each transform works
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Architecture](docs/ARCHITECTURE.md)** - How Headroom works internally

---

## Examples

See the [`examples/`](examples/) directory for complete, runnable examples:

- `basic_usage.py` - Simple SDK usage
- `proxy_integration.py` - Using the proxy with different clients
- `custom_compression.py` - Advanced compression configuration
- `metrics_dashboard.py` - Building a metrics dashboard

---

## Contributing

We welcome contributions!

```bash
# Development setup
git clone https://github.com/chopratejas/headroom.git
cd headroom
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
mypy headroom
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for the AI developer community</sub>
</p>
