# Headroom

A safe, deterministic Context Budget Controller for LLM APIs.

**Increase effective TPM headroom. Reduce latency. Never break correctness.**

## Features

- **Context MRI (Audit Mode)**: Analyze context waste without modifying requests
- **Tool Output Compression**: Safely compress large tool outputs
- **Cache-Aligned Prefixes**: Optimize for provider caching (OpenAI, etc.)
- **Rolling Window Management**: Keep context within token limits
- **Streaming Support**: Full pass-through streaming with metrics
- **Simulate Mode**: Preview optimizations before applying

## Installation

```bash
pip install headroom
```

Or install from source:

```bash
git clone https://github.com/headroom-sdk/headroom
cd headroom
pip install -e ".[dev]"
```

## Quick Start

```python
from headroom import HeadroomClient
from openai import OpenAI

# Wrap any OpenAI-compatible client
base = OpenAI(api_key="...")
client = HeadroomClient(
    original_client=base,
    store_url="sqlite:///headroom.db",
    default_mode="audit",  # Start in observation mode
)

# Use exactly like the original client
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
print(response.choices[0].message.content)
```

## Modes

### Audit Mode (Default)

Observe and log without making changes:

```python
client = HeadroomClient(
    original_client=base,
    default_mode="audit",
)

# Logs metrics to SQLite but doesn't modify requests
response = client.chat.completions.create(...)
```

### Optimize Mode

Apply safe, deterministic transforms:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    headroom_mode="optimize",  # Enable optimization
)
```

### Simulate Mode

Preview what optimizations would do:

```python
plan = client.chat.completions.simulate(
    model="gpt-4o",
    messages=[...],
)

print(f"Tokens before: {plan.tokens_before}")
print(f"Tokens after: {plan.tokens_after}")
print(f"Tokens saved: {plan.tokens_saved}")
print(f"Transforms: {plan.transforms}")
print(f"Estimated savings: {plan.estimated_savings}")
```

## Configuration

### Headroom Parameters

All headroom parameters are optional:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],

    # Headroom-specific parameters
    headroom_mode="optimize",           # "audit" | "optimize"
    headroom_output_buffer_tokens=4000, # Reserve for output
    headroom_keep_turns=2,              # Never drop last N turns
    headroom_tool_profiles={            # Per-tool compression
        "search": {"max_array_items": 5},
    },

    # All other OpenAI parameters work normally
    temperature=0.7,
    max_tokens=1000,
)
```

### Model Context Limits

Override default context limits:

```python
client = HeadroomClient(
    original_client=base,
    model_context_limits={
        "gpt-4o": 128000,
        "my-custom-model": 32000,
    },
)
```

## Transforms

### 1. Tool Output Compression

Compresses large tool outputs while preserving structure:

- Truncates long arrays (keeps first N items)
- Truncates long strings with markers
- Limits nesting depth
- **Safe**: Malformed JSON is never modified

```python
# Before: 50KB tool response
{"results": [{"id": 1, ...}, {"id": 2, ...}, ... 1000 items ...]}

# After: ~2KB with marker
{"results": [{"id": 1, ...}, ..., {"__headroom_truncated": 995}]}
<headroom:tool_digest sha256="abc123">
```

### 2. Cache Alignment

Stabilizes prefixes for better cache hit rates:

- Extracts dynamic dates from system prompts
- Normalizes whitespace
- Computes stable prefix hash

```python
# Before: Cache miss every day due to date
"You are helpful. Current Date: 2024-01-15"

# After: Stable prefix, date moved to context
"You are helpful.

[Context: Current Date: 2024-01-15]"
```

### 3. Rolling Window

Keeps context within token limits:

- Drops oldest tool call units first
- Never orphans tool responses
- Preserves system prompt and recent turns
- Inserts dropped context markers

## Reporting

Generate HTML reports of context waste:

```python
from headroom import generate_report

generate_report(
    store_url="sqlite:///headroom.db",
    output_path="report.html",
)
```

Reports include:
- Waste histogram by category
- Top high-waste requests
- Cache alignment analysis
- Actionable recommendations

## Safety Guarantees

Headroom follows strict safety rules:

1. **Never removes human content**: User/assistant text is sacred
2. **Never breaks tool ordering**: Tool calls and responses stay paired
3. **Parse failures are no-ops**: Malformed content passes through unchanged
4. **Preserves recency**: Last N turns are always kept

## Streaming

Full streaming support:

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stream=True,
    headroom_mode="optimize",
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

## Storage Options

### SQLite (Default)

```python
client = HeadroomClient(
    original_client=base,
    store_url="sqlite:///headroom.db",
)
```

### JSONL

```python
client = HeadroomClient(
    original_client=base,
    store_url="jsonl:///var/log/headroom.jsonl",
)
```

## Metrics

Access stored metrics programmatically:

```python
# Get recent metrics
metrics = client.get_metrics(limit=100)

# Get summary stats
summary = client.get_summary()
print(f"Total tokens saved: {summary['total_tokens_saved']}")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .

# Type check
mypy headroom
```

## License

MIT

## Contributing

Contributions welcome! Please read the contributing guidelines first.
