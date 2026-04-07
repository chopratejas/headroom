# Proxy Server Documentation

The Headroom proxy server is a production-ready HTTP server that applies context optimization to all requests passing through it.

> **New:** The proxy now supports the [TypeScript SDK](typescript-sdk.md) via the `POST /v1/compress` endpoint, enabling compression-as-a-service for any HTTP client without calling an LLM.

## Starting the Proxy

```bash
# Basic usage
headroom proxy

# Custom port
headroom proxy --port 8080

# With all options
headroom proxy \
  --host 0.0.0.0 \
  --port 8787 \
  --log-file /var/log/headroom.jsonl \
  --budget 100.0
```

Telemetry is enabled by default. Opt out with `HEADROOM_TELEMETRY=off` or `headroom proxy --no-telemetry`. Downstream apps can set `HEADROOM_SDK=headroom-app` to override the anonymous telemetry `sdk` label; the default remains `proxy`.

## Command Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8787` | Port to bind to |
| `--mode` | `token` | Run mode: `token` (maximize compression) or `cache` (freeze prior turns) |
| `--no-optimize` | `false` | Disable optimization (passthrough mode) |
| `--no-cache` | `false` | Disable semantic caching |
| `--no-rate-limit` | `false` | Disable rate limiting |
| `--log-file` | None | Path to JSONL log file |
| `--budget` | None | Daily budget limit in USD |
| `--openai-api-url` | `https://api.openai.com` | Custom OpenAI API URL endpoint |

### Run Modes

Headroom proxy has two explicit run modes:

- `token` mode: prioritize token reduction. Prior history may be rewritten when that improves compression.
- `cache` mode: prioritize provider prefix cache stability. Prior turns are frozen; only the newest turn is mutable.

Set via CLI or env:

```bash
headroom proxy --mode token
HEADROOM_MODE=cache headroom proxy
```

When to pick each:

- `token`: best for maximizing immediate compression savings.
- `cache`: best for long conversations where preserving prior-turn bytes improves prefix-cache reuse.

Legacy values (`token_headroom`, `cost_savings`) are still accepted as aliases.

### Context Management Options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-intelligent-context` | `false` | Disable IntelligentContextManager (fall back to RollingWindow) |
| `--no-intelligent-scoring` | `false` | Disable multi-factor importance scoring (use position-based) |
| `--no-compress-first` | `false` | Disable trying deeper compression before dropping messages |

By default, the proxy uses **IntelligentContextManager** which scores messages by multiple factors (recency, semantic similarity, TOIN-learned patterns, error indicators, forward references) and drops lowest-scored messages first. This is smarter than simple age-based truncation.

**CCR Integration:** When messages are dropped, they're stored in CCR so the LLM can retrieve them if needed. The inserted marker includes the CCR reference. Drops are also recorded to TOIN, so the system learns which message patterns are important across all users.

```bash
# Use legacy RollingWindow (drops oldest first)
headroom proxy --no-intelligent-context

# Disable semantic scoring (faster, but less intelligent)
headroom proxy --no-intelligent-scoring
```

### LLMLingua Options (ML Compression)

| Option | Default | Description |
|--------|---------|-------------|
| `--llmlingua` | `false` | Enable LLMLingua-2 ML-based compression |
| `--llmlingua-device` | `auto` | Device for model: `auto`, `cuda`, `cpu`, `mps` |
| `--llmlingua-rate` | `0.3` | Target compression rate (0.3 = keep 30% of tokens) |

**Note:** LLMLingua requires additional dependencies: `pip install headroom-ai[llmlingua]`

```bash
# Enable LLMLingua with GPU acceleration
headroom proxy --llmlingua --llmlingua-device cuda

# More aggressive compression (keep only 20%)
headroom proxy --llmlingua --llmlingua-rate 0.2

# Conservative compression for code (keep 50%)
headroom proxy --llmlingua --llmlingua-rate 0.5
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8787/health
```

Response:
```json
{
  "status": "healthy",
  "optimize": true,
  "stats": {
    "total_requests": 42,
    "tokens_saved": 15000,
    "savings_percent": 45.2
  }
}
```

### Detailed Statistics

```bash
curl http://localhost:8787/stats
```

`/stats` remains the live/session-oriented endpoint and now also includes a
`persistent_savings` block with durable proxy compression lifetime totals plus a
small recent preview. The existing `savings_history` field is still present and
remains session-scoped for backward compatibility.

For providers that return cache-write TTL bucket usage, `/stats` also includes
observed TTL breakdowns under `prefix_cache`:

- `observed_ttl_buckets.5m.tokens`
- `observed_ttl_buckets.1h.tokens`
- `observed_ttl_mix`

These are provider-reported observations, not configured TTL and not remaining
expiration time.

### Historical Savings

```bash
curl http://localhost:8787/stats-history
```

`/stats-history` exposes durable proxy compression history for dashboards and
other Headroom frontends. It returns:

- lifetime proxy compression totals
- bounded persisted checkpoint history
- derived hourly, daily, weekly, and monthly rollups for charts
- UTC timestamps throughout

By default the proxy stores this history at `~/.headroom/proxy_savings.json`.
Set `HEADROOM_SAVINGS_PATH` to override the location.

`/dashboard` uses this endpoint directly for its historical view, including the
daily/weekly/monthly rollups and built-in JSON / CSV export buttons.

```bash
curl "http://localhost:8787/stats-history?format=csv&series=weekly"
curl "http://localhost:8787/stats-history?format=csv&series=monthly"
```

### Prometheus Metrics

```bash
curl http://localhost:8787/metrics
```

### LLM APIs

The proxy supports both Anthropic and OpenAI API formats:

```bash
# Anthropic format
POST /v1/messages

# OpenAI format
POST /v1/chat/completions
```

### `POST /v1/compress`

Compression-only endpoint. Compresses messages without calling any LLM. Used by the [TypeScript SDK](typescript-sdk.md) and any HTTP client that wants compression as a service.

**Request:**
```json
{
  "messages": [...],     // OpenAI chat format
  "model": "gpt-4o"     // model name (for token counting)
}
```

**Response:**
```json
{
  "messages": [...],            // compressed messages
  "tokens_before": 15000,
  "tokens_after": 3500,
  "tokens_saved": 11500,
  "compression_ratio": 0.23,
  "transforms_applied": ["router:smart_crusher:0.35"],
  "ccr_hashes": ["a1b2c3"]
}
```

**Headers:**
- `x-headroom-bypass: true` — skip compression, return messages as-is

**Error responses:** 400 (missing fields), 401 (bad API key), 503 (compression failed)

## Using with Claude Code

```bash
# Start proxy
headroom proxy --port 8787

# In another terminal
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

## Using with Cursor

1. Start the proxy: `headroom proxy`
2. In Cursor settings, set the base URL to `http://localhost:8787`

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8787/v1",
    api_key="your-api-key",  # Still needed for upstream
)
```

## Features

### LLMLingua ML Compression (Opt-In)

When enabled, the proxy uses Microsoft's LLMLingua-2 model for ML-based token compression:

```bash
headroom proxy --llmlingua
```

**How it works:**
- LLMLinguaCompressor is added to the transform pipeline (before RollingWindow)
- Automatically detects content type (JSON, code, text) and adjusts compression
- Stores original content in CCR for retrieval if needed

**Startup feedback:**

```
# When enabled and available:
LLMLingua: ENABLED  (device=cuda, rate=0.3)

# When installed but not enabled (helpful hint):
LLMLingua: available (enable with --llmlingua for ML compression)

# When enabled but not installed:
WARNING: LLMLingua requested but not installed. Install with: pip install headroom-ai[llmlingua]
```

**Why opt-in?**
| Concern | Default Proxy | With LLMLingua |
|---------|---------------|----------------|
| Dependencies | ~50MB | +2GB (torch, transformers) |
| Cold start | <1s | 10-30s (model load) |
| Memory | ~100MB | +1GB (model in RAM) |
| Overhead | <5ms | 50-200ms per request |

Enable LLMLingua when maximum compression justifies the resource cost.

### Semantic Caching

The proxy caches responses for repeated queries:

- LRU eviction with configurable max entries
- TTL-based expiration
- Cache key based on message content hash

### Rate Limiting

Token bucket rate limiting protects against runaway costs:

- Configurable requests per minute
- Configurable tokens per minute
- Per-API-key tracking

### Cost Tracking

Track spending and enforce budgets:

- Real-time cost estimation
- Budget periods: hourly, daily, monthly
- Automatic request rejection when over budget

### Prometheus Metrics

Export metrics for monitoring:

```
headroom_requests_total
headroom_tokens_saved_total
headroom_cost_usd_total
headroom_latency_ms_sum
```

## Configuration via Environment

```bash
export HEADROOM_HOST=0.0.0.0
export HEADROOM_PORT=8787
export HEADROOM_BUDGET=100.0
export OPENAI_TARGET_API_URL=https://custom.openai.endpoint.com
headroom proxy
```

## Running in Production

For production deployments:

```bash
# Use a process manager
pip install gunicorn

# Run with gunicorn
gunicorn headroom.proxy.server:app \
  --workers 4 \
  --bind 0.0.0.0:8787 \
  --worker-class uvicorn.workers.UvicornWorker
```

Or with Docker:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install "headroom-ai[proxy]" \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
EXPOSE 8787
CMD ["headroom", "proxy", "--host", "0.0.0.0"]
```

> **Note:** `build-essential` is required at install time because `headroom-ai` includes `hnswlib`, a C++ extension that must be compiled from source. It is removed after installation to keep the image slim.
