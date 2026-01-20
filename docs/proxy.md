# Proxy Server Documentation

The Headroom proxy server is a production-ready HTTP server that applies context optimization to all requests passing through it.

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

## Command Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8787` | Port to bind to |
| `--no-optimize` | `false` | Disable optimization (passthrough mode) |
| `--no-cache` | `false` | Disable semantic caching |
| `--no-rate-limit` | `false` | Disable rate limiting |
| `--log-file` | None | Path to JSONL log file |
| `--budget` | None | Daily budget limit in USD |
| `--openai-api-url` | `https://api.openai.com` | Custom OpenAI API URL endpoint |

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
RUN pip install headroom[proxy]
EXPOSE 8787
CMD ["headroom", "proxy", "--host", "0.0.0.0"]
```
