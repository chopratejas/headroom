# Metrics & Monitoring

Headroom provides comprehensive metrics for monitoring compression performance, cost savings, and system health.

## Proxy Metrics

### Stats Endpoint

```bash
curl http://localhost:8787/stats
```

```json
{
  "requests": {
    "total": 42,
    "cached": 5,
    "rate_limited": 0,
    "failed": 0
  },
  "tokens": {
    "input": 50000,
    "output": 8000,
    "saved": 12500,
    "savings_percent": 25.0
  },
  "cost": {
    "total_cost_usd": 0.15,
    "total_savings_usd": 0.04
  },
  "cache": {
    "entries": 10,
    "total_hits": 5
  }
}
```

### Prometheus Metrics

```bash
curl http://localhost:8787/metrics
```

```prometheus
# HELP headroom_requests_total Total requests processed
headroom_requests_total{mode="optimize"} 1234

# HELP headroom_tokens_saved_total Total tokens saved
headroom_tokens_saved_total 5678900

# HELP headroom_compression_ratio Compression ratio histogram
headroom_compression_ratio_bucket{le="0.5"} 890
headroom_compression_ratio_bucket{le="0.7"} 1100
headroom_compression_ratio_bucket{le="0.9"} 1200

# HELP headroom_latency_seconds Request latency histogram
headroom_latency_seconds_bucket{le="0.01"} 800
headroom_latency_seconds_bucket{le="0.1"} 1150

# HELP headroom_cache_hits_total Cache hit counter
headroom_cache_hits_total 456

# HELP headroom_cache_misses_total Cache miss counter
headroom_cache_misses_total 778
```

### Health Check

```bash
curl http://localhost:8787/health
```

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "llmlingua_enabled": false
}
```

## SDK Metrics

### Session Stats

Quick stats for the current session (no database query):

```python
stats = client.get_stats()
print(stats)
```

```python
{
    "session": {
        "requests_total": 10,
        "tokens_input_before": 50000,
        "tokens_input_after": 35000,
        "tokens_saved_total": 15000,
        "tokens_output_total": 8000,
        "cache_hits": 3,
        "compression_ratio_avg": 0.70
    },
    "config": {
        "mode": "optimize",
        "provider": "openai",
        "cache_optimizer_enabled": True,
        "semantic_cache_enabled": False
    },
    "transforms": {
        "smart_crusher_enabled": True,
        "cache_aligner_enabled": True,
        "rolling_window_enabled": True
    }
}
```

### Historical Metrics

Query stored metrics from the database:

```python
from datetime import datetime, timedelta

# Get recent metrics
metrics = client.get_metrics(
    start_time=datetime.utcnow() - timedelta(hours=1),
    limit=100,
)

for m in metrics:
    print(f"{m.timestamp}: {m.tokens_input_before} -> {m.tokens_input_after}")
```

### Summary Statistics

Aggregate statistics across all stored metrics:

```python
summary = client.get_summary()
print(f"Total requests: {summary['total_requests']}")
print(f"Total tokens saved: {summary['total_tokens_saved']}")
print(f"Average compression: {summary['avg_compression_ratio']:.1%}")
print(f"Total cost savings: ${summary['total_cost_saved_usd']:.2f}")
```

## Logging

### Enable Logging

```python
import logging

# INFO level shows compression summaries
logging.basicConfig(level=logging.INFO)

# DEBUG level shows detailed transform decisions
logging.basicConfig(level=logging.DEBUG)
```

### Log Output Examples

```
INFO:headroom.transforms.pipeline:Pipeline complete: 45000 -> 4500 tokens (saved 40500, 90.0% reduction)
INFO:headroom.transforms.smart_crusher:SmartCrusher applied top_n strategy: kept 15 of 1000 items
INFO:headroom.cache.compression_store:CCR cache hit: hash=abc123, retrieved 1000 items
DEBUG:headroom.transforms.smart_crusher:Kept items: [0,1,2,42,77,97,98,99] (errors at 42, warnings at 77)
```

### Proxy Logging

```bash
# Log to file
headroom proxy --log-file headroom.jsonl

# Increase verbosity
headroom proxy --log-level debug
```

## Grafana Dashboard

Example Grafana dashboard configuration for Prometheus metrics:

```json
{
  "panels": [
    {
      "title": "Tokens Saved",
      "type": "stat",
      "targets": [{"expr": "headroom_tokens_saved_total"}]
    },
    {
      "title": "Compression Ratio",
      "type": "gauge",
      "targets": [{"expr": "histogram_quantile(0.5, headroom_compression_ratio_bucket)"}]
    },
    {
      "title": "Request Latency (p99)",
      "type": "graph",
      "targets": [{"expr": "histogram_quantile(0.99, headroom_latency_seconds_bucket)"}]
    },
    {
      "title": "Cache Hit Rate",
      "type": "gauge",
      "targets": [{"expr": "headroom_cache_hits_total / (headroom_cache_hits_total + headroom_cache_misses_total)"}]
    }
  ]
}
```

## Cost Tracking

### Per-Request Cost

Each request includes cost metadata in the response:

```python
response = client.chat.completions.create(...)

# Access via response metadata (if available)
# Cost is calculated based on model pricing and token counts
```

### Budget Alerts

Set a budget limit in the proxy:

```bash
headroom proxy --budget 10.00
```

When the budget is exceeded:
- Requests return a budget exceeded error
- The `/stats` endpoint shows budget status
- Logs indicate budget state

## Validation

Validate your setup is correct:

```python
result = client.validate_setup()

if result["valid"]:
    print("Setup is correct!")
else:
    print("Issues found:")
    for issue in result["issues"]:
        print(f"  - {issue}")
```

## Key Metrics to Monitor

| Metric | What It Tells You | Target |
|--------|------------------|--------|
| `tokens_saved_total` | Total cost savings | Higher is better |
| `compression_ratio_avg` | Efficiency | 0.7-0.9 typical |
| `cache_hit_rate` | Cache effectiveness | >20% is good |
| `latency_p99` | Performance impact | <10ms |
| `failed_requests` | Reliability | 0 |
