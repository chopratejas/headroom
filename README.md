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

## What It Does

Headroom is a **smart compression proxy** for LLM applications:

- **Compresses tool outputs** — 1000 search results → 15 items (keeps errors, anomalies, relevant items)
- **Enables provider caching** — Stabilizes prefixes so cache hits actually happen
- **Manages context windows** — Prevents token limit failures without breaking tool calls
- **Reversible compression** — LLM can retrieve original data if needed ([CCR architecture](docs/ccr.md))

**Zero code changes required** — point your existing tools at the proxy.

---

## 30-Second Quickstart

```bash
# Install
pip install "headroom-ai[proxy]"

# Start proxy
headroom proxy --port 8787

# Verify
curl http://localhost:8787/health
```

**Use with your tools:**

```bash
# Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude

# Cursor / Continue / any OpenAI client
OPENAI_BASE_URL=http://localhost:8787/v1 cursor

# Python scripts
export OPENAI_BASE_URL=http://localhost:8787/v1
python your_script.py
```

That's it. You're saving tokens.

---

## Verify It's Working

```bash
curl http://localhost:8787/stats
```

```json
{
  "tokens": {"saved": 12500, "savings_percent": 25.0},
  "cost": {"total_savings_usd": 0.04}
}
```

---

## Installation

```bash
pip install "headroom-ai[proxy]"     # Proxy server (recommended)
pip install headroom-ai              # SDK only
pip install "headroom-ai[all]"       # Everything
```

**Requirements**: Python 3.10+

---

## Features

| Feature | Description | Docs |
|---------|-------------|------|
| **SmartCrusher** | Compresses JSON tool outputs statistically | [Transforms](docs/transforms.md) |
| **CacheAligner** | Stabilizes prefixes for provider caching | [Transforms](docs/transforms.md) |
| **RollingWindow** | Manages context limits without breaking tools | [Transforms](docs/transforms.md) |
| **CCR** | Reversible compression with automatic retrieval | [CCR Guide](docs/ccr.md) |
| **Text Utilities** | Opt-in compression for search/logs | [Text Compression](docs/text-compression.md) |
| **LLMLingua-2** | ML-based 20x compression (opt-in) | [LLMLingua](docs/llmlingua.md) |

---

## Providers

| Provider | Token Counting | Cache Optimization |
|----------|----------------|-------------------|
| OpenAI | tiktoken (exact) | Automatic prefix caching |
| Anthropic | Official API | cache_control blocks |
| Google | Official API | Context caching |
| Cohere | Official API | - |
| Mistral | Official tokenizer | - |

---

## Performance

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| Search results (1000 items) | 45,000 tokens | 4,500 tokens | 90% |
| Log analysis (500 entries) | 22,000 tokens | 3,300 tokens | 85% |
| Long conversation (50 turns) | 80,000 tokens | 32,000 tokens | 60% |

Overhead: ~1-5ms per request.

---

## Safety

- **Never removes human content** — User/assistant messages are never compressed
- **Never breaks tool ordering** — Tool calls and responses stay paired
- **Parse failures are no-ops** — Malformed content passes through unchanged
- **Compression is reversible** — LLM can retrieve original data via CCR

---

## Documentation

| Guide | Description |
|-------|-------------|
| [SDK Guide](docs/sdk.md) | Wrap your client for fine-grained control |
| [Proxy Guide](docs/proxy.md) | Production deployment |
| [Configuration](docs/configuration.md) | All configuration options |
| [CCR Guide](docs/ccr.md) | Reversible compression architecture |
| [Metrics](docs/metrics.md) | Monitoring and observability |
| [Troubleshooting](docs/troubleshooting.md) | Common issues |
| [Architecture](docs/ARCHITECTURE.md) | How it works internally |

---

## Examples

See [`examples/`](examples/) for runnable code:

- `basic_usage.py` — Simple SDK usage
- `proxy_integration.py` — Using with different clients
- `ccr_demo.py` — CCR architecture demonstration

---

## Contributing

```bash
git clone https://github.com/chopratejas/headroom.git
cd headroom
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Built for the AI developer community</sub>
</p>
