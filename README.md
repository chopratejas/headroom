<p align="center">
  <h1 align="center">Headroom</h1>
  <p align="center">
    <strong>The Context Optimization Layer for LLM Applications</strong>
  </p>
  <p align="center">
    Tool outputs are 70-95% redundant boilerplate. Headroom compresses that away.
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
  <a href="https://pypistats.org/packages/headroom-ai">
    <img src="https://img.shields.io/pypi/dm/headroom-ai.svg" alt="Downloads">
  </a>
  <a href="https://github.com/chopratejas/headroom/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://chopratejas.github.io/headroom/">
    <img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="Documentation">
  </a>
</p>

---

## Demo

<p align="center">
  <img src="Headroom-2.gif" alt="Headroom Demo" width="800">
</p>

---

## Quick Start

```bash
pip install "headroom-ai[all]"
```

```python
from headroom import compress

messages = [
    {"role": "user", "content": "What caused the outage?"},
    {"role": "tool", "content": huge_log_output, "tool_call_id": "call_1"},
]

result = compress(messages, model="claude-sonnet-4-5-20250929")
# result.messages → same format, 50-90% fewer tokens
# result.tokens_saved → 8,000
# result.compression_ratio → 0.87

response = client.messages.create(model="claude-sonnet-4-5-20250929", messages=result.messages)
```

**Same answer. 87% fewer tokens.**

---

## How to Use Headroom

Headroom is a compression library, not just a proxy. Use whichever integration fits your stack:

| You have... | Use this | Code |
|-------------|----------|------|
| Any Python app | `compress()` | `result = compress(messages, model="gpt-4o")` |
| LiteLLM | Callback | `litellm.callbacks = [HeadroomCallback()]` |
| Python proxy (FastAPI) | ASGI Middleware | `app.add_middleware(CompressionMiddleware)` |
| Claude Code / Cursor | Proxy | `ANTHROPIC_BASE_URL=http://localhost:8787 claude` |
| Agno agents | Wrap model | `HeadroomAgnoModel(your_model)` |
| LangChain | Wrap model | `HeadroomChatModel(your_llm)` *(experimental)* |

**Already have a proxy?** You don't need another one. See the **[Integration Guide](docs/integration-guide.md)** for detailed setup with LiteLLM, ASGI middleware, and direct `compress()` usage.

---

## How It Works

```
Your App → Headroom → LLM Provider
              ↓
   CacheAligner: stabilizes prefix for KV cache hits
   ContentRouter: routes to optimal compressor per content type
     → SmartCrusher (JSON) | CodeCompressor (code) | LLMLingua (text)
   IntelligentContext: score-based token fitting
   CCR: stores originals for retrieval if LLM needs more
```

Headroom never throws data away. It compresses aggressively and retrieves precisely.

---

## Verified Performance

| Scenario | Tokens Before | Tokens After | Savings |
|----------|--------------|-------------|---------|
| Code search (100 results) | 17,765 | 1,408 | **92%** |
| SRE incident debugging | 65,694 | 5,118 | **92%** |
| Codebase exploration | 78,502 | 41,254 | **47%** |
| GitHub issue triage | 54,174 | 14,761 | **73%** |

**Overhead**: 1-5ms. **Accuracy**: [benchmarked](docs/benchmarks.md) across 12+ datasets.

---

## Integrations

| Integration | Status | Docs |
|-------------|--------|------|
| `compress()` — one function | **Stable** | [Integration Guide](docs/integration-guide.md) |
| LiteLLM callback | **Stable** | [Integration Guide](docs/integration-guide.md#litellm) |
| ASGI middleware | **Stable** | [Integration Guide](docs/integration-guide.md#asgi-middleware) |
| Proxy server | **Stable** | [Proxy Docs](docs/proxy.md) |
| Agno | **Stable** | [Agno Guide](docs/agno.md) |
| MCP (Claude Code) | **Stable** | [MCP Guide](docs/mcp.md) |
| Strands | **Stable** | [Strands Guide](docs/strands.md) |
| LangChain | **Experimental** | [LangChain Guide](docs/langchain.md) |

---

## Features

| Feature | What it does |
|---------|-------------|
| **Content Router** | Auto-detects content type, routes to optimal compressor |
| **SmartCrusher** | Statistically compresses JSON arrays (tool outputs, API responses) |
| **CodeCompressor** | AST-aware code compression (Python, JS, Go, Rust, Java) |
| **LLMLingua-2** | ML-based 20x text compression |
| **CCR** | Reversible compression — LLM retrieves originals when needed |
| **CacheAligner** | Stabilizes prefixes for provider KV cache hits |
| **IntelligentContext** | Score-based context management with learned importance |
| **Image Compression** | 40-90% token reduction via trained ML router |
| **Memory** | Persistent memory across conversations |
| **Compression Hooks** | Customize compression with pre/post hooks |
| **Query Echo** | Re-injects user question after compressed data for better attention |

---

## Cloud Providers

```bash
headroom proxy --backend bedrock --region us-east-1     # AWS Bedrock
headroom proxy --backend vertex_ai --region us-central1 # Google Vertex
headroom proxy --backend azure                          # Azure OpenAI
headroom proxy --backend openrouter                     # OpenRouter (400+ models)
```

---

## Installation

```bash
pip install headroom-ai                # Core library
pip install "headroom-ai[all]"         # Everything (recommended)
pip install "headroom-ai[proxy]"       # Proxy server
pip install "headroom-ai[mcp]"         # MCP for Claude Code
pip install "headroom-ai[agno]"        # Agno integration
pip install "headroom-ai[langchain]"   # LangChain (experimental)
pip install "headroom-ai[evals]"       # Evaluation framework
```

Python 3.10+

---

## Documentation

| | |
|---|---|
| [Integration Guide](docs/integration-guide.md) | LiteLLM, ASGI, compress(), proxy |
| [Proxy Docs](docs/proxy.md) | Proxy server configuration |
| [Architecture](docs/ARCHITECTURE.md) | How the pipeline works |
| [CCR Guide](docs/ccr.md) | Reversible compression |
| [Benchmarks](docs/benchmarks.md) | Accuracy validation |
| [Evals Framework](headroom/evals/README.md) | Prove compression preserves accuracy |
| [Memory](docs/memory.md) | Persistent memory |
| [Agno](docs/agno.md) | Agno agent framework |
| [MCP](docs/mcp.md) | Claude Code subscriptions |
| [Configuration](docs/configuration.md) | All options |

---

## Contributing

```bash
git clone https://github.com/chopratejas/headroom.git && cd headroom
pip install -e ".[dev]" && pytest
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
