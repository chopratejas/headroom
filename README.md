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

- **Zero code changes** - works as a transparent proxy
- **50-90% cost savings** - verified on real workloads
- **Reversible compression** - LLM retrieves original data via CCR
- **Content-aware** - code, logs, JSON each handled optimally
- **Provider caching** - automatic prefix optimization for cache hits
- **Persistent memory** - remember across conversations with zero-latency extraction
- **Framework native** - LangChain, Agno, MCP, agents supported

---

## Headroom vs Alternatives

| Approach | Token Reduction | Accuracy | Reversible | Latency |
|----------|-----------------|----------|------------|---------|
| **Headroom** | 50-90% | No loss | Yes (CCR) | ~1-5ms |
| Truncation | Variable | Data loss | No | ~0ms |
| Summarization | 60-80% | Lossy | No | ~500ms+ |
| No optimization | 0% | Full | N/A | 0ms |

**Headroom wins** because it intelligently selects relevant content while keeping a retrieval path to the original data.

---

## 30-Second Quickstart

### Option 1: Proxy (Zero Code Changes)

```bash
pip install "headroom-ai[proxy]"
headroom proxy --port 8787
```

Point your tools at the proxy:

```bash
# Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude

# Any OpenAI-compatible client
OPENAI_BASE_URL=http://localhost:8787/v1 cursor
```

### Option 2: LangChain Integration

```bash
pip install "headroom-ai[langchain]"
```

```python
from langchain_openai import ChatOpenAI
from headroom.integrations import HeadroomChatModel

# Wrap your model - that's it!
llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))

# Use exactly like before
response = llm.invoke("Hello!")
```

See the full [LangChain Integration Guide](docs/langchain.md) for memory, retrievers, agents, and more.

### Option 3: Agno Integration

```bash
pip install "headroom-ai[agno]"
```

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from headroom.integrations.agno import HeadroomAgnoModel

# Wrap your model - that's it!
model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))
agent = Agent(model=model)

# Use exactly like before
response = agent.run("Hello!")

# Check savings
print(f"Tokens saved: {model.total_tokens_saved}")
```

See the full [Agno Integration Guide](docs/agno.md) for hooks, multi-provider support, and more.

---

## Framework Integrations

| Framework | Integration | Docs |
|-----------|-------------|------|
| **LangChain** | `HeadroomChatModel`, memory, retrievers, agents | [Guide](docs/langchain.md) |
| **Agno** | `HeadroomAgnoModel`, hooks, multi-provider | [Guide](docs/agno.md) |
| **MCP** | Tool output compression for Claude | [Guide](docs/ccr.md) |
| **Any OpenAI Client** | Proxy server | [Guide](docs/proxy.md) |

---

## Features

| Feature | Description | Docs |
|---------|-------------|------|
| **Memory** | Persistent memory across conversations (zero-latency inline extraction) | [Memory](docs/memory.md) |
| **Universal Compression** | ML-based content detection + structure-preserving compression | [Compression](docs/compression.md) |
| **SmartCrusher** | Compresses JSON tool outputs statistically | [Transforms](docs/transforms.md) |
| **CacheAligner** | Stabilizes prefixes for provider caching | [Transforms](docs/transforms.md) |
| **RollingWindow** | Manages context limits without breaking tools | [Transforms](docs/transforms.md) |
| **CCR** | Reversible compression with automatic retrieval | [CCR Guide](docs/ccr.md) |
| **LangChain** | Memory, retrievers, agents, streaming | [LangChain](docs/langchain.md) |
| **Agno** | Agent framework integration with hooks | [Agno](docs/agno.md) |
| **Text Utilities** | Opt-in compression for search/logs | [Text Compression](docs/text-compression.md) |
| **LLMLingua-2** | ML-based 20x compression (opt-in) | [LLMLingua](docs/llmlingua.md) |
| **Code-Aware** | AST-based code compression (tree-sitter) | [Transforms](docs/transforms.md) |

---

## Performance

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| Search results (1000 items) | 45,000 tokens | 4,500 tokens | 90% |
| Log analysis (500 entries) | 22,000 tokens | 3,300 tokens | 85% |
| Long conversation (50 turns) | 80,000 tokens | 32,000 tokens | 60% |
| Agent with tools (10 calls) | 100,000 tokens | 15,000 tokens | 85% |

**Overhead**: ~1-5ms per request

---

## Providers

| Provider | Token Counting | Cache Optimization |
|----------|----------------|-------------------|
| OpenAI | tiktoken (exact) | Automatic prefix caching |
| Anthropic | Official API | cache_control blocks |
| Google | Official API | Context caching |
| Cohere | Official API | - |
| Mistral | Official tokenizer | - |

New models auto-supported via naming pattern detection.

---

## Safety Guarantees

- **Never removes human content** - user/assistant messages preserved
- **Never breaks tool ordering** - tool calls and responses stay paired
- **Parse failures are no-ops** - malformed content passes through unchanged
- **Compression is reversible** - LLM retrieves original data via CCR

---

## Installation

```bash
pip install headroom-ai              # SDK only
pip install "headroom-ai[proxy]"     # Proxy server
pip install "headroom-ai[langchain]" # LangChain integration
pip install "headroom-ai[agno]"      # Agno agent framework
pip install "headroom-ai[code]"      # AST-based code compression
pip install "headroom-ai[llmlingua]" # ML-based compression
pip install "headroom-ai[all]"       # Everything
```

**Requirements**: Python 3.10+

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Memory Guide](docs/memory.md) | Persistent memory for LLMs |
| [Compression Guide](docs/compression.md) | Universal compression with ML detection |
| [LangChain Integration](docs/langchain.md) | Full LangChain support |
| [Agno Integration](docs/agno.md) | Full Agno agent framework support |
| [SDK Guide](docs/sdk.md) | Fine-grained control |
| [Proxy Guide](docs/proxy.md) | Production deployment |
| [Configuration](docs/configuration.md) | All options |
| [CCR Guide](docs/ccr.md) | Reversible compression |
| [Metrics](docs/metrics.md) | Monitoring |
| [Troubleshooting](docs/troubleshooting.md) | Common issues |

---

## Who's Using Headroom?

> Add your project here! [Open a PR](https://github.com/chopratejas/headroom/pulls) or [start a discussion](https://github.com/chopratejas/headroom/discussions).

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

Apache License 2.0 - see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Built for the AI developer community</sub>
</p>
