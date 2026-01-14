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

Headroom is a **smart compression layer** for LLM applications:

- **Compresses tool outputs** — 1000 search results → 15 items (keeps errors, anomalies, relevant items)
- **Enables provider caching** — Stabilizes prefixes so cache hits actually happen
- **Manages context windows** — Prevents token limit failures without breaking tool calls
- **Reversible compression** — LLM can retrieve original data if needed ([CCR architecture](docs/ccr.md))

Works as a **proxy** (zero code changes) or **SDK** (fine-grained control).

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

---

## Framework Integrations

| Framework | Integration | Docs |
|-----------|-------------|------|
| **LangChain** | `HeadroomChatModel`, memory, retrievers, agents | [Guide](docs/langchain.md) |
| **MCP** | Tool output compression for Claude | [Guide](docs/ccr.md) |
| **Any OpenAI Client** | Proxy server | [Guide](docs/proxy.md) |

### LangChain Highlights

```python
from headroom.integrations import (
    HeadroomChatModel,           # Wrap any chat model
    HeadroomChatMessageHistory,  # Auto-compress conversation history
    HeadroomDocumentCompressor,  # Filter retrieved documents
    wrap_tools_with_headroom,    # Compress agent tool outputs
)

# Memory that auto-compresses when over 4K tokens
memory = ConversationBufferMemory(
    chat_memory=HeadroomChatMessageHistory(base_history)
)

# Retriever that keeps only relevant docs
retriever = ContextualCompressionRetriever(
    base_compressor=HeadroomDocumentCompressor(max_documents=10),
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 50}),
)

# Agent tools with automatic output compression
tools = wrap_tools_with_headroom([search_tool, database_tool])
```

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

Or in Python:

```python
print(llm.get_metrics())
# {'tokens_saved': 12500, 'savings_percent': 45.2}
```

---

## Installation

```bash
pip install headroom-ai              # SDK only
pip install "headroom-ai[proxy]"     # Proxy server
pip install "headroom-ai[langchain]" # LangChain integration
pip install "headroom-ai[code]"      # AST-based code compression
pip install "headroom-ai[llmlingua]" # ML-based compression
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
| **LangChain** | Memory, retrievers, agents, streaming | [LangChain](docs/langchain.md) |
| **Text Utilities** | Opt-in compression for search/logs | [Text Compression](docs/text-compression.md) |
| **LLMLingua-2** | ML-based 20x compression (opt-in) | [LLMLingua](docs/llmlingua.md) |
| **Code-Aware** | AST-based code compression (tree-sitter) | [Transforms](docs/transforms.md) |

---

## Providers

| Provider | Token Counting | Cache Optimization |
|----------|----------------|-------------------|
| OpenAI | tiktoken (exact) | Automatic prefix caching |
| Anthropic | Official API | cache_control blocks |
| Google | Official API | Context caching |
| Cohere | Official API | - |
| Mistral | Official tokenizer | - |

**New models auto-supported** — Unknown models get sensible defaults based on naming patterns.

---

## Performance

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| Search results (1000 items) | 45,000 tokens | 4,500 tokens | 90% |
| Log analysis (500 entries) | 22,000 tokens | 3,300 tokens | 85% |
| Long conversation (50 turns) | 80,000 tokens | 32,000 tokens | 60% |
| Agent with tools (10 calls) | 100,000 tokens | 15,000 tokens | 85% |

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
| [LangChain Integration](docs/langchain.md) | Full LangChain support |
| [SDK Guide](docs/sdk.md) | Wrap your client for fine-grained control |
| [Proxy Guide](docs/proxy.md) | Production deployment |
| [Configuration](docs/configuration.md) | All configuration options |
| [CCR Guide](docs/ccr.md) | Reversible compression architecture |
| [Metrics](docs/metrics.md) | Monitoring and observability |
| [Troubleshooting](docs/troubleshooting.md) | Common issues |

---

## Examples

See [`examples/`](examples/) for runnable code:

- `basic_usage.py` — Simple SDK usage
- `proxy_integration.py` — Using with different clients
- `langchain_agent.py` — LangChain ReAct agent with Headroom
- `rag_pipeline.py` — RAG with document compression
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
