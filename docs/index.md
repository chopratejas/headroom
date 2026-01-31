# Headroom

**The Context Optimization Layer for LLM Applications**

Tool outputs are 70-95% redundant. Headroom compresses that away—without losing information.

---

## Quick Install

```bash
pip install headroom-ai[all]
```

## Quick Start

### Option 1: Proxy (Zero Code Changes)

Start the proxy:

```bash
headroom proxy
```

Point your tools at it:

```bash
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

That's it. Your existing code works unchanged, with 40-90% fewer tokens.

### Option 2: Python SDK

```python
from headroom import Headroom

hr = Headroom()

# Compress tool output before sending to LLM
compressed = hr.compress(large_tool_output)

# If LLM needs the full data, retrieve it
original = hr.retrieve(compressed)
```

---

## Why Headroom?

| Problem | Solution |
|---------|----------|
| Tool outputs bloat context with repetitive JSON | Statistical compression removes redundancy |
| Dynamic content breaks provider caching | Cache alignment stabilizes prefixes |
| Long conversations exceed context limits | Intelligent scoring drops low-value messages |
| Compressed data might be needed later | CCR stores originals for on-demand retrieval |

---

## Results

**100 log entries. One critical error buried at position 67.**

| Metric | Baseline | Headroom |
|--------|----------|----------|
| Input tokens | 10,144 | 1,260 |
| Correct answers | 4/4 | 4/4 |

**87.6% fewer tokens. Same answer.**

The FATAL error was automatically preserved—no configuration needed.

---

## How It Works

```
Your App → Headroom → LLM Provider
              ↓
         Compression
         Caching
         Retrieval
```

1. **Intercepts context** — Tool outputs, logs, search results
2. **Compresses intelligently** — Keeps errors, outliers, boundaries
3. **Stores originals** — Full data available if LLM requests it
4. **Aligns for caching** — Provider caches actually hit

---

## Integrations

=== "LangChain"

    ```python
    from langchain_openai import ChatOpenAI
    from headroom.integrations import HeadroomChatModel

    llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))
    response = llm.invoke("Hello!")
    ```

=== "Agno"

    ```python
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from headroom.integrations.agno import HeadroomAgnoModel

    model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))
    agent = Agent(model=model)
    ```

=== "AWS Bedrock"

    ```bash
    # Start proxy with Bedrock backend
    headroom proxy --backend bedrock --region us-east-1

    # Point Claude Code at it
    ANTHROPIC_API_KEY="sk-ant-dummy" \
    ANTHROPIC_BASE_URL=http://localhost:8787 \
    claude
    ```

---

## Features

**Compression**

- Statistical JSON array compression (no hardcoded rules)
- ML-based text compression via LLMLingua
- AST-aware code compression
- Image optimization (40-90% reduction)

**Context Management**

- Intelligent message scoring and dropping
- Compress-Cache-Retrieve (CCR) for lossless compression
- Provider cache alignment for better hit rates

**Operations**

- Prometheus metrics endpoint
- Request logging and cost tracking
- Budget limits and rate limiting

---

## Next Steps

- [Quickstart Guide](quickstart.md) — Get running in 5 minutes
- [Proxy Documentation](proxy.md) — Configure the optimization proxy
- [Architecture](ARCHITECTURE.md) — Deep dive into how it works

---

## License

Apache 2.0 — Free for commercial use.
