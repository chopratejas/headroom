<p align="center">
  <h1 align="center">Headroom</h1>
  <p align="center">
    <strong>Compress everything your AI agent reads. Same answers, fraction of the tokens.</strong>
  </p>
  <p align="center">
    Every tool call, DB query, file read, and RAG retrieval your agent makes is 70-95% boilerplate.<br>
    Headroom compresses it away before it hits the model.<br><br>
    Works with <b>any agent</b> — coding agents (Claude Code, Codex, Cursor, Aider), custom agents<br>
    (LangChain, LangGraph, Agno, Strands, OpenClaw), or your own Python and TypeScript code.
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
  <a href="https://www.npmjs.com/package/headroom-ai">
    <img src="https://img.shields.io/npm/v/headroom-ai.svg" alt="npm">
  </a>
  <a href="https://github.com/chopratejas/headroom/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://chopratejas.github.io/headroom/">
    <img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="Documentation">
  </a>
  <a href="https://discord.gg/yRmaUNpsPJ">
    <img src="https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white" alt="Discord">
  </a>
</p>

---

## Where Headroom Fits

```
Your Agent / App
  (coding agents, customer support bots, RAG pipelines,
   data analysis agents, research agents, any LLM app)
      │
      │  tool calls, logs, DB reads, RAG results, file reads, API responses
      ▼
   Headroom  ← proxy, Python/TypeScript SDK, or framework integration
      │
      ▼
 LLM Provider  (OpenAI, Anthropic, Google, Bedrock, 100+ via LiteLLM)
```

Headroom sits between your application and the LLM provider. It intercepts requests, compresses the context, and forwards an optimized prompt. Use it as a transparent proxy (zero code changes), a Python function (`compress()`), or a framework integration (LangChain, LiteLLM, Agno).

### What gets compressed

Headroom optimizes any data your agent injects into a prompt:

- **Tool outputs** — shell commands, API calls, search results
- **Database queries** — SQL results, key-value lookups
- **RAG retrievals** — document chunks, embeddings results
- **File reads** — code, logs, configs, CSVs
- **API responses** — JSON, XML, HTML
- **Conversation history** — long agent sessions with repetitive context

---

## Quick Start

**Python:**
```bash
pip install "headroom-ai[all]"
```

**TypeScript / Node.js:**
```bash
npm install headroom-ai
```

### Any agent — one function

**Python:**
```python
from headroom import compress

result = compress(messages, model="claude-sonnet-4-5-20250929")
response = client.messages.create(model="claude-sonnet-4-5-20250929", messages=result.messages)
print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.0%})")
```

**TypeScript:**
```typescript
import { compress } from 'headroom-ai';

const result = await compress(messages, { model: 'gpt-4o' });
const response = await openai.chat.completions.create({ model: 'gpt-4o', messages: result.messages });
console.log(`Saved ${result.tokensSaved} tokens`);
```

Works with any LLM client — Anthropic, OpenAI, LiteLLM, Bedrock, Vercel AI SDK, or your own code.

### Any agent — proxy (zero code changes)

```bash
headroom proxy --port 8787
```

```bash
# Point any LLM client at the proxy
ANTHROPIC_BASE_URL=http://localhost:8787 your-app
OPENAI_BASE_URL=http://localhost:8787/v1 your-app
```

Works with any language, any tool, any framework. **[Proxy docs](docs/proxy.md)**

### Coding agents — one command

```bash
headroom wrap claude       # Starts proxy + launches Claude Code
headroom wrap codex        # Starts proxy + launches OpenAI Codex CLI
headroom wrap aider        # Starts proxy + launches Aider
headroom wrap cursor       # Starts proxy + prints Cursor config
```

Headroom starts a proxy, points your tool at it, and compresses everything automatically.

### Multi-agent — SharedContext

```python
from headroom import SharedContext

ctx = SharedContext()
ctx.put("research", big_agent_output)      # Agent A stores (compressed)
summary = ctx.get("research")               # Agent B reads (~80% smaller)
full = ctx.get("research", full=True)       # Agent B gets original if needed
```

Compress what moves between agents — any framework. **[SharedContext Guide](docs/shared-context.md)**

### MCP Tools (Claude Code, Cursor)

```bash
headroom mcp install && claude
```

Gives your AI tool three MCP tools: `headroom_compress`, `headroom_retrieve`, `headroom_stats`. **[MCP Guide](docs/mcp.md)**

### Drop into your existing stack

| Your setup | Add Headroom | One-liner |
|------------|-------------|-----------|
| **Any Python app** | `compress()` | `result = compress(messages, model="gpt-4o")` |
| **Any TypeScript app** | `compress()` | `const result = await compress(messages, { model: 'gpt-4o' })` |
| **Vercel AI SDK** | Middleware | `wrapLanguageModel({ model, middleware: headroomMiddleware() })` |
| **OpenAI Node SDK** | Wrap client | `const client = withHeadroom(new OpenAI())` |
| **Anthropic TS SDK** | Wrap client | `const client = withHeadroom(new Anthropic())` |
| **Multi-agent** | SharedContext | `ctx = SharedContext(); ctx.put("key", data)` |
| **LiteLLM** | Callback | `litellm.callbacks = [HeadroomCallback()]` |
| **Any Python proxy** | ASGI Middleware | `app.add_middleware(CompressionMiddleware)` |
| **Agno agents** | Wrap model | `HeadroomAgnoModel(your_model)` |
| **LangChain** | Wrap model | `HeadroomChatModel(your_llm)` |
| **OpenClaw** | ContextEngine plugin | `openclaw plugins install headroom-openclaw` |
| **Claude Code** | Wrap | `headroom wrap claude` |
| **Codex / Aider** | Wrap | `headroom wrap codex` or `headroom wrap aider` |

**[Full Integration Guide](docs/integration-guide.md)** | **[TypeScript SDK](docs/typescript-sdk.md)**

---

## Demo

<p align="center">
  <img src="Headroom-2.gif" alt="Headroom Demo" width="800">
</p>

---

## Does It Actually Work?

**100 production log entries. One critical error buried at position 67.**

|  | Baseline | Headroom |
|--|----------|----------|
| Input tokens | 10,144 | 1,260 |
| Correct answers | **4/4** | **4/4** |

Both responses: *"payment-gateway, error PG-5523, fix: Increase max_connections to 500, 1,847 transactions affected."*

**87.6% fewer tokens. Same answer.** Run it: `python examples/needle_in_haystack_test.py`

<details>
<summary><b>What Headroom kept</b></summary>

From 100 log entries, SmartCrusher kept 6: first 3 (boundary), the FATAL error at position 67 (anomaly detection), and last 2 (recency). The error was automatically preserved — not by keyword matching, but by statistical analysis of field variance.
</details>

### Real Workloads

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| Code search (100 results) | 17,765 | 1,408 | **92%** |
| SRE incident debugging | 65,694 | 5,118 | **92%** |
| Codebase exploration | 78,502 | 41,254 | **47%** |
| GitHub issue triage | 54,174 | 14,761 | **73%** |

### Accuracy Benchmarks

Compression preserves accuracy — tested on real OSS benchmarks.

**Standard Benchmarks** — Baseline (direct to API) vs Headroom (through proxy):

| Benchmark | Category | N | Baseline | Headroom | Delta |
|-----------|----------|---|----------|----------|-------|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Math | 100 | 0.870 | 0.870 | **0.000** |
| [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Factual | 100 | 0.530 | 0.560 | **+0.030** |

**Compression Benchmarks** — Accuracy after full compression stack:

| Benchmark | Category | N | Accuracy | Compression | Method |
|-----------|----------|---|----------|-------------|--------|
| [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2) | QA | 100 | **97%** | 19% | Before/After |
| [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | Tool/Function | 100 | **97%** | 32% | LLM-as-Judge |
| Tool Outputs (built-in) | Agent | 8 | **100%** | 20% | Before/After |
| CCR Needle Retention | Lossless | 50 | **100%** | 77% | Exact Match |

Run it yourself:

```bash
# Quick smoke test (8 cases, ~10s)
python -m headroom.evals quick -n 8 --provider openai --model gpt-4o-mini

# Full Tier 1 suite (~$3, ~15 min)
python -m headroom.evals suite --tier 1 -o eval_results/

# CI mode (exit 1 on regression)
python -m headroom.evals suite --tier 1 --ci
```

Full methodology: [Benchmarks](docs/benchmarks.md) | [Evals Framework](headroom/evals/README.md)

---

## Key Capabilities

### Lossless Compression

Headroom never throws data away. It compresses aggressively, stores the originals, and gives the LLM a tool to retrieve full details when needed. When it compresses 500 items to 20, it tells the model *what was omitted* ("87 passed, 2 failed, 1 error") so the model knows when to ask for more.

### Smart Content Detection

Auto-detects what's in your context — JSON arrays, code, logs, plain text — and routes each to the best compressor. JSON goes to SmartCrusher, code goes through AST-aware compression (Python, JS, Go, Rust, Java, C++), text goes to Kompress (ModernBERT-based, with `[ml]` extra).

### Cache Optimization

Stabilizes message prefixes so your provider's KV cache actually works. Claude offers a 90% read discount on cached prefixes — but almost no framework takes advantage of it. Headroom does.

### Failure Learning

```bash
headroom learn                   # Analyze past Claude Code sessions, show recommendations
headroom learn --apply           # Write learnings to CLAUDE.md and MEMORY.md
headroom learn --all --apply     # Learn across all your projects
```

Reads your conversation history, finds every failed tool call, correlates it with what eventually succeeded, and writes specific corrections into your project files. Next session starts smarter. **[Learn docs](docs/learn.md)**

<p align="center">
  <img src="headroom_learn.gif" alt="headroom learn demo" width="800">
</p>

### Image Compression

40-90% token reduction via trained ML router. Automatically selects the right resize/quality tradeoff per image.

<details>
<summary><b>All features</b></summary>

| Feature | What it does |
|---------|-------------|
| **Content Router** | Auto-detects content type, routes to optimal compressor |
| **SmartCrusher** | Universal JSON compression — arrays of dicts, strings, numbers, mixed types, nested objects |
| **CodeCompressor** | AST-aware compression for Python, JS, Go, Rust, Java, C++ |
| **Kompress** | ModernBERT token compression (replaces LLMLingua-2) |
| **CCR** | Reversible compression — LLM retrieves originals when needed |
| **Compression Summaries** | Tells the LLM what was omitted ("3 errors, 12 failures") |
| **CacheAligner** | Stabilizes prefixes for provider KV cache hits |
| **IntelligentContext** | Score-based context management with learned importance |
| **Image Compression** | 40-90% token reduction via trained ML router |
| **Memory** | Persistent memory across conversations |
| **Compression Hooks** | Customize compression with pre/post hooks |
| **Read Lifecycle** | Detects stale/superseded Read outputs, replaces with CCR markers |
| **`headroom learn`** | Analyzes past failures, writes project-specific learnings to CLAUDE.md/MEMORY.md |
| **`headroom wrap`** | One-command setup for Claude Code, Codex, Aider, Cursor |
| **SharedContext** | Compressed inter-agent context sharing for multi-agent workflows |
| **MCP Tools** | headroom_compress, headroom_retrieve, headroom_stats for Claude Code/Cursor |

</details>

---

## Headroom vs Alternatives

Context compression is a new space. Here's how the approaches differ:

| | Approach | Scope | Deploy as | Framework integrations | Data stays local? | Reversible |
|---|---|---|---|---|---|---|
| **Headroom** | Multi-algorithm compression | All context (tool outputs, DB reads, RAG, files, logs, history) | Proxy, Python library, ASGI middleware, or callback | LangChain, LangGraph, Agno, Strands, LiteLLM, MCP | Yes (OSS) | Yes (CCR) |
| **[RTK](https://github.com/rtk-ai/rtk)** | CLI command rewriter | Shell command outputs | CLI wrapper | None | Yes (OSS) | No |
| **[Compresr](https://compresr.ai)** | Cloud compression API | Text sent to their API | API call | None | No | No |
| **[Token Company](https://thetokencompany.ai)** | Cloud compression API | Text sent to their API | API call | None | No | No |

**Use it however you want.** Headroom works as a standalone proxy (`headroom proxy`), a one-function Python library (`compress()`), ASGI middleware, or a LiteLLM callback. Already using LiteLLM, LangChain, or Agno? Drop Headroom in without replacing anything.

**Headroom + RTK work well together.** RTK rewrites CLI commands (`git show` → `git show --short`), Headroom compresses everything else (JSON arrays, code, logs, RAG results, conversation history). Use both.

**Headroom vs cloud APIs.** Compresr and Token Company are hosted services — you send your context to their servers, they compress and return it. Headroom runs locally. Your data never leaves your machine. You also get lossless compression (CCR): the LLM can retrieve the full original when it needs more detail.

---

## How It Works Inside

```
  Your prompt
      │
      ▼
  1. CacheAligner            Stabilize prefix for KV cache
      │
      ▼
  2. ContentRouter           Route each content type:
      │                         → SmartCrusher    (JSON)
      │                         → CodeCompressor  (code)
      │                         → Kompress        (text, with [ml])
      ▼
  3. IntelligentContext      Score-based token fitting
      │
      ▼
  LLM Provider

  Needs full details? LLM calls headroom_retrieve.
  Originals are in the Compressed Store — nothing is thrown away.
```

**Overhead**: 15-200ms compression latency (net positive for Sonnet/Opus). Full data: [Latency Benchmarks](docs/LATENCY_BENCHMARKS.md)

---

## Integrations

| Integration | Status | Docs |
|-------------|--------|------|
| `headroom wrap claude/codex/aider/cursor` | **Stable** | [Proxy Docs](docs/proxy.md) |
| `compress()` — one function | **Stable** | [Integration Guide](docs/integration-guide.md) |
| `SharedContext` — multi-agent | **Stable** | [SharedContext Guide](docs/shared-context.md) |
| LiteLLM callback | **Stable** | [Integration Guide](docs/integration-guide.md#litellm) |
| ASGI middleware | **Stable** | [Integration Guide](docs/integration-guide.md#asgi-middleware) |
| Proxy server | **Stable** | [Proxy Docs](docs/proxy.md) |
| Agno | **Stable** | [Agno Guide](docs/agno.md) |
| MCP (Claude Code, Cursor, etc.) | **Stable** | [MCP Guide](docs/mcp.md) |
| Strands | **Stable** | [Strands Guide](docs/strands.md) |
| LangChain | **Stable** | [LangChain Guide](docs/langchain.md) |

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
pip install "headroom-ai[all]"         # Everything including evals (recommended)
pip install "headroom-ai[proxy]"       # Proxy server + MCP tools
pip install "headroom-ai[mcp]"         # MCP tools only (no proxy)
pip install "headroom-ai[ml]"          # ML compression (Kompress, requires torch)
pip install "headroom-ai[agno]"        # Agno integration
pip install "headroom-ai[langchain]"   # LangChain (experimental)
pip install "headroom-ai[evals]"       # Evaluation framework only
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
| [Latency Benchmarks](docs/LATENCY_BENCHMARKS.md) | Compression overhead & cost-benefit analysis |
| [Limitations](docs/LIMITATIONS.md) | When compression helps, when it doesn't |
| [Evals Framework](headroom/evals/README.md) | Prove compression preserves accuracy |
| [Memory](docs/memory.md) | Persistent memory |
| [Agno](docs/agno.md) | Agno agent framework |
| [MCP](docs/mcp.md) | Context engineering toolkit (compress, retrieve, stats) |
| [SharedContext](docs/shared-context.md) | Compressed inter-agent context sharing |
| [Learn](docs/learn.md) | Offline failure learning for coding agents |
| [Configuration](docs/configuration.md) | All options |

---

## Community

Questions, feedback, or just want to follow along? **[Join us on Discord](https://discord.gg/yRmaUNpsPJ)**

---

## Contributing

```bash
git clone https://github.com/chopratejas/headroom.git && cd headroom
pip install -e ".[dev]" && pytest
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
