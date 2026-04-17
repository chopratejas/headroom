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

**Docker-native (no Python or Node on host):**
```bash
curl -fsSL https://raw.githubusercontent.com/chopratejas/headroom/main/scripts/install.sh | bash
```

macOS uses Bash 4.3+, so run the installer with a newer Bash such as Homebrew's `bash`.

PowerShell:
```powershell
irm https://raw.githubusercontent.com/chopratejas/headroom/main/scripts/install.ps1 | iex
```

**Persistent local runtime (Python-native service/task flow):**
```bash
headroom install apply --preset persistent-service --providers auto
```

**Persistent local runtime (Docker-native wrapper / compose flow):**
```bash
headroom install apply --preset persistent-docker
```

### Any agent — one function

**Python:**
```python
from headroom import compress

# Default (coding agents — protects user messages, compresses tool outputs)
result = compress(messages, model="claude-sonnet-4-5-20250929")
response = client.messages.create(model="claude-sonnet-4-5-20250929", messages=result.messages)
print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.0%})")

# Document compression (financial, legal, clinical — compress everything, keep 50%)
result = compress(messages, model="claude-opus-4-20250514",
    compress_user_messages=True,   # Compress user messages too
    target_ratio=0.5,              # Keep 50% (preserves numbers/entities)
    protect_recent=0,              # Don't protect recent messages
)
```

**TypeScript:**
```typescript
import { compress } from 'headroom-ai';

const result = await compress(messages, { model: 'gpt-4o' });
const response = await openai.chat.completions.create({ model: 'gpt-4o', messages: result.messages });
console.log(`Saved ${result.tokensSaved} tokens`);
```

Works with any LLM client — Anthropic, OpenAI, LiteLLM, Bedrock, Vercel AI SDK, or your own code. Full options via `CompressConfig`: `compress_user_messages`, `target_ratio`, `protect_recent`, `protect_analysis_context`.

### Any agent — proxy (zero code changes)

```bash
headroom proxy --port 8787
```

```bash
# Run mode (default: token)
headroom proxy --mode token   # maximize compression
headroom proxy --mode cache   # preserve Anthropic/OpenAI prefix cache stability
```

```bash
# Point any LLM client at the proxy
ANTHROPIC_BASE_URL=http://localhost:8787 your-app
OPENAI_BASE_URL=http://localhost:8787/v1 your-app
```

Use `token` mode for short/medium sessions where raw compression savings matter most.
Use `cache` mode for long-running chats where preserving prior-turn bytes improves provider cache reuse.

Works with any language, any tool, any framework. **[Proxy docs](docs/content/docs/proxy.mdx)**

Prefer Docker as the runtime provider? See **[Installation — Docker](docs/content/docs/installation.mdx)**.

### Coding agents — one command

```bash
headroom wrap claude              # Starts proxy + launches Claude Code
headroom wrap copilot -- --model claude-sonnet-4-20250514
                                   # Starts proxy + launches GitHub Copilot CLI
headroom wrap codex               # Starts proxy + launches OpenAI Codex CLI
headroom wrap aider               # Starts proxy + launches Aider
headroom wrap cursor              # Starts proxy + prints Cursor config
headroom wrap openclaw            # Installs + configures OpenClaw plugin
headroom wrap claude --memory     # With persistent cross-agent memory
headroom wrap codex --memory      # Shares the same memory store
headroom wrap claude --code-graph # With code graph intelligence (codebase-memory-mcp)
```

Headroom starts a proxy, points your tool at it, and compresses everything automatically. Add `--memory` for persistent memory that's shared across agents. Add `--code-graph` for code intelligence via [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) — indexes your codebase into a knowledge graph for call-chain traversal, impact analysis, and architectural queries. `wrap copilot` is part of the Python-native CLI; the Docker-native wrapper currently supports `claude`, `codex`, `aider`, `cursor`, and `openclaw`.

In Docker-native mode, Headroom still runs in Docker while wrapped tools run on the host. `wrap claude`, `wrap codex`, `wrap aider`, `wrap cursor`, and OpenClaw plugin setup (`wrap openclaw` / `unwrap openclaw`) are host-managed through the installed wrapper.

### Multi-agent — SharedContext

```python
from headroom import SharedContext

ctx = SharedContext()
ctx.put("research", big_agent_output)      # Agent A stores (compressed)
summary = ctx.get("research")               # Agent B reads (~80% smaller)
full = ctx.get("research", full=True)       # Agent B gets original if needed
```

Compress what moves between agents — any framework. **[SharedContext Guide](docs/content/docs/shared-context.mdx)**

### MCP Tools (Claude Code, Cursor)

```bash
headroom mcp install && claude
```

Gives your AI tool three MCP tools: `headroom_compress`, `headroom_retrieve`, `headroom_stats`. **[MCP Guide](docs/content/docs/mcp.mdx)**

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
| **OpenClaw** | One-command wrap/unwrap | `headroom wrap openclaw` / `headroom unwrap openclaw` |
| **Claude Code** | Wrap | `headroom wrap claude` |
| **GitHub Copilot CLI** | Wrap | `headroom wrap copilot -- --model claude-sonnet-4-20250514` |
| **Codex / Aider** | Wrap | `headroom wrap codex` or `headroom wrap aider` |
| **Always-on local proxy** | Persistent install | `headroom install apply --preset persistent-service --providers auto` |

**[Full Integration Guide](docs/content/docs/index.mdx)**

---

## Demo

<p align="center">
  <img src="HeadroomDemo-Fast.gif" alt="Headroom Demo" width="800">
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

Full methodology: [Benchmarks](docs/content/docs/benchmarks.mdx) | [Evals Framework](headroom/evals/README.md)

---

## Key Capabilities

### Lossless Compression

Headroom never throws data away. It compresses aggressively, stores the originals, and gives the LLM a tool to retrieve full details when needed. When it compresses 500 items to 20, it tells the model *what was omitted* ("87 passed, 2 failed, 1 error") so the model knows when to ask for more.

### Smart Content Detection

Auto-detects what's in your context — JSON arrays, code, logs, plain text — and routes each to the best compressor. JSON goes to SmartCrusher, code goes through AST-aware compression (Python, JS, Go, Rust, Java, C++), text goes to Kompress (ModernBERT-based, with `[ml]` extra).

### Cache Optimization

Stabilizes message prefixes so your provider's KV cache actually works. Claude offers a 90% read discount on cached prefixes — but almost no framework takes advantage of it. Headroom does.

### Cross-Agent Memory

```bash
headroom wrap claude --memory    # Claude with persistent memory
headroom wrap codex --memory     # Codex shares the SAME memory store
```

Claude saves a fact, Codex reads it back. All agents sharing one proxy share one memory — project-scoped, user-isolated, with agent provenance tracking and automatic deduplication. No SDK changes needed. **[Memory docs](docs/content/docs/memory.mdx)**

### Failure Learning

```bash
headroom learn                        # Auto-detect agent (Claude, Codex, Gemini)
headroom learn --apply                # Write learnings to agent-native files
headroom learn --agent codex --all    # Analyze all Codex sessions
```

Plugin-based: reads conversation history from Claude Code, Codex, or Gemini CLI. Finds failure patterns, correlates with successes, writes corrections to CLAUDE.md / AGENTS.md / GEMINI.md. External plugins via entry points. **[Learn docs](docs/content/docs/failure-learning.mdx)**

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
| **Memory** | Cross-agent persistent memory — Claude saves, Codex reads it back. Agent provenance + auto-dedup |
| **Compression Hooks** | Customize compression with pre/post hooks |
| **Read Lifecycle** | Detects stale/superseded Read outputs, replaces with CCR markers |
| **`headroom learn`** | Plugin-based failure learning for Claude Code, Codex, Gemini CLI (extensible via entry points) |
| **`headroom wrap`** | One-command setup for Claude Code, GitHub Copilot CLI, Codex, Aider, Cursor |
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

**Overhead**: 15-200ms compression latency (net positive for Sonnet/Opus). Full data: [Benchmarks](docs/content/docs/benchmarks.mdx)

---

## Integrations

| Integration | Status | Docs |
|-------------|--------|------|
| `headroom wrap claude/copilot/codex/aider/cursor` | **Stable** | [Proxy Docs](docs/content/docs/proxy.mdx) |
| `compress()` — one function | **Stable** | [Integration Guide](docs/content/docs/index.mdx) |
| `SharedContext` — multi-agent | **Stable** | [SharedContext Guide](docs/content/docs/shared-context.mdx) |
| LiteLLM callback | **Stable** | [LiteLLM Guide](docs/content/docs/litellm.mdx) |
| ASGI middleware | **Stable** | [Integration Guide](docs/content/docs/index.mdx) |
| Proxy server | **Stable** | [Proxy Docs](docs/content/docs/proxy.mdx) |
| Agno | **Stable** | [Agno Guide](docs/content/docs/agno.mdx) |
| MCP (Claude Code, Cursor, etc.) | **Stable** | [MCP Guide](docs/content/docs/mcp.mdx) |
| Strands | **Stable** | [Strands Guide](docs/content/docs/strands.mdx) |
| LangChain | **Stable** | [LangChain Guide](docs/content/docs/langchain.mdx) |
| **OpenClaw** | **Stable** | [OpenClaw plugin](#openclaw-plugin) |

---

## OpenClaw Plugin

The [`@headroom-ai/openclaw`](plugins/openclaw) plugin integrates Headroom as a ContextEngine for [OpenClaw](https://github.com/openclaw/openclaw). It compresses tool outputs, code, logs, and structured data inline — 70-90% token savings with zero LLM calls. The plugin can connect to a local or remote Headroom proxy and will auto-start one locally if needed.

### Install

```bash
pip install "headroom-ai[proxy]"
openclaw plugins install --dangerously-force-unsafe-install headroom-ai/openclaw
```

> **Why `--dangerously-force-unsafe-install`?** The plugin auto-starts `headroom proxy` as a subprocess when no running proxy is detected. OpenClaw blocks process-launching plugins by default, so this flag is required to permit that behavior.

Once installed, assign Headroom as the context engine in your OpenClaw config:

```json
{
  "plugins": {
    "entries": { "headroom": { "enabled": true } },
    "slots": { "contextEngine": "headroom" }
  }
}
```

The plugin auto-detects and auto-starts the proxy — no manual proxy management needed. See the [plugin README](plugins/openclaw/README.md) for full configuration options, local development setup, and launcher details.

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

### Container images (GHCR tags)

- supported platforms: `linux/amd64`, `linux/arm64`
- tags `:code` - image with Code-Aware Compression (AST-based) i.e. `pip install "headroom-ai[proxy,code]"`
- tags `:slim` - image with distorless base

| Tag                 |                                                      | Extras       | Docker Bake target          |
|---------------------|------------------------------------------------------|--------------|-----------------------------|
| `<version>`         | ```ghcr.io/chopratejas/headroom:<version>```         | `proxy`      | `runtime`                   |
| `latest`            | ```ghcr.io/chopratejas/headroom:latest```            | `proxy`      | `runtime`                   |
| `nonroot`           | ```ghcr.io/chopratejas/headroom:nonroot```           | `proxy`      | `runtime-nonroot`           |
| `code`              | ```ghcr.io/chopratejas/headroom:code```              | `proxy,code` | `runtime-code`              |
| `code-nonroot`      | ```ghcr.io/chopratejas/headroom:code-nonroot```      | `proxy,code` | `runtime-code-nonroot`      |
| `slim`              | ```ghcr.io/chopratejas/headroom:slim```              | `proxy`      | `runtime-slim`              |
| `slim-nonroot`      | ```ghcr.io/chopratejas/headroom:slim-nonroot```      | `proxy`      | `runtime-slim-nonroot`      |
| `code-slim`         | ```ghcr.io/chopratejas/headroom:code-slim```         | `proxy,code` | `runtime-code-slim`         |
| `code-slim-nonroot` | ```ghcr.io/chopratejas/headroom:code-slim-nonroot``` | `proxy,code` | `runtime-code-slim-nonroot` |

### Docker Bake

```bash
# List all available build targets
docker buildx bake --list targets

# Build default image locally (proxy + nonroot)
docker buildx bake runtime-default

# Build one variant and load to local Docker image store
docker buildx bake runtime-code-slim-nonroot \
  --set runtime-code-slim-nonroot.platform=linux/amd64 \
  --set runtime-code-slim-nonroot.tags=headroom:local \
  --load
```

Python 3.10+

---

## Documentation

| | |
|---|---|
| [Integration Guide](docs/content/docs/index.mdx) | LiteLLM, ASGI, compress(), proxy |
| [Proxy Docs](docs/content/docs/proxy.mdx) | Proxy server configuration |
| [Architecture](docs/content/docs/architecture.mdx) | How the pipeline works |
| [CCR Guide](docs/content/docs/ccr.mdx) | Reversible compression |
| [Benchmarks](docs/content/docs/benchmarks.mdx) | Accuracy validation |
| [Limitations](docs/content/docs/limitations.mdx) | When compression helps, when it doesn't |
| [Evals Framework](headroom/evals/README.md) | Prove compression preserves accuracy |
| [Memory](docs/content/docs/memory.mdx) | Cross-agent persistent memory with provenance + dedup |
| [Agno](docs/content/docs/agno.mdx) | Agno agent framework |
| [MCP](docs/content/docs/mcp.mdx) | Context engineering toolkit (compress, retrieve, stats) |
| [SharedContext](docs/content/docs/shared-context.mdx) | Compressed inter-agent context sharing |
| [Learn](docs/content/docs/failure-learning.mdx) | Plugin-based failure learning (Claude, Codex, Gemini, extensible) |
| [Installation](docs/content/docs/installation.mdx) | pip, npm, Docker install methods |
| [Configuration](docs/content/docs/configuration.mdx) | All options |
| [Filesystem Contract](wiki/filesystem-contract.md) | `HEADROOM_CONFIG_DIR` / `HEADROOM_WORKSPACE_DIR` and per-resource path overrides |

---

## Community

Questions, feedback, or just want to follow along? **[Join us on Discord](https://discord.gg/yRmaUNpsPJ)**

---

## Contributing

```bash
git clone https://github.com/chopratejas/headroom.git && cd headroom
pip install -e ".[dev]" && pytest
```

Prefer a containerized setup? Open the repo in **`.devcontainer/devcontainer.json`** for the default Python/uv workflow, or **`.devcontainer/memory-stack/devcontainer.json`** when you need local Qdrant + Neo4j services and the locked `memory-stack` extra for the `qdrant-neo4j` memory backend. Inside that container, use `qdrant:6333` and `neo4j://neo4j:7687` instead of `localhost`.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
