# I Was Wasting 85% of My LLM Tokens on JSON Boilerplate

I recently built an agent to handle some SRE tasks—fetching logs, querying databases, searching code. It worked, but when I looked at the traces, I was annoyed.

It wasn't just that it was expensive (though the bill was climbing). It was the sheer **inefficiency**.

I looked at a single tool output—a search for Python files. It was 40,000 tokens.
About 35,000 of those tokens were just `"type": "file"` and `"language": "python"` repeated 2,000 times.

We are paying premium compute prices to force state-of-the-art models to read standard JSON boilerplate.

I couldn't find a tool that solved this without breaking the agent, so I wrote one. It's called **Headroom**. It's a context optimization layer that sits between your app and your LLM. It compresses context by ~85% without losing semantic meaning.

It's open source (Apache-2.0). If you just want the code:
**[github.com/chopratejas/headroom](https://github.com/chopratejas/headroom)**

---

## Why Truncation and Summarization Don't Work

When your context window fills up, the standard industry solution is **truncation** (chopping off the oldest messages or the middle of the document).

But for an agent, truncation is dangerous.

* If you chop the middle of a log file, you might lose the one error line that explains the crash.
* If you chop a file list, you might lose the exact config file the user asked for.

I tried **summarization** (using a cheaper model to summarize the data first), but that introduced hallucination. I had a summarizer tell me a deployment "looked fine" because it ignored specific error codes in the raw log.

I needed a third option: **Lossless compression.** Or at least, "intent-lossless."

---

## The Core Idea: Statistical Analysis, Not Blind Truncation

I realized that 90% of the data in a tool output is just schema scaffolding. The LLM doesn't need to see `status: active` repeated a thousand times. It needs the **anomalies**.

Headroom's SmartCrusher runs statistical analysis before touching your data:

**1. Constant Factoring**
If every item in an array has `"type": "file"`, it doesn't repeat that 2,000 times. It extracts constants once.

**2. Outlier Detection**
It calculates standard deviation of numerical fields. It preserves the spikes—the values that are >2σ from the mean. Those are usually what matters.

**3. Error Preservation**
Hard rule: never discard strings that look like stack traces, error messages, or failures. Errors are sacred.

**4. Relevance Scoring**
If you searched for "auth", items containing "auth" get preserved. Uses BM25 + semantic embeddings (hybrid scoring) to match items against the user's query context.

**5. First/Last Retention**
Always keeps first few and last few items. The LLM expects to see some examples, and recency matters.

The result: 40,000 tokens → 4,000 tokens. Same information density. No hallucination risk.

---

## CCR: Making Compression Reversible

Here's the insight that changed everything: **compression should be reversible**.

I call the architecture **CCR** (Compress-Cache-Retrieve):

### 1. Compress
SmartCrusher compresses the tool output from 2,000 items to 20.

### 2. Cache
The original 2,000 items are cached locally (5-minute TTL, LRU eviction).

### 3. Retrieve
Headroom injects a tool called `headroom_retrieve()` into the LLM's context. If the model looks at the compressed summary and decides it needs more data—maybe the user asked a follow-up question—it can call that tool. Headroom fetches from the cache and returns the relevant items.

This changes the risk calculus. You can compress aggressively (90%+) because **nothing is ever truly lost**. The model can always "unzip" what it needs.

I've had conversations like this:

```
Turn 1: "Search for all Python files"
        → 1000 files returned, compressed to 15

Turn 5: "Actually, what was that file handling JWT tokens?"
        → LLM calls headroom_retrieve("jwt")
        → Returns jwt_handler.py from cached data
```

No extra API calls. No "sorry, I don't have that information anymore."

---

## TOIN: The Network Effect

Here's where it gets interesting. Headroom learns from compression patterns.

**TOIN** (Tool Output Intelligence Network) tracks—anonymously—what happens after compression:
- Which fields get retrieved most often?
- Which tool types have high retrieval rates?
- What query patterns trigger retrievals?

This data feeds back into compression recommendations. If TOIN learns that users frequently retrieve `error_code` fields after compression, it tells SmartCrusher to preserve `error_code` more aggressively next time.

Privacy is built in:
- No actual data values stored
- Tool names are structure hashes
- Field names are SHA256[:8] hashes
- No user identifiers

The network effect: more users → more compression events → better recommendations for everyone.

---

## Memory: Cross-Conversation Learning

Agents often need to remember things across conversations. "I prefer dark mode." "My timezone is PST." "I'm working on the auth refactor."

Headroom has a memory system that extracts and stores these facts automatically.

Two approaches:

**Fast Memory (Recommended)**
Zero extra latency. The LLM outputs a `<memory>` block inline with its response. Headroom parses it out and stores the memory.

```python
from headroom.memory import with_fast_memory
client = with_fast_memory(OpenAI(), user_id="alice")

# Memories extracted automatically from responses
# Injected automatically into future requests
```

**Background Memory**
Separate LLM call extracts memories asynchronously. More accurate but adds latency.

```python
from headroom import with_memory
client = with_memory(OpenAI(), user_id="alice")
```

Memories are stored locally (SQLite) and injected into future conversations. The model remembers that Alice prefers dark mode without you managing state.

---

## The Transform Pipeline

Headroom runs four transforms on each request:

### 1. CacheAligner
LLM providers offer cached token pricing (Anthropic: 90% off, OpenAI: 50% off). But caching only works if your prompt prefix is stable.

Problem: your system prompt probably has a timestamp. `Current time: 2024-01-15 10:32:45`. That breaks caching.

CacheAligner extracts dynamic content and moves it to the end, stabilizing the prefix. Same information, better cache hits.

### 2. SmartCrusher
The statistical compression engine. Analyzes arrays, detects patterns, preserves anomalies, factors constants.

### 3. ContentRouter
Different content needs different compression. Code isn't JSON isn't logs isn't prose.

ContentRouter uses ML-based content detection to route data to specialized compressors:
- **Code** → AST-aware compression (tree-sitter)
- **JSON** → SmartCrusher
- **Logs** → LogCompressor (clusters similar messages)
- **Text** → Optional LLMLingua integration (20x compression, adds latency)

### 4. RollingWindow
When context exceeds the model limit, something has to go. RollingWindow drops oldest tool calls + responses together (never orphans data), preserves system prompt and recent turns.

---

## Three Ways to Use It

### Option 1: Proxy Server (Zero Code Changes)

```bash
pip install headroom-ai
headroom proxy --port 8787
```

Point your OpenAI client to `http://localhost:8787/v1`. Done.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8787/v1")
# No other changes
```

Works with Claude Code, Cursor, any OpenAI-compatible client.

### Option 2: SDK Wrapper

```python
from headroom import HeadroomClient
from openai import OpenAI

client = HeadroomClient(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    headroom_mode="optimize"  # or "audit" or "simulate"
)
```

Three modes:
- **audit**: Observe only. Logs what would be optimized, doesn't change anything.
- **optimize**: Apply compression. This is what saves tokens.
- **simulate**: Dry run. Returns the optimized messages without calling the API.

Start with `audit` to see potential savings, then flip to `optimize` when you're confident.

### Option 3: Framework Integrations

**LangChain:**
```python
from langchain_openai import ChatOpenAI
from headroom.integrations.langchain import HeadroomChatModel

base_model = ChatOpenAI(model="gpt-4o")
model = HeadroomChatModel(base_model, mode="optimize")

# Use in any chain or agent
chain = prompt | model | parser
```

**Agno:**
```python
from agno.agent import Agent
from headroom.integrations.agno import HeadroomAgnoModel

model = HeadroomAgnoModel(original_model, mode="optimize")
agent = Agent(model=model, tools=[...])
```

**MCP (Model Context Protocol):**
```python
from headroom.integrations.mcp import compress_tool_result

# Compress any tool result before returning to LLM
compressed = compress_tool_result(tool_name, result_data)
```

---

## Real Numbers

I've been running this in production for months. Here's what the token reduction looks like:

| Workload | Before | After | Savings |
|----------|--------|-------|---------|
| Log Analysis | 22,000 | 3,300 | 85% |
| Code Search | 45,000 | 4,500 | 90% |
| Database Queries | 18,000 | 2,700 | 85% |
| Long Conversations | 80,000 | 32,000 | 60% |

Latency overhead: 3-5ms per request. No extra LLM calls.

---

## What's Coming Next

This is actively maintained. On the roadmap:

**More Frameworks**
- CrewAI integration
- AutoGen integration
- Semantic Kernel integration

**Managed Storage**
- Cloud-hosted TOIN backend (opt-in)
- Cross-device memory sync
- Team-shared compression patterns

**Better Compression**
- Domain-specific profiles (SRE, coding, data analysis)
- Custom compressor plugins
- Streaming compression for real-time tools

---

## Why I Built This

I'm a believer that we're in the "optimization phase" of the AI hype cycle. Getting things to work is table stakes; getting them to work cheaply and reliably is the actual engineering work.

Headroom is my attempt to fix the "context bloat" problem properly. Not with heuristics or truncation, but with statistical analysis and reversible compression.

It runs entirely locally. No data leaves your machine (except to OpenAI/Anthropic as usual). Apache-2.0 licensed.

**Repo:** [github.com/chopratejas/headroom](https://github.com/chopratejas/headroom)

If you find bugs or have ideas, open an issue. I'm actively maintaining this.

---

*Tags: #llm #ai #python #openai #anthropic #agents #optimization*
