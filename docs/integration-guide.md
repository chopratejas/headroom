# Integration Guide

You don't need to run the Headroom proxy. Headroom is a compression library that works with **any** LLM client, proxy, or framework.

## Pick Your Path

| You have... | Use this | Setup |
|-------------|----------|-------|
| Any Python app | [`compress()`](#compress-function) | 2 lines |
| LiteLLM | [LiteLLM callback](#litellm) | 1 line |
| A Python proxy (FastAPI, custom) | [ASGI middleware](#asgi-middleware) | 1 line |
| Claude Code / Cursor | [Headroom proxy](#proxy) | 1 env var |
| Agno agents | [Agno integration](#agno) | Wrap model |
| LangChain | [LangChain integration](#langchain) | Wrap model |
| Non-Python app | [Headroom proxy](#proxy) | HTTP |

---

## compress() Function

The simplest integration. Works with any LLM client.

```python
from headroom import compress

# Before sending to your LLM:
result = compress(messages, model="claude-sonnet-4-5-20250929")
response = your_client.create(messages=result.messages)  # Fewer tokens, same answer

print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.0%})")
```

### With Anthropic SDK

```python
from anthropic import Anthropic
from headroom import compress

client = Anthropic()
messages = [
    {"role": "user", "content": "What went wrong?"},
    {"role": "assistant", "content": "Let me check.", "tool_use": [...]},
    {"role": "user", "content": [{"type": "tool_result", "content": huge_json}]},
]

compressed = compress(messages, model="claude-sonnet-4-5-20250929")
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    messages=compressed.messages,
    max_tokens=1000,
)
```

### With OpenAI SDK

```python
from openai import OpenAI
from headroom import compress

client = OpenAI()
messages = [
    {"role": "user", "content": "Analyze these results"},
    {"role": "tool", "content": big_json_output, "tool_call_id": "call_1"},
]

compressed = compress(messages, model="gpt-4o")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=compressed.messages,
)
```

### With LiteLLM (direct)

```python
import litellm
from headroom import compress

messages = [...]
compressed = compress(messages, model="bedrock/claude-sonnet")
response = litellm.completion(model="bedrock/claude-sonnet", messages=compressed.messages)
```

### With any HTTP client

```python
import httpx
from headroom import compress

compressed = compress(messages, model="claude-sonnet-4-5-20250929")
httpx.post("https://api.anthropic.com/v1/messages", json={
    "model": "claude-sonnet-4-5-20250929",
    "messages": compressed.messages,
}, headers={"X-Api-Key": api_key, "anthropic-version": "2023-06-01"})
```

### What compress() returns

```python
result = compress(messages, model="gpt-4o")
result.messages           # list[dict] — compressed messages, same format as input
result.tokens_before      # int — original token count
result.tokens_after       # int — compressed token count
result.tokens_saved       # int — tokens removed
result.compression_ratio  # float — 0.0 (no savings) to 1.0 (100% removed)
result.transforms_applied # list[str] — what ran (e.g., ["router:smart_crusher:0.35"])
```

---

## LiteLLM

If you're already using LiteLLM as your LLM gateway, add Headroom as a callback:

```python
import litellm
from headroom.integrations.litellm_callback import HeadroomCallback

litellm.callbacks = [HeadroomCallback()]

# All calls now compressed automatically
response = litellm.completion(model="gpt-4o", messages=[...])
response = litellm.completion(model="bedrock/claude-sonnet", messages=[...])
response = litellm.completion(model="azure/gpt-4o", messages=[...])
```

The callback compresses messages in LiteLLM's `pre_call_hook` before they're sent to the provider. Works with all 100+ LiteLLM-supported providers.

### With LiteLLM Proxy

If you run LiteLLM as a proxy server, use the ASGI middleware instead:

```python
# In your LiteLLM proxy startup
from litellm.proxy.proxy_server import app
from headroom.integrations.asgi import CompressionMiddleware

app.add_middleware(CompressionMiddleware)
```

Or use the callback in your LiteLLM config:

```yaml
# litellm_config.yaml
litellm_settings:
  callbacks: ["headroom.integrations.litellm_callback.HeadroomCallback"]
```

---

## ASGI Middleware

Drop-in middleware for any ASGI application (FastAPI, Starlette, LiteLLM proxy, custom proxies).

```python
from headroom.integrations.asgi import CompressionMiddleware

# FastAPI
app = FastAPI()
app.add_middleware(CompressionMiddleware)

# Starlette
app = Starlette(routes=[...])
app.add_middleware(CompressionMiddleware)

# LiteLLM proxy
from litellm.proxy.proxy_server import app
app.add_middleware(CompressionMiddleware)
```

The middleware intercepts POST requests to `/v1/messages`, `/v1/chat/completions`, `/v1/responses`, and `/chat/completions`. All other requests pass through untouched.

Response headers include:
- `x-headroom-compressed: true` — compression was applied
- `x-headroom-tokens-saved: 1234` — tokens removed

---

## Proxy

The Headroom proxy is a standalone HTTP server. Best for non-Python apps or tools that only support base URL configuration (Claude Code, Cursor).

```bash
pip install "headroom-ai[all]"
headroom proxy --port 8787
```

```bash
# Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude

# Cursor / Any OpenAI client
OPENAI_BASE_URL=http://localhost:8787/v1 cursor
```

### With Cloud Providers

```bash
# AWS Bedrock
headroom proxy --backend bedrock --region us-east-1

# Google Vertex AI
headroom proxy --backend vertex_ai --region us-central1

# Azure OpenAI
headroom proxy --backend azure

# OpenRouter (400+ models)
OPENROUTER_API_KEY=sk-or-... headroom proxy --backend openrouter
```

See [Proxy Documentation](proxy.md) for all options.

---

## Agno

Full integration with the Agno agent framework.

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from headroom.integrations.agno import HeadroomAgnoModel

model = HeadroomAgnoModel(Claude(id="claude-sonnet-4-20250514"))
agent = Agent(model=model, tools=[your_tools])
response = agent.run("Investigate the issue")

print(f"Tokens saved: {model.total_tokens_saved}")
```

See [Agno Guide](agno.md) for hooks, multi-provider, and streaming.

---

## LangChain

> **Experimental.** Core compression works. Streaming callbacks and async chains are still being tested.

```python
from langchain_openai import ChatOpenAI
from headroom.integrations import HeadroomChatModel

llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))
response = llm.invoke("Hello!")
```

See [LangChain Guide](langchain.md) for details and known limitations.

---

## Compression Hooks (Advanced)

Customize compression behavior without modifying Headroom's code:

```python
from headroom import compress, CompressionHooks, CompressContext

class MyHooks(CompressionHooks):
    def pre_compress(self, messages, ctx):
        # Modify messages before compression (dedup, filter, inject)
        return messages

    def compute_biases(self, messages, ctx):
        # Per-message compression aggressiveness
        # >1.0 = keep more, <1.0 = compress more
        return {5: 1.5, 6: 0.5}  # Keep message 5, compress message 6

    def post_compress(self, event):
        # Observe results (logging, analytics, learning)
        print(f"Saved {event.tokens_saved} tokens")

result = compress(messages, model="gpt-4o", hooks=MyHooks())
```

See [Architecture](ARCHITECTURE.md) for how hooks integrate with the pipeline.

---

## FAQ

**Q: Does Headroom change the response format?**
No. Your LLM returns the same response format. Headroom only modifies the input messages.

**Q: What if compression removes something the LLM needs?**
Headroom stores originals in CCR (Compress-Cache-Retrieve). The LLM can call `headroom_retrieve` to get full uncompressed content. Compression summaries tell the LLM what's available.

**Q: Does it work with streaming?**
Yes. Compression happens before the request is sent. Streaming responses are unaffected.

**Q: How much latency does it add?**
1-5ms for compression. The token savings typically save more time on the LLM side than compression adds.
