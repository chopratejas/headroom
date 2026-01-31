# Strands SDK + Headroom Demo

This directory contains an example showing how to use **Headroom** with **AWS Strands Agents SDK** to automatically compress tool outputs and reduce token costs.

## Installation

```bash
# Install Headroom with Strands support
pip install "headroom-ai[strands]"

# Set your API key
export OPENAI_API_KEY="your-api-key"
```

## Running the Demo

```bash
python run_openai_agent.py
```

The demo has **two parts**:

### Part 1: Before/After Compression Visualization

Shows exactly what Headroom's compression does to tool output:

```
┌────────────────────────────────────────────────────────────────────┐
│ BEFORE COMPRESSION (first 800 chars of server logs)               │
├────────────────────────────────────────────────────────────────────┤
│ [                                                                  │
│   {                                                                │
│     "timestamp": "2025-01-30T10:00:00Z",                           │
│     "level": "INFO",                                               │
│     "service": "prod-server-1",                                    │
│     "message": "Request processed successfully - latency=50ms",    │
│     ...                                                            │
│   },                                                               │
│   ... (20 entries)                                                 │
└────────────────────────────────────────────────────────────────────┘
   📊 Tokens: 2142

┌────────────────────────────────────────────────────────────────────┐
│ AFTER COMPRESSION (output)                                        │
├────────────────────────────────────────────────────────────────────┤
│ [                                                                  │
│   { "timestamp": "2025-01-30T10:00:00Z", ... },  <- first items    │
│   { "timestamp": "2025-01-30T10:07:00Z", ... },  <- ERROR kept!    │
│   { "timestamp": "2025-01-30T10:17:00Z", ... },  <- ERROR kept!    │
│   ... (10 items total)                                             │
│ ]                                                                  │
│ 📝 Items kept: 10 (from 20 original)                                │
└────────────────────────────────────────────────────────────────────┘
   📊 Tokens: 803

┌────────────────────────────────────────────────────────────────────┐
│ COMPRESSION RESULTS                                               │
├────────────────────────────────────────────────────────────────────┤
│   Before:  2142 tokens                                             │
│   After:    803 tokens                                             │
│   Saved:   1339 tokens (62.5%)                                     │
│                                                                    │
│   ✅ Error messages preserved                                      │
│   ✅ Timestamps preserved                                          │
│   ✅ Redundant entries removed                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Part 2: Live Agent with Headroom Hooks

A DevOps assistant agent with three verbose tools:
- `search_documentation` - Returns 20 search results
- `get_server_logs` - Returns 50 log entries  
- `get_metrics` - Returns 60 data points

Headroom's `HeadroomHookProvider` automatically compresses these outputs before they're added to context.

## How It Works

Headroom integrates with Strands SDK using the **HookProvider pattern**:

```python
from strands import Agent
from strands.models.openai import OpenAIModel
from headroom.integrations.strands import HeadroomHookProvider

# Create Headroom hook provider
headroom_hook = HeadroomHookProvider(
    compress_tool_outputs=True,
    min_tokens_to_compress=100,
)

# Create agent with hooks
agent = Agent(
    model=OpenAIModel(model_id="gpt-4o-mini"),
    tools=[my_tool],
    hooks=[headroom_hook],  # Strands calls register_hooks() automatically
)

# Run agent - tool outputs are compressed automatically
result = agent("Your query here")

# Check savings
print(f"Tokens saved: {headroom_hook.total_tokens_saved}")
```

## Hooks Registered

Headroom registers callbacks for these Strands lifecycle events:

| Event | What Headroom Does |
|-------|-------------------|
| `BeforeInvocationEvent` | Track invocation start |
| `AfterInvocationEvent` | Track invocation end |
| `BeforeToolCallEvent` | Log tool start |
| `AfterToolCallEvent` | **Compress tool output** ⭐ |
| `BeforeModelCallEvent` | Check token alert threshold |
| `AfterModelCallEvent` | Track response |

## Expected Savings

With verbose tool outputs (JSON search results, logs, metrics):

| Scenario | Typical Savings |
|----------|----------------|
| Search results (20 items) | 50-60% |
| Server logs (50 entries) | 80-85% |
| Time-series metrics | 35-40% |

> **Note:** Actual savings vary based on data redundancy. Log entries with repeated patterns compress better than unique metric data points.

## Reference

- [Strands SDK Hooks Documentation](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/hooks/)
- [Headroom Documentation](https://github.com/chopratejas/headroom)
