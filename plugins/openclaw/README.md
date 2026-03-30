# @headroom-ai/openclaw

Context compression plugin for [OpenClaw](https://github.com/openclaw/openclaw). Compresses tool outputs, code, logs, and structured data — 70-90% token savings with zero LLM calls.

## Install

```bash
pip install "headroom-ai[proxy]"
openclaw plugins install @headroom-ai/openclaw
```

## Configure

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "headroom"
    }
  }
}
```

That's it. The plugin auto-starts the Headroom proxy if it's not already running.

## How It Works

Every time OpenClaw assembles context for the model, the plugin compresses tool outputs and large messages:

- **JSON arrays** (tool outputs, search results) — statistical selection keeps anomalies, errors, boundaries
- **Code** — AST-aware compression via tree-sitter
- **Logs** — pattern deduplication, keeps errors and boundaries
- **Text** — ML-based token compression

Compression is lossless via CCR (Compress-Cache-Retrieve): originals are stored and the agent gets a `headroom_retrieve` tool to access full details when needed.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `proxyUrl` | auto-detected | URL of the Headroom proxy |
| `autoStart` | `true` | Start proxy automatically if not running |
| `pythonPath` | auto-detected | Path to Python binary |
| `proxyPort` | `8787` | Port for auto-started proxy |

## Comparison with lossless-claw

| | lossless-claw | headroom |
|---|---|---|
| Compaction method | LLM summarization (DAG) | Content-aware compression (zero LLM) |
| Cost of compaction | Tokens (LLM calls) | Zero |
| Best for | Long conversations | Tool-heavy agents with large outputs |
| Retrieval | `lcm_grep`, `lcm_expand` | `headroom_retrieve` (instant) |

## License

Apache-2.0
