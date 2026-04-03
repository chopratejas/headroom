# @headroom-ai/openclaw

Context compression plugin for [OpenClaw](https://github.com/openclaw/openclaw). Compresses tool outputs, code, logs, and structured data — 70-90% token savings with zero LLM calls.

## Install

```bash
pip install "headroom-ai[proxy]"
openclaw plugins install --dangerously-force-unsafe-install headroom-ai/openclaw
```

This plugin auto-starts `headroom proxy` when needed. OpenClaw treats process-launching plugins as unsafe by default, so `--dangerously-force-unsafe-install` is required.

## Local Development Install (Detection-Friendly)

If you are testing from this repo, run npm install/build from the plugin directory so local launcher detection aligns with runtime paths:

```bash
cd plugins/openclaw
npm install
npm run build
openclaw plugins install --dangerously-force-unsafe-install --link .
```

Why this matters:
- The plugin checks launchers in this order: PATH -> local npm bin -> global npm -> python.
- "local npm bin" means `plugins/openclaw/node_modules/.bin/headroom` relative to the installed plugin root.
- Using `--link .` from `plugins/openclaw` keeps that local path aligned for detection.
- If you install from a `.tgz`, local npm bin may not exist in the installed extension and detection will fall back to PATH/global/python.

## Configure

```json
{
  "plugins": {
    "entries": {
      "headroom": {
        "enabled": true,
        "config": {
          "proxyUrl": "http://127.0.0.1:8787"
        }
      }
    },
    "slots": {
      "contextEngine": "headroom"
    }
  }
}
```

`proxyUrl` must be localhost (`127.0.0.1` or `localhost`). If the proxy is not running, the plugin will try to start it with `headroom proxy --host ... --port ...`.
`proxyUrl` is optional. If omitted, the plugin auto-detects on:
- `http://127.0.0.1:<proxyPort>`
- `http://localhost:<proxyPort>`
Default `proxyPort` is `8787`.

If set, `proxyUrl` must be localhost (`127.0.0.1` or `localhost`).
Auto-start launch order is:
1. `headroom` from `PATH`
2. local npm bin (`node_modules/.bin/headroom`)
3. global npm bin
4. Python module (`python -m headroom.cli proxy ...`)
If `pythonPath` is set, it is tried first in the Python fallback step.

## Required Proxy Setup

Optional: run Headroom proxy yourself before launching OpenClaw.

Python install:

```bash
pip install "headroom-ai[proxy]"
headroom proxy --host 127.0.0.1 --port 8787
```

NPM install:

```bash
npm install -g headroom-ai
headroom proxy --host 127.0.0.1 --port 8787
```

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
| `proxyUrl` | auto-detected | Optional URL of a Headroom proxy (`http://127.0.0.1:<port>` or `http://localhost:<port>`). |
| `proxyPort` | `8787` | Port used for default auto-detect/auto-start when `proxyUrl` is not set. |
| `pythonPath` | auto-detected | Optional Python executable override for Python fallback launcher. |
| `autoStart` | `true` | Auto-start `headroom proxy` if not already running |
| `startupTimeoutMs` | `20000` | Time to wait for auto-started proxy to become healthy |

## Comparison with lossless-claw

| | lossless-claw | headroom |
|---|---|---|
| Compaction method | LLM summarization (DAG) | Content-aware compression (zero LLM) |
| Cost of compaction | Tokens (LLM calls) | Zero |
| Best for | Long conversations | Tool-heavy agents with large outputs |
| Retrieval | `lcm_grep`, `lcm_expand` | `headroom_retrieve` (instant) |

## License

Apache-2.0
