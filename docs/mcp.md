# MCP Server for Claude Code Subscriptions

Headroom's MCP (Model Context Protocol) server enables **CCR (Compress-Cache-Retrieve)** for Claude Code subscription users who don't have direct API access.

## Quick Start

```bash
# Install MCP dependencies
pip install "headroom-ai[mcp]"

# Configure Claude Code (one-time)
headroom mcp install

# Start the proxy
headroom proxy

# Use Claude Code - it now has headroom_retrieve!
claude
```

## Why MCP?

| Authentication | Custom Tools | Solution |
|----------------|--------------|----------|
| **API Key** | Direct injection via Messages API | Works automatically |
| **Subscription** | Claude Code's built-in tools only | MCP server |

Claude Code subscription users can't inject custom tools programmatically. MCP is Claude's official extension mechanism that works with subscriptions.

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Claude Code   │────▶│  Headroom Proxy │────▶│   LLM Provider  │
└────────┬────────┘     └────────┬────────┘     └─────────────────┘
         │                       │
         │ MCP                   │ Stores compressed
         │                       │ content
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   MCP Server    │◀───▶│ Compression     │
│ (headroom_      │     │ Store           │
│  retrieve)      │     └─────────────────┘
└─────────────────┘
```

1. **Proxy compresses** large tool outputs (file listings, search results, logs)
2. **Claude sees** compressed summaries with hash markers: `[47 items compressed... hash=abc123]`
3. **When needed**, Claude calls `headroom_retrieve` to get the original content
4. **MCP server** fetches from the proxy's compression store

## CLI Commands

### Install MCP Configuration

```bash
headroom mcp install
```

This writes to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "headroom": {
      "command": "headroom",
      "args": ["mcp", "serve"]
    }
  }
}
```

Options:
- `--proxy-url URL` - Custom proxy URL (default: `http://127.0.0.1:8787`)
- `--force` - Overwrite existing configuration

### Check Status

```bash
headroom mcp status
```

Shows:
- MCP SDK installation status
- Claude Code configuration status
- Proxy connectivity

Example output:
```
Headroom MCP Status
========================================
MCP SDK:        ✓ Installed
Claude Config:  ✓ Configured
                /Users/you/.claude/mcp.json
Proxy URL:      http://127.0.0.1:8787
Proxy Status:   ✓ Running at http://127.0.0.1:8787
```

### Uninstall

```bash
headroom mcp uninstall
```

Removes headroom from `~/.claude/mcp.json` while preserving other MCP servers.

### Manual Server Start

```bash
headroom mcp serve
```

This is called by Claude Code automatically. For debugging:

```bash
headroom mcp serve --debug
```

## The headroom_retrieve Tool

When the MCP server is active, Claude has access to:

```
Tool: headroom_retrieve

Parameters:
  - hash (required): Hash key from compression marker
  - query (optional): Search query to filter results

Returns:
  - Full original content, or
  - Filtered results matching query
```

Example interaction:

```
Claude sees:
  [47 log entries compressed. Showing first 3 + anomalies.
   Use headroom_retrieve(hash="a1b2c3") for full logs]

Claude calls:
  headroom_retrieve(hash="a1b2c3", query="error")

Returns:
  [All log entries containing "error"]
```

## Custom Proxy URL

If your proxy runs on a different port:

```bash
# During install
headroom mcp install --proxy-url http://localhost:9000

# Or via environment variable
export HEADROOM_PROXY_URL=http://localhost:9000
headroom mcp serve
```

## Troubleshooting

### "MCP SDK not installed"

```bash
pip install "headroom-ai[mcp]"
```

### "Proxy not running"

Start the proxy in another terminal:

```bash
headroom proxy
```

### "Entry not found or expired"

Compressed entries expire after 5 minutes (TTL). The proxy must be running continuously during your session.

### Claude doesn't see headroom_retrieve

1. Check status: `headroom mcp status`
2. Restart Claude Code after installing MCP
3. Verify `~/.claude/mcp.json` exists and contains headroom

## API Users

If you have an `ANTHROPIC_API_KEY`, you don't need MCP. The proxy automatically injects the `headroom_retrieve` tool into API requests.

MCP is specifically for subscription users who authenticate via Claude Code's OAuth flow rather than an API key.
