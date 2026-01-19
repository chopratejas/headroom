# Headroom Documentation

Welcome to the Headroom documentation.

## Getting Started

| Guide | Description |
|-------|-------------|
| [Quickstart](quickstart.md) | 5-minute setup |
| [SDK Guide](sdk.md) | Python SDK usage |
| [Proxy Guide](proxy.md) | Proxy server deployment |

## Framework Integrations

| Framework | Description |
|-----------|-------------|
| [LangChain](langchain.md) | Chat models, memory, retrievers, agents, streaming |
| [Agno](agno.md) | Model wrapper, hooks, multi-provider support |
| MCP | See [CCR Guide](ccr.md) for tool compression |

## Core Concepts

| Topic | Description |
|-------|-------------|
| [Universal Compression](compression.md) | ML-based content detection + structure preservation |
| [Transforms](transforms.md) | How compression works |
| [CCR](ccr.md) | Reversible compression architecture |
| [Configuration](configuration.md) | All configuration options |

## Advanced

| Topic | Description |
|-------|-------------|
| [Text Compression](text-compression.md) | Opt-in utilities for search/logs |
| [LLMLingua](llmlingua.md) | ML-based compression |
| [Metrics](metrics.md) | Monitoring and observability |
| [Errors](errors.md) | Error handling |

## Deployment & Operations

| Guide | Description |
|-------|-------------|
| [macOS Deployment](macos-deployment.md) | Run proxy as background service on macOS |

## Reference

| Topic | Description |
|-------|-------------|
| [API Reference](api.md) | Complete API docs |
| [Architecture](ARCHITECTURE.md) | Internal design |
| [Troubleshooting](troubleshooting.md) | Common issues |

## Overview

Headroom is the Context Optimization Layer for LLM applications. It reduces your LLM costs by 50-90% through intelligent context compression.

### How It Works

1. **Universal Compression** — ML-based content detection with structure-preserving compression
2. **SmartCrusher** — Compresses JSON tool outputs, keeping errors, anomalies, and relevant items
3. **CacheAligner** — Stabilizes message prefixes so provider caching works
4. **RollingWindow** — Manages context limits without breaking tool call pairs
5. **CCR** — Caches original data so compression is reversible

### Safety Guarantees

- Never removes human content
- Never breaks tool call ordering
- Parse failures pass through unchanged
- LLM can always retrieve original data

### Getting Help

- [GitHub Issues](https://github.com/chopratejas/headroom/issues) — Bug reports
- [GitHub Discussions](https://github.com/chopratejas/headroom/discussions) — Questions
