# Changelog

All notable changes to Headroom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Telemetry stack & install-mode identity fields** — anonymous beacon now
  reports `headroom_stack` (how Headroom is invoked: `proxy`, `wrap_claude`,
  `adapter_ts_openai`, ...) and `install_mode` (`wrapped` / `persistent` /
  `on_demand`), plus `requests_by_stack` for proxies that serve multiple
  integrations. Proxy exposes a `by_stack` bucket alongside `by_provider` /
  `by_model` on `/stats`, a matching `headroom_requests_by_stack` Prometheus
  counter, and an `X-Headroom-Stack` header honored by the FastAPI middleware.
  `headroom wrap <tool>` sets `HEADROOM_STACK=wrap_<agent>`; the TS SDK and
  all four adapters (`openai`, `anthropic`, `gemini`, `vercel-ai`) tag their
  compress calls. Schema migration:
  [`sql/upgrade_telemetry_stack_context.sql`](sql/upgrade_telemetry_stack_context.sql).
- **Canonical filesystem contract** (issue #175) — new `HEADROOM_CONFIG_DIR`
  (default `~/.headroom/config`, read-mostly) and `HEADROOM_WORKSPACE_DIR`
  (default `~/.headroom`, read-write state) env vars recognized by the Python
  proxy/CLI and the npm SDK. Additive; all existing per-resource env vars
  (`HEADROOM_SAVINGS_PATH`, `HEADROOM_TOIN_PATH`,
  `HEADROOM_SUBSCRIPTION_STATE_PATH`, `HEADROOM_MODEL_LIMITS`) continue to
  work with identical semantics. Docker install scripts and
  `docker-compose.native.yml` forward the new vars into containers so
  savings, logs, and telemetry resolve to the bind-mounted `.headroom` path.
  See [`wiki/filesystem-contract.md`](wiki/filesystem-contract.md).

## [0.5.22] - 2026-04-11

### Added
- **Cross-agent memory** — Claude saves a fact, Codex reads it back. All agents sharing one proxy share one memory store. Project-scoped DB at `.headroom/memory.db`, auto user_id from `$USER`.
- **Agent provenance tracking** — every memory records which agent saved it (`source_agent`, `source_provider`, `created_via`), with edit history on updates.
- **LLM-mediated dedup** — on `memory_save`, enriched response hints similar existing memories to the LLM. Background async dedup auto-removes >92% cosine duplicates. Zero extra LLM calls.
- **Memory for OpenAI and Gemini handlers** — context injection + tool handling wired into all three provider handlers (Anthropic, OpenAI, Gemini).
- **Plugin architecture for `headroom learn`** — each agent (Claude, Codex, Gemini) is a self-contained plugin. External plugins register via `headroom.learn_plugin` entry points. `--agent` flag for CLI.
- **GeminiScanner** for `headroom learn` — reads `~/.gemini/tmp/*/chats/session-*.json` and `.jsonl`.
- **Code graph integration** — `headroom wrap claude --code-graph` auto-indexes the project via [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) for call-chain traversal, impact analysis, and architectural queries. Opt-in, ~200 token overhead with Claude Code's MCP Tool Search.
- **OpenAI embedder auto-detection** — memory backend uses OpenAI embeddings when `sentence-transformers` is unavailable (no torch/2GB dependency needed).
- **Live traffic learning flush** — `headroom wrap <agent> --learn` flushes learned patterns to the correct agent-native file (MEMORY.md / AGENTS.md / GEMINI.md) at proxy shutdown.

### Changed
- **CodeCompressor disabled by default** — AST-based code compression produced invalid syntax on 40% of real files. Code now passes through uncompressed. Use `--code-graph` for code intelligence instead, or re-enable with `--code-aware`.
- **Shared tool name map** — consolidated tool normalization across all learn plugins into `_shared.py`.
- **Dynamic CLI agent detection** — `headroom learn` discovers agents via plugin registry, no hardcoded choices.

### Fixed
- **CodeCompressor statement-based truncation** — body truncation now walks AST statements (not lines), never cuts mid-expression. Fixes syntax errors on multi-line dict literals and function calls.
- **Docstring FIRST_LINE mode** — uses source lines directly instead of reconstructing from byte offsets. Properly handles all quote styles.
- **Memory shutdown queue drain** — patterns in the save queue were lost on proxy shutdown. Now drained before exit.

## [Unreleased]

### Added
- **Proxy liveness and readiness health checks**
  - Adds `GET /livez` for process liveness and `GET /readyz` for traffic readiness
  - Keeps `GET /health` backward compatible while expanding it with readiness details and subsystem checks
  - Eagerly initializes configured memory backends during proxy startup so readiness reflects real serving capability
  - Wires `/readyz` into the Docker image `HEALTHCHECK` and the example `docker-compose.yml`
- **Durable proxy savings history**
  - Persists proxy compression savings history locally at `~/.headroom/proxy_savings.json`
  - Supports `HEADROOM_SAVINGS_PATH` to override the storage location
  - Adds `/stats-history` with lifetime totals plus hourly/daily/weekly/monthly rollups
  - Supports JSON and CSV export from `/stats-history`
  - Extends `/stats` with a `persistent_savings` block while keeping `savings_history` backward compatible
  - Adds a historical mode to `/dashboard` backed by `/stats-history`, including export actions
- **Proxy telemetry SDK override** via `HEADROOM_SDK`
  - Downstream apps can override the anonymous telemetry `sdk` field without patching installed files
  - Blank values fall back to the default `proxy` label
- **`headroom learn`** — Offline failure learning for coding agents
  - Analyzes past conversation history (Claude Code, extensible to Cursor/Codex)
  - **Success correlation**: for each failure, finds what succeeded after and extracts the specific correction
  - 5 analyzers: Environment, Structure, Command Patterns, Retry Prevention, Cross-Session
  - Writes specific learnings to CLAUDE.md (stable project facts) and MEMORY.md (session patterns)
  - Generic architecture: tool-agnostic `ToolCall` model, pluggable Scanner/Writer adapters
  - Dry-run by default, `--apply` to write, `--all` for all projects
  - Example output: "FirstClassEntity.java is not at axion-formats/ — actually at axion-scala-common/"
- **Read Lifecycle Management** — Event-driven compression of stale/superseded Read outputs
  - Detects when a Read output becomes stale (file was edited after) or superseded (file was re-read)
  - Replaces stale/superseded content with compact CCR markers, stores originals for retrieval
  - 75% of Read output bytes are provably stale or redundant (from real-world analysis of 66K tool calls)
  - Fresh Reads (latest read, no subsequent edit) are never touched — Edit safety preserved
  - Opt-in via `ReadLifecycleConfig(enabled=True)`, disabled by default
  - Handles both OpenAI and Anthropic message formats
- **any-llm backend** - Route requests through 38+ LLM providers (OpenAI, Mistral, Groq, Ollama, etc.) via [any-llm](https://mozilla-ai.github.io/any-llm/providers/)
  - Enable with `--backend anyllm --anyllm-provider <provider>`
  - Install with: `pip install 'headroom-ai[anyllm]'`
- Production-ready proxy server with caching, rate limiting, and metrics
- CLI command `headroom proxy` to start the proxy server
- **IntelligentContextManager** (semantic-aware context management)
  - Multi-factor importance scoring: recency, semantic similarity, TOIN importance, error indicators, forward references, token density
  - No hardcoded patterns - all importance signals learned from TOIN or computed from metrics
  - TOIN integration for retrieval_rate and field_semantics-based scoring
  - Strategy selection: NONE, COMPRESS_FIRST, DROP_BY_SCORE based on budget overage
  - Atomic tool unit handling (call + response dropped together)
  - Configurable scoring weights via `ScoringWeights` dataclass
  - `IntelligentContextConfig` for full configuration control
  - Backwards compatible with `RollingWindowConfig`
- **LLMLingua-2 Integration** (opt-in ML-based compression)
  - `LLMLinguaCompressor` transform using Microsoft's LLMLingua-2 model
  - Content-aware compression rates (code: 0.4, JSON: 0.35, text: 0.3)
  - Memory management utilities: `unload_llmlingua_model()`, `is_llmlingua_model_loaded()`
  - Proxy integration via `--llmlingua` flag
  - Device selection: `--llmlingua-device` (auto/cuda/cpu/mps)
  - Custom compression rate: `--llmlingua-rate`
  - Helpful startup hints when llmlingua is available but not enabled
  - Install with: `pip install headroom-ai[llmlingua]`
- **Code-Aware Compression** (AST-based, syntax-preserving)
  - `CodeAwareCompressor` transform using tree-sitter for AST parsing
  - Supports Python, JavaScript, TypeScript, Go, Rust, Java, C, C++
  - Preserves imports, function signatures, type annotations, error handlers
  - Compresses function bodies while maintaining structural integrity
  - Guarantees syntactically valid output (no broken code)
  - Automatic language detection from code patterns
  - Memory management: `is_tree_sitter_available()`, `unload_tree_sitter()`
  - Uses `tree-sitter-language-pack` for broad language support
  - Install with: `pip install headroom-ai[code]`
- **ContentRouter** (intelligent compression orchestrator)
  - Auto-routes content to optimal compressor based on type detection
  - Source hint support for high-confidence routing (file paths, tool names)
  - Handles mixed content (e.g., markdown with code blocks)
  - Strategies: CODE_AWARE, SMART_CRUSHER, SEARCH, LOG, TEXT, LLMLINGUA
  - Configurable strategy preferences and fallbacks
  - Routing decision log for transparency and debugging
- **Custom Model Configuration**
  - Support for new models: Claude 4.5 (Opus), Claude 4 (Sonnet, Haiku), o3, o3-mini
  - Pattern-based inference for unknown models (opus/sonnet/haiku tiers)
  - Custom model config via `HEADROOM_MODEL_LIMITS` environment variable
  - Config file support: `~/.headroom/models.json`
  - Graceful fallback for unknown models (no crashes)
  - Updated pricing data for all current models

## [0.2.0] - 2025-01-07

### Added
- **SmartCrusher**: Statistical compression for tool outputs
  - Keeps first/last K items, errors, anomalies, and relevance matches
  - Variance-based change point detection
  - Pattern detection (time series, logs, search results)
- **Relevance Scoring Engine**: ML-powered item relevance
  - `BM25Scorer`: Fast keyword matching (zero dependencies)
  - `EmbeddingScorer`: Semantic similarity with sentence-transformers
  - `HybridScorer`: Adaptive combination of both methods
- **CacheAligner**: Prefix stabilization for better cache hits
  - Dynamic date extraction
  - Whitespace normalization
  - Stable prefix hashing
- **RollingWindow**: Context management within token limits
  - Drops oldest tool units first
  - Never orphans tool results
  - Preserves recent turns
- **Multi-Provider Support**:
  - Anthropic with official `count_tokens` API
  - Google with official `countTokens` API
  - Cohere with official `tokenize` API
  - Mistral with official tokenizer
  - LiteLLM for unified interface
- **Integrations**:
  - LangChain callback handler (`HeadroomOptimizer`)
  - MCP (Model Context Protocol) utilities
- **Proxy Server** (`headroom.proxy`):
  - Semantic caching with LRU eviction
  - Token bucket rate limiting
  - Retry with exponential backoff
  - Cost tracking with budget enforcement
  - Prometheus metrics endpoint
  - Request logging (JSONL)
- **Pricing Registry**: Centralized model pricing with staleness tracking
- **Benchmarks**: Performance benchmarks for transforms and relevance scoring

### Changed
- Improved token counting accuracy across all providers
- Enhanced tool output compression with relevance-aware selection

### Fixed
- Mistral tokenizer API compatibility
- Google token counting for multi-turn conversations

## [0.1.0] - 2025-01-05

### Added
- Initial release
- `HeadroomClient`: OpenAI-compatible client wrapper
- `ToolCrusher`: Basic tool output compression
- Audit mode for observation without modification
- Optimize mode for applying transforms
- Simulate mode for previewing changes
- SQLite and JSONL storage backends
- HTML report generation
- Streaming support

### Safety Guarantees
- Never removes human content
- Never breaks tool ordering
- Parse failures are no-ops
- Preserves recency (last N turns)

---

## Migration Guide

### From 0.1.x to 0.2.x

The 0.2.0 release is backward compatible. New features are opt-in:

```python
# Old code still works
from headroom import HeadroomClient, OpenAIProvider

# New SmartCrusher (replaces ToolCrusher for better compression)
from headroom import SmartCrusher, SmartCrusherConfig

config = SmartCrusherConfig(
    min_tokens_to_crush=200,
    max_items_after_crush=50,
)
crusher = SmartCrusher(config)

# New relevance scoring
from headroom import create_scorer

scorer = create_scorer("hybrid")  # or "bm25" for zero deps
```

### Using the Proxy

New in 0.2.0 - run Headroom as a proxy server:

```bash
# Start the proxy
python -m headroom.proxy.server --port 8787

# Use with Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

[Unreleased]: https://github.com/headroom-sdk/headroom/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/headroom-sdk/headroom/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/headroom-sdk/headroom/releases/tag/v0.1.0
