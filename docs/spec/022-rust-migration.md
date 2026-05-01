# 022. Rust Migration

**Status:** in-progress (Stage 0 complete)

Headroom is evolving from a pure-Python proxy into a Rust engine with a Python SDK layer. This document describes the motivation, target architecture, and phased execution plan. It is the source of truth for the migration; the Discord announcement and contributor docs are derived from it.

## Why Rust

Three forces push us toward Rust:

1. **Latency.** Every LLM request flows through compression transforms on the hot path. Rust runs that math 2–10× faster than Python, with negligible interpreter overhead and near-instant cold starts.
2. **Deployment.** A Rust proxy is a single static binary (~10 MB). No Python interpreter, no pip, no wheel matrix, no model downloads at startup. It drops cleanly into containers, serverless runtimes, or bare hosts.
3. **Non-Python consumers.** Integrations like the TypeScript OpenClaw plugin talk to Headroom over HTTP. A faster proxy makes every downstream client faster without changing any client code.

The Python code is not being thrown away. Rust lives alongside Python in the same repository, the HTTP contract stays stable, and existing users see no breaking changes.

## Target architecture

Three Rust artifacts, built over time:

- **`headroom-proxy`** — a standalone Rust binary that speaks the existing Headroom HTTP API and contains native implementations of the proxy's hot-path logic (routing, streaming, compression, telemetry). This is the deployable artifact.
- **`headroom-core`** — a Rust library with the compression transforms (CCR, log compressor, diff compressor, tokenizer, code compressor, content router, etc.). Consumed by `headroom-proxy` directly; optionally exposed to Python via a PyO3 binding if/when embedded SDK use warrants it.
- **`headroom-runtime`** — a tiny zero-dependency contract crate that defines the fixed execution lifecycle, event model, and plugin surface shared by `headroom-proxy`, `headroom-core`, future persistence adapters, and enterprise extensions.

Both live in a Cargo workspace under `crates/` at the repo root. They are built, tested, and released together with the Python package.

### Runtime contract

The Rust rewrite standardizes on the same canonical lifecycle already exposed in Python:

`setup -> pre_start -> post_start -> input_received -> input_cached -> input_routed -> input_compressed -> input_remembered -> pre_send -> post_send -> response_received`

That sequence is fixed. Individual stages may be no-ops while a route is still in passthrough mode, but the lifecycle itself does not change. Every future Rust route, transform pipeline, telemetry exporter, storage adapter, and enterprise plugin should compose around that contract instead of inventing route-specific hooks.

## Migration strategy

**Trunk-based, no long-lived branch.** Every Rust change ships as a small PR to `main`. Rust and Python live side by side. CI enforces parity on every PR — if the Rust port diverges from the Python reference, the build fails.

**Proxy-first, not transforms-first.** We build the Rust proxy binary as the primary deliverable, starting from a pure HTTP passthrough that forwards upstream to the existing Python proxy. Native Rust implementations of individual routes then replace passthroughs one at a time, gated by feature flags. This lets us ship a deployable Rust binary on day one and iterate without modifying Python internals.

**Feature-flagged cutover.** Each native Rust route is toggled by config. Default is passthrough-to-Python until a route has been shadow-tested and validated. Rollback is a flag flip, not a redeploy.

**Parity by shadow traffic.** Recorded input/output fixtures give us a unit-test-level parity check. Shadow mode — running both proxies on live traffic and diffing outputs — gives us the real validation gate before any cutover.

## Stages

### Stage 0 — Foundation ✅

Cargo workspace with the initial Rust crates (`headroom-core`, `headroom-proxy`, `headroom-py`, `headroom-parity`), CI, build tooling (Makefile, GitHub Actions), and a parity test harness seeded with 125 recorded fixtures across 5 leaf transforms. No production behavior changed.

### Stage 1 — Rust proxy as passthrough

`headroom-proxy` accepts requests on the same HTTP contract as the Python proxy and forwards everything upstream to Python. No transforms yet, no intelligence. The point is a deployable binary that can run in front of the existing stack with zero risk.

**Current extension of Stage 1:** the passthrough proxy now emits the canonical runtime lifecycle through `headroom-runtime`, exposes native `/livez`, `/readyz`, `/health`, and `/metrics`, keeps `/healthz` plus `/healthz/upstream` for compatibility, and includes explicit native-route flags so cutovers stay opt-in. The branch now has native-route scaffolds for OpenAI chat, the OpenAI `/v1/responses` alias family, the OpenAI batch family (`/v1/batches`, `/v1/batches/{batch_id}`, `/v1/batches/{batch_id}/cancel`), provider-runtime-aware OpenAI-style model metadata routes (`/v1/models`, `/v1/models/{model_id}`), the OpenAI utility family (`/v1/embeddings`, `/v1/moderations`, `/v1/images/generations`, `/v1/audio/transcriptions`, `/v1/audio/speech`), Anthropic messages plus `count_tokens` and the Anthropic batch family (`/v1/messages/batches`, `/v1/messages/batches/{batch_id}`, `/v1/messages/batches/{batch_id}/results`, `/v1/messages/batches/{batch_id}/cancel`), the Databricks invocation route (`/serving-endpoints/{model}/invocations`), Gemini generateContent, the dedicated Gemini streamGenerateContent alias, Gemini countTokens, and the remaining Gemini passthrough family (`/v1beta/models`, `/v1beta/models/{model_name}`, `:embedContent`, `:batchEmbedContents`, `:batchGenerateContent`, `/v1beta/batches/{batch_name}`, `/v1beta/batches/{batch_name}:cancel`, `/v1beta/cachedContents`, `/v1beta/cachedContents/{cache_id}`), and now native-local telemetry endpoints (`/v1/telemetry`, `/v1/telemetry/export`, `/v1/telemetry/import`, `/v1/telemetry/tools`, `/v1/telemetry/tools/{signature_hash}`), native-local admin utility endpoints (`/stats`, `/stats-history`, `/debug/memory`, `/debug/tasks`, `/debug/ws-sessions`, `/debug/warmup`, `/dashboard`, `/transformations/feed`, `/subscription-window`, `/quota`, `/cache/clear`), native-local CCR/TOIN read endpoints (`/v1/retrieve*`, `/v1/feedback*`, `/v1/toin*`), and a native-local `/v1/compress` route that preserves the Python request/error contract while applying Rust-native token accounting plus a deeper router split across SmartCrusher for structured JSON arrays, DiffCompressor for unified diffs, SearchCompressor for grep-style results, LogCompressor for build/test output, TextCompressor for plain text, conservative HTML extraction for web-page payloads, mixed-content splitting across prose/code/search sections including unfenced source-code blocks, an in-process result/skip cache with `/cache/clear` integration, explicit protection markers for user/system/recent-code/analysis-context paths, Python-style detector thresholds plus metadata for the content classifier, Python-style `router:<strategy>:<ratio>` markers without leaking Rust-only cache-hit/internal markers into the client payload, a Python-matching 50-token route threshold on `/v1/compress`, request-body parity for `min_tokens_to_compress` and `compress_system_messages` by ignoring route-level overrides that Python does not forward, Python-style rolling-window budget enforcement for both explicit `token_budget` and default OpenAI model context limits, atomic tool-unit dropping plus Python-style dropped-context markers on over-budget requests, Python-matching success payload fields for bypass/empty/compressed responses including per-occurrence `transforms_applied` and counted-map `transforms_summary`, Anthropic count-token passthrough parity for non-standard `system` shapes, Gemini passthrough parity for missing `contents` on native generate/count-token routes, Anthropic passthrough parity for missing `model` / `max_tokens` / `messages` fields on native `/v1/messages`, request-level native response caching for buffered OpenAI chat and Anthropic messages, real request guardrails for max body size and oversized request arrays on native buffered routes, loopback-only protection for `/debug/memory` plus the full debug-introspection family, a bounded native request log surfaced through `/transformations/feed` and `/stats.recent_requests` for `/v1/compress`, buffered OpenAI/Anthropic routes, Gemini direct routes, Google Cloud Code direct routes, and the corresponding shadowed native provider executions, a native WebSocket session registry that feeds `/debug/ws-sessions`, `/debug/tasks`, and health runtime counters, a short-TTL `/stats?cached=1` dashboard snapshot path, telemetry-enabled reporting that now follows `HEADROOM_TELEMETRY` instead of a hardcoded enabled flag, request-log-derived `/stats` aggregation for real live token totals, cumulative savings history, display-session summaries, persistent-savings previews, and provider/model/stack request breakdowns, an in-memory `/stats-history` implementation with Python-style `history_mode` handling, hourly/daily/weekly/monthly rollups, and frontend-friendly CSV exports derived from the same native request-log history, semantic JSON/SSE shadow comparison that already tolerates equivalent OpenAI, Anthropic, Gemini, and Google Cloud Code streaming chunk boundaries, Python-matching OpenAI responses WebSocket defaults in Rust (`OpenAI-Beta` injection plus `OPENAI_API_KEY` fallback auth when the client omits `Authorization`), tighter Gemini system-field validation on native token-count routes, a native `headroom-core` CacheAligner with a stateful parity comparator and deterministic fixture ordering for chained prefix-hash metrics, Python-matching `LogCompressor` ratio accounting that preserves the pre-CCR-marker `compression_ratio`, a real built-in `ccr` parity comparator, provider-side request compression parity for direct OpenAI chat, direct Anthropic `/v1/messages`, Anthropic batch-create, Gemini `:generateContent`, Gemini `:countTokens`, and Google Cloud Code request bodies, direct OpenAI chat-completion batch-create orchestration that downloads batch JSONL from the OpenAI target, rewrites each line through the native compression pipeline, uploads the replacement file, and creates the batch against that new file id with Headroom compression metadata plus response savings headers, and the Google Cloud Code streaming aliases, with parity instrumentation layered on top. Health/config output now also publishes a machine-readable route manifest plus provider target settings (`OPENAI_TARGET_API_URL`, `ANTHROPIC_TARGET_API_URL`, `GEMINI_TARGET_API_URL`, `DATABRICKS_TARGET_API_URL`, `CLOUDCODE_TARGET_API_URL`) so operators can inspect the Rust-owned HTTP surface directly. Cache/memory are still explicit no-op stages in passthrough mode, but observability, policy, and plugin work now have one stable surface before deeper native compression pipeline work lands.

### Stage 2 — First native route

Replace the passthrough for `/v1/chat/completions` (OpenAI) with a native Rust implementation: Rust transforms, direct provider call, streamed response. Feature-flagged. The branch now includes native OpenAI chat with configurable direct OpenAI target selection (`OPENAI_TARGET_API_URL`, plus Azure-style `x-headroom-base-url` override when `api-key` is present), direct OpenAI execution for the `/v1/responses`, utility, batch, and model metadata families, direct Anthropic execution for `/v1/messages`, `/v1/messages/count_tokens`, and Anthropic batches via `ANTHROPIC_TARGET_API_URL`, direct Gemini execution for `:generateContent`, `:streamGenerateContent`, `:countTokens`, and the remaining Gemini passthrough family via `GEMINI_TARGET_API_URL`, direct Databricks execution for `/serving-endpoints/{model}/invocations` via `DATABRICKS_TARGET_API_URL`, direct Google Cloud Code execution for the streaming aliases via `CLOUDCODE_TARGET_API_URL` plus the antigravity sandbox override, native-local Rust telemetry for `/v1/telemetry*`, native-local Rust admin utility behavior for `/stats`, `/stats-history`, `/debug/memory`, `/transformations/feed`, `/subscription-window`, `/quota`, and `/cache/clear`, native-local Rust CCR/TOIN read behavior for `/v1/retrieve*`, `/v1/feedback*`, and `/v1/toin*`, and a native-local `/v1/compress` implementation built around Rust token counting, thresholded Python-style content detection, mixed-content splitting that now preserves both fenced and unfenced source blocks, an in-process compression cache, SmartCrusher for structured JSON, DiffCompressor for diffs, SearchCompressor for search results, LogCompressor for build/log output, TextCompressor for plain text, conservative HTML extraction for web-page payloads, conservative pinning rules for already-compressed or code-sensitive payloads, a Python-matching 50-token default threshold on the route, Python-matching omission of route-level `min_tokens_to_compress` and `compress_system_messages` overrides, Python-style rolling-window budget enforcement for both explicit `token_budget` and default OpenAI model context limits, atomic tool-unit dropping plus Python-style dropped-context markers on over-budget requests, Python-matching success payload fields for bypass/empty/compressed responses including counted `transforms_summary` and per-occurrence `transforms_applied`, Anthropic count-token passthrough parity for non-standard `system` shapes, Gemini generate/count-token passthrough parity for missing `contents`, Anthropic message-route passthrough parity for missing `model` / `max_tokens` / `messages` fields, request-level response caching for buffered OpenAI chat and Anthropic message executions, explicit request guardrails for body-size and oversized-array rejection, `/v1/responses` WebSocket auth/header parity, stricter Gemini system-field validation on native token-count routes, provider-side request compression parity for direct OpenAI chat, direct Anthropic `/v1/messages`, Anthropic batch-create, Gemini `:generateContent`, Gemini `:countTokens`, Google Cloud Code request bodies, and OpenAI chat-completion batch-create JSONL rewriting, app-scoped CCR-backed storage for native `/v1/compress` and provider-side request compression so emitted hashes are retrievable from the Rust `/v1/retrieve*` surface, real native feedback/TOIN learning that now updates tool-level retrieval rates, query/field hints, and TOIN pattern recommendations from Rust-owned compression/retrieval events, live native telemetry learning that now records the same Rust compression/retrieval activity surfaced by `/v1/telemetry*`, a native `CacheAligner` port plus stateful parity coverage for recorded prefix-hash chains, a real `log_compressor` parity comparator, a real `ccr` parity comparator, a bounded request-log/feed implementation that now reaches `/v1/compress`, buffered OpenAI/Anthropic, Gemini direct routes, Google Cloud Code direct routes, and their shadowed native executions, plus dashboard-oriented `/stats` parity for cached polling, telemetry enablement semantics, live request-log-backed token/request aggregation, restart-persistent request-log-backed savings/history state, restart-persistent product and telemetry stores for Rust-owned local surfaces, a first explicit local-state backend interface shared by request-log/product/telemetry stores with both file-backed and SQLite-backed implementations, surfaced backend metadata in native `/health` runtime payloads and `/stats`, relational SQLite schemas for request-log/product/telemetry persistence instead of a single opaque state blob per surface, request-log-derived overhead metrics, request-log-derived TTFB metrics on native buffered routes, list-price cost estimation for the common OpenAI/Anthropic/Gemini models already exercised here, Python-compatible `prefix_cache` response shaping, per-model cost breakdowns, and in-memory `/stats-history` rollups/exports. The remaining parity work is now concentrated in the Stage 6 storage layer, the remaining native learning/persistence surfaces, and the final Python-retirement blockers rather than route ownership, provider forwarding, or the `/v1/compress` router contract.

### Stage 3 — Shadow mode validation

Run both proxies on real traffic. Diff outputs with tolerance for chunk timing (SSE). The branch now includes local shadow-comparison mechanics for streamed and non-streamed OpenAI, Anthropic, Gemini generateContent, Gemini streamGenerateContent, Gemini countTokens, and Google Cloud Code streaming alias requests, plus semantic JSON/SSE comparison that already tolerates equivalent OpenAI, Anthropic, Gemini, and Google Cloud Code streaming chunk boundaries instead of requiring raw byte equality. Gate: one week of ≤ 0.1% content divergence before flipping the flag for real.

### Stage 4 — Provider expansion

Anthropic, Google, Cohere, Mistral, Bedrock, others. Each provider goes through its own shadow period before cutover.

### Stage 5 — Code-aware transforms

Port `code_compressor` (tree-sitter, already Rust-native upstream), `content_router`, `content_detector`. These are larger transforms but have no ML dependencies.

### Stage 6 — Storage layer

SQLite (`rusqlite` + `sqlite-vec`), HNSW vector index (`instant-distance`), graph store (`neo4rs`). Feature-flagged backend for the memory subsystem. The first storage prerequisites are now in place: Rust-owned local state already sits behind an explicit backend boundary, the proxy can switch request-log/product/telemetry persistence onto a shared SQLite backend when `savings_path` points at a `.db` / `.sqlite` / `.sqlite3` file, those three local-state surfaces now persist through dedicated SQLite tables rather than one opaque blob per surface, the request-log/telemetry/product read surfaces now query those normalized SQLite tables directly instead of relying only on boot-time memory snapshots, and the product hot path now refreshes from SQLite before live retrieval/compression mutations so externally persisted CCR rows can still participate in native learning and retrieval flows. The next Stage 6 work is to reduce the remaining JSON-valued relational columns and eventually extend the storage layer into vector/graph backends rather than stopping at SQLite-backed local state.

### Stage 7 — ONNX migration

Remove the torch-dependent LLMLingua compressor. Convert remaining ML models (SmartCrusher, IntelligentContext, memory embedders) to ONNX with fixed opset. Run via the `ort` crate. Remove `torch`, `transformers`, `sentence-transformers`, `llmlingua` from runtime dependencies.

### Stage 8 — Retire the Python proxy

Delete `headroom/proxy/server.py` and the Python HTTP routes. The Python package (`import headroom`) continues to exist for SDK users and integration adapters (LangChain, Agno, MCP, Strands), which talk HTTP and don't care which proxy implementation is behind the endpoint.

## What does not change

- `pip install headroom` continues to work.
- The HTTP API contract is preserved.
- LangChain, Agno, MCP, Strands integrations keep working unchanged.
- The CLI, dashboard, eval framework, examples, and documentation site are all unaffected.
- Python contributions remain welcome for integrations, tooling, examples, and any code paths not yet ported.

## Contributing

- **Python contributors** do not need to learn Rust. CI will flag a PR if a Python change diverges from a Rust-ported counterpart; in that case either update both implementations or disable the feature flag for the affected route.
- **Rust contributors** should read `RUST_DEV.md` for workspace setup, then pick a transform or proxy route from the open issues.

## Risks and open questions

- **ONNX export parity** for embedding models: numerical reproducibility must be validated per model before cutover. Some models may resist clean export and require keeping a Python inference path behind PyO3 as a fallback.
- **LLMLingua removal** is a feature removal visible to users relying on it; deprecation timing will be announced on Discord before Stage 7 begins.
- **PyO3 binding scope**: we may ultimately not need a PyO3-exposed `headroom._core` at all, if existing Python SDK users are happy with the HTTP contract. Decision deferred to Stage 8.

## References

- `RUST_DEV.md` — developer setup and workspace reference
- `crates/` — Rust sources
- `tests/parity/` — fixtures and parity harness
- `Makefile` — `make test`, `make test-parity`, `make build-proxy`, `make build-wheel`, `make fmt`, `make lint`
