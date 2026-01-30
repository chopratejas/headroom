"""Headroom Proxy Server - Production Ready.

A full-featured LLM proxy with optimization, caching, rate limiting,
and observability.

Features:
- Context optimization (SmartCrusher, CacheAligner, RollingWindow)
- Semantic caching (save costs on repeated queries)
- Rate limiting (token bucket)
- Retry with exponential backoff
- Cost tracking and budgets
- Request tagging and metadata
- Provider fallback
- Prometheus metrics
- Full request/response logging

Usage:
    python -m headroom.proxy.server --port 8787

    # With Claude Code:
    ANTHROPIC_BASE_URL=http://localhost:8787 claude
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from headroom.backends import LiteLLMBackend
from headroom.backends.base import Backend
from headroom.cache.compression_feedback import get_compression_feedback
from headroom.cache.compression_store import get_compression_store
from headroom.ccr import (
    CCR_TOOL_NAME,
    # Batch processing
    BatchContext,
    BatchRequestContext,
    BatchResultProcessor,
    CCRResponseHandler,
    CCRToolInjector,
    ContextTracker,
    ContextTrackerConfig,
    ResponseHandlerConfig,
    get_batch_context_store,
    parse_tool_call,
)
from headroom.config import (
    CacheAlignerConfig,
    CCRConfig,
    IntelligentContextConfig,
    RollingWindowConfig,
    SmartCrusherConfig,
)
from headroom.providers import AnthropicProvider, OpenAIProvider
from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler
from headroom.telemetry import get_telemetry_collector
from headroom.telemetry.toin import get_toin
from headroom.tokenizers import get_tokenizer
from headroom.transforms import (
    _LLMLINGUA_AVAILABLE,
    CacheAligner,
    CodeAwareCompressor,
    CodeCompressorConfig,
    ContentRouter,
    ContentRouterConfig,
    IntelligentContextManager,
    RollingWindow,
    SmartCrusher,
    Transform,
    TransformPipeline,
    is_tree_sitter_available,
)

# Image compression (lazy-loaded to avoid heavy dependencies at startup)
_image_compressor = None


def _get_image_compressor():
    """Lazy load image compressor to avoid startup overhead."""
    global _image_compressor
    if _image_compressor is None:
        try:
            from headroom.image import ImageCompressor

            _image_compressor = ImageCompressor()
            logger.info("Image compression enabled (model: chopratejas/technique-router)")
        except ImportError as e:
            logger.warning(f"Image compression not available: {e}")
            _image_compressor = False  # Mark as unavailable
    return _image_compressor if _image_compressor else None


# Conditionally import LLMLingua if available
if _LLMLINGUA_AVAILABLE:
    from headroom.transforms import LLMLinguaCompressor, LLMLinguaConfig

# Try to import LiteLLM for pricing
try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("headroom.proxy")

# Maximum request body size (10MB)
MAX_REQUEST_BODY_SIZE = 10 * 1024 * 1024


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RequestLog:
    """Complete log of a single request."""

    request_id: str
    timestamp: str
    provider: str
    model: str

    # Tokens
    input_tokens_original: int
    input_tokens_optimized: int
    output_tokens: int | None
    tokens_saved: int
    savings_percent: float

    # Cost
    estimated_cost_usd: float | None
    estimated_savings_usd: float | None

    # Performance
    optimization_latency_ms: float
    total_latency_ms: float | None

    # Metadata
    tags: dict[str, str]
    cache_hit: bool
    transforms_applied: list[str]

    # Request/Response (optional, for debugging)
    request_messages: list[dict] | None = None
    response_content: str | None = None
    error: str | None = None


@dataclass
class CacheEntry:
    """Cached response entry."""

    response_body: bytes
    response_headers: dict[str, str]
    created_at: datetime
    ttl_seconds: int
    hit_count: int = 0
    tokens_saved_per_hit: int = 0


@dataclass
class RateLimitState:
    """Token bucket rate limiter state."""

    tokens: float
    last_update: float


@dataclass
class ProxyConfig:
    """Proxy configuration."""

    # Server
    host: str = "127.0.0.1"
    port: int = 8787
    openai_api_url: str | None = None  # Custom OpenAI API URL override
    gemini_api_url: str | None = None  # Custom Gemini API URL override

    # Backend: "anthropic" (direct API), "bedrock" (AWS Bedrock), or "litellm-*" (via LiteLLM)
    # LiteLLM backends: "litellm-bedrock", "litellm-vertex", "litellm-azure", etc.
    backend: str = "anthropic"
    bedrock_region: str = "us-west-2"  # AWS region for Bedrock/LiteLLM
    bedrock_profile: str | None = None  # AWS profile (optional)

    # Optimization
    optimize: bool = True
    image_optimize: bool = True  # Compress images using trained ML router
    min_tokens_to_crush: int = 500
    max_items_after_crush: int = 50
    keep_last_turns: int = 4

    # CCR Tool Injection
    ccr_inject_tool: bool = True  # Inject headroom_retrieve tool when compression occurs
    ccr_inject_system_instructions: bool = False  # Add instructions to system message

    # CCR Response Handling (intercept and handle CCR tool calls automatically)
    ccr_handle_responses: bool = True  # Handle headroom_retrieve calls in responses
    ccr_max_retrieval_rounds: int = 3  # Max rounds of retrieval before returning

    # CCR Context Tracking (track compressed content across turns)
    ccr_context_tracking: bool = True  # Track compressed contexts for proactive expansion
    ccr_proactive_expansion: bool = True  # Proactively expand based on query relevance
    ccr_max_proactive_expansions: int = 2  # Max contexts to proactively expand per turn

    # LLMLingua ML-based compression (ON by default if installed)
    llmlingua_enabled: bool = True  # Enable LLMLingua-2 for ML-based compression
    llmlingua_device: str = "auto"  # Device: 'auto', 'cuda', 'cpu', 'mps'
    llmlingua_target_rate: float = 0.3  # Target compression rate (0.3 = keep 30%)

    # Code-aware compression (ON by default if installed)
    code_aware_enabled: bool = True  # Enable AST-based code compression

    # Smart content routing (routes each message to optimal compressor)
    smart_routing: bool = True  # Use ContentRouter for intelligent compression

    # Intelligent context management (score-based dropping instead of age-based)
    intelligent_context: bool = True  # Use IntelligentContextManager instead of RollingWindow
    intelligent_context_scoring: bool = True  # Use multi-factor importance scoring
    intelligent_context_compress_first: bool = True  # Try deeper compression before dropping

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_max_entries: int = 1000

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 100000

    # Retry
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_base_delay_ms: int = 1000
    retry_max_delay_ms: int = 30000

    # Cost tracking
    cost_tracking_enabled: bool = True
    budget_limit_usd: float | None = None  # None = unlimited
    budget_period: Literal["hourly", "daily", "monthly"] = "daily"

    # Logging
    log_requests: bool = True
    log_file: str | None = None
    log_full_messages: bool = False  # Privacy: don't log content by default

    # Fallback
    fallback_enabled: bool = False
    fallback_provider: str | None = None  # "openai" or "anthropic"

    # Timeouts
    request_timeout_seconds: int = 300
    connect_timeout_seconds: int = 10

    # Connection pool (for high concurrency with multiple agents)
    max_connections: int = 500  # Max total connections to upstream APIs
    max_keepalive_connections: int = 100  # Max idle connections to keep alive
    http2: bool = True  # Enable HTTP/2 multiplexing for better throughput

    # Memory System
    memory_enabled: bool = False  # Enable memory integration
    memory_backend: Literal["local", "qdrant-neo4j"] = "local"  # Backend type
    memory_db_path: str = "headroom_memory.db"  # Path for local backend
    memory_inject_tools: bool = True  # Auto-inject memory tools
    memory_inject_context: bool = True  # Inject searched memories into context
    memory_top_k: int = 10  # Number of memories to inject
    memory_min_similarity: float = 0.3  # Minimum similarity threshold
    # Qdrant+Neo4j config (only used when memory_backend="qdrant-neo4j")
    memory_qdrant_host: str = "localhost"
    memory_qdrant_port: int = 6333
    memory_neo4j_uri: str = "neo4j://localhost:7687"
    memory_neo4j_user: str = "neo4j"
    memory_neo4j_password: str = "password"


# =============================================================================
# Caching
# =============================================================================


class SemanticCache:
    """Simple semantic cache based on message content hash.

    Uses OrderedDict for O(1) LRU eviction instead of list with O(n) pop(0).
    """

    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        # OrderedDict maintains insertion order and supports O(1) move_to_end/popitem
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    def _compute_key(self, messages: list[dict], model: str) -> str:
        """Compute cache key from messages and model."""
        # Normalize messages for consistent hashing
        normalized = json.dumps(
            {
                "model": model,
                "messages": messages,
            },
            sort_keys=True,
        )
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    async def get(self, messages: list[dict], model: str) -> CacheEntry | None:
        """Get cached response if exists and not expired."""
        key = self._compute_key(messages, model)
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            # Check expiration
            age = (datetime.now() - entry.created_at).total_seconds()
            if age > entry.ttl_seconds:
                del self._cache[key]
                return None

            entry.hit_count += 1
            # Move to end for LRU (O(1) operation)
            self._cache.move_to_end(key)
            return entry

    async def set(
        self,
        messages: list[dict],
        model: str,
        response_body: bytes,
        response_headers: dict[str, str],
        tokens_saved: int = 0,
    ):
        """Cache a response."""
        key = self._compute_key(messages, model)

        async with self._lock:
            # If key already exists, remove it first to update position
            if key in self._cache:
                del self._cache[key]

            # Evict oldest entries if at capacity (LRU) - O(1) with popitem
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)  # Remove oldest (first) entry

            self._cache[key] = CacheEntry(
                response_body=response_body,
                response_headers=response_headers,
                created_at=datetime.now(),
                ttl_seconds=self.ttl_seconds,
                tokens_saved_per_hit=tokens_saved,
            )

    async def stats(self) -> dict:
        """Get cache statistics."""
        async with self._lock:
            total_hits = sum(e.hit_count for e in self._cache.values())
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "total_hits": total_hits,
                "ttl_seconds": self.ttl_seconds,
            }

    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()


# =============================================================================
# Rate Limiting
# =============================================================================


class TokenBucketRateLimiter:
    """Token bucket rate limiter for requests and tokens."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # Per-key buckets (key = API key or IP)
        self._request_buckets: dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(tokens=requests_per_minute, last_update=time.time())
        )
        self._token_buckets: dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(tokens=tokens_per_minute, last_update=time.time())
        )
        self._lock = asyncio.Lock()

    def _refill(self, state: RateLimitState, rate_per_minute: float) -> float:
        """Refill bucket based on elapsed time."""
        now = time.time()
        elapsed = now - state.last_update
        refill = elapsed * (rate_per_minute / 60.0)
        state.tokens = min(rate_per_minute, state.tokens + refill)
        state.last_update = now
        return state.tokens

    async def check_request(self, key: str = "default") -> tuple[bool, float]:
        """Check if request is allowed. Returns (allowed, wait_seconds)."""
        async with self._lock:
            state = self._request_buckets[key]
            available = self._refill(state, self.requests_per_minute)

            if available >= 1:
                state.tokens -= 1
                return True, 0

            wait_seconds = (1 - available) * (60.0 / self.requests_per_minute)
            return False, wait_seconds

    async def check_tokens(self, key: str, token_count: int) -> tuple[bool, float]:
        """Check if token usage is allowed."""
        async with self._lock:
            state = self._token_buckets[key]
            available = self._refill(state, self.tokens_per_minute)

            if available >= token_count:
                state.tokens -= token_count
                return True, 0

            wait_seconds = (token_count - available) * (60.0 / self.tokens_per_minute)
            return False, wait_seconds

    async def stats(self) -> dict:
        """Get rate limiter statistics."""
        async with self._lock:
            return {
                "requests_per_minute": self.requests_per_minute,
                "tokens_per_minute": self.tokens_per_minute,
                "active_keys": len(self._request_buckets),
            }


# =============================================================================
# Cost Tracking
# =============================================================================


class CostTracker:
    """Track costs and enforce budgets.

    Cost history is automatically pruned to prevent unbounded memory growth:
    - Entries older than 24 hours are removed
    - Maximum of 100,000 entries are kept

    Uses LiteLLM's community-maintained pricing database for accurate costs.
    See: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
    """

    MAX_COST_ENTRIES = 100_000
    COST_RETENTION_HOURS = 24

    def __init__(self, budget_limit_usd: float | None = None, budget_period: str = "daily"):
        self.budget_limit_usd = budget_limit_usd
        self.budget_period = budget_period

        # Cost tracking - using deque for efficient left-side removal
        self._costs: deque[tuple[datetime, float]] = deque(maxlen=self.MAX_COST_ENTRIES)
        self._total_cost_usd: float = 0
        self._total_savings_usd: float = 0
        self._last_prune_time: datetime = datetime.now()

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float | None:
        """Estimate cost in USD using LiteLLM's pricing database.

        Args:
            model: Model name for pricing lookup
            input_tokens: Input tokens sent to API (does NOT include cache_read, which is served from cache)
            output_tokens: Output tokens
            cache_read_tokens: Tokens read from cache (charged at ~10% of input rate)
            cache_write_tokens: Tokens written to cache - this is a SUBSET of input_tokens (charged at ~125% of input rate)
        """
        if not LITELLM_AVAILABLE:
            logger.warning("LiteLLM not available - cannot calculate costs")
            return None

        try:
            # cost_per_token returns (total_input_cost, total_output_cost) for the given token counts
            # Despite the name, it returns total cost not per-token cost

            # Anthropic's token semantics (all three are SEPARATE, not overlapping):
            # - input_tokens: tokens sent that are NOT cached (neither read nor written)
            # - cache_read_input_tokens: tokens served from existing cache
            # - cache_creation_input_tokens: tokens being written to cache
            # Total billable = input_tokens + cache_read + cache_write (each at different rates)
            regular_input = input_tokens  # Don't subtract cache_write, they're separate

            # Get cost for regular (non-cached) input tokens
            input_cost, _ = litellm.cost_per_token(
                model=model,
                prompt_tokens=regular_input,
                completion_tokens=0,
            )

            # Get cost for output tokens
            _, output_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=0,
                completion_tokens=output_tokens,
            )

            # Get model info for cache pricing
            model_info: dict[str, Any] = {}
            try:
                model_info = dict(litellm.get_model_info(model))
            except Exception:
                pass

            # Calculate cache read cost (typically 10% of input price)
            cache_read_cost = 0.0
            if cache_read_tokens > 0:
                cache_read_cost_per_token = model_info.get("cache_read_input_token_cost")
                if cache_read_cost_per_token:
                    cache_read_cost = cache_read_tokens * cache_read_cost_per_token
                else:
                    # Fallback: most providers charge ~10% of input price for cache reads
                    cache_read_full_cost, _ = litellm.cost_per_token(
                        model=model,
                        prompt_tokens=cache_read_tokens,
                        completion_tokens=0,
                    )
                    cache_read_cost = cache_read_full_cost * 0.1

            # Calculate cache write cost (typically 125% of input price)
            cache_write_cost = 0.0
            if cache_write_tokens > 0:
                cache_write_cost_per_token = model_info.get("cache_creation_input_token_cost")
                if cache_write_cost_per_token:
                    cache_write_cost = cache_write_tokens * cache_write_cost_per_token
                else:
                    # Fallback: most providers charge ~125% of input price for cache writes
                    cache_write_full_cost, _ = litellm.cost_per_token(
                        model=model,
                        prompt_tokens=cache_write_tokens,
                        completion_tokens=0,
                    )
                    cache_write_cost = cache_write_full_cost * 1.25

            total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost
            return float(total_cost) if total_cost > 0 else None

        except Exception as e:
            logger.warning(f"Failed to get pricing for model {model}: {e}")
            return None

    def _prune_old_costs(self):
        """Remove cost entries older than retention period.

        Called periodically (every 5 minutes) to prevent unbounded memory growth.
        The deque maxlen provides a hard cap, but time-based pruning keeps
        memory usage proportional to actual traffic patterns.
        """
        now = datetime.now()
        # Only prune every 5 minutes to avoid overhead
        if (now - self._last_prune_time).total_seconds() < 300:
            return

        self._last_prune_time = now
        cutoff = now - timedelta(hours=self.COST_RETENTION_HOURS)

        # Remove entries from the left (oldest) while they're older than cutoff
        while self._costs and self._costs[0][0] < cutoff:
            self._costs.popleft()

    def record_cost(self, cost_usd: float):
        """Record a cost. Periodically prunes old entries."""
        self._costs.append((datetime.now(), cost_usd))
        self._total_cost_usd += cost_usd
        # Periodically prune old costs to prevent memory growth
        self._prune_old_costs()

    def record_savings(self, savings_usd: float):
        """Record savings from optimization."""
        self._total_savings_usd += savings_usd

    def get_period_cost(self) -> float:
        """Get cost for current budget period."""
        now = datetime.now()

        if self.budget_period == "hourly":
            cutoff = now - timedelta(hours=1)
        elif self.budget_period == "daily":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # monthly
            cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        return sum(cost for ts, cost in self._costs if ts >= cutoff)

    def check_budget(self) -> tuple[bool, float]:
        """Check if within budget. Returns (allowed, remaining)."""
        if self.budget_limit_usd is None:
            return True, float("inf")

        period_cost = self.get_period_cost()
        remaining = self.budget_limit_usd - period_cost
        return remaining > 0, max(0, remaining)

    def stats(self) -> dict:
        """Get cost statistics."""
        return {
            "total_cost_usd": round(self._total_cost_usd, 4),
            "total_savings_usd": round(self._total_savings_usd, 4),
            "period_cost_usd": round(self.get_period_cost(), 4),
            "budget_limit_usd": self.budget_limit_usd,
            "budget_period": self.budget_period,
            "budget_remaining_usd": round(self.check_budget()[1], 4)
            if self.budget_limit_usd
            else None,
        }


# =============================================================================
# Prometheus Metrics
# =============================================================================


class PrometheusMetrics:
    """Prometheus-compatible metrics."""

    def __init__(self):
        self.requests_total = 0
        self.requests_by_provider: dict[str, int] = defaultdict(int)
        self.requests_by_model: dict[str, int] = defaultdict(int)
        self.requests_cached = 0
        self.requests_rate_limited = 0
        self.requests_failed = 0

        self.tokens_input_total = 0
        self.tokens_output_total = 0
        self.tokens_saved_total = 0

        self.latency_sum_ms = 0.0
        self.latency_count = 0

        self.cost_total_usd = 0.0
        self.savings_total_usd = 0.0

        self._lock = asyncio.Lock()

    async def record_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tokens_saved: int,
        latency_ms: float,
        cached: bool = False,
        cost_usd: float = 0,
        savings_usd: float = 0,
    ):
        """Record metrics for a request."""
        async with self._lock:
            self.requests_total += 1
            self.requests_by_provider[provider] += 1
            self.requests_by_model[model] += 1

            if cached:
                self.requests_cached += 1

            self.tokens_input_total += input_tokens
            self.tokens_output_total += output_tokens
            self.tokens_saved_total += tokens_saved

            self.latency_sum_ms += latency_ms
            self.latency_count += 1

            self.cost_total_usd += cost_usd
            self.savings_total_usd += savings_usd

    async def record_rate_limited(self):
        async with self._lock:
            self.requests_rate_limited += 1

    async def record_failed(self):
        async with self._lock:
            self.requests_failed += 1

    async def export(self) -> str:
        """Export metrics in Prometheus format."""
        async with self._lock:
            lines = [
                "# HELP headroom_requests_total Total number of requests",
                "# TYPE headroom_requests_total counter",
                f"headroom_requests_total {self.requests_total}",
                "",
                "# HELP headroom_requests_cached_total Cached request count",
                "# TYPE headroom_requests_cached_total counter",
                f"headroom_requests_cached_total {self.requests_cached}",
                "",
                "# HELP headroom_requests_rate_limited_total Rate limited requests",
                "# TYPE headroom_requests_rate_limited_total counter",
                f"headroom_requests_rate_limited_total {self.requests_rate_limited}",
                "",
                "# HELP headroom_requests_failed_total Failed requests",
                "# TYPE headroom_requests_failed_total counter",
                f"headroom_requests_failed_total {self.requests_failed}",
                "",
                "# HELP headroom_tokens_input_total Total input tokens",
                "# TYPE headroom_tokens_input_total counter",
                f"headroom_tokens_input_total {self.tokens_input_total}",
                "",
                "# HELP headroom_tokens_output_total Total output tokens",
                "# TYPE headroom_tokens_output_total counter",
                f"headroom_tokens_output_total {self.tokens_output_total}",
                "",
                "# HELP headroom_tokens_saved_total Tokens saved by optimization",
                "# TYPE headroom_tokens_saved_total counter",
                f"headroom_tokens_saved_total {self.tokens_saved_total}",
                "",
                "# HELP headroom_latency_ms_sum Sum of request latencies",
                "# TYPE headroom_latency_ms_sum counter",
                f"headroom_latency_ms_sum {self.latency_sum_ms:.2f}",
                "",
                "# HELP headroom_cost_usd_total Total cost in USD",
                "# TYPE headroom_cost_usd_total counter",
                f"headroom_cost_usd_total {self.cost_total_usd:.6f}",
                "",
                "# HELP headroom_savings_usd_total Total savings in USD",
                "# TYPE headroom_savings_usd_total counter",
                f"headroom_savings_usd_total {self.savings_total_usd:.6f}",
            ]

            # Per-provider metrics
            lines.extend(
                [
                    "",
                    "# HELP headroom_requests_by_provider Requests by provider",
                    "# TYPE headroom_requests_by_provider counter",
                ]
            )
            for provider, count in self.requests_by_provider.items():
                lines.append(f'headroom_requests_by_provider{{provider="{provider}"}} {count}')

            # Per-model metrics
            lines.extend(
                [
                    "",
                    "# HELP headroom_requests_by_model Requests by model",
                    "# TYPE headroom_requests_by_model counter",
                ]
            )
            for model, count in self.requests_by_model.items():
                lines.append(f'headroom_requests_by_model{{model="{model}"}} {count}')

            return "\n".join(lines)


# =============================================================================
# Request Logger
# =============================================================================


class RequestLogger:
    """Log requests to JSONL file.

    Uses a deque with max 10,000 entries to prevent unbounded memory growth.
    """

    MAX_LOG_ENTRIES = 10_000

    def __init__(self, log_file: str | None = None, log_full_messages: bool = False):
        self.log_file = Path(log_file) if log_file else None
        self.log_full_messages = log_full_messages
        # Use deque with maxlen for automatic FIFO eviction
        self._logs: deque[RequestLog] = deque(maxlen=self.MAX_LOG_ENTRIES)

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: RequestLog):
        """Log a request. Oldest entries are automatically removed when limit reached."""
        self._logs.append(entry)

        if self.log_file:
            with open(self.log_file, "a") as f:
                log_dict = asdict(entry)
                if not self.log_full_messages:
                    log_dict.pop("request_messages", None)
                    log_dict.pop("response_content", None)
                f.write(json.dumps(log_dict) + "\n")

    def get_recent(self, n: int = 100) -> list[dict]:
        """Get recent log entries."""
        # Convert deque to list for slicing (deque doesn't support slicing)
        entries = list(self._logs)[-n:]
        return [
            {
                k: v
                for k, v in asdict(e).items()
                if k not in ("request_messages", "response_content")
            }
            for e in entries
        ]

    def stats(self) -> dict:
        """Get logging statistics."""
        return {
            "total_logged": len(self._logs),
            "log_file": str(self.log_file) if self.log_file else None,
        }


# =============================================================================
# Main Proxy
# =============================================================================


class HeadroomProxy:
    """Production-ready Headroom optimization proxy."""

    ANTHROPIC_API_URL = "https://api.anthropic.com"
    OPENAI_API_URL = "https://api.openai.com"
    GEMINI_API_URL = "https://generativelanguage.googleapis.com"

    def __init__(self, config: ProxyConfig):
        self.config = config

        # Override OPENAI_API_URL with config if set
        if config.openai_api_url:
            HeadroomProxy.OPENAI_API_URL = config.openai_api_url

        # Override GEMINI_API_URL with config if set
        if config.gemini_api_url:
            HeadroomProxy.GEMINI_API_URL = config.gemini_api_url

        # Initialize providers
        self.anthropic_provider = AnthropicProvider()
        self.openai_provider = OpenAIProvider()

        # Initialize transforms based on routing mode
        # Choose context manager: IntelligentContextManager (smart) or RollingWindow (legacy)
        context_manager: Transform  # Can be either IntelligentContextManager or RollingWindow
        if config.intelligent_context:
            # Get TOIN instance for learned pattern integration
            toin = get_toin() if config.intelligent_context_scoring else None
            context_manager = IntelligentContextManager(
                config=IntelligentContextConfig(
                    enabled=True,
                    keep_system=True,
                    keep_last_turns=config.keep_last_turns,
                    use_importance_scoring=config.intelligent_context_scoring,
                    toin_integration=config.intelligent_context_scoring,
                    compress_threshold=0.10 if config.intelligent_context_compress_first else 0.0,
                ),
                toin=toin,
            )
            self._context_manager_status = "intelligent"
        else:
            context_manager = RollingWindow(
                RollingWindowConfig(
                    enabled=True,
                    keep_system=True,
                    keep_last_turns=config.keep_last_turns,
                )
            )
            self._context_manager_status = "rolling_window"

        if config.smart_routing:
            # Smart routing: ContentRouter handles all content types intelligently
            # It lazy-loads compressors (including LLMLingua) only when needed
            router_config = ContentRouterConfig(
                enable_llmlingua=config.llmlingua_enabled,
                enable_code_aware=config.code_aware_enabled,
            )
            transforms = [
                CacheAligner(CacheAlignerConfig(enabled=True)),
                ContentRouter(router_config),
                context_manager,
            ]
            self._llmlingua_status = "lazy" if config.llmlingua_enabled else "disabled"
            self._code_aware_status = "lazy" if config.code_aware_enabled else "disabled"
        else:
            # Legacy mode: sequential pipeline
            transforms = [
                CacheAligner(CacheAlignerConfig(enabled=True)),
                SmartCrusher(
                    SmartCrusherConfig(  # type: ignore[arg-type]
                        enabled=True,
                        min_tokens_to_crush=config.min_tokens_to_crush,
                        max_items_after_crush=config.max_items_after_crush,
                    ),
                    ccr_config=CCRConfig(
                        enabled=config.ccr_inject_tool,
                        inject_retrieval_marker=config.ccr_inject_tool,  # Add CCR markers
                    ),
                ),
                context_manager,
            ]
            # Add LLMLingua if enabled and available
            self._llmlingua_status = self._setup_llmlingua(config, transforms)
            # Add CodeAware if enabled and available
            self._code_aware_status = self._setup_code_aware(config, transforms)

        self.anthropic_pipeline = TransformPipeline(
            transforms=transforms,
            provider=self.anthropic_provider,
        )
        self.openai_pipeline = TransformPipeline(
            transforms=transforms,
            provider=self.openai_provider,
        )

        # Initialize components
        self.cache = (
            SemanticCache(
                max_entries=config.cache_max_entries,
                ttl_seconds=config.cache_ttl_seconds,
            )
            if config.cache_enabled
            else None
        )

        self.rate_limiter = (
            TokenBucketRateLimiter(
                requests_per_minute=config.rate_limit_requests_per_minute,
                tokens_per_minute=config.rate_limit_tokens_per_minute,
            )
            if config.rate_limit_enabled
            else None
        )

        self.cost_tracker = (
            CostTracker(
                budget_limit_usd=config.budget_limit_usd,
                budget_period=config.budget_period,
            )
            if config.cost_tracking_enabled
            else None
        )

        self.metrics = PrometheusMetrics()

        self.logger = (
            RequestLogger(
                log_file=config.log_file,
                log_full_messages=config.log_full_messages,
            )
            if config.log_requests
            else None
        )

        # HTTP client
        self.http_client: httpx.AsyncClient | None = None

        # Backend for Anthropic API (direct or via LiteLLM)
        # Supports: "anthropic" (direct), "bedrock", "vertex", or "litellm-<provider>"
        self.anthropic_backend: Backend | None = None
        if config.backend != "anthropic":
            # Normalize backend name: "bedrock" -> "litellm-bedrock"
            backend = config.backend
            if not backend.startswith("litellm-"):
                backend = f"litellm-{backend}"
            provider = backend.replace("litellm-", "")

            try:
                self.anthropic_backend = LiteLLMBackend(
                    provider=provider,
                    region=config.bedrock_region,
                )
                logger.info(
                    f"LiteLLM backend enabled (provider={provider}, region={config.bedrock_region})"
                )
            except ImportError as e:
                logger.warning(f"LiteLLM backend not available: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize LiteLLM backend: {e}")

        # Request counter for IDs
        self._request_counter = 0
        self._request_counter_lock = asyncio.Lock()

        # CCR tool injectors (one per provider)
        self.anthropic_tool_injector = CCRToolInjector(
            provider="anthropic",
            inject_tool=config.ccr_inject_tool,
            inject_system_instructions=config.ccr_inject_system_instructions,
        )
        self.openai_tool_injector = CCRToolInjector(
            provider="openai",
            inject_tool=config.ccr_inject_tool,
            inject_system_instructions=config.ccr_inject_system_instructions,
        )

        # CCR Response Handler (handles CCR tool calls automatically)
        self.ccr_response_handler = (
            CCRResponseHandler(
                ResponseHandlerConfig(
                    enabled=True,
                    max_retrieval_rounds=config.ccr_max_retrieval_rounds,
                )
            )
            if config.ccr_handle_responses
            else None
        )

        # CCR Context Tracker (tracks compressed content across turns)
        self.ccr_context_tracker = (
            ContextTracker(
                ContextTrackerConfig(
                    enabled=True,
                    proactive_expansion=config.ccr_proactive_expansion,
                    max_proactive_expansions=config.ccr_max_proactive_expansions,
                )
            )
            if config.ccr_context_tracking
            else None
        )

        # Turn counter for context tracking
        self._turn_counter = 0

        # Memory Handler (persistent user memory)
        self.memory_handler: MemoryHandler | None = None
        if config.memory_enabled:
            memory_config = MemoryConfig(
                enabled=True,
                backend=config.memory_backend,
                db_path=config.memory_db_path,
                inject_tools=config.memory_inject_tools,
                inject_context=config.memory_inject_context,
                top_k=config.memory_top_k,
                min_similarity=config.memory_min_similarity,
                qdrant_host=config.memory_qdrant_host,
                qdrant_port=config.memory_qdrant_port,
                neo4j_uri=config.memory_neo4j_uri,
                neo4j_user=config.memory_neo4j_user,
                neo4j_password=config.memory_neo4j_password,
            )
            self.memory_handler = MemoryHandler(memory_config)

    def _setup_llmlingua(self, config: ProxyConfig, transforms: list) -> str:
        """Set up LLMLingua compression if enabled.

        Args:
            config: Proxy configuration
            transforms: Transform list to append to

        Returns:
            Status string for logging: 'enabled', 'disabled', 'available', 'unavailable'
        """
        if config.llmlingua_enabled:
            if _LLMLINGUA_AVAILABLE:
                llmlingua_config = LLMLinguaConfig(
                    device=config.llmlingua_device,
                    target_compression_rate=config.llmlingua_target_rate,
                    enable_ccr=config.ccr_inject_tool,  # Link to CCR
                )
                # Insert before RollingWindow (which should be last)
                # LLMLingua works best on individual tool outputs before windowing
                transforms.insert(-1, LLMLinguaCompressor(llmlingua_config))
                return "enabled"
            else:
                logger.warning(
                    "LLMLingua requested but not installed. "
                    "Install with: pip install headroom-ai[llmlingua]"
                )
                return "unavailable"
        else:
            if _LLMLINGUA_AVAILABLE:
                return "available"  # Available but not enabled - hint to user
            return "disabled"

    def _setup_code_aware(self, config: ProxyConfig, transforms: list) -> str:
        """Set up code-aware compression if enabled.

        Args:
            config: Proxy configuration
            transforms: Transform list to append to

        Returns:
            Status string for logging: 'enabled', 'disabled', 'available', 'unavailable'
        """
        if config.code_aware_enabled:
            if is_tree_sitter_available():
                code_config = CodeCompressorConfig(
                    preserve_imports=True,
                    preserve_signatures=True,
                    preserve_type_annotations=True,
                    preserve_error_handlers=True,
                )
                # Insert before RollingWindow (which should be last)
                transforms.insert(-1, CodeAwareCompressor(code_config))
                return "enabled"
            else:
                logger.warning(
                    "Code-aware compression requested but tree-sitter not installed. "
                    "Install with: pip install headroom-ai[code]"
                )
                return "unavailable"
        else:
            if is_tree_sitter_available():
                return "available"  # Available but not enabled
            return "disabled"

    async def startup(self):
        """Initialize async resources."""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout_seconds,
                read=self.config.request_timeout_seconds,
                write=self.config.request_timeout_seconds,
                pool=self.config.connect_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
            ),
            http2=self.config.http2,
        )
        logger.info("Headroom Proxy started")
        logger.info(f"Optimization: {'ENABLED' if self.config.optimize else 'DISABLED'}")
        logger.info(f"Caching: {'ENABLED' if self.config.cache_enabled else 'DISABLED'}")
        logger.info(f"Rate Limiting: {'ENABLED' if self.config.rate_limit_enabled else 'DISABLED'}")
        logger.info(
            f"Connection Pool: max_connections={self.config.max_connections}, "
            f"max_keepalive={self.config.max_keepalive_connections}, "
            f"http2={'ENABLED' if self.config.http2 else 'DISABLED'}"
        )

        # Smart routing status
        if self.config.smart_routing:
            logger.info("Smart Routing: ENABLED (intelligent content detection)")
        else:
            logger.info("Smart Routing: DISABLED (legacy sequential mode)")

        # LLMLingua status with helpful hint
        if self._llmlingua_status == "enabled":
            logger.info(
                f"LLMLingua: ENABLED (device={self.config.llmlingua_device}, "
                f"rate={self.config.llmlingua_target_rate})"
            )
        elif self._llmlingua_status == "lazy":
            logger.info("LLMLingua: LAZY (will load when prose content detected)")
        elif self._llmlingua_status == "available":
            logger.info("LLMLingua: available but disabled (use --llmlingua)")
        elif self._llmlingua_status == "unavailable":
            logger.info("LLMLingua: not installed (pip install headroom-ai[llmlingua])")
        elif self._llmlingua_status == "disabled":
            logger.info("LLMLingua: DISABLED")

        # Code-aware status
        if self._code_aware_status == "enabled":
            logger.info("Code-Aware: ENABLED (AST-based compression)")
        elif self._code_aware_status == "lazy":
            logger.info("Code-Aware: LAZY (will load when code content detected)")
        elif self._code_aware_status == "available":
            logger.info("Code-Aware: available but disabled (use --code-aware)")
        elif self._code_aware_status == "unavailable":
            logger.info("Code-Aware: not installed (pip install headroom-ai[code])")
        elif self._code_aware_status == "disabled":
            logger.info("Code-Aware: DISABLED")

        # CCR status
        ccr_features = []
        if self.config.ccr_inject_tool:
            ccr_features.append("tool_injection")
        if self.config.ccr_handle_responses:
            ccr_features.append("response_handling")
        if self.config.ccr_context_tracking:
            ccr_features.append("context_tracking")
        if self.config.ccr_proactive_expansion:
            ccr_features.append("proactive_expansion")
        if ccr_features:
            logger.info(f"CCR (Compress-Cache-Retrieve): ENABLED ({', '.join(ccr_features)})")
        else:
            logger.info("CCR: DISABLED")

    async def shutdown(self):
        """Cleanup async resources."""
        if self.http_client:
            await self.http_client.aclose()

        # Print final stats
        self._print_summary()

    def _print_summary(self):
        """Print session summary."""
        m = self.metrics
        logger.info("=" * 70)
        logger.info("HEADROOM PROXY SESSION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total requests:        {m.requests_total}")
        logger.info(f"Cached responses:      {m.requests_cached}")
        logger.info(f"Rate limited:          {m.requests_rate_limited}")
        logger.info(f"Failed:                {m.requests_failed}")
        logger.info(f"Input tokens:          {m.tokens_input_total:,}")
        logger.info(f"Output tokens:         {m.tokens_output_total:,}")
        logger.info(f"Tokens saved:          {m.tokens_saved_total:,}")
        if m.tokens_input_total > 0:
            savings_pct = (
                m.tokens_saved_total / (m.tokens_input_total + m.tokens_saved_total)
            ) * 100
            logger.info(f"Token savings:         {savings_pct:.1f}%")
        logger.info(f"Total cost:            ${m.cost_total_usd:.4f}")
        logger.info(f"Total savings:         ${m.savings_total_usd:.4f}")
        if m.latency_count > 0:
            avg_latency = m.latency_sum_ms / m.latency_count
            logger.info(f"Avg latency:           {avg_latency:.0f}ms")
        logger.info("=" * 70)

    async def _next_request_id(self) -> str:
        """Generate unique request ID."""
        async with self._request_counter_lock:
            self._request_counter += 1
            return f"hr_{int(time.time())}_{self._request_counter:06d}"

    def _extract_tags(self, headers: dict) -> dict[str, str]:
        """Extract Headroom tags from headers."""
        tags = {}
        for key, value in headers.items():
            if key.lower().startswith("x-headroom-"):
                tag_name = key.lower().replace("x-headroom-", "")
                tags[tag_name] = value
        return tags

    def _inject_system_context(
        self,
        messages: list[dict[str, Any]],
        context: str,
        body: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Inject context into the system message/parameter.

        For Anthropic API: Uses top-level 'system' parameter (not messages array).
        For OpenAI API: Uses system role in messages array.

        Args:
            messages: The messages list.
            context: Context to inject.
            body: Optional request body to update system parameter (for Anthropic).

        Returns:
            Updated messages list.
        """
        messages = list(messages)  # Copy to avoid mutation

        # For Anthropic API: use top-level 'system' parameter
        if body is not None:
            existing_system = body.get("system", "")
            if isinstance(existing_system, str):
                body["system"] = (existing_system + "\n\n" + context).strip()
            else:
                # system can be a list of content blocks
                body["system"] = context
            return messages

        # For OpenAI API: use system role in messages
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    messages[i] = {**msg, "content": content + "\n\n" + context}
                return messages

        # No system message found - prepend one
        messages.insert(0, {"role": "system", "content": context})
        return messages

    async def _retry_request(
        self,
        method: str,
        url: str,
        headers: dict,
        body: dict,
        stream: bool = False,
    ) -> httpx.Response:
        """Make request with retry and exponential backoff."""
        last_error = None

        for attempt in range(self.config.retry_max_attempts):
            try:
                if stream:
                    # For streaming, we return early - retry happens at higher level
                    return await self.http_client.post(url, json=body, headers=headers)  # type: ignore[union-attr]
                else:
                    response = await self.http_client.post(url, json=body, headers=headers)  # type: ignore[union-attr]

                    # Don't retry client errors (4xx)
                    if 400 <= response.status_code < 500:
                        return response

                    # Retry server errors (5xx)
                    if response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Server error: {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                    return response

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                last_error = e

                if not self.config.retry_enabled or attempt >= self.config.retry_max_attempts - 1:
                    raise

                # Exponential backoff with jitter
                delay = min(
                    self.config.retry_base_delay_ms * (2**attempt),
                    self.config.retry_max_delay_ms,
                )
                delay_with_jitter = delay * (0.5 + random.random())

                logger.warning(
                    f"Request failed (attempt {attempt + 1}), retrying in {delay_with_jitter:.0f}ms: {e}"
                )
                await asyncio.sleep(delay_with_jitter / 1000)

        raise last_error  # type: ignore[misc]

    async def handle_anthropic_messages(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle Anthropic /v1/messages endpoint."""
        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "type": "error",
                    "error": {
                        "type": "request_too_large",
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                    },
                },
            )

        # Parse request
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid JSON in request body: {e!s}",
                    },
                },
            )
        model = body.get("model", "unknown")
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Image compression (before text optimization)
        if self.config.image_optimize and messages:
            compressor = _get_image_compressor()
            if compressor and compressor.has_images(messages):
                messages = compressor.compress(messages, provider="anthropic")
                if compressor.last_result:
                    logger.info(
                        f"Image compression: {compressor.last_result.technique.value} "
                        f"({compressor.last_result.savings_percent:.0f}% saved, "
                        f"{compressor.last_result.original_tokens} -> "
                        f"{compressor.last_result.compressed_tokens} tokens)"
                    )

        # Extract headers and tags
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)

        # Rate limiting
        if self.rate_limiter:
            rate_key = headers.get("x-api-key", "default")[:16]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited()
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                    headers={"Retry-After": str(int(wait_seconds) + 1)},
                )

        # Budget check
        if self.cost_tracker:
            allowed, remaining = self.cost_tracker.check_budget()
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Budget exceeded for {self.config.budget_period} period",
                )

        # Memory: Get user ID when memory is enabled (fallback to "default" for simple DevEx)
        memory_user_id: str | None = None
        if self.memory_handler:
            memory_user_id = headers.get("x-headroom-user-id", "default")

        # Check cache (non-streaming only)
        cache_hit = False
        if self.cache and not stream:
            cached = await self.cache.get(messages, model)
            if cached:
                cache_hit = True
                optimization_latency = (time.time() - start_time) * 1000

                await self.metrics.record_request(
                    provider="anthropic",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_saved=cached.tokens_saved_per_hit,
                    latency_ms=optimization_latency,
                    cached=True,
                )

                # Remove compression headers from cached response
                response_headers = dict(cached.response_headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(
                    content=cached.response_body,
                    headers=response_headers,
                    media_type="application/json",
                )

        # Count original tokens
        tokenizer = get_tokenizer(model)
        original_tokens = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)

        # Apply optimization
        transforms_applied = []
        optimized_messages = messages
        optimized_tokens = original_tokens

        if self.config.optimize and messages:
            try:
                context_limit = self.anthropic_provider.get_context_limit(model)
                result = self.anthropic_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                )

                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    optimized_tokens = sum(
                        tokenizer.count_text(str(m.get("content", ""))) for m in optimized_messages
                    )
            except Exception as e:
                logger.warning(f"Optimization failed: {e}")

        tokens_saved = original_tokens - optimized_tokens
        optimization_latency = (time.time() - start_time) * 1000

        # CCR Tool Injection: Inject retrieval tool if compression occurred
        tools = body.get("tools")
        if self.config.ccr_inject_tool or self.config.ccr_inject_system_instructions:
            # Create fresh injector to avoid state leakage between requests
            injector = CCRToolInjector(
                provider="anthropic",
                inject_tool=self.config.ccr_inject_tool,
                inject_system_instructions=self.config.ccr_inject_system_instructions,
            )
            optimized_messages, tools, was_injected = injector.process_request(
                optimized_messages, tools
            )

            if injector.has_compressed_content:
                if was_injected:
                    logger.debug(
                        f"[{request_id}] CCR: Injected retrieval tool for hashes: {injector.detected_hashes}"
                    )
                else:
                    logger.debug(
                        f"[{request_id}] CCR: Tool already present (MCP?), skipped injection for hashes: {injector.detected_hashes}"
                    )

                # Track compression in context tracker for multi-turn awareness
                if self.ccr_context_tracker:
                    self._turn_counter += 1
                    for hash_key in injector.detected_hashes:
                        # Get compression metadata from store
                        store = get_compression_store()
                        entry = store.get_metadata(hash_key)
                        if entry:
                            self.ccr_context_tracker.track_compression(
                                hash_key=hash_key,
                                turn_number=self._turn_counter,
                                tool_name=entry.get("tool_name"),
                                original_count=entry.get("original_item_count", 0),
                                compressed_count=entry.get("compressed_item_count", 0),
                                query_context=entry.get("query_context", ""),
                                sample_content=entry.get("compressed_content", "")[:500],
                            )

        # CCR Proactive Expansion: Check if current query needs expanded context
        if self.ccr_context_tracker and self.config.ccr_proactive_expansion:
            # Extract user query from messages
            user_query = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_query = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                user_query = block.get("text", "")
                                break
                    break

            if user_query:
                recommendations = self.ccr_context_tracker.analyze_query(
                    user_query, self._turn_counter
                )
                if recommendations:
                    expansions = self.ccr_context_tracker.execute_expansions(recommendations)
                    if expansions:
                        # Add expanded context to the system message or as additional context
                        expansion_text = self.ccr_context_tracker.format_expansions_for_context(
                            expansions
                        )
                        logger.info(
                            f"[{request_id}] CCR: Proactively expanded {len(expansions)} context(s) "
                            f"based on query relevance"
                        )
                        # Append to the last user message
                        if optimized_messages and optimized_messages[-1].get("role") == "user":
                            last_msg = optimized_messages[-1]
                            content = last_msg.get("content", "")
                            if isinstance(content, str):
                                optimized_messages[-1] = {
                                    **last_msg,
                                    "content": content + "\n\n" + expansion_text,
                                }

        # Memory: Inject context and tools
        if self.memory_handler and memory_user_id:
            # Search and inject memory context
            if self.memory_handler.config.inject_context:
                try:
                    memory_context = await self.memory_handler.search_and_format_context(
                        memory_user_id, optimized_messages
                    )
                    if memory_context:
                        optimized_messages = self._inject_system_context(
                            optimized_messages, memory_context, body=body
                        )
                        logger.info(
                            f"[{request_id}] Memory: Injected {len(memory_context)} chars of context"
                        )
                except Exception as e:
                    logger.warning(f"[{request_id}] Memory: Context injection failed: {e}")

            # Inject memory tools
            if self.memory_handler.config.inject_tools:
                tools, mem_tools_injected = self.memory_handler.inject_tools(tools, "anthropic")
                if mem_tools_injected:
                    tool_names = [
                        t.get("name") for t in tools if t.get("name", "").startswith("memory_")
                    ]
                    logger.info(f"[{request_id}] Memory: Injected tools: {tool_names}")

        # Update body
        body["messages"] = optimized_messages
        if tools is not None:
            body["tools"] = tools

        # Forward request - use Bedrock backend if configured, otherwise direct API
        if self.anthropic_backend is not None:
            # Route through Bedrock backend
            try:
                if stream:
                    return await self._stream_response_bedrock(
                        body,
                        headers,
                        "anthropic",
                        model,
                        request_id,
                        original_tokens,
                        optimized_tokens,
                        tokens_saved,
                        transforms_applied,
                        tags,
                        optimization_latency,
                    )
                else:
                    backend_response = await self.anthropic_backend.send_message(body, headers)

                    if backend_response.error:
                        return JSONResponse(
                            status_code=backend_response.status_code,
                            content=backend_response.body,
                        )

                    # Track metrics
                    total_latency = (time.time() - start_time) * 1000
                    usage = backend_response.body.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)

                    await self.metrics.record_request(
                        provider="bedrock",
                        model=model,
                        input_tokens=optimized_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=tokens_saved,
                        latency_ms=total_latency,
                        cached=False,
                    )

                    if self.cost_tracker:
                        cost_usd = self.cost_tracker.estimate_cost(
                            model, optimized_tokens, output_tokens
                        )
                        if cost_usd:
                            self.cost_tracker.record_cost(cost_usd)

                    return JSONResponse(
                        status_code=backend_response.status_code,
                        content=backend_response.body,
                    )
            except Exception as e:
                logger.error(f"[{request_id}] Bedrock backend error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "type": "error",
                        "error": {"type": "api_error", "message": str(e)},
                    },
                )

        # Direct Anthropic API
        url = f"{self.ANTHROPIC_API_URL}/v1/messages"

        try:
            if stream:
                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "anthropic",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                    memory_user_id=memory_user_id,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)

                # Parse response for CCR handling
                resp_json = None
                try:
                    resp_json = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to parse response JSON for CCR handling: {e}"
                    )

                # CCR Response Handling: Handle headroom_retrieve tool calls automatically
                if (
                    self.ccr_response_handler
                    and resp_json
                    and response.status_code == 200
                    and self.ccr_response_handler.has_ccr_tool_calls(resp_json, "anthropic")
                ):
                    logger.info(f"[{request_id}] CCR: Detected retrieval tool call, handling...")

                    # Create API call function for continuation
                    # Use a fresh client to avoid potential decompression state issues
                    async def api_call_fn(
                        msgs: list[dict], tls: list[dict] | None
                    ) -> dict[str, Any]:
                        continuation_body = {
                            **body,
                            "messages": msgs,
                        }
                        if tls is not None:
                            continuation_body["tools"] = tls

                        # Use clean headers for continuation
                        continuation_headers = {
                            k: v
                            for k, v in headers.items()
                            if k.lower()
                            not in (
                                "content-encoding",
                                "transfer-encoding",
                                "accept-encoding",
                                "content-length",
                            )
                        }

                        # Reuse main client for CCR continuations (connection pooling)
                        logger.info(f"CCR: Making continuation request with {len(msgs)} messages")
                        assert self.http_client is not None, "HTTP client not initialized"
                        try:
                            cont_response = await self.http_client.post(
                                url,
                                json=continuation_body,
                                headers=continuation_headers,
                                timeout=httpx.Timeout(120.0),  # Override timeout for CCR
                            )
                            logger.info(
                                f"CCR: Got response status={cont_response.status_code}, "
                                f"content-encoding={cont_response.headers.get('content-encoding')}"
                            )
                            result: dict[str, Any] = cont_response.json()
                            logger.info("CCR: Parsed JSON successfully")
                            return result
                        except Exception as e:
                            logger.error(
                                f"CCR: API call failed: {e}, "
                                f"response headers: {dict(cont_response.headers) if 'cont_response' in dir() else 'N/A'}"
                            )
                            raise

                    # Handle CCR tool calls
                    try:
                        final_resp_json = await self.ccr_response_handler.handle_response(
                            resp_json,
                            optimized_messages,
                            tools,
                            api_call_fn,
                            provider="anthropic",
                        )
                        # Update response content with final response
                        resp_json = final_resp_json
                        # Remove encoding headers since content is now uncompressed JSON
                        ccr_response_headers = {
                            k: v
                            for k, v in response.headers.items()
                            if k.lower() not in ("content-encoding", "content-length")
                        }
                        response = httpx.Response(
                            status_code=200,
                            content=json.dumps(final_resp_json).encode(),
                            headers=ccr_response_headers,
                        )
                        logger.info(f"[{request_id}] CCR: Retrieval handled successfully")
                    except Exception as e:
                        import traceback

                        logger.warning(
                            f"[{request_id}] CCR: Response handling failed: {e}\n"
                            f"Traceback: {traceback.format_exc()}"
                        )
                        # Continue with original response

                # Memory: Handle memory tool calls in response
                if (
                    self.memory_handler
                    and memory_user_id
                    and resp_json
                    and response.status_code == 200
                    and self.memory_handler.has_memory_tool_calls(resp_json, "anthropic")
                ):
                    logger.info(f"[{request_id}] Memory: Detected memory tool call, handling...")

                    try:
                        # Execute memory tool calls
                        tool_results = await self.memory_handler.handle_memory_tool_calls(
                            resp_json, memory_user_id, "anthropic"
                        )

                        if tool_results:
                            # Create continuation messages
                            assistant_msg = {
                                "role": "assistant",
                                "content": resp_json.get("content", []),
                            }
                            user_msg = {
                                "role": "user",
                                "content": tool_results,
                            }

                            continuation_messages = optimized_messages + [assistant_msg, user_msg]

                            # Make continuation API call
                            continuation_body = {**body, "messages": continuation_messages}
                            if tools:
                                continuation_body["tools"] = tools

                            cont_response = await self._retry_request(
                                "POST", url, headers, continuation_body
                            )

                            # Update response with continuation
                            resp_json = cont_response.json()
                            response = cont_response
                            logger.info(
                                f"[{request_id}] Memory: Tool calls handled, continuation complete"
                            )

                    except Exception as e:
                        logger.warning(f"[{request_id}] Memory: Tool call handling failed: {e}")
                        # Continue with original response

                total_latency = (time.time() - start_time) * 1000

                # Parse response for actual token counts from API
                actual_input_tokens = optimized_tokens  # fallback
                output_tokens = 0
                cache_read_tokens = 0
                cache_write_tokens = 0
                if resp_json:
                    usage = resp_json.get("usage", {})
                    actual_input_tokens = usage.get("input_tokens", optimized_tokens)
                    output_tokens = usage.get("output_tokens", 0)
                    # Anthropic returns cache_read_input_tokens for cached prompt tokens
                    # These are charged at 10% of the input price
                    cache_read_tokens = usage.get("cache_read_input_tokens", 0)
                    # Anthropic returns cache_creation_input_tokens for tokens written to cache
                    # These are charged at 125% of the input price
                    cache_write_tokens = usage.get("cache_creation_input_tokens", 0)

                # Calculate cost using actual API tokens with proper cache pricing
                cost_usd = None
                savings_usd = None
                if self.cost_tracker:
                    cost_usd = self.cost_tracker.estimate_cost(
                        model,
                        actual_input_tokens,
                        output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                    )
                    original_cost = self.cost_tracker.estimate_cost(
                        model,
                        original_tokens,
                        output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                    )
                    if cost_usd and original_cost:
                        savings_usd = original_cost - cost_usd
                        self.cost_tracker.record_cost(cost_usd)
                        self.cost_tracker.record_savings(savings_usd)

                # Cache response
                if self.cache and response.status_code == 200:
                    await self.cache.set(
                        messages,
                        model,
                        response.content,
                        dict(response.headers),
                        tokens_saved=tokens_saved,
                    )

                # Record metrics with actual API tokens
                await self.metrics.record_request(
                    provider="anthropic",
                    model=model,
                    input_tokens=actual_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    cost_usd=cost_usd or 0,
                    savings_usd=savings_usd or 0,
                )

                # Log request
                if self.logger:
                    self.logger.log(
                        RequestLog(
                            request_id=request_id,
                            timestamp=datetime.now().isoformat(),
                            provider="anthropic",
                            model=model,
                            input_tokens_original=original_tokens,
                            input_tokens_optimized=optimized_tokens,
                            output_tokens=output_tokens,
                            tokens_saved=tokens_saved,
                            savings_percent=(tokens_saved / original_tokens * 100)
                            if original_tokens > 0
                            else 0,
                            estimated_cost_usd=cost_usd,
                            estimated_savings_usd=savings_usd,
                            optimization_latency_ms=optimization_latency,
                            total_latency_ms=total_latency,
                            tags=tags,
                            cache_hit=cache_hit,
                            transforms_applied=transforms_applied,
                            request_messages=messages if self.config.log_full_messages else None,
                        )
                    )

                # Log to console
                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens, ${savings_usd:.4f})"
                        if savings_usd
                        else f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens)"
                    )

                # Remove compression headers since httpx already decompressed the response
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)  # Length changed after decompression

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )

        except Exception as e:
            await self.metrics.record_failed()
            # Log full error details internally for debugging
            logger.error(f"[{request_id}] Request failed: {type(e).__name__}: {e}")

            # Try fallback if enabled
            if self.config.fallback_enabled and self.config.fallback_provider == "openai":
                logger.info(f"[{request_id}] Attempting fallback to OpenAI")
                # Convert to OpenAI format and retry
                # (simplified - would need message format conversion)

            # Return sanitized error message to client (don't expose internal details)
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "An error occurred while processing your request. Please try again.",
                    },
                },
            )

    async def handle_anthropic_batch_create(
        self,
        request: Request,
    ) -> Response:
        """Handle Anthropic POST /v1/messages/batches endpoint with compression.

        Anthropic batch format:
        {
            "requests": [
                {
                    "custom_id": "req-1",
                    "params": {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                },
                ...
            ]
        }

        This method applies compression to each request's messages before forwarding.
        """
        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "type": "error",
                    "error": {
                        "type": "request_too_large",
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                    },
                },
            )

        # Parse request
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid JSON in request body: {e!s}",
                    },
                },
            )

        requests_list = body.get("requests", [])
        if not requests_list:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Missing or empty 'requests' field in batch request",
                    },
                },
            )

        # Extract headers
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)

        # Track compression stats across all batch requests
        total_original_tokens = 0
        total_optimized_tokens = 0
        total_tokens_saved = 0
        compressed_requests = []

        # Apply compression to each request in the batch
        for batch_req in requests_list:
            custom_id = batch_req.get("custom_id", "")
            params = batch_req.get("params", {})
            messages = params.get("messages", [])
            model = params.get("model", "unknown")

            if not messages or not self.config.optimize:
                # No messages or optimization disabled - pass through unchanged
                compressed_requests.append(batch_req)
                continue

            # Count original tokens
            tokenizer = get_tokenizer(model)
            original_tokens = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
            total_original_tokens += original_tokens

            # Apply optimization
            try:
                context_limit = self.anthropic_provider.get_context_limit(model)
                result = self.anthropic_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                )

                optimized_messages = result.messages
                optimized_tokens = sum(
                    tokenizer.count_text(str(m.get("content", ""))) for m in optimized_messages
                )
                total_optimized_tokens += optimized_tokens
                tokens_saved = original_tokens - optimized_tokens
                total_tokens_saved += tokens_saved

                # CCR Tool Injection: Inject retrieval tool if compression occurred
                tools = params.get("tools")
                if self.config.ccr_inject_tool and tokens_saved > 0:
                    injector = CCRToolInjector(
                        provider="anthropic",
                        inject_tool=True,
                        inject_system_instructions=self.config.ccr_inject_system_instructions,
                    )
                    optimized_messages, tools, was_injected = injector.process_request(
                        optimized_messages, tools
                    )
                    if was_injected:
                        logger.debug(
                            f"[{request_id}] CCR: Injected retrieval tool for batch request '{custom_id}'"
                        )

                # Create compressed batch request
                compressed_params = {**params, "messages": optimized_messages}
                if tools is not None:
                    compressed_params["tools"] = tools
                compressed_requests.append(
                    {
                        "custom_id": custom_id,
                        "params": compressed_params,
                    }
                )

                if tokens_saved > 0:
                    logger.debug(
                        f"[{request_id}] Batch request '{custom_id}': "
                        f"{original_tokens:,} -> {optimized_tokens:,} tokens "
                        f"(saved {tokens_saved:,})"
                    )

            except Exception as e:
                logger.warning(
                    f"[{request_id}] Optimization failed for batch request '{custom_id}': {e}"
                )
                # Pass through unchanged on failure
                compressed_requests.append(batch_req)
                total_optimized_tokens += original_tokens

        # Update body with compressed requests
        body["requests"] = compressed_requests

        optimization_latency = (time.time() - start_time) * 1000

        # Forward request to Anthropic
        url = f"{self.ANTHROPIC_API_URL}/v1/messages/batches"

        try:
            response = await self._retry_request("POST", url, headers, body)

            # Record metrics
            await self.metrics.record_request(
                provider="anthropic",
                model="batch",
                input_tokens=total_optimized_tokens,
                output_tokens=0,
                tokens_saved=total_tokens_saved,
                latency_ms=optimization_latency,
            )

            # Log compression stats
            if total_tokens_saved > 0:
                savings_percent = (
                    (total_tokens_saved / total_original_tokens * 100)
                    if total_original_tokens > 0
                    else 0
                )
                logger.info(
                    f"[{request_id}] Batch ({len(compressed_requests)} requests): "
                    f"{total_original_tokens:,} -> {total_optimized_tokens:,} tokens "
                    f"(saved {total_tokens_saved:,}, {savings_percent:.1f}%)"
                )

            # Store batch context for CCR result processing
            if response.status_code == 200 and self.config.ccr_inject_tool:
                try:
                    response_data = response.json()
                    batch_id = response_data.get("id")
                    if batch_id:
                        await self._store_anthropic_batch_context(
                            batch_id,
                            requests_list,
                            headers.get("x-api-key"),
                        )
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to store batch context: {e}")

            # Remove compression headers
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        except Exception as e:
            await self.metrics.record_failed()
            logger.error(f"[{request_id}] Batch request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "An error occurred while processing your batch request. Please try again.",
                    },
                },
            )

    async def handle_anthropic_batch_passthrough(
        self,
        request: Request,
        batch_id: str | None = None,
    ) -> Response:
        """Handle Anthropic batch passthrough endpoints.

        Used for:
        - GET /v1/messages/batches - List batches
        - GET /v1/messages/batches/{batch_id} - Get batch
        - GET /v1/messages/batches/{batch_id}/results - Get batch results
        - POST /v1/messages/batches/{batch_id}/cancel - Cancel batch
        """
        start_time = time.time()
        path = request.url.path
        url = f"{self.ANTHROPIC_API_URL}{path}"

        # Preserve query string parameters (e.g., limit, after_id for list endpoint)
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)

        body = await request.body()

        response = await self.http_client.request(  # type: ignore[union-attr]
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="anthropic",
            model="passthrough:batches",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        # Remove compression headers
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def _store_anthropic_batch_context(
        self,
        batch_id: str,
        requests_list: list[dict[str, Any]],
        api_key: str | None,
    ) -> None:
        """Store batch context for CCR result processing.

        Args:
            batch_id: The batch ID from the API response.
            requests_list: The original batch requests.
            api_key: The API key for continuation calls.
        """
        store = get_batch_context_store()
        context = BatchContext(
            batch_id=batch_id,
            provider="anthropic",
            api_key=api_key,
            api_base_url=self.ANTHROPIC_API_URL,
        )

        for batch_req in requests_list:
            custom_id = batch_req.get("custom_id", "")
            params = batch_req.get("params", {})
            context.add_request(
                BatchRequestContext(
                    custom_id=custom_id,
                    messages=params.get("messages", []),
                    tools=params.get("tools"),
                    model=params.get("model", ""),
                    extras={
                        "max_tokens": params.get("max_tokens", 4096),
                        "system": params.get("system"),
                    },
                )
            )

        await store.store(context)
        logger.debug(f"Stored batch context for {batch_id} with {len(requests_list)} requests")

    async def handle_anthropic_batch_results(
        self,
        request: Request,
        batch_id: str,
    ) -> Response:
        """Handle Anthropic batch results with CCR post-processing.

        This endpoint:
        1. Fetches raw results from Anthropic
        2. Detects CCR tool calls in each result
        3. Executes retrieval and makes continuation calls
        4. Returns processed results with complete responses
        """
        start_time = time.time()

        # Forward request to get raw results
        url = f"{self.ANTHROPIC_API_URL}/v1/messages/batches/{batch_id}/results"

        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)

        response = await self.http_client.get(url, headers=headers)  # type: ignore[union-attr]

        if response.status_code != 200:
            # Error - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Parse results - Anthropic batch results are JSONL format
        raw_content = response.content.decode("utf-8")
        results = []
        for line in raw_content.strip().split("\n"):
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not results:
            # No results to process
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Check if we have context and CCR processing is enabled
        store = get_batch_context_store()
        batch_context = await store.get(batch_id)

        if batch_context is None or not self.config.ccr_inject_tool:
            # No context or CCR disabled - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Process results with CCR handler
        processor = BatchResultProcessor(self.http_client)  # type: ignore[arg-type]
        processed = await processor.process_results(batch_id, results, "anthropic")

        # Convert back to JSONL format
        processed_lines = []
        for p in processed:
            processed_lines.append(json.dumps(p.result))
            if p.was_processed:
                logger.info(
                    f"CCR: Processed batch result {p.custom_id} "
                    f"({p.continuation_rounds} continuation rounds)"
                )

        processed_content = "\n".join(processed_lines)

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="anthropic",
            model="batch:ccr-processed",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        return Response(
            content=processed_content.encode("utf-8"),
            status_code=200,
            media_type="application/jsonl",
        )

    # =========================================================================
    # Google/Gemini Batch API Handlers
    # =========================================================================

    async def handle_google_batch_create(
        self,
        request: Request,
        model: str,
    ) -> Response:
        """Handle Google POST /v1beta/models/{model}:batchGenerateContent endpoint.

        Google batch format:
        {
            "batch": {
                "display_name": "my-batch",
                "input_config": {
                    "requests": {
                        "requests": [
                            {
                                "request": {"contents": [{"parts": [{"text": "..."}]}]},
                                "metadata": {"key": "request-1"}
                            }
                        ]
                    }
                }
            }
        }

        This method applies compression to each request's contents before forwarding.
        """
        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": 413,
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )

        # Parse request
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": f"Invalid JSON in request body: {e!s}",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )

        # Extract batch config
        batch_config = body.get("batch", {})
        input_config = batch_config.get("input_config", {})
        requests_wrapper = input_config.get("requests", {})
        requests_list = requests_wrapper.get("requests", [])

        if not requests_list:
            # No inline requests - might be using file input, pass through
            logger.debug(f"[{request_id}] Google batch: No inline requests, passing through")
            return await self._google_batch_passthrough(request, model, body)

        # Extract headers
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)

        # Track compression stats
        total_original_tokens = 0
        total_optimized_tokens = 0
        total_tokens_saved = 0
        compressed_requests = []

        # Apply compression to each request in the batch
        for idx, batch_req in enumerate(requests_list):
            req_content = batch_req.get("request", {})
            metadata = batch_req.get("metadata", {})
            contents = req_content.get("contents", [])

            if not contents or not self.config.optimize:
                # No contents or optimization disabled - pass through unchanged
                compressed_requests.append(batch_req)
                continue

            # Convert Google format to messages for compression
            system_instruction = req_content.get("systemInstruction")
            messages, preserved_indices = self._gemini_contents_to_messages(
                contents, system_instruction
            )

            # Store original content entries that have non-text parts before compression
            preserved_contents = {idx: contents[idx] for idx in preserved_indices}

            # Early exit if ALL content has non-text parts (nothing to compress)
            if len(preserved_indices) == len(contents):
                # All content has non-text parts, skip compression
                compressed_requests.append(batch_req)
                continue

            # Count original tokens
            tokenizer = get_tokenizer(model)
            original_tokens = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
            total_original_tokens += original_tokens

            # Apply optimization
            try:
                # Default context limit for most models
                context_limit = 128000

                # Use OpenAI pipeline (similar message format after conversion)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                )

                optimized_messages = result.messages
                optimized_tokens = sum(
                    tokenizer.count_text(str(m.get("content", ""))) for m in optimized_messages
                )
                total_optimized_tokens += optimized_tokens
                tokens_saved = original_tokens - optimized_tokens
                total_tokens_saved += tokens_saved

                # CCR Tool Injection: Inject retrieval tool if compression occurred
                tools = req_content.get("tools")
                # Extract existing function declarations if present
                existing_funcs = None
                if tools:
                    for tool in tools:
                        if "functionDeclarations" in tool:
                            existing_funcs = tool["functionDeclarations"]
                            break

                if self.config.ccr_inject_tool and tokens_saved > 0:
                    injector = CCRToolInjector(
                        provider="google",
                        inject_tool=True,
                        inject_system_instructions=self.config.ccr_inject_system_instructions,
                    )
                    optimized_messages, injected_funcs, was_injected = injector.process_request(
                        optimized_messages, existing_funcs
                    )
                    if was_injected:
                        logger.debug(
                            f"[{request_id}] CCR: Injected retrieval tool for Google batch request {idx}"
                        )
                        existing_funcs = injected_funcs

                # Convert back to Google contents format
                optimized_contents, optimized_sys_inst = self._messages_to_gemini_contents(
                    optimized_messages
                )

                # Restore preserved content entries that had non-text parts
                for orig_idx, original_content in preserved_contents.items():
                    if orig_idx < len(optimized_contents):
                        optimized_contents[orig_idx] = original_content

                # Create compressed batch request
                compressed_req_content = {**req_content, "contents": optimized_contents}
                if optimized_sys_inst:
                    compressed_req_content["systemInstruction"] = optimized_sys_inst
                if existing_funcs is not None:
                    compressed_req_content["tools"] = [{"functionDeclarations": existing_funcs}]

                compressed_req = {
                    "request": compressed_req_content,
                    "metadata": metadata,
                }

                compressed_requests.append(compressed_req)

                if tokens_saved > 0:
                    logger.debug(
                        f"[{request_id}] Google batch request {idx}: "
                        f"{original_tokens:,} -> {optimized_tokens:,} tokens "
                        f"(saved {tokens_saved:,})"
                    )

            except Exception as e:
                logger.warning(
                    f"[{request_id}] Optimization failed for Google batch request {idx}: {e}"
                )
                # Pass through unchanged on failure
                compressed_requests.append(batch_req)
                total_optimized_tokens += original_tokens

        # Update body with compressed requests
        body["batch"]["input_config"]["requests"]["requests"] = compressed_requests

        optimization_latency = (time.time() - start_time) * 1000

        # Forward request to Google
        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:batchGenerateContent"

        # Add API key to URL if present in headers
        api_key = headers.pop("x-goog-api-key", None)
        if api_key:
            url = f"{url}?key={api_key}"

        try:
            response = await self._retry_request("POST", url, headers, body)

            # Record metrics
            await self.metrics.record_request(
                provider="google",
                model=f"batch:{model}",
                input_tokens=total_optimized_tokens,
                output_tokens=0,
                tokens_saved=total_tokens_saved,
                latency_ms=optimization_latency,
            )

            # Log compression stats
            if total_tokens_saved > 0:
                savings_percent = (
                    (total_tokens_saved / total_original_tokens * 100)
                    if total_original_tokens > 0
                    else 0
                )
                logger.info(
                    f"[{request_id}] Google batch compression: "
                    f"{total_original_tokens:,} -> {total_optimized_tokens:,} tokens "
                    f"({savings_percent:.1f}% saved across {len(requests_list)} requests)"
                )

            # Store batch context for CCR result processing
            if response.status_code == 200 and self.config.ccr_inject_tool:
                try:
                    response_data = response.json()
                    batch_name = response_data.get("name")
                    if batch_name:
                        await self._store_google_batch_context(
                            batch_name,
                            requests_list,
                            model,
                            api_key,
                        )
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to store Google batch context: {e}")

            # Remove compression headers
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        except Exception as e:
            logger.error(f"[{request_id}] Google batch request failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": f"Failed to forward batch request: {e!s}",
                        "status": "INTERNAL",
                    }
                },
            )

    async def _google_batch_passthrough(
        self,
        request: Request,
        model: str,
        body: dict | None = None,
    ) -> Response:
        """Pass through Google batch request without modification."""
        start_time = time.time()

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)

        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:batchGenerateContent"

        # Add API key to URL if present in headers
        api_key = headers.pop("x-goog-api-key", None)
        if api_key:
            url = f"{url}?key={api_key}"

        if body is None:
            body_content = await request.body()
        else:
            body_content = json.dumps(body).encode()

        response = await self.http_client.post(  # type: ignore[union-attr]
            url,
            headers=headers,
            content=body_content,
        )

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="google",
            model=f"passthrough:batch:{model}",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_google_batch_passthrough(
        self,
        request: Request,
        batch_name: str | None = None,
    ) -> Response:
        """Handle Google batch passthrough endpoints.

        Used for:
        - GET /v1beta/batches/{batch_name} - Get batch status
        - POST /v1beta/batches/{batch_name}:cancel - Cancel batch
        - DELETE /v1beta/batches/{batch_name} - Delete batch
        """
        start_time = time.time()
        path = request.url.path
        url = f"{self.GEMINI_API_URL}{path}"

        # Preserve query string parameters
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)

        # Handle API key
        api_key = headers.pop("x-goog-api-key", None)
        if api_key:
            if "?" in url:
                url = f"{url}&key={api_key}"
            else:
                url = f"{url}?key={api_key}"

        body = await request.body()

        response = await self.http_client.request(  # type: ignore[union-attr]
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="google",
            model="passthrough:batches",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def _store_google_batch_context(
        self,
        batch_name: str,
        requests_list: list[dict[str, Any]],
        model: str,
        api_key: str | None,
    ) -> None:
        """Store Google batch context for CCR result processing.

        Args:
            batch_name: The batch name from the API response.
            requests_list: The original batch requests.
            model: The model used for the batch.
            api_key: The API key for continuation calls.
        """
        store = get_batch_context_store()
        context = BatchContext(
            batch_id=batch_name,
            provider="google",
            api_key=api_key,
            api_base_url=self.GEMINI_API_URL,
        )

        for batch_req in requests_list:
            metadata = batch_req.get("metadata", {})
            custom_id = metadata.get("key", "")
            req_content = batch_req.get("request", {})
            contents = req_content.get("contents", [])
            system_instruction = req_content.get("systemInstruction")

            # Convert contents to messages format for CCR handler
            messages, _ = self._gemini_contents_to_messages(contents, system_instruction)

            # Extract system instruction text if present
            sys_text = None
            if system_instruction:
                parts = system_instruction.get("parts", [])
                if parts and isinstance(parts[0], dict):
                    sys_text = parts[0].get("text")

            context.add_request(
                BatchRequestContext(
                    custom_id=custom_id,
                    messages=messages,
                    tools=req_content.get("tools"),
                    model=model,
                    system_instruction=sys_text,
                )
            )

        await store.store(context)
        logger.debug(
            f"Stored Google batch context for {batch_name} with {len(requests_list)} requests"
        )

    async def handle_google_batch_results(
        self,
        request: Request,
        batch_name: str,
    ) -> Response:
        """Handle Google batch results with CCR post-processing.

        Google batch results endpoint returns the batch operation status.
        When status is SUCCEEDED, results are embedded in the response.
        This handler processes CCR tool calls in those results.
        """
        start_time = time.time()

        # Forward request to get batch status/results
        url = f"{self.GEMINI_API_URL}/v1beta/{batch_name}"

        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)

        # Handle API key
        api_key = headers.pop("x-goog-api-key", None)
        if api_key:
            if "?" in url:
                url = f"{url}&key={api_key}"
            else:
                url = f"{url}?key={api_key}"

        response = await self.http_client.get(url, headers=headers)  # type: ignore[union-attr]

        if response.status_code != 200:
            # Error - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Parse response
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            # Not JSON - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Check if batch has results (state must be SUCCEEDED)
        metadata = response_data.get("metadata", {})
        state = metadata.get("state")

        if state != "SUCCEEDED":
            # Batch not complete - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Extract results from response
        # Google embeds results in the batch response
        results = response_data.get("response", {}).get("responses", [])

        if not results:
            # No results to process
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Check if we have context and CCR processing is enabled
        store = get_batch_context_store()
        batch_context = await store.get(batch_name)

        if batch_context is None or not self.config.ccr_inject_tool:
            # No context or CCR disabled - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Process results with CCR handler
        processor = BatchResultProcessor(self.http_client)  # type: ignore[arg-type]
        processed = await processor.process_results(batch_name, results, "google")

        # Update response with processed results
        processed_results = [p.result for p in processed]
        response_data["response"]["responses"] = processed_results

        for p in processed:
            if p.was_processed:
                logger.info(
                    f"CCR: Processed Google batch result {p.custom_id} "
                    f"({p.continuation_rounds} continuation rounds)"
                )

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="google",
            model="batch:ccr-processed",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        return JSONResponse(content=response_data, status_code=200)

    def _parse_sse_usage(self, chunk: bytes, provider: str) -> dict[str, int] | None:
        """Parse usage information from SSE chunk.

        For Anthropic: Looks for message_start (input tokens) and message_delta (output tokens)
        For OpenAI: Looks for final chunk with usage object (requires stream_options.include_usage=true)
        For Gemini: Looks for usageMetadata in each chunk

        Returns dict with keys: input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens
        Returns None if no usage found in this chunk.
        """
        try:
            text = chunk.decode("utf-8", errors="ignore")
            # SSE format: "data: {...}\n\n" or "event: ...\ndata: {...}\n\n"
            for line in text.split("\n"):
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = {}

                if provider == "anthropic":
                    # Anthropic sends message_start with input tokens
                    # and message_delta with output tokens
                    event_type = data.get("type", "")

                    if event_type == "message_start":
                        msg = data.get("message", {})
                        msg_usage = msg.get("usage", {})
                        if msg_usage:
                            usage["input_tokens"] = msg_usage.get("input_tokens", 0)
                            usage["cache_read_input_tokens"] = msg_usage.get(
                                "cache_read_input_tokens", 0
                            )
                            usage["cache_creation_input_tokens"] = msg_usage.get(
                                "cache_creation_input_tokens", 0
                            )

                    elif event_type == "message_delta":
                        delta_usage = data.get("usage", {})
                        if delta_usage:
                            usage["output_tokens"] = delta_usage.get("output_tokens", 0)

                elif provider == "openai":
                    # OpenAI sends usage in final chunk (when stream_options.include_usage=true)
                    chunk_usage = data.get("usage")
                    if chunk_usage:
                        usage["input_tokens"] = chunk_usage.get("prompt_tokens", 0)
                        usage["output_tokens"] = chunk_usage.get("completion_tokens", 0)
                        # OpenAI has cached tokens in prompt_tokens_details
                        details = chunk_usage.get("prompt_tokens_details", {})
                        usage["cache_read_input_tokens"] = details.get("cached_tokens", 0)

                elif provider == "gemini":
                    # Gemini sends usageMetadata in each streaming chunk
                    # Format: {"usageMetadata": {"promptTokenCount": N, "candidatesTokenCount": M}}
                    usage_meta = data.get("usageMetadata")
                    if usage_meta:
                        usage["input_tokens"] = usage_meta.get("promptTokenCount", 0)
                        usage["output_tokens"] = usage_meta.get("candidatesTokenCount", 0)
                        # Gemini also has cachedContentTokenCount for context caching
                        usage["cache_read_input_tokens"] = usage_meta.get(
                            "cachedContentTokenCount", 0
                        )

                if usage:
                    return usage

        except (UnicodeDecodeError, KeyError, TypeError) as e:
            # Don't fail streaming on parse errors
            logger.debug(f"SSE usage parsing error for {provider}: {e}")

        return None

    def _parse_sse_usage_from_buffer(
        self, stream_state: dict[str, Any], provider: str
    ) -> dict[str, int] | None:
        """Parse usage from buffered SSE data, handling split chunks.

        Processes complete SSE events (ending with double newline) from the buffer
        and removes them from the buffer. Incomplete events are kept in the buffer
        for the next chunk.
        """
        buffer = stream_state["sse_buffer"]
        usage_found: dict[str, int] = {}

        # Process complete SSE events (separated by double newlines)
        while "\n\n" in buffer:
            event_end = buffer.index("\n\n")
            event_text = buffer[: event_end + 2]
            buffer = buffer[event_end + 2 :]

            # Parse this complete event
            for line in event_text.split("\n"):
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if provider == "anthropic":
                    event_type = data.get("type", "")
                    if event_type == "message_start":
                        msg = data.get("message", {})
                        msg_usage = msg.get("usage", {})
                        if msg_usage:
                            usage_found["input_tokens"] = msg_usage.get("input_tokens", 0)
                            usage_found["cache_read_input_tokens"] = msg_usage.get(
                                "cache_read_input_tokens", 0
                            )
                            usage_found["cache_creation_input_tokens"] = msg_usage.get(
                                "cache_creation_input_tokens", 0
                            )
                            # INFO logging for cache token tracking (temporary for debugging)
                            logger.info(
                                f"[CACHE] Anthropic usage: input={usage_found.get('input_tokens')}, "
                                f"cache_read={usage_found.get('cache_read_input_tokens')}, "
                                f"cache_write={usage_found.get('cache_creation_input_tokens')}"
                            )
                    elif event_type == "message_delta":
                        delta_usage = data.get("usage", {})
                        if delta_usage:
                            usage_found["output_tokens"] = delta_usage.get("output_tokens", 0)

                elif provider == "openai":
                    chunk_usage = data.get("usage")
                    if chunk_usage:
                        usage_found["input_tokens"] = chunk_usage.get("prompt_tokens", 0)
                        usage_found["output_tokens"] = chunk_usage.get("completion_tokens", 0)
                        details = chunk_usage.get("prompt_tokens_details", {})
                        usage_found["cache_read_input_tokens"] = details.get("cached_tokens", 0)

                elif provider == "gemini":
                    usage_meta = data.get("usageMetadata")
                    if usage_meta:
                        usage_found["input_tokens"] = usage_meta.get("promptTokenCount", 0)
                        usage_found["output_tokens"] = usage_meta.get("candidatesTokenCount", 0)
                        usage_found["cache_read_input_tokens"] = usage_meta.get(
                            "cachedContentTokenCount", 0
                        )

        # Update buffer with remaining incomplete data
        stream_state["sse_buffer"] = buffer

        return usage_found if usage_found else None

    def _parse_sse_to_response(self, sse_data: str, provider: str) -> dict[str, Any] | None:
        """Parse SSE data to reconstruct the API response JSON.

        Args:
            sse_data: Raw SSE data string.
            provider: Provider type for parsing.

        Returns:
            Reconstructed response dict or None if parsing fails.
        """
        if provider != "anthropic":
            return None  # Only implemented for Anthropic

        response: dict[str, Any] = {"content": [], "usage": {}}
        current_block: dict[str, Any] | None = None

        for line in sse_data.split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            if event_type == "message_start":
                msg = data.get("message", {})
                response["id"] = msg.get("id")
                response["model"] = msg.get("model")
                response["role"] = msg.get("role", "assistant")
                response["stop_reason"] = msg.get("stop_reason")
                if msg.get("usage"):
                    response["usage"].update(msg["usage"])

            elif event_type == "content_block_start":
                block = data.get("content_block", {})
                current_block = {
                    "type": block.get("type"),
                    "index": data.get("index", len(response["content"])),
                }
                if block.get("type") == "text":
                    current_block["text"] = block.get("text", "")
                elif block.get("type") == "tool_use":
                    current_block["id"] = block.get("id")
                    current_block["name"] = block.get("name")
                    current_block["input"] = {}

            elif event_type == "content_block_delta":
                if current_block:
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        current_block["text"] = current_block.get("text", "") + delta.get(
                            "text", ""
                        )
                    elif delta.get("type") == "input_json_delta":
                        # Accumulate partial JSON for tool input
                        partial = delta.get("partial_json", "")
                        current_block["_partial_json"] = (
                            current_block.get("_partial_json", "") + partial
                        )

            elif event_type == "content_block_stop":
                if current_block:
                    # Parse accumulated JSON for tool_use blocks
                    if current_block.get("type") == "tool_use" and "_partial_json" in current_block:
                        try:
                            current_block["input"] = json.loads(current_block["_partial_json"])
                        except json.JSONDecodeError:
                            current_block["input"] = {}
                        del current_block["_partial_json"]
                    response["content"].append(current_block)
                    current_block = None

            elif event_type == "message_delta":
                delta = data.get("delta", {})
                if delta.get("stop_reason"):
                    response["stop_reason"] = delta["stop_reason"]
                if data.get("usage"):
                    response["usage"].update(data["usage"])

        return response if response.get("content") else None

    def _response_to_sse(self, response: dict[str, Any], provider: str) -> list[bytes]:
        """Convert a response dict back to SSE format.

        Args:
            response: API response dict.
            provider: Provider type for formatting.

        Returns:
            List of SSE event bytes.
        """
        if provider != "anthropic":
            return []

        events: list[bytes] = []

        # message_start
        msg_start = {
            "type": "message_start",
            "message": {
                "id": response.get("id", "msg_generated"),
                "type": "message",
                "role": response.get("role", "assistant"),
                "model": response.get("model", "unknown"),
                "content": [],
                "stop_reason": None,
                "usage": response.get("usage", {}),
            },
        }
        events.append(f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode())

        # Content blocks
        for idx, block in enumerate(response.get("content", [])):
            # content_block_start
            if block.get("type") == "text":
                block_start = {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                }
            elif block.get("type") == "tool_use":
                block_start = {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id", f"toolu_{idx}"),
                        "name": block.get("name", ""),
                        "input": {},
                    },
                }
            else:
                continue

            events.append(
                f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode()
            )

            # content_block_delta(s)
            if block.get("type") == "text" and block.get("text"):
                delta = {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": block["text"]},
                }
                events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())
            elif block.get("type") == "tool_use" and block.get("input"):
                delta = {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                }
                events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())

            # content_block_stop
            block_stop = {"type": "content_block_stop", "index": idx}
            events.append(f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n".encode())

        # message_delta
        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": response.get("stop_reason", "end_turn")},
            "usage": {"output_tokens": response.get("usage", {}).get("output_tokens", 0)},
        }
        events.append(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode())

        # message_stop
        events.append(b'event: message_stop\ndata: {"type": "message_stop"}\n\n')

        return events

    async def _stream_response(
        self,
        url: str,
        headers: dict,
        body: dict,
        provider: str,
        model: str,
        request_id: str,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: dict[str, str],
        optimization_latency: float,
        memory_user_id: str | None = None,
    ) -> StreamingResponse:
        """Stream response with metrics tracking and memory tool handling.

        Parses SSE events to extract actual usage information from the API response
        for accurate token counting and cost calculation.

        When memory is enabled (memory_user_id provided), this method:
        1. Buffers the response to detect memory tool calls
        2. Executes memory tools if found
        3. Makes continuation requests until no memory tools remain
        4. Streams the final response to the client
        """
        start_time = time.time()

        # Mutable state for the generator to update
        stream_state: dict[str, Any] = {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "total_bytes": 0,
            "sse_buffer": "",  # Buffer for incomplete SSE events
        }

        # Track if we need to handle memory tools
        memory_enabled = (
            memory_user_id is not None
            and self.memory_handler is not None
            and provider == "anthropic"
        )

        async def generate():
            nonlocal body  # May need to modify for continuation requests

            # For memory mode, we buffer the response to check for tool calls
            buffered_chunks: list[bytes] = []
            full_sse_data = ""

            try:
                async with self.http_client.stream(
                    "POST", url, json=body, headers=headers
                ) as response:
                    async for chunk in response.aiter_bytes():
                        stream_state["total_bytes"] += len(chunk)

                        # Buffer SSE data to handle chunks split across calls
                        chunk_str = chunk.decode("utf-8", errors="ignore")
                        stream_state["sse_buffer"] += chunk_str

                        if memory_enabled:
                            # Buffer for memory tool detection
                            buffered_chunks.append(chunk)
                            full_sse_data += chunk_str
                        else:
                            # Immediate streaming when memory not enabled
                            yield chunk

                        # Parse complete SSE events from buffer
                        usage = self._parse_sse_usage_from_buffer(stream_state, provider)
                        if usage:
                            if "input_tokens" in usage:
                                stream_state["input_tokens"] = usage["input_tokens"]
                            if "output_tokens" in usage:
                                stream_state["output_tokens"] = usage["output_tokens"]
                            if "cache_read_input_tokens" in usage:
                                stream_state["cache_read_input_tokens"] = usage[
                                    "cache_read_input_tokens"
                                ]
                            if "cache_creation_input_tokens" in usage:
                                stream_state["cache_creation_input_tokens"] = usage[
                                    "cache_creation_input_tokens"
                                ]

                # Memory tool handling after stream completes
                if memory_enabled and full_sse_data:
                    # Check for Claude Code credential error in initial response
                    if "only authorized for use with Claude Code" in full_sse_data:
                        logger.warning(
                            f"[{request_id}] Memory: Claude Code subscription credentials "
                            "do not support custom tool injection. Set ANTHROPIC_API_KEY "
                            "environment variable or use --no-memory-tools flag."
                        )
                        # Yield buffered error response as-is (contains error details)
                        for chunk in buffered_chunks:
                            yield chunk
                        return

                    # Parse SSE to get response JSON
                    parsed_response = self._parse_sse_to_response(full_sse_data, provider)

                    if parsed_response and self.memory_handler.has_memory_tool_calls(
                        parsed_response, provider
                    ):
                        logger.info(
                            f"[{request_id}] Memory: Detected tool calls in streaming response"
                        )

                        # Execute memory tool calls
                        tool_results = await self.memory_handler.handle_memory_tool_calls(
                            parsed_response, memory_user_id, provider
                        )

                        if tool_results:
                            # Build continuation messages
                            # Filter out system role messages (Anthropic uses top-level 'system' param)
                            messages = [
                                m for m in body.get("messages", []) if m.get("role") != "system"
                            ]
                            assistant_msg = {
                                "role": "assistant",
                                "content": parsed_response.get("content", []),
                            }
                            user_msg = {"role": "user", "content": tool_results}
                            continuation_messages = messages + [assistant_msg, user_msg]

                            # Make continuation request (streaming to support Claude Code API key)
                            continuation_body = {
                                **body,
                                "messages": continuation_messages,
                                "stream": True,
                            }

                            logger.info(
                                f"[{request_id}] Memory: Tool execution complete, streaming continuation"
                            )

                            # Stream continuation response directly
                            async with self.http_client.stream(
                                "POST", url, json=continuation_body, headers=headers
                            ) as cont_response:
                                cont_buffer = ""
                                async for chunk in cont_response.aiter_bytes():
                                    chunk_str = chunk.decode("utf-8", errors="ignore")
                                    cont_buffer += chunk_str

                                    # Check for Claude Code credential error
                                    if "only authorized for use with Claude Code" in cont_buffer:
                                        logger.warning(
                                            f"[{request_id}] Memory: Claude Code subscription "
                                            "credentials do not support custom tool injection. "
                                            "Set ANTHROPIC_API_KEY environment variable to use "
                                            "memory tools, or disable memory tools with "
                                            "--no-memory-tools flag."
                                        )
                                        # Yield a helpful error message in SSE format
                                        error_msg = (
                                            "Memory tools require a regular Anthropic API key. "
                                            "Claude Code subscription credentials do not allow "
                                            "custom tool injection. "
                                            "To fix: (1) Set ANTHROPIC_API_KEY=your_api_key before "
                                            "starting the proxy, or (2) Run proxy with "
                                            "--no-memory-tools flag."
                                        )
                                        error_event = (
                                            f'data: {{"type":"error","error":{{"type":"permission_error",'
                                            f'"message":"{error_msg}"}}}}\n\n'
                                        )
                                        yield error_event.encode()
                                        return

                                    yield chunk
                        else:
                            # No tool results, yield original buffered chunks
                            for chunk in buffered_chunks:
                                yield chunk
                    else:
                        # No memory tool calls, yield original buffered chunks
                        for chunk in buffered_chunks:
                            yield chunk
            finally:
                # Record metrics after stream completes
                total_latency = (time.time() - start_time) * 1000

                # Use actual output tokens from API if available, otherwise estimate
                output_tokens = stream_state["output_tokens"]
                if output_tokens is None:
                    # Fallback: estimate from bytes (but this is inaccurate for SSE)
                    # Use a more conservative estimate - SSE overhead is ~10-20x
                    output_tokens = stream_state["total_bytes"] // 40
                    logger.debug(
                        f"[{request_id}] No usage in stream, estimated {output_tokens} output tokens"
                    )

                # Use actual tokens from API if available, fallback to estimates
                # Note: use 'is not None' instead of 'or' to handle 0 correctly
                api_input_tokens = stream_state["input_tokens"]
                total_input_tokens = (
                    api_input_tokens if api_input_tokens is not None else optimized_tokens
                )
                cache_read_tokens = stream_state["cache_read_input_tokens"]
                cache_write_tokens = stream_state["cache_creation_input_tokens"]

                # INFO logging for token tracking (temporary for debugging)
                if api_input_tokens is None:
                    logger.info(
                        f"[{request_id}] No input_tokens from API, using estimate: {optimized_tokens}"
                    )
                else:
                    logger.info(
                        f"[{request_id}] Final tokens: input={api_input_tokens}, "
                        f"cache_read={cache_read_tokens}, cache_write={cache_write_tokens}"
                    )

                # Normalize input tokens based on provider semantics:
                # - Anthropic: input_tokens excludes cache_read (it's separate), pass as-is
                # - OpenAI/Gemini: input_tokens includes cache_read (it's a subset), subtract it
                if provider == "anthropic":
                    # Anthropic's input_tokens = non-cached tokens sent (excludes cache_read)
                    non_cached_input = total_input_tokens
                else:
                    # OpenAI/Gemini's input_tokens = total (includes cache_read)
                    non_cached_input = total_input_tokens - cache_read_tokens

                # Calculate cost using actual API tokens with proper cache pricing
                cost_usd = None
                savings_usd = None
                if self.cost_tracker:
                    cost_usd = self.cost_tracker.estimate_cost(
                        model,
                        non_cached_input,
                        output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                    )
                    # For savings calculation, compare compression benefit using base token rates only
                    # (cache effects are Anthropic's feature, not Headroom's compression benefit)
                    compressed_base_cost = self.cost_tracker.estimate_cost(
                        model,
                        non_cached_input,
                        output_tokens,
                    )
                    original_base_cost = self.cost_tracker.estimate_cost(
                        model,
                        original_tokens,
                        output_tokens,
                    )
                    if cost_usd:
                        self.cost_tracker.record_cost(cost_usd)
                    if compressed_base_cost and original_base_cost:
                        savings_usd = original_base_cost - compressed_base_cost
                        self.cost_tracker.record_savings(max(0, savings_usd))

                await self.metrics.record_request(
                    provider=provider,
                    model=model,
                    input_tokens=total_input_tokens,  # Record total for accurate tracking
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    cost_usd=cost_usd or 0,
                    savings_usd=savings_usd or 0,
                )

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: saved {tokens_saved:,} tokens (streaming)"
                    )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    async def _stream_response_bedrock(
        self,
        body: dict,
        headers: dict,
        provider: str,
        model: str,
        request_id: str,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: dict[str, str],
        optimization_latency: float,
    ) -> StreamingResponse:
        """Stream response from Bedrock backend with metrics tracking.

        Translates Bedrock streaming events to Anthropic SSE format.
        """
        start_time = time.time()

        # Mutable state for the generator
        stream_state: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

        async def generate():
            try:
                assert self.anthropic_backend is not None

                async for event in self.anthropic_backend.stream_message(body, headers):
                    # Format as SSE
                    if event.raw_sse:
                        yield event.raw_sse.encode()
                    else:
                        sse_line = f"event: {event.event_type}\ndata: {json.dumps(event.data)}\n\n"
                        yield sse_line.encode()

                    # Track usage from message_start event
                    if event.event_type == "message_start":
                        msg = event.data.get("message", {})
                        usage = msg.get("usage", {})
                        if "input_tokens" in usage:
                            stream_state["input_tokens"] = usage["input_tokens"]

                    # Track output tokens from message_delta
                    if event.event_type == "message_delta":
                        usage = event.data.get("usage", {})
                        if "output_tokens" in usage:
                            stream_state["output_tokens"] = usage["output_tokens"]

                    # Handle errors
                    if event.event_type == "error":
                        logger.error(f"[{request_id}] Bedrock stream error: {event.data}")

            except Exception as e:
                logger.error(f"[{request_id}] Bedrock streaming error: {e}")
                error_event = {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()

            finally:
                # Record metrics
                total_latency = (time.time() - start_time) * 1000
                output_tokens = stream_state["output_tokens"]

                await self.metrics.record_request(
                    provider="bedrock",
                    model=model,
                    input_tokens=optimized_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    cached=False,
                )

                if self.cost_tracker:
                    cost_usd = self.cost_tracker.estimate_cost(
                        model, optimized_tokens, output_tokens
                    )
                    if cost_usd:
                        self.cost_tracker.record_cost(cost_usd)

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] Bedrock {model}: saved {tokens_saved:,} tokens (streaming)"
                    )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    async def handle_openai_chat(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle OpenAI /v1/chat/completions endpoint."""
        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "type": "invalid_request_error",
                        "code": "request_too_large",
                    }
                },
            )

        # Parse request
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON in request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )
        model = body.get("model", "unknown")
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Image compression (before text optimization)
        if self.config.image_optimize and messages:
            compressor = _get_image_compressor()
            if compressor and compressor.has_images(messages):
                messages = compressor.compress(messages, provider="openai")
                if compressor.last_result:
                    logger.info(
                        f"Image compression: {compressor.last_result.technique.value} "
                        f"({compressor.last_result.savings_percent:.0f}% saved, "
                        f"{compressor.last_result.original_tokens} -> "
                        f"{compressor.last_result.compressed_tokens} tokens)"
                    )

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)

        # Rate limiting
        if self.rate_limiter:
            rate_key = headers.get("authorization", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited()
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Check cache
        if self.cache and not stream:
            cached = await self.cache.get(messages, model)
            if cached:
                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_saved=cached.tokens_saved_per_hit,
                    latency_ms=(time.time() - start_time) * 1000,
                    cached=True,
                )

                # Remove compression headers from cached response
                response_headers = dict(cached.response_headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(content=cached.response_body, headers=response_headers)

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)

        # Optimization
        transforms_applied = []
        optimized_messages = messages
        optimized_tokens = original_tokens

        if self.config.optimize and messages:
            try:
                context_limit = self.openai_provider.get_context_limit(model)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    optimized_tokens = sum(
                        tokenizer.count_text(str(m.get("content", ""))) for m in optimized_messages
                    )
            except Exception as e:
                logger.warning(f"Optimization failed: {e}")

        tokens_saved = original_tokens - optimized_tokens
        optimization_latency = (time.time() - start_time) * 1000

        # CCR Tool Injection: Inject retrieval tool if compression occurred
        tools = body.get("tools")
        if self.config.ccr_inject_tool or self.config.ccr_inject_system_instructions:
            injector = CCRToolInjector(
                provider="openai",
                inject_tool=self.config.ccr_inject_tool,
                inject_system_instructions=self.config.ccr_inject_system_instructions,
            )
            optimized_messages, tools, was_injected = injector.process_request(
                optimized_messages, tools
            )

            if injector.has_compressed_content:
                if was_injected:
                    logger.debug(
                        f"[{request_id}] CCR: Injected retrieval tool for hashes: {injector.detected_hashes}"
                    )
                else:
                    logger.debug(
                        f"[{request_id}] CCR: Tool already present (MCP?), skipped injection for hashes: {injector.detected_hashes}"
                    )

        body["messages"] = optimized_messages
        if tools is not None:
            body["tools"] = tools
        url = f"{self.OPENAI_API_URL}/v1/chat/completions"

        try:
            if stream:
                # Inject stream_options to get usage stats in streaming response
                # This allows accurate token counting instead of byte-based estimation
                if "stream_options" not in body:
                    body["stream_options"] = {"include_usage": True}
                elif isinstance(body.get("stream_options"), dict):
                    body["stream_options"]["include_usage"] = True

                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "openai",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)
                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = optimized_tokens  # fallback
                output_tokens = 0
                cache_read_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    total_input_tokens = usage.get("prompt_tokens", optimized_tokens)
                    output_tokens = usage.get("completion_tokens", 0)
                    # OpenAI returns cached_tokens in prompt_tokens_details
                    # These are charged at 50% of the input price
                    prompt_details = usage.get("prompt_tokens_details", {})
                    cache_read_tokens = prompt_details.get("cached_tokens", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from OpenAI response: {e}"
                    )

                # For OpenAI, prompt_tokens is TOTAL (includes cached)
                # Normalize to non-cached input for consistent cost calculation
                non_cached_input = total_input_tokens - cache_read_tokens

                # Cost tracking using actual API tokens
                cost_usd = savings_usd = None
                if self.cost_tracker:
                    cost_usd = self.cost_tracker.estimate_cost(
                        model,
                        non_cached_input,  # Pass non-cached portion
                        output_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )
                    original_cost = self.cost_tracker.estimate_cost(
                        model,
                        original_tokens,
                        output_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )
                    if cost_usd and original_cost:
                        savings_usd = original_cost - cost_usd
                        self.cost_tracker.record_cost(cost_usd)
                        self.cost_tracker.record_savings(savings_usd)

                # Cache
                if self.cache and response.status_code == 200:
                    await self.cache.set(
                        messages, model, response.content, dict(response.headers), tokens_saved
                    )

                # Metrics with actual API tokens (total, for accurate tracking)
                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    cost_usd=cost_usd or 0,
                    savings_usd=savings_usd or 0,
                )

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens)"
                    )

                # Remove compression headers since httpx already decompressed the response
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)  # Length changed after decompression

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
        except Exception as e:
            await self.metrics.record_failed()
            # Log full error details internally for debugging
            logger.error(f"[{request_id}] OpenAI request failed: {type(e).__name__}: {e}")
            # Return sanitized error message to client (don't expose internal details)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "type": "server_error",
                        "code": "proxy_error",
                    }
                },
            )

    async def handle_passthrough(
        self,
        request: Request,
        base_url: str,
        endpoint_name: str | None = None,
        provider: str | None = None,
    ) -> Response:
        """Pass through request unchanged.

        Args:
            request: The incoming request
            base_url: The upstream API base URL
            endpoint_name: Optional name for stats tracking (e.g., "models", "embeddings")
            provider: Optional provider name for stats (e.g., "openai", "anthropic", "gemini")
        """
        start_time = time.time()
        path = request.url.path
        url = f"{base_url}{path}"

        # Preserve query string parameters
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)

        body = await request.body()

        response = await self.http_client.request(  # type: ignore[union-attr]
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )

        # Remove compression headers since httpx already decompressed the response
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)  # Length changed after decompression

        # Track stats for passthrough requests
        if endpoint_name and provider:
            latency_ms = (time.time() - start_time) * 1000
            await self.metrics.record_request(
                provider=provider,
                model=f"passthrough:{endpoint_name}",
                input_tokens=0,
                output_tokens=0,
                tokens_saved=0,
                latency_ms=latency_ms,
                cached=False,
                cost_usd=0,
                savings_usd=0,
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    # =========================================================================
    # OpenAI Batch API with Compression
    # =========================================================================

    async def handle_batch_create(self, request: Request) -> Response:
        """Handle POST /v1/batches - Create a batch with compression.

        Flow:
        1. Parse request to get input_file_id
        2. Download the JSONL file content from OpenAI
        3. Parse each line and compress the messages
        4. Create a new compressed JSONL file
        5. Upload compressed file to OpenAI
        6. Create batch with the new compressed file_id
        7. Return batch object with compression stats in metadata
        """
        start_time = time.time()
        request_id = await self._next_request_id()

        # Parse request
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON in request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )

        input_file_id = body.get("input_file_id")
        endpoint = body.get("endpoint")
        completion_window = body.get("completion_window", "24h")
        metadata = body.get("metadata", {})

        if not input_file_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "input_file_id is required",
                        "type": "invalid_request_error",
                        "code": "missing_parameter",
                    }
                },
            )

        if not endpoint:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "endpoint is required",
                        "type": "invalid_request_error",
                        "code": "missing_parameter",
                    }
                },
            )

        # Only compress chat completions endpoint
        if endpoint != "/v1/chat/completions":
            # Pass through for other endpoints
            return await self._batch_passthrough(request, body)

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)

        try:
            # Step 1: Download the input file from OpenAI
            logger.info(f"[{request_id}] Batch: Downloading input file {input_file_id}")
            file_content = await self._download_openai_file(input_file_id, headers)

            if file_content is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": {
                            "message": f"Failed to download file {input_file_id}",
                            "type": "invalid_request_error",
                            "code": "file_not_found",
                        }
                    },
                )

            # Step 2: Parse and compress each line
            logger.info(f"[{request_id}] Batch: Compressing JSONL content")
            compressed_lines, stats = await self._compress_batch_jsonl(file_content, request_id)

            if stats["total_requests"] == 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "No valid requests found in input file",
                            "type": "invalid_request_error",
                            "code": "empty_file",
                        }
                    },
                )

            # Step 3: Create compressed JSONL content
            compressed_content = "\n".join(compressed_lines)

            # Step 4: Upload compressed file to OpenAI
            logger.info(f"[{request_id}] Batch: Uploading compressed file")
            new_file_id = await self._upload_openai_file(
                compressed_content, f"compressed_{input_file_id}.jsonl", headers
            )

            if new_file_id is None:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": "Failed to upload compressed file",
                            "type": "server_error",
                            "code": "upload_failed",
                        }
                    },
                )

            # Step 5: Create batch with compressed file
            logger.info(f"[{request_id}] Batch: Creating batch with compressed file {new_file_id}")

            # Add compression stats to metadata
            compression_metadata = {
                **metadata,
                "headroom_compressed": "true",
                "headroom_original_file_id": input_file_id,
                "headroom_total_requests": str(stats["total_requests"]),
                "headroom_tokens_saved": str(stats["total_tokens_saved"]),
                "headroom_original_tokens": str(stats["total_original_tokens"]),
                "headroom_compressed_tokens": str(stats["total_compressed_tokens"]),
                "headroom_savings_percent": f"{stats['savings_percent']:.1f}",
            }

            batch_body = {
                "input_file_id": new_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
                "metadata": compression_metadata,
            }

            url = f"{self.OPENAI_API_URL}/v1/batches"
            response = await self.http_client.post(url, json=batch_body, headers=headers)  # type: ignore[union-attr]

            total_latency = (time.time() - start_time) * 1000

            # Log compression stats
            logger.info(
                f"[{request_id}] Batch created: {stats['total_requests']} requests, "
                f"{stats['total_original_tokens']:,} -> {stats['total_compressed_tokens']:,} tokens "
                f"(saved {stats['total_tokens_saved']:,} tokens, {stats['savings_percent']:.1f}%) "
                f"in {total_latency:.0f}ms"
            )

            # Record metrics
            await self.metrics.record_request(
                provider="openai",
                model="batch",
                input_tokens=stats["total_compressed_tokens"],
                output_tokens=0,
                tokens_saved=stats["total_tokens_saved"],
                latency_ms=total_latency,
            )

            # Return response with compression info in headers
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            response_headers["x-headroom-tokens-saved"] = str(stats["total_tokens_saved"])
            response_headers["x-headroom-savings-percent"] = f"{stats['savings_percent']:.1f}"

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        except Exception as e:
            logger.error(f"[{request_id}] Batch creation failed: {type(e).__name__}: {e}")
            await self.metrics.record_failed()
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "An error occurred while processing the batch request",
                        "type": "server_error",
                        "code": "batch_processing_error",
                    }
                },
            )

    async def _download_openai_file(self, file_id: str, headers: dict) -> str | None:
        """Download file content from OpenAI."""
        url = f"{self.OPENAI_API_URL}/v1/files/{file_id}/content"
        try:
            response = await self.http_client.get(url, headers=headers)  # type: ignore[union-attr]
            if response.status_code == 200:
                return str(response.text)
            logger.error(f"Failed to download file {file_id}: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return None

    async def _upload_openai_file(self, content: str, filename: str, headers: dict) -> str | None:
        """Upload a file to OpenAI for batch processing."""
        url = f"{self.OPENAI_API_URL}/v1/files"

        # Prepare multipart form data
        # We need to use httpx's files parameter for multipart upload
        files = {
            "file": (filename, content.encode("utf-8"), "application/jsonl"),
        }
        data = {
            "purpose": "batch",
        }

        # Remove content-type from headers (httpx will set it for multipart)
        upload_headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}

        try:
            response = await self.http_client.post(  # type: ignore[union-attr]
                url, files=files, data=data, headers=upload_headers
            )
            if response.status_code == 200:
                result = response.json()
                file_id: str | None = result.get("id")
                return file_id
            logger.error(f"Failed to upload file: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return None

    async def _compress_batch_jsonl(self, content: str, request_id: str) -> tuple[list[str], dict]:
        """Compress messages in each line of a batch JSONL file.

        Returns:
            Tuple of (compressed_lines, stats_dict)
        """
        lines = content.strip().split("\n")
        compressed_lines = []
        total_original_tokens = 0
        total_compressed_tokens = 0
        total_requests = 0
        errors = 0

        tokenizer = get_tokenizer("gpt-4")  # Use gpt-4 tokenizer for batch

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            try:
                request_obj = json.loads(line)
                body = request_obj.get("body", {})
                messages = body.get("messages", [])
                model = body.get("model", "gpt-4")

                if not messages:
                    # No messages to compress, pass through
                    compressed_lines.append(line)
                    total_requests += 1
                    continue

                # Count original tokens
                original_tokens = sum(
                    tokenizer.count_text(str(m.get("content", ""))) for m in messages
                )
                total_original_tokens += original_tokens

                # Compress messages using the OpenAI pipeline
                if self.config.optimize:
                    try:
                        context_limit = self.openai_provider.get_context_limit(model)
                        result = self.openai_pipeline.apply(
                            messages=messages,
                            model=model,
                            model_limit=context_limit,
                        )
                        compressed_messages = result.messages
                    except Exception as e:
                        logger.warning(f"[{request_id}] Compression failed for line {i}: {e}")
                        compressed_messages = messages
                else:
                    compressed_messages = messages

                # Count compressed tokens
                compressed_tokens = sum(
                    tokenizer.count_text(str(m.get("content", ""))) for m in compressed_messages
                )
                total_compressed_tokens += compressed_tokens
                tokens_saved = original_tokens - compressed_tokens

                # CCR Tool Injection: Inject retrieval tool if compression occurred
                tools = body.get("tools")
                if self.config.ccr_inject_tool and tokens_saved > 0:
                    injector = CCRToolInjector(
                        provider="openai",
                        inject_tool=True,
                        inject_system_instructions=self.config.ccr_inject_system_instructions,
                    )
                    compressed_messages, tools, was_injected = injector.process_request(
                        compressed_messages, tools
                    )
                    if was_injected:
                        logger.debug(
                            f"[{request_id}] CCR: Injected retrieval tool for batch line {i}"
                        )

                # Update body with compressed messages
                body["messages"] = compressed_messages
                if tools is not None:
                    body["tools"] = tools
                request_obj["body"] = body

                compressed_lines.append(json.dumps(request_obj))
                total_requests += 1

            except json.JSONDecodeError as e:
                logger.warning(f"[{request_id}] Invalid JSON on line {i}: {e}")
                errors += 1
                # Keep original line on error
                compressed_lines.append(line)
                total_requests += 1

        total_tokens_saved = total_original_tokens - total_compressed_tokens
        savings_percent = (
            (total_tokens_saved / total_original_tokens * 100) if total_original_tokens > 0 else 0
        )

        stats = {
            "total_requests": total_requests,
            "total_original_tokens": total_original_tokens,
            "total_compressed_tokens": total_compressed_tokens,
            "total_tokens_saved": total_tokens_saved,
            "savings_percent": savings_percent,
            "errors": errors,
        }

        return compressed_lines, stats

    async def _batch_passthrough(self, request: Request, body: dict) -> Response:
        """Pass through batch request to OpenAI without compression."""
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)

        url = f"{self.OPENAI_API_URL}/v1/batches"
        response = await self.http_client.post(url, json=body, headers=headers)  # type: ignore[union-attr]

        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_batch_list(self, request: Request) -> Response:
        """Handle GET /v1/batches - List batches (passthrough)."""
        return await self.handle_passthrough(request, self.OPENAI_API_URL)

    async def handle_batch_get(self, request: Request, batch_id: str) -> Response:
        """Handle GET /v1/batches/{batch_id} - Get batch (passthrough)."""
        return await self.handle_passthrough(request, self.OPENAI_API_URL)

    async def handle_batch_cancel(self, request: Request, batch_id: str) -> Response:
        """Handle POST /v1/batches/{batch_id}/cancel - Cancel batch (passthrough)."""
        return await self.handle_passthrough(request, self.OPENAI_API_URL)

    def _has_non_text_parts(self, content: dict) -> bool:
        """Check if a Gemini content entry has non-text parts.

        Non-text parts include:
        - inlineData: Base64-encoded images/media
        - fileData: File references (URI + MIME type)
        - functionCall: Function calls from model
        - functionResponse: Responses to function calls

        Args:
            content: A single Gemini content entry with 'parts' list.

        Returns:
            True if any part contains non-text data.
        """
        parts = content.get("parts", [])
        for part in parts:
            if any(
                key in part
                for key in ("inlineData", "fileData", "functionCall", "functionResponse")
            ):
                return True
        return False

    def _gemini_contents_to_messages(
        self, contents: list[dict], system_instruction: dict | None = None
    ) -> tuple[list[dict], set[int]]:
        """Convert Gemini contents[] format to OpenAI messages[] format for optimization.

        Gemini format:
            contents: [{"role": "user", "parts": [{"text": "..."}]}]
            systemInstruction: {"parts": [{"text": "..."}]}

        OpenAI format:
            messages: [{"role": "user", "content": "..."}]

        Returns:
            Tuple of (messages, preserved_indices) where preserved_indices contains
            the indices of content entries that have non-text parts (images, function
            calls, etc.) and should not be compressed.
        """
        messages = []
        preserved_indices: set[int] = set()

        # Add system instruction as system message
        if system_instruction:
            parts = system_instruction.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

        # Convert contents to messages
        for idx, content in enumerate(contents):
            # Track content entries with non-text parts
            if self._has_non_text_parts(content):
                preserved_indices.add(idx)

            role = content.get("role", "user")
            # Map Gemini roles to OpenAI roles
            if role == "model":
                role = "assistant"

            parts = content.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]

            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})

        return messages, preserved_indices

    def _messages_to_gemini_contents(self, messages: list[dict]) -> tuple[list[dict], dict | None]:
        """Convert OpenAI messages[] format back to Gemini contents[] format.

        Returns:
            (contents, system_instruction) tuple
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Extract as systemInstruction
                system_instruction = {"parts": [{"text": content}]}
            else:
                # Map OpenAI roles to Gemini roles
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

        return contents, system_instruction

    async def handle_openai_responses(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle OpenAI /v1/responses endpoint (new Responses API).

        The Responses API differs from /v1/chat/completions:
        - Input: `input` (string or array) instead of `messages`
        - System: `instructions` instead of system message
        - Output: `output[]` array instead of `choices[].message`
        - State: `previous_response_id` for multi-turn
        - Built-in tools: web_search, file_search, code_interpreter
        """
        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "type": "invalid_request_error",
                        "code": "request_too_large",
                    }
                },
            )

        # Parse request
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON in request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )

        model = body.get("model", "unknown")
        stream = body.get("stream", False)

        # Convert Responses API input to messages format for optimization
        # The Responses API accepts either a string or array of messages
        input_data = body.get("input", "")
        instructions = body.get("instructions")

        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})

        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            # Input is already an array of message objects
            messages.extend(input_data)

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)

        # Rate limiting
        if self.rate_limiter:
            rate_key = headers.get("authorization", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited()
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Token counting on converted messages
        tokenizer = get_tokenizer(model)
        original_tokens = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)

        # Note: We pass through to OpenAI without optimization for now
        # The Responses API has different semantics that may not work well with compression
        tokens_saved = 0
        transforms_applied: list[str] = []
        optimization_latency = (time.time() - start_time) * 1000

        url = f"{self.OPENAI_API_URL}/v1/responses"

        try:
            if stream:
                # Streaming for Responses API uses semantic events
                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "openai",
                    model,
                    request_id,
                    original_tokens,
                    original_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)
                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = original_tokens  # fallback
                output_tokens = 0
                cache_read_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    total_input_tokens = usage.get("input_tokens", original_tokens)
                    output_tokens = usage.get("output_tokens", 0)
                    # OpenAI returns cached_tokens in prompt_tokens_details (or input_tokens_details)
                    prompt_details = usage.get(
                        "prompt_tokens_details", usage.get("input_tokens_details", {})
                    )
                    cache_read_tokens = prompt_details.get("cached_tokens", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from OpenAI passthrough response: {e}"
                    )

                # For OpenAI, input_tokens is TOTAL (includes cached)
                # Normalize to non-cached input for consistent cost calculation
                non_cached_input = total_input_tokens - cache_read_tokens

                # Cost tracking using actual API tokens
                cost_usd = savings_usd = None
                if self.cost_tracker:
                    cost_usd = self.cost_tracker.estimate_cost(
                        model,
                        non_cached_input,  # Pass non-cached portion
                        output_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )
                    if cost_usd:
                        self.cost_tracker.record_cost(cost_usd)

                # Metrics with actual API tokens (total, for accurate tracking)
                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    cost_usd=cost_usd or 0,
                    savings_usd=savings_usd or 0,
                )

                logger.info(f"[{request_id}] /v1/responses {model}: {total_input_tokens:,} tokens")

                # Remove compression headers
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
        except Exception as e:
            await self.metrics.record_failed()
            logger.error(f"[{request_id}] OpenAI responses request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "type": "server_error",
                        "code": "proxy_error",
                    }
                },
            )

    async def handle_gemini_generate_content(
        self,
        request: Request,
        model: str,
    ) -> Response | StreamingResponse:
        """Handle Gemini native /v1beta/models/{model}:generateContent endpoint.

        Gemini's native API differs from OpenAI:
        - Input: `contents[]` with `parts[]` instead of `messages`
        - System: `systemInstruction` instead of system message
        - Auth: `x-goog-api-key` header instead of `Authorization: Bearer`
        - Output: `candidates[].content.parts[].text`
        """
        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "code": 413,
                    }
                },
            )

        # Parse request
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON in request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        contents = body.get("contents", [])

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)

        # Rate limiting (use Gemini API key)
        if self.rate_limiter:
            rate_key = headers.get("x-goog-api-key", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited()
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Convert Gemini format to messages for optimization
        system_instruction = body.get("systemInstruction")
        messages, preserved_indices = self._gemini_contents_to_messages(
            contents, system_instruction
        )

        # Store original content entries that have non-text parts before compression
        preserved_contents = {idx: contents[idx] for idx in preserved_indices}

        # Early exit if ALL content has non-text parts (nothing to compress)
        if len(preserved_indices) == len(contents):
            # All content has non-text parts, skip compression entirely
            # Just forward the request as-is
            url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:generateContent"
            query_params = dict(request.query_params)
            is_streaming = query_params.get("alt") == "sse"
            if "key" in query_params:
                url += f"?key={query_params['key']}"

            if is_streaming:
                stream_url = (
                    f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?alt=sse"
                )
                if "key" in query_params:
                    stream_url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?key={query_params['key']}&alt=sse"
                return await self._stream_response(
                    stream_url,
                    headers,
                    body,
                    "gemini",
                    model,
                    request_id,
                    0,
                    0,
                    0,
                    [],
                    tags,
                    0,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)

        # Optimization
        transforms_applied: list[str] = []
        optimized_messages = messages
        optimized_tokens = original_tokens

        if self.config.optimize and messages:
            try:
                # Use OpenAI pipeline (similar message format)
                context_limit = self.openai_provider.get_context_limit(model)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    optimized_tokens = sum(
                        tokenizer.count_text(str(m.get("content", ""))) for m in optimized_messages
                    )
            except Exception as e:
                logger.warning(f"[{request_id}] Gemini optimization failed: {e}")

        tokens_saved = original_tokens - optimized_tokens
        optimization_latency = (time.time() - start_time) * 1000

        # Convert back to Gemini format if optimized
        if optimized_messages != messages:
            optimized_contents, optimized_system = self._messages_to_gemini_contents(
                optimized_messages
            )

            # Restore preserved content entries that had non-text parts
            for orig_idx, original_content in preserved_contents.items():
                if orig_idx < len(optimized_contents):
                    optimized_contents[orig_idx] = original_content

            body["contents"] = optimized_contents
            if optimized_system:
                body["systemInstruction"] = optimized_system
            elif "systemInstruction" in body:
                del body["systemInstruction"]

        # Build URL - model is extracted from path
        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:generateContent"

        # Check if streaming requested via query param
        query_params = dict(request.query_params)
        is_streaming = query_params.get("alt") == "sse"

        # Preserve API key in query params if present
        if "key" in query_params:
            url += f"?key={query_params['key']}"

        try:
            if is_streaming:
                # For streaming, use streamGenerateContent endpoint
                stream_url = (
                    f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?alt=sse"
                )
                if "key" in query_params:
                    stream_url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?key={query_params['key']}&alt=sse"

                return await self._stream_response(
                    stream_url,
                    headers,
                    body,
                    "gemini",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)
                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = optimized_tokens  # fallback
                output_tokens = 0
                cache_read_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usageMetadata", {})
                    total_input_tokens = usage.get("promptTokenCount", optimized_tokens)
                    output_tokens = usage.get("candidatesTokenCount", 0)
                    # Gemini returns cachedContentTokenCount for context-cached tokens
                    # These are charged at 10-25% of the input price depending on model
                    cache_read_tokens = usage.get("cachedContentTokenCount", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from Gemini response: {e}"
                    )

                # For Gemini, promptTokenCount is TOTAL (includes cached)
                # Normalize to non-cached input for consistent cost calculation
                non_cached_input = total_input_tokens - cache_read_tokens

                # Cost tracking using actual API tokens
                cost_usd = savings_usd = None
                if self.cost_tracker:
                    cost_usd = self.cost_tracker.estimate_cost(
                        model,
                        non_cached_input,  # Pass non-cached portion
                        output_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )
                    original_cost = self.cost_tracker.estimate_cost(
                        model,
                        original_tokens,
                        output_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )
                    if cost_usd and original_cost:
                        savings_usd = original_cost - cost_usd
                        self.cost_tracker.record_cost(cost_usd)
                        self.cost_tracker.record_savings(savings_usd)

                # Metrics with actual API tokens (total, for accurate tracking)
                await self.metrics.record_request(
                    provider="gemini",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    cost_usd=cost_usd or 0,
                    savings_usd=savings_usd or 0,
                )

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] Gemini {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens)"
                    )
                else:
                    logger.info(f"[{request_id}] Gemini {model}: {original_tokens:,} tokens")

                # Remove compression headers
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
        except Exception as e:
            await self.metrics.record_failed()
            logger.error(f"[{request_id}] Gemini request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "code": 502,
                    }
                },
            )

    async def handle_gemini_stream_generate_content(
        self,
        request: Request,
        model: str,
    ) -> StreamingResponse | JSONResponse:
        """Handle Gemini streaming endpoint /v1beta/models/{model}:streamGenerateContent."""
        start_time = time.time()
        request_id = await self._next_request_id()

        # Parse request
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON in request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        contents = body.get("contents", [])

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = 0
        for content in contents:
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    original_tokens += tokenizer.count_text(part["text"])

        optimization_latency = (time.time() - start_time) * 1000

        # Build URL with SSE param
        query_params = dict(request.query_params)
        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?alt=sse"
        if "key" in query_params:
            url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?key={query_params['key']}&alt=sse"

        return await self._stream_response(
            url,
            headers,
            body,
            "gemini",
            model,
            request_id,
            original_tokens,
            original_tokens,
            0,  # tokens_saved
            [],  # transforms_applied
            tags,
            optimization_latency,
        )

    async def handle_gemini_count_tokens(
        self,
        request: Request,
        model: str,
    ) -> Response:
        """Handle Gemini /v1beta/models/{model}:countTokens endpoint with compression.

        This endpoint counts tokens AFTER applying compression, so users can see
        how many tokens they'll actually use after optimization.

        The request format is the same as generateContent:
            {"contents": [...], "systemInstruction": {...}}
        """
        start_time = time.time()
        request_id = await self._next_request_id()

        # Parse request
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON in request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        contents = body.get("contents", [])

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)

        # Convert Gemini format to messages for optimization
        system_instruction = body.get("systemInstruction")
        messages, preserved_indices = self._gemini_contents_to_messages(
            contents, system_instruction
        )

        # Store original content entries that have non-text parts before compression
        preserved_contents = {idx: contents[idx] for idx in preserved_indices}

        # Early exit if ALL content has non-text parts (nothing to compress)
        if len(preserved_indices) == len(contents):
            # All content has non-text parts, skip compression entirely
            # Just forward the countTokens request as-is
            url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:countTokens"
            query_params = dict(request.query_params)
            if "key" in query_params:
                url += f"?key={query_params['key']}"

            response = await self._retry_request("POST", url, headers, body)
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Token counting (original)
        tokenizer = get_tokenizer(model)
        original_tokens = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)

        # Apply compression using the same pipeline as generateContent
        transforms_applied: list[str] = []
        optimized_messages = messages

        if self.config.optimize and messages:
            try:
                context_limit = self.openai_provider.get_context_limit(model)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
            except Exception as e:
                logger.warning(f"[{request_id}] Gemini countTokens optimization failed: {e}")

        # Convert back to Gemini format for the API call
        if optimized_messages != messages:
            optimized_contents, optimized_system = self._messages_to_gemini_contents(
                optimized_messages
            )

            # Restore preserved content entries that had non-text parts
            for orig_idx, original_content in preserved_contents.items():
                if orig_idx < len(optimized_contents):
                    optimized_contents[orig_idx] = original_content

            body["contents"] = optimized_contents
            if optimized_system:
                body["systemInstruction"] = optimized_system
            elif "systemInstruction" in body:
                del body["systemInstruction"]

        # Build URL
        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:countTokens"

        # Preserve API key in query params if present
        query_params = dict(request.query_params)
        if "key" in query_params:
            url += f"?key={query_params['key']}"

        try:
            response = await self._retry_request("POST", url, headers, body)
            total_latency = (time.time() - start_time) * 1000

            # Parse response to get token count
            compressed_tokens = 0
            try:
                resp_json = response.json()
                compressed_tokens = resp_json.get("totalTokens", 0)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"[{request_id}] Failed to parse Gemini token count response: {e}")

            # Track stats
            tokens_saved = original_tokens - compressed_tokens if compressed_tokens > 0 else 0

            await self.metrics.record_request(
                provider="gemini",
                model=model,
                input_tokens=compressed_tokens,
                output_tokens=0,
                tokens_saved=tokens_saved,
                latency_ms=total_latency,
                cost_usd=0,
                savings_usd=0,
            )

            if tokens_saved > 0:
                logger.info(
                    f"[{request_id}] Gemini countTokens {model}: {original_tokens:,} → {compressed_tokens:,} "
                    f"(saved {tokens_saved:,} tokens, transforms: {transforms_applied})"
                )
            else:
                logger.info(
                    f"[{request_id}] Gemini countTokens {model}: {compressed_tokens:,} tokens"
                )

            # Remove compression headers
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )
        except Exception as e:
            await self.metrics.record_failed()
            logger.error(f"[{request_id}] Gemini countTokens failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "code": 502,
                    }
                },
            )


# =============================================================================
# FastAPI App
# =============================================================================


async def _log_toin_stats_periodically(interval_seconds: int = 300) -> None:
    """Background task that logs TOIN stats periodically.

    Args:
        interval_seconds: How often to log stats (default: 5 minutes).
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            toin = get_toin()
            stats = toin.get_stats()
            total_compressions = stats.get("total_compressions", 0)
            if total_compressions > 0:
                patterns = stats.get("patterns_tracked", 0)
                retrievals = stats.get("total_retrievals", 0)
                retrieval_rate = stats.get("global_retrieval_rate", 0.0)
                logger.info(
                    "TOIN: %d patterns, %d compressions, %d retrievals, %.1f%% retrieval rate",
                    patterns,
                    total_compressions,
                    retrievals,
                    retrieval_rate * 100,
                )
        except Exception as e:
            logger.debug("Failed to log TOIN stats: %s", e)


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required. Install: pip install fastapi uvicorn httpx")

    config = config or ProxyConfig()

    app = FastAPI(
        title="Headroom Proxy",
        description="Production-ready LLM optimization proxy",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    proxy = HeadroomProxy(config)

    @app.on_event("startup")
    async def startup():
        await proxy.startup()
        # Start background task for periodic TOIN stats logging
        asyncio.create_task(_log_toin_stats_periodically())

    @app.on_event("shutdown")
    async def shutdown():
        await proxy.shutdown()

    # Health & Metrics
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "config": {
                "optimize": config.optimize,
                "cache": config.cache_enabled,
                "rate_limit": config.rate_limit_enabled,
            },
        }

    @app.get("/stats")
    async def stats():
        """Get comprehensive proxy statistics.

        This is the main stats endpoint - it aggregates data from all subsystems:
        - Request metrics (total, cached, failed, by model/provider)
        - Token usage and savings
        - Cost tracking
        - Compression (CCR) statistics
        - Telemetry/TOIN (data flywheel) statistics
        - Cache and rate limiter stats
        """
        m = proxy.metrics

        # Calculate average latency
        avg_latency_ms = round(m.latency_sum_ms / m.latency_count, 2) if m.latency_count > 0 else 0

        # Get compression store stats
        store = get_compression_store()
        compression_stats = store.get_stats()

        # Get telemetry/TOIN stats
        telemetry = get_telemetry_collector()
        telemetry_stats = telemetry.get_stats()

        # Get feedback loop stats
        feedback = get_compression_feedback()
        feedback_stats = feedback.get_stats()

        # Calculate total tokens before compression
        total_tokens_before = m.tokens_input_total + m.tokens_saved_total

        return {
            "requests": {
                "total": m.requests_total,
                "cached": m.requests_cached,
                "rate_limited": m.requests_rate_limited,
                "failed": m.requests_failed,
                "by_provider": dict(m.requests_by_provider),
                "by_model": dict(m.requests_by_model),
            },
            "tokens": {
                "input": m.tokens_input_total,
                "output": m.tokens_output_total,
                "saved": m.tokens_saved_total,
                "total_before_compression": total_tokens_before,
                "savings_percent": round(
                    (m.tokens_saved_total / total_tokens_before * 100)
                    if total_tokens_before > 0
                    else 0,
                    2,
                ),
            },
            "latency": {
                "average_ms": avg_latency_ms,
                "total_requests": m.latency_count,
            },
            "cost": proxy.cost_tracker.stats() if proxy.cost_tracker else None,
            "compression": {
                "ccr_entries": compression_stats.get("entry_count", 0),
                "ccr_max_entries": compression_stats.get("max_entries", 0),
                "original_tokens_cached": compression_stats.get("total_original_tokens", 0),
                "compressed_tokens_cached": compression_stats.get("total_compressed_tokens", 0),
                "ccr_retrievals": compression_stats.get("total_retrievals", 0),
            },
            "telemetry": {
                "enabled": telemetry_stats.get("enabled", False),
                "total_compressions": telemetry_stats.get("total_compressions", 0),
                "total_retrievals": telemetry_stats.get("total_retrievals", 0),
                "global_retrieval_rate": round(telemetry_stats.get("global_retrieval_rate", 0), 4),
                "tool_signatures_tracked": telemetry_stats.get("tool_signatures_tracked", 0),
                "avg_compression_ratio": round(telemetry_stats.get("avg_compression_ratio", 0), 4),
                "avg_token_reduction": round(telemetry_stats.get("avg_token_reduction", 0), 4),
            },
            "feedback_loop": {
                "tools_tracked": feedback_stats.get("tools_tracked", 0),
                "total_compressions": feedback_stats.get("total_compressions", 0),
                "total_retrievals": feedback_stats.get("total_retrievals", 0),
                "global_retrieval_rate": round(feedback_stats.get("global_retrieval_rate", 0), 4),
                "tools_with_high_retrieval": sum(
                    1
                    for p in feedback_stats.get("tool_patterns", {}).values()
                    if p.get("retrieval_rate", 0) > 0.3
                ),
            },
            "toin": get_toin().get_stats(),
            "cache": await proxy.cache.stats() if proxy.cache else None,
            "rate_limiter": await proxy.rate_limiter.stats() if proxy.rate_limiter else None,
            "recent_requests": proxy.logger.get_recent(10) if proxy.logger else [],
        }

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            await proxy.metrics.export(),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/cache/clear")
    async def clear_cache():
        """Clear the response cache."""
        if proxy.cache:
            await proxy.cache.clear()
            return {"status": "cleared"}
        return {"status": "cache disabled"}

    # CCR (Compress-Cache-Retrieve) endpoints
    @app.post("/v1/retrieve")
    async def ccr_retrieve(request: Request):
        """Retrieve original content from CCR compression cache.

        This is the "Retrieve" part of CCR (Compress-Cache-Retrieve).
        When SmartCrusher compresses tool outputs, the original data is cached.
        LLMs can call this endpoint to get more data if needed.

        Request body:
            hash (str): Hash key from compression marker (required)
            query (str): Optional search query to filter results

        Response:
            Full retrieval: {"hash": "...", "original_content": "...", ...}
            Search: {"hash": "...", "query": "...", "results": [...], "count": N}
        """
        data = await request.json()
        hash_key = data.get("hash")
        query = data.get("query")

        if not hash_key:
            raise HTTPException(status_code=400, detail="hash required")

        store = get_compression_store()

        if query:
            # Search within cached content
            results = store.search(hash_key, query)
            return {
                "hash": hash_key,
                "query": query,
                "results": results,
                "count": len(results),
            }
        else:
            # Return full original content
            entry = store.retrieve(hash_key)
            if entry:
                return {
                    "hash": hash_key,
                    "original_content": entry.original_content,
                    "original_tokens": entry.original_tokens,
                    "original_item_count": entry.original_item_count,
                    "compressed_item_count": entry.compressed_item_count,
                    "tool_name": entry.tool_name,
                    "retrieval_count": entry.retrieval_count,
                }
            raise HTTPException(
                status_code=404, detail="Entry not found or expired (TTL: 5 minutes)"
            )

    @app.get("/v1/retrieve/stats")
    async def ccr_stats():
        """Get CCR compression store statistics."""
        store = get_compression_store()
        stats = store.get_stats()
        events = store.get_retrieval_events(limit=20)
        return {
            "store": stats,
            "recent_retrievals": [
                {
                    "hash": e.hash,
                    "query": e.query,
                    "items_retrieved": e.items_retrieved,
                    "total_items": e.total_items,
                    "tool_name": e.tool_name,
                    "retrieval_type": e.retrieval_type,
                }
                for e in events
            ],
        }

    @app.get("/v1/feedback")
    async def ccr_feedback():
        """Get CCR feedback loop statistics and learned patterns.

        This endpoint exposes the feedback loop's learned patterns for monitoring
        and debugging. It shows:
        - Per-tool retrieval rates (high = compress less aggressively)
        - Common search queries per tool
        - Queried fields (suggest what to preserve)

        Use this to understand how well compression is working and whether
        the feedback loop is adjusting appropriately.
        """
        feedback = get_compression_feedback()
        stats = feedback.get_stats()
        return {
            "feedback": stats,
            "hints_example": {
                tool_name: {
                    "hints": {
                        "max_items": hints.max_items
                        if (hints := feedback.get_compression_hints(tool_name))
                        else 15,
                        "suggested_items": hints.suggested_items if hints else None,
                        "skip_compression": hints.skip_compression if hints else False,
                        "preserve_fields": hints.preserve_fields if hints else [],
                        "reason": hints.reason if hints else "",
                    }
                }
                for tool_name in list(stats.get("tool_patterns", {}).keys())[:5]
            },
        }

    @app.get("/v1/feedback/{tool_name}")
    async def ccr_feedback_for_tool(tool_name: str):
        """Get compression hints for a specific tool.

        Returns feedback-based hints that would be used for compressing
        this tool's output.
        """
        feedback = get_compression_feedback()
        hints = feedback.get_compression_hints(tool_name)
        patterns = feedback.get_all_patterns().get(tool_name)

        return {
            "tool_name": tool_name,
            "hints": {
                "max_items": hints.max_items,
                "min_items": hints.min_items,
                "suggested_items": hints.suggested_items,
                "aggressiveness": hints.aggressiveness,
                "skip_compression": hints.skip_compression,
                "preserve_fields": hints.preserve_fields,
                "reason": hints.reason,
            },
            "pattern": {
                "total_compressions": patterns.total_compressions if patterns else 0,
                "total_retrievals": patterns.total_retrievals if patterns else 0,
                "retrieval_rate": patterns.retrieval_rate if patterns else 0.0,
                "full_retrieval_rate": patterns.full_retrieval_rate if patterns else 0.0,
                "search_rate": patterns.search_rate if patterns else 0.0,
                "common_queries": list(patterns.common_queries.keys())[:10] if patterns else [],
                "queried_fields": list(patterns.queried_fields.keys())[:10] if patterns else [],
            }
            if patterns
            else None,
        }

    # Telemetry endpoints (Data Flywheel)
    @app.get("/v1/telemetry")
    async def telemetry_stats():
        """Get telemetry statistics for the data flywheel.

        This endpoint exposes privacy-preserving telemetry data that powers
        the data flywheel - learning optimal compression strategies across
        tool types based on usage patterns.

        What's collected (anonymized):
        - Tool output structure patterns (field types, not values)
        - Compression decisions and ratios
        - Retrieval patterns (rate, type, not content)
        - Strategy effectiveness

        What's NOT collected:
        - Actual data values
        - User identifiers
        - Queries or search terms
        - File paths or tool names (hashed by default)
        """
        telemetry = get_telemetry_collector()
        return telemetry.get_stats()

    @app.get("/v1/telemetry/export")
    async def telemetry_export():
        """Export full telemetry data for aggregation.

        This endpoint exports all telemetry data in a format suitable for
        cross-user aggregation. The data is privacy-preserving - no actual
        values are included, only structural patterns and statistics.

        Use this for:
        - Building a central learning service
        - Sharing learned patterns across instances
        - Analysis and debugging
        """
        telemetry = get_telemetry_collector()
        return telemetry.export_stats()

    @app.post("/v1/telemetry/import")
    async def telemetry_import(request: Request):
        """Import telemetry data from another source.

        This allows merging telemetry from multiple sources for cross-user
        learning. The imported data is merged with existing statistics.

        Request body: Telemetry export data from /v1/telemetry/export
        """
        telemetry = get_telemetry_collector()
        data = await request.json()
        telemetry.import_stats(data)
        return {"status": "imported", "current_stats": telemetry.get_stats()}

    @app.get("/v1/telemetry/tools")
    async def telemetry_tools():
        """Get telemetry statistics for all tracked tool signatures.

        Returns statistics per tool signature (anonymized), including:
        - Compression ratios and strategy usage
        - Retrieval rates (high = compression too aggressive)
        - Learned recommendations
        """
        telemetry = get_telemetry_collector()
        all_stats = telemetry.get_all_tool_stats()
        return {
            "tool_count": len(all_stats),
            "tools": {sig_hash: stats.to_dict() for sig_hash, stats in all_stats.items()},
        }

    @app.get("/v1/telemetry/tools/{signature_hash}")
    async def telemetry_tool_detail(signature_hash: str):
        """Get detailed telemetry for a specific tool signature.

        Includes learned recommendations if enough data has been collected.
        """
        telemetry = get_telemetry_collector()
        stats = telemetry.get_tool_stats(signature_hash)
        recommendations = telemetry.get_recommendations(signature_hash)

        if stats is None:
            raise HTTPException(
                status_code=404, detail=f"No telemetry found for signature: {signature_hash}"
            )

        return {
            "signature_hash": signature_hash,
            "stats": stats.to_dict(),
            "recommendations": recommendations,
        }

    # TOIN (Tool Output Intelligence Network) endpoints
    @app.get("/v1/toin/stats")
    async def toin_stats():
        """Get overall TOIN statistics.

        Returns aggregated statistics from the Tool Output Intelligence Network,
        which learns optimal compression strategies across all tool types.

        Response includes:
        - enabled: Whether TOIN is enabled
        - patterns_tracked: Number of unique tool patterns being tracked
        - total_compressions: Total compression events recorded
        - total_retrievals: Total retrieval events recorded
        - global_retrieval_rate: Overall retrieval rate (high = compression too aggressive)
        - patterns_with_recommendations: Patterns with enough data for recommendations
        """
        toin = get_toin()
        return toin.get_stats()

    @app.get("/v1/toin/patterns")
    async def toin_patterns(limit: int = 20):
        """List TOIN patterns with most samples.

        Returns patterns sorted by sample_size descending. Use this to see
        which tool types have the most data and their learned behaviors.

        Query params:
            limit: Maximum number of patterns to return (default 20)

        Response includes for each pattern:
        - hash: Truncated tool signature hash (12 chars)
        - compressions: Total compression events
        - retrievals: Total retrieval events
        - retrieval_rate: Percentage of compressions that triggered retrieval
        - confidence: Confidence level in recommendations (0.0-1.0)
        - skip_recommended: Whether TOIN recommends skipping compression
        - optimal_max_items: Learned optimal max_items setting
        """
        toin = get_toin()
        exported = toin.export_patterns()
        patterns_data = exported.get("patterns", {})

        # Convert to list and sort by sample_size
        patterns_list = []
        for sig_hash, pattern_dict in patterns_data.items():
            sample_size = pattern_dict.get("sample_size", 0)
            total_compressions = pattern_dict.get("total_compressions", 0)
            total_retrievals = pattern_dict.get("total_retrievals", 0)
            retrieval_rate = (
                total_retrievals / total_compressions if total_compressions > 0 else 0.0
            )

            patterns_list.append(
                {
                    "hash": sig_hash[:12],
                    "compressions": total_compressions,
                    "retrievals": total_retrievals,
                    "retrieval_rate": f"{retrieval_rate:.1%}",
                    "confidence": round(pattern_dict.get("confidence", 0.0), 3),
                    "skip_recommended": pattern_dict.get("skip_compression_recommended", False),
                    "optimal_max_items": pattern_dict.get("optimal_max_items", 20),
                    "sample_size": sample_size,
                }
            )

        # Sort by sample_size descending
        patterns_list.sort(key=lambda p: p["sample_size"], reverse=True)

        # Remove sample_size from output (used only for sorting)
        for p in patterns_list:
            del p["sample_size"]

        return patterns_list[:limit]

    @app.get("/v1/toin/pattern/{hash_prefix}")
    async def toin_pattern_detail(hash_prefix: str):
        """Get detailed TOIN pattern info by hash prefix.

        Searches for a pattern where the tool signature hash starts with
        the provided prefix. Returns full pattern details if found.

        Path params:
            hash_prefix: Beginning of the tool signature hash (min 4 chars recommended)

        Response: Full pattern.to_dict() with all learned statistics and recommendations.
        """
        toin = get_toin()
        exported = toin.export_patterns()
        patterns_data = exported.get("patterns", {})

        # Search for pattern with matching hash prefix
        for sig_hash, pattern_dict in patterns_data.items():
            if sig_hash.startswith(hash_prefix):
                return pattern_dict

        raise HTTPException(
            status_code=404, detail=f"No TOIN pattern found with hash starting with: {hash_prefix}"
        )

    @app.get("/v1/retrieve/{hash_key}")
    async def ccr_retrieve_get(hash_key: str, query: str | None = None):
        """GET version of CCR retrieve for easier testing."""
        store = get_compression_store()

        if query:
            results = store.search(hash_key, query)
            return {
                "hash": hash_key,
                "query": query,
                "results": results,
                "count": len(results),
            }
        else:
            entry = store.retrieve(hash_key)
            if entry:
                return {
                    "hash": hash_key,
                    "original_content": entry.original_content,
                    "original_tokens": entry.original_tokens,
                    "original_item_count": entry.original_item_count,
                    "compressed_item_count": entry.compressed_item_count,
                    "tool_name": entry.tool_name,
                    "retrieval_count": entry.retrieval_count,
                }
            raise HTTPException(status_code=404, detail="Entry not found or expired")

    # CCR Tool Call Handler - for agent frameworks to call when LLM uses headroom_retrieve
    @app.post("/v1/retrieve/tool_call")
    async def ccr_handle_tool_call(request: Request):
        """Handle a CCR tool call from an LLM response.

        This endpoint accepts tool call formats from various providers and returns
        a properly formatted tool result. Agent frameworks can use this to handle
        CCR tool calls without implementing the retrieval logic themselves.

        Request body (Anthropic format):
            {
                "tool_call": {
                    "id": "toolu_123",
                    "name": "headroom_retrieve",
                    "input": {"hash": "abc123", "query": "optional search"}
                },
                "provider": "anthropic"
            }

        Request body (OpenAI format):
            {
                "tool_call": {
                    "id": "call_123",
                    "function": {
                        "name": "headroom_retrieve",
                        "arguments": "{\"hash\": \"abc123\"}"
                    }
                },
                "provider": "openai"
            }

        Response:
            {
                "tool_result": {...},  # Formatted for the provider
                "success": true,
                "data": {...}  # Raw retrieval data
            }
        """
        data = await request.json()
        tool_call = data.get("tool_call", {})
        provider = data.get("provider", "anthropic")

        # Parse the tool call
        hash_key, query = parse_tool_call(tool_call, provider)

        if hash_key is None:
            raise HTTPException(
                status_code=400, detail=f"Invalid tool call or not a {CCR_TOOL_NAME} call"
            )

        # Perform retrieval
        store = get_compression_store()

        if query:
            results = store.search(hash_key, query)
            retrieval_data = {
                "hash": hash_key,
                "query": query,
                "results": results,
                "count": len(results),
            }
        else:
            entry = store.retrieve(hash_key)
            if entry:
                retrieval_data = {
                    "hash": hash_key,
                    "original_content": entry.original_content,
                    "original_item_count": entry.original_item_count,
                    "compressed_item_count": entry.compressed_item_count,
                }
            else:
                retrieval_data = {
                    "error": "Entry not found or expired (TTL: 5 minutes)",
                    "hash": hash_key,
                }

        # Format tool result for provider
        tool_call_id = tool_call.get("id", "")
        result_content = json.dumps(retrieval_data, indent=2)

        if provider == "anthropic":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result_content,
            }
        elif provider == "openai":
            tool_result = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_content,
            }
        else:
            tool_result = {
                "tool_call_id": tool_call_id,
                "content": result_content,
            }

        return {
            "tool_result": tool_result,
            "success": "error" not in retrieval_data,
            "data": retrieval_data,
        }

    # Anthropic endpoints
    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        return await proxy.handle_anthropic_messages(request)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        return await proxy.handle_passthrough(
            request, proxy.ANTHROPIC_API_URL, "count_tokens", "anthropic"
        )

    # Anthropic Message Batches API endpoints
    @app.post("/v1/messages/batches")
    async def anthropic_batch_create(request: Request):
        """Create a message batch with compression applied to all requests."""
        return await proxy.handle_anthropic_batch_create(request)

    @app.get("/v1/messages/batches")
    async def anthropic_batch_list(request: Request):
        """List message batches (passthrough)."""
        return await proxy.handle_anthropic_batch_passthrough(request)

    @app.get("/v1/messages/batches/{batch_id}")
    async def anthropic_batch_get(request: Request, batch_id: str):
        """Get a specific message batch (passthrough)."""
        return await proxy.handle_anthropic_batch_passthrough(request, batch_id)

    @app.get("/v1/messages/batches/{batch_id}/results")
    async def anthropic_batch_results(request: Request, batch_id: str):
        """Get results for a message batch with CCR post-processing."""
        return await proxy.handle_anthropic_batch_results(request, batch_id)

    @app.post("/v1/messages/batches/{batch_id}/cancel")
    async def anthropic_batch_cancel(request: Request, batch_id: str):
        """Cancel a message batch (passthrough)."""
        return await proxy.handle_anthropic_batch_passthrough(request, batch_id)

    # OpenAI endpoints
    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        return await proxy.handle_openai_chat(request)

    @app.post("/v1/responses")
    async def openai_responses(request: Request):
        """OpenAI Responses API (new API introduced March 2025)."""
        return await proxy.handle_openai_responses(request)

    # OpenAI Batch API endpoints (with compression!)
    @app.post("/v1/batches")
    async def create_batch(request: Request):
        """Create a batch with automatic compression of messages."""
        return await proxy.handle_batch_create(request)

    @app.get("/v1/batches")
    async def list_batches(request: Request):
        """List batches (passthrough to OpenAI)."""
        return await proxy.handle_batch_list(request)

    @app.get("/v1/batches/{batch_id}")
    async def get_batch(request: Request, batch_id: str):
        """Get batch details (passthrough to OpenAI)."""
        return await proxy.handle_batch_get(request, batch_id)

    @app.post("/v1/batches/{batch_id}/cancel")
    async def cancel_batch(request: Request, batch_id: str):
        """Cancel a batch (passthrough to OpenAI)."""
        return await proxy.handle_batch_cancel(request, batch_id)

    # Gemini native endpoints
    @app.post("/v1beta/models/{model}:generateContent")
    async def gemini_generate_content(request: Request, model: str):
        """Gemini native generateContent API."""
        return await proxy.handle_gemini_generate_content(request, model)

    @app.post("/v1beta/models/{model}:streamGenerateContent")
    async def gemini_stream_generate_content(request: Request, model: str):
        """Gemini native streaming generateContent API."""
        return await proxy.handle_gemini_stream_generate_content(request, model)

    @app.post("/v1beta/models/{model}:countTokens")
    async def gemini_count_tokens(request: Request, model: str):
        """Gemini countTokens API with compression applied."""
        return await proxy.handle_gemini_count_tokens(request, model)

    # =========================================================================
    # Passthrough Endpoints (no compression needed)
    # =========================================================================

    # --- OpenAI Passthrough Endpoints ---

    @app.get("/v1/models")
    async def list_models(request: Request):
        """List models - route based on auth header.

        - x-api-key header present -> Anthropic
        - Authorization: Bearer header -> OpenAI
        """
        if request.headers.get("x-api-key"):
            return await proxy.handle_passthrough(
                request, proxy.ANTHROPIC_API_URL, "models", "anthropic"
            )
        return await proxy.handle_passthrough(request, proxy.OPENAI_API_URL, "models", "openai")

    @app.get("/v1/models/{model_id}")
    async def get_model(request: Request, model_id: str):
        """Get model details - route based on auth header.

        - x-api-key header present -> Anthropic
        - Authorization: Bearer header -> OpenAI
        """
        if request.headers.get("x-api-key"):
            return await proxy.handle_passthrough(
                request, proxy.ANTHROPIC_API_URL, "models", "anthropic"
            )
        return await proxy.handle_passthrough(request, proxy.OPENAI_API_URL, "models", "openai")

    @app.post("/v1/embeddings")
    async def openai_embeddings(request: Request):
        """OpenAI embeddings API - passthrough."""
        return await proxy.handle_passthrough(request, proxy.OPENAI_API_URL, "embeddings", "openai")

    @app.post("/v1/moderations")
    async def openai_moderations(request: Request):
        """OpenAI moderations API - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.OPENAI_API_URL, "moderations", "openai"
        )

    @app.post("/v1/images/generations")
    async def openai_images_generations(request: Request):
        """OpenAI image generation API - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.OPENAI_API_URL, "images/generations", "openai"
        )

    @app.post("/v1/audio/transcriptions")
    async def openai_audio_transcriptions(request: Request):
        """OpenAI audio transcription API (multipart/form-data) - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.OPENAI_API_URL, "audio/transcriptions", "openai"
        )

    @app.post("/v1/audio/speech")
    async def openai_audio_speech(request: Request):
        """OpenAI text-to-speech API - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.OPENAI_API_URL, "audio/speech", "openai"
        )

    # --- Gemini Passthrough Endpoints ---

    @app.get("/v1beta/models")
    async def gemini_list_models(request: Request):
        """Gemini list models API - passthrough."""
        return await proxy.handle_passthrough(request, proxy.GEMINI_API_URL, "models", "gemini")

    @app.get("/v1beta/models/{model_name}")
    async def gemini_get_model(request: Request, model_name: str):
        """Gemini get model API - passthrough.

        Note: This handles GET /v1beta/models/{model_name} but NOT :countTokens
        which is handled by a separate POST route above.
        """
        return await proxy.handle_passthrough(request, proxy.GEMINI_API_URL, "models", "gemini")

    @app.post("/v1beta/models/{model}:embedContent")
    async def gemini_embed_content(request: Request, model: str):
        """Gemini embedding API - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.GEMINI_API_URL, "embedContent", "gemini"
        )

    @app.post("/v1beta/models/{model}:batchEmbedContents")
    async def gemini_batch_embed_contents(request: Request, model: str):
        """Gemini batch embeddings API - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.GEMINI_API_URL, "batchEmbedContents", "gemini"
        )

    # Google/Gemini Batch API endpoints (with compression!)
    @app.post("/v1beta/models/{model}:batchGenerateContent")
    async def gemini_batch_create(request: Request, model: str):
        """Create a Gemini batch with compression applied to all requests."""
        return await proxy.handle_google_batch_create(request, model)

    @app.get("/v1beta/batches/{batch_name}")
    async def gemini_batch_get(request: Request, batch_name: str):
        """Get a specific Gemini batch with CCR post-processing."""
        return await proxy.handle_google_batch_results(request, batch_name)

    @app.post("/v1beta/batches/{batch_name}:cancel")
    async def gemini_batch_cancel(request: Request, batch_name: str):
        """Cancel a Gemini batch (passthrough)."""
        return await proxy.handle_google_batch_passthrough(request, batch_name)

    @app.delete("/v1beta/batches/{batch_name}")
    async def gemini_batch_delete(request: Request, batch_name: str):
        """Delete a Gemini batch (passthrough)."""
        return await proxy.handle_google_batch_passthrough(request, batch_name)

    @app.post("/v1beta/cachedContents")
    async def gemini_create_cached_content(request: Request):
        """Gemini create cached content API - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.GEMINI_API_URL, "cachedContents", "gemini"
        )

    @app.get("/v1beta/cachedContents")
    async def gemini_list_cached_contents(request: Request):
        """Gemini list cached contents API - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.GEMINI_API_URL, "cachedContents", "gemini"
        )

    @app.get("/v1beta/cachedContents/{cache_id}")
    async def gemini_get_cached_content(request: Request, cache_id: str):
        """Gemini get cached content API - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.GEMINI_API_URL, "cachedContents", "gemini"
        )

    @app.delete("/v1beta/cachedContents/{cache_id}")
    async def gemini_delete_cached_content(request: Request, cache_id: str):
        """Gemini delete cached content API - passthrough."""
        return await proxy.handle_passthrough(
            request, proxy.GEMINI_API_URL, "cachedContents", "gemini"
        )

    # =========================================================================
    # Catch-all Passthrough
    # =========================================================================

    # Passthrough - route to correct backend based on headers
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def passthrough(request: Request, path: str):
        # Anthropic SDK always sends anthropic-version header and uses x-api-key for auth
        # OpenAI SDK uses Authorization: Bearer for auth
        if request.headers.get("anthropic-version") or request.headers.get("x-api-key"):
            base_url = proxy.ANTHROPIC_API_URL
        else:
            base_url = proxy.OPENAI_API_URL
        return await proxy.handle_passthrough(request, base_url)

    return app


def _get_llmlingua_banner_status(config: ProxyConfig) -> str:
    """Get LLMLingua status line for banner."""
    if config.llmlingua_enabled:
        if _LLMLINGUA_AVAILABLE:
            return (
                f"ENABLED  (device={config.llmlingua_device}, rate={config.llmlingua_target_rate})"
            )
        else:
            return "NOT INSTALLED (pip install headroom-ai[llmlingua])"
    else:
        if _LLMLINGUA_AVAILABLE:
            return "DISABLED (remove --no-llmlingua to enable)"
        return "DISABLED"


def _get_code_aware_banner_status(config: ProxyConfig) -> str:
    """Get code-aware compression status line for banner."""
    if config.code_aware_enabled:
        if is_tree_sitter_available():
            return "ENABLED  (AST-based)"
        else:
            return "NOT INSTALLED (pip install headroom-ai[code])"
    else:
        if is_tree_sitter_available():
            return "DISABLED (remove --no-code-aware to enable)"
        return "DISABLED"


def run_server(
    config: ProxyConfig | None = None,
    workers: int = 1,
    limit_concurrency: int = 1000,
):
    """Run the proxy server.

    Args:
        config: Proxy configuration
        workers: Number of worker processes (use N for multi-core scaling)
        limit_concurrency: Max concurrent connections before 503 response
    """
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI required. Install: pip install fastapi uvicorn httpx")
        sys.exit(1)

    config = config or ProxyConfig()
    app = create_app(config)

    llmlingua_status = _get_llmlingua_banner_status(config)
    code_aware_status = _get_code_aware_banner_status(config)

    # Format connection pool info
    pool_info = f"max={config.max_connections}, keepalive={config.max_keepalive_connections}"
    http2_status = "ENABLED" if config.http2 else "DISABLED"

    # Backend status
    if config.backend == "anthropic":
        backend_status = "ANTHROPIC (direct API)"
    else:
        # Normalize: "bedrock" -> "litellm-bedrock"
        backend = config.backend
        if not backend.startswith("litellm-"):
            backend = f"litellm-{backend}"
        provider = backend.replace("litellm-", "")
        backend_status = f"{provider.upper()} via LiteLLM (region={config.bedrock_region})"

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      HEADROOM PROXY SERVER                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Version: 1.0.0                                                      ║
║  Listening: http://{config.host}:{config.port:<5}                                      ║
║  Workers: {workers:<3}  Concurrency Limit: {limit_concurrency:<5}                          ║
║  Backend: {backend_status:<59}║
╠══════════════════════════════════════════════════════════════════════╣
║  FEATURES:                                                           ║
║    Optimization:    {"ENABLED " if config.optimize else "DISABLED"}                                       ║
║    Caching:         {"ENABLED " if config.cache_enabled else "DISABLED"}   (TTL: {config.cache_ttl_seconds}s)                          ║
║    Rate Limiting:   {"ENABLED " if config.rate_limit_enabled else "DISABLED"}   ({config.rate_limit_requests_per_minute} req/min, {config.rate_limit_tokens_per_minute:,} tok/min)       ║
║    Retry:           {"ENABLED " if config.retry_enabled else "DISABLED"}   (max {config.retry_max_attempts} attempts)                       ║
║    Cost Tracking:   {"ENABLED " if config.cost_tracking_enabled else "DISABLED"}   (budget: {"$" + str(config.budget_limit_usd) + "/" + config.budget_period if config.budget_limit_usd else "unlimited"})          ║
║    LLMLingua:       {llmlingua_status:<52}║
║    Code-Aware:      {code_aware_status:<52}║
║    HTTP/2:          {http2_status:<52}║
║    Conn Pool:       {pool_info:<52}║
╠══════════════════════════════════════════════════════════════════════╣
║  USAGE:                                                              ║
║    Claude Code:   ANTHROPIC_BASE_URL=http://{config.host}:{config.port} claude     ║
║    Cursor:        Set base URL in settings                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  ENDPOINTS:                                                          ║
║    /health                  Health check                             ║
║    /stats                   Detailed statistics                      ║
║    /metrics                 Prometheus metrics                       ║
║    /cache/clear             Clear response cache                     ║
║    /v1/retrieve             CCR: Retrieve compressed content         ║
║    /v1/retrieve/stats       CCR: Compression store stats             ║
║    /v1/retrieve/tool_call   CCR: Handle LLM tool calls               ║
║    /v1/feedback             CCR: Feedback loop stats & patterns      ║
║    /v1/feedback/{{tool}}    CCR: Compression hints for a tool        ║
║    /v1/telemetry            Data flywheel: Telemetry stats           ║
║    /v1/telemetry/export     Data flywheel: Export for aggregation    ║
║    /v1/telemetry/tools      Data flywheel: Per-tool stats            ║
║    /v1/toin/stats           TOIN: Overall intelligence stats         ║
║    /v1/toin/patterns        TOIN: List learned patterns              ║
║    /v1/toin/pattern/{{hash}} TOIN: Pattern details by hash            ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="warning",
        workers=workers if workers > 1 else None,  # None = single process (default)
        limit_concurrency=limit_concurrency,
    )


def _get_env_bool(name: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def _get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_str(name: str, default: str) -> str:
    """Get string from environment variable."""
    return os.environ.get(name, default)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headroom Proxy Server")

    # Server
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--openai-api-url", help=f"Custom OpenAI API URL (default: {HeadroomProxy.OPENAI_API_URL})"
    )

    # Backend (anthropic direct or bedrock)
    parser.add_argument(
        "--backend",
        choices=["anthropic", "bedrock"],
        default="anthropic",
        help="Backend for Anthropic API: 'anthropic' (direct) or 'bedrock' (AWS Bedrock)",
    )
    parser.add_argument(
        "--bedrock-region",
        default="us-west-2",
        help="AWS region for Bedrock backend (default: us-west-2)",
    )
    parser.add_argument(
        "--bedrock-profile",
        help="AWS profile for Bedrock backend (default: use default credentials)",
    )

    # Connection pool (scalability)
    parser.add_argument(
        "--max-connections",
        type=int,
        default=500,
        help="Max connections to upstream APIs (default: 500)",
    )
    parser.add_argument(
        "--max-keepalive", type=int, default=100, help="Max keepalive connections (default: 100)"
    )
    parser.add_argument(
        "--no-http2",
        action="store_true",
        help="Disable HTTP/2 (enabled by default for better throughput)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, use N for multi-core)",
    )
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=1000,
        help="Max concurrent connections before 503 (default: 1000)",
    )

    # Optimization
    parser.add_argument("--no-optimize", action="store_true", help="Disable optimization")
    parser.add_argument("--min-tokens", type=int, default=500, help="Min tokens to crush")
    parser.add_argument("--max-items", type=int, default=50, help="Max items after crush")

    # Caching
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL seconds")

    # Rate limiting
    parser.add_argument("--no-rate-limit", action="store_true", help="Disable rate limiting")
    parser.add_argument("--rpm", type=int, default=60, help="Requests per minute")
    parser.add_argument("--tpm", type=int, default=100000, help="Tokens per minute")

    # Cost
    parser.add_argument("--budget", type=float, help="Budget limit in USD")
    parser.add_argument("--budget-period", choices=["hourly", "daily", "monthly"], default="daily")

    # Logging
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--log-messages", action="store_true", help="Log full messages")

    # Smart routing (content-aware compression)
    parser.add_argument(
        "--no-smart-routing",
        action="store_true",
        help="Disable smart routing (use legacy sequential pipeline)",
    )

    # LLMLingua ML-based compression
    parser.add_argument(
        "--llmlingua",
        action="store_true",
        help="Enable LLMLingua-2 ML-based compression (requires: pip install headroom-ai[llmlingua])",
    )
    parser.add_argument(
        "--no-llmlingua",
        action="store_true",
        help="Disable LLMLingua compression",
    )
    parser.add_argument(
        "--llmlingua-device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device for LLMLingua model (default: auto)",
    )
    parser.add_argument(
        "--llmlingua-rate",
        type=float,
        default=0.3,
        help="LLMLingua target compression rate, 0.0-1.0 (default: 0.3 = keep 30%%)",
    )

    # Code-aware compression
    parser.add_argument(
        "--code-aware",
        action="store_true",
        help="Enable AST-based code compression (requires: pip install headroom-ai[code])",
    )
    parser.add_argument(
        "--no-code-aware",
        action="store_true",
        help="Disable code-aware compression",
    )

    args = parser.parse_args()

    # Environment variable defaults (HEADROOM_* prefix)
    # CLI args override env vars, env vars override ProxyConfig defaults
    env_smart_routing = _get_env_bool("HEADROOM_SMART_ROUTING", True)
    env_llmlingua = _get_env_bool("HEADROOM_LLMLINGUA_ENABLED", True)
    env_code_aware = _get_env_bool("HEADROOM_CODE_AWARE_ENABLED", True)
    env_optimize = _get_env_bool("HEADROOM_OPTIMIZE", True)
    env_cache = _get_env_bool("HEADROOM_CACHE_ENABLED", True)
    env_rate_limit = _get_env_bool("HEADROOM_RATE_LIMIT_ENABLED", True)

    # Determine settings: CLI flags override env vars
    # --no-X explicitly disables, --X explicitly enables, neither uses env var
    smart_routing = env_smart_routing if not args.no_smart_routing else False
    llmlingua_enabled = (
        env_llmlingua
        if not (args.llmlingua or args.no_llmlingua)
        else (args.llmlingua or not args.no_llmlingua)
    )
    code_aware_enabled = (
        env_code_aware
        if not (args.code_aware or args.no_code_aware)
        else (args.code_aware or not args.no_code_aware)
    )
    optimize = env_optimize if not args.no_optimize else False
    cache_enabled = env_cache if not args.no_cache else False
    rate_limit_enabled = env_rate_limit if not args.no_rate_limit else False

    config = ProxyConfig(
        host=_get_env_str("HEADROOM_HOST", args.host),
        port=_get_env_int("HEADROOM_PORT", args.port),
        openai_api_url=_get_env_str("OPENAI_TARGET_API_URL", args.openai_api_url),
        # Backend settings
        backend=_get_env_str("HEADROOM_BACKEND", args.backend),  # type: ignore[arg-type]
        bedrock_region=_get_env_str("HEADROOM_BEDROCK_REGION", args.bedrock_region),
        bedrock_profile=args.bedrock_profile or os.environ.get("AWS_PROFILE"),
        optimize=optimize,
        min_tokens_to_crush=_get_env_int("HEADROOM_MIN_TOKENS", args.min_tokens),
        max_items_after_crush=_get_env_int("HEADROOM_MAX_ITEMS", args.max_items),
        cache_enabled=cache_enabled,
        cache_ttl_seconds=_get_env_int("HEADROOM_CACHE_TTL", args.cache_ttl),
        rate_limit_enabled=rate_limit_enabled,
        rate_limit_requests_per_minute=_get_env_int("HEADROOM_RPM", args.rpm),
        rate_limit_tokens_per_minute=_get_env_int("HEADROOM_TPM", args.tpm),
        budget_limit_usd=args.budget,
        budget_period=args.budget_period,
        log_file=_get_env_str("HEADROOM_LOG_FILE", args.log_file)
        if args.log_file
        else os.environ.get("HEADROOM_LOG_FILE"),
        log_full_messages=args.log_messages or _get_env_bool("HEADROOM_LOG_MESSAGES", False),
        smart_routing=smart_routing,
        llmlingua_enabled=llmlingua_enabled,
        llmlingua_device=_get_env_str("HEADROOM_LLMLINGUA_DEVICE", args.llmlingua_device),
        llmlingua_target_rate=_get_env_float("HEADROOM_LLMLINGUA_RATE", args.llmlingua_rate),
        code_aware_enabled=code_aware_enabled,
        # Connection pool settings
        max_connections=_get_env_int("HEADROOM_MAX_CONNECTIONS", args.max_connections),
        max_keepalive_connections=_get_env_int("HEADROOM_MAX_KEEPALIVE", args.max_keepalive),
        http2=not args.no_http2 and _get_env_bool("HEADROOM_HTTP2", True),
    )

    # Get worker and concurrency settings
    workers = _get_env_int("HEADROOM_WORKERS", args.workers)
    limit_concurrency = _get_env_int("HEADROOM_LIMIT_CONCURRENCY", args.limit_concurrency)

    run_server(config, workers=workers, limit_concurrency=limit_concurrency)
