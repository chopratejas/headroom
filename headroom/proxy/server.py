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

from headroom.cache.compression_feedback import get_compression_feedback
from headroom.cache.compression_store import get_compression_store
from headroom.ccr import (
    CCR_TOOL_NAME,
    CCRResponseHandler,
    CCRToolInjector,
    ContextTracker,
    ContextTrackerConfig,
    ResponseHandlerConfig,
    parse_tool_call,
)
from headroom.config import CacheAlignerConfig, CCRConfig, RollingWindowConfig, SmartCrusherConfig
from headroom.providers import AnthropicProvider, OpenAIProvider
from headroom.telemetry import get_telemetry_collector
from headroom.tokenizers import get_tokenizer
from headroom.transforms import (
    _LLMLINGUA_AVAILABLE,
    CacheAligner,
    CodeAwareCompressor,
    CodeCompressorConfig,
    ContentRouter,
    ContentRouterConfig,
    RollingWindow,
    SmartCrusher,
    TransformPipeline,
    is_tree_sitter_available,
)

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

    # Optimization
    optimize: bool = True
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
    """

    # Fallback pricing - LiteLLM is preferred source
    # Pricing per 1M tokens (input, output, cached_input)
    PRICING = {
        # Anthropic
        "claude-3-5-sonnet": (3.00, 15.00, 0.30),
        "claude-3-5-haiku": (0.80, 4.00, 0.08),
        "claude-3-opus": (15.00, 75.00, 1.50),
        "claude-sonnet-4": (3.00, 15.00, 0.30),
        "claude-opus-4": (15.00, 75.00, 1.50),
        # OpenAI
        "gpt-4o": (2.50, 10.00, 1.25),
        "gpt-4o-mini": (0.15, 0.60, 0.075),
        "o1": (15.00, 60.00, 7.50),
        "o1-mini": (1.10, 4.40, 0.55),
        "o3-mini": (1.10, 4.40, 0.55),
        "gpt-4-turbo": (10.00, 30.00, 5.00),
    }

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

    def _get_pricing(self, model: str) -> tuple[float, float, float] | None:
        """Get pricing for model."""
        model_lower = model.lower()
        for prefix, pricing in self.PRICING.items():
            if prefix in model_lower:
                return pricing
        return None

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost in USD."""
        # Try LiteLLM first
        if LITELLM_AVAILABLE:
            try:
                cost = litellm.completion_cost(
                    model=model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )
                if cost is not None and cost > 0:
                    return float(cost)
            except Exception:
                pass

        # Fall back to hardcoded pricing
        pricing = self._get_pricing(model)
        if pricing is None:
            return None

        input_price, output_price, cached_price = pricing

        regular_input = input_tokens - cached_tokens
        cost = (
            (regular_input / 1_000_000) * input_price
            + (cached_tokens / 1_000_000) * cached_price
            + (output_tokens / 1_000_000) * output_price
        )
        return cost

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

    def __init__(self, config: ProxyConfig):
        self.config = config

        # Override OPENAI_API_URL with config if set
        if config.openai_api_url:
            HeadroomProxy.OPENAI_API_URL = config.openai_api_url

        # Initialize providers
        self.anthropic_provider = AnthropicProvider()
        self.openai_provider = OpenAIProvider()

        # Initialize transforms based on routing mode
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
                RollingWindow(
                    RollingWindowConfig(
                        enabled=True,
                        keep_system=True,
                        keep_last_turns=config.keep_last_turns,
                    )
                ),
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
                RollingWindow(
                    RollingWindowConfig(
                        enabled=True,
                        keep_system=True,
                        keep_last_turns=config.keep_last_turns,
                    )
                ),
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
            )
        )
        logger.info("Headroom Proxy started")
        logger.info(f"Optimization: {'ENABLED' if self.config.optimize else 'DISABLED'}")
        logger.info(f"Caching: {'ENABLED' if self.config.cache_enabled else 'DISABLED'}")
        logger.info(f"Rate Limiting: {'ENABLED' if self.config.rate_limit_enabled else 'DISABLED'}")

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

        # Update body
        body["messages"] = optimized_messages
        if tools is not None:
            body["tools"] = tools

        # Forward request
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
                )
            else:
                response = await self._retry_request("POST", url, headers, body)

                # Parse response for CCR handling
                resp_json = None
                try:
                    resp_json = response.json()
                except Exception:
                    pass

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

                        # Use a fresh client for CCR continuations
                        logger.info(f"CCR: Making continuation request with {len(msgs)} messages")
                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(120.0),
                        ) as ccr_client:
                            try:
                                cont_response = await ccr_client.post(
                                    url,
                                    json=continuation_body,
                                    headers=continuation_headers,
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

                total_latency = (time.time() - start_time) * 1000

                # Parse response for output tokens
                output_tokens = 0
                if resp_json:
                    usage = resp_json.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)

                # Calculate cost
                cost_usd = None
                savings_usd = None
                if self.cost_tracker:
                    cost_usd = self.cost_tracker.estimate_cost(
                        model, optimized_tokens, output_tokens
                    )
                    original_cost = self.cost_tracker.estimate_cost(
                        model, original_tokens, output_tokens
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

                # Record metrics
                await self.metrics.record_request(
                    provider="anthropic",
                    model=model,
                    input_tokens=optimized_tokens,
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
    ) -> StreamingResponse:
        """Stream response with metrics tracking.

        Calculates output size incrementally to avoid accumulating all chunks in memory.
        """
        start_time = time.time()

        async def generate():
            # Track total bytes incrementally instead of accumulating chunks
            total_bytes = 0
            try:
                async with self.http_client.stream(
                    "POST", url, json=body, headers=headers
                ) as response:
                    async for chunk in response.aiter_bytes():
                        total_bytes += len(chunk)
                        yield chunk
            finally:
                # Record metrics after stream completes
                total_latency = (time.time() - start_time) * 1000

                # Estimate output tokens from total bytes (rough estimate: ~4 bytes per token)
                output_tokens = total_bytes // 4

                await self.metrics.record_request(
                    provider=provider,
                    model=model,
                    input_tokens=optimized_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                )

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: saved {tokens_saved:,} tokens (streaming)"
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

                output_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    output_tokens = usage.get("completion_tokens", 0)
                except Exception:
                    pass

                # Cost tracking
                cost_usd = savings_usd = None
                if self.cost_tracker:
                    cost_usd = self.cost_tracker.estimate_cost(
                        model, optimized_tokens, output_tokens
                    )
                    original_cost = self.cost_tracker.estimate_cost(
                        model, original_tokens, output_tokens
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

                # Metrics
                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=optimized_tokens,
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

    async def handle_passthrough(self, request: Request, base_url: str) -> Response:
        """Pass through request unchanged."""
        path = request.url.path
        url = f"{base_url}{path}"

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

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )


# =============================================================================
# FastAPI App
# =============================================================================


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
        return await proxy.handle_passthrough(request, proxy.ANTHROPIC_API_URL)

    # OpenAI endpoints
    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        return await proxy.handle_openai_chat(request)

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


def run_server(config: ProxyConfig | None = None):
    """Run the proxy server."""
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI required. Install: pip install fastapi uvicorn httpx")
        sys.exit(1)

    config = config or ProxyConfig()
    app = create_app(config)

    llmlingua_status = _get_llmlingua_banner_status(config)
    code_aware_status = _get_code_aware_banner_status(config)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      HEADROOM PROXY SERVER                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Version: 1.0.0                                                      ║
║  Listening: http://{config.host}:{config.port:<5}                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  FEATURES:                                                           ║
║    Optimization:    {"ENABLED " if config.optimize else "DISABLED"}                                       ║
║    Caching:         {"ENABLED " if config.cache_enabled else "DISABLED"}   (TTL: {config.cache_ttl_seconds}s)                          ║
║    Rate Limiting:   {"ENABLED " if config.rate_limit_enabled else "DISABLED"}   ({config.rate_limit_requests_per_minute} req/min, {config.rate_limit_tokens_per_minute:,} tok/min)       ║
║    Retry:           {"ENABLED " if config.retry_enabled else "DISABLED"}   (max {config.retry_max_attempts} attempts)                       ║
║    Cost Tracking:   {"ENABLED " if config.cost_tracking_enabled else "DISABLED"}   (budget: {"$" + str(config.budget_limit_usd) + "/" + config.budget_period if config.budget_limit_usd else "unlimited"})          ║
║    LLMLingua:       {llmlingua_status:<52}║
║    Code-Aware:      {code_aware_status:<52}║
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
╚══════════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


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
    parser.add_argument("--openai-api-url", help=f"Custom OpenAI API URL (default: {HeadroomProxy.OPENAI_API_URL})")

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
    )

    run_server(config)
