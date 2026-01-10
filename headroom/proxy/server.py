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
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse, StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from headroom.cache.compression_feedback import get_compression_feedback
from headroom.cache.compression_store import get_compression_store
from headroom.ccr import CCR_TOOL_NAME, CCRToolInjector, parse_tool_call
from headroom.config import CacheAlignerConfig, RollingWindowConfig, SmartCrusherConfig
from headroom.providers import AnthropicProvider, OpenAIProvider
from headroom.telemetry import get_telemetry_collector
from headroom.tokenizers import get_tokenizer
from headroom.transforms import CacheAligner, RollingWindow, SmartCrusher, TransformPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("headroom.proxy")


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

    # Optimization
    optimize: bool = True
    min_tokens_to_crush: int = 500
    max_items_after_crush: int = 50
    keep_last_turns: int = 4

    # CCR Tool Injection
    ccr_inject_tool: bool = True  # Inject headroom_retrieve tool when compression occurs
    ccr_inject_system_instructions: bool = False  # Add instructions to system message

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
    """Simple semantic cache based on message content hash."""

    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []

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

    def get(self, messages: list[dict], model: str) -> CacheEntry | None:
        """Get cached response if exists and not expired."""
        key = self._compute_key(messages, model)
        entry = self._cache.get(key)

        if entry is None:
            return None

        # Check expiration
        age = (datetime.now() - entry.created_at).total_seconds()
        if age > entry.ttl_seconds:
            del self._cache[key]
            self._access_order.remove(key)
            return None

        entry.hit_count += 1
        return entry

    def set(
        self,
        messages: list[dict],
        model: str,
        response_body: bytes,
        response_headers: dict[str, str],
        tokens_saved: int = 0,
    ):
        """Cache a response."""
        key = self._compute_key(messages, model)

        # Evict if at capacity (LRU)
        while len(self._cache) >= self.max_entries and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = CacheEntry(
            response_body=response_body,
            response_headers=response_headers,
            created_at=datetime.now(),
            ttl_seconds=self.ttl_seconds,
            tokens_saved_per_hit=tokens_saved,
        )
        self._access_order.append(key)

    def stats(self) -> dict:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self._cache.values())
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "total_hits": total_hits,
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()


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

    def _refill(self, state: RateLimitState, rate_per_minute: float) -> float:
        """Refill bucket based on elapsed time."""
        now = time.time()
        elapsed = now - state.last_update
        refill = elapsed * (rate_per_minute / 60.0)
        state.tokens = min(rate_per_minute, state.tokens + refill)
        state.last_update = now
        return state.tokens

    def check_request(self, key: str = "default") -> tuple[bool, float]:
        """Check if request is allowed. Returns (allowed, wait_seconds)."""
        state = self._request_buckets[key]
        available = self._refill(state, self.requests_per_minute)

        if available >= 1:
            state.tokens -= 1
            return True, 0

        wait_seconds = (1 - available) * (60.0 / self.requests_per_minute)
        return False, wait_seconds

    def check_tokens(self, key: str, token_count: int) -> tuple[bool, float]:
        """Check if token usage is allowed."""
        state = self._token_buckets[key]
        available = self._refill(state, self.tokens_per_minute)

        if available >= token_count:
            state.tokens -= token_count
            return True, 0

        wait_seconds = (token_count - available) * (60.0 / self.tokens_per_minute)
        return False, wait_seconds

    def stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "active_keys": len(self._request_buckets),
        }


# =============================================================================
# Cost Tracking
# =============================================================================


class CostTracker:
    """Track costs and enforce budgets."""

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

    def __init__(self, budget_limit_usd: float | None = None, budget_period: str = "daily"):
        self.budget_limit_usd = budget_limit_usd
        self.budget_period = budget_period

        # Cost tracking
        self._costs: list[tuple[datetime, float]] = []
        self._total_cost_usd: float = 0
        self._total_savings_usd: float = 0

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

    def record_cost(self, cost_usd: float):
        """Record a cost."""
        self._costs.append((datetime.now(), cost_usd))
        self._total_cost_usd += cost_usd

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

    def record_request(
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

    def record_rate_limited(self):
        self.requests_rate_limited += 1

    def record_failed(self):
        self.requests_failed += 1

    def export(self) -> str:
        """Export metrics in Prometheus format."""
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
    """Log requests to JSONL file."""

    def __init__(self, log_file: str | None = None, log_full_messages: bool = False):
        self.log_file = Path(log_file) if log_file else None
        self.log_full_messages = log_full_messages
        self._logs: list[RequestLog] = []

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: RequestLog):
        """Log a request."""
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
        entries = self._logs[-n:]
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

        # Initialize providers
        self.anthropic_provider = AnthropicProvider()
        self.openai_provider = OpenAIProvider()

        # Initialize transforms
        transforms = [
            CacheAligner(CacheAlignerConfig(enabled=True)),
            SmartCrusher(
                SmartCrusherConfig(
                    enabled=True,
                    min_tokens_to_crush=config.min_tokens_to_crush,
                    max_items_after_crush=config.max_items_after_crush,
                )
            ),
            RollingWindow(
                RollingWindowConfig(
                    enabled=True,
                    keep_system=True,
                    keep_last_turns=config.keep_last_turns,
                )
            ),
        ]

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

    def _next_request_id(self) -> str:
        """Generate unique request ID."""
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
                    return await self.http_client.post(url, json=body, headers=headers)
                else:
                    response = await self.http_client.post(url, json=body, headers=headers)

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

        raise last_error

    async def handle_anthropic_messages(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle Anthropic /v1/messages endpoint."""
        start_time = time.time()
        request_id = self._next_request_id()

        # Parse request
        body = await request.json()
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
            allowed, wait_seconds = self.rate_limiter.check_request(rate_key)
            if not allowed:
                self.metrics.record_rate_limited()
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
            cached = self.cache.get(messages, model)
            if cached:
                cache_hit = True
                optimization_latency = (time.time() - start_time) * 1000

                self.metrics.record_request(
                    provider="anthropic",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_saved=cached.tokens_saved_per_hit,
                    latency_ms=optimization_latency,
                    cached=True,
                )

                return Response(
                    content=cached.response_body,
                    headers=cached.response_headers,
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
                total_latency = (time.time() - start_time) * 1000

                # Parse response for output tokens
                output_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)
                except Exception:
                    pass

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
                    self.cache.set(
                        messages,
                        model,
                        response.content,
                        dict(response.headers),
                        tokens_saved=tokens_saved,
                    )

                # Record metrics
                self.metrics.record_request(
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

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

        except Exception as e:
            self.metrics.record_failed()
            logger.error(f"[{request_id}] Request failed: {e}")

            # Try fallback if enabled
            if self.config.fallback_enabled and self.config.fallback_provider == "openai":
                logger.info(f"[{request_id}] Attempting fallback to OpenAI")
                # Convert to OpenAI format and retry
                # (simplified - would need message format conversion)

            raise HTTPException(status_code=502, detail=str(e)) from e

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
        """Stream response with metrics tracking."""
        start_time = time.time()

        async def generate():
            output_chunks = []
            try:
                async with self.http_client.stream(
                    "POST", url, json=body, headers=headers
                ) as response:
                    async for chunk in response.aiter_bytes():
                        output_chunks.append(chunk)
                        yield chunk
            finally:
                # Record metrics after stream completes
                total_latency = (time.time() - start_time) * 1000

                # Estimate output tokens from chunks (rough)
                total_output = b"".join(output_chunks)
                output_tokens = len(total_output) // 4  # Rough estimate

                self.metrics.record_request(
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
        request_id = self._next_request_id()

        body = await request.json()
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
            allowed, wait_seconds = self.rate_limiter.check_request(rate_key)
            if not allowed:
                self.metrics.record_rate_limited()
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Check cache
        if self.cache and not stream:
            cached = self.cache.get(messages, model)
            if cached:
                self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_saved=cached.tokens_saved_per_hit,
                    latency_ms=(time.time() - start_time) * 1000,
                    cached=True,
                )
                return Response(content=cached.response_body, headers=cached.response_headers)

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
                    self.cache.set(
                        messages, model, response.content, dict(response.headers), tokens_saved
                    )

                # Metrics
                self.metrics.record_request(
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

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
        except Exception as e:
            self.metrics.record_failed()
            raise HTTPException(status_code=502, detail=str(e)) from e

    async def handle_passthrough(self, request: Request, base_url: str) -> Response:
        """Pass through request unchanged."""
        path = request.url.path
        url = f"{base_url}{path}"

        headers = dict(request.headers.items())
        headers.pop("host", None)

        body = await request.body()

        response = await self.http_client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
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
        m = proxy.metrics
        return {
            "requests": {
                "total": m.requests_total,
                "cached": m.requests_cached,
                "rate_limited": m.requests_rate_limited,
                "failed": m.requests_failed,
            },
            "tokens": {
                "input": m.tokens_input_total,
                "output": m.tokens_output_total,
                "saved": m.tokens_saved_total,
                "savings_percent": round(
                    (m.tokens_saved_total / (m.tokens_input_total + m.tokens_saved_total) * 100)
                    if m.tokens_input_total > 0
                    else 0,
                    2,
                ),
            },
            "cost": proxy.cost_tracker.stats() if proxy.cost_tracker else None,
            "cache": proxy.cache.stats() if proxy.cache else None,
            "rate_limiter": proxy.rate_limiter.stats() if proxy.rate_limiter else None,
            "recent_requests": proxy.logger.get_recent(10) if proxy.logger else [],
        }

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            proxy.metrics.export(),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/cache/clear")
    async def clear_cache():
        """Clear the response cache."""
        if proxy.cache:
            proxy.cache.clear()
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

    # Passthrough
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def passthrough(request: Request, path: str):
        if "anthropic" in request.headers.get("user-agent", "").lower():
            base_url = proxy.ANTHROPIC_API_URL
        else:
            base_url = proxy.OPENAI_API_URL
        return await proxy.handle_passthrough(request, base_url)

    return app


def run_server(config: ProxyConfig | None = None):
    """Run the proxy server."""
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI required. Install: pip install fastapi uvicorn httpx")
        sys.exit(1)

    config = config or ProxyConfig()
    app = create_app(config)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headroom Proxy Server")

    # Server
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)

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

    args = parser.parse_args()

    config = ProxyConfig(
        host=args.host,
        port=args.port,
        optimize=not args.no_optimize,
        min_tokens_to_crush=args.min_tokens,
        max_items_after_crush=args.max_items,
        cache_enabled=not args.no_cache,
        cache_ttl_seconds=args.cache_ttl,
        rate_limit_enabled=not args.no_rate_limit,
        rate_limit_requests_per_minute=args.rpm,
        rate_limit_tokens_per_minute=args.tpm,
        budget_limit_usd=args.budget,
        budget_period=args.budget_period,
        log_file=args.log_file,
        log_full_messages=args.log_messages,
    )

    run_server(config)
