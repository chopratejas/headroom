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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..cache.compression_cache import CompressionCache
    from ..memory.tracker import ComponentStats, MemoryTracker

import contextlib

import httpx

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from headroom import __version__
from headroom.backends import AnyLLMBackend, LiteLLMBackend
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
    ReadLifecycleConfig,
    RollingWindowConfig,
    SmartCrusherConfig,
)
from headroom.dashboard import get_dashboard_html
from headroom.providers import AnthropicProvider, OpenAIProvider
from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler
from headroom.telemetry import get_telemetry_collector
from headroom.telemetry.toin import get_toin
from headroom.tokenizers import get_tokenizer
from headroom.transforms import (
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
from headroom.utils import extract_user_query

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

# Always-on file logging to ~/.headroom/logs/ for `headroom perf` analysis
_HEADROOM_LOG_DIR = Path.home() / ".headroom" / "logs"


def _setup_file_logging() -> None:
    """Add a RotatingFileHandler to the headroom root logger.

    Writes to ~/.headroom/logs/proxy.log with automatic rotation:
    - Rotates at 10 MB
    - Keeps 5 backups (~50 MB max)
    """
    from logging.handlers import RotatingFileHandler

    try:
        _HEADROOM_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _HEADROOM_LOG_DIR / "proxy.log"
        handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        # Attach to the headroom root logger so all sub-loggers are captured
        logging.getLogger("headroom").addHandler(handler)
    except OSError:
        # Non-fatal: can't write logs (read-only fs, permissions, etc.)
        pass


_setup_file_logging()


def _summarize_transforms(transforms: list[str]) -> str:
    """Collapse repeated transforms into counted summary.

    e.g. ['router:excluded:tool', 'router:excluded:tool', 'read_lifecycle:stale']
      → 'router:excluded:tool*2 read_lifecycle:stale'
    """
    if not transforms:
        return "none"
    counts: dict[str, int] = {}
    for t in transforms:
        counts[t] = counts.get(t, 0) + 1
    parts = [f"{k}*{v}" if v > 1 else k for k, v in counts.items()]
    return " ".join(parts)


# Provider-specific cache discount multipliers (what fraction of input price)
# Used to calculate dollar savings from prefix caching
_CACHE_ECONOMICS = {
    "anthropic": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.25,
        "label": "Explicit breakpoints, 5-min TTL",
    },
    "openai": {
        "read_multiplier": 0.5,
        "write_multiplier": 1.0,
        "label": "Automatic, no TTL control",
    },
    "gemini": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.0,
        "label": "Explicit cachedContent, configurable TTL",
    },
    "bedrock": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.25,
        "label": "Same as Anthropic (Bedrock)",
    },
}


def _get_rtk_stats() -> dict[str, Any] | None:
    """Get rtk (Rust Token Killer) savings stats if rtk is installed.

    Reads from rtk's tracking database via `rtk gain --format json`.
    Returns None if rtk is not installed.
    """
    import shutil
    import subprocess as _sp

    rtk_bin = shutil.which("rtk")
    if not rtk_bin:
        # Check headroom-managed install
        rtk_managed = Path.home() / ".headroom" / "bin" / "rtk"
        if rtk_managed.exists():
            rtk_bin = str(rtk_managed)
        else:
            return None

    try:
        result = _sp.run(
            [rtk_bin, "gain", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            summary = data.get("summary", {})
            return {
                "installed": True,
                "total_commands": summary.get("total_commands", 0),
                "tokens_saved": summary.get("total_saved", 0),
                "avg_savings_pct": summary.get("avg_savings_pct", 0.0),
            }
    except Exception:
        pass

    return {"installed": True, "total_commands": 0, "tokens_saved": 0, "avg_savings_pct": 0.0}


def _build_prefix_cache_stats(
    metrics: PrometheusMetrics,
    cost_tracker: CostTracker | None,
) -> dict:
    """Build provider-aware prefix cache statistics for the dashboard."""
    by_provider = {}
    totals = {
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "requests": 0,
        "hit_requests": 0,
        "bust_count": 0,
        "bust_write_tokens": 0,
        "savings_usd": 0.0,
        "write_premium_usd": 0.0,
    }

    for provider, pc in metrics.cache_by_provider.items():
        if pc["requests"] == 0:
            continue

        econ = _CACHE_ECONOMICS.get(provider, _CACHE_ECONOMICS["anthropic"])
        read_mult: float = econ["read_multiplier"]  # type: ignore[assignment]
        write_mult: float = econ["write_multiplier"]  # type: ignore[assignment]

        # Get the base input price per token for the most-used model on this provider
        input_price_per_token = None
        if cost_tracker:
            for model_name in cost_tracker._tokens_sent_by_model:
                # Match model to provider
                _openai_prefixes = ("gpt", "o1", "o3", "o4")
                is_match = (
                    (provider == "anthropic" and "claude" in model_name)
                    or (provider == "openai" and any(p in model_name for p in _openai_prefixes))
                    or (provider == "gemini" and "gemini" in model_name)
                    or (provider == "bedrock" and "claude" in model_name)
                )
                if is_match:
                    price_per_1m = cost_tracker._get_list_price(model_name)
                    if price_per_1m:
                        input_price_per_token = price_per_1m / 1_000_000
                        break

        # Calculate savings:
        # Cache reads save (1.0 - read_mult) per token vs uncached input price.
        # Cache write premium is NOT deducted — it's baseline cost that the
        # client (e.g. Claude Code) pays regardless of Headroom. We track it
        # for observability but don't penalise our savings number.
        read_tokens: int = pc["cache_read_tokens"]  # type: ignore[assignment]
        write_tokens: int = pc["cache_write_tokens"]  # type: ignore[assignment]
        savings_usd = 0.0
        write_premium_usd = 0.0

        if input_price_per_token:
            # Savings from reads: tokens * price * (1.0 - read_multiplier)
            savings_usd = read_tokens * input_price_per_token * (1.0 - read_mult)
            # Write premium (observability only — not subtracted from savings)
            if write_mult > 1.0:
                write_premium_usd = write_tokens * input_price_per_token * (write_mult - 1.0)

        hit_rate = round(pc["hit_requests"] / pc["requests"] * 100, 1) if pc["requests"] > 0 else 0

        provider_stats = {
            "cache_read_tokens": read_tokens,
            "cache_write_tokens": write_tokens,
            "requests": pc["requests"],
            "hit_requests": pc["hit_requests"],
            "hit_rate": hit_rate,
            "bust_count": pc["bust_count"],
            "bust_write_tokens": pc["bust_write_tokens"],
            "read_discount": f"{(1.0 - read_mult) * 100:.0f}%",
            "write_premium": f"{(write_mult - 1.0) * 100:.0f}%" if write_mult > 1.0 else "none",
            "savings_usd": round(savings_usd, 4),
            "write_premium_usd": round(write_premium_usd, 4),
            "net_savings_usd": round(savings_usd, 4),
            "label": str(econ["label"]),
        }
        by_provider[provider] = provider_stats

        # Accumulate totals
        totals["cache_read_tokens"] += read_tokens
        totals["cache_write_tokens"] += write_tokens
        totals["requests"] += pc["requests"]
        totals["hit_requests"] += pc["hit_requests"]
        totals["bust_count"] += pc["bust_count"]
        totals["bust_write_tokens"] += pc["bust_write_tokens"]
        totals["savings_usd"] += savings_usd
        totals["write_premium_usd"] += write_premium_usd

    totals["net_savings_usd"] = round(totals["savings_usd"], 4)
    totals["savings_usd"] = round(totals["savings_usd"], 4)
    totals["write_premium_usd"] = round(totals["write_premium_usd"], 4)
    totals["hit_rate"] = (
        round(totals["hit_requests"] / totals["requests"] * 100, 1) if totals["requests"] > 0 else 0
    )

    return {
        "by_provider": by_provider,
        "totals": totals,
        "prefix_freeze": {
            "busts_avoided": metrics.prefix_freeze_busts_avoided,
            "tokens_preserved": metrics.prefix_freeze_tokens_preserved,
            "compression_foregone_tokens": metrics.prefix_freeze_compression_foregone,
            "net_benefit_tokens": (
                metrics.prefix_freeze_tokens_preserved - metrics.prefix_freeze_compression_foregone
            ),
        },
        "attribution": (
            "Prefix caching is performed by the LLM provider (Anthropic, OpenAI). "
            "Headroom reports cache stats as observed from API responses. "
            "CacheAligner and prefix freeze improve cache hit rates by stabilizing "
            "the message prefix, but baseline caching happens without Headroom."
        ),
    }


def _merge_cost_stats(
    cost_stats: dict | None,
    cache_stats: dict,
    cli_tokens_avoided: int = 0,
) -> dict | None:
    """Add prefix cache and CLI filtering savings to overall cost stats.

    Three savings layers, each on a different scope:
    - compression_savings_usd: tokens removed by proxy (SmartCrusher, etc.)
    - cache_savings_usd: prefix cache discount from provider
    - cli_filtering_savings_usd: tokens avoided by rtk before reaching context
    - savings_usd: sum of all (hero metric)
    """
    if cost_stats is None:
        return None

    cache_net = cache_stats.get("totals", {}).get("net_savings_usd", 0.0)
    compression_savings = cost_stats.get("savings_usd", 0.0)

    # Estimate CLI filtering savings: tokens_avoided * avg input price
    # Use the average input price from the cost tracker if available
    cli_savings_usd = 0.0
    if cli_tokens_avoided > 0 and LITELLM_AVAILABLE:
        # Use a conservative estimate: average across models seen
        total_input_cost = cost_stats.get("total_input_cost_usd", 0.0)
        total_input_tokens = cost_stats.get("total_input_tokens", 0)
        if total_input_tokens > 0:
            avg_price_per_token = total_input_cost / total_input_tokens
            cli_savings_usd = cli_tokens_avoided * avg_price_per_token

    return {
        **cost_stats,
        # savings_usd = Headroom's direct savings only (compression + CLI filtering)
        # Cache savings are provider-native and reported separately — NOT added here.
        "savings_usd": round(compression_savings + cli_savings_usd, 4),
        "compression_savings_usd": round(compression_savings, 4),
        "cache_savings_usd": round(cache_net, 4),
        "cli_filtering_savings_usd": round(cli_savings_usd, 4),
        "cli_tokens_avoided": cli_tokens_avoided,
    }


def _build_session_summary(
    proxy: HeadroomProxy,
    metrics: Any,
    prefix_cache_stats: dict,
    cli_tokens_avoided: int,
    total_tokens_before: int,
) -> dict[str, Any]:
    """Build a human-readable session summary from metrics and request logs.

    This is the headline view users see first in /stats — designed to answer
    "is Headroom working?" at a glance.
    """
    # Analyze per-request compression from the logger
    compressed_requests: list[dict] = []
    uncompressed_reasons: dict[str, int] = {
        "prefix_frozen": 0,
        "too_small": 0,
        "passthrough": 0,
        "no_compressible_content": 0,
    }

    if proxy.logger:
        for entry in proxy.logger._logs:
            if entry.model and "count_tokens" in entry.model:
                uncompressed_reasons["passthrough"] += 1
                continue
            if entry.tokens_saved > 0 and entry.savings_percent > 0:
                compressed_requests.append(
                    {
                        "savings_pct": round(entry.savings_percent, 1),
                        "tokens_saved": entry.tokens_saved,
                        "original": entry.input_tokens_original,
                        "optimized": entry.input_tokens_optimized,
                    }
                )
            elif entry.input_tokens_original > 0:
                # Categorize why it wasn't compressed
                transforms = entry.transforms_applied or []
                if not transforms:
                    # Pipeline returned unchanged — likely all frozen
                    uncompressed_reasons["prefix_frozen"] += 1
                elif all("excluded" in t or "protected" in t for t in transforms):
                    uncompressed_reasons["no_compressible_content"] += 1
                elif entry.input_tokens_original < 500:
                    uncompressed_reasons["too_small"] += 1
                else:
                    uncompressed_reasons["prefix_frozen"] += 1

    # Compute compression stats for requests that DID compress
    avg_compression = 0.0
    best_compression = 0.0
    best_detail = ""
    if compressed_requests:
        avg_compression = round(
            sum(r["savings_pct"] for r in compressed_requests) / len(compressed_requests),
            1,
        )
        best = max(compressed_requests, key=lambda r: r["savings_pct"])
        best_compression = best["savings_pct"]
        best_detail = f"{best['original']:,} → {best['optimized']:,} tokens"

    # Cost summary
    # cost_with_headroom = what you actually paid (includes cache discounts)
    # cost_without_headroom = counterfactual (with extra tokens at avg effective rate)
    # compression_savings = cost_without - cost_with (pure Headroom value)
    # cache_savings = provider caching discount (reported separately, NOT additive)
    cost_stats = proxy.cost_tracker.stats() if proxy.cost_tracker else {}
    cost_with = cost_stats.get("cost_with_headroom_usd", 0.0)
    cost_without = cost_stats.get("cost_without_headroom_usd", 0.0)
    cache_net = prefix_cache_stats.get("totals", {}).get("net_savings_usd", 0.0)
    compression_savings = round(cost_without - cost_with, 4) if cost_without > cost_with else 0.0
    # Total saved = compression savings only (cache is provider-native, not Headroom's)
    total_saved_usd = round(compression_savings, 2)
    savings_pct_cost = round(total_saved_usd / cost_without * 100, 1) if cost_without > 0 else 0.0

    # Primary models used
    models = dict(metrics.requests_by_model)
    primary_model = max(models, key=lambda k: models[k]) if models else "unknown"
    api_requests = sum(v for k, v in models.items() if "count_tokens" not in k)

    # Build the summary
    summary: dict[str, Any] = {
        "mode": proxy.config.mode,
        "api_requests": api_requests,
        "primary_model": primary_model,
        "compression": {
            "requests_compressed": len(compressed_requests),
            "avg_compression_pct": avg_compression,
            "best_compression_pct": best_compression,
            "best_detail": best_detail,
            "total_tokens_removed": metrics.tokens_saved_total,
        },
        "uncompressed_requests": {k: v for k, v in uncompressed_reasons.items() if v > 0},
        "cost": {
            "without_headroom_usd": round(cost_without, 2),
            "with_headroom_usd": round(cost_with, 2),
            "total_saved_usd": total_saved_usd,
            "savings_pct": savings_pct_cost,
            "breakdown": {
                "cache_savings_usd": round(cache_net, 2),
                "compression_savings_usd": round(compression_savings, 2),
            },
        },
    }

    # Add tip if token_headroom mode would help
    if proxy.config.mode == "cost_savings" and uncompressed_reasons["prefix_frozen"] > 10:
        summary["tip"] = (
            "Most requests are prefix-frozen. Set HEADROOM_MODE=token_headroom "
            "to compress frozen messages and extend your session by ~25-35%."
        )

    return summary


# Maximum request body size (100MB - increased to support image-heavy requests)
MAX_REQUEST_BODY_SIZE = 100 * 1024 * 1024

# Maximum SSE buffer size (10MB - prevents memory exhaustion from malformed streams)
MAX_SSE_BUFFER_SIZE = 10 * 1024 * 1024

# Maximum message array length (prevents DoS from deeply nested payloads)
MAX_MESSAGE_ARRAY_LENGTH = 10000


async def _read_request_json(request: Request) -> dict[str, Any]:
    """Read and parse JSON from a request, handling compressed bodies.

    Clients like OpenAI Codex may send zstd, gzip, or deflate-compressed
    request bodies.  Starlette's ``request.json()`` does not decompress
    automatically, causing a UnicodeDecodeError on compressed bytes.

    This helper inspects ``Content-Encoding``, decompresses if needed,
    then JSON-decodes the result.  It raises ``ValueError`` on any
    decompression or parse failure so callers can return a clean 400.
    """
    encoding = (request.headers.get("content-encoding") or "").lower().strip()
    raw = await request.body()

    if encoding in ("zstd", "zstandard"):
        try:
            import zstandard

            dctx = zstandard.ZstdDecompressor()
            # Use stream_reader for streaming zstd frames (no content size in header).
            # Plain decompress() fails when the frame header omits the size, which
            # is common with clients like OpenAI Codex.
            reader = dctx.stream_reader(raw)
            raw = reader.read()
            reader.close()
        except ImportError:
            raise ValueError(
                "Request body is zstd-compressed but the 'zstandard' package is not installed. "
                "Install it with: pip install zstandard"
            ) from None
        except Exception as exc:
            raise ValueError(f"Failed to decompress zstd request body: {exc}") from exc
    elif encoding == "gzip":
        import gzip as _gzip

        try:
            raw = _gzip.decompress(raw)
        except Exception as exc:
            raise ValueError(f"Failed to decompress gzip request body: {exc}") from exc
    elif encoding == "deflate":
        import zlib

        try:
            raw = zlib.decompress(raw)
        except Exception as exc:
            raise ValueError(f"Failed to decompress deflate request body: {exc}") from exc
    elif encoding == "br":
        try:
            import brotli

            raw = brotli.decompress(raw)
        except ImportError:
            raise ValueError(
                "Request body is brotli-compressed but the 'brotli' package is not installed."
            ) from None
        except Exception as exc:
            raise ValueError(f"Failed to decompress brotli request body: {exc}") from exc
    elif encoding and encoding != "identity":
        raise ValueError(f"Unsupported Content-Encoding: {encoding}")

    # Decode and parse JSON
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Request body is not valid UTF-8 (possibly compressed?): {exc}") from exc

    result: dict[str, Any] = json.loads(text)
    return result


# Maximum compression cache sessions (prevents unbounded memory growth)
MAX_COMPRESSION_CACHE_SESSIONS = 500

# Maximum rate limiter buckets (prevents DoS via spoofed API keys)
MAX_RATE_LIMITER_BUCKETS = 1000

# Compression pipeline timeout in seconds
COMPRESSION_TIMEOUT_SECONDS = 30


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

    # Performance
    optimization_latency_ms: float
    total_latency_ms: float | None

    # Metadata
    tags: dict[str, str]
    cache_hit: bool
    transforms_applied: list[str]

    # Waste signals detected in original messages
    waste_signals: dict[str, int] | None = None

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

    # Backend: "anthropic" (direct API), "litellm-*" (via LiteLLM), or "anyllm" (via any-llm)
    # LiteLLM backends: "litellm-bedrock", "litellm-vertex", "litellm-azure", etc.
    # any-llm backends: "anyllm" with --anyllm-provider (openai, mistral, groq, etc.)
    backend: str = "anthropic"
    bedrock_region: str = "us-west-2"  # AWS region for Bedrock/LiteLLM
    bedrock_profile: str | None = None  # AWS profile (optional)
    anyllm_provider: str = "openai"  # any-llm provider (openai, mistral, groq, etc.)

    # Optimization mode: "token_headroom" (default) or "cost_savings"
    # token_headroom: compress older messages for session extension
    # cost_savings: preserve prefix cache for cost reduction
    mode: str = "token_headroom"

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

    # Code-aware compression (ON by default if installed)
    code_aware_enabled: bool = True  # Enable AST-based code compression

    # Per-tool compression profiles (parsed from CLI/env)
    tool_profiles: dict[str, Any] | None = None

    # Read lifecycle management (compress stale/superseded Read outputs)
    read_lifecycle: bool = True  # ON by default: stale/superseded are provably safe

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

    # Prefix freeze: skip compression on already-cached messages
    prefix_freeze_enabled: bool = True  # Respect provider's prefix cache
    prefix_freeze_session_ttl: int = 600  # Session tracker TTL (seconds)

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
    traffic_learning_enabled: bool = False  # Live traffic pattern learning (--learn)
    memory_use_native_tool: bool = False  # Use Anthropic's native memory_20250818 tool
    memory_inject_context: bool = True  # Inject searched memories into context
    memory_top_k: int = 10  # Number of memories to inject
    memory_min_similarity: float = 0.3  # Minimum similarity threshold
    # Qdrant+Neo4j config (only used when memory_backend="qdrant-neo4j")
    memory_qdrant_host: str = "localhost"
    memory_qdrant_port: int = 6333
    memory_neo4j_uri: str = "neo4j://localhost:7687"
    memory_neo4j_user: str = "neo4j"
    memory_neo4j_password: str = "password"
    # Memory Bridge (bidirectional markdown <-> Headroom sync)
    memory_bridge_enabled: bool = False
    memory_bridge_md_paths: list[str] = field(default_factory=list)
    memory_bridge_md_format: str = "auto"
    memory_bridge_auto_import: bool = False
    memory_bridge_export_path: str = ""

    # License / Usage Reporting (managed/enterprise deployments)
    license_key: str | None = None  # HEADROOM_LICENSE_KEY env var
    license_cloud_url: str = "https://app.headroomlabs.ai"
    license_report_interval: int = 300  # seconds (5 min)

    # Compression Hooks (for SaaS and advanced customization)
    hooks: Any = None  # CompressionHooks instance, or None for default behavior


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

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        from ..memory.tracker import ComponentStats

        # Calculate size - this is sync but we access _cache directly
        # Note: This is a rough estimate, not perfectly accurate under async load
        size_bytes = sys.getsizeof(self._cache)
        total_hits = 0

        for entry in self._cache.values():
            size_bytes += sys.getsizeof(entry)
            size_bytes += len(entry.response_body)
            size_bytes += sys.getsizeof(entry.response_headers)
            for k, v in entry.response_headers.items():
                size_bytes += len(k) + len(v)
            total_hits += entry.hit_count

        return ComponentStats(
            name="semantic_cache",
            entry_count=len(self._cache),
            size_bytes=size_bytes,
            budget_bytes=None,
            hits=total_hits,
            misses=0,  # Would need to track this separately
            evictions=0,  # Would need to track this separately
        )


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

    async def _cleanup_stale_buckets(self) -> None:
        """Remove buckets that haven't been used in the last 10 minutes."""
        now = time.time()
        stale_threshold = now - 600  # 10 minutes
        stale_keys = [
            k for k, v in self._request_buckets.items() if v.last_update < stale_threshold
        ]
        for k in stale_keys:
            del self._request_buckets[k]
            self._token_buckets.pop(k, None)
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limiter buckets")

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
            # Prevent unbounded bucket growth from spoofed keys
            if len(self._request_buckets) > MAX_RATE_LIMITER_BUCKETS:
                await self._cleanup_stale_buckets()
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
        self._last_prune_time: datetime = datetime.now()

        # Token savings per model (exact, no dollar estimation)
        self._tokens_saved_by_model: dict[str, int] = {}
        self._tokens_sent_by_model: dict[str, int] = {}
        self._requests_by_model: dict[str, int] = {}

        # API-reported cache breakdown per model (for accurate cost calculation)
        self._api_cache_read_by_model: dict[str, int] = {}
        self._api_cache_write_by_model: dict[str, int] = {}
        self._api_uncached_by_model: dict[str, int] = {}

    # Cache resolved model names to avoid repeated litellm lookups.
    # This is critical: litellm.cost_per_token() is synchronous and can block
    # the async event loop if it triggers I/O (lazy model info download).
    _resolved_model_cache: dict[str, str] = {}

    @classmethod
    def _resolve_litellm_model(cls, model: str) -> str:
        """Resolve model name to one LiteLLM recognizes, adding provider prefix if needed.

        Results are cached per model name to avoid blocking the event loop
        with repeated synchronous litellm lookups.
        """
        if model in cls._resolved_model_cache:
            return cls._resolved_model_cache[model]

        resolved = cls._resolve_litellm_model_uncached(model)
        cls._resolved_model_cache[model] = resolved
        return resolved

    @staticmethod
    def _resolve_litellm_model_uncached(model: str) -> str:
        """Uncached resolution — called once per unique model name."""
        if not LITELLM_AVAILABLE:
            return model

        # Try as-is first
        try:
            litellm.cost_per_token(model=model, prompt_tokens=1, completion_tokens=0)
            return model
        except Exception:
            pass

        # Try with provider prefix
        prefixes = {
            "claude-": "anthropic/",
            "gpt-": "openai/",
            "o1-": "openai/",
            "o3-": "openai/",
            "o4-": "openai/",
            "gemini-": "google/",
        }
        for pattern, prefix in prefixes.items():
            if model.startswith(pattern):
                prefixed = f"{prefix}{model}"
                try:
                    litellm.cost_per_token(model=prefixed, prompt_tokens=1, completion_tokens=0)
                    return prefixed
                except Exception:
                    break

        return model

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float | None:
        """Estimate cost in USD using LiteLLM's pricing database.

        LiteLLM natively handles cache_read and cache_creation pricing
        for all providers (Anthropic, OpenAI, Google, etc.) in a single call.

        Args:
            model: Model name for pricing lookup
            input_tokens: Non-cached input tokens (excludes cache_read)
            output_tokens: Output tokens
            cache_read_tokens: Tokens served from cache (~10% of input rate)
            cache_write_tokens: Tokens written to cache (~125% of input rate)
        """
        if not LITELLM_AVAILABLE:
            logger.warning("LiteLLM not available - cannot calculate costs")
            return None

        try:
            resolved_model = self._resolve_litellm_model(model)

            # litellm.cost_per_token handles all token types natively:
            # prompt_tokens at input rate, cache_read at ~10%, cache_creation at ~125%
            input_cost, output_cost = litellm.cost_per_token(
                model=resolved_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cache_read_input_tokens=cache_read_tokens,
                cache_creation_input_tokens=cache_write_tokens,
            )

            total_cost = input_cost + output_cost
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

    def record_tokens(
        self,
        model: str,
        tokens_saved: int,
        tokens_sent: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        uncached_tokens: int = 0,
    ):
        """Record token counts per model.

        Args:
            model: Model name.
            tokens_saved: Tokens removed by compression (Headroom's count).
            tokens_sent: Compressed message tokens sent (Headroom's count).
            cache_read_tokens: Cache read tokens from API response usage.
            cache_write_tokens: Cache write tokens from API response usage.
            uncached_tokens: Non-cached input tokens from API response usage.
        """
        self._tokens_saved_by_model[model] = (
            self._tokens_saved_by_model.get(model, 0) + tokens_saved
        )
        self._tokens_sent_by_model[model] = self._tokens_sent_by_model.get(model, 0) + tokens_sent
        self._requests_by_model[model] = self._requests_by_model.get(model, 0) + 1
        self._api_cache_read_by_model[model] = (
            self._api_cache_read_by_model.get(model, 0) + cache_read_tokens
        )
        self._api_cache_write_by_model[model] = (
            self._api_cache_write_by_model.get(model, 0) + cache_write_tokens
        )
        self._api_uncached_by_model[model] = (
            self._api_uncached_by_model.get(model, 0) + uncached_tokens
        )

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

    def _get_list_price(self, model: str) -> float | None:
        """Get list input price per 1M tokens for a model."""
        if not LITELLM_AVAILABLE:
            return None
        try:
            resolved = self._resolve_litellm_model(model)
            info = litellm.model_cost.get(resolved, {})
            cost_per_token = info.get("input_cost_per_token")
            return cost_per_token * 1_000_000 if cost_per_token else None
        except Exception:
            return None

    def _get_cache_prices(self, model: str) -> tuple[float, float, float] | None:
        """Get per-token prices for cache read, cache write, and uncached input.

        Returns (cache_read, cache_write, uncached) per-token costs, or None
        if pricing is unavailable. Uses LiteLLM's native cache pricing data.
        """
        if not LITELLM_AVAILABLE:
            return None
        try:
            resolved = self._resolve_litellm_model(model)
            info = litellm.model_cost.get(resolved, {})
            uncached = info.get("input_cost_per_token")
            if not uncached:
                return None
            cache_read = info.get("cache_read_input_token_cost", uncached)
            cache_write = info.get("cache_creation_input_token_cost", uncached)
            return (cache_read, cache_write, uncached)
        except Exception:
            return None

    def stats(self) -> dict:
        """Get token statistics per model."""
        per_model = {}
        total_saved = 0
        for model in sorted(self._tokens_saved_by_model.keys()):
            saved = self._tokens_saved_by_model[model]
            sent = self._tokens_sent_by_model.get(model, 0)
            reqs = self._requests_by_model.get(model, 0)
            total_saved += saved
            per_model[model] = {
                "requests": reqs,
                "tokens_saved": saved,
                "tokens_sent": sent,
                "reduction_pct": round(saved / (saved + sent) * 100, 1)
                if (saved + sent) > 0
                else 0,
            }

        # Compute actual input cost using API-reported cache breakdown and
        # LiteLLM's per-category pricing (cache reads discounted, writes at
        # premium, uncached at list). Falls back to list price when cache
        # data is unavailable.
        cost_with_headroom = 0.0
        total_billed_input_tokens = 0
        total_input_tokens = 0
        for model in self._tokens_saved_by_model:
            saved = self._tokens_saved_by_model[model]
            sent = self._tokens_sent_by_model.get(model, 0)
            cr = self._api_cache_read_by_model.get(model, 0)
            cw = self._api_cache_write_by_model.get(model, 0)
            uncached = self._api_uncached_by_model.get(model, 0)
            total_input_tokens += sent

            prices = self._get_cache_prices(model)
            if prices:
                cr_price, cw_price, uncached_price = prices
                if cr + cw + uncached > 0:
                    # Use API's real cache breakdown with LiteLLM pricing
                    model_cost = cr * cr_price + cw * cw_price + uncached * uncached_price
                    billed_tokens = cr + cw + uncached
                else:
                    # No cache data from API — fall back to list price
                    model_cost = sent * uncached_price
                    billed_tokens = sent
                cost_with_headroom += model_cost
                total_billed_input_tokens += billed_tokens

        # Compression counterfactual: value removed tokens at the effective average
        # $/input token observed in this tracker (actual spend / billed input tokens),
        # not at uncached list price — aligns savings $ with real billing mix (cache
        # reads, writes, uncached). Fallback to list-price-per-removed-token when
        # there is no billable traffic yet.
        if total_billed_input_tokens > 0:
            avg_price_per_input_token = cost_with_headroom / total_billed_input_tokens
            cost_without_headroom = cost_with_headroom + total_saved * avg_price_per_input_token
        else:
            cost_without_headroom = 0.0
            for model in self._tokens_saved_by_model:
                saved = self._tokens_saved_by_model[model]
                sent = self._tokens_sent_by_model.get(model, 0)
                cr = self._api_cache_read_by_model.get(model, 0)
                cw = self._api_cache_write_by_model.get(model, 0)
                uncached_tk = self._api_uncached_by_model.get(model, 0)
                prices = self._get_cache_prices(model)
                if prices:
                    cr_price, cw_price, uncached_price = prices
                    if cr + cw + uncached_tk > 0:
                        model_cost = cr * cr_price + cw * cw_price + uncached_tk * uncached_price
                    else:
                        model_cost = sent * uncached_price
                    cost_without_headroom += model_cost + saved * uncached_price

        return {
            "total_tokens_saved": total_saved,
            "total_input_tokens": total_input_tokens,
            "total_input_cost_usd": round(cost_with_headroom, 4),
            "per_model": per_model,
            "cost_with_headroom_usd": round(cost_with_headroom, 4),
            "cost_without_headroom_usd": round(cost_without_headroom, 4),
            "savings_usd": round(cost_without_headroom - cost_with_headroom, 4),
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
        self.latency_min_ms = float("inf")
        self.latency_max_ms = 0.0
        self.latency_count = 0

        # Headroom overhead (optimization time only, excludes LLM)
        self.overhead_sum_ms = 0.0
        self.overhead_min_ms = float("inf")
        self.overhead_max_ms = 0.0
        self.overhead_count = 0

        # Time to first byte (TTFB) from upstream — what the user actually feels
        self.ttfb_sum_ms = 0.0
        self.ttfb_min_ms = float("inf")
        self.ttfb_max_ms = 0.0
        self.ttfb_count = 0

        # Per-transform timing (name → cumulative ms, count)
        self.transform_timing_sum: dict[str, float] = defaultdict(float)
        self.transform_timing_count: dict[str, int] = defaultdict(int)
        self.transform_timing_max: dict[str, float] = defaultdict(float)

        # Aggregate waste signals
        self.waste_signals_total: dict[str, int] = defaultdict(int)

        # Provider-specific prefix cache tracking
        # Each provider has different cache economics:
        #   Anthropic: cache_read=0.1x, cache_write=1.25x, explicit breakpoints
        #   OpenAI: cache_read=0.5x, no write penalty, automatic
        #   Google: cache_read=~0.1x, explicit cachedContent API, storage cost
        #   Bedrock: no cache metrics
        self.cache_by_provider: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "requests": 0,
                "hit_requests": 0,  # requests with cache_read > 0
                "bust_count": 0,
                "bust_write_tokens": 0,
            }
        )
        # Track per-model cache request count to distinguish cold starts from busts
        self._cache_requests_by_model: dict[str, int] = defaultdict(int)

        # Prefix freeze stats (cache-aware compression)
        self.prefix_freeze_busts_avoided: int = 0
        self.prefix_freeze_tokens_preserved: int = 0
        self.prefix_freeze_compression_foregone: int = 0

        # Cumulative savings history (timestamp → cumulative tokens saved)
        self.savings_history: list[tuple[str, int]] = []

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
        overhead_ms: float = 0,
        ttfb_ms: float = 0,
        pipeline_timing: dict[str, float] | None = None,
        waste_signals: dict[str, int] | None = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
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

            # Track provider-specific prefix cache metrics
            if cache_read_tokens > 0 or cache_write_tokens > 0:
                pc = self.cache_by_provider[provider]
                pc["cache_read_tokens"] += cache_read_tokens
                pc["cache_write_tokens"] += cache_write_tokens
                pc["requests"] += 1
                if cache_read_tokens > 0:
                    pc["hit_requests"] += 1
                # Model-aware bust detection: the first request for any model
                # is always a cold start (100% write, 0% read) — not a bust.
                # Only flag as bust when a previously-warm model suddenly has
                # high write ratio, indicating prefix invalidation.
                model_req_num = self._cache_requests_by_model[model]
                self._cache_requests_by_model[model] += 1
                if provider == "anthropic" and model_req_num > 0:
                    total_cached = cache_read_tokens + cache_write_tokens
                    if total_cached > 0 and cache_write_tokens > total_cached * 0.5:
                        pc["bust_count"] += 1
                        pc["bust_write_tokens"] += cache_write_tokens

            self.latency_sum_ms += latency_ms
            self.latency_min_ms = min(self.latency_min_ms, latency_ms)
            self.latency_max_ms = max(self.latency_max_ms, latency_ms)
            self.latency_count += 1

            # Track Headroom overhead separately
            if overhead_ms > 0:
                self.overhead_sum_ms += overhead_ms
                self.overhead_min_ms = min(self.overhead_min_ms, overhead_ms)
                self.overhead_max_ms = max(self.overhead_max_ms, overhead_ms)
                self.overhead_count += 1

            # Track TTFB (time to first byte from upstream)
            if ttfb_ms > 0:
                self.ttfb_sum_ms += ttfb_ms
                self.ttfb_min_ms = min(self.ttfb_min_ms, ttfb_ms)
                self.ttfb_max_ms = max(self.ttfb_max_ms, ttfb_ms)
                self.ttfb_count += 1

            # Track per-transform timing
            if pipeline_timing:
                for name, ms in pipeline_timing.items():
                    self.transform_timing_sum[name] += ms
                    self.transform_timing_count[name] += 1
                    self.transform_timing_max[name] = max(self.transform_timing_max[name], ms)

            # Track waste signals
            if waste_signals:
                for signal_name, token_count in waste_signals.items():
                    self.waste_signals_total[signal_name] += token_count

            # Track cumulative savings history (record every request)
            from datetime import datetime

            self.savings_history.append((datetime.now().isoformat(), self.tokens_saved_total))
            # Keep last 500 data points
            if len(self.savings_history) > 500:
                self.savings_history = self.savings_history[-500:]

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

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        from ..memory.tracker import ComponentStats

        # Calculate size
        size_bytes = sys.getsizeof(self._logs)

        for log_entry in self._logs:
            size_bytes += sys.getsizeof(log_entry)
            # Add string fields
            if log_entry.request_id:
                size_bytes += len(log_entry.request_id)
            if log_entry.provider:
                size_bytes += len(log_entry.provider)
            if log_entry.model:
                size_bytes += len(log_entry.model)
            if log_entry.error:
                size_bytes += len(log_entry.error)
            # Messages and response can be large
            if log_entry.request_messages:
                size_bytes += sys.getsizeof(log_entry.request_messages)
            if log_entry.response_content:
                size_bytes += len(log_entry.response_content)

        return ComponentStats(
            name="request_logger",
            entry_count=len(self._logs),
            size_bytes=size_bytes,
            budget_bytes=None,
            hits=0,
            misses=0,
            evictions=0,
        )


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
        # Strip trailing /v1 or /v1/ to avoid double-path (e.g., .../v1/v1/models)
        if config.openai_api_url:
            url = config.openai_api_url.rstrip("/")
            if url.endswith("/v1"):
                url = url[:-3]
            HeadroomProxy.OPENAI_API_URL = url

        # Override GEMINI_API_URL with config if set
        if config.gemini_api_url:
            gurl = config.gemini_api_url.rstrip("/")
            HeadroomProxy.GEMINI_API_URL = gurl

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
            # It lazy-loads compressors only when needed
            router_config = ContentRouterConfig(
                enable_code_aware=config.code_aware_enabled,
                tool_profiles=config.tool_profiles,
                read_lifecycle=ReadLifecycleConfig(enabled=config.read_lifecycle),
            )
            # Token headroom mode: allow compression of older excluded-tool results
            if config.mode == "token_headroom":
                router_config.protect_recent_reads_fraction = 0.3
            transforms = [
                CacheAligner(CacheAlignerConfig(enabled=True)),
                ContentRouter(router_config),
                context_manager,
            ]
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

        # Prefix cache tracking: freeze already-cached messages to avoid
        # invalidating the provider's prefix cache with our transforms
        from headroom.cache.prefix_tracker import PrefixFreezeConfig, SessionTrackerStore

        self.session_tracker_store = SessionTrackerStore(
            default_config=PrefixFreezeConfig(
                enabled=config.prefix_freeze_enabled,
                session_ttl_seconds=config.prefix_freeze_session_ttl,
            )
        )

        # Compression cache store for token_headroom mode (session-scoped)
        self._compression_caches: dict[str, CompressionCache] = {}

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

        # Backend for Anthropic API (direct, LiteLLM, or any-llm)
        # Supports: "anthropic" (direct), "bedrock", "vertex", "litellm-<provider>", or "anyllm"
        self.anthropic_backend: Backend | None = None
        if config.backend != "anthropic":
            backend = config.backend

            # Handle any-llm backend
            if backend == "anyllm" or backend.startswith("anyllm-"):
                provider = config.anyllm_provider
                try:
                    self.anthropic_backend = AnyLLMBackend(provider=provider)
                    logger.info(f"any-llm backend enabled (provider={provider})")
                except ImportError as e:
                    logger.warning(f"any-llm backend not available: {e}")
                except Exception as e:
                    logger.error(f"Failed to initialize any-llm backend: {e}")
            else:
                # Handle LiteLLM backend
                # Normalize backend name: "bedrock" -> "litellm-bedrock"
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
                use_native_tool=config.memory_use_native_tool,
                inject_context=config.memory_inject_context,
                top_k=config.memory_top_k,
                min_similarity=config.memory_min_similarity,
                qdrant_host=config.memory_qdrant_host,
                qdrant_port=config.memory_qdrant_port,
                neo4j_uri=config.memory_neo4j_uri,
                neo4j_user=config.memory_neo4j_user,
                neo4j_password=config.memory_neo4j_password,
                bridge_enabled=config.memory_bridge_enabled,
                bridge_md_paths=config.memory_bridge_md_paths,
                bridge_md_format=config.memory_bridge_md_format,
                bridge_auto_import=config.memory_bridge_auto_import,
                bridge_export_path=config.memory_bridge_export_path,
            )
            self.memory_handler = MemoryHandler(memory_config)

        # Usage Reporter (license validation + phone-home for managed/enterprise)
        self.usage_reporter: UsageReporter | None = None
        if config.license_key:
            from headroom.telemetry.reporter import UsageReporter

            self.usage_reporter = UsageReporter(
                license_key=config.license_key,
                cloud_url=config.license_cloud_url,
                report_interval=config.license_report_interval,
            )

        # Traffic Learner (live pattern extraction from proxy traffic)
        # Only activates with --learn flag; requires --memory for backend
        self.traffic_learner: TrafficLearner | None = None
        if config.traffic_learning_enabled:
            from headroom.memory.traffic_learner import TrafficLearner

            self.traffic_learner = TrafficLearner(
                user_id=os.environ.get("HEADROOM_USER_ID", os.environ.get("USER", "default")),
            )

    def _get_compression_cache(self, session_id: str) -> CompressionCache:
        """Get or create a CompressionCache for a session."""
        if session_id not in self._compression_caches:
            from headroom.cache.compression_cache import CompressionCache

            # Evict oldest caches if at capacity
            if len(self._compression_caches) >= MAX_COMPRESSION_CACHE_SESSIONS:
                # Remove oldest quarter to amortize cleanup cost
                oldest_keys = list(self._compression_caches.keys())[
                    : MAX_COMPRESSION_CACHE_SESSIONS // 4
                ]
                for key in oldest_keys:
                    del self._compression_caches[key]
                logger.info(
                    "Evicted %d compression caches (exceeded %d max sessions)",
                    len(oldest_keys),
                    MAX_COMPRESSION_CACHE_SESSIONS,
                )

            self._compression_caches[session_id] = CompressionCache()
        return self._compression_caches[session_id]

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
        if self.config.mode not in ("cost_savings", "token_headroom"):
            logger.warning(
                f"Unknown HEADROOM_MODE '{self.config.mode}', falling back to 'cost_savings'"
            )
            self.config.mode = "token_headroom"
        logger.info(f"Mode: {self.config.mode}")
        if self.config.mode == "token_headroom":
            logger.info("  Prefix freeze: re-freeze after compression")
            logger.info("  Read protection window: 30%% of excluded-tool messages")
            logger.info("  CCR TTL: extended for session lifetime")
            logger.info("  Compression cache: active")
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

        # Eagerly load ALL compressors, parsers, and detectors at startup
        # This eliminates cold-start latency spikes on first requests
        self._kompress_status = "not installed"
        eager_status: dict[str, str] = {}

        if self.config.optimize:
            logger.info("Pre-loading compressors and parsers...")
            for transform in self.anthropic_pipeline.transforms:
                if hasattr(transform, "eager_load_compressors"):
                    eager_status = transform.eager_load_compressors()
                    break

        # Update internal status from eager loading results
        if eager_status.get("kompress") == "enabled":
            self._kompress_status = "enabled"
        if eager_status.get("code_aware") == "enabled":
            self._code_aware_status = "enabled"

        # Log component status
        if self._kompress_status == "enabled":
            logger.info("Kompress: ENABLED (ModernBERT token compressor)")
        elif self.config.optimize:
            logger.info("Kompress: not installed (pip install headroom-ai[ml] for ML compression)")

        if self._code_aware_status == "enabled":
            logger.info("Code-Aware: ENABLED (AST-based compression)")
            if "tree_sitter" in eager_status:
                logger.info(f"Tree-Sitter: {eager_status['tree_sitter']}")
        elif self._code_aware_status == "lazy":
            logger.info("Code-Aware: LAZY (will load when code content detected)")
        elif self._code_aware_status == "available":
            logger.info("Code-Aware: available but disabled (use --code-aware)")
        elif self._code_aware_status == "unavailable":
            logger.info("Code-Aware: not installed (pip install headroom-ai[code])")
        elif self._code_aware_status == "disabled":
            logger.info("Code-Aware: DISABLED")

        if eager_status.get("magika") == "enabled":
            logger.info("Magika: ENABLED (ML content detection)")

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
            elif isinstance(existing_system, list):
                # system is a list of content blocks (e.g., with cache_control).
                # Append memory context as a new text block — never overwrite.
                body["system"] = existing_system + [{"type": "text", "text": context}]
            else:
                # No existing system prompt — set as string
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

    async def handle_compress(self, request: Request) -> JSONResponse:
        """Compress messages without calling an LLM.

        POST /v1/compress
        Body: {"messages": [...], "model": "...", "config": {}}
        Returns compressed messages + metrics.
        """
        # Check bypass header
        if request.headers.get("x-headroom-bypass", "").lower() == "true":
            try:
                body = await _read_request_json(request)
            except (json.JSONDecodeError, ValueError) as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid request body: {e!s}"},
                )
            messages = body.get("messages", [])
            return JSONResponse(
                {
                    "messages": messages,
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "tokens_saved": 0,
                    "compression_ratio": 1.0,
                    "transforms_applied": [],
                    "ccr_hashes": [],
                }
            )

        try:
            body = await _read_request_json(request)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Invalid JSON in request body.",
                    }
                },
            )

        messages = body.get("messages")
        model = body.get("model")

        if messages is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Missing required field: messages",
                    }
                },
            )

        if model is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Missing required field: model",
                    }
                },
            )

        if not messages:
            return JSONResponse(
                {
                    "messages": [],
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "tokens_saved": 0,
                    "compression_ratio": 1.0,
                    "transforms_applied": [],
                    "ccr_hashes": [],
                }
            )

        try:
            # Use OpenAI pipeline (messages are in OpenAI format from TS SDK)
            # Allow optional token_budget to override model's context limit
            # (used by OpenClaw compact() and other callers that need tighter budgets)
            token_budget = body.get("token_budget")
            context_limit = (
                token_budget
                if token_budget and isinstance(token_budget, int)
                else self.openai_provider.get_context_limit(model)
            )
            result = self.openai_pipeline.apply(
                messages=messages,
                model=model,
                model_limit=context_limit,
            )

            return JSONResponse(
                {
                    "messages": result.messages,
                    "tokens_before": result.tokens_before,
                    "tokens_after": result.tokens_after,
                    "tokens_saved": result.tokens_before - result.tokens_after,
                    "compression_ratio": (
                        result.tokens_after / result.tokens_before
                        if result.tokens_before > 0
                        else 1.0
                    ),
                    "transforms_applied": result.transforms_applied,
                    "ccr_hashes": result.markers_inserted,
                }
            )
        except Exception as e:
            logger.exception("Compression failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "compression_error",
                        "message": str(e),
                    }
                },
            )

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
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid request body: {e!s}",
                    },
                },
            )
        model = body.get("model", "unknown")
        messages = body.get("messages", [])

        # Validate message array size
        if len(messages) > MAX_MESSAGE_ARRAY_LENGTH:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Message array too large ({len(messages)} messages). "
                        f"Maximum is {MAX_MESSAGE_ARRAY_LENGTH}.",
                    },
                },
            )

        stream = body.get("stream", False)

        # Bypass: skip ALL compression, TOIN learning, and CCR injection
        # when the caller explicitly opts out via header.
        # Prevents Headroom from corrupting sub-agent API calls
        # (e.g., Claude Code sub-agents that inherit ANTHROPIC_BASE_URL).
        _bypass = (
            request.headers.get("x-headroom-bypass", "").lower() == "true"
            or request.headers.get("x-headroom-mode", "").lower() == "passthrough"
        )
        if _bypass:
            logger.info(f"[{request_id}] Bypass: skipping compression (header)")

        # Image compression (before text optimization)
        if self.config.image_optimize and messages and not _bypass:
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
            api_key = headers.get("x-api-key", "")
            client_ip = request.client.host if request.client else "unknown"
            rate_key = f"{api_key[:16]}:{client_ip}" if api_key else client_ip
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
                    tokens_saved=0,  # Savings already counted when response was cached
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
        original_tokens = tokenizer.count_messages(messages)

        # Hook: pre_compress — let hooks modify messages before compression
        from headroom.transforms.query_echo import extract_user_query

        if self.config.hooks:
            from headroom.hooks import CompressContext

            _hook_ctx = CompressContext(
                model=model,
                user_query=extract_user_query(messages),
                provider="anthropic",
            )
            try:
                messages = self.config.hooks.pre_compress(messages, _hook_ctx)
            except Exception as e:
                logger.debug(f"[{request_id}] pre_compress hook error: {e}")

        # Apply optimization
        transforms_applied = []
        pipeline_timing: dict[str, float] = {}
        waste_signals_dict: dict[str, int] | None = None
        optimized_messages = messages
        optimized_tokens = original_tokens

        # Get prefix cache tracker for this session
        session_id = self.session_tracker_store.compute_session_id(request, model, messages)
        prefix_tracker = self.session_tracker_store.get_or_create(session_id, "anthropic")
        frozen_message_count = prefix_tracker.get_frozen_message_count()

        _compression_failed = False
        original_messages = messages  # Preserve for 400-retry fallback
        _license_ok = self.usage_reporter.should_compress if self.usage_reporter else True
        if self.config.optimize and messages and not _bypass and _license_ok:
            try:
                context_limit = self.anthropic_provider.get_context_limit(model)
                biases = (
                    self.config.hooks.compute_biases(messages, _hook_ctx)
                    if self.config.hooks
                    else None
                )

                if self.config.mode == "token_headroom":
                    comp_cache = self._get_compression_cache(session_id)

                    # Zone 1: Swap cached compressed versions into working copy
                    working_messages = comp_cache.apply_cached(messages)

                    # Re-freeze boundary: consecutive stable messages from start
                    frozen_message_count = comp_cache.compute_frozen_count(messages)

                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: self.anthropic_pipeline.apply(
                                messages=working_messages,
                                model=model,
                                model_limit=context_limit,
                                context=extract_user_query(working_messages),
                                frozen_message_count=frozen_message_count,
                                biases=biases,
                            )
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    # Cache newly compressed messages (index-aligned diff)
                    if result.messages != working_messages:
                        comp_cache.update_from_result(messages, result.messages)

                    # Always use pipeline result — Zone 1 swaps are already applied
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    # Keep original_tokens as the REAL original (pre-Zone-1-swap)
                    # so tokens_saved captures both Zone 1 + Zone 2 savings.
                    # original_tokens was set at line ~2183 from uncompressed messages.
                    optimized_tokens = result.tokens_after
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: self.anthropic_pipeline.apply(
                                messages=messages,
                                model=model,
                                model_limit=context_limit,
                                context=extract_user_query(messages),
                                frozen_message_count=frozen_message_count,
                                biases=biases,
                            )
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    if result.messages != messages:
                        optimized_messages = result.messages
                        transforms_applied = result.transforms_applied
                        pipeline_timing = result.timing
                        original_tokens = result.tokens_before
                        optimized_tokens = result.tokens_after

                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
            except Exception as e:
                logger.warning(f"Optimization failed: {e}")
                # Flag compression failure for observability
                _compression_failed = True

        tokens_saved = max(0, original_tokens - optimized_tokens)
        optimization_latency = (time.time() - start_time) * 1000

        # Hook: post_compress — let hooks observe compression results
        if self.config.hooks and tokens_saved > 0:
            from headroom.hooks import CompressEvent

            try:
                self.config.hooks.post_compress(
                    CompressEvent(
                        tokens_before=original_tokens,
                        tokens_after=optimized_tokens,
                        tokens_saved=tokens_saved,
                        compression_ratio=tokens_saved / original_tokens
                        if original_tokens > 0
                        else 0,
                        transforms_applied=transforms_applied,
                        model=model,
                        user_query=_hook_ctx.user_query if self.config.hooks else "",
                        provider="anthropic",
                    )
                )
            except Exception as e:
                logger.debug(f"[{request_id}] post_compress hook error: {e}")

        # CCR Tool Injection: Inject retrieval tool if compression occurred
        tools = body.get("tools")
        _original_tools = tools  # Preserve for diagnostic / future retry
        if (
            self.config.ccr_inject_tool or self.config.ccr_inject_system_instructions
        ) and not _bypass:
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

        # Traffic Learner: Extract patterns from inbound tool results
        if self.traffic_learner:
            try:
                # Wire backend on first use (lazy init after memory handler is ready)
                if (
                    self.traffic_learner._backend is None
                    and self.memory_handler
                    and self.memory_handler.initialized
                    and self.memory_handler.backend
                ):
                    self.traffic_learner.set_backend(self.memory_handler.backend)

                # Extract tool results from messages and learn from them
                tool_results = self.traffic_learner.extract_tool_results_from_messages(
                    optimized_messages
                )
                for tr in tool_results[-5:]:  # Only recent results
                    await self.traffic_learner.on_tool_result(
                        tool_name=tr["tool_name"],
                        tool_input=tr["input"],
                        tool_output=tr["output"],
                        is_error=tr["is_error"],
                    )

                # Also extract preference signals from user messages
                await self.traffic_learner.on_messages(optimized_messages)
            except Exception as e:
                logger.debug(f"[{request_id}] Traffic learner: {e}")

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
                        t.get("name") or t.get("type", "")
                        for t in tools
                        if t.get("name", "").startswith("memory")
                        or t.get("type", "").startswith("memory")
                    ]
                    logger.info(f"[{request_id}] Memory: Injected tools: {tool_names}")

                    # Add beta headers for native memory tool
                    beta_headers = self.memory_handler.get_beta_headers()
                    if beta_headers:
                        for key, value in beta_headers.items():
                            # Merge with existing beta header if present
                            existing = headers.get(key, "")
                            if existing and value not in existing:
                                headers[key] = f"{existing},{value}"
                            else:
                                headers[key] = value
                            logger.info(
                                f"[{request_id}] Memory: Added beta header: {key}={headers[key]}"
                            )

        # Query Echo: disabled — hurts prefix caching in long conversations.
        # The echo changes every turn, invalidating the cached prefix.
        # To re-enable, uncomment and set query_echo_enabled on ProxyConfig.

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
                        pipeline_timing=pipeline_timing,
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
                        overhead_ms=optimization_latency,
                        pipeline_timing=pipeline_timing,
                    )

                    if self.cost_tracker:
                        self.cost_tracker.record_tokens(model, tokens_saved, optimized_tokens)

                    # Log request
                    if self.logger:
                        self.logger.log(
                            RequestLog(
                                request_id=request_id,
                                timestamp=datetime.now().isoformat(),
                                provider="bedrock",
                                model=model,
                                input_tokens_original=original_tokens,
                                input_tokens_optimized=optimized_tokens,
                                output_tokens=output_tokens,
                                tokens_saved=tokens_saved,
                                savings_percent=(tokens_saved / original_tokens * 100)
                                if original_tokens > 0
                                else 0,
                                optimization_latency_ms=optimization_latency,
                                total_latency_ms=total_latency,
                                tags=tags,
                                cache_hit=False,
                                transforms_applied=transforms_applied,
                                request_messages=body.get("messages")
                                if self.config.log_full_messages
                                else None,
                            )
                        )

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
                    pipeline_timing=pipeline_timing,
                    prefix_tracker=prefix_tracker,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)

                # Full diagnostic dump on upstream errors.
                # Writes pre/post compression messages, tools, and error
                # to ~/.headroom/logs/debug_400/ for offline analysis.
                if response.status_code >= 400:
                    try:
                        err_body = response.json()
                        err_msg = err_body.get("error", {}).get("message", "")
                        err_type = err_body.get("error", {}).get("type", "")
                    except Exception:
                        err_body = {"raw": response.text[:2000]}
                        err_msg = str(response.text[:500])
                        err_type = "parse_error"

                    logger.warning(
                        f"[{request_id}] UPSTREAM_ERROR "
                        f"status={response.status_code} "
                        f"error_type={err_type} "
                        f"error_msg={err_msg!r} "
                        f"model={model} "
                        f"compressed={'yes' if transforms_applied else 'no'} "
                        f"transforms={transforms_applied} "
                        f"original_tokens={original_tokens} "
                        f"optimized_tokens={optimized_tokens} "
                        f"message_count={len(body.get('messages', []))} "
                        f"stream={stream}"
                    )

                    # Dump full request details to debug file
                    try:
                        debug_dir = Path.home() / ".headroom" / "logs" / "debug_400"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_file = debug_dir / f"{ts}_{request_id}.json"

                        # Sanitize headers (redact API keys)
                        safe_headers = {}
                        for k, v in headers.items():
                            if k.lower() in ("x-api-key", "authorization"):
                                safe_headers[k] = v[:12] + "..." if v else ""
                            else:
                                safe_headers[k] = v

                        debug_payload = {
                            "request_id": request_id,
                            "timestamp": datetime.now().isoformat(),
                            "status_code": response.status_code,
                            "error_response": err_body,
                            "model": model,
                            "stream": stream,
                            "headers": safe_headers,
                            "compression": {
                                "was_compressed": bool(transforms_applied),
                                "transforms": transforms_applied,
                                "original_tokens": original_tokens,
                                "optimized_tokens": optimized_tokens,
                                "tokens_saved": tokens_saved,
                                "compression_failed": _compression_failed,
                            },
                            "tools_sent": body.get("tools"),
                            "tool_count": len(body.get("tools") or []),
                            "original_tool_count": len(_original_tools or []),
                            "messages_sent": body.get("messages"),
                            "message_count": len(body.get("messages", [])),
                            "original_messages": (
                                original_messages
                                if original_messages is not body.get("messages")
                                else "__same_as_sent__"
                            ),
                            "original_message_count": len(original_messages),
                            "system_prompt": body.get("system"),
                        }

                        with open(debug_file, "w") as f:
                            json.dump(debug_payload, f, indent=2, default=str)

                        logger.warning(f"[{request_id}] Full debug dump: {debug_file}")
                    except Exception as dump_err:
                        logger.error(f"[{request_id}] Failed to write debug dump: {dump_err}")

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
                            resp_headers: str | dict[str, str] = "N/A"
                            try:
                                resp_headers = dict(cont_response.headers)
                            except Exception:
                                pass
                            logger.error(
                                f"CCR: API call failed: {e}, response headers: {resp_headers}"
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
                        try:
                            ccr_content = json.dumps(final_resp_json).encode()
                        except (TypeError, ValueError) as json_err:
                            logger.warning(
                                f"[{request_id}] CCR: JSON serialization failed: {json_err}"
                            )
                            ccr_content = json.dumps(resp_json).encode()
                        response = httpx.Response(
                            status_code=200,
                            content=ccr_content,
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

                # Parse response for output token count and cache metrics
                output_tokens = 0
                cr_tokens = 0
                cw_tokens = 0
                uncached_input_tokens = 0
                if resp_json:
                    usage = resp_json.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)
                    cr_tokens = usage.get("cache_read_input_tokens", 0)
                    cw_tokens = usage.get("cache_creation_input_tokens", 0)
                    uncached_input_tokens = usage.get("input_tokens", 0)

                # Update prefix cache tracker for next turn
                prefix_tracker.update_from_response(
                    cache_read_tokens=cr_tokens,
                    cache_write_tokens=cw_tokens,
                    messages=optimized_messages,
                )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        optimized_tokens,
                        cache_read_tokens=cr_tokens,
                        cache_write_tokens=cw_tokens,
                        uncached_tokens=uncached_input_tokens,
                    )

                # Cache response
                if self.cache and response.status_code == 200:
                    await self.cache.set(
                        messages,
                        model,
                        response.content,
                        dict(response.headers),
                        tokens_saved=tokens_saved,
                    )

                # Record metrics — use optimized_tokens (what we sent), not API's
                # input_tokens which is just the non-cached portion with prompt caching
                await self.metrics.record_request(
                    provider="anthropic",
                    model=model,
                    input_tokens=optimized_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    pipeline_timing=pipeline_timing,
                    waste_signals=waste_signals_dict,
                    cache_read_tokens=cr_tokens,
                    cache_write_tokens=cw_tokens,
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
                            optimization_latency_ms=optimization_latency,
                            total_latency_ms=total_latency,
                            tags=tags,
                            cache_hit=cache_hit,
                            transforms_applied=transforms_applied,
                            waste_signals=waste_signals_dict,
                            request_messages=messages if self.config.log_full_messages else None,
                        )
                    )

                # Structured perf log line for `headroom perf` analysis
                num_msgs = len(messages)
                resp_usage = resp_json.get("usage", {}) if resp_json else {}
                cr = resp_usage.get("cache_read_input_tokens", 0)
                cw = resp_usage.get("cache_creation_input_tokens", 0)
                chp = round(cr / (cr + cw) * 100) if (cr + cw) > 0 else 0
                timing_str = (
                    " ".join(f"{k}={v:.0f}ms" for k, v in pipeline_timing.items())
                    if pipeline_timing
                    else ""
                )
                logger.info(
                    f"[{request_id}] PERF "
                    f"model={model} msgs={num_msgs} "
                    f"tok_before={original_tokens} tok_after={optimized_tokens} "
                    f"tok_saved={tokens_saved} "
                    f"cache_read={cr} cache_write={cw} cache_hit_pct={chp} "
                    f"opt_ms={optimization_latency:.0f} "
                    f"transforms={_summarize_transforms(transforms_applied)}"
                    f"{' timing=' + timing_str if timing_str else ''}"
                )

                # Remove compression headers since httpx already decompressed the response
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)  # Length changed after decompression

                # Inject Headroom compression metrics (for SaaS metering)
                response_headers["x-headroom-tokens-before"] = str(original_tokens)
                response_headers["x-headroom-tokens-after"] = str(optimized_tokens)
                response_headers["x-headroom-tokens-saved"] = str(tokens_saved)
                response_headers["x-headroom-model"] = model
                if transforms_applied:
                    response_headers["x-headroom-transforms"] = ",".join(transforms_applied)
                if cache_hit:
                    response_headers["x-headroom-cached"] = "true"
                if _compression_failed:
                    response_headers["x-headroom-compression-failed"] = "true"

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
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid request body: {e!s}",
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
        pipeline_timing: dict[str, float] = {}

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

            # Apply optimization
            try:
                context_limit = self.anthropic_provider.get_context_limit(model)
                result = self.anthropic_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(messages),
                )

                optimized_messages = result.messages
                for k, v in result.timing.items():
                    pipeline_timing[k] = pipeline_timing.get(k, 0.0) + v
                # Use pipeline's token counts for consistency with pipeline logs
                original_tokens = result.tokens_before
                optimized_tokens = result.tokens_after
                total_original_tokens += original_tokens
                total_optimized_tokens += optimized_tokens
                tokens_saved = max(0, original_tokens - optimized_tokens)
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
                overhead_ms=optimization_latency,
                pipeline_timing=pipeline_timing,
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
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": f"Invalid request body: {e!s}",
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
        pipeline_timing: dict[str, float] = {}

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

            # Apply optimization
            try:
                # Default context limit for most models
                context_limit = 128000

                # Use OpenAI pipeline (similar message format after conversion)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(messages),
                )

                optimized_messages = result.messages
                for k, v in result.timing.items():
                    pipeline_timing[k] = pipeline_timing.get(k, 0.0) + v
                # Use pipeline's token counts for consistency with pipeline logs
                original_tokens = result.tokens_before
                optimized_tokens = result.tokens_after
                total_original_tokens += original_tokens
                total_optimized_tokens += optimized_tokens
                tokens_saved = max(0, original_tokens - optimized_tokens)
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
                overhead_ms=optimization_latency,
                pipeline_timing=pipeline_timing,
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

    def _record_ccr_feedback_from_response(
        self, response: dict, provider: str, request_id: str
    ) -> None:
        """Extract headroom_retrieve tool calls from a response and record feedback.

        This closes the TOIN feedback loop for streaming responses where
        the proxy can't intercept and handle retrieval calls inline.
        """
        from headroom.cache.compression_store import get_compression_store

        content = response.get("content", [])
        if not isinstance(content, list):
            return

        store = get_compression_store()

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") != "headroom_retrieve":
                continue

            input_data = block.get("input", {})
            hash_key = input_data.get("hash")
            query = input_data.get("query")

            if not hash_key:
                continue

            logger.info(
                f"[{request_id}] CCR Feedback: Recording retrieval "
                f"hash={hash_key[:8]}... query={query!r}"
            )

            # Call store.retrieve()/search() for the side effect of triggering
            # the feedback chain: _log_retrieval -> process_pending_feedback
            # -> toin.record_retrieval(). We discard the returned content.
            try:
                if query:
                    store.search(hash_key, query)
                else:
                    store.retrieve(hash_key, query=None)
            except Exception as e:
                logger.debug(f"[{request_id}] CCR Feedback recording failed: {e}")

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
        pipeline_timing: dict[str, float] | None = None,
        prefix_tracker: Any | None = None,
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
            "ttfb_ms": None,  # Time to first byte from upstream
        }

        # Track if we need to handle memory tools
        memory_enabled = (
            memory_user_id is not None
            and self.memory_handler is not None
            and provider == "anthropic"
        )

        # Open connection before generator to capture upstream response headers
        # (needed to forward ratelimit headers to the client via StreamingResponse)
        assert self.http_client is not None, "http_client must be initialized before streaming"
        try:
            _upstream_req = self.http_client.build_request("POST", url, json=body, headers=headers)
            upstream_response = await self.http_client.send(_upstream_req, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            error_msg = str(e)
            logger.error(f"[{request_id}] Connection error to upstream API: {error_msg}")

            async def _error_gen():
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "connection_error",
                        "message": f"Failed to connect to upstream API: {error_msg}",
                    },
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()

            return StreamingResponse(_error_gen(), media_type="text/event-stream")

        # Forward upstream ratelimit headers to the client
        forwarded_headers = {
            k: v for k, v in upstream_response.headers.items() if "ratelimit" in k.lower()
        }

        async def generate():
            nonlocal body, memory_enabled  # May need to modify for continuation requests

            # For memory mode, we buffer the response to check for tool calls
            buffered_chunks: list[bytes] = []
            full_sse_data = ""

            try:
                async with contextlib.aclosing(upstream_response) as response:
                    async for chunk in response.aiter_bytes():
                        # Record TTFB on first chunk
                        if stream_state["ttfb_ms"] is None:
                            stream_state["ttfb_ms"] = (time.time() - start_time) * 1000

                        stream_state["total_bytes"] += len(chunk)

                        # Buffer SSE data to handle chunks split across calls
                        chunk_str = chunk.decode("utf-8", errors="ignore")
                        stream_state["sse_buffer"] += chunk_str

                        # Safety: prevent unbounded buffer growth
                        if len(stream_state["sse_buffer"]) > MAX_SSE_BUFFER_SIZE:
                            logger.error(
                                "SSE buffer exceeded maximum size (%d bytes), "
                                "truncating to prevent memory exhaustion",
                                MAX_SSE_BUFFER_SIZE,
                            )
                            stream_state["sse_buffer"] = stream_state["sse_buffer"][
                                -MAX_SSE_BUFFER_SIZE // 2 :
                            ]

                        # Always stream immediately — buffering breaks
                        # real-time clients (LangGraph, LangChain, etc.)
                        yield chunk

                        if memory_enabled:
                            # Also buffer for post-stream memory processing
                            buffered_chunks.append(chunk)
                            full_sse_data += chunk_str
                            if len(full_sse_data) > MAX_SSE_BUFFER_SIZE:
                                logger.warning(
                                    "Memory-mode SSE buffer exceeded maximum size, "
                                    "disabling memory detection for this request"
                                )
                                memory_enabled = False

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
                # Chunks were already yielded in real-time above, so we only
                # do silent background processing here — no yielding.
                if memory_enabled and full_sse_data:
                    # Check for Claude Code credential error
                    if "only authorized for use with Claude Code" in full_sse_data:
                        logger.warning(
                            f"[{request_id}] Memory: Claude Code subscription credentials "
                            "do not support custom tool injection. Set ANTHROPIC_API_KEY "
                            "environment variable or use --no-memory-tools flag."
                        )
                        return

                    # Parse SSE to get response JSON
                    parsed_response = self._parse_sse_to_response(full_sse_data, provider)

                    if parsed_response and self.memory_handler.has_memory_tool_calls(
                        parsed_response, provider
                    ):
                        logger.info(
                            f"[{request_id}] Memory: Detected tool calls in streaming response"
                        )

                        # Execute memory tool calls silently — response already
                        # streamed so we cannot make a continuation request.
                        tool_results = await self.memory_handler.handle_memory_tool_calls(
                            parsed_response, memory_user_id, provider
                        )
                        if tool_results:
                            logger.info(
                                f"[{request_id}] Memory: Tool calls executed silently "
                                "(streaming mode — no continuation)"
                            )

                # CCR Feedback: Record headroom_retrieve tool calls for TOIN learning.
                # In streaming mode, the client handles actual retrieval, but we
                # still need to record the event so TOIN learns which fields matter.
                if self.config.ccr_inject_tool and full_sse_data:
                    ccr_parsed = (
                        parsed_response
                        if parsed_response
                        else self._parse_sse_to_response(full_sse_data, provider)
                    )
                    if ccr_parsed:
                        self._record_ccr_feedback_from_response(ccr_parsed, provider, request_id)

            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                logger.error(f"[{request_id}] Connection error to upstream API: {e}")
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "connection_error",
                        "message": f"Failed to connect to upstream API: {e}",
                    },
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
            except httpx.HTTPStatusError as e:
                logger.error(f"[{request_id}] HTTP error from upstream API: {e}")
                # Forward the upstream error response
                yield e.response.content
            except Exception as e:
                logger.error(f"[{request_id}] Unexpected streaming error: {e}")
                error_event = {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
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

                # Use optimized_tokens for dashboard metrics (what we actually sent).
                # API's input_tokens is the non-cached portion only, which is
                # misleading for aggregation (often just 1 with prompt caching).
                cache_read_tokens = stream_state["cache_read_input_tokens"]
                cache_write_tokens = stream_state["cache_creation_input_tokens"]
                uncached_input_tokens = stream_state.get("input_tokens") or 0

                # Structured perf log line for `headroom perf` analysis
                num_msgs = len(body.get("messages", []))
                cache_hit_pct = (
                    round(cache_read_tokens / (cache_read_tokens + cache_write_tokens) * 100)
                    if (cache_read_tokens + cache_write_tokens) > 0
                    else 0
                )
                logger.info(
                    f"[{request_id}] PERF "
                    f"model={model} msgs={num_msgs} "
                    f"tok_before={original_tokens} tok_after={optimized_tokens} "
                    f"tok_saved={tokens_saved} "
                    f"cache_read={cache_read_tokens} cache_write={cache_write_tokens} "
                    f"cache_hit_pct={cache_hit_pct} "
                    f"opt_ms={optimization_latency:.0f} "
                    f"transforms={_summarize_transforms(transforms_applied)}"
                )

                # Update prefix cache tracker for next turn (streaming path)
                if prefix_tracker is not None:
                    prefix_tracker.update_from_response(
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        messages=body.get("messages", []),
                    )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        optimized_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        uncached_tokens=uncached_input_tokens,
                    )

                await self.metrics.record_request(
                    provider=provider,
                    model=model,
                    input_tokens=optimized_tokens,  # What we sent, not API's non-cached count
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    ttfb_ms=stream_state["ttfb_ms"] or 0,
                    pipeline_timing=pipeline_timing,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers=forwarded_headers,
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
        pipeline_timing: dict[str, float] | None = None,
    ) -> StreamingResponse:
        """Stream response from Bedrock backend with metrics tracking.

        Translates Bedrock streaming events to Anthropic SSE format.
        """
        start_time = time.time()

        # Mutable state for the generator
        stream_state: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "ttfb_ms": None,
        }

        async def generate():
            try:
                assert self.anthropic_backend is not None

                async for event in self.anthropic_backend.stream_message(body, headers):
                    # Record TTFB on first event
                    if stream_state["ttfb_ms"] is None:
                        stream_state["ttfb_ms"] = (time.time() - start_time) * 1000

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
                    overhead_ms=optimization_latency,
                    ttfb_ms=stream_state["ttfb_ms"] or 0,
                    pipeline_timing=pipeline_timing,
                )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(model, tokens_saved, optimized_tokens)

                # Log request
                if self.logger:
                    self.logger.log(
                        RequestLog(
                            request_id=request_id,
                            timestamp=datetime.now().isoformat(),
                            provider="bedrock",
                            model=model,
                            input_tokens_original=original_tokens,
                            input_tokens_optimized=optimized_tokens,
                            output_tokens=output_tokens,
                            tokens_saved=tokens_saved,
                            savings_percent=(tokens_saved / original_tokens * 100)
                            if original_tokens > 0
                            else 0,
                            optimization_latency_ms=optimization_latency,
                            total_latency_ms=total_latency,
                            tags=tags,
                            cache_hit=False,
                            transforms_applied=transforms_applied,
                            request_messages=body.get("messages")
                            if self.config.log_full_messages
                            else None,
                        )
                    )

                # Structured perf log line for `headroom perf` analysis
                num_msgs = len(body.get("messages", []))
                logger.info(
                    f"[{request_id}] PERF "
                    f"model={model} msgs={num_msgs} "
                    f"tok_before={original_tokens} tok_after={optimized_tokens} "
                    f"tok_saved={tokens_saved} "
                    f"cache_read=0 cache_write=0 cache_hit_pct=0 "
                    f"opt_ms={optimization_latency:.0f} "
                    f"transforms={_summarize_transforms(transforms_applied)}"
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    async def _stream_openai_via_backend(
        self,
        body: dict,
        headers: dict,
        model: str,
        request_id: str,
        start_time: float,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: dict[str, str],
        optimization_latency: float,
        pipeline_timing: dict[str, float] | None = None,
    ) -> StreamingResponse:
        """Stream OpenAI chat completion response from backend.

        Routes stream:true requests through the backend's stream_openai_message(),
        yielding SSE events to the client.
        """
        assert self.anthropic_backend is not None

        async def generate():
            try:
                async for sse_chunk in self.anthropic_backend.stream_openai_message(body, headers):
                    yield sse_chunk.encode() if isinstance(sse_chunk, str) else sse_chunk
            except Exception as e:
                logger.error(f"[{request_id}] Backend streaming error: {e}")
                error_data = {
                    "error": {
                        "message": str(e),
                        "type": "api_error",
                        "code": "backend_error",
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                total_latency = (time.time() - start_time) * 1000
                await self.metrics.record_request(
                    provider=self.anthropic_backend.name,
                    model=model,
                    input_tokens=optimized_tokens,
                    output_tokens=0,  # Unknown in streaming
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    cached=False,
                    overhead_ms=optimization_latency,
                    pipeline_timing=pipeline_timing,
                )
                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens) via {self.anthropic_backend.name} [stream]"
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
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )
        model = body.get("model", "unknown")
        messages = body.get("messages", [])

        # Validate message array size
        if len(messages) > MAX_MESSAGE_ARRAY_LENGTH:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Message array too large ({len(messages)} messages). "
                        f"Maximum is {MAX_MESSAGE_ARRAY_LENGTH}.",
                        "type": "invalid_request_error",
                        "code": "invalid_request",
                    }
                },
            )

        stream = body.get("stream", False)

        # Bypass: skip ALL compression for explicit opt-out
        _bypass = (
            request.headers.get("x-headroom-bypass", "").lower() == "true"
            or request.headers.get("x-headroom-mode", "").lower() == "passthrough"
        )
        if _bypass:
            logger.info(f"[{request_id}] Bypass: skipping compression (header)")

        # Image compression (before text optimization)
        if self.config.image_optimize and messages and not _bypass:
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
                    tokens_saved=0,  # Savings already counted when response was cached
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
        original_tokens = tokenizer.count_messages(messages)

        # Hook: pre_compress
        _hook_biases = None
        if self.config.hooks:
            from headroom.hooks import CompressContext

            _hook_ctx = CompressContext(model=model, provider="openai")
            try:
                messages = self.config.hooks.pre_compress(messages, _hook_ctx)
                _hook_biases = self.config.hooks.compute_biases(messages, _hook_ctx)
            except Exception as e:
                logger.debug(f"[{request_id}] Hook error: {e}")

        # Optimization
        transforms_applied = []
        pipeline_timing: dict[str, float] = {}
        waste_signals_dict: dict[str, int] | None = None
        optimized_messages = messages
        optimized_tokens = original_tokens

        # Get prefix cache tracker for this session
        openai_session_id = self.session_tracker_store.compute_session_id(request, model, messages)
        openai_prefix_tracker = self.session_tracker_store.get_or_create(
            openai_session_id, "openai"
        )
        openai_frozen_count = openai_prefix_tracker.get_frozen_message_count()

        _compression_failed = False
        original_messages = messages  # Preserve for 400-retry fallback
        _license_ok = self.usage_reporter.should_compress if self.usage_reporter else True
        if self.config.optimize and messages and not _bypass and _license_ok:
            try:
                context_limit = self.openai_provider.get_context_limit(model)

                if self.config.mode == "token_headroom":
                    comp_cache = self._get_compression_cache(openai_session_id)

                    # Zone 1: Swap cached compressed versions
                    working_messages = comp_cache.apply_cached(messages)

                    # Re-freeze boundary
                    openai_frozen_count = comp_cache.compute_frozen_count(messages)

                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: self.openai_pipeline.apply(
                                messages=working_messages,
                                model=model,
                                model_limit=context_limit,
                                context=extract_user_query(working_messages),
                                frozen_message_count=openai_frozen_count,
                                biases=_hook_biases,
                            )
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    if result.messages != working_messages:
                        comp_cache.update_from_result(messages, result.messages)

                    # Always use pipeline result in token_headroom mode
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    # Keep original_tokens as the REAL original (pre-Zone-1-swap)
                    # so tokens_saved captures both Zone 1 + Zone 2 savings.
                    optimized_tokens = result.tokens_after
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: self.openai_pipeline.apply(
                                messages=messages,
                                model=model,
                                model_limit=context_limit,
                                context=extract_user_query(messages),
                                frozen_message_count=openai_frozen_count,
                                biases=_hook_biases,
                            )
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    if result.messages != messages:
                        optimized_messages = result.messages
                        transforms_applied = result.transforms_applied
                        pipeline_timing = result.timing
                        original_tokens = result.tokens_before
                        optimized_tokens = result.tokens_after

                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
            except Exception as e:
                logger.warning(f"Optimization failed: {e}")
                # Flag compression failure for observability
                _compression_failed = True

        tokens_saved = max(0, original_tokens - optimized_tokens)
        optimization_latency = (time.time() - start_time) * 1000

        # Hook: post_compress
        if self.config.hooks and tokens_saved > 0:
            from headroom.hooks import CompressEvent

            try:
                self.config.hooks.post_compress(
                    CompressEvent(
                        tokens_before=original_tokens,
                        tokens_after=optimized_tokens,
                        tokens_saved=tokens_saved,
                        compression_ratio=tokens_saved / original_tokens
                        if original_tokens > 0
                        else 0,
                        transforms_applied=transforms_applied,
                        model=model,
                        provider="openai",
                    )
                )
            except Exception as e:
                logger.debug(f"[{request_id}] post_compress hook error: {e}")

        # CCR Tool Injection: Inject retrieval tool if compression occurred
        tools = body.get("tools")
        _original_tools = tools  # Preserve for diagnostic / future retry
        if (
            self.config.ccr_inject_tool or self.config.ccr_inject_system_instructions
        ) and not _bypass:
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

        # Query Echo: disabled — hurts prefix caching in long conversations.

        body["messages"] = optimized_messages
        if tools is not None:
            body["tools"] = tools

        # Route through LiteLLM/any-llm backend if configured
        if self.anthropic_backend is not None:
            try:
                if stream:
                    # Streaming: use stream_openai_message() → SSE events
                    return await self._stream_openai_via_backend(
                        body,
                        headers,
                        model,
                        request_id,
                        start_time,
                        original_tokens,
                        optimized_tokens,
                        tokens_saved,
                        transforms_applied,
                        tags,
                        optimization_latency,
                        pipeline_timing=pipeline_timing,
                    )
                else:
                    # Non-streaming: use send_openai_message() → JSON
                    backend_response = await self.anthropic_backend.send_openai_message(
                        body, headers
                    )

                    if backend_response.error:
                        return JSONResponse(
                            status_code=backend_response.status_code,
                            content=backend_response.body,
                        )

                    # Track metrics
                    total_latency = (time.time() - start_time) * 1000
                    usage = backend_response.body.get("usage", {})
                    output_tokens = usage.get("completion_tokens", 0)
                    total_input_tokens = usage.get("prompt_tokens", optimized_tokens)

                    await self.metrics.record_request(
                        provider=self.anthropic_backend.name,
                        model=model,
                        input_tokens=total_input_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=tokens_saved,
                        latency_ms=total_latency,
                        cached=False,
                        overhead_ms=optimization_latency,
                        pipeline_timing=pipeline_timing,
                    )

                    if tokens_saved > 0:
                        logger.info(
                            f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                            f"(saved {tokens_saved:,} tokens) via {self.anthropic_backend.name}"
                        )

                    return JSONResponse(
                        status_code=backend_response.status_code,
                        content=backend_response.body,
                    )
            except Exception as e:
                logger.error(f"[{request_id}] Backend error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": str(e),
                            "type": "api_error",
                            "code": "backend_error",
                        }
                    },
                )

        # Direct OpenAI API (no backend configured)
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
                    pipeline_timing=pipeline_timing,
                    prefix_tracker=openai_prefix_tracker,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)

                # Full diagnostic dump on upstream errors (OpenAI handler)
                if response.status_code >= 400:
                    try:
                        err_body = response.json()
                        err_msg = err_body.get("error", {}).get("message", "")
                        err_type = err_body.get("error", {}).get("type", "")
                    except Exception:
                        err_body = {"raw": response.text[:2000]}
                        err_msg = str(response.text[:500])
                        err_type = "parse_error"

                    logger.warning(
                        f"[{request_id}] UPSTREAM_ERROR "
                        f"status={response.status_code} "
                        f"error_type={err_type} "
                        f"error_msg={err_msg!r} "
                        f"model={model} "
                        f"compressed={'yes' if transforms_applied else 'no'} "
                        f"transforms={transforms_applied} "
                        f"original_tokens={original_tokens} "
                        f"optimized_tokens={optimized_tokens} "
                        f"message_count={len(body.get('messages', []))} "
                        f"stream={stream}"
                    )

                    try:
                        debug_dir = Path.home() / ".headroom" / "logs" / "debug_400"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_file = debug_dir / f"{ts}_{request_id}.json"

                        safe_headers = {}
                        for k, v in headers.items():
                            if k.lower() in ("x-api-key", "authorization"):
                                safe_headers[k] = v[:12] + "..." if v else ""
                            else:
                                safe_headers[k] = v

                        debug_payload = {
                            "request_id": request_id,
                            "timestamp": datetime.now().isoformat(),
                            "status_code": response.status_code,
                            "error_response": err_body,
                            "model": model,
                            "stream": stream,
                            "headers": safe_headers,
                            "compression": {
                                "was_compressed": bool(transforms_applied),
                                "transforms": transforms_applied,
                                "original_tokens": original_tokens,
                                "optimized_tokens": optimized_tokens,
                                "tokens_saved": tokens_saved,
                                "compression_failed": _compression_failed,
                            },
                            "tools_sent": body.get("tools"),
                            "tool_count": len(body.get("tools") or []),
                            "original_tool_count": len(_original_tools or []),
                            "messages_sent": body.get("messages"),
                            "message_count": len(body.get("messages", [])),
                            "original_messages": (
                                original_messages
                                if original_messages is not body.get("messages")
                                else "__same_as_sent__"
                            ),
                            "original_message_count": len(original_messages),
                            "system_prompt": body.get("system"),
                        }

                        with open(debug_file, "w") as f:
                            json.dump(debug_payload, f, indent=2, default=str)

                        logger.warning(f"[{request_id}] Full debug dump: {debug_file}")
                    except Exception as dump_err:
                        logger.error(f"[{request_id}] Failed to write debug dump: {dump_err}")

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

                # Update prefix cache tracker for next turn
                openai_prefix_tracker.update_from_response(
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=0,  # OpenAI doesn't report write tokens
                    messages=optimized_messages,
                )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        optimized_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )

                # Cache
                if self.cache and response.status_code == 200:
                    await self.cache.set(
                        messages, model, response.content, dict(response.headers), tokens_saved
                    )

                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    pipeline_timing=pipeline_timing,
                    waste_signals=waste_signals_dict,
                    cache_read_tokens=cache_read_tokens,
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

                # Inject Headroom compression metrics (for SaaS metering)
                response_headers["x-headroom-tokens-before"] = str(original_tokens)
                response_headers["x-headroom-tokens-after"] = str(optimized_tokens)
                response_headers["x-headroom-tokens-saved"] = str(tokens_saved)
                response_headers["x-headroom-model"] = model
                if transforms_applied:
                    response_headers["x-headroom-transforms"] = ",".join(transforms_applied)
                if cache_read_tokens > 0:
                    response_headers["x-headroom-cached"] = "true"
                if _compression_failed:
                    response_headers["x-headroom-compression-failed"] = "true"

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
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    # =========================================================================
    # Databricks Native API
    # =========================================================================

    async def handle_databricks_invocations(
        self,
        request: Request,
        model: str,
    ) -> Response | StreamingResponse:
        """Handle Databricks native /serving-endpoints/{model}/invocations endpoint.

        This enables using the Databricks CLI directly with Headroom:
            databricks serving-endpoints query <model> --profile HEADROOM --json '{"messages": [...]}'

        The request/response format is identical to OpenAI chat completions,
        so we inject the model from the path and delegate to handle_openai_chat.
        """
        request_id = await self._next_request_id()

        try:
            body = await _read_request_json(request)
        except Exception as e:
            logger.error(f"[{request_id}] Failed to parse Databricks request body: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}
                },
            )

        # Inject model from path into body (Databricks CLI passes model in URL, not body)
        body["model"] = model

        logger.info(f"[{request_id}] Databricks invocation: model={model}")

        # Create a new request with the modified body
        # We reuse the OpenAI chat handler since the format is identical
        from starlette.requests import Request as StarletteRequest

        # Build new scope with the body already parsed
        scope = dict(request.scope)

        # Create a simple receive function that returns our modified body
        body_bytes = json.dumps(body).encode()

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        modified_request = StarletteRequest(scope, receive)

        # Delegate to the OpenAI chat handler (same format)
        return await self.handle_openai_chat(modified_request)

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
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
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

                # Compress messages using the OpenAI pipeline
                if self.config.optimize:
                    try:
                        context_limit = self.openai_provider.get_context_limit(model)
                        result = self.openai_pipeline.apply(
                            messages=messages,
                            model=model,
                            model_limit=context_limit,
                            context=extract_user_query(messages),
                        )
                        compressed_messages = result.messages
                        # Use pipeline's token counts for consistency with pipeline logs
                        original_tokens = result.tokens_before
                        compressed_tokens = result.tokens_after
                    except Exception as e:
                        logger.warning(f"[{request_id}] Compression failed for line {i}: {e}")
                        compressed_messages = messages
                        original_tokens = tokenizer.count_messages(messages)
                        compressed_tokens = original_tokens
                else:
                    compressed_messages = messages
                    original_tokens = tokenizer.count_messages(messages)
                    compressed_tokens = original_tokens

                total_original_tokens += original_tokens
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
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
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
        original_tokens = tokenizer.count_messages(messages)

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
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    total_input_tokens = usage.get("input_tokens", original_tokens)
                    output_tokens = usage.get("output_tokens", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from OpenAI passthrough response: {e}"
                    )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(model, tokens_saved, total_input_tokens)

                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
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
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
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
        original_tokens = tokenizer.count_messages(messages)

        # Optimization
        transforms_applied: list[str] = []
        waste_signals_dict: dict[str, int] | None = None
        optimized_messages = messages
        optimized_tokens = original_tokens

        _compression_failed = False
        _license_ok = self.usage_reporter.should_compress if self.usage_reporter else True
        if self.config.optimize and messages and _license_ok:
            try:
                # Use OpenAI pipeline (similar message format)
                context_limit = self.openai_provider.get_context_limit(model)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(messages),
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    # Use pipeline's token counts for consistency with pipeline logs
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
            except Exception as e:
                _compression_failed = True
                logger.warning(f"[{request_id}] Gemini optimization failed: {e}")

        tokens_saved = max(0, original_tokens - optimized_tokens)
        optimization_latency = (time.time() - start_time) * 1000

        # Query Echo: disabled — hurts prefix caching in long conversations.

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

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        optimized_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )

                await self.metrics.record_request(
                    provider="gemini",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    waste_signals=waste_signals_dict,
                    cache_read_tokens=cache_read_tokens,
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

                # Inject Headroom compression metrics (for SaaS metering)
                response_headers["x-headroom-tokens-before"] = str(original_tokens)
                response_headers["x-headroom-tokens-after"] = str(optimized_tokens)
                response_headers["x-headroom-tokens-saved"] = str(tokens_saved)
                response_headers["x-headroom-model"] = model
                if transforms_applied:
                    response_headers["x-headroom-transforms"] = ",".join(transforms_applied)
                if cache_read_tokens > 0:
                    response_headers["x-headroom-cached"] = "true"
                if _compression_failed:
                    response_headers["x-headroom-compression-failed"] = "true"

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
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
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
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
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
        original_tokens = tokenizer.count_messages(messages)

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
                    context=extract_user_query(messages),
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
            tokens_saved = (
                max(0, original_tokens - compressed_tokens) if compressed_tokens > 0 else 0
            )

            await self.metrics.record_request(
                provider="gemini",
                model=model,
                input_tokens=compressed_tokens,
                output_tokens=0,
                tokens_saved=tokens_saved,
                latency_ms=total_latency,
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


def _register_memory_components(proxy: HeadroomProxy, tracker: MemoryTracker) -> None:
    """Register all memory-tracked components with the tracker.

    This function is idempotent - it checks if components are already registered.

    Args:
        proxy: The HeadroomProxy instance.
        tracker: The MemoryTracker instance.
    """
    # Register compression store (global singleton)
    if "compression_store" not in tracker.registered_components:
        store = get_compression_store()
        tracker.register("compression_store", store.get_memory_stats)

    # Register semantic cache (instance on proxy)
    if proxy.cache and "semantic_cache" not in tracker.registered_components:
        tracker.register("semantic_cache", proxy.cache.get_memory_stats)

    # Register request logger (instance on proxy)
    if proxy.logger and "request_logger" not in tracker.registered_components:
        tracker.register("request_logger", proxy.logger.get_memory_stats)

    # Register batch context store (global singleton)
    if "batch_context_store" not in tracker.registered_components:
        try:
            from ..ccr.batch_store import get_batch_context_store

            batch_store = get_batch_context_store()
            if hasattr(batch_store, "get_memory_stats"):
                tracker.register("batch_context_store", batch_store.get_memory_stats)
        except ImportError:
            pass

    # Note: graph_store and vector_index are created per-user within the
    # LocalMemoryBackend, not as global singletons. They would need to be
    # registered when the memory system is initialized with specific backends.


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required. Install: pip install fastapi uvicorn httpx")

    from contextlib import asynccontextmanager

    config = config or ProxyConfig()

    proxy = HeadroomProxy(config)

    # Telemetry beacon (anonymous aggregate stats)
    from headroom.telemetry.beacon import TelemetryBeacon

    _beacon = TelemetryBeacon(
        port=config.port if hasattr(config, "port") else 8787,
        sdk=os.environ.get("HEADROOM_SDK", "proxy").strip() or "proxy",
        backend=config.backend if hasattr(config, "backend") else "anthropic",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        # Startup
        await proxy.startup()
        asyncio.create_task(_log_toin_stats_periodically())
        if proxy.usage_reporter:
            await proxy.usage_reporter.start(proxy)
        if proxy.traffic_learner:
            await proxy.traffic_learner.start()
        await _beacon.start()

        yield

        # Shutdown
        await _beacon.stop()
        if proxy.usage_reporter:
            await proxy.usage_reporter.stop()
        if proxy.traffic_learner:
            await proxy.traffic_learner.stop()
        await proxy.shutdown()

    app = FastAPI(
        title="Headroom Proxy",
        description="Production-ready LLM optimization proxy",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health & Metrics
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "version": __version__,
            "config": {
                "optimize": config.optimize,
                "cache": config.cache_enabled,
                "rate_limit": config.rate_limit_enabled,
            },
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Serve the Headroom dashboard UI."""
        return get_dashboard_html()

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
        min_latency_ms = (
            round(m.latency_min_ms, 2)
            if m.latency_count > 0 and m.latency_min_ms != float("inf")
            else 0
        )
        max_latency_ms = round(m.latency_max_ms, 2) if m.latency_count > 0 else 0

        # Calculate Headroom overhead (optimization time only, excludes pass-through requests)
        avg_overhead_ms = (
            round(m.overhead_sum_ms / m.overhead_count, 2) if m.overhead_count > 0 else 0
        )
        min_overhead_ms = (
            round(m.overhead_min_ms, 2)
            if m.overhead_count > 0 and m.overhead_min_ms != float("inf")
            else 0
        )
        max_overhead_ms = round(m.overhead_max_ms, 2) if m.overhead_count > 0 else 0

        # Calculate TTFB (time to first byte)
        avg_ttfb_ms = round(m.ttfb_sum_ms / m.ttfb_count, 2) if m.ttfb_count > 0 else 0
        min_ttfb_ms = (
            round(m.ttfb_min_ms, 2) if m.ttfb_count > 0 and m.ttfb_min_ms != float("inf") else 0
        )
        max_ttfb_ms = round(m.ttfb_max_ms, 2) if m.ttfb_count > 0 else 0

        # Get compression store stats
        store = get_compression_store()
        compression_stats = store.get_stats()

        # Get telemetry/TOIN stats
        telemetry = get_telemetry_collector()
        telemetry_stats = telemetry.get_stats()

        # Get feedback loop stats
        feedback = get_compression_feedback()
        feedback_stats = feedback.get_stats()

        # Build prefix cache stats once (used in both prefix_cache and cost)
        prefix_cache_stats = _build_prefix_cache_stats(m, proxy.cost_tracker)

        # Fetch CLI filtering savings (rtk — tokens avoided before reaching context)
        cli_filtering_stats = _get_rtk_stats()
        cli_tokens_avoided = (
            cli_filtering_stats.get("tokens_saved", 0) if cli_filtering_stats else 0
        )

        # Calculate total tokens before compression
        total_tokens_before = m.tokens_input_total + m.tokens_saved_total

        # Build human-readable summary
        summary = _build_session_summary(
            proxy, m, prefix_cache_stats, cli_tokens_avoided, total_tokens_before
        )
        # DEBUG: log the summary payload for external upsert consumers
        try:
            logger.debug("/stats summary data: %r", summary)
        except Exception:
            logger.warning("Failed to log /stats summary payload")

        # Compression cache stats (token_headroom mode)
        compression_cache_stats: dict = {}
        if proxy.config.mode == "token_headroom" and proxy._compression_caches:
            total_entries = 0
            total_hits = 0
            total_misses = 0
            total_tokens_saved = 0
            for cache in proxy._compression_caches.values():
                s = cache.get_stats()
                total_entries += s.get("entries", 0)
                total_hits += s.get("hits", 0)
                total_misses += s.get("misses", 0)
                total_tokens_saved += s.get("total_tokens_saved", 0)
            compression_cache_stats = {
                "mode": "token_headroom",
                "active_sessions": len(proxy._compression_caches),
                "total_entries": total_entries,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": round(total_hits / max(1, total_hits + total_misses) * 100, 1),
                "total_tokens_saved": total_tokens_saved,
            }
        else:
            compression_cache_stats = {"mode": "token_headroom"}

        # Build unified savings summary (all layers)
        compression_tokens = m.tokens_saved_total
        cache_net_usd = prefix_cache_stats.get("totals", {}).get("net_savings_usd", 0.0)
        total_tokens_all_layers = compression_tokens + cli_tokens_avoided

        return {
            "summary": summary,
            "savings": {
                "total_tokens": total_tokens_all_layers,
                "by_layer": {
                    "cli_filtering": {
                        "tokens": cli_tokens_avoided,
                        "description": "Tokens avoided by CLI output filtering (rtk) before reaching context",
                    },
                    "compression": {
                        "tokens": compression_tokens,
                        "description": "Tokens removed by proxy compression (SmartCrusher, ContentRouter, etc.)",
                    },
                    "prefix_cache": {
                        "discount_usd": round(cache_net_usd, 4),
                        "description": (
                            "Cost discount from provider prefix caching. "
                            "Headroom's CacheAligner improves hit rates; "
                            "baseline caching is provider-native."
                        ),
                    },
                },
            },
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
                "cli_tokens_avoided": cli_tokens_avoided,
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
                "min_ms": min_latency_ms,
                "max_ms": max_latency_ms,
                "total_requests": m.latency_count,
            },
            "overhead": {
                "average_ms": avg_overhead_ms,
                "min_ms": min_overhead_ms,
                "max_ms": max_overhead_ms,
            },
            "ttfb": {
                "average_ms": avg_ttfb_ms,
                "min_ms": min_ttfb_ms,
                "max_ms": max_ttfb_ms,
            },
            "pipeline_timing": {
                name: {
                    "average_ms": round(
                        m.transform_timing_sum[name] / m.transform_timing_count[name], 2
                    ),
                    "max_ms": round(m.transform_timing_max[name], 2),
                    "count": m.transform_timing_count[name],
                }
                for name in sorted(m.transform_timing_sum.keys())
            }
            if m.transform_timing_sum
            else {},
            "waste_signals": dict(m.waste_signals_total) if m.waste_signals_total else {},
            "savings_history": m.savings_history[-100:],  # Last 100 data points
            "prefix_cache": prefix_cache_stats,
            "cost": _merge_cost_stats(
                proxy.cost_tracker.stats() if proxy.cost_tracker else None,
                prefix_cache_stats,
                cli_tokens_avoided=cli_tokens_avoided,
            ),
            "compression": {
                "ccr_entries": compression_stats.get("entry_count", 0),
                "ccr_max_entries": compression_stats.get("max_entries", 0),
                "original_tokens_cached": compression_stats.get("total_original_tokens", 0),
                "compressed_tokens_cached": compression_stats.get("total_compressed_tokens", 0),
                "ccr_retrievals": compression_stats.get("total_retrievals", 0),
            },
            "compression_cache": compression_cache_stats,
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
            "cli_filtering": cli_filtering_stats,
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

    # Debug endpoints
    @app.get("/debug/memory")
    async def debug_memory():
        """Get detailed memory usage statistics.

        Returns memory usage for all tracked components including:
        - Process-level memory (RSS, VMS, percent)
        - Per-component memory usage and budgets
        - Cache hit/miss statistics
        - Total tracked vs target budget

        This endpoint is useful for debugging memory issues and
        monitoring memory budgets.
        """
        from ..memory.tracker import MemoryTracker

        tracker = MemoryTracker.get()

        # Register components if not already registered
        _register_memory_components(proxy, tracker)

        report = tracker.get_report()
        return report.to_dict()

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

    # Compression-only endpoint (for TypeScript SDK and other HTTP clients)
    @app.post("/v1/compress")
    async def compress_messages(request: Request):
        return await proxy.handle_compress(request)

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
    # Databricks Native Endpoints
    # =========================================================================

    @app.post("/serving-endpoints/{model}/invocations")
    async def databricks_invocations(request: Request, model: str):
        """Databricks native serving endpoint - compatible with Databricks CLI.

        This allows using the Databricks CLI directly with Headroom proxy:
            databricks serving-endpoints query <model> --profile HEADROOM --json '{"messages": [...]}'

        The request format is identical to OpenAI chat completions.
        """
        return await proxy.handle_databricks_invocations(request, model)

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
        # Allow explicit base URL override (for Azure, custom endpoints, etc.)
        custom_base = request.headers.get("x-headroom-base-url")
        if custom_base:
            return await proxy.handle_passthrough(request, custom_base.rstrip("/"))

        # Anthropic: sends anthropic-version header and x-api-key
        if request.headers.get("anthropic-version") or request.headers.get("x-api-key"):
            base_url = proxy.ANTHROPIC_API_URL
        # Gemini: sends x-goog-api-key
        elif request.headers.get("x-goog-api-key"):
            base_url = proxy.GEMINI_API_URL
        # Azure OpenAI: sends api-key header (not x-api-key)
        elif request.headers.get("api-key"):
            # Azure requires explicit base URL (varies per deployment)
            azure_base = request.headers.get("x-headroom-base-url", "")
            if azure_base:
                base_url = azure_base.rstrip("/")
            else:
                base_url = proxy.OPENAI_API_URL  # Fallback
        else:
            # Default: OpenAI
            base_url = proxy.OPENAI_API_URL
        return await proxy.handle_passthrough(request, base_url)

    return app


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

    code_aware_status = _get_code_aware_banner_status(config)

    # Format connection pool info
    pool_info = f"max={config.max_connections}, keepalive={config.max_keepalive_connections}"
    http2_status = "ENABLED" if config.http2 else "DISABLED"

    # Backend status - use provider registry for display info
    if config.backend == "anthropic":
        backend_status = "ANTHROPIC (direct API)"
    elif config.backend == "anyllm" or config.backend.startswith("anyllm-"):
        backend_status = f"{config.anyllm_provider.title()} via any-llm"
    else:
        from headroom.backends.litellm import get_provider_config

        provider = config.backend.replace("litellm-", "")
        provider_config = get_provider_config(provider)
        if provider_config.uses_region:
            backend_status = (
                f"{provider_config.display_name} via LiteLLM (region={config.bedrock_region})"
            )
        else:
            backend_status = f"{provider_config.display_name} via LiteLLM"

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


def _parse_tool_profiles(cli_profiles: list[str]) -> dict[str, Any]:
    """Parse tool profiles from CLI args and HEADROOM_TOOL_PROFILES env var.

    Format: ToolName:level (e.g., Grep:conservative, Bash:moderate)
    Env var format: comma-separated (e.g., "Grep:conservative,Bash:moderate")

    Returns:
        Dict mapping tool names to CompressionProfile instances.
    """
    from headroom.config import PROFILE_PRESETS, CompressionProfile

    profiles: dict[str, CompressionProfile] = {}
    raw_entries: list[str] = list(cli_profiles)

    # Also check env var
    env_val = os.environ.get("HEADROOM_TOOL_PROFILES", "")
    if env_val:
        raw_entries.extend(e.strip() for e in env_val.split(",") if e.strip())

    for entry in raw_entries:
        if ":" not in entry:
            logger.warning("Invalid tool profile format (expected ToolName:level): %s", entry)
            continue
        tool_name, level = entry.split(":", 1)
        tool_name = tool_name.strip()
        level = level.strip().lower()

        if level in PROFILE_PRESETS:
            profiles[tool_name] = PROFILE_PRESETS[level]
        else:
            logger.warning(
                "Unknown profile level '%s' for tool '%s'. Use: conservative, moderate, aggressive",
                level,
                tool_name,
            )

    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headroom Proxy Server")

    # Server
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--openai-api-url", help=f"Custom OpenAI API URL (default: {HeadroomProxy.OPENAI_API_URL})"
    )

    # Backend (anthropic direct, bedrock, openrouter, anyllm, or litellm-<provider>)
    parser.add_argument(
        "--backend",
        default="anthropic",
        help=(
            "Backend: 'anthropic' (direct), 'bedrock' (AWS), 'openrouter', "
            "'anyllm' (any-llm), or 'litellm-<provider>' (e.g., litellm-hosted_vllm, litellm-vertex)"
        ),
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
    parser.add_argument(
        "--openrouter-api-key",
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--anyllm-provider",
        default="openai",
        help="any-llm provider: openai, anthropic, mistral, groq, ollama, bedrock, etc. (default: openai)",
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
    parser.add_argument(
        "--tool-profile",
        action="append",
        default=[],
        help="Per-tool compression profile: ToolName:level (e.g., Grep:conservative, Bash:moderate, WebFetch:aggressive). "
        "Can be specified multiple times. Also settable via HEADROOM_TOOL_PROFILES env var.",
    )

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
    env_code_aware = _get_env_bool("HEADROOM_CODE_AWARE_ENABLED", True)
    env_optimize = _get_env_bool("HEADROOM_OPTIMIZE", True)
    env_cache = _get_env_bool("HEADROOM_CACHE_ENABLED", True)
    env_rate_limit = _get_env_bool("HEADROOM_RATE_LIMIT_ENABLED", True)

    # Determine settings: CLI flags override env vars
    # --no-X explicitly disables, --X explicitly enables, neither uses env var
    smart_routing = env_smart_routing if not args.no_smart_routing else False
    code_aware_enabled = (
        env_code_aware
        if not (args.code_aware or args.no_code_aware)
        else (args.code_aware or not args.no_code_aware)
    )
    optimize = env_optimize if not args.no_optimize else False
    cache_enabled = env_cache if not args.no_cache else False
    rate_limit_enabled = env_rate_limit if not args.no_rate_limit else False

    # Set OpenRouter API key from CLI if provided
    if hasattr(args, "openrouter_api_key") and args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key

    # Parse per-tool compression profiles from CLI and env var
    tool_profiles = _parse_tool_profiles(args.tool_profile)

    config = ProxyConfig(
        host=_get_env_str("HEADROOM_HOST", args.host),
        port=_get_env_int("HEADROOM_PORT", args.port),
        openai_api_url=_get_env_str("OPENAI_TARGET_API_URL", args.openai_api_url),
        # Backend settings
        backend=_get_env_str("HEADROOM_BACKEND", args.backend),  # type: ignore[arg-type]
        bedrock_region=_get_env_str("HEADROOM_BEDROCK_REGION", args.bedrock_region),
        bedrock_profile=args.bedrock_profile or os.environ.get("AWS_PROFILE"),
        anyllm_provider=_get_env_str("HEADROOM_ANYLLM_PROVIDER", args.anyllm_provider),
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
        code_aware_enabled=code_aware_enabled,
        # Connection pool settings
        max_connections=_get_env_int("HEADROOM_MAX_CONNECTIONS", args.max_connections),
        max_keepalive_connections=_get_env_int("HEADROOM_MAX_KEEPALIVE", args.max_keepalive),
        http2=not args.no_http2 and _get_env_bool("HEADROOM_HTTP2", True),
        tool_profiles=tool_profiles if tool_profiles else None,
        mode=_get_env_str("HEADROOM_MODE", "token_headroom"),
    )

    # Get worker and concurrency settings
    workers = _get_env_int("HEADROOM_WORKERS", args.workers)
    limit_concurrency = _get_env_int("HEADROOM_LIMIT_CONCURRENCY", args.limit_concurrency)

    run_server(config, workers=workers, limit_concurrency=limit_concurrency)
