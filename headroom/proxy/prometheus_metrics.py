"""Prometheus-compatible metrics for the Headroom proxy.

Tracks request counts, token usage, latency, overhead, TTFB,
per-transform timing, waste signals, prefix cache stats, and
cumulative savings history.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from headroom.proxy.cost import CostTracker

from headroom.proxy.savings_tracker import SavingsTracker

logger = logging.getLogger("headroom.proxy")


class PrometheusMetrics:
    """Prometheus-compatible metrics."""

    def __init__(
        self,
        savings_tracker: SavingsTracker | None = None,
        cost_tracker: CostTracker | None = None,
    ):
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
                "cache_write_5m_tokens": 0,
                "cache_write_1h_tokens": 0,
                "cache_write_5m_requests": 0,
                "cache_write_1h_requests": 0,
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
        self.savings_tracker = savings_tracker or SavingsTracker()
        self.cost_tracker = cost_tracker
        tracker_lifetime = self.savings_tracker.snapshot()["lifetime"]
        self._savings_tracker_input_tokens_offset = max(
            int(tracker_lifetime.get("total_input_tokens", 0) or 0),
            0,
        )
        self._savings_tracker_input_cost_usd_offset = max(
            float(tracker_lifetime.get("total_input_cost_usd", 0.0) or 0.0),
            0.0,
        )

        self._lock = asyncio.Lock()

    def _current_savings_tracker_totals(self) -> tuple[int, float]:
        total_input_tokens = self._savings_tracker_input_tokens_offset + self.tokens_input_total
        total_input_cost_usd = self._savings_tracker_input_cost_usd_offset

        if self.cost_tracker is None:
            return total_input_tokens, total_input_cost_usd

        try:
            cost_stats = self.cost_tracker.stats()
        except Exception:
            logger.debug("Failed to read cost tracker totals for savings history", exc_info=True)
            return total_input_tokens, total_input_cost_usd

        tracked_input_tokens = cost_stats.get("total_input_tokens")
        tracked_input_cost_usd = cost_stats.get("total_input_cost_usd")

        if tracked_input_tokens is not None:
            try:
                total_input_tokens = self._savings_tracker_input_tokens_offset + max(
                    int(tracked_input_tokens),
                    0,
                )
            except (TypeError, ValueError):
                pass

        if tracked_input_cost_usd is not None:
            try:
                total_input_cost_usd = self._savings_tracker_input_cost_usd_offset + max(
                    float(tracked_input_cost_usd),
                    0.0,
                )
            except (TypeError, ValueError):
                pass

        return total_input_tokens, total_input_cost_usd

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
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
        uncached_input_tokens: int = 0,
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
                pc["cache_write_5m_tokens"] += cache_write_5m_tokens
                pc["cache_write_1h_tokens"] += cache_write_1h_tokens
                if cache_write_5m_tokens > 0:
                    pc["cache_write_5m_requests"] += 1
                if cache_write_1h_tokens > 0:
                    pc["cache_write_1h_requests"] += 1
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
            self.savings_history.append((datetime.now().isoformat(), self.tokens_saved_total))
            # Keep last 500 data points
            if len(self.savings_history) > 500:
                self.savings_history = self.savings_history[-500:]

            total_input_tokens, total_input_cost_usd = self._current_savings_tracker_totals()
            self.savings_tracker.record_request(
                model=model,
                input_tokens=input_tokens,
                tokens_saved=tokens_saved,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                uncached_input_tokens=uncached_input_tokens,
                total_input_tokens=total_input_tokens,
                total_input_cost_usd=total_input_cost_usd,
            )

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
