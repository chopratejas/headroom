"""Prometheus-compatible metrics for the Headroom proxy.

Tracks request counts, token usage, latency, overhead, TTFB,
per-transform timing, waste signals, prefix cache stats, and
cumulative savings history.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from headroom.observability import HeadroomOtelMetrics
    from headroom.proxy.cost import CostTracker

from headroom.observability import get_otel_metrics
from headroom.proxy.savings_tracker import SavingsTracker

logger = logging.getLogger("headroom.proxy")


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: dict[str, str] | None = None) -> str:
    if not labels:
        return ""

    rendered = ",".join(
        f'{key}="{_escape_label_value(str(value))}"' for key, value in sorted(labels.items())
    )
    return f"{{{rendered}}}"


def _append_metric(
    lines: list[str],
    *,
    name: str,
    metric_type: str,
    help_text: str,
    value: int | float,
    labels: dict[str, str] | None = None,
) -> None:
    lines.extend(
        [
            f"# HELP {name} {help_text}",
            f"# TYPE {name} {metric_type}",
            f"{name}{_format_labels(labels)} {value}",
            "",
        ]
    )


class PrometheusMetrics:
    """Prometheus-compatible metrics."""

    def __init__(
        self,
        savings_tracker: SavingsTracker | None = None,
        cost_tracker: CostTracker | None = None,
        otel_metrics: HeadroomOtelMetrics | None = None,
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

        # Per-stage timing (Unit 2). Keyed by ``(path, stage)`` tuples so
        # a single metric name can distinguish between, e.g.,
        # ``openai_responses_ws`` ``upstream_connect`` and
        # ``anthropic_messages`` ``upstream_connect``.
        self.stage_timing_sum: dict[tuple[str, str], float] = defaultdict(float)
        self.stage_timing_count: dict[tuple[str, str], int] = defaultdict(int)
        self.stage_timing_max: dict[tuple[str, str], float] = defaultdict(float)

        # WS session lifecycle (Unit 3). Gauges are live counters updated
        # by the Codex handler on register/deregister + attach_tasks/
        # detach. Histograms record completed-session durations bucketed
        # by termination cause so we can distinguish slow happy-path
        # sessions from long client-hold followed by client_disconnect.
        self.active_ws_sessions: int = 0
        self.active_relay_tasks: int = 0
        self.ws_session_duration_sum_ms: dict[str, float] = defaultdict(float)
        self.ws_session_duration_count: dict[str, int] = defaultdict(int)
        self.ws_session_duration_max_ms: dict[str, float] = defaultdict(float)

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
                "uncached_input_tokens": 0,
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

        # Cache bust tracking: how many tokens lost their cache discount due to compression
        self.cache_bust_tokens_lost: int = 0
        self.cache_bust_count: int = 0

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
        # Tiny synchronous critical section for stage-timing triple updates
        # (sum + count + max must move together for a consistent scrape).
        # threading.Lock is cheaper than asyncio.Lock and does NOT contend
        # with the async ``export()`` path — scrapes snapshot these dicts
        # under this lock in a microsecond block, then build the metrics
        # string without holding anything.
        self._stage_timing_lock = threading.Lock()
        self._otel_metrics = otel_metrics

    def _get_otel_metrics(self) -> HeadroomOtelMetrics:
        return self._otel_metrics or get_otel_metrics()

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
                pc["uncached_input_tokens"] += uncached_input_tokens
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

        self._get_otel_metrics().record_proxy_request(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_saved=tokens_saved,
            latency_ms=latency_ms,
            cached=cached,
            overhead_ms=overhead_ms,
            ttfb_ms=ttfb_ms,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_5m_tokens=cache_write_5m_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
            uncached_input_tokens=uncached_input_tokens,
        )

    async def record_stage_timings(
        self,
        path: str,
        timings: dict[str, float],
    ) -> None:
        """Record per-stage timings as histogram-style observations.

        ``path`` identifies the code path that emitted the timings (e.g.
        ``openai_responses_ws`` or ``anthropic_messages``). ``timings``
        maps stage names to millisecond durations. Mirrors the
        ``transform_timing_*`` aggregation pattern so the ``/metrics``
        endpoint exposes sum/count/max series per ``(path, stage)``.

        Uses a tiny synchronous ``threading.Lock`` around the triple
        update (sum + count + max) rather than the async
        ``self._lock``: (1) the updates have no awaits, so there is no
        async contention benefit, and (2) the async lock is also held
        by ``export()`` during Prometheus scrapes — which does
        string-building while holding it. Under N concurrent request
        finalizations + an active scrape, callers would queue behind
        the scrape's string-building.
        """
        if not timings:
            return
        with self._stage_timing_lock:
            for stage, ms in timings.items():
                try:
                    ms_val = float(ms)
                except (TypeError, ValueError):
                    continue
                key = (path, stage)
                self.stage_timing_sum[key] += ms_val
                self.stage_timing_count[key] += 1
                if ms_val > self.stage_timing_max[key]:
                    self.stage_timing_max[key] = ms_val

    async def record_cache_bust(self, tokens_lost: int) -> None:
        """Record tokens that lost their cache discount due to compression."""
        async with self._lock:
            self.cache_bust_tokens_lost += tokens_lost
            self.cache_bust_count += 1
        self._get_otel_metrics().record_proxy_cache_bust(tokens_lost=tokens_lost)

    # ------------------------------------------------------------------
    # Unit 3: WS session lifecycle gauges / histogram
    # ------------------------------------------------------------------

    def inc_active_ws_sessions(self) -> None:
        """Increment the live WS session gauge (called on register)."""
        self.active_ws_sessions += 1

    def dec_active_ws_sessions(self) -> None:
        """Decrement the live WS session gauge (called on deregister)."""
        self.active_ws_sessions = max(0, self.active_ws_sessions - 1)

    def inc_active_relay_tasks(self, n: int = 1) -> None:
        """Increment the live relay-task gauge (attach_tasks)."""
        self.active_relay_tasks += n

    def dec_active_relay_tasks(self, n: int = 1) -> None:
        """Decrement the live relay-task gauge (deregister)."""
        self.active_relay_tasks = max(0, self.active_relay_tasks - n)

    def record_ws_session_duration(
        self,
        duration_ms: float,
        cause: str = "unknown",
    ) -> None:
        """Record a completed WS session's duration, bucketed by cause.

        Mirrors the ``stage_timing_*`` shape so ``/metrics`` exposes
        sum/count/max per termination cause. Uses synchronous dict
        updates (no ``_lock``) because Unit 3 callers run on the event
        loop — matching the gauges above.
        """
        try:
            ms_val = float(duration_ms)
        except (TypeError, ValueError):
            return
        self.ws_session_duration_sum_ms[cause] += ms_val
        self.ws_session_duration_count[cause] += 1
        if ms_val > self.ws_session_duration_max_ms[cause]:
            self.ws_session_duration_max_ms[cause] = ms_val

    async def record_rate_limited(self, *, provider: str | None = None, model: str | None = None):
        async with self._lock:
            self.requests_rate_limited += 1
        self._get_otel_metrics().record_proxy_rate_limited(provider=provider, model=model)

    async def record_failed(self, *, provider: str | None = None, model: str | None = None):
        async with self._lock:
            self.requests_failed += 1
        self._get_otel_metrics().record_proxy_failed(provider=provider, model=model)

    async def export(self) -> str:
        """Export metrics in Prometheus format."""
        # Snapshot stage-timing dicts under the tiny synchronous lock so
        # we don't race a concurrent ``record_stage_timings`` and observe
        # an inconsistent (sum, count, max) triple. Freeze into plain
        # dicts so the scrape's string-building below doesn't hold the
        # stage-timing lock during I/O-ish work.
        with self._stage_timing_lock:
            stage_timing_sum_snapshot = dict(self.stage_timing_sum)
            stage_timing_count_snapshot = dict(self.stage_timing_count)
            stage_timing_max_snapshot = dict(self.stage_timing_max)
        async with self._lock:
            lines: list[str] = []
            _append_metric(
                lines,
                name="headroom_requests_total",
                metric_type="counter",
                help_text="Total number of requests",
                value=self.requests_total,
            )
            _append_metric(
                lines,
                name="headroom_requests_cached_total",
                metric_type="counter",
                help_text="Cached request count",
                value=self.requests_cached,
            )
            _append_metric(
                lines,
                name="headroom_requests_rate_limited_total",
                metric_type="counter",
                help_text="Rate limited requests",
                value=self.requests_rate_limited,
            )
            _append_metric(
                lines,
                name="headroom_requests_failed_total",
                metric_type="counter",
                help_text="Failed requests",
                value=self.requests_failed,
            )
            _append_metric(
                lines,
                name="headroom_tokens_input_total",
                metric_type="counter",
                help_text="Total input tokens",
                value=self.tokens_input_total,
            )
            _append_metric(
                lines,
                name="headroom_tokens_output_total",
                metric_type="counter",
                help_text="Total output tokens",
                value=self.tokens_output_total,
            )
            _append_metric(
                lines,
                name="headroom_tokens_saved_total",
                metric_type="counter",
                help_text="Tokens saved by optimization",
                value=self.tokens_saved_total,
            )
            _append_metric(
                lines,
                name="headroom_latency_ms_sum",
                metric_type="counter",
                help_text="Sum of request latencies in milliseconds",
                value=round(self.latency_sum_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_latency_ms_count",
                metric_type="counter",
                help_text="Count of observed request latencies",
                value=self.latency_count,
            )
            _append_metric(
                lines,
                name="headroom_latency_ms_min",
                metric_type="gauge",
                help_text="Minimum observed request latency in milliseconds",
                value=0 if self.latency_count == 0 else round(self.latency_min_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_latency_ms_max",
                metric_type="gauge",
                help_text="Maximum observed request latency in milliseconds",
                value=round(self.latency_max_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_overhead_ms_sum",
                metric_type="counter",
                help_text="Sum of Headroom processing overhead in milliseconds",
                value=round(self.overhead_sum_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_overhead_ms_count",
                metric_type="counter",
                help_text="Count of observed Headroom overhead samples",
                value=self.overhead_count,
            )
            _append_metric(
                lines,
                name="headroom_overhead_ms_min",
                metric_type="gauge",
                help_text="Minimum observed Headroom overhead in milliseconds",
                value=0 if self.overhead_count == 0 else round(self.overhead_min_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_overhead_ms_max",
                metric_type="gauge",
                help_text="Maximum observed Headroom overhead in milliseconds",
                value=round(self.overhead_max_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_ttfb_ms_sum",
                metric_type="counter",
                help_text="Sum of time to first byte in milliseconds",
                value=round(self.ttfb_sum_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_ttfb_ms_count",
                metric_type="counter",
                help_text="Count of observed time-to-first-byte samples",
                value=self.ttfb_count,
            )
            _append_metric(
                lines,
                name="headroom_ttfb_ms_min",
                metric_type="gauge",
                help_text="Minimum observed time to first byte in milliseconds",
                value=0 if self.ttfb_count == 0 else round(self.ttfb_min_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_ttfb_ms_max",
                metric_type="gauge",
                help_text="Maximum observed time to first byte in milliseconds",
                value=round(self.ttfb_max_ms, 2),
            )
            _append_metric(
                lines,
                name="headroom_cache_bust_total",
                metric_type="counter",
                help_text="Requests that lost provider cache efficiency because of compression",
                value=self.cache_bust_count,
            )
            _append_metric(
                lines,
                name="headroom_cache_bust_tokens_lost_total",
                metric_type="counter",
                help_text="Tokens that lost provider cache discount because of compression",
                value=self.cache_bust_tokens_lost,
            )

            lines.extend(
                [
                    "# HELP headroom_requests_by_provider Requests by provider",
                    "# TYPE headroom_requests_by_provider counter",
                ]
            )
            for provider, count in self.requests_by_provider.items():
                lines.append(f'headroom_requests_by_provider{{provider="{provider}"}} {count}')
            lines.append("")

            lines.extend(
                [
                    "# HELP headroom_requests_by_model Requests by model",
                    "# TYPE headroom_requests_by_model counter",
                ]
            )
            for model, count in self.requests_by_model.items():
                lines.append(f'headroom_requests_by_model{{model="{model}"}} {count}')
            lines.append("")

            if self.transform_timing_sum:
                lines.extend(
                    [
                        "# HELP headroom_transform_timing_ms_sum Sum of transform timing in milliseconds",
                        "# TYPE headroom_transform_timing_ms_sum counter",
                    ]
                )
                for name, total in self.transform_timing_sum.items():
                    lines.append(
                        f'headroom_transform_timing_ms_sum{{transform="{_escape_label_value(name)}"}} {round(total, 2)}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_transform_timing_ms_count Count of transform timing samples",
                        "# TYPE headroom_transform_timing_ms_count counter",
                    ]
                )
                for name, count in self.transform_timing_count.items():
                    lines.append(
                        f'headroom_transform_timing_ms_count{{transform="{_escape_label_value(name)}"}} {count}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_transform_timing_ms_max Maximum transform timing in milliseconds",
                        "# TYPE headroom_transform_timing_ms_max gauge",
                    ]
                )
                for name, max_value in self.transform_timing_max.items():
                    lines.append(
                        f'headroom_transform_timing_ms_max{{transform="{_escape_label_value(name)}"}} {round(max_value, 2)}'
                    )
                lines.append("")

            if stage_timing_sum_snapshot:
                lines.extend(
                    [
                        "# HELP headroom_stage_timing_ms_sum Sum of per-stage handler timings in milliseconds",
                        "# TYPE headroom_stage_timing_ms_sum counter",
                    ]
                )
                for (path_label, stage), total in stage_timing_sum_snapshot.items():
                    lines.append(
                        f'headroom_stage_timing_ms_sum{{path="{_escape_label_value(path_label)}",stage="{_escape_label_value(stage)}"}} {round(total, 2)}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_stage_timing_ms_count Count of per-stage handler timing samples",
                        "# TYPE headroom_stage_timing_ms_count counter",
                    ]
                )
                for (path_label, stage), count in stage_timing_count_snapshot.items():
                    lines.append(
                        f'headroom_stage_timing_ms_count{{path="{_escape_label_value(path_label)}",stage="{_escape_label_value(stage)}"}} {count}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_stage_timing_ms_max Maximum per-stage handler timing in milliseconds",
                        "# TYPE headroom_stage_timing_ms_max gauge",
                    ]
                )
                for (path_label, stage), max_value in stage_timing_max_snapshot.items():
                    lines.append(
                        f'headroom_stage_timing_ms_max{{path="{_escape_label_value(path_label)}",stage="{_escape_label_value(stage)}"}} {round(max_value, 2)}'
                    )
                lines.append("")

            # Unit 3: WS session lifecycle gauges + duration histogram.
            lines.extend(
                [
                    "# HELP headroom_active_ws_sessions Active Codex WebSocket sessions",
                    "# TYPE headroom_active_ws_sessions gauge",
                    f"headroom_active_ws_sessions {self.active_ws_sessions}",
                    "",
                    "# HELP headroom_active_relay_tasks Active Codex WS relay tasks",
                    "# TYPE headroom_active_relay_tasks gauge",
                    f"headroom_active_relay_tasks {self.active_relay_tasks}",
                    "",
                ]
            )
            if self.ws_session_duration_sum_ms:
                lines.extend(
                    [
                        "# HELP headroom_ws_session_duration_ms_sum Sum of Codex WS session durations",
                        "# TYPE headroom_ws_session_duration_ms_sum counter",
                    ]
                )
                for cause, total in self.ws_session_duration_sum_ms.items():
                    lines.append(
                        f'headroom_ws_session_duration_ms_sum{{cause="{_escape_label_value(cause)}"}} {round(total, 2)}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_ws_session_duration_ms_count Count of completed Codex WS sessions",
                        "# TYPE headroom_ws_session_duration_ms_count counter",
                    ]
                )
                for cause, count in self.ws_session_duration_count.items():
                    lines.append(
                        f'headroom_ws_session_duration_ms_count{{cause="{_escape_label_value(cause)}"}} {count}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_ws_session_duration_ms_max Maximum Codex WS session duration",
                        "# TYPE headroom_ws_session_duration_ms_max gauge",
                    ]
                )
                for cause, max_value in self.ws_session_duration_max_ms.items():
                    lines.append(
                        f'headroom_ws_session_duration_ms_max{{cause="{_escape_label_value(cause)}"}} {round(max_value, 2)}'
                    )
                lines.append("")

            if self.waste_signals_total:
                lines.extend(
                    [
                        "# HELP headroom_waste_signal_tokens_total Tokens attributed to detected waste signals",
                        "# TYPE headroom_waste_signal_tokens_total counter",
                    ]
                )
                for signal_name, token_count in self.waste_signals_total.items():
                    lines.append(
                        f'headroom_waste_signal_tokens_total{{signal="{_escape_label_value(signal_name)}"}} {token_count}'
                    )
                lines.append("")

            if self.cache_by_provider:
                lines.extend(
                    [
                        "# HELP headroom_cache_read_tokens_total Provider cache read tokens",
                        "# TYPE headroom_cache_read_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_cache_read_tokens_total{{provider="{provider}"}} {stats["cache_read_tokens"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_cache_write_tokens_total Provider cache write tokens",
                        "# TYPE headroom_cache_write_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_cache_write_tokens_total{{provider="{provider}"}} {stats["cache_write_tokens"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_cache_write_ttl_tokens_total Provider cache write tokens by observed TTL bucket",
                        "# TYPE headroom_cache_write_ttl_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_cache_write_ttl_tokens_total{{provider="{provider}",ttl="5m"}} {stats["cache_write_5m_tokens"]}'
                    )
                    lines.append(
                        f'headroom_cache_write_ttl_tokens_total{{provider="{provider}",ttl="1h"}} {stats["cache_write_1h_tokens"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_cache_write_ttl_requests_total Provider cache write requests by observed TTL bucket",
                        "# TYPE headroom_cache_write_ttl_requests_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_cache_write_ttl_requests_total{{provider="{provider}",ttl="5m"}} {stats["cache_write_5m_requests"]}'
                    )
                    lines.append(
                        f'headroom_cache_write_ttl_requests_total{{provider="{provider}",ttl="1h"}} {stats["cache_write_1h_requests"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_uncached_input_tokens_total Input tokens not served from provider cache",
                        "# TYPE headroom_uncached_input_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_uncached_input_tokens_total{{provider="{provider}"}} {stats["uncached_input_tokens"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_provider_cache_requests_total Requests with provider cache observations",
                        "# TYPE headroom_provider_cache_requests_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_provider_cache_requests_total{{provider="{provider}"}} {stats["requests"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_provider_cache_hit_requests_total Requests with provider cache reads",
                        "# TYPE headroom_provider_cache_hit_requests_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_provider_cache_hit_requests_total{{provider="{provider}"}} {stats["hit_requests"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_provider_cache_bust_total Provider-specific cache bust count",
                        "# TYPE headroom_provider_cache_bust_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_provider_cache_bust_total{{provider="{provider}"}} {stats["bust_count"]}'
                    )
                lines.extend(
                    [
                        "",
                        "# HELP headroom_provider_cache_bust_write_tokens_total Provider cache write tokens attributed to busts",
                        "# TYPE headroom_provider_cache_bust_write_tokens_total counter",
                    ]
                )
                for provider, stats in self.cache_by_provider.items():
                    lines.append(
                        f'headroom_provider_cache_bust_write_tokens_total{{provider="{provider}"}} {stats["bust_write_tokens"]}'
                    )
                lines.append("")

            return "\n".join(lines)
