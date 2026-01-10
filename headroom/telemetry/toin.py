"""Tool Output Intelligence Network (TOIN) - Cross-user learning for compression.

TOIN aggregates anonymized compression patterns across all Headroom users to
create a network effect: every user's compression decisions improve the
recommendations for everyone.

Key concepts:
- ToolPattern: Aggregated intelligence about a tool type (by structure hash)
- CompressionHint: Recommendations for how to compress a specific tool output
- ToolIntelligenceNetwork: Central aggregator that learns from all users

How it works:
1. When SmartCrusher compresses data, it records the outcome via telemetry
2. When LLM retrieves compressed data, TOIN tracks what was needed
3. TOIN learns: "For tools with structure X, retrieval rate is high when
   compressing field Y - preserve it"
4. Next time: SmartCrusher asks TOIN for hints before compressing

Privacy:
- No actual data values are stored
- Tool names are structure hashes
- Field names are SHA256[:8] hashes
- No user identifiers

Network Effect:
- More users → more compression events → better recommendations
- Cross-user patterns reveal universal tool behaviors
- Federated learning: aggregate patterns, not data

Usage:
    from headroom.telemetry.toin import get_toin

    # Before compression, get recommendations
    hint = get_toin().get_recommendation(tool_signature, query_context)

    # Apply hint
    if hint.skip_compression:
        return original_data
    config.preserve_fields = hint.preserve_fields
    config.max_items = hint.max_items
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .models import ToolSignature

logger = logging.getLogger(__name__)

# LOW FIX #22: Define callback types for metrics/monitoring hooks
# These allow users to plug in their own metrics collection (Prometheus, StatsD, etc.)
MetricsCallback = Callable[[str, dict[str, Any]], None]  # (event_name, event_data) -> None


@dataclass
class ToolPattern:
    """Aggregated intelligence about a tool type across all users.

    This is the core TOIN data structure. It represents everything we've
    learned about how to compress outputs from tools with a specific structure.
    """

    tool_signature_hash: str

    # === Compression Statistics ===
    total_compressions: int = 0
    total_items_seen: int = 0
    total_items_kept: int = 0
    avg_compression_ratio: float = 0.0
    avg_token_reduction: float = 0.0

    # === Retrieval Statistics ===
    total_retrievals: int = 0
    full_retrievals: int = 0  # Retrieved everything
    search_retrievals: int = 0  # Used search filter

    @property
    def retrieval_rate(self) -> float:
        """Fraction of compressions that triggered retrieval."""
        if self.total_compressions == 0:
            return 0.0
        return self.total_retrievals / self.total_compressions

    @property
    def full_retrieval_rate(self) -> float:
        """Fraction of retrievals that were full (not search)."""
        if self.total_retrievals == 0:
            return 0.0
        return self.full_retrievals / self.total_retrievals

    # === Learned Patterns ===
    # Fields that are frequently retrieved (should preserve)
    commonly_retrieved_fields: list[str] = field(default_factory=list)
    field_retrieval_frequency: dict[str, int] = field(default_factory=dict)

    # Query patterns that trigger retrieval
    common_query_patterns: list[str] = field(default_factory=list)
    # MEDIUM FIX #10: Track query pattern frequency to keep most common, not just recent
    query_pattern_frequency: dict[str, int] = field(default_factory=dict)

    # Best compression strategy for this tool type
    optimal_strategy: str = "default"
    strategy_success_rates: dict[str, float] = field(default_factory=dict)

    # === Learned Recommendations ===
    optimal_max_items: int = 20
    skip_compression_recommended: bool = False
    preserve_fields: list[str] = field(default_factory=list)

    # === Confidence ===
    sample_size: int = 0
    user_count: int = 0  # Number of unique users (anonymized)
    confidence: float = 0.0  # 0.0 = no data, 1.0 = high confidence
    last_updated: float = 0.0

    # === Instance Tracking (for user_count) ===
    # Hashed instance IDs of users who have contributed to this pattern
    # Limited to avoid unbounded growth (for serialization)
    _seen_instance_hashes: list[str] = field(default_factory=list)
    # FIX: Separate set for ALL seen instances to prevent double-counting
    # CRITICAL FIX #1: Capped at MAX_SEEN_INSTANCES to prevent OOM with millions of users.
    # When cap is reached, we rely on user_count for accurate counting and
    # accept some potential double-counting for new users (negligible at scale).
    _all_seen_instances: set[str] = field(default_factory=set)

    # CRITICAL FIX: Track whether instance tracking was truncated during serialization
    # If True, we know some users were lost and should be conservative about user_count
    _tracking_truncated: bool = False

    # CRITICAL FIX #1: Maximum entries in _all_seen_instances to prevent OOM
    # This is a class constant, not a field (not serialized)
    MAX_SEEN_INSTANCES: int = 10000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_signature_hash": self.tool_signature_hash,
            "total_compressions": self.total_compressions,
            "total_items_seen": self.total_items_seen,
            "total_items_kept": self.total_items_kept,
            "avg_compression_ratio": self.avg_compression_ratio,
            "avg_token_reduction": self.avg_token_reduction,
            "total_retrievals": self.total_retrievals,
            "full_retrievals": self.full_retrievals,
            "search_retrievals": self.search_retrievals,
            "retrieval_rate": self.retrieval_rate,
            "full_retrieval_rate": self.full_retrieval_rate,
            "commonly_retrieved_fields": self.commonly_retrieved_fields,
            "field_retrieval_frequency": self.field_retrieval_frequency,
            "common_query_patterns": self.common_query_patterns,
            "query_pattern_frequency": self.query_pattern_frequency,
            "optimal_strategy": self.optimal_strategy,
            "strategy_success_rates": self.strategy_success_rates,
            "optimal_max_items": self.optimal_max_items,
            "skip_compression_recommended": self.skip_compression_recommended,
            "preserve_fields": self.preserve_fields,
            "sample_size": self.sample_size,
            "user_count": self.user_count,
            "confidence": self.confidence,
            "last_updated": self.last_updated,
            # Serialize instance hashes (limited to 100 for bounded storage)
            "seen_instance_hashes": self._seen_instance_hashes[:100],
            # CRITICAL FIX: Track if truncation occurred during serialization
            # This tells from_dict() that some users were lost and prevents double-counting
            "tracking_truncated": (
                self._tracking_truncated
                or self.user_count > len(self._seen_instance_hashes)
                or len(self._all_seen_instances) > 100
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolPattern:
        """Create from dictionary."""
        # Filter to only valid fields
        valid_fields = {
            "tool_signature_hash",
            "total_compressions",
            "total_items_seen",
            "total_items_kept",
            "avg_compression_ratio",
            "avg_token_reduction",
            "total_retrievals",
            "full_retrievals",
            "search_retrievals",
            "commonly_retrieved_fields",
            "field_retrieval_frequency",
            "common_query_patterns",
            "query_pattern_frequency",
            "optimal_strategy",
            "strategy_success_rates",
            "optimal_max_items",
            "skip_compression_recommended",
            "preserve_fields",
            "sample_size",
            "user_count",
            "confidence",
            "last_updated",
        }
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        # Handle seen_instance_hashes (serialized without underscore prefix)
        seen_hashes = data.get("seen_instance_hashes", [])

        pattern = cls(**filtered)
        pattern._seen_instance_hashes = seen_hashes[:100]  # Limit on load

        # CRITICAL FIX: Populate _all_seen_instances from loaded hashes
        # This prevents double-counting after restart - without this, the same
        # instances would be counted again because the lookup set was empty
        pattern._all_seen_instances = set(pattern._seen_instance_hashes)

        # CRITICAL FIX: Restore truncation flag to prevent double-counting
        # If truncated, we know some users were lost in serialization
        pattern._tracking_truncated = data.get("tracking_truncated", False)
        # Also detect truncation if user_count > loaded hashes (backward compat)
        if pattern.user_count > len(pattern._seen_instance_hashes):
            pattern._tracking_truncated = True

        return pattern


@dataclass
class CompressionHint:
    """Recommendation for how to compress a specific tool output.

    This is what TOIN returns when asked for advice before compression.
    """

    # Should we compress at all?
    skip_compression: bool = False

    # How aggressively to compress
    max_items: int = 20
    compression_level: Literal["none", "conservative", "moderate", "aggressive"] = "moderate"

    # Which fields to preserve (never remove)
    preserve_fields: list[str] = field(default_factory=list)

    # Which strategy to use
    recommended_strategy: str = "default"

    # Why this recommendation
    reason: str = ""
    confidence: float = 0.0

    # Source of recommendation
    source: Literal["network", "local", "default"] = "default"
    based_on_samples: int = 0


@dataclass
class TOINConfig:
    """Configuration for the Tool Output Intelligence Network."""

    # Enable/disable TOIN
    enabled: bool = True

    # Storage
    storage_path: str | None = None  # Path to store TOIN data
    auto_save_interval: int = 600  # Auto-save every 10 minutes

    # Network learning thresholds
    min_samples_for_recommendation: int = 10
    min_users_for_network_effect: int = 3

    # Recommendation thresholds
    high_retrieval_threshold: float = 0.5  # Above this = compress less
    medium_retrieval_threshold: float = 0.2  # Between medium and high = moderate

    # Privacy
    anonymize_queries: bool = True
    max_query_patterns: int = 10

    # LOW FIX #22: Metrics/monitoring hooks
    # Callback for emitting metrics events. Signature: (event_name, event_data) -> None
    # Event names: "toin.compression", "toin.retrieval", "toin.recommendation", "toin.save"
    # This allows integration with Prometheus, StatsD, OpenTelemetry, etc.
    metrics_callback: MetricsCallback | None = None


class ToolIntelligenceNetwork:
    """Aggregates tool patterns across all Headroom users.

    This is the brain of TOIN. It maintains a database of learned patterns
    for different tool types and provides recommendations based on
    cross-user intelligence.

    Thread-safe for concurrent access.
    """

    def __init__(self, config: TOINConfig | None = None):
        """Initialize TOIN.

        Args:
            config: Configuration options.
        """
        self._config = config or TOINConfig()
        self._lock = threading.RLock()  # RLock for reentrant locking (save calls export_patterns)

        # Pattern database: structure_hash -> ToolPattern
        self._patterns: dict[str, ToolPattern] = {}

        # Instance ID for user counting (anonymized)
        # IMPORTANT: Must be STABLE across restarts to avoid false user count inflation
        # Derive from storage path if available, otherwise use machine-specific ID
        self._instance_id = self._generate_stable_instance_id()

        # Tracking
        self._last_save_time = time.time()
        self._dirty = False

        # Load existing data
        if self._config.storage_path:
            self._load_from_disk()

    def _generate_stable_instance_id(self) -> str:
        """Generate a stable instance ID that doesn't change across restarts.

        Uses storage path if available, otherwise uses machine-specific info.
        This prevents false user count inflation when reloading from disk.

        HIGH FIX: Instance ID collision risk
        Previously used SHA256[:8] (32 bits) which has 50% collision probability
        at sqrt(2^32) ≈ 65,536 users (birthday paradox). Increased to SHA256[:16]
        (64 bits) for 50% collision at ~4 billion users, which is acceptable.
        """
        if self._config.storage_path:
            # Derive from storage path - same path = same instance
            return hashlib.sha256(self._config.storage_path.encode()).hexdigest()[
                :16
            ]  # HIGH FIX: 64 bits instead of 32
        else:
            # No storage - use a combination of hostname and process info
            # This is less stable but better than pure random
            import os
            import socket

            machine_info = (
                f"{socket.gethostname()}:{os.getuid() if hasattr(os, 'getuid') else 'unknown'}"
            )
            return hashlib.sha256(machine_info.encode()).hexdigest()[:16]  # HIGH FIX: 64 bits

    def _emit_metric(self, event_name: str, event_data: dict[str, Any]) -> None:
        """Emit a metrics event via the configured callback.

        LOW FIX #22: Provides monitoring integration for external metrics systems.

        Args:
            event_name: Name of the event (e.g., "toin.compression").
            event_data: Dictionary of event data to emit.
        """
        if self._config.metrics_callback is not None:
            try:
                self._config.metrics_callback(event_name, event_data)
            except Exception as e:
                # Never let metrics callback failures break TOIN
                logger.debug(f"Metrics callback failed for {event_name}: {e}")

    def record_compression(
        self,
        tool_signature: ToolSignature,
        original_count: int,
        compressed_count: int,
        original_tokens: int,
        compressed_tokens: int,
        strategy: str,
        query_context: str | None = None,
    ) -> None:
        """Record a compression event.

        Called after SmartCrusher compresses data. Updates the pattern
        for this tool type.

        Args:
            tool_signature: Signature of the tool output structure.
            original_count: Original number of items.
            compressed_count: Number of items after compression.
            original_tokens: Original token count.
            compressed_tokens: Compressed token count.
            strategy: Compression strategy used.
            query_context: Optional user query that triggered this tool call.
        """
        # HIGH FIX: Check enabled FIRST to avoid computing structure_hash if disabled
        # This saves CPU when TOIN is turned off
        if not self._config.enabled:
            return

        # Computing structure_hash can be expensive for large structures
        sig_hash = tool_signature.structure_hash

        # LOW FIX #22: Emit compression metric
        self._emit_metric(
            "toin.compression",
            {
                "signature_hash": sig_hash,
                "original_count": original_count,
                "compressed_count": compressed_count,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "strategy": strategy,
                "compression_ratio": compressed_count / original_count if original_count > 0 else 0,
            },
        )

        with self._lock:
            # Get or create pattern
            if sig_hash not in self._patterns:
                self._patterns[sig_hash] = ToolPattern(tool_signature_hash=sig_hash)

            pattern = self._patterns[sig_hash]

            # Update compression stats
            pattern.total_compressions += 1
            pattern.total_items_seen += original_count
            pattern.total_items_kept += compressed_count
            pattern.sample_size += 1

            # Update rolling averages
            n = pattern.total_compressions
            compression_ratio = compressed_count / original_count if original_count > 0 else 0.0
            token_reduction = (
                1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0
            )

            pattern.avg_compression_ratio = (
                pattern.avg_compression_ratio * (n - 1) + compression_ratio
            ) / n
            pattern.avg_token_reduction = (
                pattern.avg_token_reduction * (n - 1) + token_reduction
            ) / n

            # Update strategy stats
            if strategy not in pattern.strategy_success_rates:
                pattern.strategy_success_rates[strategy] = 1.0  # Start optimistic
            else:
                # Give a small boost for each compression without retrieval
                # This counteracts the penalty from record_retrieval() and prevents
                # all strategies from trending to 0.0 over time (one-way ratchet fix)
                # The boost is small (0.02) because retrieval penalties are larger (0.05-0.15)
                # This means strategies that cause retrievals will still trend down
                current_rate = pattern.strategy_success_rates[strategy]
                pattern.strategy_success_rates[strategy] = min(1.0, current_rate + 0.02)

            # HIGH FIX: Bound strategy_success_rates to prevent unbounded growth
            # Keep top 20 strategies by success rate
            if len(pattern.strategy_success_rates) > 20:
                sorted_strategies = sorted(
                    pattern.strategy_success_rates.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]
                pattern.strategy_success_rates = dict(sorted_strategies)

            # Track unique users via instance_id
            # FIX: Use _all_seen_instances set for lookup to prevent double-counting
            # after the storage list hits its cap
            # CRITICAL FIX #1: Check cap before adding to prevent OOM
            if self._instance_id not in pattern._all_seen_instances:
                # CRITICAL FIX: Check if we can verify this is a new user
                # If tracking was truncated (users lost after restart), we can only
                # count new users if we can add them to _all_seen_instances for dedup
                can_track = len(pattern._all_seen_instances) < ToolPattern.MAX_SEEN_INSTANCES

                if can_track:
                    # Add to the lookup set - we can verify this is new
                    pattern._all_seen_instances.add(self._instance_id)
                    # Also add to storage list (capped at 100 for serialization)
                    if len(pattern._seen_instance_hashes) < 100:
                        pattern._seen_instance_hashes.append(self._instance_id)
                    # Safe to increment user_count - we verified it's new
                    pattern.user_count += 1
                elif not pattern._tracking_truncated:
                    # Tracking set is full but we weren't truncated before
                    # This is a truly new user beyond our tracking capacity
                    pattern.user_count += 1
                # else: Can't verify if new, skip incrementing to prevent double-count

            # Track query context patterns for learning (privacy-preserving)
            if query_context and len(query_context) >= 3:
                # Normalize and anonymize: extract keywords, remove values
                query_pattern = self._anonymize_query_pattern(query_context)
                if query_pattern:
                    # MEDIUM FIX #10: Track frequency to keep most common patterns
                    pattern.query_pattern_frequency[query_pattern] = (
                        pattern.query_pattern_frequency.get(query_pattern, 0) + 1
                    )
                    # Update the list to contain top patterns by frequency
                    if query_pattern not in pattern.common_query_patterns:
                        pattern.common_query_patterns.append(query_pattern)
                    # Keep only the most common patterns (by frequency)
                    if len(pattern.common_query_patterns) > self._config.max_query_patterns:
                        pattern.common_query_patterns = sorted(
                            pattern.common_query_patterns,
                            key=lambda p: pattern.query_pattern_frequency.get(p, 0),
                            reverse=True,
                        )[: self._config.max_query_patterns]
                    # Also limit the frequency dict
                    if len(pattern.query_pattern_frequency) > self._config.max_query_patterns * 2:
                        top_patterns = sorted(
                            pattern.query_pattern_frequency.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[: self._config.max_query_patterns * 2]
                        pattern.query_pattern_frequency = dict(top_patterns)

            # Periodically update recommendations even without retrievals
            # This ensures optimal_strategy is updated based on success rates
            if pattern.total_compressions % 10 == 0:
                self._update_recommendations(pattern)

            pattern.last_updated = time.time()
            pattern.confidence = self._calculate_confidence(pattern)
            self._dirty = True

        # Auto-save if needed (outside lock)
        self._maybe_auto_save()

    def record_retrieval(
        self,
        tool_signature_hash: str,
        retrieval_type: str,
        query: str | None = None,
        query_fields: list[str] | None = None,
        strategy: str | None = None,
    ) -> None:
        """Record a retrieval event.

        Called when LLM retrieves compressed content. This is the key
        feedback signal - it means compression was too aggressive.

        Args:
            tool_signature_hash: Hash of the tool signature.
            retrieval_type: "full" or "search".
            query: Optional search query (will be anonymized).
            query_fields: Fields mentioned in query (will be hashed).
            strategy: Compression strategy that was used (for success rate tracking).
        """
        if not self._config.enabled:
            return

        # LOW FIX #22: Emit retrieval metric
        self._emit_metric(
            "toin.retrieval",
            {
                "signature_hash": tool_signature_hash,
                "retrieval_type": retrieval_type,
                "has_query": query is not None,
                "query_fields_count": len(query_fields) if query_fields else 0,
                "strategy": strategy,
            },
        )

        with self._lock:
            if tool_signature_hash not in self._patterns:
                # First time seeing this tool via retrieval
                self._patterns[tool_signature_hash] = ToolPattern(
                    tool_signature_hash=tool_signature_hash
                )

            pattern = self._patterns[tool_signature_hash]

            # Update retrieval stats
            pattern.total_retrievals += 1
            if retrieval_type == "full":
                pattern.full_retrievals += 1
            else:
                pattern.search_retrievals += 1

            # Update strategy success rates - retrieval means the strategy was TOO aggressive
            # Decrease success rate for this strategy
            if strategy and strategy in pattern.strategy_success_rates:
                # Exponential moving average: penalize strategies that trigger retrieval
                # Full retrievals are worse than search retrievals
                penalty = 0.15 if retrieval_type == "full" else 0.05
                current_rate = pattern.strategy_success_rates[strategy]
                pattern.strategy_success_rates[strategy] = max(0.0, current_rate - penalty)

            # Track queried fields (anonymized)
            if query_fields:
                for field_name in query_fields:
                    field_hash = self._hash_field_name(field_name)
                    pattern.field_retrieval_frequency[field_hash] = (
                        pattern.field_retrieval_frequency.get(field_hash, 0) + 1
                    )

                    # Update commonly retrieved fields
                    if field_hash not in pattern.commonly_retrieved_fields:
                        # Add if frequently retrieved (check count from dict)
                        freq = pattern.field_retrieval_frequency.get(field_hash, 0)
                        if freq >= 3:
                            pattern.commonly_retrieved_fields.append(field_hash)
                            # HIGH: Limit commonly_retrieved_fields to prevent unbounded growth
                            if len(pattern.commonly_retrieved_fields) > 20:
                                # Keep only the most frequently retrieved fields
                                sorted_fields = sorted(
                                    pattern.commonly_retrieved_fields,
                                    key=lambda f: pattern.field_retrieval_frequency.get(f, 0),
                                    reverse=True,
                                )
                                pattern.commonly_retrieved_fields = sorted_fields[:20]

                # HIGH: Limit field_retrieval_frequency dict to prevent unbounded growth
                if len(pattern.field_retrieval_frequency) > 100:
                    sorted_fields = sorted(
                        pattern.field_retrieval_frequency.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:100]
                    pattern.field_retrieval_frequency = dict(sorted_fields)

            # Track query patterns (anonymized)
            if query and self._config.anonymize_queries:
                query_pattern = self._anonymize_query_pattern(query)
                if query_pattern:
                    # MEDIUM FIX #10: Track frequency to keep most common patterns
                    pattern.query_pattern_frequency[query_pattern] = (
                        pattern.query_pattern_frequency.get(query_pattern, 0) + 1
                    )
                    if query_pattern not in pattern.common_query_patterns:
                        pattern.common_query_patterns.append(query_pattern)
                    # Keep only the most common patterns (by frequency)
                    if len(pattern.common_query_patterns) > self._config.max_query_patterns:
                        pattern.common_query_patterns = sorted(
                            pattern.common_query_patterns,
                            key=lambda p: pattern.query_pattern_frequency.get(p, 0),
                            reverse=True,
                        )[: self._config.max_query_patterns]

            # Update recommendations based on new retrieval data
            self._update_recommendations(pattern)

            pattern.last_updated = time.time()
            self._dirty = True

        self._maybe_auto_save()

    def get_recommendation(
        self,
        tool_signature: ToolSignature,
        query_context: str | None = None,
    ) -> CompressionHint:
        """Get compression recommendation for a tool output.

        This is the main API for SmartCrusher to consult before compressing.

        Args:
            tool_signature: Signature of the tool output structure.
            query_context: User query for context-aware recommendations.

        Returns:
            CompressionHint with recommendations.
        """
        if not self._config.enabled:
            return CompressionHint(source="default", reason="TOIN disabled")

        sig_hash = tool_signature.structure_hash

        with self._lock:
            pattern = self._patterns.get(sig_hash)

            if pattern is None:
                # No data for this tool type
                return CompressionHint(
                    source="default",
                    reason="No pattern data for this tool type",
                )

            # Not enough samples for reliable recommendation
            if pattern.sample_size < self._config.min_samples_for_recommendation:
                hint = CompressionHint(
                    source="local",
                    reason=f"Only {pattern.sample_size} samples (need {self._config.min_samples_for_recommendation})",
                    confidence=pattern.confidence,
                    based_on_samples=pattern.sample_size,
                )
                # LOW FIX #22: Emit recommendation metric
                self._emit_metric(
                    "toin.recommendation",
                    {
                        "signature_hash": sig_hash,
                        "source": hint.source,
                        "confidence": hint.confidence,
                        "skip_compression": hint.skip_compression,
                        "max_items": hint.max_items,
                        "compression_level": hint.compression_level,
                        "based_on_samples": hint.based_on_samples,
                    },
                )
                return hint

            # Build recommendation based on learned patterns
            hint = self._build_recommendation(pattern, query_context)

            # LOW FIX #22: Emit recommendation metric
            self._emit_metric(
                "toin.recommendation",
                {
                    "signature_hash": sig_hash,
                    "source": hint.source,
                    "confidence": hint.confidence,
                    "skip_compression": hint.skip_compression,
                    "max_items": hint.max_items,
                    "compression_level": hint.compression_level,
                    "based_on_samples": hint.based_on_samples,
                },
            )
            return hint

    def _build_recommendation(
        self,
        pattern: ToolPattern,
        query_context: str | None,
    ) -> CompressionHint:
        """Build a recommendation based on pattern data and query context."""
        hint = CompressionHint(
            source="network"
            if pattern.user_count >= self._config.min_users_for_network_effect
            else "local",
            confidence=pattern.confidence,
            based_on_samples=pattern.sample_size,
        )

        retrieval_rate = pattern.retrieval_rate
        full_retrieval_rate = pattern.full_retrieval_rate

        # High retrieval rate = compression too aggressive
        if retrieval_rate > self._config.high_retrieval_threshold:
            if full_retrieval_rate > 0.8:
                # Almost all retrievals are full = don't compress
                hint.skip_compression = True
                hint.compression_level = "none"
                hint.reason = f"Very high full retrieval rate ({full_retrieval_rate:.1%})"
            else:
                # High retrieval but mostly search = compress conservatively
                hint.max_items = pattern.optimal_max_items
                hint.compression_level = "conservative"
                hint.reason = f"High retrieval rate ({retrieval_rate:.1%})"

        elif retrieval_rate > self._config.medium_retrieval_threshold:
            # Moderate retrieval = moderate compression
            hint.max_items = max(20, pattern.optimal_max_items)
            hint.compression_level = "moderate"
            hint.reason = f"Moderate retrieval rate ({retrieval_rate:.1%})"

        else:
            # Low retrieval = aggressive compression works
            hint.max_items = min(15, pattern.optimal_max_items)
            hint.compression_level = "aggressive"
            hint.reason = f"Low retrieval rate ({retrieval_rate:.1%})"

        # Build preserve_fields list weighted by retrieval frequency
        # Start with pattern's preserve_fields, then enhance based on query
        preserve_fields = pattern.preserve_fields.copy()
        query_fields_count = 0

        # If we have query context, extract field names and prioritize them
        if query_context and pattern.field_retrieval_frequency:
            # Extract field names from query context
            import re

            query_field_names = re.findall(r"(\w+)[=:]", query_context.lower())

            # Hash them and check if they're in our frequency data
            for field_name in query_field_names:
                field_hash = self._hash_field_name(field_name)
                if field_hash in pattern.field_retrieval_frequency:
                    # This field is known to be retrieved - prioritize it
                    if field_hash in preserve_fields:
                        # Move to front
                        preserve_fields.remove(field_hash)
                    preserve_fields.insert(0, field_hash)
                    query_fields_count += 1

        # Sort remaining fields by retrieval frequency (most frequent first)
        if pattern.field_retrieval_frequency and len(preserve_fields) > 1:
            # Separate query-mentioned fields (already at front) from others
            if query_fields_count < len(preserve_fields):
                rest = preserve_fields[query_fields_count:]
                rest.sort(
                    key=lambda f: pattern.field_retrieval_frequency.get(f, 0),
                    reverse=True,
                )
                preserve_fields = preserve_fields[:query_fields_count] + rest

        hint.preserve_fields = preserve_fields[:10]  # Limit to top 10

        # Use optimal strategy if known AND it has good success rate
        if pattern.optimal_strategy != "default":
            success_rate = pattern.strategy_success_rates.get(pattern.optimal_strategy, 1.0)
            # Only recommend strategy if success rate >= 0.5
            # Lower success rates mean this strategy often causes retrievals
            if success_rate >= 0.5:
                hint.recommended_strategy = pattern.optimal_strategy
            else:
                # Strategy has poor success rate - reduce confidence
                hint.confidence *= success_rate
                hint.reason += (
                    f" (strategy {pattern.optimal_strategy} has low success: {success_rate:.1%})"
                )
                # Try to find a better strategy
                best_strategy = self._find_best_strategy(pattern)
                if best_strategy and best_strategy != pattern.optimal_strategy:
                    hint.recommended_strategy = best_strategy
                    hint.reason += f", using {best_strategy} instead"

        # Boost max_items if query_context matches common retrieval patterns
        # This prevents unnecessary retrieval when we can predict what's needed
        if query_context:
            query_lower = query_context.lower()

            # Check for exhaustive query keywords that suggest user needs all data
            exhaustive_keywords = ["all", "every", "complete", "full", "entire", "list all"]
            if any(kw in query_lower for kw in exhaustive_keywords):
                # User likely needs more data - be conservative
                hint.max_items = max(hint.max_items, 40)
                hint.compression_level = "conservative"
                hint.reason += " (exhaustive query detected)"

            # Check against common retrieval patterns
            if pattern.common_query_patterns:
                query_pattern = self._anonymize_query_pattern(query_context)
                if query_pattern:
                    # Exact match
                    if query_pattern in pattern.common_query_patterns:
                        hint.max_items = max(hint.max_items, 30)
                        hint.reason += " (query matches retrieval pattern)"
                    else:
                        # Partial match: check if any stored pattern is contained in query
                        for stored_pattern in pattern.common_query_patterns:
                            # Check if key fields match (e.g., "status:*" in both)
                            stored_fields = {
                                f.split(":")[0] for f in stored_pattern.split() if ":" in f
                            }
                            query_fields = {
                                f.split(":")[0] for f in query_pattern.split() if ":" in f
                            }
                            # If query uses same fields as a problematic pattern, be conservative
                            if stored_fields and stored_fields.issubset(query_fields):
                                hint.max_items = max(hint.max_items, 25)
                                hint.reason += " (query uses fields from retrieval pattern)"
                                break

        return hint

    def _find_best_strategy(self, pattern: ToolPattern) -> str | None:
        """Find the strategy with the best success rate.

        Returns None if no strategies have been tried or all have low success.
        """
        if not pattern.strategy_success_rates:
            return None

        # Find strategy with highest success rate above threshold
        best_strategy = None
        best_rate = 0.5  # Minimum acceptable rate

        for strategy, rate in pattern.strategy_success_rates.items():
            if rate > best_rate:
                best_rate = rate
                best_strategy = strategy

        return best_strategy

    def _update_recommendations(self, pattern: ToolPattern) -> None:
        """Update learned recommendations for a pattern."""
        # Calculate optimal max_items based on retrieval rate
        retrieval_rate = pattern.retrieval_rate

        if retrieval_rate > self._config.high_retrieval_threshold:
            if pattern.full_retrieval_rate > 0.8:
                pattern.skip_compression_recommended = True
                pattern.optimal_max_items = pattern.total_items_seen // max(
                    1, pattern.total_compressions
                )
            else:
                pattern.optimal_max_items = 50
        elif retrieval_rate > self._config.medium_retrieval_threshold:
            pattern.optimal_max_items = 30
        else:
            pattern.optimal_max_items = 20

        # Update preserve_fields from frequently retrieved fields
        if pattern.field_retrieval_frequency:
            # Get top 5 most retrieved fields
            sorted_fields = sorted(
                pattern.field_retrieval_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            pattern.preserve_fields = [f for f, _ in sorted_fields]

        # Update optimal strategy (pick most successful)
        if pattern.strategy_success_rates:
            best_strategy = max(
                pattern.strategy_success_rates.items(),
                key=lambda x: x[1],
            )[0]
            pattern.optimal_strategy = best_strategy

    def _calculate_confidence(self, pattern: ToolPattern) -> float:
        """Calculate confidence level for a pattern."""
        # Base confidence on sample size
        sample_confidence = min(0.7, pattern.sample_size / 100)

        # Boost if from multiple users
        # FIX: Changed from `user_count / 10 * 0.1` (= user_count * 0.01, too small)
        # to `user_count * 0.03` for meaningful boost at low user counts
        # - 3 users: 0.09 boost
        # - 10 users: 0.30 boost (capped)
        user_boost = 0.0
        if pattern.user_count >= self._config.min_users_for_network_effect:
            user_boost = min(0.3, pattern.user_count * 0.03)

        return min(0.95, sample_confidence + user_boost)

    def _hash_field_name(self, field_name: str) -> str:
        """Hash a field name for anonymization."""
        return hashlib.sha256(field_name.encode()).hexdigest()[:8]

    def _anonymize_query_pattern(self, query: str) -> str | None:
        """Extract anonymized pattern from a query.

        Keeps structural patterns, removes specific values.
        E.g., "status:error AND user:john" -> "status:* AND user:*"
        """
        if not query:
            return None

        # Simple pattern extraction: replace values after : or =
        import re

        # Match field:value or field="value" patterns, but don't include spaces in unquoted values
        pattern = re.sub(r'(\w+)[=:](?:"[^"]*"|\'[^\']*\'|\w+)', r"\1:*", query)

        # Remove if it's just generic
        if pattern in ("*", ""):
            return None

        return pattern

    def get_stats(self) -> dict[str, Any]:
        """Get overall TOIN statistics."""
        with self._lock:
            total_compressions = sum(p.total_compressions for p in self._patterns.values())
            total_retrievals = sum(p.total_retrievals for p in self._patterns.values())

            return {
                "enabled": self._config.enabled,
                "patterns_tracked": len(self._patterns),
                "total_compressions": total_compressions,
                "total_retrievals": total_retrievals,
                "global_retrieval_rate": (
                    total_retrievals / total_compressions if total_compressions > 0 else 0.0
                ),
                "patterns_with_recommendations": sum(
                    1
                    for p in self._patterns.values()
                    if p.sample_size >= self._config.min_samples_for_recommendation
                ),
            }

    def get_pattern(self, signature_hash: str) -> ToolPattern | None:
        """Get pattern data for a specific tool signature.

        HIGH FIX: Returns a deep copy to prevent external mutation of internal state.
        """
        import copy

        with self._lock:
            pattern = self._patterns.get(signature_hash)
            if pattern is not None:
                return copy.deepcopy(pattern)
            return None

    def export_patterns(self) -> dict[str, Any]:
        """Export all patterns for sharing/aggregation."""
        with self._lock:
            return {
                "version": "1.0",
                "export_timestamp": time.time(),
                "instance_id": self._instance_id,
                "patterns": {
                    sig_hash: pattern.to_dict() for sig_hash, pattern in self._patterns.items()
                },
            }

    def import_patterns(self, data: dict[str, Any]) -> None:
        """Import patterns from another source.

        Used for federated learning: aggregate patterns from multiple
        Headroom instances without sharing actual data.

        Args:
            data: Exported pattern data.
        """
        if not self._config.enabled:
            return

        patterns_data = data.get("patterns", {})
        source_instance = data.get("instance_id", "unknown")

        with self._lock:
            for sig_hash, pattern_dict in patterns_data.items():
                imported = ToolPattern.from_dict(pattern_dict)

                if sig_hash in self._patterns:
                    # Merge with existing
                    self._merge_patterns(self._patterns[sig_hash], imported)
                else:
                    # Add new pattern - need to track source instance
                    self._patterns[sig_hash] = imported

                    # For NEW patterns from another instance, track the source in
                    # _seen_instance_hashes so user_count reflects cross-user data
                    if source_instance != self._instance_id:
                        pattern = self._patterns[sig_hash]
                        if source_instance not in pattern._seen_instance_hashes:
                            # Limit storage to 100 unique instances to bound memory
                            if len(pattern._seen_instance_hashes) < 100:
                                pattern._seen_instance_hashes.append(source_instance)
                            # CRITICAL: Always increment user_count (even after cap)
                            pattern.user_count += 1

            self._dirty = True

    def _merge_patterns(self, existing: ToolPattern, imported: ToolPattern) -> None:
        """Merge imported pattern into existing."""
        total = existing.sample_size + imported.sample_size
        if total == 0:
            return

        w_existing = existing.sample_size / total
        w_imported = imported.sample_size / total

        # Merge counts
        existing.total_compressions += imported.total_compressions
        existing.total_retrievals += imported.total_retrievals
        existing.full_retrievals += imported.full_retrievals
        existing.search_retrievals += imported.search_retrievals
        existing.total_items_seen += imported.total_items_seen
        existing.total_items_kept += imported.total_items_kept

        # Weighted averages
        existing.avg_compression_ratio = (
            existing.avg_compression_ratio * w_existing
            + imported.avg_compression_ratio * w_imported
        )
        existing.avg_token_reduction = (
            existing.avg_token_reduction * w_existing + imported.avg_token_reduction * w_imported
        )

        # Merge field frequencies
        for field_hash, count in imported.field_retrieval_frequency.items():
            existing.field_retrieval_frequency[field_hash] = (
                existing.field_retrieval_frequency.get(field_hash, 0) + count
            )
        # HIGH: Limit field_retrieval_frequency dict to prevent unbounded growth
        if len(existing.field_retrieval_frequency) > 100:
            # Keep only the most frequently retrieved fields
            sorted_fields = sorted(
                existing.field_retrieval_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:100]
            existing.field_retrieval_frequency = dict(sorted_fields)

        # Merge commonly retrieved fields
        for field_hash in imported.commonly_retrieved_fields:
            if field_hash not in existing.commonly_retrieved_fields:
                existing.commonly_retrieved_fields.append(field_hash)
        # HIGH: Limit commonly_retrieved_fields to prevent unbounded growth
        if len(existing.commonly_retrieved_fields) > 20:
            # Prioritize by retrieval frequency if available
            if existing.field_retrieval_frequency:
                existing.commonly_retrieved_fields = sorted(
                    existing.commonly_retrieved_fields,
                    key=lambda f: existing.field_retrieval_frequency.get(f, 0),
                    reverse=True,
                )[:20]
            else:
                existing.commonly_retrieved_fields = existing.commonly_retrieved_fields[:20]

        # Merge query patterns (for federated learning)
        # MEDIUM FIX #10: Also merge query_pattern_frequency for proper ranking
        for query_pattern, freq in imported.query_pattern_frequency.items():
            existing.query_pattern_frequency[query_pattern] = (
                existing.query_pattern_frequency.get(query_pattern, 0) + freq
            )
        for query_pattern in imported.common_query_patterns:
            if query_pattern not in existing.common_query_patterns:
                existing.common_query_patterns.append(query_pattern)
        # Keep only the most common patterns (by frequency)
        if len(existing.common_query_patterns) > self._config.max_query_patterns:
            existing.common_query_patterns = sorted(
                existing.common_query_patterns,
                key=lambda p: existing.query_pattern_frequency.get(p, 0),
                reverse=True,
            )[: self._config.max_query_patterns]
        # Limit frequency dict
        if len(existing.query_pattern_frequency) > self._config.max_query_patterns * 2:
            top_patterns = sorted(
                existing.query_pattern_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[: self._config.max_query_patterns * 2]
            existing.query_pattern_frequency = dict(top_patterns)

        # Merge strategy success rates (weighted average)
        for strategy, rate in imported.strategy_success_rates.items():
            if strategy in existing.strategy_success_rates:
                existing.strategy_success_rates[strategy] = (
                    existing.strategy_success_rates[strategy] * w_existing + rate * w_imported
                )
            else:
                existing.strategy_success_rates[strategy] = rate

        # HIGH FIX: Bound strategy_success_rates after merge
        if len(existing.strategy_success_rates) > 20:
            sorted_strategies = sorted(
                existing.strategy_success_rates.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:20]
            existing.strategy_success_rates = dict(sorted_strategies)

        # Merge preserve_fields (union of both, deduplicated)
        for preserve_field in imported.preserve_fields:
            if preserve_field not in existing.preserve_fields:
                existing.preserve_fields.append(preserve_field)
        # Keep only top 10 most important fields
        if len(existing.preserve_fields) > 10:
            # Prioritize by retrieval frequency if available
            if existing.field_retrieval_frequency:
                existing.preserve_fields = sorted(
                    existing.preserve_fields,
                    key=lambda f: existing.field_retrieval_frequency.get(f, 0),
                    reverse=True,
                )[:10]
            else:
                existing.preserve_fields = existing.preserve_fields[:10]

        # Merge skip_compression_recommended (true if either recommends skip)
        if imported.skip_compression_recommended:
            # Imported has more data suggesting skip - consider it
            if imported.sample_size > existing.sample_size // 2:
                existing.skip_compression_recommended = True

        # Merge optimal_strategy (prefer the one with better success rate)
        if imported.optimal_strategy != "default":
            imported_rate = imported.strategy_success_rates.get(imported.optimal_strategy, 0.5)
            existing_rate = (
                existing.strategy_success_rates.get(existing.optimal_strategy, 0.5)
                if existing.optimal_strategy != "default"
                else 0.0
            )

            if imported_rate > existing_rate:
                existing.optimal_strategy = imported.optimal_strategy

        # Merge optimal_max_items (weighted average with bounds)
        if imported.optimal_max_items > 0:
            merged_max_items = int(
                existing.optimal_max_items * w_existing + imported.optimal_max_items * w_imported
            )
            # Ensure valid bounds: min 3 items, max 1000 items
            existing.optimal_max_items = max(3, min(1000, merged_max_items))

        existing.sample_size = total

        # Merge seen instance hashes (union of both, limited to 100 for storage)
        # CRITICAL FIX #1 & #3: Simplified user count merge logic with cap enforcement.
        # user_count is the authoritative count even when sets hit their caps.
        new_users_found = 0
        for instance_hash in imported._seen_instance_hashes:
            # Use _all_seen_instances for deduplication (the authoritative set)
            if instance_hash not in existing._all_seen_instances:
                # Add to lookup set (with cap to prevent OOM)
                if len(existing._all_seen_instances) < ToolPattern.MAX_SEEN_INSTANCES:
                    existing._all_seen_instances.add(instance_hash)
                # Limit storage list to 100 unique instances to bound serialization
                if len(existing._seen_instance_hashes) < 100:
                    existing._seen_instance_hashes.append(instance_hash)
                new_users_found += 1

        # Also merge instances from imported._all_seen_instances that weren't in list
        # (in case imported had more than 100 instances)
        for instance_hash in imported._all_seen_instances:
            if instance_hash not in existing._all_seen_instances:
                # Add with cap check
                if len(existing._all_seen_instances) < ToolPattern.MAX_SEEN_INSTANCES:
                    existing._all_seen_instances.add(instance_hash)
                # Storage list already at limit, just track for dedup
                new_users_found += 1

        # CRITICAL FIX #3: Simplified user count calculation.
        # We count new users from both the list and set, then add any users
        # that imported had beyond what we could deduplicate (when both hit caps).
        # imported.user_count may be > len(imported._all_seen_instances) if they hit cap
        users_beyond_imported_tracking = max(
            0, imported.user_count - len(imported._all_seen_instances)
        )
        existing.user_count += new_users_found + users_beyond_imported_tracking

        existing.last_updated = time.time()

        # Recalculate recommendations based on merged data
        self._update_recommendations(existing)

    def save(self) -> None:
        """Save TOIN data to disk with atomic write.

        Uses a temporary file and rename to ensure atomicity.
        If the write fails, the original file is preserved.

        HIGH FIX: Serialize under lock but write outside lock to prevent
        blocking other threads during slow file I/O.
        """
        if not self._config.storage_path:
            return

        import tempfile

        # Step 1: Serialize under lock (fast in-memory operation)
        with self._lock:
            data = self.export_patterns()

        # Step 2: Write outside lock (slow I/O operation)
        path = Path(self._config.storage_path)

        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize to string (outside lock but before file ops)
            json_data = json.dumps(data, indent=2)

            # Write to temporary file first (atomic write pattern)
            # Use same directory to ensure same filesystem for rename
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".toin_", suffix=".tmp")
            try:
                with open(fd, "w") as f:
                    f.write(json_data)

                # Atomic rename (on POSIX systems)
                Path(tmp_path).replace(path)

            except Exception:
                # Clean up temp file on failure
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass
                raise

            # Step 3: Update state under lock (fast)
            with self._lock:
                self._dirty = False
                self._last_save_time = time.time()

        except OSError as e:
            # Log error but don't crash - TOIN should be resilient
            logger.warning(f"Failed to save TOIN data: {e}")

    def _load_from_disk(self) -> None:
        """Load TOIN data from disk."""
        if not self._config.storage_path:
            return

        path = Path(self._config.storage_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)
            self.import_patterns(data)
            self._dirty = False
        except (json.JSONDecodeError, OSError):
            pass  # Start fresh if corrupted

    def _maybe_auto_save(self) -> None:
        """Auto-save if enough time has passed.

        HIGH FIX: Check conditions under lock to prevent race where another
        thread modifies _dirty or _last_save_time between check and save.
        The save() method already acquires the lock, and we use RLock so
        it's safe to hold the lock when calling save().
        """
        if not self._config.storage_path or not self._config.auto_save_interval:
            return

        # Check under lock to prevent race conditions
        with self._lock:
            if not self._dirty:
                return

            elapsed = time.time() - self._last_save_time
            if elapsed >= self._config.auto_save_interval:
                # save() uses the same RLock, so this is safe
                self.save()

    def clear(self) -> None:
        """Clear all TOIN data. Mainly for testing."""
        with self._lock:
            self._patterns.clear()
            self._dirty = False


# Global TOIN instance (lazy initialization)
_toin_instance: ToolIntelligenceNetwork | None = None
_toin_lock = threading.Lock()


def get_toin(config: TOINConfig | None = None) -> ToolIntelligenceNetwork:
    """Get the global TOIN instance.

    Thread-safe singleton pattern. Always acquires lock to avoid subtle
    race conditions in double-checked locking on non-CPython implementations.

    Args:
        config: Configuration (only used on first call). If the instance
            already exists, config is ignored and a warning is logged.

    Returns:
        Global ToolIntelligenceNetwork instance.
    """
    global _toin_instance

    # CRITICAL FIX: Always acquire lock for thread safety across all Python
    # implementations. The overhead is negligible since we only construct once.
    with _toin_lock:
        if _toin_instance is None:
            _toin_instance = ToolIntelligenceNetwork(config)
        elif config is not None:
            # Warn when config is silently ignored
            logger.warning(
                "TOIN config ignored: instance already exists. "
                "Call reset_toin() first if you need to change config."
            )

    return _toin_instance


def reset_toin() -> None:
    """Reset the global TOIN instance. Mainly for testing."""
    global _toin_instance

    with _toin_lock:
        if _toin_instance is not None:
            _toin_instance.clear()
        _toin_instance = None
