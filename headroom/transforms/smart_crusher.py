"""Smart statistical tool output compression for Headroom SDK.

This module provides intelligent JSON compression based on statistical analysis
rather than fixed rules. It analyzes data patterns and applies optimal compression
strategies to maximize token reduction while preserving important information.

SCHEMA-PRESERVING: Output contains only items from the original array.
No wrappers, no generated text, no metadata keys. This ensures downstream
tools and parsers work unchanged.

Safe V1 Compression Recipe - Always keeps:
- First K items (default 3)
- Last K items (default 2)
- Error items (containing 'error', 'exception', 'failed', 'critical')
- Anomalous numeric items (> 2 std from mean)
- Items around detected change points
- Top-K by score if score field present
- Items with high relevance score to user query (via RelevanceScorer)

Key Features:
- RelevanceScorer: ML-powered or BM25-based relevance matching (replaces regex)
- Variance-based change point detection (preserve anomalies)
- Error item detection (never lose error messages)
- Pattern detection (time series, logs, search results)
- Strategy selection based on data characteristics
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..config import RelevanceScorerConfig, TransformResult
from ..relevance import BM25Scorer, RelevanceScorer, create_scorer


# Legacy patterns for backwards compatibility (extract_query_anchors)
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_NUMERIC_ID_PATTERN = re.compile(r"\b\d{4,}\b")  # 4+ digit numbers (likely IDs)
_HOSTNAME_PATTERN = re.compile(
    r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z]{2,})?\b"
)
_QUOTED_STRING_PATTERN = re.compile(r"['\"]([^'\"]{1,50})['\"]")  # Short quoted strings
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


def extract_query_anchors(text: str) -> set[str]:
    """Extract query anchors from user text (legacy regex-based method).

    DEPRECATED: Use RelevanceScorer.score_batch() for better semantic matching.

    Query anchors are identifiers or values that the user is likely searching for.
    When crushing tool outputs, items matching these anchors should be preserved.

    Extracts:
    - UUIDs (e.g., "550e8400-e29b-41d4-a716-446655440000")
    - Numeric IDs (4+ digits, e.g., "12345", "1001234")
    - Hostnames (e.g., "api.example.com", "server-01.prod")
    - Quoted strings (e.g., 'Alice', "error_code")
    - Email addresses (e.g., "user@example.com")

    Args:
        text: User message text to extract anchors from.

    Returns:
        Set of anchor strings (lowercased for case-insensitive matching).
    """
    anchors: set[str] = set()

    if not text:
        return anchors

    # UUIDs
    for match in _UUID_PATTERN.findall(text):
        anchors.add(match.lower())

    # Numeric IDs
    for match in _NUMERIC_ID_PATTERN.findall(text):
        anchors.add(match)

    # Hostnames
    for match in _HOSTNAME_PATTERN.findall(text):
        # Filter out common false positives
        if match.lower() not in ("e.g", "i.e", "etc."):
            anchors.add(match.lower())

    # Quoted strings
    for match in _QUOTED_STRING_PATTERN.findall(text):
        if len(match.strip()) >= 2:  # Skip very short matches
            anchors.add(match.lower())

    # Email addresses
    for match in _EMAIL_PATTERN.findall(text):
        anchors.add(match.lower())

    return anchors


def item_matches_anchors(item: dict, anchors: set[str]) -> bool:
    """Check if an item matches any query anchors (legacy method).

    DEPRECATED: Use RelevanceScorer for better matching.

    Args:
        item: Dictionary item from tool output.
        anchors: Set of anchor strings to match.

    Returns:
        True if any anchor is found in the item's string representation.
    """
    if not anchors:
        return False

    item_str = str(item).lower()
    return any(anchor in item_str for anchor in anchors)
from ..tokenizer import Tokenizer
from ..utils import (
    compute_short_hash,
    create_tool_digest_marker,
    deep_copy_messages,
    safe_json_dumps,
    safe_json_loads,
)
from .base import Transform


class CompressionStrategy(Enum):
    """Compression strategies based on data patterns."""
    NONE = "none"                    # No compression needed
    TIME_SERIES = "time_series"      # Keep change points, summarize stable
    CLUSTER_SAMPLE = "cluster"       # Dedupe similar items
    TOP_N = "top_n"                  # Keep highest scored items
    SMART_SAMPLE = "smart_sample"    # Statistical sampling with constants


@dataclass
class FieldStats:
    """Statistics for a single field across array items."""
    name: str
    field_type: str  # "numeric", "string", "boolean", "object", "array", "null"
    count: int
    unique_count: int
    unique_ratio: float
    is_constant: bool
    constant_value: Any = None

    # Numeric-specific stats
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    variance: float | None = None
    change_points: list[int] = field(default_factory=list)

    # String-specific stats
    avg_length: float | None = None
    top_values: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class ArrayAnalysis:
    """Complete analysis of an array."""
    item_count: int
    field_stats: dict[str, FieldStats]
    detected_pattern: str  # "time_series", "logs", "search_results", "generic"
    recommended_strategy: CompressionStrategy
    constant_fields: dict[str, Any]
    estimated_reduction: float


@dataclass
class CompressionPlan:
    """Plan for how to compress an array."""
    strategy: CompressionStrategy
    keep_indices: list[int] = field(default_factory=list)
    constant_fields: dict[str, Any] = field(default_factory=dict)
    summary_ranges: list[tuple[int, int, dict]] = field(default_factory=list)
    cluster_field: str | None = None
    sort_field: str | None = None
    keep_count: int = 10


@dataclass
class SmartCrusherConfig:
    """Configuration for smart crusher.

    SCHEMA-PRESERVING: Output contains only items from the original array.
    No wrappers, no generated text, no metadata keys.
    """
    enabled: bool = True
    min_items_to_analyze: int = 5      # Don't analyze tiny arrays
    min_tokens_to_crush: int = 200     # Only crush if > N tokens
    variance_threshold: float = 2.0    # Std devs for change point detection
    uniqueness_threshold: float = 0.1  # Below this = nearly constant
    similarity_threshold: float = 0.8  # For clustering similar strings
    max_items_after_crush: int = 15    # Target max items in output
    preserve_change_points: bool = True
    factor_out_constants: bool = False  # Disabled - preserves original schema
    include_summaries: bool = False     # Disabled - no generated text


class SmartAnalyzer:
    """Analyzes JSON arrays to determine optimal compression strategy."""

    def __init__(self, config: SmartCrusherConfig | None = None):
        self.config = config or SmartCrusherConfig()

    def analyze_array(self, items: list[dict]) -> ArrayAnalysis:
        """Perform complete statistical analysis of an array."""
        if not items or not isinstance(items[0], dict):
            return ArrayAnalysis(
                item_count=len(items) if items else 0,
                field_stats={},
                detected_pattern="generic",
                recommended_strategy=CompressionStrategy.NONE,
                constant_fields={},
                estimated_reduction=0.0,
            )

        # Analyze each field
        field_stats = {}
        all_keys = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        for key in all_keys:
            field_stats[key] = self._analyze_field(key, items)

        # Detect pattern
        pattern = self._detect_pattern(field_stats, items)

        # Extract constants
        constant_fields = {
            k: v.constant_value
            for k, v in field_stats.items()
            if v.is_constant
        }

        # Select strategy
        strategy = self._select_strategy(field_stats, pattern, len(items))

        # Estimate reduction
        reduction = self._estimate_reduction(field_stats, strategy, len(items))

        return ArrayAnalysis(
            item_count=len(items),
            field_stats=field_stats,
            detected_pattern=pattern,
            recommended_strategy=strategy,
            constant_fields=constant_fields,
            estimated_reduction=reduction,
        )

    def _analyze_field(self, key: str, items: list[dict]) -> FieldStats:
        """Analyze a single field across all items."""
        values = [item.get(key) for item in items if isinstance(item, dict)]
        non_null_values = [v for v in values if v is not None]

        if not non_null_values:
            return FieldStats(
                name=key,
                field_type="null",
                count=len(values),
                unique_count=0,
                unique_ratio=0.0,
                is_constant=True,
                constant_value=None,
            )

        # Determine type from first non-null value
        first_val = non_null_values[0]
        if isinstance(first_val, bool):
            field_type = "boolean"
        elif isinstance(first_val, (int, float)):
            field_type = "numeric"
        elif isinstance(first_val, str):
            field_type = "string"
        elif isinstance(first_val, dict):
            field_type = "object"
        elif isinstance(first_val, list):
            field_type = "array"
        else:
            field_type = "unknown"

        # Compute uniqueness
        str_values = [str(v) for v in values]
        unique_values = set(str_values)
        unique_count = len(unique_values)
        unique_ratio = unique_count / len(values) if values else 0

        # Check if constant
        is_constant = unique_count == 1
        constant_value = non_null_values[0] if is_constant else None

        stats = FieldStats(
            name=key,
            field_type=field_type,
            count=len(values),
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            is_constant=is_constant,
            constant_value=constant_value,
        )

        # Numeric-specific analysis
        if field_type == "numeric":
            nums = [v for v in non_null_values if isinstance(v, (int, float))]
            if nums:
                stats.min_val = min(nums)
                stats.max_val = max(nums)
                stats.mean_val = statistics.mean(nums)
                stats.variance = statistics.variance(nums) if len(nums) > 1 else 0
                stats.change_points = self._detect_change_points(nums)

        # String-specific analysis
        elif field_type == "string":
            strs = [v for v in non_null_values if isinstance(v, str)]
            if strs:
                stats.avg_length = statistics.mean(len(s) for s in strs)
                stats.top_values = Counter(strs).most_common(5)

        return stats

    def _detect_change_points(self, values: list[float], window: int = 5) -> list[int]:
        """Detect indices where values change significantly."""
        if len(values) < window * 2:
            return []

        change_points = []

        # Calculate overall statistics
        overall_std = statistics.stdev(values) if len(values) > 1 else 0
        if overall_std == 0:
            return []

        threshold = self.config.variance_threshold * overall_std

        # Sliding window comparison
        for i in range(window, len(values) - window):
            before_mean = statistics.mean(values[i-window:i])
            after_mean = statistics.mean(values[i:i+window])

            if abs(after_mean - before_mean) > threshold:
                change_points.append(i)

        # Deduplicate nearby change points
        if change_points:
            deduped = [change_points[0]]
            for cp in change_points[1:]:
                if cp - deduped[-1] > window:
                    deduped.append(cp)
            return deduped

        return []

    def _detect_pattern(self, field_stats: dict[str, FieldStats], items: list[dict]) -> str:
        """Detect the data pattern (time_series, logs, search_results, generic)."""
        keys_lower = {k.lower(): k for k in field_stats.keys()}

        # Check for time series pattern
        time_indicators = ["timestamp", "time", "date", "created", "updated", "@timestamp"]
        has_timestamp = any(t in keys_lower for t in time_indicators)

        numeric_fields = [k for k, v in field_stats.items() if v.field_type == "numeric"]
        has_numeric_with_variance = any(
            field_stats[k].variance and field_stats[k].variance > 0
            for k in numeric_fields
        )

        if has_timestamp and has_numeric_with_variance:
            return "time_series"

        # Check for logs pattern
        log_indicators = ["message", "msg", "log", "level", "severity"]
        has_message = any(t in keys_lower for t in log_indicators)
        has_level = any(t in keys_lower for t in ["level", "severity", "loglevel"])

        if has_message and has_level:
            return "logs"

        # Check for search results pattern
        score_indicators = ["score", "rank", "relevance", "confidence", "_score"]
        has_score = any(t in keys_lower for t in score_indicators)

        if has_score:
            return "search_results"

        return "generic"

    def _select_strategy(
        self,
        field_stats: dict[str, FieldStats],
        pattern: str,
        item_count: int
    ) -> CompressionStrategy:
        """Select optimal compression strategy based on analysis."""
        if item_count < self.config.min_items_to_analyze:
            return CompressionStrategy.NONE

        if pattern == "time_series":
            # Check if there are change points worth preserving
            numeric_fields = [v for v in field_stats.values() if v.field_type == "numeric"]
            has_change_points = any(f.change_points for f in numeric_fields)
            if has_change_points:
                return CompressionStrategy.TIME_SERIES

        if pattern == "logs":
            # Check if messages are clusterable (low-medium uniqueness)
            message_field = next(
                (v for k, v in field_stats.items() if "message" in k.lower()),
                None
            )
            if message_field and message_field.unique_ratio < 0.5:
                return CompressionStrategy.CLUSTER_SAMPLE

        if pattern == "search_results":
            return CompressionStrategy.TOP_N

        # Default: smart sampling
        return CompressionStrategy.SMART_SAMPLE

    def _estimate_reduction(
        self,
        field_stats: dict[str, FieldStats],
        strategy: CompressionStrategy,
        item_count: int
    ) -> float:
        """Estimate token reduction ratio."""
        if strategy == CompressionStrategy.NONE:
            return 0.0

        # Count constant fields (will be factored out)
        constant_ratio = sum(1 for v in field_stats.values() if v.is_constant) / len(field_stats)

        # Estimate based on strategy
        base_reduction = {
            CompressionStrategy.TIME_SERIES: 0.7,
            CompressionStrategy.CLUSTER_SAMPLE: 0.8,
            CompressionStrategy.TOP_N: 0.6,
            CompressionStrategy.SMART_SAMPLE: 0.5,
        }.get(strategy, 0.3)

        # Adjust for constants
        reduction = base_reduction + (constant_ratio * 0.2)

        return min(reduction, 0.95)


class SmartCrusher(Transform):
    """
    Intelligent tool output compression using statistical analysis.

    Unlike fixed-rule crushing, SmartCrusher:
    1. Analyzes JSON structure and computes field statistics
    2. Detects data patterns (time series, logs, search results)
    3. Identifies constant fields to factor out
    4. Finds change points in numeric data to preserve
    5. Applies optimal compression strategy per data type
    6. Uses RelevanceScorer for semantic matching of user queries

    This results in higher compression with lower information loss.
    """

    name = "smart_crusher"

    def __init__(
        self,
        config: SmartCrusherConfig | None = None,
        relevance_config: RelevanceScorerConfig | None = None,
        scorer: RelevanceScorer | None = None,
    ):
        self.config = config or SmartCrusherConfig()
        self.analyzer = SmartAnalyzer(self.config)

        # Initialize relevance scorer
        if scorer is not None:
            self._scorer = scorer
        else:
            rel_config = relevance_config or RelevanceScorerConfig()
            # Build kwargs based on tier - BM25 params only apply to bm25 tier
            scorer_kwargs = {}
            if rel_config.tier == "bm25":
                scorer_kwargs = {"k1": rel_config.bm25_k1, "b": rel_config.bm25_b}
            elif rel_config.tier == "hybrid":
                scorer_kwargs = {
                    "alpha": rel_config.hybrid_alpha,
                    "adaptive": rel_config.adaptive_alpha,
                }
            self._scorer = create_scorer(tier=rel_config.tier, **scorer_kwargs)
        # Use threshold from config, or default from RelevanceScorerConfig
        rel_cfg = relevance_config or RelevanceScorerConfig()
        self._relevance_threshold = rel_cfg.relevance_threshold

        # Error keywords for detection (CRITICAL: never lose errors)
        self._error_keywords = {'error', 'exception', 'failed', 'failure', 'critical', 'fatal'}

    def _prioritize_indices(
        self,
        keep_indices: set[int],
        items: list[dict],
        n: int,
        analysis: ArrayAnalysis | None = None,
    ) -> set[int]:
        """Prioritize indices when we exceed max_items, ALWAYS keeping errors and anomalies.

        Priority order:
        1. ALL error items (non-negotiable)
        2. ALL numeric anomalies (non-negotiable) - e.g., unusual values like 999999
        3. First 3 items (context)
        4. Last 2 items (context)
        5. Other important items by index order
        """
        if len(keep_indices) <= self.config.max_items_after_crush:
            return keep_indices

        # Identify error indices (MUST keep ALL of them)
        error_indices = set()
        for i, item in enumerate(items):
            item_str = str(item).lower()
            if any(kw in item_str for kw in self._error_keywords):
                error_indices.add(i)

        # Identify numeric anomalies (MUST keep ALL of them)
        anomaly_indices = set()
        if analysis and analysis.field_stats:
            for field_name, stats in analysis.field_stats.items():
                if stats.field_type == "numeric" and stats.mean_val is not None and stats.variance:
                    std = stats.variance ** 0.5
                    if std > 0:
                        threshold = self.config.variance_threshold * std
                        for i, item in enumerate(items):
                            val = item.get(field_name)
                            if isinstance(val, (int, float)):
                                if abs(val - stats.mean_val) > threshold:
                                    anomaly_indices.add(i)

        # Start with all errors and anomalies (these are non-negotiable)
        prioritized = error_indices | anomaly_indices

        # Add first/last items if we have room
        remaining_slots = self.config.max_items_after_crush - len(prioritized)
        if remaining_slots > 0:
            # First 3 items
            for i in range(min(3, n)):
                if i not in prioritized and remaining_slots > 0:
                    prioritized.add(i)
                    remaining_slots -= 1
            # Last 2 items
            for i in range(max(0, n - 2), n):
                if i not in prioritized and remaining_slots > 0:
                    prioritized.add(i)
                    remaining_slots -= 1

        # Fill remaining slots with other important indices (by index order)
        if remaining_slots > 0:
            other_indices = sorted(keep_indices - prioritized)
            for i in other_indices:
                if remaining_slots <= 0:
                    break
                prioritized.add(i)
                remaining_slots -= 1

        return prioritized

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if any tool messages would benefit from smart crushing."""
        if not self.config.enabled:
            return False

        for msg in messages:
            # OpenAI style: role="tool"
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str):
                    tokens = tokenizer.count_text(content)
                    if tokens > self.config.min_tokens_to_crush:
                        # Check if it's JSON with arrays
                        parsed, success = safe_json_loads(content)
                        if success and self._has_crushable_arrays(parsed):
                            return True

            # Anthropic style: role="user" with tool_result content blocks
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, str):
                            tokens = tokenizer.count_text(tool_content)
                            if tokens > self.config.min_tokens_to_crush:
                                parsed, success = safe_json_loads(tool_content)
                                if success and self._has_crushable_arrays(parsed):
                                    return True

        return False

    def _has_crushable_arrays(self, data: Any, depth: int = 0) -> bool:
        """Check if data contains arrays large enough to crush."""
        if depth > 5:
            return False

        if isinstance(data, list):
            if len(data) >= self.config.min_items_to_analyze:
                if data and isinstance(data[0], dict):
                    return True
            for item in data[:10]:  # Check first few items
                if self._has_crushable_arrays(item, depth + 1):
                    return True

        elif isinstance(data, dict):
            for value in data.values():
                if self._has_crushable_arrays(value, depth + 1):
                    return True

        return False

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Apply smart crushing to messages."""
        tokens_before = tokenizer.count_messages(messages)
        result_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        markers_inserted: list[str] = []
        warnings: list[str] = []

        # Extract query context from recent user messages for relevance scoring
        query_context = self._extract_context_from_messages(result_messages)

        crushed_count = 0

        for msg in result_messages:
            # OpenAI style
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue

                tokens = tokenizer.count_text(content)
                if tokens <= self.config.min_tokens_to_crush:
                    continue

                crushed, was_modified, analysis_info = self._smart_crush_content(
                    content, query_context
                )

                if was_modified:
                    original_hash = compute_short_hash(content)
                    marker = create_tool_digest_marker(original_hash)
                    msg["content"] = crushed + "\n" + marker
                    crushed_count += 1
                    markers_inserted.append(marker)
                    if analysis_info:
                        transforms_applied.append(f"smart:{analysis_info}")

            # Anthropic style
            content = msg.get("content")
            if isinstance(content, list):
                for i, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue

                    tool_content = block.get("content", "")
                    if not isinstance(tool_content, str):
                        continue

                    tokens = tokenizer.count_text(tool_content)
                    if tokens <= self.config.min_tokens_to_crush:
                        continue

                    crushed, was_modified, analysis_info = self._smart_crush_content(
                        tool_content, query_context
                    )

                    if was_modified:
                        original_hash = compute_short_hash(tool_content)
                        marker = create_tool_digest_marker(original_hash)
                        content[i]["content"] = crushed + "\n" + marker
                        crushed_count += 1
                        markers_inserted.append(marker)
                        if analysis_info:
                            transforms_applied.append(f"smart:{analysis_info}")

        if crushed_count > 0:
            transforms_applied.insert(0, f"smart_crush:{crushed_count}")

        tokens_after = tokenizer.count_messages(result_messages)

        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
            warnings=warnings,
        )

    def _extract_context_from_messages(
        self, messages: list[dict[str, Any]]
    ) -> str:
        """Extract query context from recent messages for relevance scoring.

        Builds a context string from:
        - Recent user messages (what the user is asking about)
        - Recent tool call arguments (what data was requested)

        This context is used by RelevanceScorer to determine which items
        to preserve during crushing.

        Args:
            messages: Full message list.

        Returns:
            Context string for relevance scoring.
        """
        context_parts: list[str] = []

        # Look at last 5 user messages (most relevant to recent tool calls)
        user_message_count = 0
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    context_parts.append(content)
                elif isinstance(content, list):
                    # Anthropic style - extract from text blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                context_parts.append(text)

                user_message_count += 1
                if user_message_count >= 5:
                    break

            # Also check assistant tool_calls for function arguments
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        args = func.get("arguments", "")
                        if isinstance(args, str) and args:
                            context_parts.append(args)

        return " ".join(context_parts)

    def _smart_crush_content(
        self, content: str, query_context: str = ""
    ) -> tuple[str, bool, str]:
        """
        Apply smart crushing to content.

        Args:
            content: JSON string to crush.
            query_context: Context string from user messages for relevance scoring.

        Returns:
            Tuple of (crushed_content, was_modified, analysis_info).
        """
        parsed, success = safe_json_loads(content)
        if not success:
            return content, False, ""

        # Recursively process and crush arrays
        crushed, info = self._process_value(parsed, query_context=query_context)

        result = safe_json_dumps(crushed, indent=None)
        was_modified = result != content.strip()

        return result, was_modified, info

    def _process_value(
        self, value: Any, depth: int = 0, query_context: str = ""
    ) -> tuple[Any, str]:
        """Recursively process a value, crushing arrays where appropriate."""
        info_parts = []

        if isinstance(value, list):
            # Check if this array should be crushed
            if (len(value) >= self.config.min_items_to_analyze and
                value and isinstance(value[0], dict)):

                crushed, strategy = self._crush_array(value, query_context)
                info_parts.append(f"{strategy}({len(value)}->{len(crushed)})")
                return crushed, ",".join(info_parts)
            else:
                # Process items recursively
                processed = []
                for item in value:
                    p_item, p_info = self._process_value(item, depth + 1, query_context)
                    processed.append(p_item)
                    if p_info:
                        info_parts.append(p_info)
                return processed, ",".join(info_parts)

        elif isinstance(value, dict):
            # Process values recursively
            processed = {}
            for k, v in value.items():
                p_val, p_info = self._process_value(v, depth + 1, query_context)
                processed[k] = p_val
                if p_info:
                    info_parts.append(p_info)
            return processed, ",".join(info_parts)

        else:
            return value, ""

    def _crush_array(
        self, items: list[dict], query_context: str = ""
    ) -> tuple[list, str]:
        """Crush an array using statistical analysis and relevance scoring."""
        # Analyze the array
        analysis = self.analyzer.analyze_array(items)

        # Create compression plan with relevance scoring
        plan = self._create_plan(analysis, items, query_context)

        # Execute compression
        result = self._execute_plan(plan, items, analysis)

        return result, analysis.recommended_strategy.value

    def _create_plan(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        query_context: str = "",
    ) -> CompressionPlan:
        """Create a detailed compression plan using relevance scoring."""
        plan = CompressionPlan(
            strategy=analysis.recommended_strategy,
            constant_fields=analysis.constant_fields if self.config.factor_out_constants else {},
        )

        if analysis.recommended_strategy == CompressionStrategy.TIME_SERIES:
            plan = self._plan_time_series(analysis, items, plan, query_context)

        elif analysis.recommended_strategy == CompressionStrategy.CLUSTER_SAMPLE:
            plan = self._plan_cluster_sample(analysis, items, plan, query_context)

        elif analysis.recommended_strategy == CompressionStrategy.TOP_N:
            plan = self._plan_top_n(analysis, items, plan, query_context)

        else:  # SMART_SAMPLE or NONE
            plan = self._plan_smart_sample(analysis, items, plan, query_context)

        return plan

    def _plan_time_series(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        plan: CompressionPlan,
        query_context: str = "",
    ) -> CompressionPlan:
        """Plan compression for time series data.

        Keeps items around change points (anomalies) plus first/last items.
        Uses Safe V1 Recipe for additional error detection.
        Uses RelevanceScorer for semantic matching of user queries.
        """
        n = len(items)
        keep_indices = set()

        # 1. First 3 items
        for i in range(min(3, n)):
            keep_indices.add(i)

        # 2. Last 2 items
        for i in range(max(0, n - 2), n):
            keep_indices.add(i)

        # 3. Items around change points from numeric fields
        for stats in analysis.field_stats.values():
            if stats.change_points:
                for cp in stats.change_points:
                    # Keep a window around each change point
                    for offset in range(-2, 3):
                        idx = cp + offset
                        if 0 <= idx < n:
                            keep_indices.add(idx)

        # 4. Error items
        error_keywords = {'error', 'exception', 'failed', 'failure', 'critical', 'fatal'}
        for i, item in enumerate(items):
            item_str = str(item).lower()
            if any(kw in item_str for kw in error_keywords):
                keep_indices.add(i)

        # 5. Items with high relevance to query context (CRITICAL: preserve needle records)
        if query_context:
            item_strs = [json.dumps(item, default=str) for item in items]
            scores = self._scorer.score_batch(item_strs, query_context)
            for i, score in enumerate(scores):
                if score.score >= self._relevance_threshold:
                    keep_indices.add(i)

        # Limit to max_items_after_crush while ALWAYS preserving errors and anomalies
        keep_indices = self._prioritize_indices(keep_indices, items, n, analysis)

        plan.keep_indices = sorted(keep_indices)
        return plan

    def _plan_cluster_sample(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        plan: CompressionPlan,
        query_context: str = "",
    ) -> CompressionPlan:
        """Plan compression for clusterable data (like logs).

        Uses clustering plus Safe V1 Recipe for error detection.
        Uses RelevanceScorer for semantic matching of user queries.
        """
        n = len(items)
        keep_indices = set()

        # 1. First 3 items (Safe V1)
        for i in range(min(3, n)):
            keep_indices.add(i)

        # 2. Last 2 items (Safe V1)
        for i in range(max(0, n - 2), n):
            keep_indices.add(i)

        # 3. Error items (Safe V1 - never lose errors)
        error_keywords = {'error', 'exception', 'failed', 'failure', 'critical', 'fatal'}
        for i, item in enumerate(items):
            item_str = str(item).lower()
            if any(kw in item_str for kw in error_keywords):
                keep_indices.add(i)

        # 4. Cluster by message field and keep representatives
        message_field = None
        for name, stats in analysis.field_stats.items():
            if "message" in name.lower() or "msg" in name.lower():
                message_field = name
                break

        if message_field:
            plan.cluster_field = message_field

            # Simple clustering: group by first 50 chars of message
            clusters: dict[str, list[int]] = {}
            for i, item in enumerate(items):
                msg = str(item.get(message_field, ""))[:50]
                msg_hash = hashlib.md5(msg.encode()).hexdigest()[:8]
                if msg_hash not in clusters:
                    clusters[msg_hash] = []
                clusters[msg_hash].append(i)

            # Keep 1-2 representatives from each cluster
            for indices in clusters.values():
                for idx in indices[:2]:
                    keep_indices.add(idx)

        # 5. Items with high relevance to query context (CRITICAL: preserve needle records)
        if query_context:
            item_strs = [json.dumps(item, default=str) for item in items]
            scores = self._scorer.score_batch(item_strs, query_context)
            for i, score in enumerate(scores):
                if score.score >= self._relevance_threshold:
                    keep_indices.add(i)

        # Limit total while ALWAYS preserving errors and anomalies
        keep_indices = self._prioritize_indices(keep_indices, items, n, analysis)

        plan.keep_indices = sorted(keep_indices)
        return plan

    def _plan_top_n(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        plan: CompressionPlan,
        query_context: str = "",
    ) -> CompressionPlan:
        """Plan compression for scored/ranked data using Safe V1 Recipe.

        Keeps top N by score PLUS error items and relevance-matched items.
        Uses RelevanceScorer for semantic matching of user queries.
        """
        # Find score field
        score_field = None
        for name in analysis.field_stats.keys():
            if any(s in name.lower() for s in ["score", "rank", "relevance", "_score"]):
                score_field = name
                break

        if not score_field:
            return self._plan_smart_sample(analysis, items, plan, query_context)

        plan.sort_field = score_field
        keep_indices = set()

        # 1. Items with high relevance FIRST (CRITICAL: preserve needle records)
        # These are given priority over top N since the user is specifically looking for them
        if query_context:
            item_strs = [json.dumps(item, default=str) for item in items]
            scores = self._scorer.score_batch(item_strs, query_context)
            for i, score in enumerate(scores):
                if score.score >= self._relevance_threshold:
                    keep_indices.add(i)

        # 2. Top N by score (adjusted for relevance matches already kept)
        scored_items = [
            (i, item.get(score_field, 0))
            for i, item in enumerate(items)
        ]
        scored_items.sort(key=lambda x: x[1], reverse=True)

        remaining_slots = self.config.max_items_after_crush - len(keep_indices) - 3  # Reserve for errors
        top_count = min(max(0, remaining_slots), len(items))
        for idx, _ in scored_items[:top_count]:
            keep_indices.add(idx)

        # 3. Error items (Safe V1 Recipe - always keep errors regardless of score)
        error_keywords = {'error', 'exception', 'failed', 'failure', 'critical', 'fatal'}
        for i, item in enumerate(items):
            item_str = str(item).lower()
            if any(kw in item_str for kw in error_keywords):
                keep_indices.add(i)

        # Limit total
        plan.keep_count = len(keep_indices)
        plan.keep_indices = sorted(keep_indices)
        return plan

    def _plan_smart_sample(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        plan: CompressionPlan,
        query_context: str = "",
    ) -> CompressionPlan:
        """Plan smart statistical sampling using Safe V1 Recipe.

        Safe V1 Recipe - Always keeps:
        - First K items (default 3)
        - Last K items (default 2)
        - Error items (containing 'error', 'exception', 'failed', 'critical')
        - Anomalous numeric items (> 2 std from mean)
        - Items around change points
        - Items with high relevance to query context (via RelevanceScorer)
        """
        n = len(items)
        keep_indices = set()

        # 1. First K items (default 3)
        for i in range(min(3, n)):
            keep_indices.add(i)

        # 2. Last K items (default 2)
        for i in range(max(0, n - 2), n):
            keep_indices.add(i)

        # 3. Error items (containing error keywords)
        error_keywords = {'error', 'exception', 'failed', 'failure', 'critical', 'fatal'}
        for i, item in enumerate(items):
            item_str = str(item).lower()
            if any(kw in item_str for kw in error_keywords):
                keep_indices.add(i)

        # 4. Anomalous numeric items (> 2 std from mean)
        for name, stats in analysis.field_stats.items():
            if stats.field_type == "numeric" and stats.mean_val is not None and stats.variance:
                std = stats.variance ** 0.5
                if std > 0:
                    threshold = self.config.variance_threshold * std
                    for i, item in enumerate(items):
                        val = item.get(name)
                        if isinstance(val, (int, float)):
                            if abs(val - stats.mean_val) > threshold:
                                keep_indices.add(i)

        # 5. Items around change points (if detected)
        if self.config.preserve_change_points:
            for stats in analysis.field_stats.values():
                if stats.change_points:
                    for cp in stats.change_points:
                        # Keep items around change point
                        for offset in range(-1, 2):
                            idx = cp + offset
                            if 0 <= idx < n:
                                keep_indices.add(idx)

        # 6. Items with high relevance to query context (CRITICAL: preserve needle records)
        if query_context:
            item_strs = [json.dumps(item, default=str) for item in items]
            scores = self._scorer.score_batch(item_strs, query_context)
            for i, score in enumerate(scores):
                if score.score >= self._relevance_threshold:
                    keep_indices.add(i)

        # Limit to max_items_after_crush while ALWAYS preserving errors and anomalies
        keep_indices = self._prioritize_indices(keep_indices, items, n, analysis)

        plan.keep_indices = sorted(keep_indices)
        return plan

    def _execute_plan(
        self,
        plan: CompressionPlan,
        items: list[dict],
        analysis: ArrayAnalysis
    ) -> list:
        """Execute a compression plan and return crushed array.

        SCHEMA-PRESERVING: Returns only items from the original array.
        No wrappers, no generated text, no metadata keys.
        """
        result = []

        # Return only the kept items, preserving original schema
        for idx in sorted(plan.keep_indices):
            if 0 <= idx < len(items):
                # Copy item unchanged - no modifications to schema
                result.append(items[idx].copy())

        return result


def smart_crush_tool_output(
    content: str,
    config: SmartCrusherConfig | None = None,
) -> tuple[str, bool, str]:
    """
    Convenience function to smart-crush a single tool output.

    Args:
        content: The tool output content (JSON string).
        config: Optional configuration.

    Returns:
        Tuple of (crushed_content, was_modified, analysis_info).
    """
    cfg = config or SmartCrusherConfig()
    crusher = SmartCrusher(cfg)
    return crusher._smart_crush_content(content)
