"""Smart statistical tool output compression for Headroom SDK.

This module provides intelligent JSON compression based on statistical analysis
rather than fixed rules. It analyzes data patterns and applies optimal compression
strategies to maximize token reduction while preserving important information.

SCOPE: SmartCrusher handles JSON arrays of ANY type — dicts, strings, numbers,
mixed types, and nested arrays. Non-JSON content (plain text, search results,
logs, code, diffs) passes through UNCHANGED.

TEXT COMPRESSION IS OPT-IN: For text-based content, Headroom provides standalone
utilities that applications can use explicitly:
- SearchCompressor: For grep/ripgrep output (file:line:content format)
- LogCompressor: For build/test logs (pytest, npm, cargo output)
- TextCompressor: For generic plain text with anchor preservation

Applications should decide when and how to use text compression based on their
specific needs. This design prevents lossy text compression from being applied
automatically, which could lose important context in coding tasks.

SCHEMA-PRESERVING: Output contains only items from the original array.
No wrappers, no generated text, no metadata keys. This ensures downstream
tools and parsers work unchanged.

Supported JSON types:
- Arrays of dicts: Full statistical analysis with adaptive K (Kneedle algorithm)
- Arrays of strings: Dedup + adaptive sampling + error preservation
- Arrays of numbers: Statistical summary + outlier/change-point preservation
- Mixed-type arrays: Grouped by type, each group compressed independently
- Flat objects (many keys): Key-level adaptive sampling
- Nested objects: Recursive compression of inner arrays/objects

Safety guarantees (consistent across ALL types):
- First K, last K items always kept (K is adaptive, not hardcoded)
- Error items (containing 'error', 'exception', 'failed', 'critical') never dropped
- Anomalous numeric items (> 2 std from mean) always kept
- Items around detected change points preserved
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
import logging
import math
import re
import statistics
import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..cache.compression_feedback import CompressionFeedback, get_compression_feedback
from ..cache.compression_store import CompressionStore, get_compression_store
from ..config import AnchorConfig, CCRConfig, RelevanceScorerConfig, TransformResult
from ..relevance import RelevanceScorer, create_scorer
from ..telemetry import TelemetryCollector, ToolSignature, get_telemetry_collector
from ..telemetry.models import FieldSemantics
from ..telemetry.toin import ToolIntelligenceNetwork, get_toin
from ..tokenizer import Tokenizer
from ..utils import (
    compute_short_hash,
    create_tool_digest_marker,
    deep_copy_messages,
    safe_json_dumps,
    safe_json_loads,
)
from .anchor_selector import AnchorSelector
from .anchor_selector import DataPattern as AnchorDataPattern
from .base import Transform

logger = logging.getLogger(__name__)

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


def _hash_field_name(field_name: str) -> str:
    """Hash a field name to match TOIN's anonymized preserve_fields.

    TOIN stores field names as SHA256[:8] hashes for privacy.
    This function produces the same hash format.
    """
    return hashlib.sha256(field_name.encode()).hexdigest()[:8]


def _get_preserve_field_values(
    item: dict,
    preserve_field_hashes: list[str],
) -> list[tuple[str, Any]]:
    """Get values from item fields that match TOIN's preserve_field hashes.

    TOIN stores preserve_fields as hashed field names (SHA256[:8]).
    This function iterates over item fields, hashes each, and returns
    matching field names and values.

    Args:
        item: Dictionary item from tool output.
        preserve_field_hashes: List of SHA256[:8] hashed field names from TOIN.

    Returns:
        List of (field_name, value) tuples for fields that match.
    """
    if not preserve_field_hashes or not item:
        return []

    # Convert preserve_fields to set for O(1) lookup
    hash_set = set(preserve_field_hashes)

    matches = []
    for field_name, value in item.items():
        field_hash = _hash_field_name(field_name)
        if field_hash in hash_set:
            matches.append((field_name, value))

    return matches


def _item_has_preserve_field_match(
    item: dict,
    preserve_field_hashes: list[str],
    query_context: str,
) -> bool:
    """Check if item has a preserve_field value that matches query context.

    Args:
        item: Dictionary item from tool output.
        preserve_field_hashes: List of SHA256[:8] hashed field names from TOIN.
        query_context: User's query to match against field values.

    Returns:
        True if any preserve_field value matches the query context.
    """
    if not query_context:
        return False

    query_lower = query_context.lower()

    for _field_name, value in _get_preserve_field_values(item, preserve_field_hashes):
        if value is not None:
            value_str = str(value).lower()
            if value_str in query_lower or query_lower in value_str:
                return True

    return False


class CompressionStrategy(Enum):
    """Compression strategies based on data patterns."""

    NONE = "none"  # No compression needed
    SKIP = "skip"  # Explicitly skip - not safe to crush
    TIME_SERIES = "time_series"  # Keep change points, summarize stable
    CLUSTER_SAMPLE = "cluster"  # Dedupe similar items
    TOP_N = "top_n"  # Keep highest scored items
    SMART_SAMPLE = "smart_sample"  # Statistical sampling with constants


class ArrayType(Enum):
    """JSON array element type classification."""

    DICT_ARRAY = "dict_array"  # [{...}, {...}, ...]
    STRING_ARRAY = "string_array"  # ["a", "b", "c", ...]
    NUMBER_ARRAY = "number_array"  # [1, 2.5, 3, ...]
    BOOL_ARRAY = "bool_array"  # [true, false, ...]
    NESTED_ARRAY = "nested_array"  # [[...], [...], ...]
    MIXED_ARRAY = "mixed_array"  # [{"a":1}, "str", 42, ...]
    EMPTY = "empty"


def _classify_array(items: list) -> ArrayType:
    """Classify a JSON array by its element types.

    Uses set-of-types check on ALL elements (not sampling) to guarantee
    correct classification. Fast because type() is O(1).
    """
    if not items:
        return ArrayType.EMPTY
    # Note: bool is a subclass of int in Python, so check bool first
    types = set()
    has_bool = False
    for item in items:
        if isinstance(item, bool):
            has_bool = True
        types.add(type(item))
    if has_bool and types <= {bool, int}:
        # All bools (Python's True/False are int subclass)
        if all(isinstance(i, bool) for i in items):
            return ArrayType.BOOL_ARRAY
    if types == {dict}:
        return ArrayType.DICT_ARRAY
    if types == {str}:
        return ArrayType.STRING_ARRAY
    if types <= {int, float} and not has_bool:
        return ArrayType.NUMBER_ARRAY
    if types == {list}:
        return ArrayType.NESTED_ARRAY
    return ArrayType.MIXED_ARRAY


# =====================================================================
# STATISTICAL FIELD DETECTION (replaces hardcoded string patterns)
# =====================================================================
# Instead of matching field names like "id", "score", "error", we use
# statistical and structural properties of the data to detect field types.


def _is_uuid_format(value: str) -> bool:
    """Check if a string looks like a UUID (structural pattern)."""
    if not isinstance(value, str) or len(value) != 36:
        return False
    # UUID format: 8-4-4-4-12 hex chars
    parts = value.split("-")
    if len(parts) != 5:
        return False
    expected_lens = [8, 4, 4, 4, 12]
    for part, expected_len in zip(parts, expected_lens):
        if len(part) != expected_len:
            return False
        if not all(c in "0123456789abcdefABCDEF" for c in part):
            return False
    return True


def _calculate_string_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string, normalized to [0, 1].

    High entropy (>0.7) suggests random/ID-like content.
    Low entropy (<0.3) suggests repetitive/predictable content.
    """
    if not s or len(s) < 2:
        return 0.0

    # Count character frequencies
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1

    # Calculate entropy
    import math

    entropy = 0.0
    length = len(s)
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalize by max possible entropy for this length
    max_entropy = math.log2(min(len(freq), length))
    if max_entropy > 0:
        return entropy / max_entropy
    return 0.0


def _detect_sequential_pattern(values: list[Any], check_order: bool = True) -> bool:
    """Detect if numeric values form a sequential pattern (like IDs: 1,2,3,...).

    Returns True if values appear to be auto-incrementing or sequential.

    Args:
        values: List of values to check.
        check_order: If True, also check if values are in ascending order in the array.
                     Score fields are often sorted descending, while IDs are ascending.
    """
    if len(values) < 5:
        return False

    # Get numeric values
    nums = []
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nums.append(v)
        elif isinstance(v, str):
            try:
                nums.append(int(v))
            except ValueError:
                pass

    if len(nums) < 5:
        return False

    # Check if sorted values form a near-sequence
    sorted_nums = sorted(nums)
    diffs = [sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]

    if not diffs:
        return False

    # If most differences are 1 (or small constant), it's sequential
    avg_diff = sum(diffs) / len(diffs)
    if 0.5 <= avg_diff <= 2.0:
        # Check consistency - sequential IDs have consistent spacing
        consistent_count = sum(1 for d in diffs if 0.5 <= d <= 2.0)
        is_sequential = consistent_count / len(diffs) > 0.8

        # Additional check: IDs are typically in ASCENDING order in the array
        # Scores sorted by relevance are typically in DESCENDING order
        if check_order and is_sequential:
            # Check if original order is ascending (like IDs)
            ascending_count = sum(1 for i in range(len(nums) - 1) if nums[i] <= nums[i + 1])
            is_ascending = ascending_count / (len(nums) - 1) > 0.7
            return is_ascending  # Only flag as sequential if ascending (ID-like)

        return is_sequential

    return False


def _detect_id_field_statistically(stats: FieldStats, values: list[Any]) -> tuple[bool, float]:
    """Detect if a field is an ID field using statistical properties.

    Returns (is_id_field, confidence).

    ID fields have:
    - Very high uniqueness (>0.95)
    - Sequential numeric pattern OR UUID format OR high entropy strings
    """
    # Must have high uniqueness
    if stats.unique_ratio < 0.9:
        return False, 0.0

    confidence = 0.0

    # Check for UUID format (structural detection)
    if stats.field_type == "string":
        sample_values = [v for v in values[:20] if isinstance(v, str)]
        uuid_count = sum(1 for v in sample_values if _is_uuid_format(v))
        if sample_values and uuid_count / len(sample_values) > 0.8:
            return True, 0.95

        # Check for high entropy (random string IDs)
        if sample_values:
            avg_entropy = sum(_calculate_string_entropy(v) for v in sample_values) / len(
                sample_values
            )
            if avg_entropy > 0.7 and stats.unique_ratio > 0.95:
                confidence = 0.8
                return True, confidence

    # Check for sequential numeric pattern
    if stats.field_type == "numeric":
        if _detect_sequential_pattern(values) and stats.unique_ratio > 0.95:
            return True, 0.9

        # High uniqueness numeric with high range suggests ID
        if stats.min_val is not None and stats.max_val is not None:
            value_range = stats.max_val - stats.min_val
            if value_range > 0 and stats.unique_ratio > 0.95:
                return True, 0.85

    # Very high uniqueness alone is a signal (even without other patterns)
    if stats.unique_ratio > 0.98:
        return True, 0.7

    return False, 0.0


def _detect_score_field_statistically(stats: FieldStats, items: list[dict]) -> tuple[bool, float]:
    """Detect if a field is a score/ranking field using statistical properties.

    Returns (is_score_field, confidence).

    Score fields have:
    - Numeric type
    - Bounded range (0-1, 0-10, 0-100, or similar)
    - NOT sequential (unlike IDs)
    - Often the data appears sorted by this field (descending)
    """
    if stats.field_type != "numeric":
        return False, 0.0

    if stats.min_val is None or stats.max_val is None:
        return False, 0.0

    confidence = 0.0

    # Check for bounded range typical of scores
    stats.max_val - stats.min_val
    min_val, max_val = stats.min_val, stats.max_val

    # Common score ranges: [0,1], [0,10], [0,100], [-1,1], [0,5]
    is_bounded = False
    if 0 <= min_val <= 1 and 0 <= max_val <= 1:  # [0,1] range
        is_bounded = True
        confidence += 0.4
    elif 0 <= min_val <= 10 and 0 <= max_val <= 10:  # [0,10] range
        is_bounded = True
        confidence += 0.3
    elif 0 <= min_val <= 100 and 0 <= max_val <= 100:  # [0,100] range
        is_bounded = True
        confidence += 0.25
    elif -1 <= min_val and max_val <= 1:  # [-1,1] range
        is_bounded = True
        confidence += 0.35

    if not is_bounded:
        return False, 0.0

    # Should NOT be sequential (IDs are sequential, scores are not)
    sample_values = [item.get(stats.name) for item in items[:50] if stats.name in item]
    if _detect_sequential_pattern(sample_values):
        return False, 0.0

    # Check if data appears sorted by this field (descending = relevance sorted)
    # Filter out NaN/Inf which break comparisons
    values_in_order: list[float] = []
    for item in items:
        if stats.name in item:
            val = item.get(stats.name)
            if isinstance(val, (int, float)) and math.isfinite(val):
                values_in_order.append(float(val))
    if len(values_in_order) >= 5:
        # Check for descending sort
        descending_count = sum(
            1
            for i in range(len(values_in_order) - 1)
            if values_in_order[i] >= values_in_order[i + 1]
        )
        if descending_count / (len(values_in_order) - 1) > 0.7:
            confidence += 0.3

    # Score fields often have floating point values
    # Filter out NaN/Inf which can't be converted to int
    float_count = sum(
        1 for v in values_in_order[:20] if isinstance(v, float) and math.isfinite(v) and v != int(v)
    )
    if float_count > len(values_in_order[:20]) * 0.3:
        confidence += 0.1

    return confidence >= 0.4, min(confidence, 0.95)


def _detect_structural_outliers(items: list[dict]) -> list[int]:
    """Detect items that are structural outliers (error-like items).

    Instead of looking for "error" keywords, we detect:
    1. Items with extra fields that others don't have
    2. Items with rare status/state values
    3. Items with significantly different structure

    Returns indices of outlier items.
    """
    if len(items) < 5:
        return []

    outlier_indices: list[int] = []

    # 1. Detect items with extra fields
    # Find the "common" field set (fields present in >80% of items)
    field_counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            for key in item.keys():
                field_counts[key] = field_counts.get(key, 0) + 1

    n = len(items)
    common_fields = {k for k, v in field_counts.items() if v >= n * 0.8}
    rare_fields = {k for k, v in field_counts.items() if v < n * 0.2}

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        item_fields = set(item.keys())

        # Has rare fields that most items don't have
        has_rare = bool(item_fields & rare_fields)
        if has_rare:
            outlier_indices.append(i)
            continue

    # 2. Detect rare status/state values
    # Find fields that look like status fields (low cardinality, categorical)
    status_outliers = _detect_rare_status_values(items, common_fields)
    outlier_indices.extend(status_outliers)

    return list(set(outlier_indices))


def _detect_rare_status_values(items: list[dict], common_fields: set[str]) -> list[int]:
    """Detect items with rare values in status-like fields.

    A status field has low cardinality (few distinct values).
    If 95%+ have the same value, items with different values are interesting.
    """
    outlier_indices: list[int] = []

    # Find potential status fields (low cardinality)
    for field_name in common_fields:
        values = [
            item.get(field_name) for item in items if isinstance(item, dict) and field_name in item
        ]

        # Skip if too few values or non-hashable
        try:
            unique_values = {str(v) for v in values if v is not None}
        except Exception:
            continue

        # Status field = low cardinality (2-10 distinct values)
        if not (2 <= len(unique_values) <= 10):
            continue

        # Count value frequencies
        value_counts: dict[str, int] = {}
        for v in values:
            key = str(v) if v is not None else "__none__"
            value_counts[key] = value_counts.get(key, 0) + 1

        # Find the dominant value
        if not value_counts:
            continue

        max_count = max(value_counts.values())
        total = len(values)

        # If one value dominates (>90%), others are interesting
        if max_count >= total * 0.9:
            dominant_value = max(value_counts.keys(), key=lambda k: value_counts[k])

            for i, item in enumerate(items):
                if not isinstance(item, dict) or field_name not in item:
                    continue
                item_value = str(item[field_name]) if item[field_name] is not None else "__none__"
                if item_value != dominant_value:
                    outlier_indices.append(i)

    return outlier_indices


# Error keywords for PRESERVATION guarantee (not crushability detection)
# This is for the quality guarantee: "ALL error items are ALWAYS preserved"
# regardless of how common they are. Used in _prioritize_indices().
_ERROR_KEYWORDS_FOR_PRESERVATION = frozenset(
    {
        "error",
        "exception",
        "failed",
        "failure",
        "critical",
        "fatal",
        "crash",
        "panic",
        "abort",
        "timeout",
        "denied",
        "rejected",
    }
)


def _detect_error_items_for_preservation(items: list[dict]) -> list[int]:
    """Detect items containing error keywords for PRESERVATION guarantee.

    This is NOT for crushability analysis - it's for ensuring ALL error items
    are retained during compression. The quality guarantee is that error items
    are NEVER dropped, even if errors are common in the dataset.

    Uses keywords because error semantics are well-defined across domains.
    """
    error_indices: list[int] = []

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        # Serialize item to check all content
        try:
            item_str = json.dumps(item).lower()
        except Exception:
            continue

        # Check if any error keyword is present
        for keyword in _ERROR_KEYWORDS_FOR_PRESERVATION:
            if keyword in item_str:
                error_indices.append(i)
                break

    return error_indices


def _detect_items_by_learned_semantics(
    items: list[dict],
    field_semantics: dict[str, FieldSemantics],
) -> list[int]:
    """Detect items with important values based on learned field semantics.

    This is the TOIN Evolution integration - uses learned field semantic types
    to identify items that should be preserved during compression.

    Key insight: Instead of hardcoded patterns, we learn from user behavior
    which field values are actually important (e.g., error indicators, rare
    status values, identifiers that get queried).

    Args:
        items: List of items to analyze.
        field_semantics: Learned field semantics from TOIN (field_hash -> FieldSemantics).

    Returns:
        List of indices for items containing important values.
    """
    if not field_semantics or not items:
        return []

    important_indices: list[int] = []

    # Build a quick lookup for field_hash -> FieldSemantics
    # Pre-filter to fields with sufficient confidence
    confident_semantics = {
        fh: fs
        for fh, fs in field_semantics.items()
        if fs.confidence >= 0.3 and fs.inferred_type != "unknown"
    }

    if not confident_semantics:
        return []

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        for field_name, value in item.items():
            # Hash the field name to match TOIN's format
            field_hash = hashlib.sha256(field_name.encode()).hexdigest()[:8]

            if field_hash not in confident_semantics:
                continue

            field_sem = confident_semantics[field_hash]

            # Hash the value to check importance
            if value is None:
                value_canonical = "null"
            elif isinstance(value, bool):
                value_canonical = "true" if value else "false"
            elif isinstance(value, (int, float)):
                value_canonical = str(value)
            elif isinstance(value, str):
                value_canonical = value
            elif isinstance(value, (list, dict)):
                try:
                    value_canonical = json.dumps(value, sort_keys=True, default=str)
                except (TypeError, ValueError):
                    value_canonical = str(value)
            else:
                value_canonical = str(value)

            value_hash = hashlib.sha256(value_canonical.encode()).hexdigest()[:8]

            # Check if this value is important based on learned semantics
            if field_sem.is_value_important(value_hash):
                important_indices.append(i)
                break  # Only need to mark item once

    return important_indices


@dataclass
class CrushabilityAnalysis:
    """Analysis of whether an array is safe to crush.

    The key insight: if we don't have a reliable SIGNAL to determine
    which items are important, we should NOT crush at all.

    Signals include:
    - Score/rank fields (search results)
    - Error keywords (logs)
    - Numeric anomalies (metrics)
    - Low uniqueness (repetitive data where sampling is representative)

    High variability + No signal = DON'T CRUSH
    """

    crushable: bool
    confidence: float  # 0.0 to 1.0
    reason: str
    signals_present: list[str] = field(default_factory=list)
    signals_absent: list[str] = field(default_factory=list)

    # Detailed metrics
    has_id_field: bool = False
    id_uniqueness: float = 0.0
    avg_string_uniqueness: float = 0.0
    has_score_field: bool = False
    error_item_count: int = 0
    anomaly_count: int = 0


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
    crushability: CrushabilityAnalysis | None = None  # Whether it's safe to crush


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
class CrushResult:
    """Result from SmartCrusher.crush() method.

    Used by ContentRouter when routing JSON arrays to SmartCrusher.
    """

    compressed: str
    original: str
    was_modified: bool
    strategy: str = "passthrough"


@dataclass
class SmartCrusherConfig:
    """Configuration for smart crusher.

    SCHEMA-PRESERVING: Output contains only items from the original array.
    No wrappers, no generated text, no metadata keys.
    """

    enabled: bool = True
    min_items_to_analyze: int = 5  # Don't analyze tiny arrays
    min_tokens_to_crush: int = 200  # Only crush if > N tokens
    variance_threshold: float = 2.0  # Std devs for change point detection
    uniqueness_threshold: float = 0.1  # Below this = nearly constant
    similarity_threshold: float = 0.8  # For clustering similar strings
    max_items_after_crush: int = 15  # Target max items in output
    preserve_change_points: bool = True
    factor_out_constants: bool = False  # Disabled - preserves original schema
    include_summaries: bool = False  # Disabled - no generated text

    # Feedback loop integration
    use_feedback_hints: bool = True  # Use learned patterns to adjust compression

    # LOW FIX #21: Make TOIN confidence threshold configurable
    # Minimum confidence required to apply TOIN recommendations
    toin_confidence_threshold: float = 0.5

    # Content deduplication - prevents wasting slots on identical items
    dedup_identical_items: bool = True

    # Adaptive K boundary allocation (fraction of total K for first/last items)
    first_fraction: float = 0.3  # 30% of K from start of array
    last_fraction: float = 0.15  # 15% of K from end of array


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
        all_keys: set[str] = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        for key in all_keys:
            field_stats[key] = self._analyze_field(key, items)

        # Detect pattern
        pattern = self._detect_pattern(field_stats, items)

        # Extract constants
        constant_fields = {k: v.constant_value for k, v in field_stats.items() if v.is_constant}

        # CRITICAL: Analyze crushability BEFORE selecting strategy
        crushability = self.analyze_crushability(items, field_stats)

        # Select strategy (respects crushability)
        strategy = self._select_strategy(field_stats, pattern, len(items), crushability)

        # Estimate reduction (0 if not crushable)
        if strategy == CompressionStrategy.SKIP:
            reduction = 0.0
        else:
            reduction = self._estimate_reduction(field_stats, strategy, len(items))

        return ArrayAnalysis(
            item_count=len(items),
            field_stats=field_stats,
            detected_pattern=pattern,
            recommended_strategy=strategy,
            constant_fields=constant_fields,
            estimated_reduction=reduction,
            crushability=crushability,
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
            # Filter out NaN and Infinity which break statistics functions
            nums = [v for v in non_null_values if isinstance(v, (int, float)) and math.isfinite(v)]
            if nums:
                try:
                    stats.min_val = min(nums)
                    stats.max_val = max(nums)
                    stats.mean_val = statistics.mean(nums)
                    stats.variance = statistics.variance(nums) if len(nums) > 1 else 0
                    stats.change_points = self._detect_change_points(nums)
                except (OverflowError, ValueError):
                    # Extreme values that overflow - skip detailed statistics
                    stats.min_val = None
                    stats.max_val = None
                    stats.mean_val = None
                    stats.variance = 0
                    stats.change_points = []

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
            before_mean = statistics.mean(values[i - window : i])
            after_mean = statistics.mean(values[i : i + window])

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
        """Detect the data pattern using STATISTICAL analysis (no hardcoded field names).

        Pattern detection:
        - TIME_SERIES: Has a temporal field (detected by value format) + numeric variance
        - LOGS: Has a high-cardinality string field + low-cardinality categorical field
        - SEARCH_RESULTS: Has a score-like field (bounded numeric, possibly sorted)
        - GENERIC: Default
        """
        # Check for time series pattern using STRUCTURAL detection
        has_timestamp = self._detect_temporal_field(field_stats, items)

        numeric_fields = [k for k, v in field_stats.items() if v.field_type == "numeric"]
        has_numeric_with_variance = any(
            (field_stats[k].variance is not None and (field_stats[k].variance or 0) > 0)
            for k in numeric_fields
        )

        if has_timestamp and has_numeric_with_variance:
            return "time_series"

        # Check for logs pattern using STATISTICAL detection
        # Logs have: high-cardinality string (message) + low-cardinality categorical (level)
        has_message_like = False
        has_level_like = False

        for _name, stats in field_stats.items():
            if stats.field_type == "string":
                # High-cardinality string = likely message field
                if stats.unique_ratio > 0.5 and stats.avg_length and stats.avg_length > 20:
                    has_message_like = True
                # Low-cardinality string = likely level/status field
                elif stats.unique_ratio < 0.1 and 2 <= stats.unique_count <= 10:
                    has_level_like = True

        if has_message_like and has_level_like:
            return "logs"

        # Check for search results pattern using STATISTICAL score detection
        for _name, stats in field_stats.items():
            is_score, confidence = _detect_score_field_statistically(stats, items)
            if is_score and confidence >= 0.5:
                return "search_results"

        return "generic"

    def _detect_temporal_field(self, field_stats: dict[str, FieldStats], items: list[dict]) -> bool:
        """Detect if any field contains temporal values (dates/timestamps).

        Uses STRUCTURAL detection based on value format, not field names.
        """
        # Check string fields for ISO 8601 patterns
        iso_datetime_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
        iso_date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        for name, stats in field_stats.items():
            if stats.field_type == "string":
                # Sample some values
                sample_values = [
                    item.get(name) for item in items[:10] if isinstance(item.get(name), str)
                ]
                if sample_values:
                    # Check if values look like dates/datetimes
                    iso_count = sum(
                        1
                        for v in sample_values
                        if v is not None
                        and (iso_datetime_pattern.match(v) or iso_date_pattern.match(v))
                    )
                    if iso_count / len(sample_values) > 0.5:
                        return True

            # Check numeric fields for Unix timestamp range
            elif stats.field_type == "numeric":
                if stats.min_val and stats.max_val:
                    # Unix timestamps (seconds): 1000000000 to 2000000000 (roughly 2001-2033)
                    # Unix timestamps (milliseconds): 1000000000000 to 2000000000000
                    is_unix_seconds = 1000000000 <= stats.min_val <= 2000000000
                    is_unix_millis = 1000000000000 <= stats.min_val <= 2000000000000
                    if is_unix_seconds or is_unix_millis:
                        return True

        return False

    def analyze_crushability(
        self,
        items: list[dict],
        field_stats: dict[str, FieldStats],
    ) -> CrushabilityAnalysis:
        """Analyze whether it's SAFE to crush this array.

        The key insight: High variability + No importance signal = DON'T CRUSH.

        We use STATISTICAL detection (no hardcoded field names):
        1. ID fields detected by uniqueness + sequential/UUID/entropy patterns
        2. Score fields detected by bounded range + sorted order
        3. Error items detected by structural outliers (rare fields, rare status values)
        4. Numeric anomalies (importance signal)
        5. Low uniqueness (safe to sample)

        Returns:
            CrushabilityAnalysis with decision and reasoning.
        """
        signals_present: list[str] = []
        signals_absent: list[str] = []

        # 1. Detect ID field STATISTICALLY (no hardcoded field names)
        id_field_name = None
        id_uniqueness = 0.0
        id_confidence = 0.0
        for name, stats in field_stats.items():
            values = [item.get(name) for item in items if isinstance(item, dict)]
            is_id, confidence = _detect_id_field_statistically(stats, values)
            if is_id and confidence > id_confidence:
                id_field_name = name
                id_uniqueness = stats.unique_ratio
                id_confidence = confidence

        has_id_field = id_field_name is not None and id_confidence >= 0.7

        # 2. Detect score/rank field STATISTICALLY (no hardcoded field names)
        has_score_field = False
        for name, stats in field_stats.items():
            is_score, confidence = _detect_score_field_statistically(stats, items)
            if is_score:
                has_score_field = True
                signals_present.append(f"score_field:{name}(conf={confidence:.2f})")
                break
        if not has_score_field:
            signals_absent.append("score_field")

        # 3. Detect error items via STRUCTURAL OUTLIERS (no hardcoded keywords)
        outlier_indices = _detect_structural_outliers(items)
        structural_outlier_count = len(outlier_indices)

        if structural_outlier_count > 0:
            signals_present.append(f"structural_outliers:{structural_outlier_count}")
        else:
            signals_absent.append("structural_outliers")

        # 3b. Also detect errors via keywords in content (for log/message-style data)
        # This catches errors that are in the content but not structural outliers
        # (e.g., Slack messages where error is in the text field)
        error_keyword_indices = _detect_error_items_for_preservation(items)
        keyword_error_count = len(error_keyword_indices)

        if keyword_error_count > 0 and structural_outlier_count == 0:
            signals_present.append(f"error_keywords:{keyword_error_count}")

        # Combined error count for crushability analysis
        error_count = max(structural_outlier_count, keyword_error_count)

        # 4. Count numeric anomalies (importance signal)
        anomaly_count = 0
        anomaly_indices: set[int] = set()
        for stats in field_stats.values():
            if stats.field_type == "numeric" and stats.mean_val is not None and stats.variance:
                std = stats.variance**0.5
                if std > 0:
                    threshold = self.config.variance_threshold * std
                    for i, item in enumerate(items):
                        val = item.get(stats.name)
                        if isinstance(val, (int, float)):
                            if abs(val - stats.mean_val) > threshold:
                                anomaly_indices.add(i)

        anomaly_count = len(anomaly_indices)
        if anomaly_count > 0:
            signals_present.append(f"anomalies:{anomaly_count}")
        else:
            signals_absent.append("anomalies")

        # 5. Compute average string uniqueness (EXCLUDING statistically-detected ID fields)
        string_stats = [
            s for s in field_stats.values() if s.field_type == "string" and s.name != id_field_name
        ]
        avg_string_uniqueness = (
            statistics.mean(s.unique_ratio for s in string_stats) if string_stats else 0.0
        )

        # Compute uniqueness of non-ID numeric fields
        non_id_numeric_stats = [
            s for s in field_stats.values() if s.field_type == "numeric" and s.name != id_field_name
        ]
        avg_non_id_numeric_uniqueness = (
            statistics.mean(s.unique_ratio for s in non_id_numeric_stats)
            if non_id_numeric_stats
            else 0.0
        )

        # Combined uniqueness metric (including ID fields)
        max_uniqueness = max(avg_string_uniqueness, id_uniqueness, 0.0)

        # Non-ID content uniqueness (for detecting repetitive content with unique IDs)
        non_id_content_uniqueness = max(avg_string_uniqueness, avg_non_id_numeric_uniqueness)

        # 6. Check for change points (importance signal for time series)
        has_change_points = any(
            stats.change_points for stats in field_stats.values() if stats.field_type == "numeric"
        )
        if has_change_points:
            signals_present.append("change_points")

        # DECISION LOGIC
        has_any_signal = len(signals_present) > 0

        # Case 0: Repetitive content with unique IDs
        # If all non-ID fields are nearly constant, data is safe to sample
        # even if there's a unique ID field (e.g., status="success" for all items)
        if non_id_content_uniqueness < 0.1 and has_id_field:
            signals_present.append("repetitive_content")
            return CrushabilityAnalysis(
                crushable=True,
                confidence=0.85,
                reason="repetitive_content_with_ids",
                signals_present=signals_present,
                signals_absent=signals_absent,
                has_id_field=has_id_field,
                id_uniqueness=id_uniqueness,
                avg_string_uniqueness=avg_string_uniqueness,
                has_score_field=has_score_field,
                error_item_count=error_count,
                anomaly_count=anomaly_count,
            )

        # Case 1: Low uniqueness - safe to sample (data is repetitive)
        if max_uniqueness < 0.3:
            return CrushabilityAnalysis(
                crushable=True,
                confidence=0.9,
                reason="low_uniqueness_safe_to_sample",
                signals_present=signals_present,
                signals_absent=signals_absent,
                has_id_field=has_id_field,
                id_uniqueness=id_uniqueness,
                avg_string_uniqueness=avg_string_uniqueness,
                has_score_field=has_score_field,
                error_item_count=error_count,
                anomaly_count=anomaly_count,
            )

        # Case 2: High uniqueness + ID field + NO signal = DON'T CRUSH
        # This is the critical case: DB results, file listings, user lists
        if has_id_field and max_uniqueness > 0.8 and not has_any_signal:
            return CrushabilityAnalysis(
                crushable=False,
                confidence=0.85,
                reason="unique_entities_no_signal",
                signals_present=signals_present,
                signals_absent=signals_absent,
                has_id_field=has_id_field,
                id_uniqueness=id_uniqueness,
                avg_string_uniqueness=avg_string_uniqueness,
                has_score_field=has_score_field,
                error_item_count=error_count,
                anomaly_count=anomaly_count,
            )

        # Case 3: High uniqueness + has signal = CRUSH using signal
        if max_uniqueness > 0.8 and has_any_signal:
            return CrushabilityAnalysis(
                crushable=True,
                confidence=0.7,
                reason="unique_entities_with_signal",
                signals_present=signals_present,
                signals_absent=signals_absent,
                has_id_field=has_id_field,
                id_uniqueness=id_uniqueness,
                avg_string_uniqueness=avg_string_uniqueness,
                has_score_field=has_score_field,
                error_item_count=error_count,
                anomaly_count=anomaly_count,
            )

        # Case 4: Medium uniqueness + no signal = be cautious, don't crush
        if not has_any_signal:
            return CrushabilityAnalysis(
                crushable=False,
                confidence=0.6,
                reason="medium_uniqueness_no_signal",
                signals_present=signals_present,
                signals_absent=signals_absent,
                has_id_field=has_id_field,
                id_uniqueness=id_uniqueness,
                avg_string_uniqueness=avg_string_uniqueness,
                has_score_field=has_score_field,
                error_item_count=error_count,
                anomaly_count=anomaly_count,
            )

        # Case 5: Medium uniqueness + has signal = crush with caution
        return CrushabilityAnalysis(
            crushable=True,
            confidence=0.5,
            reason="medium_uniqueness_with_signal",
            signals_present=signals_present,
            signals_absent=signals_absent,
            has_id_field=has_id_field,
            id_uniqueness=id_uniqueness,
            avg_string_uniqueness=avg_string_uniqueness,
            has_score_field=has_score_field,
            error_item_count=error_count,
            anomaly_count=anomaly_count,
        )

    def _select_strategy(
        self,
        field_stats: dict[str, FieldStats],
        pattern: str,
        item_count: int,
        crushability: CrushabilityAnalysis | None = None,
    ) -> CompressionStrategy:
        """Select optimal compression strategy based on analysis."""
        if item_count < self.config.min_items_to_analyze:
            return CompressionStrategy.NONE

        # CRITICAL: Check crushability first
        if crushability is not None and not crushability.crushable:
            return CompressionStrategy.SKIP

        if pattern == "time_series":
            # Check if there are change points worth preserving
            numeric_fields = [v for v in field_stats.values() if v.field_type == "numeric"]
            has_change_points = any(f.change_points for f in numeric_fields)
            if has_change_points:
                return CompressionStrategy.TIME_SERIES

        if pattern == "logs":
            # Check if messages are clusterable (low-medium uniqueness)
            message_field = next(
                (v for k, v in field_stats.items() if "message" in k.lower()), None
            )
            if message_field and message_field.unique_ratio < 0.5:
                return CompressionStrategy.CLUSTER_SAMPLE

        if pattern == "search_results":
            return CompressionStrategy.TOP_N

        # Default: smart sampling
        return CompressionStrategy.SMART_SAMPLE

    def _estimate_reduction(
        self, field_stats: dict[str, FieldStats], strategy: CompressionStrategy, item_count: int
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
        ccr_config: CCRConfig | None = None,
    ):
        self.config = config or SmartCrusherConfig()
        self.analyzer = SmartAnalyzer(self.config)

        # CCR (Compress-Cache-Retrieve) configuration
        # When no ccr_config provided, default to caching enabled but markers disabled
        # This maintains backward compatibility - callers must opt-in to markers
        if ccr_config is None:
            self._ccr_config = CCRConfig(
                enabled=True,  # Still cache for potential retrieval
                inject_retrieval_marker=False,  # Don't break JSON parsing by default
            )
        else:
            self._ccr_config = ccr_config
        self._compression_store: CompressionStore | None = None

        # Feedback loop for learning compression patterns
        self._feedback: CompressionFeedback | None = None

        # CRITICAL FIX: Lock for thread-safe lazy initialization
        # Without this, multiple threads could call _get_* methods simultaneously
        # and potentially create redundant initialization calls.
        self._lazy_init_lock = threading.Lock()

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

        # Initialize AnchorSelector for dynamic position-based preservation
        anchor_config = self.config.anchor if hasattr(self.config, "anchor") else AnchorConfig()
        self._anchor_selector = AnchorSelector(anchor_config)

        # NOTE: Error detection now uses structural outlier detection (_detect_structural_outliers)
        # instead of hardcoded keywords. This scales to any data domain.

    def _map_to_anchor_pattern(self, strategy: CompressionStrategy) -> AnchorDataPattern:
        """Map SmartCrusher compression strategy to AnchorSelector data pattern.

        Args:
            strategy: The detected compression strategy.

        Returns:
            Corresponding AnchorDataPattern for anchor selection.
        """
        return {
            CompressionStrategy.TIME_SERIES: AnchorDataPattern.TIME_SERIES,
            CompressionStrategy.TOP_N: AnchorDataPattern.SEARCH_RESULTS,
            CompressionStrategy.CLUSTER_SAMPLE: AnchorDataPattern.LOGS,
            CompressionStrategy.SMART_SAMPLE: AnchorDataPattern.GENERIC,
        }.get(strategy, AnchorDataPattern.GENERIC)

    def crush(self, content: str, query: str = "", bias: float = 1.0) -> CrushResult:
        """Crush content string directly (for use by ContentRouter).

        This is a simplified interface for compressing a single content string,
        used by ContentRouter when routing JSON arrays to SmartCrusher.

        Args:
            content: JSON string content to compress.
            query: Query context for relevance-based compression.
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).

        Returns:
            CrushResult with compressed content and metadata.
        """
        compressed, was_modified, analysis_info = self._smart_crush_content(
            content, query_context=query, bias=bias
        )
        return CrushResult(
            compressed=compressed,
            original=content,
            was_modified=was_modified,
            strategy=analysis_info or "passthrough",
        )

    def _get_compression_store(self) -> CompressionStore:
        """Get the compression store for CCR (lazy initialization).

        CRITICAL FIX: Thread-safe double-checked locking pattern.
        """
        if self._compression_store is None:
            with self._lazy_init_lock:
                # Double-check after acquiring lock
                if self._compression_store is None:
                    self._compression_store = get_compression_store(
                        max_entries=self._ccr_config.store_max_entries,
                        default_ttl=self._ccr_config.store_ttl_seconds,
                    )
        return self._compression_store

    def _get_feedback(self) -> CompressionFeedback:
        """Get the feedback analyzer (lazy initialization).

        CRITICAL FIX: Thread-safe double-checked locking pattern.
        """
        if self._feedback is None:
            with self._lazy_init_lock:
                if self._feedback is None:
                    self._feedback = get_compression_feedback()
        return self._feedback

    def _get_telemetry(self) -> TelemetryCollector:
        """Get the telemetry collector (lazy initialization).

        CRITICAL FIX: Thread-safe double-checked locking pattern.
        """
        # Use getattr to avoid hasattr race condition
        if getattr(self, "_telemetry", None) is None:
            with self._lazy_init_lock:
                if getattr(self, "_telemetry", None) is None:
                    self._telemetry = get_telemetry_collector()
        return self._telemetry

    def _get_toin(self) -> ToolIntelligenceNetwork:
        """Get the TOIN instance (lazy initialization).

        CRITICAL FIX: Thread-safe double-checked locking pattern.
        """
        # Use getattr to avoid hasattr race condition
        if getattr(self, "_toin", None) is None:
            with self._lazy_init_lock:
                if getattr(self, "_toin", None) is None:
                    self._toin = get_toin()
        return self._toin

    def _record_telemetry(
        self,
        items: list[dict],
        result: list,
        analysis: ArrayAnalysis,
        plan: CompressionPlan,
        tool_name: str | None = None,
    ) -> None:
        """Record compression telemetry for the data flywheel.

        This collects anonymized statistics about compression patterns to
        enable cross-user learning and improve compression over time.

        Privacy guarantees:
        - No actual data values are stored
        - Tool names can be hashed
        - Only structural patterns are captured
        """
        try:
            telemetry = self._get_telemetry()

            # Calculate what was kept
            kept_first_n = sum(1 for i in plan.keep_indices if i < 3)
            kept_last_n = sum(1 for i in plan.keep_indices if i >= len(items) - 2)

            # Count error items in result
            error_indices = set(_detect_error_items_for_preservation(items))
            kept_errors = sum(1 for i in plan.keep_indices if i in error_indices)

            # Count anomalies (approximate from change points)
            anomaly_count = 0
            for stats in analysis.field_stats.values():
                if stats.change_points:
                    anomaly_count += len(stats.change_points)
            kept_anomalies = min(anomaly_count, len(plan.keep_indices))

            # Crushability info
            crushability_score = None
            crushability_reason = None
            if analysis.crushability:
                crushability_score = analysis.crushability.confidence
                crushability_reason = analysis.crushability.reason

            # Record the event
            telemetry.record_compression(
                items=items[:100],  # Sample for structure analysis
                original_count=len(items),
                compressed_count=len(result),
                original_tokens=0,  # Not available here
                compressed_tokens=0,  # Not available here
                strategy=analysis.recommended_strategy.value,
                tool_name=tool_name,
                strategy_reason=analysis.detected_pattern,
                crushability_score=crushability_score,
                crushability_reason=crushability_reason,
                kept_first_n=kept_first_n,
                kept_last_n=kept_last_n,
                kept_errors=kept_errors,
                kept_anomalies=kept_anomalies,
                kept_by_relevance=0,  # Would need to track separately
                kept_by_score=0,  # Would need to track separately
            )
        except Exception:
            # Telemetry should never break compression
            pass

    def _deduplicate_indices_by_content(
        self,
        keep_indices: set[int],
        items: list[dict],
    ) -> set[int]:
        """Deduplicate indices by content hash, preferring lower indices.

        When multiple indices contain identical content, only the lowest index
        is kept. This maximizes information density in the compressed output.

        Enterprise considerations:
        - Thread-safe: No shared state modified
        - Performance: O(n) single pass with hash map
        - Memory: O(n) for hash storage (16-char hashes)
        - Fault-tolerant: Serialization errors preserve the item

        Args:
            keep_indices: Set of indices to deduplicate.
            items: The items array for content lookup.

        Returns:
            Deduplicated set of indices (may be smaller than input).
        """
        if not keep_indices:
            return keep_indices

        # Track first occurrence of each content hash
        # Using dict[hash -> lowest_index] for O(1) lookup
        seen_hashes: dict[str, int] = {}
        duplicates_removed = 0

        # Process in sorted order to ensure deterministic "lowest index wins"
        for idx in sorted(keep_indices):
            # Bounds check
            if idx < 0 or idx >= len(items):
                continue

            item = items[idx]

            # Compute content hash
            try:
                if isinstance(item, dict):
                    # Canonical JSON serialization for consistent hashing
                    content = json.dumps(item, sort_keys=True, default=str)
                else:
                    # Non-dict items: use string representation
                    content = str(item)
                item_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            except (TypeError, ValueError, RecursionError) as e:
                # Serialization failed - keep the item (fail-safe)
                logger.debug("Dedup hash failed for item at index %d: %s. Keeping item.", idx, e)
                # Use index as unique "hash" to ensure item is kept
                item_hash = f"__idx_{idx}__"

            # First occurrence wins
            if item_hash not in seen_hashes:
                seen_hashes[item_hash] = idx
            else:
                duplicates_removed += 1

        # Log deduplication stats for observability
        if duplicates_removed > 0:
            logger.debug(
                "Content deduplication removed %d duplicate items from %d candidates "
                "(%.1f%% reduction). Unique items: %d",
                duplicates_removed,
                len(keep_indices),
                100 * duplicates_removed / len(keep_indices),
                len(seen_hashes),
            )

        return set(seen_hashes.values())

    def _fill_remaining_slots(
        self,
        keep_indices: set[int],
        items: list[dict],
        n: int,
        effective_max: int,
    ) -> set[int]:
        """Fill remaining slots with unique items when under budget.

        When deduplication reduces keep_indices below effective_max, this method
        fills the remaining slots with diverse items from the array. It:
        1. Computes content hashes for already-kept items
        2. Scans array for items with unique content not yet kept
        3. Distributes new items evenly across the array for coverage

        Enterprise considerations:
        - Thread-safe: No shared state modified
        - Performance: O(n) for hash computation, O(n) for filling
        - Memory: O(k) where k = len(keep_indices) for hash storage
        - Deterministic: Same input always produces same output

        Args:
            keep_indices: Current set of indices to keep (already deduplicated).
            items: The full items array.
            n: Total number of items.
            effective_max: Target maximum items.

        Returns:
            Updated set of indices, filled up to effective_max with unique items.
        """
        remaining_slots = effective_max - len(keep_indices)
        if remaining_slots <= 0:
            return keep_indices

        # Build set of content hashes for items we're already keeping
        seen_hashes: set[str] = set()
        for idx in keep_indices:
            if 0 <= idx < n:
                item = items[idx]
                try:
                    if isinstance(item, dict):
                        content = json.dumps(item, sort_keys=True, default=str)
                    else:
                        content = str(item)
                    seen_hashes.add(hashlib.sha256(content.encode()).hexdigest()[:16])
                except (TypeError, ValueError, RecursionError):
                    pass  # Skip hash computation failures

        # Find candidate indices not in keep_indices
        candidates = [i for i in range(n) if i not in keep_indices]
        if not candidates:
            return keep_indices

        # Distribute selection evenly across the array for coverage
        # Use stride-based sampling to avoid clustering
        result = keep_indices.copy()
        added = 0

        # Calculate step size for even distribution
        step = max(1, len(candidates) // (remaining_slots + 1))

        # First pass: evenly distributed unique items
        for start_offset in range(step):
            if added >= remaining_slots:
                break
            for i in range(start_offset, len(candidates), step):
                if added >= remaining_slots:
                    break
                idx = candidates[i]
                item = items[idx]

                # Check if this item's content is unique
                try:
                    if isinstance(item, dict):
                        content = json.dumps(item, sort_keys=True, default=str)
                    else:
                        content = str(item)
                    item_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                except (TypeError, ValueError, RecursionError):
                    # Hash failure - use index as unique hash (fail-safe)
                    item_hash = f"__idx_{idx}__"

                if item_hash not in seen_hashes:
                    result.add(idx)
                    seen_hashes.add(item_hash)
                    added += 1

        if added > 0:
            logger.debug(
                "Filled %d remaining slots (from %d to %d items) after deduplication",
                added,
                len(keep_indices),
                len(result),
            )

        return result

    def _prioritize_indices(
        self,
        keep_indices: set[int],
        items: list[dict],
        n: int,
        analysis: ArrayAnalysis | None = None,
        max_items: int | None = None,
        field_semantics: dict[str, FieldSemantics] | None = None,
    ) -> set[int]:
        """Prioritize indices when we exceed max_items, ALWAYS keeping critical items.

        Priority order:
        1. ALL error items (non-negotiable) - items with error keywords
        2. ALL structural outliers (non-negotiable) - items with rare fields/status values
        3. ALL numeric anomalies (non-negotiable) - e.g., unusual values like 999999
        4. ALL items with important values (learned) - TOIN field semantics
        5. First 3 items (context)
        6. Last 2 items (context)
        7. Other important items by index order

        Uses BOTH keyword detection (for preservation guarantee) AND statistical detection,
        PLUS learned field semantics from TOIN for zero-latency signal detection.

        HIGH FIX: Note that this function may return MORE items than effective_max
        when critical items (errors, outliers, anomalies) exceed the limit. This is
        intentional to preserve the quality guarantee. A warning is logged when this
        happens to help diagnose cases where compression is less effective than expected.

        Args:
            keep_indices: Initial set of indices to keep.
            items: The items being compressed.
            n: Total number of items.
            analysis: Optional analysis results for anomaly detection.
            max_items: Thread-safe max items limit (defaults to config value).
            field_semantics: Optional learned field semantics from TOIN.

        Returns:
            Set of indices to keep (may exceed max_items if critical items require it).
        """
        # Use provided max_items or fall back to config
        effective_max = max_items if max_items is not None else self.config.max_items_after_crush

        # === ENTERPRISE FIX: Content-based deduplication ===
        # Multiple preservation mechanisms (anchors, anomalies, outliers, etc.) can add
        # the same item multiple times by index. More critically, different INDICES can
        # contain IDENTICAL content (e.g., 10 identical status messages at indices 0-9).
        # Without deduplication, we waste slots on redundant information.
        #
        # This deduplication runs FIRST, before any other logic, to ensure:
        # 1. Budget calculations work with unique items only
        # 2. Critical item detection doesn't double-count
        # 3. Final output maximizes information density
        #
        # Performance: O(n) where n = len(keep_indices), single pass with hash map
        # Memory: O(n) for hash storage, hashes are 16 chars each
        if self.config.dedup_identical_items:
            keep_indices = self._deduplicate_indices_by_content(keep_indices, items)

        # === ENTERPRISE FIX: Fill up to max_items when under budget ===
        # After deduplication, we may have fewer unique items than max_items.
        # Instead of returning a sparse result, fill remaining slots with
        # diverse items from the array. This maximizes information density.
        if len(keep_indices) < effective_max and len(keep_indices) < n:
            keep_indices = self._fill_remaining_slots(keep_indices, items, n, effective_max)

        if len(keep_indices) <= effective_max:
            return keep_indices

        # Use provided field_semantics or fall back to instance variable (set by crush())
        effective_field_semantics = field_semantics or getattr(
            self, "_current_field_semantics", None
        )

        # Identify error items using KEYWORD detection (preservation guarantee)
        # This ensures ALL error items are kept, regardless of frequency
        error_indices = set(_detect_error_items_for_preservation(items))

        # Identify structural outlier indices using STATISTICAL detection
        # (items with rare fields or rare status values)
        outlier_indices = set(_detect_structural_outliers(items))

        # Identify numeric anomalies (MUST keep ALL of them)
        anomaly_indices = set()
        if analysis and analysis.field_stats:
            for field_name, stats in analysis.field_stats.items():
                if stats.field_type == "numeric" and stats.mean_val is not None and stats.variance:
                    std = stats.variance**0.5
                    if std > 0:
                        threshold = self.config.variance_threshold * std
                        for i, item in enumerate(items):
                            val = item.get(field_name)
                            if isinstance(val, (int, float)):
                                if abs(val - stats.mean_val) > threshold:
                                    anomaly_indices.add(i)

        # === TOIN Evolution: Identify items with important values (learned) ===
        # Uses learned field semantics for zero-latency signal detection
        learned_important_indices: set[int] = set()
        if effective_field_semantics:
            learned_important_indices = set(
                _detect_items_by_learned_semantics(items, effective_field_semantics)
            )

        # Start with all critical items (these are non-negotiable)
        # Error items are ALWAYS preserved (quality guarantee)
        prioritized = error_indices | outlier_indices | anomaly_indices | learned_important_indices

        # HIGH FIX: Log warning if critical items alone exceed the limit
        # This helps diagnose why compression may be less effective than expected
        critical_count = len(prioritized)
        if critical_count > effective_max:
            logger.warning(
                "Critical items (%d) exceed max_items (%d): errors=%d outliers=%d anomalies=%d learned=%d. "
                "Quality guarantee takes precedence - keeping all critical items.",
                critical_count,
                effective_max,
                len(error_indices),
                len(outlier_indices),
                len(anomaly_indices),
                len(learned_important_indices),
            )

        # Add first/last items if we have room
        remaining_slots = effective_max - len(prioritized)
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
        """Check if data contains arrays large enough to crush.

        Accepts arrays of ANY homogeneous type (dicts, strings, numbers, etc.)
        as well as mixed-type arrays. Bool-only arrays are excluded (not useful
        to compress).
        """
        if depth > 5:
            return False

        if isinstance(data, list):
            if len(data) >= self.config.min_items_to_analyze:
                arr_type = _classify_array(data)
                if arr_type not in (ArrayType.EMPTY, ArrayType.BOOL_ARRAY):
                    return True
            for item in data[:10]:  # Check first few items
                if self._has_crushable_arrays(item, depth + 1):
                    return True

        elif isinstance(data, dict):
            # Large objects with many keys are themselves crushable
            if len(data) >= self.config.min_items_to_analyze:
                return True
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

    def _extract_context_from_messages(self, messages: list[dict[str, Any]]) -> str:
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
        self,
        content: str,
        query_context: str = "",
        tool_name: str | None = None,
        bias: float = 1.0,
    ) -> tuple[str, bool, str]:
        """
        Apply smart crushing to content.

        Handles both JSON (existing SmartCrusher logic) and plain text content
        (search results, logs, generic text) using specialized compressors.

        Args:
            content: Content to crush (JSON or plain text).
            query_context: Context string from user messages for relevance scoring.
            tool_name: Name of the tool that produced this output.
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).

        Returns:
            Tuple of (crushed_content, was_modified, analysis_info).
        """
        parsed, success = safe_json_loads(content)
        if not success:
            # Not JSON - pass through unchanged
            # Text compression utilities (SearchCompressor, LogCompressor, TextCompressor)
            # are available as standalone tools for applications to use explicitly
            return content, False, ""

        # Recursively process and crush arrays
        crushed, info, ccr_markers = self._process_value(
            parsed, query_context=query_context, tool_name=tool_name, bias=bias
        )

        result = safe_json_dumps(crushed, indent=None)
        was_modified = result != content.strip()

        # CCR: Inject retrieval markers if compression happened and CCR is enabled
        if was_modified and ccr_markers and self._ccr_config.inject_retrieval_marker:
            for marker_data in ccr_markers:
                if len(marker_data) == 4:
                    ccr_hash, original_count, compressed_count, dropped_summary = marker_data
                else:
                    ccr_hash, original_count, compressed_count = marker_data
                    dropped_summary = ""
                summary_str = f" Omitted: {dropped_summary}." if dropped_summary else ""
                # Escape { } in summary to prevent .format() errors
                safe_summary = summary_str.replace("{", "{{").replace("}", "}}")
                ttl_seconds = getattr(self._ccr_config, "store_ttl_seconds", 300)
                marker = self._ccr_config.marker_template.format(
                    original_count=original_count,
                    compressed_count=compressed_count,
                    hash=ccr_hash,
                    summary=safe_summary,
                    ttl_minutes=max(1, ttl_seconds // 60),
                )
                result += marker

        return result, was_modified, info

    def _process_value(
        self,
        value: Any,
        depth: int = 0,
        query_context: str = "",
        tool_name: str | None = None,
        bias: float = 1.0,
    ) -> tuple[Any, str, list[tuple[str, int, int]]]:
        """Recursively process a value, crushing arrays where appropriate.

        Returns:
            Tuple of (processed_value, info_string, ccr_markers).
            ccr_markers is a list of (hash, original_count, compressed_count, summary) tuples.
        """
        info_parts = []
        ccr_markers: list[tuple] = []

        if isinstance(value, list):
            if len(value) >= self.config.min_items_to_analyze:
                arr_type = _classify_array(value)

                if arr_type == ArrayType.DICT_ARRAY:
                    # Existing path — dict arrays (battle-tested, unchanged)
                    crushed, strategy, ccr_hash, dropped_summary = self._crush_array(
                        value, query_context, tool_name, bias=bias
                    )
                    info_parts.append(f"{strategy}({len(value)}->{len(crushed)})")
                    if ccr_hash:
                        ccr_markers.append((ccr_hash, len(value), len(crushed), dropped_summary))
                    return crushed, ",".join(info_parts), ccr_markers

                elif arr_type == ArrayType.STRING_ARRAY:
                    crushed, strategy = self._crush_string_array(value, bias=bias)
                    info_parts.append(f"{strategy}({len(value)}->{len(crushed)})")
                    return crushed, ",".join(info_parts), ccr_markers

                elif arr_type == ArrayType.NUMBER_ARRAY:
                    crushed, strategy = self._crush_number_array(value, bias=bias)
                    if isinstance(crushed, list):
                        info_parts.append(f"{strategy}({len(value)}->{len(crushed)})")
                    else:
                        info_parts.append(f"{strategy}({len(value)}->summary)")
                    return crushed, ",".join(info_parts), ccr_markers

                elif arr_type == ArrayType.MIXED_ARRAY:
                    crushed, strategy = self._crush_mixed_array(
                        value, query_context, tool_name, bias=bias
                    )
                    info_parts.append(f"{strategy}({len(value)}->{len(crushed)})")
                    return crushed, ",".join(info_parts), ccr_markers

                # NESTED_ARRAY, BOOL_ARRAY, EMPTY — fall through to recursive

            # Not crushable or below threshold — process items recursively
            processed = []
            for item in value:
                p_item, p_info, p_markers = self._process_value(
                    item, depth + 1, query_context, tool_name, bias=bias
                )
                processed.append(p_item)
                if p_info:
                    info_parts.append(p_info)
                ccr_markers.extend(p_markers)
            return processed, ",".join(info_parts), ccr_markers

        elif isinstance(value, dict):
            # First: recurse into values to compress nested arrays
            processed_dict: dict[str, Any] = {}
            for k, v in value.items():
                p_val, p_info, p_markers = self._process_value(
                    v, depth + 1, query_context, tool_name, bias=bias
                )
                processed_dict[k] = p_val
                if p_info:
                    info_parts.append(p_info)
                ccr_markers.extend(p_markers)

            # Second: if the object itself has many keys, compress at key level
            if len(processed_dict) >= self.config.min_items_to_analyze:
                crushed_dict, strategy = self._crush_object(processed_dict, bias=bias)
                if strategy != "object:passthrough":
                    info_parts.append(strategy)
                    return crushed_dict, ",".join(info_parts), ccr_markers

            return processed_dict, ",".join(info_parts), ccr_markers

        else:
            return value, "", []

    def _crush_array(
        self,
        items: list[dict],
        query_context: str = "",
        tool_name: str | None = None,
        bias: float = 1.0,
    ) -> tuple[list, str, str | None, str]:
        """Crush an array using statistical analysis and relevance scoring.

        IMPORTANT: If crushability analysis determines it's not safe to crush
        (high variability + no importance signal), returns original array unchanged.

        TOIN-aware: Consults the Tool Output Intelligence Network for cross-user
        learned patterns. High retrieval rate across all users → compress less.

        Feedback-aware: Uses learned patterns to adjust compression aggressiveness.
        High retrieval rate for a tool → compress less aggressively.

        Args:
            items: List of dict items to compress.
            query_context: Context string from user messages for relevance scoring.
            tool_name: Name of the tool that produced this output.
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).

        Returns:
            Tuple of (crushed_items, strategy_info, ccr_hash, dropped_summary).
            ccr_hash is the hash for retrieval if CCR is enabled, None otherwise.
            dropped_summary is a categorical summary of what was dropped.
        """
        # BOUNDARY CHECK: Use adaptive sizing instead of hardcoded limit
        # compute_optimal_k handles trivial cases (n <= 8 → keep all)
        from .adaptive_sizer import compute_optimal_k

        item_strings = [json.dumps(item, default=str) for item in items]
        adaptive_k = compute_optimal_k(
            item_strings,
            bias=bias,
            min_k=3,
            max_k=self.config.max_items_after_crush if self.config.max_items_after_crush else None,
        )

        if len(items) <= adaptive_k:
            return items, "none:adaptive_at_limit", None, ""

        # Get feedback hints if enabled
        # THREAD-SAFETY: Use a local effective_max_items instead of mutating shared config
        effective_max_items = adaptive_k
        hints_applied = False
        toin_hint_applied = False

        # Create ToolSignature for TOIN lookup
        tool_signature = ToolSignature.from_items(items)

        # TOIN: Get cross-user learned recommendations
        toin = self._get_toin()
        toin_hint = toin.get_recommendation(tool_signature, query_context)

        # Log TOIN hint details
        logger.debug(
            "TOIN hint: source=%s, confidence=%.2f, skip=%s, max_items=%d",
            toin_hint.source,
            toin_hint.confidence,
            toin_hint.skip_compression,
            toin_hint.max_items,
        )

        if toin_hint.skip_compression:
            return items, f"skip:toin({toin_hint.reason})", None, ""

        # Apply TOIN recommendations if from network or local learning
        toin_preserve_fields: list[str] = []
        toin_recommended_strategy: str | None = None
        toin_compression_level: str | None = None
        # LOW FIX #21: Use configurable threshold instead of hardcoded 0.5
        if (
            toin_hint.source in ("network", "local")
            and toin_hint.confidence >= self.config.toin_confidence_threshold
        ):
            # TOIN recommendations take precedence over local feedback
            effective_max_items = toin_hint.max_items
            toin_preserve_fields = toin_hint.preserve_fields  # Fields to never remove
            toin_hint_applied = True
            # Store strategy and compression level for later use
            if toin_hint.recommended_strategy != "default":
                toin_recommended_strategy = toin_hint.recommended_strategy
            if toin_hint.compression_level != "moderate":
                toin_compression_level = toin_hint.compression_level
            # Log that TOIN hint was applied
            logger.debug(
                "TOIN hint applied: max_items=%d, strategy=%s, compression_level=%s",
                effective_max_items,
                toin_recommended_strategy or "default",
                toin_compression_level or "moderate",
            )
        elif toin_hint.source in ("network", "local"):
            # Hint available but confidence too low
            logger.debug(
                "TOIN hint not applied: confidence %.2f < threshold %.2f",
                toin_hint.confidence,
                self.config.toin_confidence_threshold,
            )

        # === TOIN Evolution: Extract field semantics for signal detection ===
        # Store temporarily on instance for use in _prioritize_indices
        # This enables learned signal detection without changing all method signatures
        self._current_field_semantics = (
            toin_hint.field_semantics if toin_hint.field_semantics else None
        )

        # Local feedback hints (if TOIN didn't apply)
        if not toin_hint_applied and self.config.use_feedback_hints and tool_name:
            feedback = self._get_feedback()
            hints = feedback.get_compression_hints(tool_name)

            # Check if hints recommend skipping compression
            if hints.skip_compression:
                return items, f"skip:feedback({hints.reason})", None, ""

            # Adjust max_items based on feedback
            if hints.suggested_items is not None:
                effective_max_items = hints.suggested_items
                hints_applied = True

            # Use preserve_fields from local feedback (hash them for TOIN compatibility)
            # Note: CompressionFeedback stores actual field names, but _plan methods
            # expect SHA256[:8] hashes for privacy-preserving comparison
            if hints.preserve_fields:
                toin_preserve_fields = [_hash_field_name(field) for field in hints.preserve_fields]

            # Use recommended_strategy from local feedback if not already set by TOIN
            if hints.recommended_strategy and not toin_recommended_strategy:
                toin_recommended_strategy = hints.recommended_strategy

        try:
            # Analyze the array (includes crushability check)
            analysis = self.analyzer.analyze_array(items)

            # CRITICAL: If not crushable, return original array unchanged
            if analysis.recommended_strategy == CompressionStrategy.SKIP:
                reason = ""
                if analysis.crushability:
                    reason = f"skip:{analysis.crushability.reason}"
                return items, reason, None, ""

            # Apply TOIN strategy recommendation if available
            # TOIN learns which strategies work best from cross-user patterns
            if toin_recommended_strategy:
                try:
                    toin_strategy = CompressionStrategy(toin_recommended_strategy)
                    # Only override if TOIN suggests a valid non-SKIP strategy
                    if toin_strategy != CompressionStrategy.SKIP:
                        analysis.recommended_strategy = toin_strategy
                except ValueError:
                    pass  # Invalid strategy name, keep analyzer's choice

            # Apply TOIN compression level to adjust effective_max_items
            if toin_compression_level:
                if toin_compression_level == "none":
                    # Don't compress - return original
                    return items, "skip:toin_level_none", None, ""
                elif toin_compression_level == "conservative":
                    # Be conservative - keep more items
                    effective_max_items = max(effective_max_items, min(50, len(items) // 2))
                elif toin_compression_level == "aggressive":
                    # Be aggressive - keep fewer items
                    effective_max_items = min(effective_max_items, 15)

            # Create compression plan with relevance scoring
            # Pass TOIN preserve_fields so items with those fields get priority
            # Pass effective_max_items for thread-safe compression
            plan = self._create_plan(
                analysis,
                items,
                query_context,
                preserve_fields=toin_preserve_fields or None,
                effective_max_items=effective_max_items,
            )

            # Execute compression
            result = self._execute_plan(plan, items, analysis)

            # CCR: Store original content for retrieval if enabled
            ccr_hash = None
            if (
                self._ccr_config.enabled
                and len(items) >= self._ccr_config.min_items_to_cache
                and len(result) < len(items)  # Only cache if compression actually happened
            ):
                store = self._get_compression_store()
                original_json = json.dumps(items, default=str)
                compressed_json = json.dumps(result, default=str)

                ccr_hash = store.store(
                    original=original_json,
                    compressed=compressed_json,
                    original_item_count=len(items),
                    compressed_item_count=len(result),
                    tool_name=tool_name,
                    query_context=query_context,
                    # CRITICAL: Pass the tool_signature_hash so retrieval events
                    # can be correlated with compression events in TOIN
                    tool_signature_hash=tool_signature.structure_hash,
                    compression_strategy=analysis.recommended_strategy.value,
                )

                # Record compression event for feedback loop
                if self.config.use_feedback_hints and tool_name:
                    feedback = self._get_feedback()
                    feedback.record_compression(
                        tool_name=tool_name,
                        original_count=len(items),
                        compressed_count=len(result),
                        strategy=analysis.recommended_strategy.value,
                        tool_signature_hash=tool_signature.structure_hash,
                    )

            # Record telemetry for data flywheel
            self._record_telemetry(
                items=items,
                result=result,
                analysis=analysis,
                plan=plan,
                tool_name=tool_name,
            )

            # TOIN: Record compression event for cross-user learning
            try:
                # Calculate token counts (approximate)
                original_tokens = len(json.dumps(items, default=str)) // 4
                compressed_tokens = len(json.dumps(result, default=str)) // 4

                toin.record_compression(
                    tool_signature=tool_signature,
                    original_count=len(items),
                    compressed_count=len(result),
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    strategy=analysis.recommended_strategy.value,
                    query_context=query_context,
                    items=items,  # Pass items for field-level semantic learning
                )
            except Exception:
                # TOIN should never break compression
                pass

            strategy_info = analysis.recommended_strategy.value
            if toin_hint_applied:
                toin_parts = [f"items={toin_hint.max_items}", f"conf={toin_hint.confidence:.2f}"]
                if toin_recommended_strategy:
                    toin_parts.append(f"strategy={toin_recommended_strategy}")
                if toin_compression_level and toin_compression_level != "moderate":
                    toin_parts.append(f"level={toin_compression_level}")
                strategy_info += f"(toin:{','.join(toin_parts)})"
            elif hints_applied:
                strategy_info += f"(feedback:{effective_max_items})"

            # Generate categorical summary of dropped items (use indices, not identity)
            from .compression_summary import summarize_dropped_items

            dropped_summary = summarize_dropped_items(
                items,
                result,
                kept_indices=set(plan.keep_indices),
            )

            # Clean up temporary instance variable
            self._current_field_semantics = None
            return result, strategy_info, ccr_hash, dropped_summary

        except Exception:
            # Clean up temporary instance variable
            self._current_field_semantics = None
            # Re-raise any exceptions (removed finally block since we no longer mutate config)
            raise

    # =================================================================
    # Universal JSON type handlers (string, number, mixed arrays)
    # =================================================================

    def _compute_k_split(self, items: list, bias: float = 1.0) -> tuple[int, int, int, int]:
        """Compute adaptive K split into first/last/importance slots.

        Uses the existing Kneedle-based adaptive_sizer for K_total, then
        splits according to configurable first_fraction / last_fraction.

        Returns:
            (k_total, k_first, k_last, k_importance)
        """
        from .adaptive_sizer import compute_optimal_k

        item_strings = [json.dumps(item, default=str) for item in items]
        k_total = compute_optimal_k(
            item_strings,
            bias=bias,
            min_k=3,
            max_k=self.config.max_items_after_crush or None,
        )
        k_first = max(1, round(k_total * self.config.first_fraction))
        k_last = max(1, round(k_total * self.config.last_fraction))
        k_importance = max(0, k_total - k_first - k_last)
        return k_total, k_first, k_last, k_importance

    def _crush_string_array(
        self,
        items: list[str],
        bias: float = 1.0,
    ) -> tuple[list[str], str]:
        """Crush an array of strings using dedup + adaptive sampling.

        Strategy:
        1. Compute adaptive K via Kneedle algorithm
        2. Always keep: error-containing strings, first K, last K
        3. Deduplicate exact matches
        4. Fill remaining budget with diverse samples (stride-based)

        Returns:
            (crushed_items, strategy_string)
        """
        n = len(items)
        if n <= 8:
            return items, "string:passthrough"

        k_total, k_first, k_last, k_importance = self._compute_k_split(items, bias)

        # Mandatory: error-containing strings (never dropped)
        error_indices: set[int] = set()
        for i, s in enumerate(items):
            s_lower = s.lower()
            for keyword in _ERROR_KEYWORDS_FOR_PRESERVATION:
                if keyword in s_lower:
                    error_indices.add(i)
                    break

        # Mandatory: strings with abnormal length (anomalies)
        lengths = [len(s) for s in items]
        if len(lengths) > 1:
            mean_len = statistics.mean(lengths)
            std_len = statistics.stdev(lengths)
            anomaly_indices = {
                i
                for i, length in enumerate(lengths)
                if std_len > 0 and abs(length - mean_len) > self.config.variance_threshold * std_len
            }
        else:
            anomaly_indices = set[int]()

        # Boundary: first K, last K
        first_indices = set(range(min(k_first, n)))
        last_indices = set(range(max(0, n - k_last), n))

        # Combine mandatory + boundary
        keep_indices = error_indices | anomaly_indices | first_indices | last_indices

        # Dedup: among remaining candidates, skip exact duplicates
        seen_strings: set[str] = set()
        dedup_count = 0
        for i in sorted(keep_indices):
            seen_strings.add(items[i])

        # Fill remaining budget with diverse stride-based samples
        remaining_budget = max(0, k_total - len(keep_indices))
        if remaining_budget > 0:
            stride = max(1, (n - 1) // (remaining_budget + 1))
            for i in range(0, n, stride):
                if len(keep_indices) >= k_total + len(error_indices) + len(anomaly_indices):
                    break
                if i not in keep_indices:
                    if items[i] not in seen_strings:
                        keep_indices.add(i)
                        seen_strings.add(items[i])
                    else:
                        dedup_count += 1

        # Build output in original order
        result = [items[i] for i in sorted(keep_indices)]

        strategy = f"string:adaptive({n}->{len(result)}"
        if dedup_count:
            strategy += f",dedup={dedup_count}"
        if error_indices:
            strategy += f",errors={len(error_indices)}"
        strategy += ")"

        return result, strategy

    def _crush_number_array(
        self,
        items: list[int | float],
        bias: float = 1.0,
    ) -> tuple[list, str]:
        """Crush an array of numbers using statistical summary + outlier preservation.

        Strategy:
        1. Compute descriptive statistics (min, max, mean, median, stddev, percentiles)
        2. Detect outliers (> variance_threshold σ from mean)
        3. Detect change points (sudden shifts in running mean)
        4. Keep: first K, last K, all outliers, change points
        5. Return kept values with a prepended stats summary string

        Returns:
            (crushed_items, strategy_string) where crushed_items is a list
            starting with a summary string followed by representative values.
        """
        n = len(items)
        if n <= 8:
            return items, "number:passthrough"

        # Filter out non-finite values for statistics
        finite = [x for x in items if isinstance(x, (int, float)) and math.isfinite(x)]
        if not finite:
            return items, "number:no_finite"

        k_total, k_first, k_last, k_importance = self._compute_k_split(items, bias)

        # Statistics
        mean_val = statistics.mean(finite)
        median_val = statistics.median(finite)
        std_val = statistics.stdev(finite) if len(finite) > 1 else 0.0
        sorted_finite = sorted(finite)
        p25 = sorted_finite[len(sorted_finite) // 4] if len(sorted_finite) >= 4 else min(finite)
        p75 = sorted_finite[3 * len(sorted_finite) // 4] if len(sorted_finite) >= 4 else max(finite)

        # Outliers (> variance_threshold σ from mean)
        outlier_indices: set[int] = set()
        if std_val > 0:
            for i, val in enumerate(items):
                if isinstance(val, (int, float)) and math.isfinite(val):
                    if abs(val - mean_val) > self.config.variance_threshold * std_val:
                        outlier_indices.add(i)

        # Change points (detect sudden shifts using running difference)
        change_indices: set[int] = set()
        if self.config.preserve_change_points and n > 10:
            window = 5
            for i in range(window, n - window):
                left = [
                    items[j]
                    for j in range(i - window, i)
                    if isinstance(items[j], (int, float)) and math.isfinite(items[j])
                ]
                right = [
                    items[j]
                    for j in range(i, i + window)
                    if isinstance(items[j], (int, float)) and math.isfinite(items[j])
                ]
                if left and right:
                    left_mean = statistics.mean(left)
                    right_mean = statistics.mean(right)
                    if (
                        std_val > 0
                        and abs(right_mean - left_mean) > self.config.variance_threshold * std_val
                    ):
                        change_indices.add(i)

        # Boundary: first K, last K
        first_indices = set(range(min(k_first, n)))
        last_indices = set(range(max(0, n - k_last), n))

        # Combine all
        keep_indices = outlier_indices | change_indices | first_indices | last_indices

        # Fill remaining budget with stride-based samples
        remaining_budget = max(0, k_total - len(keep_indices))
        if remaining_budget > 0:
            stride = max(1, (n - 1) // (remaining_budget + 1))
            for i in range(0, n, stride):
                if len(keep_indices) >= k_total + len(outlier_indices):
                    break
                if i not in keep_indices:
                    keep_indices.add(i)

        # Build output: summary string + kept values in original order
        stats_summary = (
            f"[{n} numbers: min={min(finite)}, max={max(finite)}, "
            f"mean={mean_val:.4g}, median={median_val:.4g}, "
            f"stddev={std_val:.4g}, p25={p25:.4g}, p75={p75:.4g}"
        )
        if outlier_indices:
            stats_summary += f", outliers={len(outlier_indices)}"
        if change_indices:
            stats_summary += f", change_points={len(change_indices)}"
        stats_summary += "]"

        kept_values = [items[i] for i in sorted(keep_indices)]
        result: list = [stats_summary] + kept_values

        strategy = f"number:adaptive({n}->{len(kept_values)}"
        if outlier_indices:
            strategy += f",outliers={len(outlier_indices)}"
        strategy += ")"

        return result, strategy

    def _crush_mixed_array(
        self,
        items: list,
        query_context: str = "",
        tool_name: str | None = None,
        bias: float = 1.0,
    ) -> tuple[list, str]:
        """Crush a mixed-type array by grouping items by type and compressing each group.

        Strategy:
        1. Group items by type (dict, str, number, list, None, bool)
        2. For each group with >= min_items_to_analyze items: compress with appropriate handler
        3. For small groups: keep all items
        4. Reassemble in original order

        Returns:
            (crushed_items, strategy_string)
        """
        n = len(items)
        if n <= 8:
            return items, "mixed:passthrough"

        # Group items by type, tracking original indices
        groups: dict[str, list[tuple[int, Any]]] = {}
        for i, item in enumerate(items):
            if isinstance(item, dict):
                key = "dict"
            elif isinstance(item, str):
                key = "str"
            elif isinstance(item, bool):
                key = "bool"
            elif isinstance(item, (int, float)):
                key = "number"
            elif isinstance(item, list):
                key = "list"
            elif item is None:
                key = "none"
            else:
                key = "other"
            groups.setdefault(key, []).append((i, item))

        # Compress each group independently
        keep_indices: set[int] = set()
        strategy_parts: list[str] = []

        for type_key, group_items in groups.items():
            indices = [idx for idx, _ in group_items]
            values = [val for _, val in group_items]

            if len(values) < self.config.min_items_to_analyze:
                # Small group — keep all
                keep_indices.update(indices)
                continue

            if type_key == "dict":
                # Use existing dict array crusher
                crushed, strategy, _, _ = self._crush_array(
                    values, query_context, tool_name, bias=bias
                )
                crushed_set = {json.dumps(c, sort_keys=True, default=str) for c in crushed}
                for idx, val in group_items:
                    if json.dumps(val, sort_keys=True, default=str) in crushed_set:
                        keep_indices.add(idx)
                strategy_parts.append(f"dict:{len(values)}->{len(crushed)}")

            elif type_key == "str":
                crushed, strategy = self._crush_string_array(values, bias=bias)
                crushed_set = set(crushed)
                for idx, val in group_items:
                    if val in crushed_set:
                        keep_indices.add(idx)
                strategy_parts.append(f"str:{len(values)}->{len(crushed)}")

            elif type_key == "number":
                # For numbers in mixed arrays, just do adaptive sampling (no summary prefix)
                k_total, k_first, k_last, _ = self._compute_k_split(values, bias)
                first_idx = set(indices[:k_first])
                last_idx = set(indices[-k_last:])
                keep_indices.update(first_idx | last_idx)
                # Outliers
                finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
                if len(finite) > 1:
                    mean_v = statistics.mean(finite)
                    std_v = statistics.stdev(finite)
                    if std_v > 0:
                        for idx, val in group_items:
                            if isinstance(val, (int, float)) and math.isfinite(val):
                                if abs(val - mean_v) > self.config.variance_threshold * std_v:
                                    keep_indices.add(idx)
                strategy_parts.append(f"num:{len(values)}")

            else:
                # list, bool, none, other — keep all
                keep_indices.update(indices)

        # Reassemble in original order
        result = [items[i] for i in sorted(keep_indices)]

        strategy = f"mixed:adaptive({n}->{len(result)},{','.join(strategy_parts)})"
        return result, strategy

    def _crush_object(
        self,
        obj: dict[str, Any],
        bias: float = 1.0,
    ) -> tuple[dict[str, Any], str]:
        """Crush a large JSON object by selecting the most informative keys.

        Treats key-value pairs as items and applies adaptive K to select which
        keys to retain. Preserves schema — each kept key-value pair is exact
        from the original.

        Strategy:
        1. Classify each value by size (tokens) and importance
        2. Always keep: keys with small values (cheap), keys with error content
        3. Compute adaptive K on key-value representations
        4. Fill remaining budget with diverse keys (stride-based)

        Returns:
            (compressed_object, strategy_string)
        """
        n = len(obj)
        if n <= 8:
            return obj, "object:passthrough"

        # Estimate tokens per key-value pair
        kv_tokens: list[tuple[str, int]] = []
        total_tokens = 0
        for key, val in obj.items():
            val_str = json.dumps(val, default=str)
            tokens = len(val_str) // 4 + len(key) // 4 + 2  # rough estimate
            kv_tokens.append((key, tokens))
            total_tokens += tokens

        # If already small enough, passthrough
        if total_tokens < self.config.min_tokens_to_crush:
            return obj, "object:passthrough"

        # Compute adaptive K on key-value string representations
        keys = list(obj.keys())
        kv_strings = [f"{k}: {json.dumps(obj[k], default=str)}" for k in keys]

        from .adaptive_sizer import compute_optimal_k

        k_total = compute_optimal_k(
            kv_strings,
            bias=bias,
            min_k=3,
            max_k=self.config.max_items_after_crush or None,
        )

        if k_total >= n:
            return obj, "object:passthrough"

        # Classify keys by importance
        keep_keys: set[str] = set()

        # Always keep: keys with error-containing values
        for key, val in obj.items():
            val_str = json.dumps(val, default=str).lower()
            for keyword in _ERROR_KEYWORDS_FOR_PRESERVATION:
                if keyword in val_str:
                    keep_keys.add(key)
                    break

        # Always keep: keys with small values (cheap to keep)
        small_threshold = 50  # chars
        for key, tokens in kv_tokens:
            if tokens <= small_threshold // 4:
                keep_keys.add(key)

        # Boundary: first K and last K keys
        k_first = max(1, round(k_total * self.config.first_fraction))
        k_last = max(1, round(k_total * self.config.last_fraction))
        for key in keys[:k_first]:
            keep_keys.add(key)
        for key in keys[-k_last:]:
            keep_keys.add(key)

        # Fill remaining budget with stride-based diverse sampling
        remaining = max(0, k_total - len(keep_keys))
        if remaining > 0:
            stride = max(1, (n - 1) // (remaining + 1))
            for i in range(0, n, stride):
                if len(keep_keys) >= k_total + len(
                    [
                        k
                        for k in keep_keys
                        if any(
                            kw in json.dumps(obj[k], default=str).lower()
                            for kw in _ERROR_KEYWORDS_FOR_PRESERVATION
                        )
                    ]
                ):
                    break
                keep_keys.add(keys[i])

        # Build output preserving original key order
        result = {k: obj[k] for k in keys if k in keep_keys}

        strategy = f"object:adaptive({n}->{len(result)} keys)"
        return result, strategy

    def _create_plan(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        query_context: str = "",
        preserve_fields: list[str] | None = None,
        effective_max_items: int | None = None,
    ) -> CompressionPlan:
        """Create a detailed compression plan using relevance scoring.

        Args:
            analysis: The array analysis results.
            items: The items to compress.
            query_context: Context string from user messages for relevance scoring.
            preserve_fields: TOIN-learned fields that users commonly retrieve.
                Items with values in these fields get higher priority.
            effective_max_items: Thread-safe max items limit (defaults to config value).
        """
        # Use provided effective_max_items or fall back to config
        max_items = (
            effective_max_items
            if effective_max_items is not None
            else self.config.max_items_after_crush
        )

        plan = CompressionPlan(
            strategy=analysis.recommended_strategy,
            constant_fields=analysis.constant_fields if self.config.factor_out_constants else {},
        )

        # Handle SKIP - keep all items (shouldn't normally reach here)
        if analysis.recommended_strategy == CompressionStrategy.SKIP:
            plan.keep_indices = list(range(len(items)))
            return plan

        if analysis.recommended_strategy == CompressionStrategy.TIME_SERIES:
            plan = self._plan_time_series(
                analysis, items, plan, query_context, preserve_fields, max_items
            )

        elif analysis.recommended_strategy == CompressionStrategy.CLUSTER_SAMPLE:
            plan = self._plan_cluster_sample(
                analysis, items, plan, query_context, preserve_fields, max_items
            )

        elif analysis.recommended_strategy == CompressionStrategy.TOP_N:
            plan = self._plan_top_n(
                analysis, items, plan, query_context, preserve_fields, max_items
            )

        else:  # SMART_SAMPLE or NONE
            plan = self._plan_smart_sample(
                analysis, items, plan, query_context, preserve_fields, max_items
            )

        return plan

    def _plan_time_series(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        plan: CompressionPlan,
        query_context: str = "",
        preserve_fields: list[str] | None = None,
        max_items: int | None = None,
    ) -> CompressionPlan:
        """Plan compression for time series data.

        Keeps items around change points (anomalies) plus first/last items.
        Uses STATISTICAL outlier detection for important items.
        Uses RelevanceScorer for semantic matching of user queries.

        Args:
            preserve_fields: TOIN-learned fields that users commonly retrieve.
                Items where query_context matches these field values get priority.
            max_items: Thread-safe max items limit (defaults to config value).
        """
        # Use provided max_items or fall back to config
        effective_max = max_items if max_items is not None else self.config.max_items_after_crush
        n = len(items)
        keep_indices = set()

        # 1. Dynamic anchor selection (replaces static first 3 + last 2)
        anchor_pattern = self._map_to_anchor_pattern(CompressionStrategy.TIME_SERIES)
        anchor_indices = self._anchor_selector.select_anchors(
            items=items,
            max_items=effective_max,
            pattern=anchor_pattern,
            query=query_context or None,
        )
        keep_indices.update(anchor_indices)

        # 2. Items around change points from numeric fields
        for stats in analysis.field_stats.values():
            if stats.change_points:
                for cp in stats.change_points:
                    # Keep a window around each change point
                    for offset in range(-2, 3):
                        idx = cp + offset
                        if 0 <= idx < n:
                            keep_indices.add(idx)

        # 3. Structural outlier items (STATISTICAL detection - no hardcoded keywords)
        outlier_indices = _detect_structural_outliers(items)
        keep_indices.update(outlier_indices)

        # 3b. Error items via KEYWORD detection (PRESERVATION GUARANTEE)
        # This is critical - errors must ALWAYS be preserved regardless of structure
        error_indices = _detect_error_items_for_preservation(items)
        keep_indices.update(error_indices)

        # 4. Items matching query anchors (DETERMINISTIC exact match)
        # Anchors provide reliable preservation for specific entity lookups (UUIDs, IDs, names)
        if query_context:
            anchors = extract_query_anchors(query_context)
            for i, item in enumerate(items):
                if item_matches_anchors(item, anchors):
                    keep_indices.add(i)

        # 5. Items with high relevance to query context (PROBABILISTIC semantic match)
        if query_context:
            item_strs = [json.dumps(item, default=str) for item in items]
            scores = self._scorer.score_batch(item_strs, query_context)
            for i, score in enumerate(scores):
                if score.score >= self._relevance_threshold:
                    keep_indices.add(i)

        # 5b. TOIN preserve_fields: boost items where query matches these fields
        # Note: preserve_fields are SHA256[:8] hashes, use helper to match
        if preserve_fields and query_context:
            for i, item in enumerate(items):
                if _item_has_preserve_field_match(item, preserve_fields, query_context):
                    keep_indices.add(i)

        # Limit to effective_max while ALWAYS preserving outliers and anomalies
        keep_indices = self._prioritize_indices(keep_indices, items, n, analysis, effective_max)

        plan.keep_indices = sorted(keep_indices)
        return plan

    def _plan_cluster_sample(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        plan: CompressionPlan,
        query_context: str = "",
        preserve_fields: list[str] | None = None,
        max_items: int | None = None,
    ) -> CompressionPlan:
        """Plan compression for clusterable data (like logs).

        Uses clustering plus STATISTICAL outlier detection.
        Uses RelevanceScorer for semantic matching of user queries.

        Args:
            preserve_fields: TOIN-learned fields that users commonly retrieve.
                Items where query_context matches these field values get priority.
            max_items: Thread-safe max items limit (defaults to config value).
        """
        # Use provided max_items or fall back to config
        effective_max = max_items if max_items is not None else self.config.max_items_after_crush
        n = len(items)
        keep_indices = set()

        # 1. Dynamic anchor selection (replaces static first 3 + last 2)
        anchor_pattern = self._map_to_anchor_pattern(CompressionStrategy.CLUSTER_SAMPLE)
        anchor_indices = self._anchor_selector.select_anchors(
            items=items,
            max_items=effective_max,
            pattern=anchor_pattern,
            query=query_context or None,
        )
        keep_indices.update(anchor_indices)

        # 2. Structural outlier items (STATISTICAL detection - no hardcoded keywords)
        outlier_indices = _detect_structural_outliers(items)
        keep_indices.update(outlier_indices)

        # 2b. Error items via KEYWORD detection (PRESERVATION GUARANTEE)
        # This is critical - errors must ALWAYS be preserved regardless of structure
        error_indices = _detect_error_items_for_preservation(items)
        keep_indices.update(error_indices)

        # 3. Cluster by message-like field and keep representatives
        # Find a high-cardinality string field (likely message field)
        message_field = None
        max_uniqueness = 0.0
        for name, stats in analysis.field_stats.items():
            if stats.field_type == "string" and stats.unique_ratio > max_uniqueness:
                # Prefer fields with moderate to high uniqueness (message-like)
                if stats.unique_ratio > 0.3:
                    message_field = name
                    max_uniqueness = stats.unique_ratio

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

        # 4. Items matching query anchors (DETERMINISTIC exact match)
        # Anchors provide reliable preservation for specific entity lookups (UUIDs, IDs, names)
        if query_context:
            anchors = extract_query_anchors(query_context)
            for i, item in enumerate(items):
                if item_matches_anchors(item, anchors):
                    keep_indices.add(i)

        # 5. Items with high relevance to query context (PROBABILISTIC semantic match)
        if query_context:
            item_strs = [json.dumps(item, default=str) for item in items]
            scores = self._scorer.score_batch(item_strs, query_context)
            for i, score in enumerate(scores):
                if score.score >= self._relevance_threshold:
                    keep_indices.add(i)

        # 5b. TOIN preserve_fields: boost items where query matches these fields
        # Note: preserve_fields are SHA256[:8] hashes, use helper to match
        if preserve_fields and query_context:
            for i, item in enumerate(items):
                if _item_has_preserve_field_match(item, preserve_fields, query_context):
                    keep_indices.add(i)

        # Limit total while ALWAYS preserving outliers and anomalies
        keep_indices = self._prioritize_indices(keep_indices, items, n, analysis, effective_max)

        plan.keep_indices = sorted(keep_indices)
        return plan

    def _plan_top_n(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        plan: CompressionPlan,
        query_context: str = "",
        preserve_fields: list[str] | None = None,
        max_items: int | None = None,
    ) -> CompressionPlan:
        """Plan compression for scored/ranked data.

        For data with a score/relevance field, that field IS the primary relevance
        signal. Our internal relevance scoring is SECONDARY - it's used to find
        potential "needle" items that the original scoring might have missed.

        Strategy:
        1. Keep top N by score (the original system's relevance ranking)
        2. Add structural outliers (errors, anomalies)
        3. Add high-confidence relevance matches (needles the user is looking for)

        Args:
            preserve_fields: TOIN-learned fields that users commonly retrieve.
                Items where query_context matches these field values get priority.
            max_items: Thread-safe max items limit (defaults to config value).
        """
        # Use provided max_items or fall back to config
        effective_max = max_items if max_items is not None else self.config.max_items_after_crush

        # Find score field using STATISTICAL detection (no hardcoded field names)
        score_field = None
        max_confidence = 0.0
        for name, stats in analysis.field_stats.items():
            is_score, confidence = _detect_score_field_statistically(stats, items)
            if is_score and confidence > max_confidence:
                score_field = name
                max_confidence = confidence

        if not score_field:
            return self._plan_smart_sample(
                analysis, items, plan, query_context, preserve_fields, effective_max
            )

        plan.sort_field = score_field
        keep_indices = set()

        # 1. TOP N by score FIRST (the primary relevance signal)
        # The original system's score field is the authoritative ranking
        scored_items = [(i, item.get(score_field, 0)) for i, item in enumerate(items)]
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Reserve slots for outliers
        top_count = max(0, effective_max - 3)
        for idx, _ in scored_items[:top_count]:
            keep_indices.add(idx)

        # 2. Structural outlier items (STATISTICAL detection - no hardcoded keywords)
        outlier_indices = _detect_structural_outliers(items)
        keep_indices.update(outlier_indices)

        # 2b. Error items via KEYWORD detection (PRESERVATION GUARANTEE)
        # This is critical - errors must ALWAYS be preserved regardless of structure
        error_indices = _detect_error_items_for_preservation(items)
        keep_indices.update(error_indices)

        # 3. Items matching query anchors (DETERMINISTIC exact match) - ADDITIVE
        # Anchors provide reliable preservation for specific entity lookups (UUIDs, IDs, names)
        # These are ALWAYS preserved since they represent explicit user intent
        if query_context:
            anchors = extract_query_anchors(query_context)
            for i, item in enumerate(items):
                if i not in keep_indices and item_matches_anchors(item, anchors):
                    keep_indices.add(i)

        # 4. HIGH-CONFIDENCE relevance matches (potential needles) - ADDITIVE only
        # Only add items that are NOT already in top N but match the query strongly
        # Use a higher threshold (0.5) since the score field already captures relevance
        if query_context:
            item_strs = [json.dumps(item, default=str) for item in items]
            scores = self._scorer.score_batch(item_strs, query_context)
            # Higher threshold and limit count to avoid adding everything
            high_threshold = max(0.5, self._relevance_threshold * 2)
            added_count = 0
            max_relevance_adds = 3  # Limit additional relevance matches
            for i, score in enumerate(scores):
                if i not in keep_indices and score.score >= high_threshold:
                    keep_indices.add(i)
                    added_count += 1
                    if added_count >= max_relevance_adds:
                        break

        # 4b. TOIN preserve_fields: boost items where query matches these fields
        # Note: preserve_fields are SHA256[:8] hashes, use helper to match
        if preserve_fields and query_context:
            for i, item in enumerate(items):
                if i not in keep_indices:  # Only add if not already kept
                    if _item_has_preserve_field_match(item, preserve_fields, query_context):
                        keep_indices.add(i)

        plan.keep_count = len(keep_indices)
        plan.keep_indices = sorted(keep_indices)
        return plan

    def _plan_smart_sample(
        self,
        analysis: ArrayAnalysis,
        items: list[dict],
        plan: CompressionPlan,
        query_context: str = "",
        preserve_fields: list[str] | None = None,
        max_items: int | None = None,
    ) -> CompressionPlan:
        """Plan smart statistical sampling using STATISTICAL detection.

        Always keeps:
        - Dynamic anchor positions (based on data pattern and query context)
        - Structural outliers (items with rare fields or rare status values)
        - Anomalous numeric items (> 2 std from mean)
        - Items around change points
        - Items with high relevance to query context (via RelevanceScorer)

        Uses STATISTICAL detection instead of hardcoded keywords.

        Args:
            preserve_fields: TOIN-learned fields that users commonly retrieve.
                Items where query_context matches these field values get priority.
            max_items: Thread-safe max items limit (defaults to config value).
        """
        # Use provided max_items or fall back to config
        effective_max = max_items if max_items is not None else self.config.max_items_after_crush

        n = len(items)
        keep_indices = set()

        # 1. Dynamic anchor selection (replaces static first 3 + last 2)
        anchor_pattern = self._map_to_anchor_pattern(CompressionStrategy.SMART_SAMPLE)
        anchor_indices = self._anchor_selector.select_anchors(
            items=items,
            max_items=effective_max,
            pattern=anchor_pattern,
            query=query_context or None,
        )
        keep_indices.update(anchor_indices)

        # 2. Structural outlier items (STATISTICAL detection - no hardcoded keywords)
        outlier_indices = _detect_structural_outliers(items)
        keep_indices.update(outlier_indices)

        # 2b. Error items via KEYWORD detection (PRESERVATION GUARANTEE)
        # This is critical - errors must ALWAYS be preserved regardless of structure
        error_indices = _detect_error_items_for_preservation(items)
        keep_indices.update(error_indices)

        # 3. Anomalous numeric items (> 2 std from mean)
        for name, stats in analysis.field_stats.items():
            if stats.field_type == "numeric" and stats.mean_val is not None and stats.variance:
                std = stats.variance**0.5
                if std > 0:
                    threshold = self.config.variance_threshold * std
                    for i, item in enumerate(items):
                        val = item.get(name)
                        if isinstance(val, (int, float)):
                            if abs(val - stats.mean_val) > threshold:
                                keep_indices.add(i)

        # 4. Items around change points (if detected)
        if self.config.preserve_change_points:
            for stats in analysis.field_stats.values():
                if stats.change_points:
                    for cp in stats.change_points:
                        # Keep items around change point
                        for offset in range(-1, 2):
                            idx = cp + offset
                            if 0 <= idx < n:
                                keep_indices.add(idx)

        # 5. Items matching query anchors (DETERMINISTIC exact match)
        # Anchors provide reliable preservation for specific entity lookups (UUIDs, IDs, names)
        if query_context:
            anchors = extract_query_anchors(query_context)
            for i, item in enumerate(items):
                if item_matches_anchors(item, anchors):
                    keep_indices.add(i)

        # 6. Items with high relevance to query context (PROBABILISTIC semantic match)
        if query_context:
            item_strs = [json.dumps(item, default=str) for item in items]
            scores = self._scorer.score_batch(item_strs, query_context)
            for i, score in enumerate(scores):
                if score.score >= self._relevance_threshold:
                    keep_indices.add(i)

        # 6b. TOIN preserve_fields: boost items where query matches these fields
        # Note: preserve_fields are SHA256[:8] hashes, use helper to match
        if preserve_fields and query_context:
            for i, item in enumerate(items):
                if _item_has_preserve_field_match(item, preserve_fields, query_context):
                    keep_indices.add(i)

        # Limit to effective_max while ALWAYS preserving outliers and anomalies
        keep_indices = self._prioritize_indices(keep_indices, items, n, analysis, effective_max)

        plan.keep_indices = sorted(keep_indices)
        return plan

    def _execute_plan(
        self, plan: CompressionPlan, items: list[dict], analysis: ArrayAnalysis
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
    ccr_config: CCRConfig | None = None,
) -> tuple[str, bool, str]:
    """
    Convenience function to smart-crush a single tool output.

    NOTE: CCR markers are DISABLED by default in this convenience function
    to maintain backward compatibility (output remains valid JSON).
    To enable CCR markers, pass a CCRConfig with inject_retrieval_marker=True.

    Args:
        content: The tool output content (JSON string).
        config: Optional SmartCrusher configuration.
        ccr_config: Optional CCR (Compress-Cache-Retrieve) configuration.
            By default, CCR is enabled (caching) but markers are disabled.

    Returns:
        Tuple of (crushed_content, was_modified, analysis_info).
    """
    cfg = config or SmartCrusherConfig()

    # Default: CCR enabled for caching, but markers disabled for clean JSON output
    if ccr_config is None:
        ccr_cfg = CCRConfig(
            enabled=True,  # Still cache for retrieval
            inject_retrieval_marker=False,  # Don't break JSON output
        )
    else:
        ccr_cfg = ccr_config

    crusher = SmartCrusher(cfg, ccr_config=ccr_cfg)
    return crusher._smart_crush_content(content)
