"""Configuration models for Headroom SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class HeadroomMode(str, Enum):
    """Operating modes for Headroom."""

    AUDIT = "audit"  # Observe only, no modifications
    OPTIMIZE = "optimize"  # Apply deterministic transforms
    SIMULATE = "simulate"  # Return transform plan without API call


# Model context limits should be provided by the Provider
# This dict allows user overrides only
DEFAULT_MODEL_CONTEXT_LIMITS: dict[str, int] = {}


@dataclass
class ToolCrusherConfig:
    """Configuration for tool output compression (naive/fixed-rule approach).

    GOTCHAS:
    - Keeps FIRST N items only - may miss important data later in arrays
    - A spike at index 50 will be lost if max_array_items=10
    - String truncation cuts at fixed length, may break mid-word/mid-sentence
    - No awareness of data patterns or importance

    Consider using SmartCrusherConfig instead for statistical analysis.
    """

    enabled: bool = False  # Disabled by default, SmartCrusher is preferred
    min_tokens_to_crush: int = 500  # Only crush if > N tokens
    max_array_items: int = 10  # Keep first N items
    max_string_length: int = 1000  # Truncate strings > N chars
    max_depth: int = 5  # Preserve structure to depth N
    preserve_keys: set[str] = field(
        default_factory=lambda: {"error", "status", "code", "id", "message", "name", "type"}
    )
    tool_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class CacheAlignerConfig:
    """Configuration for cache alignment.

    GOTCHAS:
    - Date regex may match non-date content (e.g., version numbers like "2024-01-15")
    - Moving dates to end of system prompt may confuse models if date was
      semantically important in its original position
    - Whitespace normalization may break:
      - Code blocks with significant indentation
      - ASCII art or formatted tables
      - Markdown that relies on specific spacing
    - ISO timestamps in tool outputs may be incorrectly flagged as "dynamic dates"

    SAFE: Only applied to SYSTEM messages, not user/assistant/tool content.
    """

    enabled: bool = True
    date_patterns: list[str] = field(
        default_factory=lambda: [
            r"Current [Dd]ate:?\s*\d{4}-\d{2}-\d{2}",
            r"Today is \w+,?\s+\w+ \d+",
            r"Today's date:?\s*\d{4}-\d{2}-\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        ]
    )
    normalize_whitespace: bool = True
    collapse_blank_lines: bool = True
    # Separator used to mark where dynamic content begins in system message
    # Content before this separator is cached; content after is dynamic
    dynamic_tail_separator: str = "\n\n---\n[Dynamic Context]\n"


@dataclass
class RollingWindowConfig:
    """Configuration for rolling window token cap.

    GOTCHAS:
    - Dropping old turns loses context the model may need:
      - "As I mentioned earlier..." - what was mentioned is now gone
      - "The user asked about X" - that turn may be dropped
      - Implicit references to prior conversation become orphaned
    - Tool call/result pairs are kept atomic (correct), BUT:
      - Assistant text referencing a dropped tool result becomes confusing
      - "Based on the search results..." when those results are gone
    - keep_last_turns=2 may not be enough for complex multi-step reasoning
    - No semantic analysis - drops oldest first regardless of importance

    SAFER ALTERNATIVES:
    - Increase keep_last_turns for agentic workloads
    - Use summarization for old context (not implemented - would add latency)
    - Set enabled=False for short conversations
    """

    enabled: bool = True
    keep_system: bool = True  # Never drop system prompt
    keep_last_turns: int = 2  # Never drop last N turns
    output_buffer_tokens: int = 4000  # Reserve for output


@dataclass
class RelevanceScorerConfig:
    """Configuration for relevance scoring in SmartCrusher.

    Relevance scoring determines which items to keep when compressing
    tool outputs. Uses the pattern: relevance(item, context) -> [0, 1].

    Available tiers:
    - "bm25": BM25 keyword matching (zero dependencies, fast)
    - "embedding": Semantic similarity via sentence-transformers
    - "hybrid": BM25 + embedding with adaptive fusion (RECOMMENDED)

    DEFAULT: "hybrid" - combines exact matching (UUIDs, IDs) with semantic
    understanding. Falls back to BM25 if sentence-transformers not installed.

    For full hybrid support, install: pip install headroom[relevance]

    WHY HYBRID IS DEFAULT:
    - Missing important items during compression is catastrophic
    - BM25 alone gives low scores for single-term matches (e.g., "Alice" = 0.07)
    - Semantic matching catches "errors" -> "failed", "issues", etc.
    - 5-10ms latency is acceptable vs. losing critical data
    """

    tier: Literal["bm25", "embedding", "hybrid"] = "hybrid"

    # BM25 parameters
    bm25_k1: float = 1.5  # Term frequency saturation
    bm25_b: float = 0.75  # Length normalization

    # Embedding parameters
    embedding_model: str = "all-MiniLM-L6-v2"  # Lightweight model

    # Hybrid parameters
    hybrid_alpha: float = 0.5  # BM25 weight (1-alpha = embedding weight)
    adaptive_alpha: bool = True  # Adjust alpha based on query type

    # Scoring thresholds
    # With hybrid/embedding: semantic scores are meaningful (0.3-0.5 for good matches)
    # With BM25 fallback: threshold is still reasonable for multi-term matches
    # Lower threshold = safer (keeps more items), higher = more aggressive compression
    relevance_threshold: float = 0.25  # Keep items above this score


@dataclass
class SmartCrusherConfig:
    """Configuration for smart statistical crusher (DEFAULT).

    Uses statistical analysis to intelligently compress tool outputs while
    PRESERVING THE ORIGINAL JSON SCHEMA. Output contains only items from
    the original array - no wrappers, no generated text, no metadata.

    Safe V1 Compression Recipe - Always keeps:
    - First K items (default 3)
    - Last K items (default 2)
    - Error items (containing 'error', 'exception', 'failed', 'critical')
    - Anomalous numeric items (> 2 std from mean)
    - Top-K by score if score field present
    - Items matching query context via RelevanceScorer

    GOTCHAS:
    - Adds ~5-10ms overhead per tool output for statistical analysis
    - Change point detection uses fixed window (5 items) - may miss:
      - Very gradual changes
      - Patterns in smaller arrays
    - TOP_N for search results assumes higher score = more relevant
      (may not be true for all APIs)

    SAFER SETTINGS:
    - Increase max_items_after_crush for critical data
    - Set variance_threshold lower (1.5) to catch more change points
    """

    enabled: bool = True  # Enabled by default (preferred over ToolCrusher)
    min_items_to_analyze: int = 5  # Don't analyze tiny arrays
    min_tokens_to_crush: int = 200  # Only crush if > N tokens
    variance_threshold: float = 2.0  # Std devs for change point detection
    uniqueness_threshold: float = 0.1  # Below this = nearly constant
    similarity_threshold: float = 0.8  # For clustering similar strings
    max_items_after_crush: int = 15  # Target max items in output
    preserve_change_points: bool = True
    factor_out_constants: bool = False  # Disabled - preserves original schema
    include_summaries: bool = False  # Disabled - no generated text

    # Relevance scoring configuration
    relevance: RelevanceScorerConfig = field(default_factory=RelevanceScorerConfig)


@dataclass
class CacheOptimizerConfig:
    """Configuration for provider-specific cache optimization.

    The CacheOptimizer system provides provider-specific caching strategies:
    - Anthropic: Explicit cache_control breakpoints for prompt caching
    - OpenAI: Prefix stabilization for automatic prefix caching
    - Google: CachedContent API lifecycle management

    This is COMPLEMENTARY to the CacheAligner transform - CacheAligner does
    basic prefix stabilization (date extraction, whitespace normalization),
    while CacheOptimizer applies provider-specific optimizations.

    Enable this for maximum cache hit rates when you know your provider.
    """

    enabled: bool = True  # Enable provider-specific cache optimization
    auto_detect_provider: bool = True  # Auto-detect from HeadroomClient provider
    min_cacheable_tokens: int = 1024  # Minimum tokens for caching (provider may override)
    enable_semantic_cache: bool = False  # Enable query-level semantic caching
    semantic_cache_similarity: float = 0.95  # Similarity threshold for semantic cache
    semantic_cache_max_entries: int = 1000  # Max semantic cache entries
    semantic_cache_ttl_seconds: int = 300  # Semantic cache TTL


@dataclass
class HeadroomConfig:
    """Main configuration for HeadroomClient."""

    store_url: str = "sqlite:///headroom.db"
    default_mode: HeadroomMode = HeadroomMode.AUDIT
    model_context_limits: dict[str, int] = field(
        default_factory=lambda: DEFAULT_MODEL_CONTEXT_LIMITS.copy()
    )
    tool_crusher: ToolCrusherConfig = field(default_factory=ToolCrusherConfig)
    smart_crusher: SmartCrusherConfig = field(default_factory=SmartCrusherConfig)
    cache_aligner: CacheAlignerConfig = field(default_factory=CacheAlignerConfig)
    rolling_window: RollingWindowConfig = field(default_factory=RollingWindowConfig)
    cache_optimizer: CacheOptimizerConfig = field(default_factory=CacheOptimizerConfig)

    # Debugging - opt-in diff artifact generation
    generate_diff_artifact: bool = False  # Enable to get detailed transform diffs

    def get_context_limit(self, model: str) -> int | None:
        """
        Get context limit for a model from user overrides.

        Args:
            model: Model name.

        Returns:
            Context limit if configured, None otherwise.
            Provider should be consulted if None is returned.
        """
        if model in self.model_context_limits:
            return self.model_context_limits[model]
        # Try prefix matching for versioned model names
        for known_model, limit in self.model_context_limits.items():
            if model.startswith(known_model):
                return limit
        return None


@dataclass
class Block:
    """Atomic unit of context analysis."""

    kind: Literal["system", "user", "assistant", "tool_call", "tool_result", "rag", "unknown"]
    text: str
    tokens_est: int
    content_hash: str
    source_index: int  # Position in original messages
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class WasteSignals:
    """Detected waste signals in a request."""

    json_bloat_tokens: int = 0  # JSON blocks > 500 tokens
    html_noise_tokens: int = 0  # HTML tags/comments
    base64_tokens: int = 0  # Base64 encoded blobs
    whitespace_tokens: int = 0  # Repeated whitespace
    dynamic_date_tokens: int = 0  # Dynamic dates in system prompt
    repetition_tokens: int = 0  # Repeated content

    def total(self) -> int:
        """Total waste tokens detected."""
        return (
            self.json_bloat_tokens
            + self.html_noise_tokens
            + self.base64_tokens
            + self.whitespace_tokens
            + self.dynamic_date_tokens
            + self.repetition_tokens
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for storage."""
        return {
            "json_bloat": self.json_bloat_tokens,
            "html_noise": self.html_noise_tokens,
            "base64": self.base64_tokens,
            "whitespace": self.whitespace_tokens,
            "dynamic_date": self.dynamic_date_tokens,
            "repetition": self.repetition_tokens,
        }


@dataclass
class CachePrefixMetrics:
    """Detailed cache prefix metrics for debugging cache misses.

    Log these per-request to understand why caching is or isn't working.
    Compare stable_prefix_hash across requests - any change means cache miss.
    """

    stable_prefix_bytes: int  # Byte length of static prefix
    stable_prefix_tokens_est: int  # Estimated token count of static prefix
    stable_prefix_hash: str  # Hash of canonicalized prefix (16 chars)
    prefix_changed: bool  # True if hash differs from previous request in session
    previous_hash: str | None = None  # Previous hash for comparison (None = first request)


@dataclass
class TransformResult:
    """Output of a transform operation."""

    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    transforms_applied: list[str]
    markers_inserted: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diff_artifact: DiffArtifact | None = None  # Populated if generate_diff_artifact=True
    cache_metrics: CachePrefixMetrics | None = None  # Populated by CacheAligner


@dataclass
class TransformDiff:
    """Diff info for a single transform (for debugging)."""

    transform_name: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    items_removed: int = 0
    items_kept: int = 0
    details: str = ""  # Human-readable description of what changed


@dataclass
class DiffArtifact:
    """Complete diff artifact for debugging transform pipeline.

    Opt-in via HeadroomConfig.generate_diff_artifact = True.
    Useful for understanding what each transform did to your messages.
    """

    request_id: str
    original_tokens: int
    optimized_tokens: int
    total_tokens_saved: int
    transforms: list[TransformDiff] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result of a simulation (dry-run)."""

    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms: list[str]
    estimated_savings: str  # Human-readable cost estimate
    messages_optimized: list[dict[str, Any]]
    block_breakdown: dict[str, int]
    waste_signals: dict[str, int]
    stable_prefix_hash: str
    cache_alignment_score: float


@dataclass
class RequestMetrics:
    """Comprehensive metrics for a single request."""

    request_id: str
    timestamp: datetime
    model: str
    stream: bool
    mode: str  # audit | optimize | simulate

    # Token breakdown
    tokens_input_before: int
    tokens_input_after: int
    tokens_output: int | None = None  # None if streaming

    # Block breakdown
    block_breakdown: dict[str, int] = field(default_factory=dict)

    # Waste signals
    waste_signals: dict[str, int] = field(default_factory=dict)

    # Cache metrics (basic)
    stable_prefix_hash: str = ""
    cache_alignment_score: float = 0.0
    cached_tokens: int | None = None  # From API response if available

    # Cache optimizer metrics (provider-specific)
    cache_optimizer_used: str | None = None  # e.g., "anthropic-cache-optimizer"
    cache_optimizer_strategy: str | None = None  # e.g., "explicit_breakpoints"
    cacheable_tokens: int = 0  # Tokens eligible for caching
    breakpoints_inserted: int = 0  # Cache breakpoints added (Anthropic)
    estimated_cache_hit: bool = False  # Whether prefix matches previous
    estimated_savings_percent: float = 0.0  # Estimated savings if cached
    semantic_cache_hit: bool = False  # Whether semantic cache was hit

    # Transform details
    transforms_applied: list[str] = field(default_factory=list)
    tool_units_dropped: int = 0
    turns_dropped: int = 0

    # For debugging
    messages_hash: str = ""
    error: str | None = None
