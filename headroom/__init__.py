"""
Headroom - A safe, deterministic Context Budget Controller for LLM APIs.

Headroom wraps LLM clients to provide:
- Context waste detection and reporting
- Tool output compression
- Cache-aligned prefix optimization
- Rolling window token management
- Full streaming support

Example usage:

    from headroom import HeadroomClient, OpenAIProvider
    from openai import OpenAI

    base = OpenAI(api_key="...")
    provider = OpenAIProvider()

    client = HeadroomClient(
        original_client=base,
        provider=provider,
        store_url="sqlite:///headroom.db",
        default_mode="audit",
    )

    # Use like normal OpenAI client
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[...],
        headroom_mode="optimize",  # Enable optimization
    )

    # Simulate without API call
    plan = client.chat.completions.simulate(
        model="gpt-4o",
        messages=[...],
    )
    print(f"Would save {plan.tokens_saved} tokens")
"""

from .cache import (
    AnthropicCacheOptimizer,
    BaseCacheOptimizer,
    CacheConfig,
    CacheMetrics,
    CacheOptimizerRegistry,
    CacheResult,
    CacheStrategy,
    GoogleCacheOptimizer,
    OpenAICacheOptimizer,
    OptimizationContext,
    SemanticCache,
    SemanticCacheLayer,
)
from .client import HeadroomClient
from .config import (
    Block,
    CacheAlignerConfig,
    CacheOptimizerConfig,
    CachePrefixMetrics,
    DiffArtifact,
    HeadroomConfig,
    HeadroomMode,
    RelevanceScorerConfig,
    RequestMetrics,
    RollingWindowConfig,
    SimulationResult,
    SmartCrusherConfig,
    ToolCrusherConfig,
    TransformDiff,
    TransformResult,
    WasteSignals,
)
from .providers import AnthropicProvider, OpenAIProvider, Provider, TokenCounter
from .relevance import (
    BM25Scorer,
    EmbeddingScorer,
    HybridScorer,
    RelevanceScore,
    RelevanceScorer,
    create_scorer,
    embedding_available,
)
from .reporting import generate_report
from .tokenizer import Tokenizer, count_tokens_messages, count_tokens_text
from .transforms import (
    CacheAligner,
    RollingWindow,
    SmartCrusher,
    ToolCrusher,
    TransformPipeline,
)

__version__ = "0.2.0"

__all__ = [
    # Main client
    "HeadroomClient",
    # Providers
    "Provider",
    "TokenCounter",
    "OpenAIProvider",
    "AnthropicProvider",
    # Config
    "HeadroomConfig",
    "HeadroomMode",
    "ToolCrusherConfig",
    "SmartCrusherConfig",
    "CacheAlignerConfig",
    "CacheOptimizerConfig",
    "RollingWindowConfig",
    "RelevanceScorerConfig",
    # Data models
    "Block",
    "CachePrefixMetrics",
    "DiffArtifact",
    "RequestMetrics",
    "SimulationResult",
    "TransformDiff",
    "TransformResult",
    "WasteSignals",
    # Transforms
    "ToolCrusher",
    "SmartCrusher",
    "CacheAligner",
    "RollingWindow",
    "TransformPipeline",
    # Cache optimizers
    "BaseCacheOptimizer",
    "CacheConfig",
    "CacheMetrics",
    "CacheResult",
    "CacheStrategy",
    "OptimizationContext",
    "CacheOptimizerRegistry",
    "AnthropicCacheOptimizer",
    "OpenAICacheOptimizer",
    "GoogleCacheOptimizer",
    "SemanticCache",
    "SemanticCacheLayer",
    # Relevance scoring
    "RelevanceScore",
    "RelevanceScorer",
    "BM25Scorer",
    "EmbeddingScorer",
    "HybridScorer",
    "create_scorer",
    "embedding_available",
    # Utilities
    "Tokenizer",
    "count_tokens_text",
    "count_tokens_messages",
    "generate_report",
]
