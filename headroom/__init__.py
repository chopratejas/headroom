"""
Headroom - The Context Optimization Layer for LLM Applications.

Cut your LLM costs by 50-90% without losing accuracy.

Headroom wraps LLM clients to provide:
- Smart compression of tool outputs (keeps errors, anomalies, relevant items)
- Cache-aligned prefix optimization for better provider cache hits
- Rolling window token management for long conversations
- Full streaming support with zero accuracy loss

Quick Start:

    from headroom import HeadroomClient, OpenAIProvider
    from openai import OpenAI

    # Wrap your existing client
    client = HeadroomClient(
        original_client=OpenAI(),
        provider=OpenAIProvider(),
        default_mode="optimize",
    )

    # Use exactly like the original client
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Hello!"},
        ],
    )

    # Check savings
    stats = client.get_stats()
    print(f"Tokens saved: {stats['session']['tokens_saved_total']}")

Verify It's Working:

    # Validate configuration
    result = client.validate_setup()
    if not result["valid"]:
        print("Issues:", result)

    # Enable logging to see what's happening
    import logging
    logging.basicConfig(level=logging.INFO)
    # INFO:headroom.transforms.pipeline:Pipeline complete: 45000 -> 4500 tokens

Simulate Before Sending:

    plan = client.chat.completions.simulate(
        model="gpt-4o",
        messages=large_messages,
    )
    print(f"Would save {plan.tokens_saved} tokens")
    print(f"Transforms: {plan.transforms}")

Error Handling:

    from headroom import HeadroomError, ConfigurationError, ProviderError

    try:
        response = client.chat.completions.create(...)
    except ConfigurationError as e:
        print(f"Config issue: {e.details}")
    except HeadroomError as e:
        print(f"Headroom error: {e}")

For more examples, see https://github.com/headroom-sdk/headroom/tree/main/examples
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
from .exceptions import (
    CacheError,
    CompressionError,
    ConfigurationError,
    HeadroomError,
    ProviderError,
    StorageError,
    TokenizationError,
    TransformError,
    ValidationError,
)

# Memory module - hierarchical memory system
from .memory import (
    EmbedderBackend,
    HierarchicalMemory,
    Memory,
    MemoryConfig,
    ScopeLevel,
    with_memory,
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

__version__ = "0.3.0"

__all__ = [
    # Main client
    "HeadroomClient",
    # Providers
    "Provider",
    "TokenCounter",
    "OpenAIProvider",
    "AnthropicProvider",
    # Exceptions
    "HeadroomError",
    "ConfigurationError",
    "ProviderError",
    "StorageError",
    "CompressionError",
    "TokenizationError",
    "CacheError",
    "ValidationError",
    "TransformError",
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
    # Memory - hierarchical memory system
    "with_memory",  # Main user-facing API
    "Memory",
    "ScopeLevel",
    "HierarchicalMemory",
    "MemoryConfig",
    "EmbedderBackend",
]
