"""One-function compression API for Headroom.

The simplest way to use Headroom — no proxy, no config, just compress:

    from headroom import compress

    result = compress(messages, model="claude-sonnet-4-5-20250929")
    result.messages          # Compressed messages (same format, fewer tokens)
    result.tokens_saved      # Tokens saved
    result.compression_ratio # e.g., 0.35 means 65% saved

Works with any LLM client, any proxy, any framework. Just compress
the messages before sending them.

Examples:

    # With Anthropic SDK
    from anthropic import Anthropic
    from headroom import compress

    client = Anthropic()
    messages = [{"role": "user", "content": huge_tool_output}]
    compressed = compress(messages, model="claude-sonnet-4-5-20250929")
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        messages=compressed.messages,
    )

    # With OpenAI SDK
    from openai import OpenAI
    from headroom import compress

    client = OpenAI()
    messages = [{"role": "user", "content": "analyze this"}, {"role": "tool", "content": big_data}]
    compressed = compress(messages, model="gpt-4o")
    response = client.chat.completions.create(model="gpt-4o", messages=compressed.messages)

    # With LiteLLM
    import litellm
    from headroom import compress

    messages = [...]
    compressed = compress(messages, model="bedrock/claude-sonnet")
    response = litellm.completion(model="bedrock/claude-sonnet", messages=compressed.messages)

    # With any HTTP client
    import httpx
    from headroom import compress

    compressed = compress(messages, model="claude-sonnet-4-5-20250929")
    httpx.post("https://api.anthropic.com/v1/messages", json={
        "model": "claude-sonnet-4-5-20250929",
        "messages": compressed.messages,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-initialized singleton pipeline
_pipeline = None
_pipeline_lock = None


@dataclass
class CompressResult:
    """Result of compressing messages.

    Attributes:
        messages: The compressed messages (same format as input).
        tokens_before: Token count before compression.
        tokens_after: Token count after compression.
        tokens_saved: Tokens removed by compression.
        compression_ratio: Ratio of tokens saved (0.0 = no savings, 1.0 = 100% removed).
        transforms_applied: List of transforms that were applied.
    """

    messages: list[dict[str, Any]]
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    transforms_applied: list[str] = field(default_factory=list)


def compress(
    messages: list[dict[str, Any]],
    model: str = "claude-sonnet-4-5-20250929",
    model_limit: int = 200000,
    optimize: bool = True,
    hooks: Any = None,
) -> CompressResult:
    """Compress messages using Headroom's full compression pipeline.

    This is the simplest way to use Headroom. No proxy, no config needed.
    Just pass messages and get compressed messages back.

    Args:
        messages: List of messages in Anthropic or OpenAI format.
        model: Model name (used for token counting and context limit).
        model_limit: Model's context window size in tokens.
        optimize: Whether to actually compress (False = passthrough for A/B testing).
        hooks: Optional CompressionHooks instance for custom behavior.

    Returns:
        CompressResult with compressed messages and metrics.
    """
    if not messages or not optimize:
        return CompressResult(messages=messages)

    pipeline = _get_pipeline()

    try:
        # Compute biases from hooks if provided
        biases = None
        if hooks:
            from headroom.hooks import CompressContext

            ctx = CompressContext(model=model)
            messages = hooks.pre_compress(messages, ctx)
            biases = hooks.compute_biases(messages, ctx)

        result = pipeline.apply(
            messages=messages,
            model=model,
            model_limit=model_limit,
            biases=biases,
        )

        tokens_before = result.tokens_before
        tokens_after = result.tokens_after
        tokens_saved = tokens_before - tokens_after
        ratio = tokens_saved / tokens_before if tokens_before > 0 else 0.0

        # Post-compress hook
        if hooks and tokens_saved > 0:
            from headroom.hooks import CompressEvent

            hooks.post_compress(
                CompressEvent(
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    tokens_saved=tokens_saved,
                    compression_ratio=ratio,
                    transforms_applied=result.transforms_applied,
                    model=model,
                )
            )

        return CompressResult(
            messages=result.messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            compression_ratio=ratio,
            transforms_applied=result.transforms_applied,
        )

    except Exception as e:
        logger.warning("Compression failed, returning original messages: %s", e)
        return CompressResult(messages=messages)


def _get_pipeline() -> Any:
    """Get or create the singleton compression pipeline."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    import threading

    global _pipeline_lock
    if _pipeline_lock is None:
        _pipeline_lock = threading.Lock()

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        from headroom.transforms import TransformPipeline

        # Default pipeline: CacheAligner → ContentRouter → IntelligentContext
        # CacheAligner: stabilizes prefix for provider KV cache hits
        # ContentRouter: routes to the right compressor per content type
        #   (SmartCrusher for JSON, CodeCompressor for code, LLMLingua for text)
        # IntelligentContext: enforces token limits with score-based dropping
        _pipeline = TransformPipeline()
        logger.debug("Headroom compression pipeline initialized")
        return _pipeline
