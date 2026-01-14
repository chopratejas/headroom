"""LLMLingua-2 compressor for ML-based prompt compression.

This module provides integration with LLMLingua-2, a BERT-based token classifier
trained via GPT-4 distillation. It achieves superior compression (up to 20x)
while maintaining high fidelity on tool outputs and structured content.

Key Features:
- Token-level classification (keep/remove) using fine-tuned BERT
- 3-6x faster than LLMLingua-1 with better results
- Especially effective on tool outputs, code, and structured data
- Reversible compression via CCR integration

Reference:
    LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression
    https://arxiv.org/abs/2403.12968

Installation:
    pip install headroom-ai[llmlingua]

Usage:
    >>> from headroom.transforms import LLMLinguaCompressor
    >>> compressor = LLMLinguaCompressor()
    >>> result = compressor.compress(long_tool_output)
    >>> print(result.compressed)  # Significantly reduced output
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer
from .base import Transform

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_llmlingua_available: bool | None = None
_llmlingua_instance: Any = None
_llmlingua_lock = threading.Lock()  # Thread safety for model access


def _check_llmlingua_available() -> bool:
    """Check if llmlingua package is available."""
    global _llmlingua_available
    if _llmlingua_available is None:
        try:
            import llmlingua  # noqa: F401

            _llmlingua_available = True
        except ImportError:
            _llmlingua_available = False
    return _llmlingua_available


def _get_llmlingua_compressor(model_name: str, device: str) -> Any:
    """Get or create the LLMLingua compressor instance.

    Uses lazy initialization and caches the instance to avoid repeated model loading.
    Thread-safe: uses lock to prevent race conditions during model initialization.

    Args:
        model_name: HuggingFace model name for the compressor.
        device: Device to run the model on ('cuda', 'cpu', or 'auto').

    Returns:
        PromptCompressor instance from llmlingua.

    Raises:
        ImportError: If llmlingua is not installed.
        RuntimeError: If model loading fails.
    """
    global _llmlingua_instance

    if not _check_llmlingua_available():
        raise ImportError(
            "llmlingua is not installed. Install with: pip install headroom-ai[llmlingua]\n"
            "Note: This requires ~2GB of disk space and ~1GB RAM for the model."
        )

    with _llmlingua_lock:
        # Double-check after acquiring lock
        if _llmlingua_instance is None or _llmlingua_instance._model_name != model_name:
            try:
                from llmlingua import PromptCompressor

                logger.info(
                    "Loading LLMLingua-2 model: %s on device: %s "
                    "(this may take 10-30s on first run)",
                    model_name,
                    device,
                )
                _llmlingua_instance = PromptCompressor(
                    model_name=model_name,
                    device_map=device,
                    use_llmlingua2=True,  # Use LLMLingua-2 (BERT classifier)
                )
                # Store model name for later comparison
                _llmlingua_instance._model_name = model_name
                logger.info("LLMLingua-2 model loaded successfully")

            except Exception as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    raise RuntimeError(
                        f"Out of memory loading LLMLingua model. Try:\n"
                        f"  1. Use device='cpu' instead of 'cuda'\n"
                        f"  2. Close other GPU applications\n"
                        f"  3. Use a smaller model\n"
                        f"Original error: {e}"
                    ) from e
                elif "not found" in error_msg or "404" in error_msg:
                    raise RuntimeError(
                        f"Model '{model_name}' not found on HuggingFace. Try:\n"
                        f"  1. Check the model name is correct\n"
                        f"  2. Use default: 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'\n"
                        f"Original error: {e}"
                    ) from e
                else:
                    raise RuntimeError(
                        f"Failed to load LLMLingua model: {e}\n"
                        f"Ensure you have sufficient disk space and memory."
                    ) from e

    return _llmlingua_instance


def unload_llmlingua_model() -> bool:
    """Unload the LLMLingua model to free memory.

    Use this when you're done with compression and want to reclaim GPU/CPU memory.
    The model will be reloaded automatically on the next compression call.

    Returns:
        True if a model was unloaded, False if no model was loaded.

    Example:
        >>> from headroom.transforms import LLMLinguaCompressor, unload_llmlingua_model
        >>> compressor = LLMLinguaCompressor()
        >>> result = compressor.compress(content)  # Model loaded here
        >>> # ... do other work ...
        >>> unload_llmlingua_model()  # Free ~1GB of memory
    """
    global _llmlingua_instance

    with _llmlingua_lock:
        if _llmlingua_instance is not None:
            model_name = getattr(_llmlingua_instance, "_model_name", "unknown")
            logger.info("Unloading LLMLingua model: %s", model_name)

            # Clear the instance
            _llmlingua_instance = None

            # Attempt to free GPU memory if torch is available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("Cleared CUDA cache")
            except ImportError:
                pass

            return True

    return False


def is_llmlingua_model_loaded() -> bool:
    """Check if an LLMLingua model is currently loaded.

    Returns:
        True if a model is loaded in memory, False otherwise.
    """
    return _llmlingua_instance is not None


@dataclass
class LLMLinguaConfig:
    """Configuration for LLMLingua-2 compression.

    Attributes:
        model_name: HuggingFace model for the compressor. Default is the
            LLMLingua-2 xlm-roberta-large model fine-tuned for compression.
        device: Device to run on ('cuda', 'cpu', 'auto'). Auto will use CUDA if available.
        target_compression_rate: Target compression ratio (e.g., 0.3 = keep 30% of tokens).
        force_tokens: Tokens to always preserve (e.g., important keywords).
        drop_consecutive: Whether to drop consecutive punctuation/whitespace.
        min_tokens_for_compression: Minimum token count to trigger compression.
            Content below this threshold is passed through unchanged.
        enable_ccr: Whether to store originals in CCR for retrieval.
        ccr_ttl: TTL for CCR entries in seconds.

    GOTCHA: Lower target_compression_rate = more aggressive compression.
        A rate of 0.2 means keeping only 20% of tokens.
    """

    # Model configuration
    model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    device: str = "auto"

    # Compression parameters
    target_compression_rate: float = 0.3
    force_tokens: list[str] = field(default_factory=list)
    drop_consecutive: bool = True

    # Thresholds
    min_tokens_for_compression: int = 100

    # CCR integration
    enable_ccr: bool = True
    ccr_ttl: int = 300  # 5 minutes

    # Content type specific settings
    code_compression_rate: float = 0.4  # More conservative for code
    json_compression_rate: float = 0.35  # Slightly conservative for JSON
    text_compression_rate: float = 0.25  # More aggressive for plain text


@dataclass
class LLMLinguaResult:
    """Result of LLMLingua-2 compression.

    Attributes:
        compressed: Compressed content.
        original: Original content before compression.
        original_tokens: Token count of original content.
        compressed_tokens: Token count after compression.
        compression_ratio: Actual compression ratio achieved.
        cache_key: CCR cache key if stored.
        model_used: Model that performed the compression.
        tokens_saved: Number of tokens saved.
    """

    compressed: str
    original: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    cache_key: str | None = None
    model_used: str | None = None

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved by compression."""
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def savings_percentage(self) -> float:
        """Percentage of tokens saved."""
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100


class LLMLinguaCompressor(Transform):
    """LLMLingua-2 based prompt compressor.

    Uses a BERT-based token classifier trained via GPT-4 distillation to
    identify and remove non-essential tokens while preserving semantic meaning.

    Key advantages over statistical compression:
    - Learned token importance from LLM feedback
    - Better handling of context-dependent importance
    - More aggressive compression with less information loss
    - Especially effective on structured outputs (JSON, code, logs)

    Example:
        >>> compressor = LLMLinguaCompressor()
        >>> result = compressor.compress(long_tool_output)
        >>> print(f"Saved {result.tokens_saved} tokens ({result.savings_percentage:.1f}%)")

        >>> # Use as a Transform in pipeline
        >>> from headroom.transforms import TransformPipeline
        >>> pipeline = TransformPipeline([LLMLinguaCompressor()])
        >>> result = pipeline.apply(messages, tokenizer)
    """

    name: str = "llmlingua_compressor"

    def __init__(self, config: LLMLinguaConfig | None = None):
        """Initialize LLMLingua compressor.

        Args:
            config: Compression configuration. If None, uses defaults.

        Note:
            The underlying model is loaded lazily on first use to avoid
            startup overhead when the compressor isn't used.
        """
        self.config = config or LLMLinguaConfig()
        self._compressor: Any = None  # Lazy loaded

    def compress(
        self,
        content: str,
        context: str = "",
        content_type: str | None = None,
    ) -> LLMLinguaResult:
        """Compress content using LLMLingua-2.

        Args:
            content: Content to compress.
            context: Optional context for relevance-aware compression.
            content_type: Type of content ('code', 'json', 'text').
                If None, auto-detected.

        Returns:
            LLMLinguaResult with compressed content and metadata.

        Raises:
            ImportError: If llmlingua is not installed.
        """
        # Check availability
        if not _check_llmlingua_available():
            logger.warning(
                "LLMLingua not available. Install with: pip install headroom-ai[llmlingua]"
            )
            return LLMLinguaResult(
                compressed=content,
                original=content,
                original_tokens=len(content.split()),  # Rough estimate
                compressed_tokens=len(content.split()),
                compression_ratio=1.0,
            )

        # Estimate token count (rough)
        estimated_tokens = len(content.split())

        # Skip compression for small content
        if estimated_tokens < self.config.min_tokens_for_compression:
            return LLMLinguaResult(
                compressed=content,
                original=content,
                original_tokens=estimated_tokens,
                compressed_tokens=estimated_tokens,
                compression_ratio=1.0,
            )

        # Get compression rate based on content type
        compression_rate = self._get_compression_rate(content, content_type)

        # Get or initialize compressor
        device = self._resolve_device()
        compressor = _get_llmlingua_compressor(self.config.model_name, device)

        # Prepare force tokens
        force_tokens = list(self.config.force_tokens)

        # Add context words as force tokens if provided
        if context:
            context_words = [w for w in context.split() if len(w) > 3]
            force_tokens.extend(context_words[:10])  # Limit to avoid overhead

        # Perform compression
        try:
            result = compressor.compress_prompt(
                original_prompt=content,
                rate=compression_rate,
                force_tokens=force_tokens if force_tokens else None,
                drop_consecutive=self.config.drop_consecutive,
            )

            compressed = result.get("compressed_prompt", content)
            original_tokens = result.get("origin_tokens", estimated_tokens)
            compressed_tokens = result.get("compressed_tokens", len(compressed.split()))

        except Exception as e:
            logger.warning("LLMLingua compression failed: %s", e)
            return LLMLinguaResult(
                compressed=content,
                original=content,
                original_tokens=estimated_tokens,
                compressed_tokens=estimated_tokens,
                compression_ratio=1.0,
            )

        # Calculate actual ratio
        ratio = compressed_tokens / max(original_tokens, 1)

        # Store in CCR if enabled
        cache_key = None
        if self.config.enable_ccr and ratio < 0.8:
            cache_key = self._store_in_ccr(content, compressed, original_tokens)
            if cache_key:
                compressed += (
                    f"\n[LLMLingua: {original_tokens}→{compressed_tokens} tokens. hash={cache_key}]"
                )

        return LLMLinguaResult(
            compressed=compressed,
            original=content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            cache_key=cache_key,
            model_used=self.config.model_name,
        )

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Apply LLMLingua compression to messages.

        This method implements the Transform interface for use in pipelines.
        It compresses tool outputs and long assistant/user messages.

        Args:
            messages: List of message dicts to transform.
            tokenizer: Tokenizer for accurate token counting.
            **kwargs: Additional arguments (e.g., 'context' for relevance).

        Returns:
            TransformResult with compressed messages and metadata.
        """
        tokens_before = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
        context = kwargs.get("context", "")

        transformed_messages = []
        transforms_applied = []
        warnings: list[str] = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # Compress tool results (highest value compression)
            if role == "tool" and content:
                result = self.compress(content, context=context, content_type="json")
                if result.compression_ratio < 0.9:
                    transformed_messages.append({**message, "content": result.compressed})
                    transforms_applied.append(f"llmlingua:tool:{result.compression_ratio:.2f}")
                else:
                    transformed_messages.append(message)

            # Compress long assistant messages (tool outputs often embedded)
            elif role == "assistant" and len(content) > 500:
                result = self.compress(content, context=context)
                if result.compression_ratio < 0.9:
                    transformed_messages.append({**message, "content": result.compressed})
                    transforms_applied.append(f"llmlingua:assistant:{result.compression_ratio:.2f}")
                else:
                    transformed_messages.append(message)

            # Pass through other messages
            else:
                transformed_messages.append(message)

        tokens_after = sum(
            tokenizer.count_text(str(m.get("content", ""))) for m in transformed_messages
        )

        # Add warning if llmlingua not available
        if not _check_llmlingua_available():
            warnings.append(
                "LLMLingua not installed. Install with: pip install headroom-ai[llmlingua]"
            )

        return TransformResult(
            messages=transformed_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied if transforms_applied else ["llmlingua:noop"],
            warnings=warnings,
        )

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if LLMLingua compression should be applied.

        Returns True if:
        - LLMLingua is available, AND
        - Total token count exceeds minimum threshold

        Args:
            messages: Messages to check.
            tokenizer: Tokenizer for counting.
            **kwargs: Additional arguments.

        Returns:
            True if compression should be applied.
        """
        if not _check_llmlingua_available():
            return False

        total_tokens = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
        return total_tokens >= self.config.min_tokens_for_compression

    def _get_compression_rate(
        self,
        content: str,
        content_type: str | None,
    ) -> float:
        """Get appropriate compression rate based on content type.

        Args:
            content: Content to analyze.
            content_type: Explicit content type or None for auto-detection.

        Returns:
            Target compression rate for this content.
        """
        if content_type == "code":
            return self.config.code_compression_rate
        elif content_type == "json":
            return self.config.json_compression_rate
        elif content_type == "text":
            return self.config.text_compression_rate

        # Auto-detect content type
        if self._looks_like_json(content):
            return self.config.json_compression_rate
        elif self._looks_like_code(content):
            return self.config.code_compression_rate
        else:
            return self.config.text_compression_rate

    def _looks_like_json(self, content: str) -> bool:
        """Check if content appears to be JSON."""
        stripped = content.strip()
        return (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        )

    def _looks_like_code(self, content: str) -> bool:
        """Check if content appears to be code."""
        code_indicators = [
            "def ",
            "class ",
            "function ",
            "import ",
            "from ",
            "const ",
            "let ",
            "var ",
            "public ",
            "private ",
            "async ",
            "await ",
            "return ",
            "if (",
            "for (",
            "while (",
        ]
        return any(indicator in content for indicator in code_indicators)

    def _resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def _store_in_ccr(
        self,
        original: str,
        compressed: str,
        original_tokens: int,
    ) -> str | None:
        """Store original content in CCR for later retrieval.

        Args:
            original: Original content before compression.
            compressed: Compressed content.
            original_tokens: Token count of original.

        Returns:
            Cache key if stored successfully, None otherwise.
        """
        try:
            from ..cache.compression_store import get_compression_store

            store = get_compression_store()
            return store.store(
                original,
                compressed,
                original_tokens=original_tokens,
                compressed_tokens=len(compressed.split()),
                compression_strategy="llmlingua2",
            )
        except ImportError:
            return None
        except Exception as e:
            logger.debug("CCR storage failed: %s", e)
            return None


def compress_with_llmlingua(
    content: str,
    compression_rate: float = 0.3,
    context: str = "",
    model_name: str | None = None,
) -> str:
    """Convenience function for one-off compression.

    Args:
        content: Content to compress.
        compression_rate: Target compression rate (0.0-1.0).
        context: Optional context for relevance-aware compression.
        model_name: Optional model name override.

    Returns:
        Compressed content string.

    Example:
        >>> compressed = compress_with_llmlingua(long_output, compression_rate=0.2)
    """
    config = LLMLinguaConfig(target_compression_rate=compression_rate)
    if model_name:
        config.model_name = model_name

    compressor = LLMLinguaCompressor(config)
    result = compressor.compress(content, context=context)
    return result.compressed
