"""Strands Agents model wrapper for Headroom optimization.

This module provides HeadroomStrandsModel, which wraps any Strands model
to apply Headroom context optimization before API calls.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from headroom import HeadroomConfig, HeadroomMode
from headroom.providers import OpenAIProvider
from headroom.transforms import TransformPipeline

logger = logging.getLogger(__name__)

# Strands imports - these are optional dependencies
STRANDS_AVAILABLE = False
try:
    # Try importing strands-agents SDK
    # Note: Actual import paths may vary based on strands-agents version
    import strands  # noqa: F401

    STRANDS_AVAILABLE = True
except ImportError:
    pass


def strands_available() -> bool:
    """Check if Strands SDK is installed."""
    return STRANDS_AVAILABLE


def _check_strands_available() -> None:
    """Raise ImportError if Strands is not installed."""
    if not STRANDS_AVAILABLE:
        raise ImportError(
            "Strands Agents SDK is required for this integration. "
            "Install with: pip install strands-agents"
        )


@dataclass
class OptimizationMetrics:
    """Metrics from a single optimization pass."""

    request_id: str
    timestamp: datetime
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    transforms_applied: list[str]
    model: str


class HeadroomStrandsModel:
    """Strands model wrapper that applies Headroom optimizations.

    Wraps any Strands Model and automatically optimizes the context
    before each API call. Works with BedrockModel, OpenAIModel, and
    other Strands model types.

    Example:
        from strands import Agent
        from strands.models import BedrockModel
        from headroom.integrations.strands import HeadroomStrandsModel

        # Wrap any Strands model
        model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        optimized = HeadroomStrandsModel(wrapped_model=model)

        # Use with agent
        agent = Agent(model=optimized)
        response = agent("Hello!")

        # Access metrics
        print(f"Saved {optimized.total_tokens_saved} tokens")

    Attributes:
        wrapped_model: The underlying Strands model
        total_tokens_saved: Running total of tokens saved
        metrics_history: List of OptimizationMetrics from recent calls
    """

    def __init__(
        self,
        wrapped_model: Any,
        config: HeadroomConfig | None = None,
        mode: HeadroomMode = HeadroomMode.OPTIMIZE,
        auto_detect_provider: bool = True,
    ) -> None:
        """Initialize HeadroomStrandsModel.

        Args:
            wrapped_model: The Strands model to wrap
            config: HeadroomConfig for optimization settings
            mode: HeadroomMode (AUDIT, OPTIMIZE, or SIMULATE)
            auto_detect_provider: Whether to auto-detect provider from model
        """
        if wrapped_model is None:
            raise ValueError("wrapped_model cannot be None")

        self.wrapped_model = wrapped_model
        self.config = config or HeadroomConfig()
        self.mode = mode
        self.auto_detect_provider = auto_detect_provider

        # Lazy-initialized pipeline
        self._pipeline: TransformPipeline | None = None
        self._provider: OpenAIProvider | None = None

        # Metrics tracking
        self._metrics_history: list[OptimizationMetrics] = []
        self._total_tokens_saved: int = 0
        self._lock = threading.Lock()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all calls (thread-safe)."""
        with self._lock:
            return self._total_tokens_saved

    @property
    def metrics_history(self) -> list[OptimizationMetrics]:
        """History of optimization metrics (thread-safe copy)."""
        with self._lock:
            return self._metrics_history.copy()

    @property
    def pipeline(self) -> TransformPipeline:
        """Lazily initialize TransformPipeline (thread-safe)."""
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._provider = OpenAIProvider()
                    self._pipeline = TransformPipeline(
                        config=self.config,
                        provider=self._provider,
                    )
        return self._pipeline

    # =========================================================================
    # Model interface - forwards to wrapped model
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        if name.startswith("_") or name in (
            "wrapped_model",
            "config",
            "mode",
            "auto_detect_provider",
            "pipeline",
            "total_tokens_saved",
            "metrics_history",
        ):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self.wrapped_model, name)

    def _get_model_name(self) -> str:
        """Extract model name from wrapped model."""
        if hasattr(self.wrapped_model, "model_id"):
            return self.wrapped_model.model_id
        if hasattr(self.wrapped_model, "model"):
            return self.wrapped_model.model
        if hasattr(self.wrapped_model, "id"):
            return self.wrapped_model.id
        return "unknown"

    def _convert_messages_to_openai(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert Strands messages to OpenAI format for Headroom."""
        result = []
        for msg in messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                entry: dict[str, Any] = {
                    "role": getattr(msg, "role", "user"),
                    "content": getattr(msg, "content", "") or "",
                }
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = msg.tool_calls
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id
                result.append(entry)
            elif isinstance(msg, dict):
                result.append(msg.copy())
            else:
                result.append({"role": "user", "content": str(msg)})
        return result

    def _optimize_messages(
        self, messages: list[Any]
    ) -> tuple[list[dict[str, Any]], OptimizationMetrics]:
        """Apply Headroom optimization to messages."""
        request_id = str(uuid4())
        model = self._get_model_name()

        # Convert to OpenAI format
        openai_messages = self._convert_messages_to_openai(messages)

        if not openai_messages:
            metrics = OptimizationMetrics(
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
                tokens_before=0,
                tokens_after=0,
                tokens_saved=0,
                savings_percent=0,
                transforms_applied=[],
                model=model,
            )
            return openai_messages, metrics

        # Get model context limit
        model_limit = self._provider.get_context_limit(model) if self._provider else 128000

        try:
            # Apply Headroom transforms
            result = self.pipeline.apply(
                messages=openai_messages,
                model=model,
                model_limit=model_limit,
            )
            optimized = result.messages
            tokens_before = result.tokens_before
            tokens_after = result.tokens_after
            transforms_applied = result.transforms_applied
        except Exception as e:
            logger.warning(f"Headroom optimization failed: {e}")
            optimized = openai_messages
            tokens_before = sum(len(str(m.get("content", ""))) // 4 for m in openai_messages)
            tokens_after = tokens_before
            transforms_applied = ["fallback:error"]

        tokens_saved = max(0, tokens_before - tokens_after)
        metrics = OptimizationMetrics(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            savings_percent=(tokens_saved / tokens_before * 100 if tokens_before > 0 else 0),
            transforms_applied=transforms_applied,
            model=model,
        )

        # Track metrics
        with self._lock:
            self._metrics_history.append(metrics)
            self._total_tokens_saved += metrics.tokens_saved
            if len(self._metrics_history) > 100:
                self._metrics_history = self._metrics_history[-100:]

        return optimized, metrics

    # =========================================================================
    # Model invocation methods
    # =========================================================================

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """Invoke the model with optimized messages.

        Args:
            messages: Messages to send to the model
            **kwargs: Additional arguments for the model

        Returns:
            Model response
        """
        optimized, metrics = self._optimize_messages(messages)

        logger.info(
            f"Headroom optimized: {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.savings_percent:.1f}% saved)"
        )

        return self.wrapped_model.invoke(optimized, **kwargs)

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """Async invoke the model with optimized messages.

        Args:
            messages: Messages to send to the model
            **kwargs: Additional arguments for the model

        Returns:
            Model response
        """
        optimized, metrics = self._optimize_messages(messages)

        logger.info(
            f"Headroom optimized (async): {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.savings_percent:.1f}% saved)"
        )

        if hasattr(self.wrapped_model, "ainvoke"):
            return await self.wrapped_model.ainvoke(optimized, **kwargs)
        return self.wrapped_model.invoke(optimized, **kwargs)

    def __call__(self, messages: list[Any], **kwargs: Any) -> Any:
        """Make the wrapper callable like the original model.

        Args:
            messages: Messages to send to the model
            **kwargs: Additional arguments for the model

        Returns:
            Model response
        """
        return self.invoke(messages, **kwargs)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_savings_summary(self) -> dict[str, Any]:
        """Get summary of token savings."""
        with self._lock:
            if not self._metrics_history:
                return {
                    "total_requests": 0,
                    "total_tokens_saved": 0,
                    "average_savings_percent": 0,
                }

            return {
                "total_requests": len(self._metrics_history),
                "total_tokens_saved": self._total_tokens_saved,
                "average_savings_percent": (
                    sum(m.savings_percent for m in self._metrics_history)
                    / len(self._metrics_history)
                ),
                "total_tokens_before": sum(m.tokens_before for m in self._metrics_history),
                "total_tokens_after": sum(m.tokens_after for m in self._metrics_history),
            }

    def reset(self) -> None:
        """Reset all tracked metrics (thread-safe)."""
        with self._lock:
            self._metrics_history = []
            self._total_tokens_saved = 0


def optimize_messages(
    messages: list[Any],
    config: HeadroomConfig | None = None,
    model: str = "claude-3-sonnet-20240229",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Standalone function to optimize Strands messages.

    Use this for manual optimization when you need fine-grained control.

    Args:
        messages: List of Strands messages
        config: HeadroomConfig for optimization settings
        model: Model name for token estimation

    Returns:
        Tuple of (optimized_messages, metrics_dict)

    Example:
        from headroom.integrations.strands import optimize_messages

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        optimized, metrics = optimize_messages(messages)
        print(f"Saved {metrics['tokens_saved']} tokens")
    """
    config = config or HeadroomConfig()
    provider = OpenAIProvider()
    pipeline = TransformPipeline(config=config, provider=provider)

    # Convert to OpenAI format
    openai_messages = []
    for msg in messages:
        if hasattr(msg, "role") and hasattr(msg, "content"):
            openai_messages.append({"role": msg.role, "content": msg.content or ""})
        elif isinstance(msg, dict):
            openai_messages.append(msg.copy())
        else:
            openai_messages.append({"role": "user", "content": str(msg)})

    # Get model context limit
    model_limit = provider.get_context_limit(model)

    # Apply transforms
    result = pipeline.apply(
        messages=openai_messages,
        model=model,
        model_limit=model_limit,
    )

    metrics = {
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_before - result.tokens_after,
        "savings_percent": (
            (result.tokens_before - result.tokens_after) / result.tokens_before * 100
            if result.tokens_before > 0
            else 0
        ),
        "transforms_applied": result.transforms_applied,
    }

    return result.messages, metrics
