"""Agno model wrapper for Headroom optimization.

This module provides HeadroomAgnoModel, which wraps any Agno model
to apply Headroom context optimization before API calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

# Agno imports - these are optional dependencies
try:
    from agno.models.base import Model
    from agno.models.message import Message
    from agno.models.response import ModelResponse

    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    Model = object  # type: ignore[misc,assignment]
    Message = dict  # type: ignore[misc,assignment]
    ModelResponse = dict  # type: ignore[misc,assignment]

from headroom import HeadroomConfig, HeadroomMode
from headroom.providers import OpenAIProvider
from headroom.transforms import TransformPipeline

from .providers import get_headroom_provider, get_model_name_from_agno

logger = logging.getLogger(__name__)


def _check_agno_available() -> None:
    """Raise ImportError if Agno is not installed."""
    if not AGNO_AVAILABLE:
        raise ImportError("Agno is required for this integration. Install with: pip install agno")


def agno_available() -> bool:
    """Check if Agno is installed."""
    return AGNO_AVAILABLE


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


class HeadroomAgnoModel:
    """Agno model wrapper that applies Headroom optimizations.

    Wraps any Agno Model and automatically optimizes the context
    before each API call. Works with OpenAIChat, Claude, Gemini, and
    other Agno model types.

    Example:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from headroom.integrations.agno import HeadroomAgnoModel

        # Basic usage
        model = OpenAIChat(id="gpt-4o")
        optimized = HeadroomAgnoModel(model)

        # Use with agent
        agent = Agent(model=optimized)
        response = agent.run("Hello!")

        # Access metrics
        print(f"Saved {optimized.total_tokens_saved} tokens")

        # With custom config
        from headroom import HeadroomConfig, HeadroomMode
        config = HeadroomConfig(default_mode=HeadroomMode.OPTIMIZE)
        optimized = HeadroomAgnoModel(model, config=config)

    Attributes:
        wrapped_model: The underlying Agno model
        total_tokens_saved: Running total of tokens saved
        metrics_history: List of OptimizationMetrics from recent calls
    """

    def __init__(
        self,
        wrapped_model: Any,
        config: HeadroomConfig | None = None,
        mode: HeadroomMode | None = None,
        auto_detect_provider: bool = True,
    ) -> None:
        """Initialize HeadroomAgnoModel.

        Args:
            wrapped_model: Any Agno Model to wrap (OpenAIChat, Claude, etc.)
            config: HeadroomConfig for optimization settings. Use
                config.default_mode to set the optimization mode.
            mode: Deprecated. Use config.default_mode instead.
            auto_detect_provider: Auto-detect provider from wrapped model.
                When True (default), automatically detects if the wrapped model
                is OpenAI, Anthropic, Google, etc. and uses the appropriate
                Headroom provider for accurate token counting.
        """
        _check_agno_available()

        if wrapped_model is None:
            raise ValueError("wrapped_model cannot be None")

        if mode is not None:
            warnings.warn(
                "The 'mode' parameter is deprecated. Use HeadroomConfig(default_mode=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.wrapped_model = wrapped_model
        self.config = config or HeadroomConfig()
        self.mode = mode or HeadroomMode.OPTIMIZE  # Kept for backwards compatibility
        self.auto_detect_provider = auto_detect_provider

        self._metrics_history: list[OptimizationMetrics] = []
        self._total_tokens_saved: int = 0
        self._pipeline: TransformPipeline | None = None
        self._provider: Any = None
        self._lock = threading.Lock()  # Thread safety for metrics

    # Forward all attribute access to wrapped model for compatibility
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

    @property
    def pipeline(self) -> TransformPipeline:
        """Lazily initialize TransformPipeline (thread-safe)."""
        if self._pipeline is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._pipeline is None:
                    if self.auto_detect_provider:
                        self._provider = get_headroom_provider(self.wrapped_model)
                        logger.debug(f"Auto-detected provider: {self._provider.__class__.__name__}")
                    else:
                        self._provider = OpenAIProvider()
                    self._pipeline = TransformPipeline(
                        config=self.config,
                        provider=self._provider,
                    )
        return self._pipeline

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all calls."""
        return self._total_tokens_saved

    @property
    def metrics_history(self) -> list[OptimizationMetrics]:
        """History of optimization metrics."""
        return self._metrics_history.copy()

    def _convert_messages_to_openai(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert Agno messages to OpenAI format for Headroom."""
        result = []
        for msg in messages:
            # Handle Agno Message objects
            if hasattr(msg, "role") and hasattr(msg, "content"):
                entry: dict[str, Any] = {
                    "role": msg.role,
                    "content": msg.content if msg.content is not None else "",
                }
                # Handle tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = msg.tool_calls
                # Handle tool call ID for tool responses
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id
                result.append(entry)
            # Handle dict format
            elif isinstance(msg, dict):
                result.append(msg.copy())
            else:
                # Try to extract content
                content = str(msg) if msg is not None else ""
                result.append({"role": "user", "content": content})
        return result

    def _convert_messages_from_openai(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Convert OpenAI format messages back to Agno format.

        Note: Agno typically accepts OpenAI-format dicts directly,
        so we may not need full conversion.
        """
        # Agno models generally accept OpenAI-format messages
        # Return as-is for compatibility
        return messages

    def _optimize_messages(self, messages: list[Any]) -> tuple[list[Any], OptimizationMetrics]:
        """Apply Headroom optimization to messages.

        Thread-safe with fallback on pipeline errors.
        """
        request_id = str(uuid4())

        # Convert to OpenAI format
        openai_messages = self._convert_messages_to_openai(messages)

        # Handle empty messages gracefully
        if not openai_messages:
            metrics = OptimizationMetrics(
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
                tokens_before=0,
                tokens_after=0,
                tokens_saved=0,
                savings_percent=0,
                transforms_applied=[],
                model=get_model_name_from_agno(self.wrapped_model),
            )
            return openai_messages, metrics

        # Get model name from wrapped model
        model = get_model_name_from_agno(self.wrapped_model)

        # Ensure pipeline is initialized
        _ = self.pipeline

        # Get model context limit
        model_limit = self._provider.get_context_limit(model) if self._provider else 128000

        try:
            # Apply Headroom transforms via pipeline
            result = self.pipeline.apply(
                messages=openai_messages,
                model=model,
                model_limit=model_limit,
            )
            optimized = result.messages
            tokens_before = result.tokens_before
            tokens_after = result.tokens_after
            transforms_applied = result.transforms_applied
        except (
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            KeyError,
            IndexError,
            ImportError,
            OSError,
        ) as e:
            # Fallback to original messages on pipeline error
            # Log at warning level (degraded behavior, not critical failure)
            logger.warning(
                f"Headroom optimization failed, using original messages: {type(e).__name__}: {e}"
            )
            optimized = openai_messages
            # Estimate token count for unoptimized messages (rough approximation)
            # Note: This uses ~4 chars/token which is approximate for English text
            tokens_before = sum(len(str(m.get("content", ""))) // 4 for m in openai_messages)
            tokens_after = tokens_before  # No optimization occurred
            transforms_applied = ["fallback:error"]

        # Create metrics
        tokens_saved = max(0, tokens_before - tokens_after)  # Never negative
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

        # Track metrics (thread-safe)
        with self._lock:
            self._metrics_history.append(metrics)
            self._total_tokens_saved += metrics.tokens_saved

            # Keep only last 100 metrics
            if len(self._metrics_history) > 100:
                self._metrics_history = self._metrics_history[-100:]

        # Convert back (Agno accepts OpenAI format)
        optimized_messages = self._convert_messages_from_openai(optimized)

        return optimized_messages, metrics

    def response(self, messages: list[Any], **kwargs: Any) -> Any:
        """Generate response with Headroom optimization.

        This is the core method that Agno agents call.
        """
        # Optimize messages
        optimized_messages, metrics = self._optimize_messages(messages)

        logger.info(
            f"Headroom optimized: {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.savings_percent:.1f}% saved)"
        )

        # Call wrapped model with optimized messages
        return self.wrapped_model.response(optimized_messages, **kwargs)

    def response_stream(self, messages: list[Any], **kwargs: Any) -> Iterator[Any]:
        """Stream response with Headroom optimization."""
        # Optimize messages
        optimized_messages, metrics = self._optimize_messages(messages)

        logger.info(
            f"Headroom optimized (streaming): {metrics.tokens_before} -> "
            f"{metrics.tokens_after} tokens"
        )

        # Stream from wrapped model
        yield from self.wrapped_model.response_stream(optimized_messages, **kwargs)

    async def aresponse(self, messages: list[Any], **kwargs: Any) -> Any:
        """Async generate response with Headroom optimization."""
        # Run optimization in executor (CPU-bound)
        loop = asyncio.get_running_loop()
        optimized_messages, metrics = await loop.run_in_executor(
            None, self._optimize_messages, messages
        )

        logger.info(
            f"Headroom optimized (async): {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.savings_percent:.1f}% saved)"
        )

        # Call wrapped model's async method
        if hasattr(self.wrapped_model, "aresponse"):
            return await self.wrapped_model.aresponse(optimized_messages, **kwargs)
        else:
            # Fallback to sync in executor (non-blocking)
            return await loop.run_in_executor(
                None, lambda: self.wrapped_model.response(optimized_messages, **kwargs)
            )

    async def aresponse_stream(self, messages: list[Any], **kwargs: Any) -> AsyncIterator[Any]:
        """Async stream response with Headroom optimization."""
        # Run optimization in executor (CPU-bound)
        loop = asyncio.get_running_loop()
        optimized_messages, metrics = await loop.run_in_executor(
            None, self._optimize_messages, messages
        )

        logger.info(
            f"Headroom optimized (async streaming): {metrics.tokens_before} -> "
            f"{metrics.tokens_after} tokens"
        )

        # Async stream from wrapped model
        if hasattr(self.wrapped_model, "aresponse_stream"):
            async for chunk in self.wrapped_model.aresponse_stream(optimized_messages, **kwargs):
                yield chunk
        else:
            # Fallback: wrap sync streaming in async iterator (non-blocking)
            # Run the entire sync iteration in executor to avoid blocking event loop
            def _sync_stream() -> list[Any]:
                return list(self.wrapped_model.response_stream(optimized_messages, **kwargs))

            chunks = await loop.run_in_executor(None, _sync_stream)
            for chunk in chunks:
                yield chunk

    def get_savings_summary(self) -> dict[str, Any]:
        """Get summary of token savings."""
        if not self._metrics_history:
            return {
                "total_requests": 0,
                "total_tokens_saved": 0,
                "average_savings_percent": 0,
            }

        return {
            "total_requests": len(self._metrics_history),
            "total_tokens_saved": self._total_tokens_saved,
            "average_savings_percent": sum(m.savings_percent for m in self._metrics_history)
            / len(self._metrics_history),
            "total_tokens_before": sum(m.tokens_before for m in self._metrics_history),
            "total_tokens_after": sum(m.tokens_after for m in self._metrics_history),
        }

    def reset(self) -> None:
        """Reset all tracked metrics (thread-safe).

        Clears the metrics history and resets the total tokens saved counter.
        Useful for starting fresh measurements or between test runs.
        """
        with self._lock:
            self._metrics_history = []
            self._total_tokens_saved = 0


def optimize_messages(
    messages: list[Any],
    config: HeadroomConfig | None = None,
    mode: HeadroomMode = HeadroomMode.OPTIMIZE,
    model: str = "gpt-4o",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Standalone function to optimize Agno messages.

    Use this for manual optimization when you need fine-grained control.

    Args:
        messages: List of Agno Message objects or dicts
        config: HeadroomConfig for optimization settings
        mode: HeadroomMode (AUDIT, OPTIMIZE, or SIMULATE)
        model: Model name for token estimation

    Returns:
        Tuple of (optimized_messages, metrics_dict)

    Example:
        from headroom.integrations.agno import optimize_messages

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        optimized, metrics = optimize_messages(messages)
        print(f"Saved {metrics['tokens_saved']} tokens")
    """
    _check_agno_available()

    config = config or HeadroomConfig()
    provider = OpenAIProvider()
    pipeline = TransformPipeline(config=config, provider=provider)

    # Convert to OpenAI format
    openai_messages = []
    for msg in messages:
        if hasattr(msg, "role") and hasattr(msg, "content"):
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            openai_messages.append(entry)
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
