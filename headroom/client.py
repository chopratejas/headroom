"""Main HeadroomClient implementation for Headroom SDK."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator

from .config import (
    HeadroomConfig,
    HeadroomMode,
    RequestMetrics,
    SimulationResult,
)
from .parser import parse_messages
from .providers.base import Provider
from .storage import create_storage
from .tokenizer import Tokenizer
from .transforms import CacheAligner, TransformPipeline
from .utils import (
    compute_messages_hash,
    compute_prefix_hash,
    estimate_cost,
    format_cost,
    generate_request_id,
)


class ChatCompletions:
    """Wrapper for chat.completions API (OpenAI-style)."""

    def __init__(self, client: HeadroomClient):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        # Headroom-specific parameters
        headroom_mode: str | None = None,
        headroom_cache_prefix_tokens: int | None = None,
        headroom_output_buffer_tokens: int | None = None,
        headroom_keep_turns: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        # Pass through all other kwargs
        **kwargs: Any,
    ) -> Any:
        """
        Create a chat completion with optional Headroom optimization.

        Args:
            model: Model name.
            messages: List of messages.
            stream: Whether to stream the response.
            headroom_mode: Override default mode ("audit" | "optimize").
            headroom_cache_prefix_tokens: Target cache-aligned prefix size.
            headroom_output_buffer_tokens: Reserve tokens for output.
            headroom_keep_turns: Never drop last N turns.
            headroom_tool_profiles: Per-tool compression config.
            **kwargs: Additional arguments passed to underlying client.

        Returns:
            Chat completion response (or stream iterator).
        """
        return self._client._create(
            model=model,
            messages=messages,
            stream=stream,
            headroom_mode=headroom_mode,
            headroom_cache_prefix_tokens=headroom_cache_prefix_tokens,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_keep_turns=headroom_keep_turns,
            headroom_tool_profiles=headroom_tool_profiles,
            api_style="openai",
            **kwargs,
        )

    def simulate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headroom_mode: str = "optimize",
        headroom_output_buffer_tokens: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Simulate optimization without calling the API.

        Args:
            model: Model name.
            messages: List of messages.
            headroom_mode: Mode to simulate.
            headroom_output_buffer_tokens: Output buffer to use.
            headroom_tool_profiles: Tool profiles to use.
            **kwargs: Additional arguments (ignored).

        Returns:
            SimulationResult with projected changes.
        """
        return self._client._simulate(
            model=model,
            messages=messages,
            headroom_mode=headroom_mode,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_tool_profiles=headroom_tool_profiles,
        )


class Messages:
    """Wrapper for messages API (Anthropic-style)."""

    def __init__(self, client: HeadroomClient):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        # Headroom-specific parameters
        headroom_mode: str | None = None,
        headroom_cache_prefix_tokens: int | None = None,
        headroom_output_buffer_tokens: int | None = None,
        headroom_keep_turns: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        # Pass through all other kwargs
        **kwargs: Any,
    ) -> Any:
        """
        Create a message with optional Headroom optimization.

        Args:
            model: Model name.
            messages: List of messages.
            max_tokens: Maximum tokens in response.
            headroom_mode: Override default mode ("audit" | "optimize").
            headroom_cache_prefix_tokens: Target cache-aligned prefix size.
            headroom_output_buffer_tokens: Reserve tokens for output.
            headroom_keep_turns: Never drop last N turns.
            headroom_tool_profiles: Per-tool compression config.
            **kwargs: Additional arguments passed to underlying client.

        Returns:
            Message response.
        """
        return self._client._create(
            model=model,
            messages=messages,
            stream=False,
            headroom_mode=headroom_mode,
            headroom_cache_prefix_tokens=headroom_cache_prefix_tokens,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_keep_turns=headroom_keep_turns,
            headroom_tool_profiles=headroom_tool_profiles,
            api_style="anthropic",
            max_tokens=max_tokens,
            **kwargs,
        )

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        # Headroom-specific parameters
        headroom_mode: str | None = None,
        headroom_cache_prefix_tokens: int | None = None,
        headroom_output_buffer_tokens: int | None = None,
        headroom_keep_turns: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        # Pass through all other kwargs
        **kwargs: Any,
    ) -> Any:
        """
        Stream a message with optional Headroom optimization.

        Args:
            model: Model name.
            messages: List of messages.
            max_tokens: Maximum tokens in response.
            headroom_mode: Override default mode ("audit" | "optimize").
            headroom_cache_prefix_tokens: Target cache-aligned prefix size.
            headroom_output_buffer_tokens: Reserve tokens for output.
            headroom_keep_turns: Never drop last N turns.
            headroom_tool_profiles: Per-tool compression config.
            **kwargs: Additional arguments passed to underlying client.

        Returns:
            Stream context manager.
        """
        return self._client._create(
            model=model,
            messages=messages,
            stream=True,
            headroom_mode=headroom_mode,
            headroom_cache_prefix_tokens=headroom_cache_prefix_tokens,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_keep_turns=headroom_keep_turns,
            headroom_tool_profiles=headroom_tool_profiles,
            api_style="anthropic",
            max_tokens=max_tokens,
            **kwargs,
        )

    def simulate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headroom_mode: str = "optimize",
        headroom_output_buffer_tokens: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Simulate optimization without calling the API.

        Args:
            model: Model name.
            messages: List of messages.
            headroom_mode: Mode to simulate.
            headroom_output_buffer_tokens: Output buffer to use.
            headroom_tool_profiles: Tool profiles to use.
            **kwargs: Additional arguments (ignored).

        Returns:
            SimulationResult with projected changes.
        """
        return self._client._simulate(
            model=model,
            messages=messages,
            headroom_mode=headroom_mode,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_tool_profiles=headroom_tool_profiles,
        )


class HeadroomClient:
    """
    Context Budget Controller wrapper for LLM API clients.

    Provides automatic context optimization, waste detection, and
    cache alignment while maintaining API compatibility.
    """

    def __init__(
        self,
        original_client: Any,
        provider: Provider,
        store_url: str = "sqlite:///headroom.db",
        default_mode: str = "audit",
        model_context_limits: dict[str, int] | None = None,
    ):
        """
        Initialize HeadroomClient.

        Args:
            original_client: The underlying LLM client (OpenAI-compatible).
            provider: Provider instance for model-specific behavior.
            store_url: Storage URL (sqlite:// or jsonl://).
            default_mode: Default mode ("audit" | "optimize").
            model_context_limits: Override context limits for models.
        """
        self._original = original_client
        self._provider = provider
        self._store_url = store_url
        self._default_mode = HeadroomMode(default_mode)

        # Build config
        self._config = HeadroomConfig()
        self._config.store_url = store_url
        self._config.default_mode = self._default_mode

        if model_context_limits:
            self._config.model_context_limits.update(model_context_limits)

        # Initialize storage
        self._storage = create_storage(store_url)

        # Initialize transform pipeline
        self._pipeline = TransformPipeline(self._config, provider=self._provider)

        # Public API - OpenAI style
        self.chat = type("Chat", (), {"completions": ChatCompletions(self)})()
        # Public API - Anthropic style
        self.messages = Messages(self)

    def _get_tokenizer(self, model: str) -> Tokenizer:
        """Get tokenizer for model using provider."""
        token_counter = self._provider.get_token_counter(model)
        return Tokenizer(token_counter, model)

    def _get_context_limit(self, model: str) -> int:
        """Get context limit from user config or provider."""
        # User override takes precedence
        limit = self._config.get_context_limit(model)
        if limit is not None:
            return limit
        # Fall back to provider
        return self._provider.get_context_limit(model)

    def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        headroom_mode: str | None = None,
        headroom_cache_prefix_tokens: int | None = None,
        headroom_output_buffer_tokens: int | None = None,
        headroom_keep_turns: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        api_style: str = "openai",
        **kwargs: Any,
    ) -> Any:
        """Internal implementation of create."""
        request_id = generate_request_id()
        timestamp = datetime.utcnow()
        mode = HeadroomMode(headroom_mode) if headroom_mode else self._default_mode

        tokenizer = self._get_tokenizer(model)

        # Analyze original messages
        blocks, block_breakdown, waste_signals = parse_messages(
            messages, tokenizer
        )
        tokens_before = tokenizer.count_messages(messages)

        # Compute cache alignment score
        aligner = CacheAligner(self._config.cache_aligner)
        cache_alignment_score = aligner.get_alignment_score(messages)

        # Compute stable prefix hash
        stable_prefix_hash = compute_prefix_hash(messages)

        # Apply transforms if in optimize mode
        if mode == HeadroomMode.OPTIMIZE:
            output_buffer = headroom_output_buffer_tokens or self._config.rolling_window.output_buffer_tokens
            model_limit = self._get_context_limit(model)

            result = self._pipeline.apply(
                messages,
                model,
                model_limit=model_limit,
                output_buffer=output_buffer,
                tool_profiles=headroom_tool_profiles or {},
            )

            optimized_messages = result.messages
            tokens_after = result.tokens_after
            transforms_applied = result.transforms_applied

            # Recalculate prefix hash after optimization
            stable_prefix_hash = compute_prefix_hash(optimized_messages)
        else:
            # Audit mode - no changes
            optimized_messages = messages
            tokens_after = tokens_before
            transforms_applied = []

        # Create metrics
        metrics = RequestMetrics(
            request_id=request_id,
            timestamp=timestamp,
            model=model,
            stream=stream,
            mode=mode.value,
            tokens_input_before=tokens_before,
            tokens_input_after=tokens_after,
            block_breakdown=block_breakdown,
            waste_signals=waste_signals.to_dict(),
            stable_prefix_hash=stable_prefix_hash,
            cache_alignment_score=cache_alignment_score,
            transforms_applied=transforms_applied,
            messages_hash=compute_messages_hash(messages),
        )

        # Call underlying client based on API style
        try:
            if api_style == "anthropic":
                return self._call_anthropic(
                    model=model,
                    messages=optimized_messages,
                    stream=stream,
                    metrics=metrics,
                    **kwargs,
                )
            else:
                return self._call_openai(
                    model=model,
                    messages=optimized_messages,
                    stream=stream,
                    metrics=metrics,
                    **kwargs,
                )

        except Exception as e:
            metrics.error = str(e)
            self._storage.save(metrics)
            raise

    def _call_openai(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        metrics: RequestMetrics,
        **kwargs: Any,
    ) -> Any:
        """Call OpenAI-style API."""
        if stream:
            response = self._original.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            return self._wrap_stream(response, metrics)
        else:
            response = self._original.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs,
            )

            # Extract output tokens from response
            if hasattr(response, "usage") and response.usage:
                metrics.tokens_output = response.usage.completion_tokens
                # Check for cached tokens in usage
                if hasattr(response.usage, "prompt_tokens_details"):
                    details = response.usage.prompt_tokens_details
                    if hasattr(details, "cached_tokens"):
                        metrics.cached_tokens = details.cached_tokens

            self._storage.save(metrics)
            return response

    def _call_anthropic(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        metrics: RequestMetrics,
        **kwargs: Any,
    ) -> Any:
        """Call Anthropic-style API."""
        if stream:
            # Anthropic streaming returns a context manager
            stream_manager = self._original.messages.stream(
                model=model,
                messages=messages,
                **kwargs,
            )
            # Save metrics when stream is created
            self._storage.save(metrics)
            return stream_manager
        else:
            response = self._original.messages.create(
                model=model,
                messages=messages,
                **kwargs,
            )

            # Extract output tokens from Anthropic response
            if hasattr(response, "usage") and response.usage:
                metrics.tokens_output = response.usage.output_tokens
                # Check for cached tokens in Anthropic usage
                if hasattr(response.usage, "cache_read_input_tokens"):
                    metrics.cached_tokens = response.usage.cache_read_input_tokens

            self._storage.save(metrics)
            return response

    def _wrap_stream(
        self,
        stream: Iterator[Any],
        metrics: RequestMetrics,
    ) -> Iterator[Any]:
        """Wrap stream to pass through chunks and save metrics at end."""
        try:
            for chunk in stream:
                yield chunk
        finally:
            # Save metrics when stream completes
            # Note: output tokens unknown for streams
            self._storage.save(metrics)

    def _simulate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headroom_mode: str = "optimize",
        headroom_output_buffer_tokens: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
    ) -> SimulationResult:
        """Internal implementation of simulate."""
        tokenizer = self._get_tokenizer(model)

        # Analyze original
        blocks, block_breakdown, waste_signals = parse_messages(
            messages, tokenizer
        )
        tokens_before = tokenizer.count_messages(messages)

        # Compute original cache alignment
        aligner = CacheAligner(self._config.cache_aligner)
        cache_alignment_score = aligner.get_alignment_score(messages)
        stable_prefix_hash = compute_prefix_hash(messages)

        # Apply transforms
        output_buffer = headroom_output_buffer_tokens or self._config.rolling_window.output_buffer_tokens
        model_limit = self._get_context_limit(model)

        result = self._pipeline.simulate(
            messages,
            model,
            model_limit=model_limit,
            output_buffer=output_buffer,
            tool_profiles=headroom_tool_profiles or {},
        )

        tokens_saved = tokens_before - result.tokens_after

        # Estimate cost savings using provider
        cost_before = estimate_cost(tokens_before, 500, model, provider=self._provider)
        cost_after = estimate_cost(result.tokens_after, 500, model, provider=self._provider)

        if cost_before is not None and cost_after is not None:
            savings = format_cost(cost_before - cost_after)
        else:
            savings = "N/A"

        # Recalculate prefix hash after optimization
        optimized_prefix_hash = compute_prefix_hash(result.messages)

        return SimulationResult(
            tokens_before=tokens_before,
            tokens_after=result.tokens_after,
            tokens_saved=tokens_saved,
            transforms=result.transforms_applied,
            estimated_savings=f"{savings} per request",
            messages_optimized=result.messages,
            block_breakdown=block_breakdown,
            waste_signals=waste_signals.to_dict(),
            stable_prefix_hash=optimized_prefix_hash,
            cache_alignment_score=cache_alignment_score,
        )

    def get_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
        limit: int = 100,
    ) -> list[RequestMetrics]:
        """
        Query stored metrics.

        Args:
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.
            model: Filter by model name.
            mode: Filter by mode.
            limit: Maximum results.

        Returns:
            List of RequestMetrics.
        """
        return self._storage.query(
            start_time=start_time,
            end_time=end_time,
            model=model,
            mode=mode,
            limit=limit,
        )

    def get_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get summary statistics.

        Args:
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.

        Returns:
            Summary statistics dict.
        """
        return self._storage.get_summary_stats(start_time, end_time)

    def close(self) -> None:
        """Close storage connection."""
        self._storage.close()

    def __enter__(self) -> HeadroomClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
