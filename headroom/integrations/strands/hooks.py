"""Strands Agents SDK hook provider for Headroom integration.

This module provides a HookProvider that integrates with Strands SDK's
hook system to apply Headroom optimization and track metrics.

Strands SDK uses a HookProvider/HookRegistry pattern with lifecycle events:
- BeforeInvocationEvent: Called before agent invocation starts
- AfterInvocationEvent: Called after agent invocation completes
- BeforeToolCallEvent: Called before a tool is executed
- AfterToolCallEvent: Called after a tool completes
- BeforeModelCallEvent: Called before model invocation
- AfterModelCallEvent: Called after model response

Reference: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/hooks/
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from headroom import HeadroomConfig, HeadroomMode
from headroom.providers import OpenAIProvider
from headroom.transforms import SmartCrusher, SmartCrusherConfig

logger = logging.getLogger(__name__)

# Type checking imports for Strands SDK
if TYPE_CHECKING:
    from strands.hooks import HookRegistry
    from strands.hooks.events import (
        AfterInvocationEvent,
        AfterModelCallEvent,
        AfterToolCallEvent,
        BeforeInvocationEvent,
        BeforeModelCallEvent,
        BeforeToolCallEvent,
    )

# Check if Strands is available
STRANDS_HOOKS_AVAILABLE = False
try:
    from strands.hooks import HookProvider, HookRegistry  # noqa: F811
    from strands.hooks.events import (  # noqa: F811
        AfterInvocationEvent,
        AfterModelCallEvent,
        AfterToolCallEvent,
        BeforeInvocationEvent,
        BeforeModelCallEvent,
        BeforeToolCallEvent,
    )

    STRANDS_HOOKS_AVAILABLE = True
except ImportError:
    # Create placeholder base class for when Strands is not installed
    class HookProvider:  # type: ignore[no-redef]
        """Placeholder HookProvider when Strands SDK is not installed."""

        def register_hooks(self, registry: Any, **kwargs: Any) -> None:
            pass


@dataclass
class CallbackMetrics:
    """Metrics collected by Headroom hook provider.

    Tracks token savings from tool output compression and context optimization.
    """

    request_id: str
    timestamp: datetime
    event_type: str  # "tool_compress", "context_optimize", "invocation_complete"
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    savings_percent: float = 0.0
    tool_name: str | None = None
    transforms_applied: list[str] = field(default_factory=list)


class HeadroomHookProvider(HookProvider):
    """Strands SDK HookProvider that applies Headroom optimization.

    This hook provider integrates with Strands SDK's hook system to:
    1. Compress tool outputs after they complete (AfterToolCallEvent)
    2. Track token savings across the agent lifecycle
    3. Provide observability into optimization effectiveness

    Reference: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/hooks/

    Example:
        from strands import Agent
        from strands.models import BedrockModel
        from headroom.integrations.strands import HeadroomHookProvider

        # Create hook provider
        headroom_hook = HeadroomHookProvider(
            compress_tool_outputs=True,
            token_alert_threshold=10000,
        )

        # Use with Strands agent
        agent = Agent(
            model=BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0"),
            hooks=[headroom_hook],
        )

        response = agent("Search for Python best practices")

        # Check savings
        print(f"Tokens saved: {headroom_hook.total_tokens_saved}")
        print(f"Tool compressions: {headroom_hook.tool_compression_count}")
    """

    def __init__(
        self,
        config: HeadroomConfig | None = None,
        mode: HeadroomMode = HeadroomMode.OPTIMIZE,
        model: str = "claude-3-sonnet-20240229",
        compress_tool_outputs: bool = True,
        min_tokens_to_compress: int = 500,
        max_items_after_crush: int = 50,
        log_level: str = "INFO",
        token_alert_threshold: int | None = None,
    ) -> None:
        """Initialize HeadroomHookProvider.

        Args:
            config: HeadroomConfig for optimization settings
            mode: HeadroomMode (AUDIT, OPTIMIZE, or SIMULATE)
            model: Default model name for token estimation
            compress_tool_outputs: Whether to compress tool outputs
            min_tokens_to_compress: Minimum tokens before compression kicks in
            max_items_after_crush: Max items to keep after crushing arrays
            log_level: Logging level ("DEBUG", "INFO", "WARNING")
            token_alert_threshold: Alert if request exceeds this many tokens
        """
        self.config = config or HeadroomConfig()
        self.mode = mode
        self.model = model
        self.compress_tool_outputs = compress_tool_outputs
        self.min_tokens_to_compress = min_tokens_to_compress
        self.max_items_after_crush = max_items_after_crush
        self.log_level = log_level
        self.token_alert_threshold = token_alert_threshold

        # Initialize SmartCrusher for tool output compression
        self._crusher = SmartCrusher(
            SmartCrusherConfig(
                min_tokens_to_crush=min_tokens_to_compress,
                max_items_after_crush=max_items_after_crush,
            )
        )
        self._provider = OpenAIProvider()

        # Metrics tracking
        self._metrics_history: list[CallbackMetrics] = []
        self._total_tokens_saved: int = 0
        self._tool_compression_count: int = 0
        self._alerts: list[str] = []
        self._lock = threading.Lock()

        # Current invocation tracking
        self._current_invocation_id: str | None = None
        self._invocation_tokens_before: int = 0

    # =========================================================================
    # HookProvider interface - register hooks with Strands SDK
    # =========================================================================

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register Headroom hooks with the Strands HookRegistry.

        This method is called by Strands SDK when the agent is initialized.
        We register callbacks for relevant lifecycle events.

        Args:
            registry: The Strands HookRegistry to register callbacks with
            **kwargs: Additional arguments for future compatibility
        """
        if not STRANDS_HOOKS_AVAILABLE:
            logger.warning(
                "Strands SDK not installed. Hooks will not be registered. "
                "Install with: pip install strands-agents"
            )
            return

        # Register lifecycle callbacks
        registry.add_callback(BeforeInvocationEvent, self._on_before_invocation)
        registry.add_callback(AfterInvocationEvent, self._on_after_invocation)
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)
        registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)
        registry.add_callback(AfterModelCallEvent, self._on_after_model_call)

        logger.info("Headroom hooks registered with Strands SDK")

    # =========================================================================
    # Public properties
    # =========================================================================

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all operations (thread-safe)."""
        with self._lock:
            return self._total_tokens_saved

    @property
    def tool_compression_count(self) -> int:
        """Number of tool outputs compressed (thread-safe)."""
        with self._lock:
            return self._tool_compression_count

    @property
    def metrics_history(self) -> list[CallbackMetrics]:
        """History of optimization metrics (thread-safe copy)."""
        with self._lock:
            return self._metrics_history.copy()

    @property
    def alerts(self) -> list[str]:
        """List of alerts triggered (thread-safe copy)."""
        with self._lock:
            return self._alerts.copy()

    # =========================================================================
    # Hook Callbacks
    # =========================================================================

    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Called before agent invocation starts.

        Args:
            event: The BeforeInvocationEvent from Strands SDK
        """
        self._current_invocation_id = str(uuid4())
        self._invocation_tokens_before = 0
        logger.debug(f"Headroom: Invocation started {self._current_invocation_id}")

    def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Called after agent invocation completes.

        Args:
            event: The AfterInvocationEvent from Strands SDK
        """
        if self._current_invocation_id:
            logger.debug(f"Headroom: Invocation completed {self._current_invocation_id}")
            self._current_invocation_id = None

    def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Called before a tool is executed.

        Args:
            event: The BeforeToolCallEvent from Strands SDK
        """
        tool_name = (
            event.tool_use.get("name", "unknown") if hasattr(event, "tool_use") else "unknown"
        )
        logger.debug(f"Headroom: Tool starting - {tool_name}")

    def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Called after a tool completes - compresses output if enabled.

        This is the main optimization point. Tool outputs are often large
        JSON responses that can be significantly compressed.

        Args:
            event: The AfterToolCallEvent from Strands SDK
        """
        if not self.compress_tool_outputs:
            return

        # Extract tool info from event
        tool_name = "unknown"
        if hasattr(event, "tool_use") and event.tool_use:
            tool_name = event.tool_use.get("name", "unknown")

        # Get the result from event
        if not hasattr(event, "result") or event.result is None:
            return

        result = event.result
        request_id = str(uuid4())

        # Extract content from result
        # Strands result format: {"content": [{"text": "..."}]}
        original_content = None
        if isinstance(result, dict):
            content_list = result.get("content", [])
            if content_list and isinstance(content_list, list):
                for item in content_list:
                    if isinstance(item, dict) and "text" in item:
                        original_content = item.get("text")
                        break
            if original_content is None:
                original_content = json.dumps(result)
        else:
            original_content = str(result)

        if not original_content:
            return

        # Estimate tokens
        tokens_before = len(original_content) // 4  # Rough estimate

        # Skip if below threshold
        if tokens_before < self.min_tokens_to_compress:
            logger.debug(
                f"Headroom: Tool output below threshold ({tokens_before} < {self.min_tokens_to_compress})"
            )
            return

        try:
            # Apply SmartCrusher compression
            crush_result = self._crusher.crush(original_content)
            compressed = crush_result.compressed
            strategy = crush_result.strategy
            tokens_after = len(compressed) // 4

            tokens_saved = max(0, tokens_before - tokens_after)
            savings_percent = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0

            # Create metrics
            metrics = CallbackMetrics(
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
                event_type="tool_compress",
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_saved=tokens_saved,
                savings_percent=savings_percent,
                tool_name=tool_name,
                transforms_applied=[f"smart_crush:{strategy}"] if strategy else [],
            )

            # Thread-safe metrics update
            with self._lock:
                self._metrics_history.append(metrics)
                self._total_tokens_saved += tokens_saved
                self._tool_compression_count += 1

                # Keep only last 100 metrics
                if len(self._metrics_history) > 100:
                    self._metrics_history = self._metrics_history[-100:]

            logger.info(
                f"Headroom: Compressed tool '{tool_name}' output: "
                f"{tokens_before} -> {tokens_after} tokens ({savings_percent:.1f}% saved)"
            )

            # Modify the event result with compressed content
            # This is the key feature - modifying event.result in AfterToolCallEvent
            if isinstance(result, dict) and "content" in result:
                content_list = result.get("content", [])
                if content_list and isinstance(content_list, list):
                    for item in content_list:
                        if isinstance(item, dict) and "text" in item:
                            item["text"] = compressed
                            break

        except Exception as e:
            logger.warning(f"Headroom: Tool compression failed for '{tool_name}': {e}")

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Called before model invocation.

        Can be used to track context size before LLM call.

        Args:
            event: The BeforeModelCallEvent from Strands SDK
        """
        # Track total tokens if available
        if hasattr(event, "messages") and event.messages:
            total_tokens = sum(
                len(str(m.get("content", ""))) // 4 for m in event.messages if isinstance(m, dict)
            )

            if self.token_alert_threshold and total_tokens > self.token_alert_threshold:
                alert = (
                    f"Token alert: {total_tokens} tokens exceeds "
                    f"threshold {self.token_alert_threshold}"
                )
                with self._lock:
                    self._alerts.append(alert)
                logger.warning(alert)

    def _on_after_model_call(self, event: AfterModelCallEvent) -> None:
        """Called after model response.

        Args:
            event: The AfterModelCallEvent from Strands SDK
        """
        # Could track response tokens here if needed
        pass

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_savings_summary(self) -> dict[str, Any]:
        """Get summary of token savings (thread-safe)."""
        with self._lock:
            if not self._metrics_history:
                return {
                    "total_events": 0,
                    "total_tokens_saved": 0,
                    "tool_compressions": 0,
                    "average_savings_percent": 0,
                    "alerts": len(self._alerts),
                }

            tool_metrics = [m for m in self._metrics_history if m.event_type == "tool_compress"]

            return {
                "total_events": len(self._metrics_history),
                "total_tokens_saved": self._total_tokens_saved,
                "tool_compressions": self._tool_compression_count,
                "average_savings_percent": (
                    sum(m.savings_percent for m in tool_metrics) / len(tool_metrics)
                    if tool_metrics
                    else 0
                ),
                "alerts": len(self._alerts),
            }

    def reset(self) -> None:
        """Reset all tracked metrics (thread-safe)."""
        with self._lock:
            self._metrics_history = []
            self._total_tokens_saved = 0
            self._tool_compression_count = 0
            self._alerts = []


# Alias for backwards compatibility
HeadroomCallbackHandler = HeadroomHookProvider


def create_headroom_callback(
    config: HeadroomConfig | None = None,
    mode: HeadroomMode = HeadroomMode.OPTIMIZE,
    model: str = "claude-3-sonnet-20240229",
    compress_tool_outputs: bool = True,
    log_level: str = "INFO",
    token_alert_threshold: int | None = None,
) -> HeadroomHookProvider:
    """Create a Headroom hook provider for Strands agents.

    This is a convenience function to create a hook provider with
    sensible defaults for Strands SDK integration.

    Reference: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/hooks/

    Args:
        config: HeadroomConfig for optimization settings
        mode: HeadroomMode (AUDIT, OPTIMIZE, or SIMULATE)
        model: Default model name for token estimation
        compress_tool_outputs: Whether to compress tool outputs
        log_level: Logging level for the handler
        token_alert_threshold: Alert threshold for token usage

    Returns:
        Configured HeadroomHookProvider

    Example:
        from strands import Agent
        from strands.models import BedrockModel
        from headroom.integrations.strands import create_headroom_callback

        headroom_hook = create_headroom_callback(
            token_alert_threshold=10000,
        )

        agent = Agent(
            model=BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0"),
            hooks=[headroom_hook],
        )

        response = agent("Analyze the server logs")
        print(f"Tokens saved: {headroom_hook.total_tokens_saved}")
    """
    return HeadroomHookProvider(
        config=config,
        mode=mode,
        model=model,
        compress_tool_outputs=compress_tool_outputs,
        log_level=log_level,
        token_alert_threshold=token_alert_threshold,
    )
