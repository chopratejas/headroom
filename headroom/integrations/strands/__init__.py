"""Strands Agents SDK integration for Headroom.

This module provides seamless integration with AWS Strands Agents SDK,
enabling automatic context optimization for Strands agents.

Components:
1. HeadroomStrandsModel - Wraps any Strands model to apply Headroom transforms
2. HeadroomHookProvider - HookProvider that registers with Strands' hook system
3. create_headroom_callback - Creates hook provider with consistent config
4. optimize_messages - Standalone function for manual optimization

Reference: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/hooks/

Example using HookProvider (recommended):
    from strands import Agent
    from strands.models import BedrockModel
    from headroom.integrations.strands import HeadroomHookProvider

    # Create hook provider
    headroom_hook = HeadroomHookProvider(
        compress_tool_outputs=True,
        token_alert_threshold=10000,
    )

    # Use with agent - Strands calls register_hooks automatically
    agent = Agent(
        model=BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0"),
        hooks=[headroom_hook],
    )

    response = agent("Search for Python best practices")
    print(f"Tokens saved: {headroom_hook.total_tokens_saved}")

Example using Model Wrapper:
    from strands import Agent
    from strands.models import BedrockModel
    from headroom.integrations.strands import HeadroomStrandsModel

    # Wrap any Strands model
    model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    optimized_model = HeadroomStrandsModel(model)

    agent = Agent(model=optimized_model)
    response = agent("Hello!")
    print(f"Tokens saved: {optimized_model.total_tokens_saved}")

Install: pip install headroom-ai strands-agents
"""

from .hooks import (
    CallbackMetrics,
    HeadroomCallbackHandler,  # Alias for HeadroomHookProvider
    HeadroomHookProvider,
    create_headroom_callback,
)
from .model import (
    HeadroomStrandsModel,
    OptimizationMetrics,
    optimize_messages,
    strands_available,
)

__all__ = [
    # Model wrapper
    "HeadroomStrandsModel",
    "OptimizationMetrics",
    "strands_available",
    "optimize_messages",
    # Hook provider (Strands SDK pattern)
    "HeadroomHookProvider",
    "HeadroomCallbackHandler",  # Alias
    "CallbackMetrics",
    "create_headroom_callback",
]
