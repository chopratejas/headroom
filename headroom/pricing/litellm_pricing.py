"""LiteLLM-based pricing for model cost estimation.

Uses LiteLLM's community-maintained model cost database instead of
hardcoded values. This provides up-to-date pricing for 100+ models.

See: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import litellm


@dataclass
class LiteLLMModelPricing:
    """Pricing information from LiteLLM's database.

    All costs are in USD per 1 million tokens.
    """

    model: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_vision: bool = False
    supports_function_calling: bool = False


def get_litellm_model_cost() -> dict[str, Any]:
    """Get LiteLLM's full model cost dictionary.

    Returns:
        Dictionary mapping model names to their pricing/capability info.
    """
    return litellm.model_cost


def get_model_pricing(model: str) -> LiteLLMModelPricing | None:
    """Get pricing for a model from LiteLLM's database.

    Args:
        model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022').

    Returns:
        LiteLLMModelPricing if found, None otherwise.
    """
    cost_data = litellm.model_cost

    # Try exact match first
    info = cost_data.get(model)

    # Try common provider prefixes if not found
    if info is None:
        for prefix in ["openai/", "anthropic/", "google/", "mistral/", "deepseek/"]:
            if f"{prefix}{model}" in cost_data:
                info = cost_data[f"{prefix}{model}"]
                break

    if info is None:
        return None

    # LiteLLM stores cost per token, convert to per 1M
    input_per_token = info.get("input_cost_per_token", 0) or 0
    output_per_token = info.get("output_cost_per_token", 0) or 0

    return LiteLLMModelPricing(
        model=model,
        input_cost_per_1m=input_per_token * 1_000_000,
        output_cost_per_1m=output_per_token * 1_000_000,
        max_tokens=info.get("max_tokens"),
        max_input_tokens=info.get("max_input_tokens"),
        max_output_tokens=info.get("max_output_tokens"),
        supports_vision=info.get("supports_vision", False),
        supports_function_calling=info.get("supports_function_calling", False),
    )


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float | None:
    """Estimate cost for a model using LiteLLM's pricing.

    Args:
        model: Model name.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD, or None if model not found.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m
    return input_cost + output_cost


def list_available_models() -> list[str]:
    """List all models with pricing info in LiteLLM's database.

    Returns:
        List of model names.
    """
    return list(litellm.model_cost.keys())
