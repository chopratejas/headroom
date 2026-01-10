"""OpenAI provider implementation for Headroom SDK.

Token counting is accurate (uses tiktoken).
Cost estimates are APPROXIMATE - always verify against your actual billing.
"""

from __future__ import annotations

import warnings
from datetime import date
from functools import lru_cache
from typing import Any

from .base import Provider, TokenCounter

# Pricing metadata for transparency
_PRICING_LAST_UPDATED = date(2024, 12, 1)
_PRICING_STALE_DAYS = 60  # Warn if pricing data is older than this

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# OpenAI model to tiktoken encoding mappings
_MODEL_ENCODINGS: dict[str, str] = {
    # GPT-4o and newer use o200k_base
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024": "o200k_base",
    "o1": "o200k_base",
    "o1-preview": "o200k_base",
    "o1-mini": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    # GPT-4 and GPT-3.5 use cl100k_base
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
}

# OpenAI context window limits
_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-2024-11-20": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "o1": 200000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 200000,
}

# OpenAI pricing per 1M tokens (input, output)
# NOTE: These are ESTIMATES. Always verify against actual OpenAI billing.
# Last updated: 2024-12-01
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-preview": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3": (10.00, 40.00),  # Estimated based on capability tier
    "o3-mini": (1.10, 4.40),  # Estimated based on capability tier
}

# Track if staleness warning has been shown
_PRICING_WARNING_SHOWN = False


def _check_pricing_staleness() -> str | None:
    """Check if pricing data is stale and return warning message if so."""
    global _PRICING_WARNING_SHOWN
    days_old = (date.today() - _PRICING_LAST_UPDATED).days
    if days_old > _PRICING_STALE_DAYS and not _PRICING_WARNING_SHOWN:
        _PRICING_WARNING_SHOWN = True
        return (
            f"OpenAI pricing data is {days_old} days old. "
            "Cost estimates may be inaccurate. Verify against actual billing."
        )
    return None


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str) -> Any:
    """Get tiktoken encoding, cached."""
    if not TIKTOKEN_AVAILABLE:
        raise RuntimeError(
            "tiktoken is required for OpenAI provider. Install with: pip install tiktoken"
        )
    return tiktoken.get_encoding(encoding_name)


def _get_encoding_name_for_model(model: str) -> str:
    """Get the encoding name for a model."""
    # Direct match
    if model in _MODEL_ENCODINGS:
        return _MODEL_ENCODINGS[model]

    # Prefix match for versioned models
    for prefix, encoding in _MODEL_ENCODINGS.items():
        if model.startswith(prefix):
            return encoding

    raise ValueError(
        f"Unknown OpenAI model: {model}. Supported models: {list(_MODEL_ENCODINGS.keys())}"
    )


class OpenAITokenCounter:
    """Token counter using tiktoken for OpenAI models."""

    def __init__(self, model: str):
        """
        Initialize token counter for a model.

        Args:
            model: OpenAI model name.

        Raises:
            ValueError: If model is not supported.
            RuntimeError: If tiktoken is not installed.
        """
        self.model = model
        encoding_name = _get_encoding_name_for_model(model)
        self._encoding = _get_encoding(encoding_name)

    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_message(self, message: dict[str, Any]) -> int:
        """
        Count tokens in a single message.

        Accounts for ChatML format overhead.
        """
        # Base overhead per message (role + delimiters)
        tokens = 4

        role = message.get("role", "")
        tokens += self.count_text(role)

        content = message.get("content")
        if content:
            if isinstance(content, str):
                tokens += self.count_text(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            tokens += self.count_text(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            tokens += 85  # Low detail image estimate
                    elif isinstance(part, str):
                        tokens += self.count_text(part)

        # Name field
        name = message.get("name")
        if name:
            tokens += self.count_text(name) + 1

        # Tool calls in assistant messages
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                tokens += self.count_text(func.get("name", ""))
                tokens += self.count_text(func.get("arguments", ""))
                tokens += self.count_text(tc.get("id", ""))
                tokens += 10  # Structural overhead

        # Tool call ID for tool responses
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            tokens += self.count_text(tool_call_id) + 2

        return tokens

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        total = sum(self.count_message(msg) for msg in messages)
        # Add priming tokens for assistant response
        total += 3
        return total


class OpenAIProvider(Provider):
    """Provider implementation for OpenAI models."""

    @property
    def name(self) -> str:
        return "openai"

    def supports_model(self, model: str) -> bool:
        """Check if model is a known OpenAI model."""
        if model in _CONTEXT_LIMITS:
            return True
        # Check prefix match
        for prefix in _CONTEXT_LIMITS:
            if model.startswith(prefix):
                return True
        return False

    def get_token_counter(self, model: str) -> TokenCounter:
        """Get token counter for an OpenAI model."""
        if not self.supports_model(model):
            raise ValueError(
                f"Model '{model}' is not recognized as an OpenAI model. "
                f"Supported models: {list(_CONTEXT_LIMITS.keys())}"
            )
        return OpenAITokenCounter(model)

    def get_context_limit(self, model: str) -> int:
        """Get context limit for an OpenAI model."""
        if model in _CONTEXT_LIMITS:
            return _CONTEXT_LIMITS[model]

        # Prefix match
        for prefix, limit in _CONTEXT_LIMITS.items():
            if model.startswith(prefix):
                return limit

        raise ValueError(
            f"Unknown context limit for model '{model}'. "
            f"Known models: {list(_CONTEXT_LIMITS.keys())}"
        )

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost for OpenAI API call.

        ⚠️ IMPORTANT: This is an ESTIMATE only.
        - Pricing data may be outdated
        - Cached token discount assumed at 50% (actual may vary)
        - Always verify against your actual OpenAI billing

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
            cached_tokens: Number of cached tokens (estimated 50% discount).

        Returns:
            Estimated cost in USD, or None if pricing unknown.
        """
        # Check for stale pricing and warn once
        staleness_warning = _check_pricing_staleness()
        if staleness_warning:
            warnings.warn(staleness_warning, UserWarning, stacklevel=2)

        # Find pricing
        input_price, output_price = None, None
        for model_prefix, (inp, outp) in _PRICING.items():
            if model.startswith(model_prefix):
                input_price, output_price = inp, outp
                break

        if input_price is None:
            return None  # Unknown pricing

        # Calculate cost (cached tokens get estimated 50% discount)
        # NOTE: Actual OpenAI cache discount may vary
        regular_input = input_tokens - cached_tokens
        cached_cost = (cached_tokens / 1_000_000) * input_price * 0.5
        regular_cost = (regular_input / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price

        return cached_cost + regular_cost + output_cost

    def get_output_buffer(self, model: str, default: int = 4000) -> int:
        """Get recommended output buffer."""
        # Reasoning models produce longer outputs
        if model.startswith("o1") or model.startswith("o3"):
            return 8000
        return default
