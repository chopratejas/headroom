"""Anthropic provider implementation for Headroom SDK.

Token counting uses Anthropic's official Token Count API when a client
is provided. This gives accurate counts for all content types including
JSON, non-English text, and tool definitions.

Usage:
    from anthropic import Anthropic
    from headroom import AnthropicProvider

    client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
    provider = AnthropicProvider(client=client)  # Accurate counting via API

    # Or without client (uses tiktoken approximation - less accurate)
    provider = AnthropicProvider()  # Warning: approximate counting
"""

import warnings
from typing import Any

from .base import Provider, TokenCounter

# Warning flags
_FALLBACK_WARNING_SHOWN = False


# Anthropic model context limits
ANTHROPIC_CONTEXT_LIMITS: dict[str, int] = {
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-latest": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-5-haiku-latest": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-opus-latest": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-2.1": 200000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,
}

# Anthropic pricing per 1M tokens
# NOTE: These are ESTIMATES. Always verify against actual Anthropic billing.
# Last updated: 2024-12-01
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-3-opus-latest": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25, "cached_input": 0.03},
}


class AnthropicTokenCounter(TokenCounter):
    """Token counter for Anthropic models.

    When an Anthropic client is provided, uses the official Token Count API
    (/v1/messages/count_tokens) for accurate counting. This handles:
    - JSON-heavy tool payloads
    - Non-English text
    - Tool definitions and structured content

    Falls back to tiktoken approximation only when no client is available.
    """

    def __init__(self, model: str, client: Any = None):
        """Initialize token counter.

        Args:
            model: Anthropic model name.
            client: Optional anthropic.Anthropic client for API-based counting.
                    If not provided, falls back to tiktoken approximation.
        """
        global _FALLBACK_WARNING_SHOWN

        self.model = model
        self._client = client
        self._encoding: Any = None
        self._use_api = client is not None

        if not self._use_api and not _FALLBACK_WARNING_SHOWN:
            warnings.warn(
                "AnthropicProvider: No client provided, using tiktoken approximation. "
                "For accurate counting, pass an Anthropic client: "
                "AnthropicProvider(client=Anthropic())",
                UserWarning,
                stacklevel=4,
            )
            _FALLBACK_WARNING_SHOWN = True

        # Load tiktoken as fallback
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            if not self._use_api:
                warnings.warn(
                    "tiktoken not installed - token counting will be very approximate. "
                    "Install tiktoken or provide an Anthropic client.",
                    UserWarning,
                    stacklevel=4,
                )

    def count_text(self, text: str) -> int:
        """Count tokens in text.

        Note: For single text strings, uses tiktoken approximation even when
        API is available (API only supports full message counting).
        """
        if not text:
            return 0

        if self._encoding:
            # tiktoken with ~1.1x multiplier for Claude
            base_count = len(self._encoding.encode(text))
            return int(base_count * 1.1)

        # Character-based fallback
        return max(1, len(text) // 3)

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a single message.

        Uses API if available, otherwise falls back to estimation.
        """
        if self._use_api:
            return self._count_message_via_api(message)
        return self._count_message_estimated(message)

    def _count_message_via_api(self, message: dict[str, Any]) -> int:
        """Count tokens using Anthropic Token Count API."""
        try:
            # Convert to Anthropic message format if needed
            messages = [self._normalize_message(message)]
            response = self._client.messages.count_tokens(
                model=self.model,
                messages=messages,
            )
            return response.input_tokens
        except Exception:
            # Fall back to estimation on API error
            return self._count_message_estimated(message)

    def _count_message_estimated(self, message: dict[str, Any]) -> int:
        """Estimate token count without API."""
        tokens = 4  # Role overhead

        content = message.get("content")
        if isinstance(content, str):
            tokens += self.count_text(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        tokens += self.count_text(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tokens += self.count_text(block.get("name", ""))
                        tokens += self.count_text(str(block.get("input", {})))
                    elif block.get("type") == "tool_result":
                        tokens += self.count_text(str(block.get("content", "")))

        # OpenAI format tool calls
        if "tool_calls" in message:
            for tool_call in message.get("tool_calls", []):
                if isinstance(tool_call, dict):
                    func = tool_call.get("function", {})
                    tokens += self.count_text(func.get("name", ""))
                    tokens += self.count_text(func.get("arguments", ""))

        return tokens

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Normalize message to Anthropic format."""
        role = message.get("role", "user")

        # Map OpenAI roles to Anthropic
        if role == "system":
            # System messages need special handling - count as user for API
            return {"role": "user", "content": message.get("content", "")}
        elif role == "tool":
            # Tool results in OpenAI format
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": message.get("content", ""),
                    }
                ],
            }

        return {"role": role, "content": message.get("content", "")}

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages.

        Uses the Token Count API for accurate counting when available.
        """
        if self._use_api:
            return self._count_messages_via_api(messages)
        return self._count_messages_estimated(messages)

    def _count_messages_via_api(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens using Anthropic Token Count API."""
        try:
            # Separate system message (Anthropic handles it differently)
            system_content = None
            api_messages = []

            for msg in messages:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                else:
                    api_messages.append(self._normalize_message(msg))

            # Ensure we have at least one message
            if not api_messages:
                api_messages = [{"role": "user", "content": ""}]

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": api_messages,
            }
            if system_content:
                kwargs["system"] = system_content

            response = self._client.messages.count_tokens(**kwargs)
            return response.input_tokens

        except Exception as e:
            # Fall back to estimation on API error
            warnings.warn(
                f"Token Count API failed ({e}), using estimation", UserWarning, stacklevel=3
            )
            return self._count_messages_estimated(messages)

    def _count_messages_estimated(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token count without API."""
        total = sum(self._count_message_estimated(msg) for msg in messages)
        return total + 3  # Base overhead


class AnthropicProvider(Provider):
    """Provider implementation for Anthropic Claude models.

    For accurate token counting, provide an Anthropic client:

        from anthropic import Anthropic
        provider = AnthropicProvider(client=Anthropic())

    This uses Anthropic's official Token Count API which accurately handles:
    - JSON-heavy tool payloads
    - Non-English text
    - Long system prompts
    - Tool definitions and structured content

    Without a client, falls back to tiktoken approximation (less accurate).
    """

    def __init__(
        self,
        client: Any = None,
        context_limits: dict[str, int] | None = None,
    ):
        """Initialize Anthropic provider.

        Args:
            client: Optional anthropic.Anthropic client for accurate token counting.
                    If not provided, uses tiktoken approximation.
            context_limits: Optional override for model context limits.

        Example:
            from anthropic import Anthropic
            provider = AnthropicProvider(client=Anthropic())
        """
        self._client = client
        self._context_limits = {**ANTHROPIC_CONTEXT_LIMITS}
        if context_limits:
            self._context_limits.update(context_limits)
        self._token_counters: dict[str, AnthropicTokenCounter] = {}

    @property
    def name(self) -> str:
        return "anthropic"

    def get_token_counter(self, model: str) -> TokenCounter:
        """Get token counter for a model.

        If a client was provided to the provider, uses the Token Count API.
        Otherwise falls back to tiktoken approximation.
        """
        if model not in self._token_counters:
            self._token_counters[model] = AnthropicTokenCounter(
                model=model,
                client=self._client,
            )
        return self._token_counters[model]

    def get_context_limit(self, model: str) -> int:
        """Get context window limit for a model."""
        if model in self._context_limits:
            return self._context_limits[model]

        # Check for partial matches (e.g., "claude-3-5-sonnet" matches "claude-3-5-sonnet-20241022")
        for known_model, limit in self._context_limits.items():
            if model in known_model or known_model in model:
                return limit

        raise ValueError(f"Unknown Anthropic model: {model}. Configure context_limits explicitly.")

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model."""
        if model in self._context_limits:
            return True
        # Check prefix matches
        return any(
            model.startswith(prefix) for prefix in ["claude-3", "claude-2", "claude-instant"]
        )

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost for a request."""
        pricing = None

        # Find pricing for model
        if model in ANTHROPIC_PRICING:
            pricing = ANTHROPIC_PRICING[model]
        else:
            # Try partial match
            for known_model, prices in ANTHROPIC_PRICING.items():
                if model in known_model or known_model in model:
                    pricing = prices
                    break

        if not pricing:
            return None

        # Calculate cost
        non_cached_input = input_tokens - cached_tokens
        cost = (
            (non_cached_input / 1_000_000) * pricing["input"]
            + (cached_tokens / 1_000_000) * pricing.get("cached_input", pricing["input"])
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        return cost
