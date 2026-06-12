"""CCR provider adapter registry."""

from __future__ import annotations

from .anthropic import AnthropicCcrAdapter
from .base import GenericCcrAdapter, ProviderCcrAdapter
from .google import GoogleCcrAdapter
from .openai import OpenAICcrAdapter

_ADAPTERS: dict[str, ProviderCcrAdapter] = {
    "anthropic": AnthropicCcrAdapter(),
    "openai": OpenAICcrAdapter(),
    "google": GoogleCcrAdapter(),
    "gemini": GoogleCcrAdapter(),
}
_GENERIC = GenericCcrAdapter()


def get_ccr_adapter(provider: str) -> ProviderCcrAdapter:
    """Resolve a CCR adapter for a provider name."""
    return _ADAPTERS.get(provider, _GENERIC)
