"""Provider-owned CCR format adapters."""

from .base import CCR_TOOL_NAME, ProviderCcrAdapter
from .registry import get_ccr_adapter

__all__ = ["CCR_TOOL_NAME", "ProviderCcrAdapter", "get_ccr_adapter"]
