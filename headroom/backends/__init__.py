"""Headroom Backends - API translation layers for different LLM providers.

Backends handle the translation between the proxy's canonical format
(Anthropic Messages API) and provider-specific APIs.

Uses LiteLLM for broad provider support:
- bedrock: AWS Bedrock (Claude, Cohere, Mistral, etc.)
- vertex_ai: Google Vertex AI (Claude, Gemini, etc.)
- azure: Azure OpenAI (GPT-4, etc.)
- And 100+ more providers...

Usage:
    headroom proxy --backend litellm-bedrock --region us-west-2
"""

from .base import Backend, BackendResponse, StreamEvent
from .litellm import LiteLLMBackend

__all__ = ["Backend", "BackendResponse", "StreamEvent", "LiteLLMBackend"]
