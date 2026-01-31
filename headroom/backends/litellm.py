"""LiteLLM-based backend for Headroom.

Uses LiteLLM to support 100+ providers with minimal code:
- AWS Bedrock: model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
- Azure OpenAI: model="azure/gpt-4"
- Google Vertex: model="vertex_ai/claude-3-5-sonnet"
- And many more...

LiteLLM handles all the auth and format translation internally.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .base import Backend, BackendResponse, StreamEvent

logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm import acompletion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore
    acompletion = None  # type: ignore


# Model mapping: Anthropic model IDs -> LiteLLM model strings
# IMPORTANT: Claude 4+ models require inference profiles (us.anthropic.* or global.anthropic.*)
# Direct model IDs (anthropic.*) don't support on-demand throughput for newer models.
# See: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html
BEDROCK_MODEL_MAP = {
    # Claude 4.5 (requires inference profiles)
    "claude-opus-4-5-20251101": "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-sonnet-4-5-20250929": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-haiku-4-5-20251001": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    # Claude 4.1
    "claude-opus-4-1-20250805": "bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0",
    # Claude 4 (requires inference profiles)
    "claude-opus-4-20250514": "bedrock/us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-sonnet-4-20250514": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
    # Claude 3.7 (requires inference profiles)
    "claude-3-7-sonnet-20250219": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    # Claude 3.5 (can use inference profiles for cross-region)
    "claude-3-5-sonnet-20241022": "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet-20240620": "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-haiku-20241022": "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
    # Claude 3 (can use inference profiles for cross-region)
    "claude-3-opus-20240229": "bedrock/us.anthropic.claude-3-opus-20240229-v1:0",
    "claude-3-sonnet-20240229": "bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku-20240307": "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0",
}

VERTEX_MODEL_MAP = {
    "claude-3-5-sonnet-20241022": "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "claude-3-5-sonnet-20240620": "vertex_ai/claude-3-5-sonnet@20240620",
    "claude-3-opus-20240229": "vertex_ai/claude-3-opus@20240229",
    "claude-3-sonnet-20240229": "vertex_ai/claude-3-sonnet@20240229",
    "claude-3-haiku-20240307": "vertex_ai/claude-3-haiku@20240307",
}


class LiteLLMBackend(Backend):
    """Backend using LiteLLM for multi-provider support.

    Supports any provider LiteLLM supports:
    - bedrock: AWS Bedrock (uses AWS credentials)
    - vertex_ai: Google Vertex AI (uses GCP credentials)
    - azure: Azure OpenAI (uses Azure credentials)
    - And 100+ more...
    """

    def __init__(
        self,
        provider: str = "bedrock",
        region: str | None = None,
        **kwargs: Any,
    ):
        """Initialize LiteLLM backend.

        Args:
            provider: LiteLLM provider prefix (bedrock, vertex_ai, azure, etc.)
            region: Cloud region (provider-specific)
            **kwargs: Additional provider-specific config
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for LiteLLMBackend. Install with: pip install litellm"
            )

        self.provider = provider
        self.region = region
        self.kwargs = kwargs

        # Select model map based on provider
        if provider == "bedrock":
            self._model_map = BEDROCK_MODEL_MAP
            # Set AWS region for litellm
            if region:
                litellm.set_verbose = False  # Reduce noise
        elif provider == "vertex_ai":
            self._model_map = VERTEX_MODEL_MAP
        else:
            self._model_map = {}

        logger.info(f"LiteLLM backend initialized (provider={provider})")

    @property
    def name(self) -> str:
        return f"litellm-{self.provider}"

    def map_model_id(self, anthropic_model: str) -> str:
        """Map Anthropic model ID to LiteLLM model string."""
        # Check direct mapping
        if anthropic_model in self._model_map:
            return self._model_map[anthropic_model]

        # If already has provider prefix, use as-is
        if "/" in anthropic_model:
            return anthropic_model

        # Fallback: construct provider/model format
        return f"{self.provider}/{anthropic_model}"

    def supports_model(self, model: str) -> bool:
        """Check if model is supported."""
        return "claude" in model.lower() or model in self._model_map

    def _convert_messages_for_litellm(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic message format to LiteLLM/OpenAI format.

        LiteLLM expects OpenAI-style messages but handles most Anthropic
        content blocks automatically.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle string content directly
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            # Handle content blocks (Anthropic style)
            if isinstance(content, list):
                # Check if it's simple text blocks only
                text_parts = []
                has_complex_content = False

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") in ("tool_use", "tool_result", "image"):
                            has_complex_content = True
                            break

                if not has_complex_content and text_parts:
                    # Simple text - join into single string
                    converted.append({"role": role, "content": "\n".join(text_parts)})
                else:
                    # Complex content - pass through (LiteLLM handles it)
                    converted.append({"role": role, "content": content})

        return converted

    def _to_anthropic_response(
        self,
        litellm_response: Any,
        original_model: str,
    ) -> dict[str, Any]:
        """Convert LiteLLM/OpenAI response to Anthropic format."""
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Extract content from OpenAI format
        choice = litellm_response.choices[0]
        message = choice.message

        # Build Anthropic content blocks
        content = []
        if message.content:
            content.append({"type": "text", "text": message.content})

        # Handle tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": tc.function.arguments,
                    }
                )

        # Map stop reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn",
        }
        stop_reason = stop_reason_map.get(choice.finish_reason, "end_turn")

        # Build usage
        usage = {
            "input_tokens": getattr(litellm_response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(litellm_response.usage, "completion_tokens", 0),
        }

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }

    async def send_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> BackendResponse:
        """Send message via LiteLLM."""
        original_model = body.get("model", "claude-3-5-sonnet-20241022")
        litellm_model = self.map_model_id(original_model)

        try:
            # Convert messages
            messages = self._convert_messages_for_litellm(body.get("messages", []))

            # Build kwargs for litellm
            kwargs: dict[str, Any] = {
                "model": litellm_model,
                "messages": messages,
            }

            # Optional parameters
            if "max_tokens" in body:
                kwargs["max_tokens"] = body["max_tokens"]
            if "temperature" in body:
                kwargs["temperature"] = body["temperature"]
            if "top_p" in body:
                kwargs["top_p"] = body["top_p"]
            if "stop_sequences" in body:
                kwargs["stop"] = body["stop_sequences"]

            # System prompt (Anthropic puts it in body, OpenAI in messages)
            if "system" in body:
                system = body["system"]
                if isinstance(system, str):
                    kwargs["messages"].insert(0, {"role": "system", "content": system})
                elif isinstance(system, list):
                    # Anthropic list format
                    system_text = " ".join(
                        s.get("text", "") if isinstance(s, dict) else str(s) for s in system
                    )
                    kwargs["messages"].insert(0, {"role": "system", "content": system_text})

            # AWS region for Bedrock
            if self.provider == "bedrock" and self.region:
                kwargs["aws_region_name"] = self.region

            logger.debug(f"LiteLLM request: model={litellm_model}")

            # Make the call
            response = await acompletion(**kwargs)

            # Convert to Anthropic format
            anthropic_response = self._to_anthropic_response(response, original_model)

            return BackendResponse(
                body=anthropic_response,
                status_code=200,
                headers={"content-type": "application/json"},
            )

        except Exception as e:
            logger.error(f"LiteLLM error: {e}")

            # Map to Anthropic error format
            error_type = "api_error"
            status_code = 500

            error_str = str(e).lower()
            if "authentication" in error_str or "credentials" in error_str:
                error_type = "authentication_error"
                status_code = 401
            elif "rate" in error_str or "limit" in error_str:
                error_type = "rate_limit_error"
                status_code = 429
            elif "not found" in error_str:
                error_type = "not_found_error"
                status_code = 404

            return BackendResponse(
                body={
                    "type": "error",
                    "error": {"type": error_type, "message": str(e)},
                },
                status_code=status_code,
                error=str(e),
            )

    async def stream_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[StreamEvent]:
        """Stream message via LiteLLM."""
        original_model = body.get("model", "claude-3-5-sonnet-20241022")
        litellm_model = self.map_model_id(original_model)

        try:
            messages = self._convert_messages_for_litellm(body.get("messages", []))

            kwargs: dict[str, Any] = {
                "model": litellm_model,
                "messages": messages,
                "stream": True,
            }

            if "max_tokens" in body:
                kwargs["max_tokens"] = body["max_tokens"]
            if "temperature" in body:
                kwargs["temperature"] = body["temperature"]
            if "system" in body:
                system = body["system"]
                if isinstance(system, str):
                    kwargs["messages"].insert(0, {"role": "system", "content": system})

            if self.provider == "bedrock" and self.region:
                kwargs["aws_region_name"] = self.region

            msg_id = f"msg_{uuid.uuid4().hex[:24]}"

            # Emit message_start
            yield StreamEvent(
                event_type="message_start",
                data={
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": original_model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )

            # Emit content_block_start
            yield StreamEvent(
                event_type="content_block_start",
                data={
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )

            # Stream content
            response = await acompletion(**kwargs)
            output_tokens = 0

            async for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield StreamEvent(
                            event_type="content_block_delta",
                            data={
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": delta.content},
                            },
                        )
                        output_tokens += 1  # Rough estimate

            # Emit content_block_stop
            yield StreamEvent(
                event_type="content_block_stop",
                data={"type": "content_block_stop", "index": 0},
            )

            # Emit message_delta with stop reason
            yield StreamEvent(
                event_type="message_delta",
                data={
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            )

            # Emit message_stop
            yield StreamEvent(
                event_type="message_stop",
                data={"type": "message_stop"},
            )

        except Exception as e:
            logger.error(f"LiteLLM streaming error: {e}")
            yield StreamEvent(
                event_type="error",
                data={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                },
            )

    async def close(self) -> None:  # noqa: B027
        """Clean up (no-op for LiteLLM)."""
        pass
