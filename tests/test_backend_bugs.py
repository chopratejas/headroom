"""Tests for backend bug fixes in LiteLLM and any-llm integrations.

Tests tool forwarding, tool argument parsing, streaming param forwarding,
and Vertex AI model mapping.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("litellm")

from headroom.backends.litellm import (
    _VERTEX_MODEL_MAP,
    LiteLLMBackend,
    _convert_anthropic_tool,
    _convert_tool_choice,
    _parse_tool_arguments,
)

# =============================================================================
# Tool Format Conversion (Bug 1)
# =============================================================================


class TestConvertAnthropicTool:
    """Test Anthropic → OpenAI tool format conversion."""

    def test_basic_tool_conversion(self):
        anthropic_tool = {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
        result = _convert_anthropic_tool(anthropic_tool)
        assert result == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }

    def test_tool_without_description(self):
        tool = {"name": "do_thing", "input_schema": {"type": "object"}}
        result = _convert_anthropic_tool(tool)
        assert result["function"]["name"] == "do_thing"
        assert "description" not in result["function"]
        assert result["function"]["parameters"] == {"type": "object"}

    def test_tool_without_input_schema(self):
        tool = {"name": "simple_tool", "description": "No params"}
        result = _convert_anthropic_tool(tool)
        assert result["function"]["name"] == "simple_tool"
        assert "parameters" not in result["function"]


class TestConvertToolChoice:
    """Test Anthropic → OpenAI tool_choice conversion."""

    def test_auto(self):
        assert _convert_tool_choice({"type": "auto"}) == "auto"

    def test_any_to_required(self):
        assert _convert_tool_choice({"type": "any"}) == "required"

    def test_specific_tool(self):
        result = _convert_tool_choice({"type": "tool", "name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_string_passthrough(self):
        assert _convert_tool_choice("auto") == "auto"
        assert _convert_tool_choice("none") == "none"


# =============================================================================
# Tool Argument Parsing (Bug 2)
# =============================================================================


class TestParseToolArguments:
    """Test that tool arguments are parsed from JSON string to dict."""

    def test_json_string_parsed(self):
        result = _parse_tool_arguments('{"location": "Paris"}')
        assert result == {"location": "Paris"}

    def test_dict_passthrough(self):
        d = {"location": "Paris"}
        result = _parse_tool_arguments(d)
        assert result == d

    def test_invalid_json_returns_original(self):
        result = _parse_tool_arguments("not json")
        assert result == "not json"

    def test_empty_string(self):
        result = _parse_tool_arguments("")
        assert result == ""

    def test_none_passthrough(self):
        result = _parse_tool_arguments(None)
        assert result is None


# =============================================================================
# LiteLLM send_message Tools Forwarding (Bug 1)
# =============================================================================


class TestLiteLLMToolsForwarding:
    """Test that tools are forwarded through LiteLLM send_message."""

    @pytest.mark.asyncio
    async def test_tools_forwarded_in_send_message(self):
        """Tools should be converted and passed to litellm.acompletion."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hello", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_response

            backend = LiteLLMBackend(provider="openrouter")
            body = {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                "tool_choice": {"type": "auto"},
            }

            await backend.send_message(body, {})

            call_kwargs = mock_acomp.call_args[1]
            assert "tools" in call_kwargs
            assert call_kwargs["tools"][0]["type"] == "function"
            assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"
            assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_tool_arguments_parsed_in_response(self):
        """Tool call arguments should be parsed from JSON string to dict."""
        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "Paris"}'

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content=None, tool_calls=[mock_tc]),
                finish_reason="tool_calls",
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_response

            backend = LiteLLMBackend(provider="openrouter")
            result = await backend.send_message(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                {},
            )

            tool_block = result.body["content"][0]
            assert tool_block["type"] == "tool_use"
            assert tool_block["input"] == {"location": "Paris"}
            assert isinstance(tool_block["input"], dict)


# =============================================================================
# Streaming Params (Bugs 3-4)
# =============================================================================


class TestLiteLLMStreamingParams:
    """Test that streaming forwards all params."""

    @pytest.mark.asyncio
    async def test_streaming_forwards_all_params(self):
        """stream_message should forward top_p, stop, and tools."""

        # Create an async iterator for the mock streaming response
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content="Hi"))]
            yield chunk

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_stream()

            backend = LiteLLMBackend(provider="openrouter")
            body = {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop_sequences": ["\n"],
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "A test",
                        "input_schema": {"type": "object"},
                    }
                ],
            }

            events = []
            async for event in backend.stream_message(body, {}):
                events.append(event)

            call_kwargs = mock_acomp.call_args[1]
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["stop"] == ["\n"]
            assert "tools" in call_kwargs
            assert call_kwargs["tools"][0]["function"]["name"] == "test_tool"


# =============================================================================
# Vertex AI Model Map (Bug 6)
# =============================================================================


class TestVertexModelMap:
    """Test that Vertex AI model map includes all current models.

    Model IDs sourced from: https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
    """

    def test_claude_46_models(self):
        assert _VERTEX_MODEL_MAP["claude-opus-4-6"] == "vertex_ai/claude-opus-4-6"
        assert _VERTEX_MODEL_MAP["claude-sonnet-4-6"] == "vertex_ai/claude-sonnet-4-6"

    def test_claude_45_models(self):
        assert (
            _VERTEX_MODEL_MAP["claude-sonnet-4-5-20250929"]
            == "vertex_ai/claude-sonnet-4-5@20250929"
        )
        assert _VERTEX_MODEL_MAP["claude-opus-4-5-20251101"] == "vertex_ai/claude-opus-4-5@20251101"

    def test_claude_4_models(self):
        assert _VERTEX_MODEL_MAP["claude-sonnet-4-20250514"] == "vertex_ai/claude-sonnet-4@20250514"
        assert _VERTEX_MODEL_MAP["claude-opus-4-20250514"] == "vertex_ai/claude-opus-4@20250514"

    def test_claude_35_models(self):
        assert (
            _VERTEX_MODEL_MAP["claude-3-5-sonnet-20241022"]
            == "vertex_ai/claude-3-5-sonnet-v2@20241022"
        )
        assert (
            _VERTEX_MODEL_MAP["claude-3-5-haiku-20241022"] == "vertex_ai/claude-3-5-haiku@20241022"
        )

    def test_claude_haiku_45(self):
        assert (
            _VERTEX_MODEL_MAP["claude-haiku-4-5-20251001"] == "vertex_ai/claude-haiku-4-5@20251001"
        )

    def test_claude_3_legacy(self):
        assert "claude-3-haiku-20240307" in _VERTEX_MODEL_MAP
