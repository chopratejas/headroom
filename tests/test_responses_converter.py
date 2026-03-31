"""Tests for OpenAI Responses API ↔ Chat Completions message conversion.

Tests cover:
1. Forward conversion (Responses items → Chat Completions messages)
2. Reverse conversion (compressed messages → Responses items)
3. Round-trip fidelity (convert → compress → convert back)
4. Edge cases (empty input, unknown types, mixed ordering)
"""

from __future__ import annotations

import json

from headroom.proxy.responses_converter import (
    messages_to_responses_items,
    responses_items_to_messages,
)

# =============================================================================
# Forward conversion: responses_items_to_messages
# =============================================================================


class TestItemsToMessages:
    """Test converting Responses API items to Chat Completions messages."""

    def test_simple_user_message(self):
        """String content user message passes through."""
        items = [{"role": "user", "content": "Hello"}]
        messages, preserved = responses_items_to_messages(items)

        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert preserved == []

    def test_content_array_input_text(self):
        """input_text content parts are extracted to plain text."""
        items = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What is 2+2?"}],
            }
        ]
        messages, preserved = responses_items_to_messages(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is 2+2?"

    def test_output_text_assistant(self):
        """output_text content parts from assistant messages are extracted."""
        items = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The answer is 4."}],
            }
        ]
        messages, preserved = responses_items_to_messages(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "The answer is 4."

    def test_function_call_to_tool_calls(self):
        """Single function_call item becomes assistant message with tool_calls."""
        items = [
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            }
        ]
        messages, preserved = responses_items_to_messages(items)

        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "Paris"}'

    def test_consecutive_function_calls_merge(self):
        """Consecutive function_call items merge into one assistant message."""
        items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "search",
                "arguments": '{"q": "foo"}',
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "read_file",
                "arguments": '{"path": "/tmp/x"}',
            },
        ]
        messages, preserved = responses_items_to_messages(items)

        assert len(messages) == 1
        assert len(messages[0]["tool_calls"]) == 2
        assert messages[0]["tool_calls"][0]["id"] == "call_1"
        assert messages[0]["tool_calls"][1]["id"] == "call_2"

    def test_function_call_output_to_tool_role(self):
        """function_call_output becomes role=tool message."""
        items = [
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": '{"temp": 22, "unit": "C"}',
            }
        ]
        messages, preserved = responses_items_to_messages(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_abc"
        assert messages[0]["content"] == '{"temp": 22, "unit": "C"}'

    def test_reasoning_preserved(self):
        """Reasoning items go to preserved_indices, not messages."""
        items = [
            {"role": "user", "content": "Think hard."},
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Thinking..."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        ]
        messages, preserved = responses_items_to_messages(items)

        assert len(messages) == 2  # user + assistant (reasoning skipped)
        assert 1 in preserved  # index 1 is the reasoning item

    def test_image_content_preserved(self):
        """Items with input_image content are preserved, not converted."""
        items = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image."},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                ],
            }
        ]
        messages, preserved = responses_items_to_messages(items)

        assert len(messages) == 0  # skipped (has non-text)
        assert 0 in preserved

    def test_developer_maps_to_system(self):
        """developer role maps to system."""
        items = [{"role": "developer", "content": "You are helpful."}]
        messages, preserved = responses_items_to_messages(items)

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

    def test_system_passthrough(self):
        """system role passes through as-is."""
        items = [{"role": "system", "content": "Be concise."}]
        messages, preserved = responses_items_to_messages(items)

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."

    def test_message_type_item(self):
        """Item with explicit type='message' is handled."""
        items = [
            {"type": "message", "role": "user", "content": "Hi"},
        ]
        messages, preserved = responses_items_to_messages(items)

        assert messages[0] == {"role": "user", "content": "Hi"}

    def test_empty_input(self):
        """Empty items list produces empty messages."""
        messages, preserved = responses_items_to_messages([])
        assert messages == []
        assert preserved == []

    def test_unknown_type_preserved(self):
        """Unknown item types are preserved, not converted."""
        items = [{"type": "some_future_type", "data": "something"}]
        messages, preserved = responses_items_to_messages(items)

        assert len(messages) == 0
        assert 0 in preserved

    def test_function_call_flush_on_non_function_call(self):
        """Pending function_calls flush when a non-function_call item arrives."""
        items = [
            {"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "result"},
        ]
        messages, preserved = responses_items_to_messages(items)

        # Should be: assistant (tool_calls), tool (output)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["id"] == "c1"
        assert messages[1]["role"] == "tool"


# =============================================================================
# Reverse conversion: messages_to_responses_items
# =============================================================================


class TestMessagesToItems:
    """Test converting compressed messages back to Responses API items."""

    def test_round_trip_simple(self):
        """Simple user/assistant conversation round-trips."""
        original = [
            {"role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Hi!"},
        ]
        messages, preserved = responses_items_to_messages(original)
        result = messages_to_responses_items(messages, original, preserved)

        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi!"

    def test_round_trip_with_tools(self):
        """Full tool call flow round-trips correctly."""
        original = [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Sunny, 22C",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "It's sunny."}],
            },
        ]
        messages, preserved = responses_items_to_messages(original)

        assert len(messages) == 4  # user, assistant(tool_calls), tool, assistant

        result = messages_to_responses_items(messages, original, preserved)

        assert len(result) == 4
        # function_call passes through unmodified
        assert result[1]["type"] == "function_call"
        assert result[1]["call_id"] == "call_1"
        # function_call_output has original content (no compression happened)
        assert result[2]["type"] == "function_call_output"
        assert result[2]["output"] == "Sunny, 22C"

    def test_round_trip_compressed_output(self):
        """Simulated compression: shortened tool output appears in result."""
        original = [
            {"role": "user", "content": "Get data"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "search",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": json.dumps([{"id": i, "name": f"item_{i}"} for i in range(100)]),
            },
        ]
        messages, preserved = responses_items_to_messages(original)

        # Simulate compression: replace tool output with shorter version
        for msg in messages:
            if msg.get("role") == "tool":
                msg["content"] = "[100 items, first: item_0, last: item_99]"

        result = messages_to_responses_items(messages, original, preserved)

        assert result[2]["type"] == "function_call_output"
        assert result[2]["output"] == "[100 items, first: item_0, last: item_99]"

    def test_round_trip_with_reasoning(self):
        """Reasoning items survive round-trip exactly."""
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_abc",
            "summary": [{"type": "summary_text", "text": "Let me think..."}],
        }
        original = [
            {"role": "user", "content": "Complex question"},
            reasoning_item,
            {"type": "message", "role": "assistant", "content": "Answer."},
        ]
        messages, preserved = responses_items_to_messages(original)
        result = messages_to_responses_items(messages, original, preserved)

        assert len(result) == 3
        assert result[1] == reasoning_item  # Exact match
        assert result[1]["type"] == "reasoning"

    def test_round_trip_content_array_preserved(self):
        """Content array structure (input_text) is preserved through round-trip."""
        original = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Original question"}],
            },
        ]
        messages, preserved = responses_items_to_messages(original)

        # Simulate compression changing the text
        messages[0]["content"] = "Compressed question"

        result = messages_to_responses_items(messages, original, preserved)

        # Should reconstruct the array structure
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "Compressed question"

    def test_mixed_ordering(self):
        """Complex sequence maintains correct ordering."""
        original = [
            {"role": "user", "content": "Do two things"},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "task_a",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "c2",
                "name": "task_b",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "Result A",
            },
            {
                "type": "function_call_output",
                "call_id": "c2",
                "output": "Result B",
            },
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Thinking..."}],
            },
            {"type": "message", "role": "assistant", "content": "All done."},
        ]
        messages, preserved = responses_items_to_messages(original)
        result = messages_to_responses_items(messages, original, preserved)

        assert len(result) == 7
        assert result[0]["role"] == "user"
        assert result[1]["type"] == "function_call"
        assert result[1]["call_id"] == "c1"
        assert result[2]["type"] == "function_call"
        assert result[2]["call_id"] == "c2"
        assert result[3]["type"] == "function_call_output"
        assert result[3]["call_id"] == "c1"
        assert result[4]["type"] == "function_call_output"
        assert result[4]["call_id"] == "c2"
        assert result[5]["type"] == "reasoning"
        assert result[6]["content"] == "All done."

    def test_image_preserved_in_round_trip(self):
        """Image items survive round-trip at their original position."""
        image_item = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this"},
                {"type": "input_image", "image_url": "https://example.com/img.png"},
            ],
        }
        original = [
            {"role": "user", "content": "Hi"},
            image_item,
            {"role": "user", "content": "Also tell me about this"},
        ]
        messages, preserved = responses_items_to_messages(original)
        result = messages_to_responses_items(messages, original, preserved)

        assert len(result) == 3
        assert result[0]["content"] == "Hi"
        assert result[1] == image_item  # Preserved exactly
        assert result[2]["content"] == "Also tell me about this"
