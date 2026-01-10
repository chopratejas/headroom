"""Tests for tool crusher transform."""

import json

from headroom import OpenAIProvider, Tokenizer, ToolCrusherConfig
from headroom.transforms import ToolCrusher

# Create a shared provider for tests
_provider = OpenAIProvider()


def get_tokenizer(model: str = "gpt-4o") -> Tokenizer:
    """Get a tokenizer for tests using OpenAI provider."""
    token_counter = _provider.get_token_counter(model)
    return Tokenizer(token_counter, model)


class TestToolCrusher:
    """Tests for ToolCrusher transform."""

    def test_small_tool_output_unchanged(self):
        """Small tool outputs should not be modified."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "ok"}'},
        ]

        crusher = ToolCrusher()
        tokenizer = get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Should not be modified (too small)
        assert result.messages[1]["content"] == '{"status": "ok"}'
        assert len(result.transforms_applied) == 0

    def test_large_json_array_truncated(self):
        """Large arrays should be truncated."""
        large_array = [{"id": i, "name": f"Item {i}"} for i in range(50)]
        large_json = json.dumps({"results": large_array})

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": large_json},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=50, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Should be modified
        tool_content = result.messages[1]["content"]
        parsed = json.loads(tool_content.split("\n<headroom:")[0])

        # Array should be truncated
        assert len(parsed["results"]) <= 6  # 5 items + truncation marker

    def test_long_strings_truncated(self):
        """Long strings should be truncated."""
        long_string = "x" * 2000
        data = {"content": long_string}

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=50, max_string_length=100)
        crusher = ToolCrusher(config)
        tokenizer = get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[1]["content"]
        parsed = json.loads(tool_content.split("\n<headroom:")[0])

        # String should be truncated
        assert len(parsed["content"]) < 200
        assert "truncated" in parsed["content"]

    def test_nested_depth_limited(self):
        """Deeply nested structures should be limited."""
        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(10):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(nested)},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=10, max_depth=3)
        crusher = ToolCrusher(config)
        tokenizer = get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[1]["content"]
        parsed = json.loads(tool_content.split("\n<headroom:")[0])

        # Deep nesting should be summarized
        # Navigate to depth limit
        current = parsed
        depth = 0
        while "nested" in current and isinstance(current["nested"], dict):
            current = current["nested"]
            depth += 1
            if depth > 5:
                break

        assert depth <= 4  # Should be limited

    def test_digest_marker_added(self):
        """Digest marker should be added to crushed content."""
        large_data = {"items": list(range(100))}

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(large_data)},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=10, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[1]["content"]

        # Should have digest marker
        assert "<headroom:tool_digest" in tool_content
        assert "sha256=" in tool_content

    def test_non_tool_messages_unchanged(self):
        """Non-tool messages should not be modified."""
        messages = [
            {"role": "system", "content": json.dumps({"large": "data" * 1000})},
            {"role": "user", "content": json.dumps({"user": "data" * 1000})},
            {"role": "assistant", "content": json.dumps({"assistant": "data" * 1000})},
        ]

        crusher = ToolCrusher()
        tokenizer = get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # All messages should be unchanged
        for i, msg in enumerate(result.messages):
            assert msg["content"] == messages[i]["content"]
