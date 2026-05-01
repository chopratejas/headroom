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

    def test_transform_tag_includes_tool_names_openai(self):
        """Tag shape is ``tool_crush:<count>:<name1,name2>`` for OpenAI format."""
        large_a = {"items": [{"id": i, "v": "x" * 10} for i in range(40)]}
        large_b = {"rows": list(range(200))}

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "Grep", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(large_a)},
            {"role": "tool", "tool_call_id": "c2", "content": json.dumps(large_b)},
        ]

        crusher = ToolCrusher(ToolCrusherConfig(min_tokens_to_crush=10, max_array_items=3))
        result = crusher.apply(messages, get_tokenizer())

        tags = [t for t in result.transforms_applied if t.startswith("tool_crush:")]
        assert len(tags) == 1
        parts = tags[0].split(":", 2)
        assert parts[0] == "tool_crush"
        assert parts[1] == "2"
        # Order follows first-crushed-first
        assert parts[2] == "Bash,Grep"

    def test_transform_tag_includes_tool_names_anthropic(self):
        """Anthropic tool_use blocks feed the tool-name index."""
        large = {"items": [{"id": i, "v": "x" * 10} for i in range(40)]}
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "u1", "name": "Read", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "u1", "content": json.dumps(large)},
                ],
            },
        ]

        crusher = ToolCrusher(ToolCrusherConfig(min_tokens_to_crush=10, max_array_items=3))
        result = crusher.apply(messages, get_tokenizer())

        assert "tool_crush:1:Read" in result.transforms_applied

    def test_transform_tag_dedupes_repeated_tool(self):
        """Same tool crushed twice shows once in the tag."""
        large = {"items": [{"id": i, "v": "x" * 10} for i in range(40)]}

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(large)},
            {"role": "tool", "tool_call_id": "c2", "content": json.dumps(large)},
        ]

        crusher = ToolCrusher(ToolCrusherConfig(min_tokens_to_crush=10, max_array_items=3))
        result = crusher.apply(messages, get_tokenizer())

        assert "tool_crush:2:Bash" in result.transforms_applied

    def test_tool_name_index_skips_entries_missing_id_or_name(self):
        """Guards: tool_calls / tool_use blocks missing id or name are skipped,
        other blocks (text, etc.) are skipped, and the crushed tag still
        reflects the entries that DO have both."""
        large = {"items": [{"id": i, "v": "x" * 10} for i in range(40)]}
        messages = [
            {
                "role": "assistant",
                "content": [
                    # tool_use block with no id → skipped
                    {"type": "tool_use", "name": "NamelessRead"},
                    # tool_use block with no name → skipped
                    {"type": "tool_use", "id": "u0"},
                    # Non-tool_use block → skipped
                    {"type": "text", "text": "thinking..."},
                    # The one good entry
                    {"type": "tool_use", "id": "u1", "name": "Grep", "input": {}},
                ],
                # OpenAI-style tool_calls missing id/name → skipped
                "tool_calls": [
                    {"id": "", "function": {"name": "Empty"}},
                    {"id": "c1", "function": {"name": ""}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "u1", "content": json.dumps(large)},
                ],
            },
        ]

        crusher = ToolCrusher(ToolCrusherConfig(min_tokens_to_crush=10, max_array_items=3))
        result = crusher.apply(messages, get_tokenizer())

        # Only Grep (u1) had both id + name AND was actually crushed.
        assert "tool_crush:1:Grep" in result.transforms_applied

    def test_transform_tag_falls_back_when_no_names(self):
        """Crushed tool with no resolvable name keeps legacy ``tool_crush:<n>`` shape."""
        large = {"items": [{"id": i, "v": "x" * 10} for i in range(40)]}
        # No assistant message → no name index entries.
        messages = [
            {"role": "tool", "tool_call_id": "orphan", "content": json.dumps(large)},
        ]

        crusher = ToolCrusher(ToolCrusherConfig(min_tokens_to_crush=10, max_array_items=3))
        result = crusher.apply(messages, get_tokenizer())

        assert "tool_crush:1" in result.transforms_applied

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
