"""Tests for tool crusher transform."""

import json

from headroom import OpenAIProvider, Tokenizer, ToolCrusherConfig
from headroom.transforms import ToolCrusher
from headroom.transforms.tool_crusher import crush_tool_output

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

    def test_should_apply_false_when_disabled_or_threshold_not_met(self):
        tokenizer = get_tokenizer()
        disabled = ToolCrusher(ToolCrusherConfig(enabled=False, min_tokens_to_crush=1))

        assert (
            disabled.should_apply(
                [{"role": "tool", "content": json.dumps({"items": list(range(100))})}],
                tokenizer,
            )
            is False
        )

        enabled = ToolCrusher(ToolCrusherConfig(enabled=True, min_tokens_to_crush=10_000))
        assert (
            enabled.should_apply(
                [{"role": "tool", "content": json.dumps({"items": list(range(100))})}],
                tokenizer,
            )
            is False
        )

    def test_should_apply_detects_anthropic_tool_result_blocks(self):
        class _Tokenizer:
            def count_text(self, text: str) -> int:
                return 42 if "tool output payload" in text else 0

        large_tool_output = json.dumps({"items": ["tool output payload"]})
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": large_tool_output,
                    }
                ],
            }
        ]

        crusher = ToolCrusher(ToolCrusherConfig(enabled=True, min_tokens_to_crush=5))

        assert crusher.should_apply(messages, _Tokenizer()) is True

    def test_apply_crushes_anthropic_tool_result_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": json.dumps({"items": list(range(50))}),
                    },
                ],
            }
        ]

        crusher = ToolCrusher(ToolCrusherConfig(min_tokens_to_crush=10, max_array_items=3))
        result = crusher.apply(messages, get_tokenizer())

        tool_block = result.messages[0]["content"][1]
        assert "<headroom:tool_digest" in tool_block["content"]
        parsed = json.loads(tool_block["content"].split("\n<headroom:")[0])
        assert parsed["items"][-1] == {"__headroom_truncated": 47}
        assert result.transforms_applied == ["tool_crush:1"]
        assert len(result.markers_inserted) == 1

    def test_apply_skips_non_string_tool_content_variants(self):
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": {"structured": True}},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": {"structured": True},
                    }
                ],
            },
        ]

        crusher = ToolCrusher(ToolCrusherConfig(min_tokens_to_crush=1))
        result = crusher.apply(messages, get_tokenizer())

        assert result.messages == messages
        assert result.transforms_applied == []
        assert result.markers_inserted == []

    def test_crush_content_returns_original_when_json_parse_fails(self):
        crusher = ToolCrusher()

        crushed, was_modified = crusher._crush_content("not-json", crusher._get_profile("", {}))

        assert crushed == "not-json"
        assert was_modified is False

    def test_get_profile_uses_default_config_values(self):
        config = ToolCrusherConfig(
            max_array_items=7,
            max_string_length=123,
            max_depth=4,
            preserve_keys=["id", "name"],
        )
        crusher = ToolCrusher(config)

        profile = crusher._get_profile("call_1", {"ignored": {"max_array_items": 1}})

        assert profile == {
            "max_array_items": 7,
            "max_string_length": 123,
            "max_depth": 4,
            "preserve_keys": ["id", "name"],
        }

    def test_crush_value_handles_depth_exceeded_for_dict_list_and_string(self):
        crusher = ToolCrusher()

        assert crusher._crush_value(
            {"a": 1, "b": 2}, depth=5, max_depth=5, max_array_items=2, max_string_length=4
        ) == {"__headroom_depth_exceeded": 2}
        assert crusher._crush_value(
            [1, 2, 3], depth=5, max_depth=5, max_array_items=2, max_string_length=4
        ) == {"__headroom_depth_exceeded": 3}
        assert crusher._crush_value(
            "abcdef", depth=5, max_depth=5, max_array_items=2, max_string_length=4
        ) == ("abcd...[truncated 2 chars]")

    def test_crush_value_passes_through_small_structures_and_scalars(self):
        crusher = ToolCrusher()

        assert crusher._crush_value(
            {"items": [1, 2], "name": "ok"},
            depth=0,
            max_depth=5,
            max_array_items=5,
            max_string_length=10,
        ) == {"items": [1, 2], "name": "ok"}
        assert (
            crusher._crush_value(
                True, depth=0, max_depth=5, max_array_items=5, max_string_length=10
            )
            is True
        )
        assert (
            crusher._crush_value(
                None, depth=0, max_depth=5, max_array_items=5, max_string_length=10
            )
            is None
        )

    def test_crush_tool_output_convenience_uses_config(self):
        content = json.dumps({"items": list(range(20))})

        crushed, was_modified = crush_tool_output(
            content,
            ToolCrusherConfig(min_tokens_to_crush=1, max_array_items=2),
        )

        assert was_modified is True
        parsed = json.loads(crushed)
        assert parsed["items"][-1] == {"__headroom_truncated": 18}
