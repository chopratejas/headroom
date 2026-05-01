"""Direct unit tests for rolling_window without provider-gated setup."""

from __future__ import annotations

from headroom.config import RollingWindowConfig
from headroom.transforms.rolling_window import RollingWindow, apply_rolling_window


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(text.split()))

    def count_message(self, message) -> int:  # noqa: ANN001, ANN201
        content = message.get("content")
        total = 1
        if isinstance(content, str):
            total += self.count_text(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += self.count_text(block.get("text", ""))
                    total += self.count_text(block.get("content", ""))
        if message.get("tool_calls"):
            total += sum(
                self.count_text(tc.get("function", {}).get("arguments", ""))
                for tc in message["tool_calls"]
            )
        return total

    def count_messages(self, messages) -> int:  # noqa: ANN001, ANN201
        return sum(self.count_message(message) for message in messages)


def _openai_tool_messages() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "System instructions stay stable"},
        {"role": "user", "content": "Old question with many words to consume budget quickly"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query":"old"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "old tool response with several extra words included here",
        },
        {"role": "assistant", "content": "Old answer after the tool call"},
        {"role": "user", "content": "Newest request must survive"},
        {"role": "assistant", "content": "Newest answer must survive too"},
    ]


def test_should_apply_respects_enabled_and_budget() -> None:
    messages = _openai_tool_messages()
    tokenizer = _Tokenizer()

    assert (
        RollingWindow(RollingWindowConfig(enabled=False)).should_apply(
            messages, tokenizer, model_limit=10, output_buffer=0
        )
        is False
    )
    assert (
        RollingWindow(RollingWindowConfig(enabled=True)).should_apply(
            messages, tokenizer, model_limit=10_000, output_buffer=0
        )
        is False
    )
    assert (
        RollingWindow(RollingWindowConfig(enabled=True)).should_apply(
            messages, tokenizer, model_limit=12, output_buffer=0
        )
        is True
    )


def test_apply_returns_unchanged_when_under_budget() -> None:
    messages = _openai_tool_messages()
    window = RollingWindow(RollingWindowConfig(enabled=True, keep_system=True, keep_last_turns=1))
    tokenizer = _Tokenizer()

    result = window.apply(messages, tokenizer, model_limit=10_000, output_buffer=0)

    assert result.messages == messages
    assert result.transforms_applied == []
    assert result.tokens_before == result.tokens_after


def test_apply_drops_oldest_tool_unit_and_inserts_marker() -> None:
    messages = _openai_tool_messages()
    window = RollingWindow(RollingWindowConfig(enabled=True, keep_system=True, keep_last_turns=1))
    tokenizer = _Tokenizer()

    result = window.apply(messages, tokenizer, model_limit=18, output_buffer=0)

    remaining_tool_ids = {
        message.get("tool_call_id") for message in result.messages if message.get("role") == "tool"
    }
    assert "call-1" not in remaining_tool_ids
    assert result.messages[0]["role"] == "system"
    assert "<headroom:dropped_context" in result.messages[1]["content"]
    assert result.transforms_applied == ["window_cap:3"]
    assert len(result.markers_inserted) == 1
    assert result.tokens_after < result.tokens_before


def test_apply_uses_block_marker_when_conversation_uses_block_content() -> None:
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": [{"type": "text", "text": "old block text with many words"}]},
        {"role": "assistant", "content": "older answer with many words"},
        {"role": "user", "content": [{"type": "text", "text": "latest block request"}]},
    ]
    window = RollingWindow(RollingWindowConfig(enabled=True, keep_system=True, keep_last_turns=1))

    result = window.apply(messages, _Tokenizer(), model_limit=6, output_buffer=0)

    assert "<headroom:dropped_context" in result.messages[1]["content"][0]["text"]


def test_get_protected_indices_handles_anthropic_tool_results() -> None:
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Older"},
        {"role": "assistant", "content": "Older answer"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup"}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "result"}],
        },
    ]
    window = RollingWindow(RollingWindowConfig(enabled=True, keep_system=True, keep_last_turns=2))

    protected = window._get_protected_indices(messages)

    assert {0, 3, 4}.issubset(protected)


def test_apply_respects_frozen_message_count() -> None:
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Frozen question with extra words"},
        {"role": "assistant", "content": "Frozen answer with extra words"},
        {"role": "user", "content": "Drop this older question"},
        {"role": "assistant", "content": "Drop this older answer"},
        {"role": "user", "content": "Keep latest"},
    ]
    window = RollingWindow(RollingWindowConfig(enabled=True, keep_system=False, keep_last_turns=1))

    result = window.apply(
        messages, _Tokenizer(), model_limit=10, output_buffer=0, frozen_message_count=3
    )

    remaining_contents = {message.get("content") for message in result.messages}
    assert "Frozen question with extra words" in remaining_contents
    assert "Frozen answer with extra words" in remaining_contents


def test_build_drop_candidates_prioritizes_tool_units_then_turns() -> None:
    messages = _openai_tool_messages()
    window = RollingWindow(RollingWindowConfig(enabled=True, keep_system=True, keep_last_turns=1))

    protected = window._get_protected_indices(messages)
    tool_units = [(2, [3])]
    candidates = window._build_drop_candidates(messages, protected, tool_units)

    assert candidates[0]["type"] == "tool_unit"
    assert candidates[0]["indices"] == [2, 3]
    assert all(candidate["priority"] >= 1 for candidate in candidates)
    assert any(candidate["type"] in {"turn", "single"} for candidate in candidates[1:])


def test_apply_rolling_window_convenience_function_returns_transform() -> None:
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Old question " * 20},
        {"role": "assistant", "content": "Old answer " * 20},
        {"role": "user", "content": "Newest request"},
    ]

    trimmed_messages, transforms = apply_rolling_window(
        messages,
        model_limit=30,
        output_buffer=0,
        keep_last_turns=1,
        config=RollingWindowConfig(enabled=True, keep_system=True, keep_last_turns=1),
    )

    assert len(trimmed_messages) < len(messages)
    assert transforms == ["window_cap:1"]
