"""Tests for rolling window transform."""

import pytest

from headroom import OpenAIProvider, RollingWindowConfig, Tokenizer
from headroom.parser import find_tool_units
from headroom.transforms import RollingWindow

# Create a shared provider for tests
_provider = OpenAIProvider()


def get_tokenizer(model: str = "gpt-4o") -> Tokenizer:
    """Get a tokenizer for tests using OpenAI provider."""
    token_counter = _provider.get_token_counter(model)
    return Tokenizer(token_counter, model)


# Fixtures for realistic message scenarios


@pytest.fixture
def messages_with_system():
    """Messages with a system prompt."""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. You help users with their tasks.",
        },
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": "Of course! What do you need help with?"},
        {"role": "user", "content": "I need to analyze some data."},
        {
            "role": "assistant",
            "content": "I'd be happy to help analyze your data. What kind of data do you have?",
        },
    ]


@pytest.fixture
def messages_with_tool_calls():
    """Messages with tool calls and responses."""
    return [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "Find user with ID 12345"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_user", "arguments": '{"user_id": "12345"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"id": "12345", "name": "Alice", "email": "alice@example.com", "status": "active"}',
        },
        {
            "role": "assistant",
            "content": "I found the user. Alice (ID: 12345) is an active user with email alice@example.com.",
        },
        {"role": "user", "content": "Can you also find user 67890?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_def456",
                    "type": "function",
                    "function": {"name": "get_user", "arguments": '{"user_id": "67890"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_def456",
            "content": '{"id": "67890", "name": "Bob", "email": "bob@example.com", "status": "inactive"}',
        },
        {"role": "assistant", "content": "Found Bob (ID: 67890). This user is currently inactive."},
        {"role": "user", "content": "Thanks for the help!"},
        {"role": "assistant", "content": "You're welcome! Let me know if you need anything else."},
    ]


@pytest.fixture
def messages_multiple_tool_calls():
    """Messages with multiple tool calls in a single assistant message."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search for users Alice and Bob"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_multi_1",
                    "type": "function",
                    "function": {"name": "search_user", "arguments": '{"name": "Alice"}'},
                },
                {
                    "id": "call_multi_2",
                    "type": "function",
                    "function": {"name": "search_user", "arguments": '{"name": "Bob"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_multi_1", "content": '{"id": "1", "name": "Alice"}'},
        {"role": "tool", "tool_call_id": "call_multi_2", "content": '{"id": "2", "name": "Bob"}'},
        {"role": "assistant", "content": "I found both users."},
    ]


@pytest.fixture
def long_conversation():
    """A longer conversation to test window dropping."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. " * 50},  # ~250 tokens
    ]
    # Add 20 turns of conversation
    for i in range(20):
        messages.append(
            {"role": "user", "content": f"This is user message number {i}. " * 10}
        )  # ~50 tokens each
        messages.append(
            {"role": "assistant", "content": f"This is assistant response number {i}. " * 10}
        )
    return messages


class TestRollingWindowProtection:
    """Tests for protected message handling."""

    def test_never_drops_system_prompt(self, messages_with_system):
        """System prompt should never be dropped even under tight budget."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=0,  # Don't protect any turns
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        # Apply with very small budget (smaller than system prompt itself isn't realistic,
        # but we test that system is retained even when budget is tight)
        result = window.apply(
            messages_with_system,
            tokenizer,
            model_limit=200,  # Very small budget
            output_buffer=0,
        )

        # System message should still be present
        system_messages = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_messages) == 1
        assert "You are a helpful assistant" in system_messages[0]["content"]

    def test_never_drops_last_n_turns(self, messages_with_system):
        """Last N turns should be protected from dropping."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=2,  # Protect last 2 turns
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        # Apply with tight budget
        result = window.apply(
            messages_with_system,
            tokenizer,
            model_limit=300,  # Tight budget
            output_buffer=0,
        )

        # Last 2 turns (4 messages) should be preserved
        # The last messages in the conversation are about analyzing data
        last_user = [m for m in result.messages if m.get("role") == "user"][-1]
        last_assistant = [m for m in result.messages if m.get("role") == "assistant"][-1]

        assert "analyze" in last_user["content"].lower() or "data" in last_user["content"].lower()
        assert "data" in last_assistant["content"].lower()

    def test_protects_tool_responses_for_protected_assistant(self, messages_with_tool_calls):
        """Tool responses for protected assistant messages should also be protected."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=1,  # Protect last turn
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        # Get original last assistant with tool calls if any
        # In our fixture, the last assistant doesn't have tool calls, but second-to-last does
        # Modify fixture for this test - use a smaller subset
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Old message"},
            {"role": "assistant", "content": "Old response"},
            {"role": "user", "content": "Find user 999"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_protected",
                        "type": "function",
                        "function": {"name": "get_user", "arguments": '{"id": "999"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_protected",
                "content": '{"id": "999", "name": "Protected User"}',
            },
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "You're welcome!"},
        ]

        window.apply(
            messages,
            tokenizer,
            model_limit=500,
            output_buffer=0,
        )

        # Check that if the assistant with tool_calls is protected, its tool response is too
        protected_indices = window._get_protected_indices(messages)

        # Find the assistant message with tool_calls
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                if i in protected_indices:
                    # The tool response should also be protected
                    for tc in msg.get("tool_calls", []):
                        tc_id = tc.get("id")
                        for j, other_msg in enumerate(messages):
                            if (
                                other_msg.get("role") == "tool"
                                and other_msg.get("tool_call_id") == tc_id
                            ):
                                assert j in protected_indices, (
                                    f"Tool response at {j} should be protected"
                                )


class TestDropPriority:
    """Tests for drop priority ordering."""

    def test_drops_oldest_tool_units_first(self, messages_with_tool_calls):
        """Tool units should be dropped before regular turns."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=1,  # Protect only the last turn
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        get_tokenizer()

        # Check drop candidates ordering
        protected = window._get_protected_indices(messages_with_tool_calls)
        tool_units = find_tool_units(messages_with_tool_calls)
        candidates = window._build_drop_candidates(messages_with_tool_calls, protected, tool_units)

        # First candidates should be tool units (priority 1)
        tool_candidates = [c for c in candidates if c["type"] == "tool_unit"]
        turn_candidates = [c for c in candidates if c["type"] in ("turn", "single")]

        if tool_candidates and turn_candidates:
            # Tool units should come before turns in the sorted list
            first_tool_idx = candidates.index(tool_candidates[0])
            first_turn_idx = candidates.index(turn_candidates[0])
            assert first_tool_idx < first_turn_idx, "Tool units should be dropped before turns"

    def test_drops_oldest_turns_second(self, long_conversation):
        """After tool units, oldest turns should be dropped."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=2,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        # Apply with budget that forces dropping some turns
        result = window.apply(
            long_conversation,
            tokenizer,
            model_limit=2000,  # Will need to drop many turns
            output_buffer=0,
        )

        # The newest messages (user message 19, assistant 19) should still be there
        # Check that at least some oldest messages were dropped
        remaining_content = " ".join(m.get("content", "") or "" for m in result.messages)

        # Last turn should be preserved
        assert "number 19" in remaining_content

        # Old turns should be dropped (message 0 or 1)
        # Due to marker insertion, check that we have fewer messages
        assert len(result.messages) < len(long_conversation)

    def test_tool_unit_atomic(self, messages_with_tool_calls):
        """Tool calls and their responses should be dropped together."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=1,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        # Find tool units before applying
        tool_units_before = find_tool_units(messages_with_tool_calls)
        assert len(tool_units_before) > 0, "Test requires messages with tool calls"

        # Apply transform with tight budget
        result = window.apply(
            messages_with_tool_calls,
            tokenizer,
            model_limit=500,
            output_buffer=0,
        )

        # For each remaining assistant with tool_calls, verify its tool responses exist
        for msg in result.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_call_ids = {tc.get("id") for tc in msg.get("tool_calls", [])}
                # Find matching tool responses
                tool_responses = [
                    m
                    for m in result.messages
                    if m.get("role") == "tool" and m.get("tool_call_id") in tool_call_ids
                ]
                # All tool calls should have their responses
                response_ids = {m.get("tool_call_id") for m in tool_responses}
                assert tool_call_ids == response_ids, "Tool calls must have matching responses"

    def test_never_orphans_tool_result(self, messages_multiple_tool_calls):
        """Tool results should never be left without their assistant message."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=1,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            messages_multiple_tool_calls,
            tokenizer,
            model_limit=300,
            output_buffer=0,
        )

        # Check that no tool message is orphaned
        for msg in result.messages:
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                # Find the corresponding assistant message
                found_assistant = False
                for other_msg in result.messages:
                    if other_msg.get("role") == "assistant" and other_msg.get("tool_calls"):
                        for tc in other_msg.get("tool_calls", []):
                            if tc.get("id") == tool_call_id:
                                found_assistant = True
                                break
                assert found_assistant, f"Tool response {tool_call_id} is orphaned"


class TestTokenBudget:
    """Tests for token budget handling."""

    def test_no_drops_under_budget(self, messages_with_system):
        """No messages should be dropped when under budget."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=2,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        # Apply with large budget
        result = window.apply(
            messages_with_system,
            tokenizer,
            model_limit=100000,  # Plenty of room
            output_buffer=0,
        )

        # No messages should be dropped
        assert len(result.messages) == len(messages_with_system)
        assert len(result.transforms_applied) == 0
        assert result.tokens_before == result.tokens_after

    def test_drops_until_under_budget(self, long_conversation):
        """Should drop messages until under budget."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=2,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        model_limit = 1500  # Tight budget

        result = window.apply(
            long_conversation,
            tokenizer,
            model_limit=model_limit,
            output_buffer=0,
        )

        # Should be under budget after transform
        assert result.tokens_after <= model_limit
        # Should have dropped something
        assert result.tokens_after < result.tokens_before

    def test_respects_output_buffer(self, long_conversation):
        """Output buffer should be subtracted from available budget."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=2,
            output_buffer_tokens=4000,  # Default buffer
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        model_limit = 5000
        output_buffer = 2000

        result = window.apply(
            long_conversation,
            tokenizer,
            model_limit=model_limit,
            output_buffer=output_buffer,
        )

        # Should be under (model_limit - output_buffer)
        available = model_limit - output_buffer
        assert result.tokens_after <= available

    def test_model_limit_parameter(self, long_conversation):
        """Model limit parameter should control the budget."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=2,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        # Test with different model limits
        result_small = window.apply(
            long_conversation,
            tokenizer,
            model_limit=1000,
            output_buffer=0,
        )

        result_large = window.apply(
            long_conversation,
            tokenizer,
            model_limit=3000,
            output_buffer=0,
        )

        # Smaller limit should result in fewer tokens
        assert result_small.tokens_after <= result_large.tokens_after


class TestMarkers:
    """Tests for dropped context markers."""

    def test_inserts_dropped_context_marker(self, long_conversation):
        """A marker should be inserted when content is dropped."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=2,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            long_conversation,
            tokenizer,
            model_limit=1500,
            output_buffer=0,
        )

        # Should have a marker in the result
        assert len(result.markers_inserted) > 0
        assert "dropped_context" in result.markers_inserted[0]

        # Marker should be in the messages
        marker_found = False
        for msg in result.messages:
            content = msg.get("content", "")
            if content and "<headroom:dropped_context" in content:
                marker_found = True
                break
        assert marker_found, "Dropped context marker should be in messages"

    def test_marker_after_system_messages(self, long_conversation):
        """Marker should be inserted after system messages."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=2,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            long_conversation,
            tokenizer,
            model_limit=1500,
            output_buffer=0,
        )

        # Find marker position
        marker_idx = None
        for i, msg in enumerate(result.messages):
            content = msg.get("content", "")
            if content and "<headroom:dropped_context" in content:
                marker_idx = i
                break

        assert marker_idx is not None

        # All system messages should come before the marker
        for i in range(marker_idx):
            assert result.messages[i].get("role") == "system" or "<headroom:" in result.messages[
                i
            ].get("content", "")

    def test_marker_contains_count(self, long_conversation):
        """Marker should contain the count of dropped items."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=2,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            long_conversation,
            tokenizer,
            model_limit=1500,
            output_buffer=0,
        )

        # Check that marker contains count
        assert len(result.markers_inserted) > 0
        marker = result.markers_inserted[0]
        assert 'count="' in marker

        # Transforms applied should indicate count
        assert len(result.transforms_applied) > 0
        assert "window_cap:" in result.transforms_applied[0]


class TestBuildDropCandidates:
    """Tests for the _build_drop_candidates method."""

    def test_tool_units_have_priority_1(self, messages_with_tool_calls):
        """Tool units should have priority 1."""
        config = RollingWindowConfig(enabled=True)
        window = RollingWindow(config)

        protected = window._get_protected_indices(messages_with_tool_calls)
        tool_units = find_tool_units(messages_with_tool_calls)
        candidates = window._build_drop_candidates(messages_with_tool_calls, protected, tool_units)

        tool_candidates = [c for c in candidates if c["type"] == "tool_unit"]
        for tc in tool_candidates:
            assert tc["priority"] == 1

    def test_turns_have_priority_2(self, messages_with_system):
        """Regular turns should have priority 2."""
        config = RollingWindowConfig(enabled=True, keep_last_turns=0)
        window = RollingWindow(config)

        protected = window._get_protected_indices(messages_with_system)
        tool_units = find_tool_units(messages_with_system)  # Empty for this fixture
        candidates = window._build_drop_candidates(messages_with_system, protected, tool_units)

        turn_candidates = [c for c in candidates if c["type"] in ("turn", "single")]
        for tc in turn_candidates:
            assert tc["priority"] == 2

    def test_candidates_sorted_by_age(self, long_conversation):
        """Candidates should be sorted by priority then by position (oldest first)."""
        config = RollingWindowConfig(enabled=True, keep_last_turns=0)
        window = RollingWindow(config)

        protected = {0}  # Only protect system
        tool_units = []
        candidates = window._build_drop_candidates(long_conversation, protected, tool_units)

        # Check that candidates are sorted: first by priority, then by position
        for i in range(1, len(candidates)):
            prev = candidates[i - 1]
            curr = candidates[i]
            # Priority should be non-decreasing
            assert prev["priority"] <= curr["priority"]
            # Within same priority, position should be increasing
            if prev["priority"] == curr["priority"]:
                assert prev["position"] <= curr["position"]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_messages(self):
        """Empty message list should be handled gracefully."""
        config = RollingWindowConfig(enabled=True)
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            [],
            tokenizer,
            model_limit=1000,
            output_buffer=0,
        )

        assert result.messages == []
        # Tokenizer may return a small overhead even for empty messages
        assert result.tokens_before == result.tokens_after
        assert len(result.transforms_applied) == 0

    def test_system_only(self):
        """Conversation with only system message should not drop anything."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        config = RollingWindowConfig(enabled=True, keep_system=True)
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            messages,
            tokenizer,
            model_limit=1000,
            output_buffer=0,
        )

        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "system"

    def test_large_conversation(self, long_conversation):
        """Large conversations should be handled without errors."""
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=3,
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            long_conversation,
            tokenizer,
            model_limit=2000,
            output_buffer=0,
        )

        # Should complete without errors
        assert result.tokens_after <= 2000
        # System should be preserved
        system_msgs = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_msgs) == 1

    def test_all_protected(self):
        """When everything is protected, nothing should be dropped."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=10,  # More than we have
            output_buffer_tokens=0,
        )
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            messages,
            tokenizer,
            model_limit=50,  # Very tight, but everything is protected
            output_buffer=0,
        )

        # Nothing should be dropped since everything is protected
        # The result might still be over budget
        assert len(result.messages) == len(messages)
        assert len(result.transforms_applied) == 0


class TestShouldApply:
    """Tests for the should_apply method."""

    def test_disabled_config_returns_false(self, messages_with_system):
        """should_apply returns False when disabled in config."""
        config = RollingWindowConfig(enabled=False)
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        assert window.should_apply(messages_with_system, tokenizer, model_limit=100) is False

    def test_under_budget_returns_false(self, messages_with_system):
        """should_apply returns False when under budget."""
        config = RollingWindowConfig(enabled=True)
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        # Large budget - should not need to apply
        assert window.should_apply(messages_with_system, tokenizer, model_limit=100000) is False

    def test_over_budget_returns_true(self, long_conversation):
        """should_apply returns True when over budget."""
        config = RollingWindowConfig(enabled=True)
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        # Small budget - should need to apply
        assert window.should_apply(long_conversation, tokenizer, model_limit=500) is True


class TestConvenienceFunction:
    """Tests for the apply_rolling_window convenience function."""

    def test_convenience_function(self, long_conversation):
        """The convenience function should work correctly."""
        from headroom.transforms.rolling_window import apply_rolling_window

        messages, transforms = apply_rolling_window(
            long_conversation,
            model_limit=1500,
            output_buffer=500,
            keep_last_turns=2,
        )

        # Should have applied transform
        assert len(messages) < len(long_conversation)
        assert len(transforms) > 0

    def test_convenience_function_with_config(self, long_conversation):
        """The convenience function should accept a config."""
        from headroom.transforms.rolling_window import apply_rolling_window

        config = RollingWindowConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=3,
        )

        messages, transforms = apply_rolling_window(
            long_conversation,
            model_limit=1500,
            output_buffer=500,
            keep_last_turns=3,
            config=config,
        )

        assert len(messages) < len(long_conversation)


class TestTransformResult:
    """Tests for TransformResult fields."""

    def test_tokens_before_after(self, long_conversation):
        """tokens_before and tokens_after should be correct."""
        config = RollingWindowConfig(enabled=True, keep_last_turns=2)
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        tokens_before = tokenizer.count_messages(long_conversation)

        result = window.apply(
            long_conversation,
            tokenizer,
            model_limit=1500,
            output_buffer=0,
        )

        assert result.tokens_before == tokens_before
        assert result.tokens_after <= result.tokens_before
        assert result.tokens_after == tokenizer.count_messages(result.messages)

    def test_transforms_applied_field(self, long_conversation):
        """transforms_applied should contain window_cap info."""
        config = RollingWindowConfig(enabled=True, keep_last_turns=2)
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            long_conversation,
            tokenizer,
            model_limit=1500,
            output_buffer=0,
        )

        assert len(result.transforms_applied) > 0
        assert any("window_cap" in t for t in result.transforms_applied)

    def test_warnings_field(self, messages_with_system):
        """warnings field should be present (may be empty)."""
        config = RollingWindowConfig(enabled=True)
        window = RollingWindow(config)
        tokenizer = get_tokenizer()

        result = window.apply(
            messages_with_system,
            tokenizer,
            model_limit=100000,
            output_buffer=0,
        )

        # warnings should be a list (possibly empty)
        assert isinstance(result.warnings, list)
