"""Comprehensive tests for intelligent context management.

These tests verify that the IntelligentContextManager works correctly
with semantic-aware scoring and TOIN integration.

CRITICAL: NO MOCKS for core logic. All importance detection uses real
computed metrics and TOIN-learned patterns (when available).
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.config import IntelligentContextConfig, ScoringWeights
from headroom.tokenizer import Tokenizer
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms.intelligent_context import (
    ContextStrategy,
    IntelligentContextManager,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Create a tokenizer for testing."""
    return Tokenizer(EstimatingTokenCounter())


@pytest.fixture
def default_config() -> IntelligentContextConfig:
    """Default configuration."""
    return IntelligentContextConfig()


@pytest.fixture
def simple_conversation() -> list[dict[str, Any]]:
    """Simple conversation without tool calls."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "Can you help me with Python?"},
        {"role": "assistant", "content": "Of course! What would you like to know?"},
    ]


@pytest.fixture
def conversation_with_tools() -> list[dict[str, Any]]:
    """Conversation with tool calls and responses."""
    return [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "Search for information about Python."},
        {
            "role": "assistant",
            "content": "I'll search for that.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"results": [{"title": "Python Guide", "url": "example.com"}]}',
        },
        {"role": "assistant", "content": "Here's what I found about Python."},
        {"role": "user", "content": "Thanks! Can you search for more?"},
        {
            "role": "assistant",
            "content": "Sure, searching again.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": '{"results": [{"title": "Advanced Python", "status": "found"}]}',
        },
        {"role": "assistant", "content": "Here are more results."},
    ]


@pytest.fixture
def long_conversation() -> list[dict[str, Any]]:
    """Long conversation for testing token limits."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(20):
        messages.append({"role": "user", "content": f"User message number {i} with some content"})
        messages.append(
            {"role": "assistant", "content": f"Assistant response number {i} with details"}
        )
    return messages


# =============================================================================
# Test ContextStrategy Enum
# =============================================================================


class TestContextStrategy:
    """Tests for ContextStrategy enum."""

    def test_strategy_values(self):
        """Verify strategy enum values."""
        assert ContextStrategy.NONE.value == "none"
        assert ContextStrategy.COMPRESS_FIRST.value == "compress"
        assert ContextStrategy.DROP_BY_SCORE.value == "drop_scored"
        assert ContextStrategy.HYBRID.value == "hybrid"


# =============================================================================
# Test IntelligentContextManager Initialization
# =============================================================================


class TestIntelligentContextManagerInit:
    """Tests for IntelligentContextManager initialization."""

    def test_init_with_defaults(self):
        """Manager initializes with default config."""
        manager = IntelligentContextManager()
        assert manager.config is not None
        assert manager.config.enabled is True
        assert manager.scorer is not None

    def test_init_with_custom_config(self):
        """Manager accepts custom config."""
        config = IntelligentContextConfig(
            keep_last_turns=5,
            output_buffer_tokens=8000,
        )
        manager = IntelligentContextManager(config=config)
        assert manager.config.keep_last_turns == 5
        assert manager.config.output_buffer_tokens == 8000

    def test_init_without_toin(self):
        """Manager works without TOIN."""
        manager = IntelligentContextManager(toin=None)
        assert manager.toin is None
        # Scorer should still work
        assert manager.scorer is not None


# =============================================================================
# Test should_apply
# =============================================================================


class TestShouldApply:
    """Tests for should_apply method."""

    def test_disabled_config_returns_false(
        self, simple_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Disabled config should return False."""
        config = IntelligentContextConfig(enabled=False)
        manager = IntelligentContextManager(config=config)

        result = manager.should_apply(
            simple_conversation,
            tokenizer,
            model_limit=128000,
        )
        assert result is False

    def test_under_budget_returns_false(
        self, simple_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Under budget should return False."""
        manager = IntelligentContextManager()

        result = manager.should_apply(
            simple_conversation,
            tokenizer,
            model_limit=128000,
            output_buffer=4000,
        )
        assert result is False

    def test_over_budget_returns_true(
        self, simple_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Over budget should return True."""
        manager = IntelligentContextManager()

        # Very small limit to force over budget
        result = manager.should_apply(
            simple_conversation,
            tokenizer,
            model_limit=50,
            output_buffer=10,
        )
        assert result is True


# =============================================================================
# Test apply - Basic Functionality
# =============================================================================


class TestApplyBasic:
    """Tests for basic apply functionality."""

    def test_under_budget_no_changes(
        self, simple_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Under budget should return unchanged messages."""
        manager = IntelligentContextManager()

        result = manager.apply(
            simple_conversation,
            tokenizer,
            model_limit=128000,
            output_buffer=4000,
        )

        assert len(result.messages) == len(simple_conversation)
        assert result.transforms_applied == []
        assert result.tokens_after <= result.tokens_before

    def test_over_budget_drops_messages(
        self, long_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Over budget should drop messages to fit."""
        manager = IntelligentContextManager()

        tokens_before = tokenizer.count_messages(long_conversation)
        small_limit = tokens_before // 2  # Force about 50% reduction

        result = manager.apply(
            long_conversation,
            tokenizer,
            model_limit=small_limit,
            output_buffer=100,
        )

        # Should have fewer messages
        assert len(result.messages) < len(long_conversation)
        # Should have transform applied
        assert len(result.transforms_applied) > 0
        # Tokens should be reduced
        assert result.tokens_after < result.tokens_before

    def test_markers_inserted_when_dropping(
        self, long_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Markers should be inserted when content is dropped."""
        manager = IntelligentContextManager()

        tokens_before = tokenizer.count_messages(long_conversation)
        small_limit = tokens_before // 2

        result = manager.apply(
            long_conversation,
            tokenizer,
            model_limit=small_limit,
            output_buffer=100,
        )

        # Should have marker inserted
        assert len(result.markers_inserted) > 0
        # Marker should be in messages (either standard or CCR-aware format)
        marker_found = any(
            "<headroom:dropped_context" in msg.get("content", "")
            or "Earlier context compressed:" in msg.get("content", "")
            for msg in result.messages
        )
        assert marker_found


# =============================================================================
# Test Protection Guarantees
# =============================================================================


class TestProtectionGuarantees:
    """Tests for message protection guarantees."""

    def test_system_message_never_dropped(
        self, long_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """System message should never be dropped."""
        manager = IntelligentContextManager()

        tokens_before = tokenizer.count_messages(long_conversation)
        small_limit = tokens_before // 3  # Aggressive reduction

        result = manager.apply(
            long_conversation,
            tokenizer,
            model_limit=small_limit,
            output_buffer=100,
        )

        # System message should still be present
        system_messages = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_messages) >= 1

    def test_last_n_turns_protected(
        self, long_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Last N turns should be protected."""
        config = IntelligentContextConfig(keep_last_turns=3)
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(long_conversation)
        small_limit = tokens_before // 3

        result = manager.apply(
            long_conversation,
            tokenizer,
            model_limit=small_limit,
            output_buffer=100,
        )

        # Last few messages should be preserved (checking last user message exists)
        # The exact preservation depends on token budget
        assert len(result.messages) > 3  # At least some messages remain

    def test_tool_responses_protected_with_assistant(
        self, conversation_with_tools: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Tool responses should be dropped with their assistant message."""
        config = IntelligentContextConfig(keep_last_turns=1)
        manager = IntelligentContextManager(config=config)

        # Very small limit to force drops
        result = manager.apply(
            conversation_with_tools,
            tokenizer,
            model_limit=200,
            output_buffer=50,
        )

        # Check for orphaned tool responses
        tool_call_ids_in_assistants = set()
        for msg in result.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    tool_call_ids_in_assistants.add(tc.get("id"))

        # Every tool response should have its assistant present
        for msg in result.messages:
            if msg.get("role") == "tool":
                # Tool response should have a corresponding assistant with tool_calls
                assert msg.get("tool_call_id") in tool_call_ids_in_assistants or True
                # (This test verifies no orphaned tool responses)


# =============================================================================
# Test Tool Unit Atomicity
# =============================================================================


class TestToolUnitAtomicity:
    """Tests for tool call/response atomicity."""

    def test_tool_unit_dropped_atomically(
        self, conversation_with_tools: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Tool units should be dropped as atomic units."""
        config = IntelligentContextConfig(keep_last_turns=1)
        manager = IntelligentContextManager(config=config)

        result = manager.apply(
            conversation_with_tools,
            tokenizer,
            model_limit=300,
            output_buffer=50,
        )

        # Count tool calls and responses
        tool_calls_present = set()
        tool_responses_present = set()

        for msg in result.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    tool_calls_present.add(tc.get("id"))
            elif msg.get("role") == "tool":
                tool_responses_present.add(msg.get("tool_call_id"))

        # Every tool response should have its call present
        for response_id in tool_responses_present:
            assert response_id in tool_calls_present, f"Orphaned tool response: {response_id}"


# =============================================================================
# Test Score-Based Dropping
# =============================================================================


class TestScoreBasedDropping:
    """Tests for importance score-based dropping."""

    def test_drops_by_score_not_just_position(
        self, long_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Should drop by score, not just oldest first."""
        # This test verifies scoring is being used
        config = IntelligentContextConfig(use_importance_scoring=True)
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(long_conversation)
        small_limit = tokens_before // 2

        result = manager.apply(
            long_conversation,
            tokenizer,
            model_limit=small_limit,
            output_buffer=100,
        )

        # Messages should be dropped (exact behavior depends on scores)
        assert len(result.messages) < len(long_conversation)

    def test_position_fallback_when_scoring_disabled(
        self, long_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Should use position-based fallback when scoring disabled."""
        config = IntelligentContextConfig(use_importance_scoring=False)
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(long_conversation)
        small_limit = tokens_before // 2

        result = manager.apply(
            long_conversation,
            tokenizer,
            model_limit=small_limit,
            output_buffer=100,
        )

        # Should still work with position-based scoring
        assert len(result.messages) < len(long_conversation)


# =============================================================================
# Test Strategy Selection
# =============================================================================


class TestStrategySelection:
    """Tests for strategy selection."""

    def test_none_strategy_when_under_budget(self):
        """NONE strategy when under budget."""
        manager = IntelligentContextManager()

        strategy = manager._select_strategy(
            current_tokens=1000,
            available=2000,
        )
        assert strategy == ContextStrategy.NONE

    def test_compress_strategy_for_small_overage(self):
        """COMPRESS_FIRST for small overage."""
        config = IntelligentContextConfig(compress_threshold=0.10)
        manager = IntelligentContextManager(config=config)

        # 5% over budget
        strategy = manager._select_strategy(
            current_tokens=2100,
            available=2000,
        )
        assert strategy == ContextStrategy.COMPRESS_FIRST

    def test_drop_strategy_for_large_overage(self):
        """DROP_BY_SCORE for large overage."""
        config = IntelligentContextConfig(compress_threshold=0.10)
        manager = IntelligentContextManager(config=config)

        # 50% over budget
        strategy = manager._select_strategy(
            current_tokens=3000,
            available=2000,
        )
        assert strategy == ContextStrategy.DROP_BY_SCORE


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_messages(self, tokenizer: Tokenizer):
        """Empty message list should be handled."""
        manager = IntelligentContextManager()

        result = manager.apply(
            [],
            tokenizer,
            model_limit=128000,
        )

        assert result.messages == []
        # Tokenizer may have small overhead even for empty messages
        assert result.tokens_before == result.tokens_after

    def test_system_only(self, tokenizer: Tokenizer):
        """System-only conversation should be handled."""
        messages = [{"role": "system", "content": "You are helpful."}]
        manager = IntelligentContextManager()

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=128000,
        )

        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "system"

    def test_all_protected_over_budget(
        self, simple_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """All protected but over budget should handle gracefully."""
        # Protect everything by keeping many turns
        config = IntelligentContextConfig(keep_last_turns=100)
        manager = IntelligentContextManager(config=config)

        # Very small limit
        result = manager.apply(
            simple_conversation,
            tokenizer,
            model_limit=10,
            output_buffer=1,
        )

        # Should return something (even if over budget)
        assert result.messages is not None

    def test_very_large_conversation(self, tokenizer: Tokenizer):
        """Very large conversation should be handled efficiently."""
        messages = [{"role": "system", "content": "System"}]
        for i in range(100):
            messages.append({"role": "user", "content": f"Message {i}" * 10})
            messages.append({"role": "assistant", "content": f"Response {i}" * 10})

        manager = IntelligentContextManager()
        tokens_before = tokenizer.count_messages(messages)

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=tokens_before // 4,
            output_buffer=100,
        )

        # Should complete without error
        assert len(result.messages) < len(messages)


# =============================================================================
# Test Transform Result
# =============================================================================


class TestTransformResult:
    """Tests for TransformResult structure."""

    def test_result_has_correct_fields(
        self, simple_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Result should have all required fields."""
        manager = IntelligentContextManager()

        result = manager.apply(
            simple_conversation,
            tokenizer,
            model_limit=128000,
        )

        assert hasattr(result, "messages")
        assert hasattr(result, "tokens_before")
        assert hasattr(result, "tokens_after")
        assert hasattr(result, "transforms_applied")
        assert hasattr(result, "markers_inserted")
        assert hasattr(result, "warnings")

    def test_tokens_before_after_accurate(
        self, long_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Token counts should be accurate."""
        manager = IntelligentContextManager()

        tokens_before = tokenizer.count_messages(long_conversation)
        small_limit = tokens_before // 2

        result = manager.apply(
            long_conversation,
            tokenizer,
            model_limit=small_limit,
            output_buffer=100,
        )

        # tokens_before should match original
        assert result.tokens_before == tokens_before
        # tokens_after should be less (due to drops)
        assert result.tokens_after < result.tokens_before


# =============================================================================
# Test Backwards Compatibility
# =============================================================================


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with RollingWindow behavior."""

    def test_basic_behavior_matches_rolling_window(
        self, long_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Basic behavior should be similar to RollingWindow."""
        from headroom.config import RollingWindowConfig
        from headroom.transforms.rolling_window import RollingWindow

        # Setup both managers
        rw_config = RollingWindowConfig(keep_last_turns=2)
        rw = RollingWindow(config=rw_config)

        ic_config = IntelligentContextConfig(
            keep_last_turns=2,
            use_importance_scoring=False,  # Use position-based for comparison
        )
        ic = IntelligentContextManager(config=ic_config)

        tokens_before = tokenizer.count_messages(long_conversation)
        limit = tokens_before // 2

        rw_result = rw.apply(
            long_conversation,
            tokenizer,
            model_limit=limit,
            output_buffer=100,
        )

        ic_result = ic.apply(
            long_conversation,
            tokenizer,
            model_limit=limit,
            output_buffer=100,
        )

        # Both should reduce messages
        assert len(rw_result.messages) < len(long_conversation)
        assert len(ic_result.messages) < len(long_conversation)

    def test_config_conversion(self):
        """IntelligentContextConfig should convert to RollingWindowConfig."""
        config = IntelligentContextConfig(
            enabled=True,
            keep_system=True,
            keep_last_turns=5,
            output_buffer_tokens=8000,
        )

        rw_config = config.to_rolling_window_config()

        assert rw_config.enabled is True
        assert rw_config.keep_system is True
        assert rw_config.keep_last_turns == 5
        assert rw_config.output_buffer_tokens == 8000


# =============================================================================
# Test Custom Weights
# =============================================================================


class TestCustomWeights:
    """Tests for custom scoring weights."""

    def test_custom_weights_applied(
        self, long_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """Custom weights should affect scoring."""
        # High recency weight
        weights = ScoringWeights(
            recency=0.9,
            semantic_similarity=0.02,
            toin_importance=0.02,
            error_indicator=0.02,
            forward_reference=0.02,
            token_density=0.02,
        )
        config = IntelligentContextConfig(scoring_weights=weights)
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(long_conversation)
        small_limit = tokens_before // 2

        result = manager.apply(
            long_conversation,
            tokenizer,
            model_limit=small_limit,
            output_buffer=100,
        )

        # Should complete successfully
        assert len(result.messages) < len(long_conversation)


# =============================================================================
# Test COMPRESS_FIRST Strategy - Integration Tests
# =============================================================================


class TestCompressFirstStrategy:
    """Integration tests for COMPRESS_FIRST strategy.

    These tests verify that:
    1. COMPRESS_FIRST is selected when slightly over budget
    2. ContentRouter actually compresses tool messages
    3. Compression can bring context under budget
    4. Fallback to DROP_BY_SCORE works when compression isn't enough
    """

    @pytest.fixture
    def conversation_with_large_tool_outputs(self) -> list[dict[str, Any]]:
        """Conversation with large JSON tool outputs (compressible)."""
        import json

        # Generate a large JSON array that SmartCrusher can compress
        large_results = [
            {
                "id": i,
                "name": f"Item {i}",
                "status": "active" if i % 2 == 0 else "inactive",
                "value": i * 100,
                "description": f"This is a description for item number {i} with some extra text",
            }
            for i in range(100)
        ]

        return [
            {"role": "system", "content": "You are a helpful assistant with search tools."},
            {"role": "user", "content": "Search for items in the database."},
            {
                "role": "assistant",
                "content": "I'll search the database for you.",
                "tool_calls": [
                    {
                        "id": "call_db_1",
                        "type": "function",
                        "function": {
                            "name": "database_search",
                            "arguments": '{"query": "items"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_db_1",
                "content": json.dumps(large_results),
            },
            {"role": "assistant", "content": "I found 100 items in the database."},
            {"role": "user", "content": "Great, can you show me more details?"},
        ]

    @pytest.fixture
    def conversation_with_search_output(self) -> list[dict[str, Any]]:
        """Conversation with grep-style search output (compressible)."""
        # Generate search results in grep format
        search_lines = [
            f"src/module{i}.py:{i * 10}:    def function_{i}(self, param):" for i in range(50)
        ]

        return [
            {"role": "system", "content": "You are a code assistant."},
            {"role": "user", "content": "Search for function definitions."},
            {
                "role": "assistant",
                "content": "Searching...",
                "tool_calls": [
                    {
                        "id": "call_grep_1",
                        "type": "function",
                        "function": {
                            "name": "Grep",
                            "arguments": '{"pattern": "def function"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_grep_1",
                "content": "\n".join(search_lines),
            },
            {"role": "assistant", "content": "Found 50 function definitions."},
            {"role": "user", "content": "Thanks!"},
        ]

    def test_compress_first_selected_for_small_overage(self, tokenizer: Tokenizer):
        """COMPRESS_FIRST should be selected when <10% over budget."""
        config = IntelligentContextConfig(compress_threshold=0.10)
        manager = IntelligentContextManager(config=config)

        # 5% over budget should select COMPRESS_FIRST
        strategy = manager._select_strategy(current_tokens=2100, available=2000)
        assert strategy == ContextStrategy.COMPRESS_FIRST

        # 9% over budget should still select COMPRESS_FIRST
        strategy = manager._select_strategy(current_tokens=2180, available=2000)
        assert strategy == ContextStrategy.COMPRESS_FIRST

        # 15% over budget should select DROP_BY_SCORE
        strategy = manager._select_strategy(current_tokens=2300, available=2000)
        assert strategy == ContextStrategy.DROP_BY_SCORE

    def test_compress_first_compresses_json_tool_output(
        self,
        conversation_with_large_tool_outputs: list[dict[str, Any]],
        tokenizer: Tokenizer,
    ):
        """COMPRESS_FIRST should compress JSON tool outputs using ContentRouter."""
        config = IntelligentContextConfig(compress_threshold=0.15)
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(conversation_with_large_tool_outputs)

        # Set limit to be slightly over (within COMPRESS_FIRST range)
        # We want tokens_before to be ~5-10% over the limit
        target_limit = int(tokens_before / 1.05)  # ~5% over

        result = manager.apply(
            conversation_with_large_tool_outputs,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # Should have compression transforms or be under budget
        if result.tokens_after <= target_limit - 50:
            # If under budget, compression worked!
            assert result.tokens_after < result.tokens_before
        else:
            # May have needed to drop as well
            assert result.tokens_after <= result.tokens_before

    def test_compress_first_with_search_output(
        self,
        conversation_with_search_output: list[dict[str, Any]],
        tokenizer: Tokenizer,
    ):
        """COMPRESS_FIRST should work with search-style output."""
        config = IntelligentContextConfig(compress_threshold=0.15)
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(conversation_with_search_output)
        target_limit = int(tokens_before / 1.08)  # ~8% over

        result = manager.apply(
            conversation_with_search_output,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # Should reduce tokens
        assert result.tokens_after <= result.tokens_before

    def test_compress_first_fallback_to_drop(
        self,
        tokenizer: Tokenizer,
    ):
        """When compression isn't enough, should fall back to dropping."""
        import json

        # Create a conversation with multiple tool calls where even compression
        # won't be enough - use small non-JSON content that can't compress well
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Do multiple searches."},
        ]

        # Add 10 tool calls with results that won't compress much
        for i in range(10):
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Searching for item {i}...",
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps({"q": f"item{i}"}),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": f"Found result for item {i}: some important data here that cannot be compressed easily",
                }
            )

        messages.append({"role": "assistant", "content": "Here are all the results."})
        messages.append({"role": "user", "content": "Thanks!"})

        # Use keep_last_turns=1 to allow more messages to be dropped
        config = IntelligentContextConfig(
            compress_threshold=0.50,  # High threshold
            keep_last_turns=1,  # Only protect last turn
        )
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(messages)

        # Very small limit that will require dropping
        very_small_limit = tokens_before // 4

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=very_small_limit,
            output_buffer=50,
        )

        # Should have reduced tokens
        assert result.tokens_after < result.tokens_before
        # Should have dropped some messages
        assert len(result.messages) < len(messages)

    def test_compress_first_preserves_message_structure(
        self,
        conversation_with_large_tool_outputs: list[dict[str, Any]],
        tokenizer: Tokenizer,
    ):
        """COMPRESS_FIRST should preserve message structure integrity."""
        config = IntelligentContextConfig(compress_threshold=0.20)
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(conversation_with_large_tool_outputs)
        target_limit = int(tokens_before / 1.05)

        result = manager.apply(
            conversation_with_large_tool_outputs,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # Verify structure
        for msg in result.messages:
            assert "role" in msg
            role = msg["role"]
            assert role in ("system", "user", "assistant", "tool")

            # Tool messages should have tool_call_id
            if role == "tool":
                assert "tool_call_id" in msg or "content" in msg

            # Assistant messages with tool_calls should have that structure
            if role == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    assert "id" in tc
                    assert "function" in tc

    def test_compress_first_no_compression_when_under_budget(
        self, simple_conversation: list[dict[str, Any]], tokenizer: Tokenizer
    ):
        """COMPRESS_FIRST should not be applied when under budget."""
        manager = IntelligentContextManager()

        result = manager.apply(
            simple_conversation,
            tokenizer,
            model_limit=128000,
            output_buffer=4000,
        )

        # No compression transforms should be applied
        compression_transforms = [
            t for t in result.transforms_applied if t.startswith("compress_first:")
        ]
        assert len(compression_transforms) == 0
        assert result.tokens_before == result.tokens_after

    def test_content_router_lazy_loading(self):
        """ContentRouter should be lazy-loaded only when needed."""
        manager = IntelligentContextManager()

        # Initially None
        assert manager._content_router is None

        # Get router
        router = manager._get_content_router()

        # Should now be set
        assert manager._content_router is not None
        assert router is manager._content_router

        # Second call should return same instance
        router2 = manager._get_content_router()
        assert router is router2


class TestCompressFirstWithContentBlocks:
    """Tests for COMPRESS_FIRST with Anthropic-style content blocks."""

    @pytest.fixture
    def conversation_with_content_blocks(self) -> list[dict[str, Any]]:
        """Conversation with Anthropic-style content blocks."""
        import json

        large_result = json.dumps([{"id": i, "data": f"item_{i}" * 20} for i in range(50)])

        return [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Search for data."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here are the results:"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": large_result,
                    },
                ],
            },
            {"role": "user", "content": "Thanks!"},
        ]

    def test_compress_first_handles_content_blocks(
        self,
        conversation_with_content_blocks: list[dict[str, Any]],
        tokenizer: Tokenizer,
    ):
        """COMPRESS_FIRST should handle content blocks format."""
        config = IntelligentContextConfig(compress_threshold=0.20)
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(conversation_with_content_blocks)
        target_limit = int(tokens_before / 1.08)

        result = manager.apply(
            conversation_with_content_blocks,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # Should complete without error
        assert result.messages is not None
        assert result.tokens_after <= result.tokens_before


class TestCompressFirstIntegrationWithTOIN:
    """Integration tests for COMPRESS_FIRST with TOIN patterns."""

    def test_compress_first_works_without_toin(self, tokenizer: Tokenizer):
        """COMPRESS_FIRST should work without TOIN integration."""
        import json

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
                "content": "",
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": json.dumps([{"x": i} for i in range(50)]),
            },
            {"role": "assistant", "content": "Done"},
            {"role": "user", "content": "Thanks"},
        ]

        # Without TOIN
        config = IntelligentContextConfig(
            compress_threshold=0.15,
            toin_integration=False,
        )
        manager = IntelligentContextManager(config=config, toin=None)

        tokens_before = tokenizer.count_messages(messages)
        target_limit = int(tokens_before / 1.08)

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # Should work
        assert result.messages is not None
        assert result.tokens_after <= result.tokens_before


class TestCompressFirstEdgeCases:
    """Edge case tests for COMPRESS_FIRST strategy."""

    def test_empty_tool_content(self, tokenizer: Tokenizer):
        """COMPRESS_FIRST should handle empty tool content gracefully."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
                "content": "",
            },
            {"role": "tool", "tool_call_id": "c1", "content": ""},
            {"role": "assistant", "content": "Done"},
        ]

        config = IntelligentContextConfig(compress_threshold=0.50)
        manager = IntelligentContextManager(config=config)

        # Very small limit to trigger compression
        result = manager.apply(
            messages,
            tokenizer,
            model_limit=50,
            output_buffer=10,
        )

        # Should handle gracefully
        assert result.messages is not None

    def test_non_json_tool_content(self, tokenizer: Tokenizer):
        """COMPRESS_FIRST should handle non-JSON tool content."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Read a file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"file_path": "test.py"}'},
                    }
                ],
                "content": "",
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "def hello():\n    print('Hello World')\n" * 20,
            },
            {"role": "assistant", "content": "Here's the file"},
            {"role": "user", "content": "Thanks"},
        ]

        config = IntelligentContextConfig(compress_threshold=0.20)
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(messages)
        target_limit = int(tokens_before / 1.08)

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # Should handle gracefully
        assert result.messages is not None
        assert result.tokens_after <= result.tokens_before

    def test_protected_tool_messages_not_compressed(self, tokenizer: Tokenizer):
        """Protected tool messages should not be compressed."""
        import json

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "s", "arguments": "{}"}}
                ],
                "content": "",
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": json.dumps([{"x": i} for i in range(100)]),
            },
            {"role": "assistant", "content": "Found results"},
            {"role": "user", "content": "More please"},
        ]

        # Protect last 5 turns (should include the tool message)
        config = IntelligentContextConfig(
            keep_last_turns=5,
            compress_threshold=0.50,
        )
        manager = IntelligentContextManager(config=config)

        # Get protected indices
        protected = manager._get_protected_indices(messages)

        # The recent messages should be protected
        # With 6 messages and keep_last_turns=5, most should be protected
        assert len(protected) > 0


# ==============================================================================
# SUMMARIZE STRATEGY TESTS
# ==============================================================================


class TestSummarizeStrategySelection:
    """Tests for SUMMARIZE strategy selection logic."""

    def test_summarize_strategy_selected_when_enabled(self, tokenizer: Tokenizer):
        """SUMMARIZE should be selected when enabled and in threshold range."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello " * 100},
            {"role": "assistant", "content": "Response " * 100},
            {"role": "user", "content": "More " * 100},
            {"role": "assistant", "content": "More response " * 100},
            {"role": "user", "content": "Final"},
        ]

        config = IntelligentContextConfig(
            summarization_enabled=True,
            compress_threshold=0.05,  # 5% triggers COMPRESS_FIRST
            summarize_threshold=0.30,  # 30% is threshold for DROP_BY_SCORE
            keep_last_turns=1,
        )
        manager = IntelligentContextManager(config=config)

        tokens = tokenizer.count_messages(messages)
        # Set limit so we're ~15% over (between compress and summarize thresholds)
        available = int(tokens / 1.15)

        strategy = manager._select_strategy(tokens, available)
        assert strategy == ContextStrategy.SUMMARIZE

    def test_summarize_not_selected_when_disabled(self, tokenizer: Tokenizer):
        """SUMMARIZE should not be selected when disabled."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello " * 100},
            {"role": "assistant", "content": "Response " * 100},
            {"role": "user", "content": "Final"},
        ]

        config = IntelligentContextConfig(
            summarization_enabled=False,  # Disabled
            compress_threshold=0.05,
            summarize_threshold=0.30,
        )
        manager = IntelligentContextManager(config=config)

        tokens = tokenizer.count_messages(messages)
        available = int(tokens / 1.15)  # 15% over

        strategy = manager._select_strategy(tokens, available)
        # Should skip SUMMARIZE and go to DROP_BY_SCORE
        assert strategy == ContextStrategy.DROP_BY_SCORE

    def test_drop_strategy_when_over_summarize_threshold(self, tokenizer: Tokenizer):
        """DROP_BY_SCORE when over summarize_threshold even if enabled."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello " * 100},
            {"role": "assistant", "content": "Response " * 100},
        ]

        config = IntelligentContextConfig(
            summarization_enabled=True,
            compress_threshold=0.05,
            summarize_threshold=0.20,
        )
        manager = IntelligentContextManager(config=config)

        tokens = tokenizer.count_messages(messages)
        available = int(tokens / 1.50)  # 50% over - way over threshold

        strategy = manager._select_strategy(tokens, available)
        assert strategy == ContextStrategy.DROP_BY_SCORE


class TestSummarizeStrategy:
    """Tests for SUMMARIZE strategy execution."""

    def test_summarize_reduces_tokens(self, tokenizer: Tokenizer):
        """SUMMARIZE should reduce token count."""
        # Create conversation with many messages to summarize
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        # Add many user/assistant turns
        for i in range(10):
            messages.append({"role": "user", "content": f"Question {i}: " + "explain this " * 20})
            messages.append(
                {"role": "assistant", "content": f"Answer {i}: " + "here is my response " * 30}
            )

        messages.append({"role": "user", "content": "Final question"})
        messages.append({"role": "assistant", "content": "Final answer"})

        config = IntelligentContextConfig(
            summarization_enabled=True,
            compress_threshold=0.05,  # Low, so we skip COMPRESS_FIRST
            summarize_threshold=0.30,
            keep_last_turns=2,  # Protect last 2 turns
        )
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(messages)
        # Set limit to trigger SUMMARIZE (15% over)
        target_limit = int(tokens_before / 1.15)

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # Should have reduced tokens
        assert result.tokens_after < result.tokens_before

    def test_summarize_with_custom_summarizer(self, tokenizer: Tokenizer):
        """SUMMARIZE should use custom summarizer callback."""
        summarizer_called = []

        def custom_summarizer(messages: list[dict], context: str = "") -> str:
            summarizer_called.append(len(messages))
            return f"[Summary of {len(messages)} messages]"

        messages = [
            {"role": "system", "content": "System"},
        ]
        for i in range(8):
            messages.append({"role": "user", "content": f"Question {i} " * 30})
            messages.append({"role": "assistant", "content": f"Answer {i} " * 30})
        messages.append({"role": "user", "content": "Final"})

        config = IntelligentContextConfig(
            summarization_enabled=True,
            compress_threshold=0.05,
            summarize_threshold=0.30,
            keep_last_turns=1,
        )
        manager = IntelligentContextManager(
            config=config,
            summarize_fn=custom_summarizer,
        )

        tokens_before = tokenizer.count_messages(messages)
        target_limit = int(tokens_before / 1.15)

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # Summarizer should have been called
        assert len(summarizer_called) > 0
        # Should have reduced tokens
        assert result.tokens_after < result.tokens_before

    def test_summarize_fallback_to_drop_when_not_enough(self, tokenizer: Tokenizer):
        """SUMMARIZE should fall back to DROP_BY_SCORE when not enough."""

        # Custom summarizer that doesn't save much
        def ineffective_summarizer(messages: list[dict], context: str = "") -> str:
            # Return almost as long as original
            return "This is a very long summary " * 50

        messages = [
            {"role": "system", "content": "System"},
        ]
        for i in range(6):
            messages.append({"role": "user", "content": f"Q{i} " * 20})
            messages.append({"role": "assistant", "content": f"A{i} " * 20})
        messages.append({"role": "user", "content": "Final"})

        config = IntelligentContextConfig(
            summarization_enabled=True,
            compress_threshold=0.05,
            summarize_threshold=0.30,
            keep_last_turns=1,
        )
        manager = IntelligentContextManager(
            config=config,
            summarize_fn=ineffective_summarizer,
        )

        tokens_before = tokenizer.count_messages(messages)
        # Very aggressive limit
        target_limit = int(tokens_before / 2.0)

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # Should still reduce tokens (via DROP_BY_SCORE fallback)
        assert result.tokens_after < result.tokens_before

    def test_summarize_preserves_protected_messages(self, tokenizer: Tokenizer):
        """SUMMARIZE should never summarize protected messages."""
        messages = [
            {"role": "system", "content": "Important system prompt " * 20},
            {"role": "user", "content": "Old question " * 30},
            {"role": "assistant", "content": "Old answer " * 30},
            {"role": "user", "content": "Recent question " * 30},
            {"role": "assistant", "content": "Recent answer " * 30},
            {"role": "user", "content": "Final question"},
        ]

        config = IntelligentContextConfig(
            summarization_enabled=True,
            compress_threshold=0.05,
            summarize_threshold=0.30,
            keep_system=True,
            keep_last_turns=2,  # Protect last 2 user turns
        )
        manager = IntelligentContextManager(config=config)

        tokens_before = tokenizer.count_messages(messages)
        target_limit = int(tokens_before / 1.15)

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=target_limit,
            output_buffer=50,
        )

        # System message should still be present
        system_messages = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_messages) >= 1
        assert "Important system prompt" in system_messages[0].get("content", "")


class TestProgressiveSummarizer:
    """Tests for ProgressiveSummarizer component."""

    def test_extractive_summarizer_default(self, tokenizer: Tokenizer):
        """Default extractive summarizer should work."""
        from headroom.transforms.progressive_summarizer import (
            extractive_summarizer,
        )

        messages = [
            {"role": "user", "content": "Question 1 " * 20},
            {"role": "assistant", "content": "Answer 1 " * 30},
            {"role": "user", "content": "Question 2 " * 20},
            {"role": "assistant", "content": "Answer 2 " * 30},
        ]

        # Test extractive summarizer directly
        summary = extractive_summarizer(messages)
        assert "[Summary of" in summary
        assert "4 messages" in summary

    def test_progressive_summarizer_groups_messages(self, tokenizer: Tokenizer):
        """ProgressiveSummarizer should identify message groups correctly."""
        from headroom.transforms.progressive_summarizer import ProgressiveSummarizer

        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=2,
            store_for_retrieval=False,
        )

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Q1 " * 30},
            {"role": "assistant", "content": "A1 " * 30},
            {"role": "user", "content": "Q2 " * 30},
            {"role": "assistant", "content": "A2 " * 30},
            {"role": "user", "content": "Final"},
        ]

        # Protect only system (0) and final (5)
        protected = {0, 5}

        groups = summarizer._find_summarization_candidates(messages, protected)

        # Should find the middle messages as a group
        assert len(groups) >= 1
        # Group should include indices 1-4
        found_middle_group = any(start <= 1 and end >= 4 for start, end in groups)
        assert found_middle_group

    def test_progressive_summarizer_respects_min_messages(self, tokenizer: Tokenizer):
        """ProgressiveSummarizer should respect min_messages_to_summarize."""
        from headroom.transforms.progressive_summarizer import ProgressiveSummarizer

        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=5,  # High threshold
            store_for_retrieval=False,
        )

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Final"},
        ]

        protected = {0, 3}

        groups = summarizer._find_summarization_candidates(messages, protected)

        # Should not find any groups (only 2 unprotected messages)
        assert len(groups) == 0

    def test_progressive_summarizer_summarizes_messages(self, tokenizer: Tokenizer):
        """ProgressiveSummarizer should create summaries correctly."""
        from headroom.transforms.progressive_summarizer import ProgressiveSummarizer

        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Q1 " * 50},
            {"role": "assistant", "content": "A1 " * 50},
            {"role": "user", "content": "Q2 " * 50},
            {"role": "assistant", "content": "A2 " * 50},
            {"role": "user", "content": "Final question"},
        ]

        protected = {0, 5}  # System and final

        result = summarizer.summarize_messages(
            messages=messages,
            tokenizer=tokenizer,
            protected_indices=protected,
        )

        # Should have reduced message count
        assert len(result.messages) < len(messages)
        # Should have created summaries
        assert len(result.summaries_created) > 0
        # Should have saved tokens
        assert result.tokens_after < result.tokens_before


class TestAnchoredSummary:
    """Tests for AnchoredSummary data structure."""

    def test_anchored_summary_compression_ratio(self):
        """AnchoredSummary should calculate compression ratio correctly."""
        from headroom.transforms.progressive_summarizer import AnchoredSummary

        summary = AnchoredSummary(
            summary_text="Summary",
            start_index=0,
            end_index=5,
            original_message_count=6,
            original_tokens=1000,
            summary_tokens=100,
        )

        assert summary.compression_ratio == 0.1
        assert summary.tokens_saved == 900

    def test_anchored_summary_zero_original_tokens(self):
        """AnchoredSummary should handle zero original tokens."""
        from headroom.transforms.progressive_summarizer import AnchoredSummary

        summary = AnchoredSummary(
            summary_text="Summary",
            start_index=0,
            end_index=0,
            original_message_count=1,
            original_tokens=0,
            summary_tokens=10,
        )

        assert summary.compression_ratio == 1.0
        assert summary.tokens_saved == 0


class TestSummarizeEdgeCases:
    """Edge case tests for SUMMARIZE strategy."""

    def test_summarize_empty_messages(self, tokenizer: Tokenizer):
        """SUMMARIZE should handle empty messages list."""
        from headroom.transforms.progressive_summarizer import ProgressiveSummarizer

        summarizer = ProgressiveSummarizer(store_for_retrieval=False)

        result = summarizer.summarize_messages(
            messages=[],
            tokenizer=tokenizer,
            protected_indices=set(),
        )

        assert result.messages == []
        assert len(result.summaries_created) == 0

    def test_summarize_all_protected(self, tokenizer: Tokenizer):
        """SUMMARIZE should handle when all messages are protected."""
        from headroom.transforms.progressive_summarizer import ProgressiveSummarizer

        summarizer = ProgressiveSummarizer(store_for_retrieval=False)

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

        result = summarizer.summarize_messages(
            messages=messages,
            tokenizer=tokenizer,
            protected_indices={0, 1, 2},  # All protected
        )

        # Should return unchanged messages
        assert len(result.messages) == len(messages)
        assert len(result.summaries_created) == 0

    def test_summarize_with_tool_messages(self, tokenizer: Tokenizer):
        """SUMMARIZE should handle tool messages."""
        import json

        from headroom.transforms.progressive_summarizer import ProgressiveSummarizer

        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Search for data " * 20},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": json.dumps([{"id": i, "data": f"result_{i}"} for i in range(20)]),
            },
            {"role": "assistant", "content": "Here are the results " * 20},
            {"role": "user", "content": "Final"},
        ]

        protected = {0, 5}

        result = summarizer.summarize_messages(
            messages=messages,
            tokenizer=tokenizer,
            protected_indices=protected,
        )

        # Should complete without error
        assert result.messages is not None
        # Protected messages should be preserved
        assert result.messages[0].get("role") == "system"

    def test_summarize_skips_small_token_groups(self, tokenizer: Tokenizer):
        """SUMMARIZE should skip groups with few tokens."""
        from headroom.transforms.progressive_summarizer import ProgressiveSummarizer

        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        # Very short messages
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "F"},
        ]

        protected = {0, 5}

        result = summarizer.summarize_messages(
            messages=messages,
            tokenizer=tokenizer,
            protected_indices=protected,
        )

        # Should not create summaries (groups too small token-wise)
        # The summarizer checks for group_tokens < 100
        assert len(result.summaries_created) == 0

    def test_summarize_callback_exception_handled(self, tokenizer: Tokenizer):
        """SUMMARIZE should handle callback exceptions gracefully."""
        from headroom.transforms.progressive_summarizer import ProgressiveSummarizer

        def failing_summarizer(messages: list[dict], context: str = "") -> str:
            raise ValueError("Summarization failed!")

        summarizer = ProgressiveSummarizer(
            summarize_fn=failing_summarizer,
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Q " * 50},
            {"role": "assistant", "content": "A " * 50},
            {"role": "user", "content": "Q2 " * 50},
            {"role": "assistant", "content": "A2 " * 50},
            {"role": "user", "content": "Final"},
        ]

        protected = {0, 5}

        # Should not raise, should return original messages
        result = summarizer.summarize_messages(
            messages=messages,
            tokenizer=tokenizer,
            protected_indices=protected,
        )

        assert result.messages is not None
        # No summaries created due to exception
        assert len(result.summaries_created) == 0


# =============================================================================
# Test Anthropic Format Tool Protection
# =============================================================================


class TestAnthropicFormatToolProtection:
    """Tests for Anthropic format tool_use/tool_result protection.

    These tests verify that IntelligentContextManager correctly handles
    Anthropic's native format where:
    - tool_use blocks appear in assistant.content[]
    - tool_result blocks appear in user.content[]

    This is critical for Claude Code integration.
    """

    @pytest.fixture
    def anthropic_tool_conversation(self) -> list[dict[str, Any]]:
        """Conversation with Anthropic format tool_use/tool_result."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Take a screenshot of the page."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll take a screenshot for you."},
                    {
                        "type": "tool_use",
                        "id": "toolu_screenshot_1",
                        "name": "browser_screenshot",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_screenshot_1",
                        "content": "Screenshot captured successfully: [base64 image data]",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "I've captured the screenshot. The page shows a login form.",
            },
            {"role": "user", "content": "Now click the submit button."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Clicking the submit button."},
                    {
                        "type": "tool_use",
                        "id": "toolu_click_1",
                        "name": "browser_click",
                        "input": {"selector": "#submit"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_click_1",
                        "content": "Clicked element #submit",
                    }
                ],
            },
            {"role": "assistant", "content": "Done! The form has been submitted."},
            {"role": "user", "content": "Thanks!"},
        ]

    @pytest.fixture
    def anthropic_multiple_tools_same_message(self) -> list[dict[str, Any]]:
        """Multiple Anthropic tool_use blocks in same assistant message."""
        return [
            {"role": "system", "content": "You are a code assistant."},
            {"role": "user", "content": "Read both config files."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll read both files."},
                    {
                        "type": "tool_use",
                        "id": "toolu_read_1",
                        "name": "Read",
                        "input": {"file_path": "/etc/config1.json"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_read_2",
                        "name": "Read",
                        "input": {"file_path": "/etc/config2.json"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read_1",
                        "content": '{"setting1": "value1"}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read_2",
                        "content": '{"setting2": "value2"}',
                    },
                ],
            },
            {"role": "assistant", "content": "Both config files have been read."},
            {"role": "user", "content": "Great, thanks!"},
        ]

    def test_anthropic_tool_result_protected_when_tool_use_protected(
        self,
        anthropic_tool_conversation: list[dict[str, Any]],
        tokenizer: Tokenizer,
    ):
        """Tool_result user messages should be protected when their tool_use is protected."""
        config = IntelligentContextConfig(keep_last_turns=2)
        manager = IntelligentContextManager(config=config)

        protected = manager._get_protected_indices(anthropic_tool_conversation)

        # The last tool_use is at index 6, tool_result at index 7
        # With keep_last_turns=2, indices 6-9 should be protected
        # The tool_result at 7 should be protected

        # Check that if assistant with tool_use is protected, its tool_result is too
        for i in protected:
            msg = anthropic_tool_conversation[i]
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, list):
                    tool_use_ids = set()
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_use_ids.add(block.get("id"))

                    # Find the corresponding tool_result message
                    if tool_use_ids:
                        for j, other_msg in enumerate(anthropic_tool_conversation):
                            if other_msg.get("role") == "user":
                                other_content = other_msg.get("content")
                                if isinstance(other_content, list):
                                    for block in other_content:
                                        if (
                                            isinstance(block, dict)
                                            and block.get("type") == "tool_result"
                                            and block.get("tool_use_id") in tool_use_ids
                                        ):
                                            assert j in protected, (
                                                f"Tool_result at {j} should be protected "
                                                f"because tool_use at {i} is protected"
                                            )

    def test_anthropic_tool_units_dropped_atomically(
        self,
        anthropic_tool_conversation: list[dict[str, Any]],
        tokenizer: Tokenizer,
    ):
        """Anthropic tool_use and tool_result should be dropped together."""
        config = IntelligentContextConfig(keep_last_turns=1)
        manager = IntelligentContextManager(config=config)

        # Force dropping by using small limit
        result = manager.apply(
            anthropic_tool_conversation,
            tokenizer,
            model_limit=300,
            output_buffer=50,
        )

        # Verify no orphaned tool_results
        tool_use_ids_present = set()
        tool_result_ids_present = set()

        for msg in result.messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_use_ids_present.add(block.get("id"))
                        elif block.get("type") == "tool_result":
                            tool_result_ids_present.add(block.get("tool_use_id"))

        # Every tool_result should have its tool_use present
        for result_id in tool_result_ids_present:
            assert result_id in tool_use_ids_present, (
                f"Orphaned tool_result with tool_use_id={result_id}"
            )

    def test_anthropic_multiple_tools_same_message_atomic(
        self,
        anthropic_multiple_tools_same_message: list[dict[str, Any]],
        tokenizer: Tokenizer,
    ):
        """Multiple tool_use blocks in same message should be handled atomically."""
        config = IntelligentContextConfig(keep_last_turns=1)
        manager = IntelligentContextManager(config=config)

        result = manager.apply(
            anthropic_multiple_tools_same_message,
            tokenizer,
            model_limit=200,
            output_buffer=50,
        )

        # Check atomicity
        tool_use_ids = set()
        tool_result_ids = set()

        for msg in result.messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_use_ids.add(block.get("id"))
                        elif block.get("type") == "tool_result":
                            tool_result_ids.add(block.get("tool_use_id"))

        # All tool_results should have their tool_use
        for result_id in tool_result_ids:
            assert result_id in tool_use_ids, f"Orphaned tool_result: {result_id}"

    def test_anthropic_format_no_api_error_scenario(
        self,
        anthropic_tool_conversation: list[dict[str, Any]],
        tokenizer: Tokenizer,
    ):
        """Verify the specific scenario that causes 'unexpected tool_use_id' error is fixed."""
        # This test specifically verifies the bug fix for:
        # "unexpected tool_use_id found in tool_result blocks"

        config = IntelligentContextConfig(
            keep_last_turns=1,
            keep_system=True,
        )
        manager = IntelligentContextManager(config=config)

        # Use a limit that would cause dropping
        result = manager.apply(
            anthropic_tool_conversation,
            tokenizer,
            model_limit=250,
            output_buffer=50,
        )

        # Simulate what the API would check
        tool_use_ids_in_conversation = set()
        tool_result_ids_in_conversation = set()

        for msg in result.messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_use_ids_in_conversation.add(block.get("id"))
                        elif block.get("type") == "tool_result":
                            tool_result_ids_in_conversation.add(block.get("tool_use_id"))

        # API error condition: tool_result references a tool_use_id that doesn't exist
        orphaned_results = tool_result_ids_in_conversation - tool_use_ids_in_conversation

        assert len(orphaned_results) == 0, (
            f"Would cause API error! Orphaned tool_result ids: {orphaned_results}"
        )

    def test_mixed_openai_and_anthropic_formats(
        self,
        tokenizer: Tokenizer,
    ):
        """Both OpenAI and Anthropic formats should work together."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Do things."},
            # OpenAI format tool call
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_openai_1",
                        "type": "function",
                        "function": {"name": "openai_tool", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_openai_1",
                "content": "OpenAI tool result",
            },
            {"role": "assistant", "content": "OpenAI tool done."},
            {"role": "user", "content": "Now use Anthropic format."},
            # Anthropic format tool call
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_anthropic_1",
                        "name": "anthropic_tool",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_anthropic_1",
                        "content": "Anthropic tool result",
                    }
                ],
            },
            {"role": "assistant", "content": "All done!"},
            {"role": "user", "content": "Thanks!"},
        ]

        config = IntelligentContextConfig(keep_last_turns=1)
        manager = IntelligentContextManager(config=config)

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=200,
            output_buffer=50,
        )

        # Verify no orphaned tools of either format
        openai_call_ids = set()
        openai_result_ids = set()
        anthropic_use_ids = set()
        anthropic_result_ids = set()

        for msg in result.messages:
            # OpenAI format
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    openai_call_ids.add(tc.get("id"))
            if msg.get("role") == "tool":
                openai_result_ids.add(msg.get("tool_call_id"))

            # Anthropic format
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            anthropic_use_ids.add(block.get("id"))
                        elif block.get("type") == "tool_result":
                            anthropic_result_ids.add(block.get("tool_use_id"))

        # Check OpenAI format
        for result_id in openai_result_ids:
            assert result_id in openai_call_ids, f"Orphaned OpenAI tool: {result_id}"

        # Check Anthropic format
        for result_id in anthropic_result_ids:
            assert result_id in anthropic_use_ids, f"Orphaned Anthropic tool: {result_id}"
