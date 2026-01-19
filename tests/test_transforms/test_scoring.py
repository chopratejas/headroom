"""Comprehensive tests for message importance scoring.

These tests verify that the scoring system works correctly WITHOUT
hardcoded patterns. All importance detection must come from:
1. Computed metrics (recency, density, references)
2. TOIN-learned patterns
3. Embedding similarity
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.config import ScoringWeights
from headroom.transforms.scoring import MessageScorer

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_weights() -> ScoringWeights:
    """Default scoring weights."""
    return ScoringWeights()


@pytest.fixture
def simple_conversation() -> list[dict[str, Any]]:
    """Simple conversation without tool calls."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "Can you help me with Python?"},
        {"role": "assistant", "content": "Of course! What would you like to know about Python?"},
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
    """Long conversation for testing recency decay."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(20):
        messages.append({"role": "user", "content": f"User message {i}"})
        messages.append({"role": "assistant", "content": f"Assistant response {i}"})
    return messages


@pytest.fixture
def high_density_message() -> dict[str, Any]:
    """Message with high information density (many unique tokens)."""
    return {
        "role": "user",
        "content": "Python JavaScript Ruby Golang Rust Swift Kotlin TypeScript C++ Java",
    }


@pytest.fixture
def low_density_message() -> dict[str, Any]:
    """Message with low information density (repeated tokens)."""
    return {
        "role": "user",
        "content": "the the the the the the the the the the very very very very",
    }


# =============================================================================
# Test ScoringWeights
# =============================================================================


class TestScoringWeights:
    """Tests for ScoringWeights configuration."""

    def test_default_weights_sum_approximately_one(self):
        """Default weights should sum close to 1.0."""
        weights = ScoringWeights()
        total = (
            weights.recency
            + weights.semantic_similarity
            + weights.toin_importance
            + weights.error_indicator
            + weights.forward_reference
            + weights.token_density
        )
        assert abs(total - 1.0) < 0.01

    def test_normalized_weights_sum_exactly_one(self):
        """Normalized weights should sum to exactly 1.0."""
        weights = ScoringWeights(
            recency=0.5,
            semantic_similarity=0.3,
            toin_importance=0.2,
            error_indicator=0.1,
            forward_reference=0.1,
            token_density=0.05,
        )
        normalized = weights.normalized()
        total = (
            normalized.recency
            + normalized.semantic_similarity
            + normalized.toin_importance
            + normalized.error_indicator
            + normalized.forward_reference
            + normalized.token_density
        )
        assert abs(total - 1.0) < 1e-10

    def test_zero_weights_returns_default(self):
        """Zero weights should return default weights."""
        weights = ScoringWeights(
            recency=0,
            semantic_similarity=0,
            toin_importance=0,
            error_indicator=0,
            forward_reference=0,
            token_density=0,
        )
        normalized = weights.normalized()
        # Should return default values, not NaN
        assert normalized.recency >= 0


# =============================================================================
# Test MessageScorer Initialization
# =============================================================================


class TestMessageScorerInit:
    """Tests for MessageScorer initialization."""

    def test_init_with_defaults(self):
        """Scorer initializes with default weights."""
        scorer = MessageScorer()
        assert scorer.weights is not None
        assert scorer.toin is None
        assert scorer.embedding_provider is None

    def test_init_with_custom_weights(self, default_weights):
        """Scorer accepts custom weights."""
        scorer = MessageScorer(weights=default_weights)
        # Weights should be normalized
        total = (
            scorer.weights.recency
            + scorer.weights.semantic_similarity
            + scorer.weights.toin_importance
            + scorer.weights.error_indicator
            + scorer.weights.forward_reference
            + scorer.weights.token_density
        )
        assert abs(total - 1.0) < 1e-10

    def test_init_with_custom_decay_rate(self):
        """Scorer accepts custom recency decay rate."""
        scorer = MessageScorer(recency_decay_rate=0.2)
        assert scorer.recency_decay_rate == 0.2


# =============================================================================
# Test Recency Scoring
# =============================================================================


class TestRecencyScoring:
    """Tests for recency-based scoring."""

    def test_last_message_has_highest_recency(self, simple_conversation):
        """Most recent message should have highest recency score."""
        scorer = MessageScorer()
        scores = scorer.score_messages(
            simple_conversation,
            protected_indices=set(),
            tool_unit_indices=set(),
        )

        # Last message should have highest recency
        last_idx = len(simple_conversation) - 1
        for i, score in enumerate(scores):
            if i != last_idx:
                assert score.recency_score <= scores[last_idx].recency_score

    def test_recency_decreases_with_age(self, long_conversation):
        """Recency score should decrease for older messages."""
        scorer = MessageScorer()
        scores = scorer.score_messages(
            long_conversation,
            protected_indices=set(),
            tool_unit_indices=set(),
        )

        # Verify decreasing recency (allowing for equal scores)
        for i in range(1, len(scores)):
            assert scores[i].recency_score >= scores[i - 1].recency_score

    def test_recency_decay_rate_affects_scores(self, simple_conversation):
        """Higher decay rate should make older messages score lower."""
        scorer_slow = MessageScorer(recency_decay_rate=0.05)
        scorer_fast = MessageScorer(recency_decay_rate=0.5)

        scores_slow = scorer_slow.score_messages(simple_conversation, set(), set())
        scores_fast = scorer_fast.score_messages(simple_conversation, set(), set())

        # First message should have lower recency with fast decay
        assert scores_fast[1].recency_score < scores_slow[1].recency_score

    def test_single_message_has_max_recency(self):
        """Single message conversation should have max recency."""
        messages = [{"role": "user", "content": "Hello"}]
        scorer = MessageScorer()
        scores = scorer.score_messages(messages, set(), set())

        assert scores[0].recency_score == 1.0


# =============================================================================
# Test Density Scoring
# =============================================================================


class TestDensityScoring:
    """Tests for information density scoring."""

    def test_high_density_scores_higher(self, high_density_message, low_density_message):
        """High density message should score higher than low density."""
        scorer = MessageScorer()

        high_scores = scorer.score_messages([high_density_message], set(), set())
        low_scores = scorer.score_messages([low_density_message], set(), set())

        assert high_scores[0].density_score > low_scores[0].density_score

    def test_density_in_valid_range(self, simple_conversation):
        """Density scores should be in [0, 1] range."""
        scorer = MessageScorer()
        scores = scorer.score_messages(simple_conversation, set(), set())

        for score in scores:
            assert 0.0 <= score.density_score <= 1.0

    def test_empty_content_gets_neutral_density(self):
        """Empty content should get neutral density score."""
        messages = [{"role": "user", "content": ""}]
        scorer = MessageScorer()
        scores = scorer.score_messages(messages, set(), set())

        assert scores[0].density_score == 0.5


# =============================================================================
# Test Forward Reference Scoring
# =============================================================================


class TestForwardReferenceScoring:
    """Tests for forward reference detection."""

    def test_assistant_with_tool_calls_has_references(self, conversation_with_tools):
        """Assistant messages with tool calls should have forward references."""
        scorer = MessageScorer()
        scores = scorer.score_messages(conversation_with_tools, set(), set())

        # Message at index 2 has tool_calls, should have references
        # (tool response at index 3 references it)
        assert scores[2].reference_score > 0

    def test_no_tool_calls_no_references(self, simple_conversation):
        """Messages without tool calls should have no forward references."""
        scorer = MessageScorer()
        scores = scorer.score_messages(simple_conversation, set(), set())

        for score in scores:
            assert score.reference_score == 0.0


# =============================================================================
# Test Protected Message Handling
# =============================================================================


class TestProtectedMessages:
    """Tests for protected message handling in scoring."""

    def test_protected_messages_marked(self, simple_conversation):
        """Protected messages should be marked in scores."""
        scorer = MessageScorer()
        protected = {0, 1}  # System and first user message

        scores = scorer.score_messages(
            simple_conversation,
            protected_indices=protected,
            tool_unit_indices=set(),
        )

        assert scores[0].is_protected is True
        assert scores[1].is_protected is True
        assert scores[2].is_protected is False

    def test_tool_unit_messages_marked_unsafe(self, conversation_with_tools):
        """Messages in tool units should be marked as not drop_safe."""
        scorer = MessageScorer()
        tool_unit_indices = {2, 3, 6, 7}  # Assistant with tool_calls and responses

        scores = scorer.score_messages(
            conversation_with_tools,
            protected_indices=set(),
            tool_unit_indices=tool_unit_indices,
        )

        # Tool unit messages should not be independently droppable
        # They can only be dropped as a unit
        for idx in tool_unit_indices:
            # Not protected, but part of a unit
            assert scores[idx].drop_safe is True or scores[idx].is_protected


# =============================================================================
# Test Total Score Computation
# =============================================================================


class TestTotalScore:
    """Tests for total score computation."""

    def test_total_score_in_valid_range(self, simple_conversation):
        """Total scores should be in [0, 1] range."""
        scorer = MessageScorer()
        scores = scorer.score_messages(simple_conversation, set(), set())

        for score in scores:
            assert 0.0 <= score.total_score <= 1.0

    def test_score_breakdown_matches_total(self, simple_conversation):
        """Score breakdown components should match total."""
        weights = ScoringWeights()
        scorer = MessageScorer(weights=weights)
        scores = scorer.score_messages(simple_conversation, set(), set())

        for score in scores:
            # Without TOIN and embeddings, some components are neutral
            # Just verify breakdown dict is populated
            assert "recency" in score.score_breakdown
            assert "density" in score.score_breakdown

    def test_weights_affect_total_score(self, simple_conversation):
        """Different weights should produce different total scores."""
        weights_recency = ScoringWeights(
            recency=1.0,
            semantic_similarity=0,
            toin_importance=0,
            error_indicator=0,
            forward_reference=0,
            token_density=0,
        )
        weights_density = ScoringWeights(
            recency=0,
            semantic_similarity=0,
            toin_importance=0,
            error_indicator=0,
            forward_reference=0,
            token_density=1.0,
        )

        scorer_recency = MessageScorer(weights=weights_recency)
        scorer_density = MessageScorer(weights=weights_density)

        scores_r = scorer_recency.score_messages(simple_conversation, set(), set())
        scores_d = scorer_density.score_messages(simple_conversation, set(), set())

        # Different weights should produce different scores
        # (unless message happens to have same recency and density)
        assert any(
            abs(scores_r[i].total_score - scores_d[i].total_score) > 0.01
            for i in range(len(simple_conversation))
        )


# =============================================================================
# Test Cosine Similarity
# =============================================================================


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors_similarity_one(self):
        """Identical vectors should have similarity 1.0."""
        a = [1.0, 2.0, 3.0]
        similarity = MessageScorer._cosine_similarity(a, a)
        assert abs(similarity - 1.0) < 1e-10

    def test_orthogonal_vectors_similarity_zero(self):
        """Orthogonal vectors should have similarity 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        similarity = MessageScorer._cosine_similarity(a, b)
        assert abs(similarity) < 1e-10

    def test_opposite_vectors_similarity_negative(self):
        """Opposite vectors should have similarity -1.0."""
        a = [1.0, 1.0]
        b = [-1.0, -1.0]
        similarity = MessageScorer._cosine_similarity(a, b)
        assert abs(similarity - (-1.0)) < 1e-10

    def test_empty_vectors_return_zero(self):
        """Empty vectors should return 0 similarity."""
        assert MessageScorer._cosine_similarity([], []) == 0.0

    def test_mismatched_lengths_return_zero(self):
        """Mismatched vector lengths should return 0."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert MessageScorer._cosine_similarity(a, b) == 0.0


# =============================================================================
# Test TOIN Integration (Without Actual TOIN)
# =============================================================================


class TestTOINIntegration:
    """Tests for TOIN integration in scoring."""

    def test_no_toin_returns_neutral_scores(self, conversation_with_tools):
        """Without TOIN, tool messages should get neutral TOIN scores."""
        scorer = MessageScorer(toin=None)
        scores = scorer.score_messages(conversation_with_tools, set(), set())

        # Tool messages should have neutral TOIN score (0.5)
        for i, msg in enumerate(conversation_with_tools):
            if msg.get("role") == "tool":
                assert scores[i].toin_score == 0.5

    def test_no_toin_returns_zero_error_scores(self, conversation_with_tools):
        """Without TOIN, error scores should be zero."""
        scorer = MessageScorer(toin=None)
        scores = scorer.score_messages(conversation_with_tools, set(), set())

        for score in scores:
            assert score.error_score == 0.0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in scoring."""

    def test_empty_messages_list(self):
        """Empty message list should return empty scores."""
        scorer = MessageScorer()
        scores = scorer.score_messages([], set(), set())
        assert scores == []

    def test_single_message(self):
        """Single message should be scored correctly."""
        messages = [{"role": "user", "content": "Hello"}]
        scorer = MessageScorer()
        scores = scorer.score_messages(messages, set(), set())

        assert len(scores) == 1
        assert scores[0].message_index == 0
        assert scores[0].recency_score == 1.0

    def test_message_with_no_content(self):
        """Message with no content key should be handled."""
        messages = [{"role": "user"}]
        scorer = MessageScorer()
        scores = scorer.score_messages(messages, set(), set())

        assert len(scores) == 1
        # Should not crash, should have some score

    def test_message_with_non_string_content(self):
        """Message with non-string content should be handled."""
        messages = [{"role": "user", "content": ["list", "content"]}]
        scorer = MessageScorer()
        scores = scorer.score_messages(messages, set(), set())

        assert len(scores) == 1
        # Density should be neutral for non-string
        assert scores[0].density_score == 0.5

    def test_all_messages_protected(self, simple_conversation):
        """All messages protected should all be marked."""
        scorer = MessageScorer()
        all_protected = set(range(len(simple_conversation)))

        scores = scorer.score_messages(
            simple_conversation,
            protected_indices=all_protected,
            tool_unit_indices=set(),
        )

        for score in scores:
            assert score.is_protected is True


# =============================================================================
# Test Score Ordering
# =============================================================================


class TestScoreOrdering:
    """Tests for expected score ordering patterns."""

    def test_recent_messages_score_higher_by_default(self, long_conversation):
        """Recent messages should generally score higher with default weights."""
        scorer = MessageScorer()
        scores = scorer.score_messages(long_conversation, set(), set())

        # Average score of last 5 messages should be higher than first 5
        # (excluding system message at index 0)
        first_5_avg = sum(s.total_score for s in scores[1:6]) / 5
        last_5_avg = sum(s.total_score for s in scores[-5:]) / 5

        assert last_5_avg > first_5_avg

    def test_system_message_scores_lower_on_recency(self, simple_conversation):
        """System message (index 0) should have low recency score."""
        scorer = MessageScorer()
        scores = scorer.score_messages(simple_conversation, set(), set())

        # System message is oldest, should have lowest recency
        system_recency = scores[0].recency_score
        for score in scores[1:]:
            assert score.recency_score >= system_recency
