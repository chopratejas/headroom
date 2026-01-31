"""Tests for Strands model integration.

Tests cover:
1. HeadroomStrandsModel - Wrapper for any Strands model
2. Message optimization pipeline
3. Metrics tracking
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Check if Strands is available
try:
    import strands  # noqa: F401

    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False

from headroom import HeadroomConfig, HeadroomMode

# Skip all tests if Strands not installed
pytestmark = pytest.mark.skipif(not STRANDS_AVAILABLE, reason="Strands not installed")


@pytest.fixture
def mock_strands_model():
    """Create a mock Strands model."""
    mock = MagicMock()
    mock.model_id = "gpt-4o-mini"
    mock.invoke = MagicMock(return_value=MagicMock(content="Hello!"))
    return mock


@pytest.fixture
def sample_messages():
    """Sample messages in OpenAI format."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


@pytest.fixture
def large_conversation():
    """Large conversation with many turns."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(50):
        messages.append({"role": "user", "content": f"Question {i}: What is {i} + {i}?"})
        messages.append({"role": "assistant", "content": f"The answer is {i + i}."})
    return messages


class TestHeadroomStrandsModel:
    """Tests for HeadroomStrandsModel wrapper."""

    def test_init_and_config(self, mock_strands_model):
        """Initialize with default and custom settings."""
        from headroom.integrations.strands import HeadroomStrandsModel

        # Default init
        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)
        assert model.wrapped_model is mock_strands_model
        assert model.total_tokens_saved == 0
        assert model.metrics_history == []

        # Custom config
        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)
        model2 = HeadroomStrandsModel(
            wrapped_model=mock_strands_model,
            config=config,
            mode=HeadroomMode.SIMULATE,
        )
        assert model2.config is config
        assert model2.mode == HeadroomMode.SIMULATE

    def test_requires_wrapped_model(self):
        """Raises ValueError if wrapped_model is None."""
        from headroom.integrations.strands import HeadroomStrandsModel

        with pytest.raises(ValueError, match="wrapped_model cannot be None"):
            HeadroomStrandsModel(wrapped_model=None)

    def test_forwards_attributes_to_wrapped_model(self, mock_strands_model):
        """Forwards unknown attributes to wrapped model."""
        from headroom.integrations.strands import HeadroomStrandsModel

        mock_strands_model.custom_attr = "test_value"
        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        assert model.custom_attr == "test_value"
        assert model.model_id == "gpt-4o-mini"

    def test_converts_messages_to_openai_format(self, mock_strands_model):
        """Converts various message formats to OpenAI format."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Dict messages (already OpenAI format)
        dict_msgs = [{"role": "user", "content": "Hello"}]
        result = model._convert_messages_to_openai(dict_msgs)
        assert result[0]["role"] == "user"

        # Message objects with tool calls
        assistant_msg = MagicMock()
        assistant_msg.role = "assistant"
        assistant_msg.content = "I'll help."
        assistant_msg.tool_calls = [{"id": "call_123", "name": "search"}]
        assistant_msg.tool_call_id = None

        result = model._convert_messages_to_openai([assistant_msg])
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]

    def test_invoke_applies_optimization(self, mock_strands_model, sample_messages):
        """invoke() applies Headroom optimization and tracks metrics."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)
        _ = model.pipeline  # Force lazy init

        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = sample_messages
            mock_result.tokens_before = 100
            mock_result.tokens_after = 80
            mock_result.transforms_applied = ["cache_aligner"]
            mock_apply.return_value = mock_result

            model.invoke(sample_messages)

            mock_apply.assert_called_once()
            assert len(model.metrics_history) == 1
            assert model.metrics_history[0].tokens_saved == 20

    def test_get_savings_summary(self, mock_strands_model):
        """get_savings_summary returns correct metrics."""
        from headroom.integrations.strands import HeadroomStrandsModel
        from headroom.integrations.strands.model import OptimizationMetrics

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Empty
        assert model.get_savings_summary()["total_requests"] == 0

        # With data
        model._metrics_history = [
            OptimizationMetrics(
                request_id="1",
                timestamp=datetime.now(),
                tokens_before=100,
                tokens_after=80,
                tokens_saved=20,
                savings_percent=20.0,
                transforms_applied=[],
                model="gpt-4o",
            ),
        ]
        model._total_tokens_saved = 20

        summary = model.get_savings_summary()
        assert summary["total_requests"] == 1
        assert summary["total_tokens_saved"] == 20

    def test_reset_clears_state(self, mock_strands_model):
        """reset() clears all tracked state."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)
        model._metrics_history = [MagicMock()]
        model._total_tokens_saved = 100

        model.reset()

        assert model._metrics_history == []
        assert model._total_tokens_saved == 0


class TestOptimizeMessages:
    """Tests for standalone optimize_messages function."""

    def test_optimizes_messages(self, sample_messages):
        """optimize_messages applies transforms and returns metrics."""
        from headroom.integrations.strands import optimize_messages

        with patch("headroom.integrations.strands.model.TransformPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.messages = sample_messages
            mock_result.tokens_before = 100
            mock_result.tokens_after = 80
            mock_result.transforms_applied = ["cache_aligner"]
            mock_instance.apply.return_value = mock_result
            MockPipeline.return_value = mock_instance

            optimized, metrics = optimize_messages(sample_messages, model="gpt-4o")

            assert len(optimized) == 2
            assert metrics["tokens_saved"] == 20
            assert metrics["savings_percent"] == 20.0


class TestRealHeadroomIntegration:
    """Integration tests with real Headroom (no mocking)."""

    def test_real_optimization_pipeline(self, sample_messages):
        """Test with real Headroom transforms (no API calls)."""
        from headroom.integrations.strands import optimize_messages

        optimized, metrics = optimize_messages(sample_messages)

        assert len(optimized) >= 1
        assert all("role" in m and "content" in m for m in optimized)
        assert "tokens_before" in metrics
        assert "tokens_after" in metrics

    def test_large_conversation_compression(self, large_conversation):
        """Large conversations get compressed."""
        from headroom.integrations.strands import optimize_messages

        optimized, metrics = optimize_messages(large_conversation)

        assert metrics["tokens_before"] >= metrics["tokens_after"]

    def test_model_wrapper_optimization(self, mock_strands_model, sample_messages):
        """HeadroomStrandsModel applies real optimization."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)
        optimized, metrics = model._optimize_messages(sample_messages)

        assert len(model.metrics_history) == 1
        assert model.metrics_history[0].tokens_before >= 0
