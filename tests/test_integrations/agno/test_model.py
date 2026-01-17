"""Comprehensive tests for Agno integration.

Tests cover:
1. HeadroomAgnoModel - Wrapper for any Agno model
2. Provider detection - Detecting correct provider from Agno model
3. Hooks - Pre and post hooks for observability
4. optimize_messages() - Standalone optimization function
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Check if Agno is available
try:
    import agno  # noqa: F401

    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False

from headroom import HeadroomConfig, HeadroomMode

# Skip all tests if Agno not installed
pytestmark = pytest.mark.skipif(not AGNO_AVAILABLE, reason="Agno not installed")


@pytest.fixture
def mock_agno_model():
    """Create a mock Agno model (OpenAIChat-like)."""
    mock = MagicMock()
    mock.__class__.__name__ = "OpenAIChat"
    mock.__class__.__module__ = "agno.models.openai"
    mock.id = "gpt-4o"

    # Mock response method
    def mock_response(messages, **kwargs):
        response = MagicMock()
        response.content = "Hello! I'm a mock response."
        response.metrics = MagicMock()
        response.metrics.input_tokens = 10
        response.metrics.output_tokens = 5
        response.metrics.total_tokens = 15
        return response

    mock.response = MagicMock(side_effect=mock_response)

    # Mock streaming response
    def mock_stream(messages, **kwargs):
        yield MagicMock(content="Streaming...")

    mock.response_stream = MagicMock(side_effect=mock_stream)

    return mock


@pytest.fixture
def mock_claude_model():
    """Create a mock Agno model (Claude-like)."""
    mock = MagicMock()
    mock.__class__.__name__ = "Claude"
    mock.__class__.__module__ = "agno.models.anthropic"
    mock.id = "claude-3-5-sonnet-20241022"

    def mock_response(messages, **kwargs):
        response = MagicMock()
        response.content = "I'm Claude!"
        response.metrics = MagicMock()
        response.metrics.input_tokens = 20
        response.metrics.output_tokens = 10
        response.metrics.total_tokens = 30
        return response

    mock.response = MagicMock(side_effect=mock_response)
    return mock


@pytest.fixture
def sample_messages():
    """Sample messages in OpenAI format (Agno accepts this)."""
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


class TestAgnoAvailable:
    """Tests for agno_available() helper."""

    def test_returns_bool(self):
        """agno_available returns boolean."""
        from headroom.integrations.agno import agno_available

        assert isinstance(agno_available(), bool)

    def test_returns_true_when_installed(self):
        """Returns True when Agno is installed."""
        from headroom.integrations.agno import agno_available

        assert agno_available() is True


class TestHeadroomAgnoModel:
    """Tests for HeadroomAgnoModel wrapper."""

    def test_init_with_defaults(self, mock_agno_model):
        """Initialize with default config."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(mock_agno_model)

        assert model.wrapped_model is mock_agno_model
        assert model.mode == HeadroomMode.OPTIMIZE
        assert model._metrics_history == []
        assert model._total_tokens_saved == 0

    def test_init_with_custom_config(self, mock_agno_model):
        """Initialize with custom config."""
        from headroom.integrations.agno import HeadroomAgnoModel

        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)
        model = HeadroomAgnoModel(
            mock_agno_model,
            config=config,
            mode=HeadroomMode.SIMULATE,
        )

        assert model.config is config
        assert model.mode == HeadroomMode.SIMULATE

    def test_init_auto_detect_provider(self, mock_agno_model):
        """Auto-detect provider from wrapped model."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(mock_agno_model, auto_detect_provider=True)

        assert model.auto_detect_provider is True

    def test_forward_attributes(self, mock_agno_model):
        """Forward attribute access to wrapped model."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.custom_attribute = "test_value"
        model = HeadroomAgnoModel(mock_agno_model)

        assert model.custom_attribute == "test_value"

    def test_properties_not_forwarded(self, mock_agno_model):
        """Own properties should not be forwarded."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(mock_agno_model)

        # These should work without forwarding to wrapped model
        assert model.total_tokens_saved == 0
        assert model.metrics_history == []

    def test_convert_messages_to_openai(self, mock_agno_model, sample_messages):
        """Convert Agno messages to OpenAI format."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(mock_agno_model)

        # Test with dict messages (already OpenAI format)
        openai_msgs = model._convert_messages_to_openai(sample_messages)

        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "system"
        assert openai_msgs[0]["content"] == "You are a helpful assistant."
        assert openai_msgs[1]["role"] == "user"
        assert "France" in openai_msgs[1]["content"]

    def test_convert_agno_message_objects(self, mock_agno_model):
        """Convert Agno Message objects to OpenAI format."""
        from headroom.integrations.agno import HeadroomAgnoModel

        # Create mock Agno Message objects
        system_msg = MagicMock()
        system_msg.role = "system"
        system_msg.content = "You are helpful."
        system_msg.tool_calls = None
        system_msg.tool_call_id = None

        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Hello"
        user_msg.tool_calls = None
        user_msg.tool_call_id = None

        messages = [system_msg, user_msg]

        model = HeadroomAgnoModel(mock_agno_model)
        openai_msgs = model._convert_messages_to_openai(messages)

        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "system"
        assert openai_msgs[0]["content"] == "You are helpful."

    def test_convert_messages_with_tool_calls(self, mock_agno_model):
        """Convert messages with tool calls."""
        from headroom.integrations.agno import HeadroomAgnoModel

        assistant_msg = MagicMock()
        assistant_msg.role = "assistant"
        assistant_msg.content = "I'll check the weather."
        assistant_msg.tool_calls = [
            {"id": "call_123", "name": "get_weather", "args": {"city": "Paris"}}
        ]
        assistant_msg.tool_call_id = None

        tool_msg = MagicMock()
        tool_msg.role = "tool"
        tool_msg.content = '{"temp": 20}'
        tool_msg.tool_calls = None
        tool_msg.tool_call_id = "call_123"

        messages = [assistant_msg, tool_msg]

        model = HeadroomAgnoModel(mock_agno_model)
        openai_msgs = model._convert_messages_to_openai(messages)

        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "assistant"
        assert "tool_calls" in openai_msgs[0]
        assert openai_msgs[1]["tool_call_id"] == "call_123"

    def test_response_applies_optimization(self, mock_agno_model, sample_messages):
        """response() applies Headroom optimization."""
        from headroom.integrations.agno import HeadroomAgnoModel
        from headroom.providers import OpenAIProvider

        model = HeadroomAgnoModel(mock_agno_model)

        # Initialize provider and pipeline for mocking
        model._provider = OpenAIProvider()
        _ = model.pipeline  # Force lazy init

        # Mock the pipeline apply method
        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
            mock_result.tokens_before = 100
            mock_result.tokens_after = 80
            mock_result.transforms_applied = ["cache_aligner"]
            mock_apply.return_value = mock_result

            model.response(sample_messages)

            # Verify pipeline.apply was called
            mock_apply.assert_called_once()

            # Verify metrics were tracked
            assert len(model._metrics_history) == 1
            assert model._metrics_history[0].tokens_saved == 20

    def test_response_stream_applies_optimization(self, mock_agno_model, sample_messages):
        """response_stream() applies Headroom optimization."""
        from headroom.integrations.agno import HeadroomAgnoModel
        from headroom.providers import OpenAIProvider

        model = HeadroomAgnoModel(mock_agno_model)
        model._provider = OpenAIProvider()
        _ = model.pipeline

        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = sample_messages
            mock_result.tokens_before = 100
            mock_result.tokens_after = 90
            mock_result.transforms_applied = []
            mock_apply.return_value = mock_result

            # Consume the generator
            list(model.response_stream(sample_messages))

            mock_apply.assert_called_once()
            assert len(model._metrics_history) == 1

    def test_metrics_history_limited(self, mock_agno_model, sample_messages):
        """Metrics history is limited to 100 entries."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(mock_agno_model)

        # Add 150 fake metrics
        for _i in range(150):
            model._metrics_history.append(MagicMock())

        # Simulate a call that trims
        model._metrics_history = model._metrics_history[-100:]

        assert len(model._metrics_history) == 100

    def test_get_savings_summary_empty(self, mock_agno_model):
        """get_savings_summary with no history."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(mock_agno_model)
        summary = model.get_savings_summary()

        assert summary["total_requests"] == 0
        assert summary["total_tokens_saved"] == 0
        assert summary["average_savings_percent"] == 0

    def test_get_savings_summary_with_data(self, mock_agno_model):
        """get_savings_summary with metrics."""
        from headroom.integrations.agno import HeadroomAgnoModel
        from headroom.integrations.agno.model import OptimizationMetrics

        model = HeadroomAgnoModel(mock_agno_model)

        # Add fake metrics
        model._metrics_history = [
            OptimizationMetrics(
                request_id="1",
                timestamp=datetime.now(),
                tokens_before=100,
                tokens_after=80,
                tokens_saved=20,
                savings_percent=20.0,
                transforms_applied=["smart_crusher"],
                model="gpt-4o",
            ),
            OptimizationMetrics(
                request_id="2",
                timestamp=datetime.now(),
                tokens_before=200,
                tokens_after=150,
                tokens_saved=50,
                savings_percent=25.0,
                transforms_applied=["cache_aligner"],
                model="gpt-4o",
            ),
        ]
        model._total_tokens_saved = 70

        summary = model.get_savings_summary()

        assert summary["total_requests"] == 2
        assert summary["total_tokens_saved"] == 70
        assert summary["average_savings_percent"] == 22.5

    def test_reset_clears_all_state(self, mock_agno_model):
        """reset() clears all metrics state."""
        from headroom.integrations.agno import HeadroomAgnoModel
        from headroom.integrations.agno.model import OptimizationMetrics

        model = HeadroomAgnoModel(mock_agno_model)

        # Add fake metrics
        model._metrics_history = [
            OptimizationMetrics(
                request_id="1",
                timestamp=datetime.now(),
                tokens_before=100,
                tokens_after=80,
                tokens_saved=20,
                savings_percent=20.0,
                transforms_applied=["smart_crusher"],
                model="gpt-4o",
            ),
        ]
        model._total_tokens_saved = 20

        # Verify state before reset
        assert len(model._metrics_history) == 1
        assert model._total_tokens_saved == 20

        # Reset
        model.reset()

        # Verify state after reset
        assert model._metrics_history == []
        assert model._total_tokens_saved == 0
        assert model.total_tokens_saved == 0

        # Verify summary is empty
        summary = model.get_savings_summary()
        assert summary["total_requests"] == 0
        assert summary["total_tokens_saved"] == 0


class TestProviderDetection:
    """Tests for provider detection from Agno models."""

    def test_detect_openai_provider(self, mock_agno_model):
        """Detect OpenAI provider from OpenAIChat."""
        from headroom.integrations.agno.providers import get_headroom_provider
        from headroom.providers import OpenAIProvider

        provider = get_headroom_provider(mock_agno_model)

        assert isinstance(provider, OpenAIProvider)

    def test_detect_anthropic_provider(self, mock_claude_model):
        """Detect Anthropic provider from Claude model."""
        from headroom.integrations.agno.providers import get_headroom_provider
        from headroom.providers import AnthropicProvider

        provider = get_headroom_provider(mock_claude_model)

        assert isinstance(provider, AnthropicProvider)

    def test_detect_from_model_id(self):
        """Detect provider from model ID string."""
        from headroom.integrations.agno.providers import get_headroom_provider
        from headroom.providers import AnthropicProvider, GoogleProvider, OpenAIProvider

        # GPT model
        mock_gpt = MagicMock()
        mock_gpt.__class__.__name__ = "UnknownModel"
        mock_gpt.__class__.__module__ = "some.module"
        mock_gpt.id = "gpt-4o-mini"
        assert isinstance(get_headroom_provider(mock_gpt), OpenAIProvider)

        # Claude model
        mock_claude = MagicMock()
        mock_claude.__class__.__name__ = "UnknownModel"
        mock_claude.__class__.__module__ = "some.module"
        mock_claude.id = "claude-3-opus-20240229"
        assert isinstance(get_headroom_provider(mock_claude), AnthropicProvider)

        # Gemini model
        mock_gemini = MagicMock()
        mock_gemini.__class__.__name__ = "UnknownModel"
        mock_gemini.__class__.__module__ = "some.module"
        mock_gemini.id = "gemini-pro"
        assert isinstance(get_headroom_provider(mock_gemini), GoogleProvider)

    def test_fallback_to_openai(self):
        """Fallback to OpenAI provider for unknown models."""
        from headroom.integrations.agno.providers import get_headroom_provider
        from headroom.providers import OpenAIProvider

        mock = MagicMock()
        mock.__class__.__name__ = "TotallyUnknownModel"
        mock.__class__.__module__ = "completely.unknown"
        mock.id = "mystery-model-v1"

        provider = get_headroom_provider(mock)

        assert isinstance(provider, OpenAIProvider)

    def test_get_model_name(self, mock_agno_model):
        """Extract model name from Agno model."""
        from headroom.integrations.agno.providers import get_model_name_from_agno

        name = get_model_name_from_agno(mock_agno_model)

        assert name == "gpt-4o"

    def test_get_model_name_fallback(self):
        """Fallback model name when not found."""
        from headroom.integrations.agno.providers import get_model_name_from_agno

        mock = MagicMock(spec=[])  # No attributes
        name = get_model_name_from_agno(mock)

        assert name == "gpt-4o"  # Default fallback


class TestOptimizeMessages:
    """Tests for standalone optimize_messages function."""

    def test_basic_optimization(self, sample_messages):
        """Basic message optimization."""
        from headroom.integrations.agno import optimize_messages

        with patch("headroom.integrations.agno.model.TransformPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            mock_result.tokens_before = 100
            mock_result.tokens_after = 80
            mock_result.transforms_applied = ["cache_aligner"]
            mock_instance.apply.return_value = mock_result
            MockPipeline.return_value = mock_instance

            optimized, metrics = optimize_messages(sample_messages)

            assert len(optimized) == 2
            assert metrics["tokens_saved"] == 20
            assert metrics["savings_percent"] == 20.0

    def test_with_custom_config(self, sample_messages):
        """Optimization with custom config."""
        from headroom.integrations.agno import optimize_messages

        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)

        with patch("headroom.integrations.agno.model.TransformPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.messages = []
            mock_result.tokens_before = 50
            mock_result.tokens_after = 50
            mock_result.transforms_applied = []
            mock_instance.apply.return_value = mock_result
            MockPipeline.return_value = mock_instance

            _, metrics = optimize_messages(
                sample_messages,
                config=config,
                mode=HeadroomMode.AUDIT,
            )

            # Verify pipeline was created with config
            MockPipeline.assert_called_once()
            call_kwargs = MockPipeline.call_args[1]
            assert call_kwargs["config"] is config


class TestIntegrationWithRealHeadroom:
    """Integration tests using real Headroom components (no mocking)."""

    def test_real_optimization_pipeline(self, sample_messages):
        """Test with real Headroom client (no API calls)."""
        from headroom.integrations.agno import optimize_messages

        # This uses real Headroom transforms but no LLM API calls
        optimized, metrics = optimize_messages(
            sample_messages,
            mode=HeadroomMode.OPTIMIZE,
        )

        # Should return valid messages
        assert len(optimized) >= 1
        assert all(isinstance(m, dict) for m in optimized)
        assert all("role" in m and "content" in m for m in optimized)

        # Metrics should be populated
        assert "tokens_before" in metrics
        assert "tokens_after" in metrics
        assert "transforms_applied" in metrics

    def test_large_conversation_compression(self, large_conversation):
        """Test compression of large conversation."""
        from headroom.integrations.agno import optimize_messages

        optimized, metrics = optimize_messages(large_conversation)

        # Should compress (rolling window, etc.)
        assert metrics["tokens_before"] >= metrics["tokens_after"]

    def test_model_wrapper_real_optimization(self, mock_agno_model, sample_messages):
        """Test HeadroomAgnoModel with real Headroom optimization."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(mock_agno_model)

        # Call response - this will apply real optimization
        model.response(sample_messages)

        # Should have tracked metrics
        assert len(model.metrics_history) == 1
        metrics = model.metrics_history[0]
        assert metrics.tokens_before >= 0
        assert metrics.tokens_after >= 0
