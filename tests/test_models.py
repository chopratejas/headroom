"""Tests for the model registry and capabilities database."""

from __future__ import annotations

import pytest

from headroom.models import (
    ModelInfo,
    ModelRegistry,
    get_model_info,
    list_models,
    register_model,
)


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_default_values(self):
        """Test default values."""
        info = ModelInfo(name="test", provider="test-provider")
        assert info.context_window == 128000
        assert info.max_output_tokens == 4096
        assert info.supports_tools is True
        assert info.supports_vision is False
        assert info.supports_streaming is True

    def test_custom_values(self):
        """Test custom values."""
        info = ModelInfo(
            name="custom-model",
            provider="custom",
            context_window=32000,
            max_output_tokens=8192,
            supports_tools=False,
            supports_vision=True,
            input_cost_per_1m=1.5,
            output_cost_per_1m=3.0,
        )
        assert info.context_window == 32000
        assert info.max_output_tokens == 8192
        assert info.supports_tools is False
        assert info.supports_vision is True
        assert info.input_cost_per_1m == 1.5

    def test_frozen(self):
        """Test that ModelInfo is frozen (immutable)."""
        info = ModelInfo(name="test", provider="test")
        with pytest.raises(AttributeError):
            info.name = "changed"


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_get_openai_model(self):
        """Test getting OpenAI model info."""
        info = ModelRegistry.get("gpt-4o")
        assert info is not None
        assert info.provider == "openai"
        assert info.context_window == 128000

    def test_get_anthropic_model(self):
        """Test getting Anthropic model info."""
        info = ModelRegistry.get("claude-3-5-sonnet-20241022")
        assert info is not None
        assert info.provider == "anthropic"
        assert info.context_window == 200000

    def test_get_google_model(self):
        """Test getting Google model info."""
        info = ModelRegistry.get("gemini-1.5-pro")
        assert info is not None
        assert info.provider == "google"
        assert info.context_window == 2000000  # 2M!

    def test_get_by_alias(self):
        """Test getting model by alias."""
        info = ModelRegistry.get("gpt-4o-2024-11-20")
        assert info is not None
        assert info.name == "gpt-4o"

    def test_get_unknown_model(self):
        """Test getting unknown model returns None."""
        info = ModelRegistry.get("unknown-model-xyz")
        assert info is None

    def test_get_prefix_matching(self):
        """Test prefix matching for versioned models."""
        info = ModelRegistry.get("gpt-4o-new-version")
        assert info is not None
        assert info.name == "gpt-4o"

    def test_register_custom_model(self):
        """Test registering custom model."""
        info = ModelRegistry.register(
            "my-custom-model",
            provider="custom",
            context_window=64000,
            supports_vision=True,
        )
        assert info.name == "my-custom-model"
        assert info.provider == "custom"
        assert info.context_window == 64000

        # Should be retrievable
        retrieved = ModelRegistry.get("my-custom-model")
        assert retrieved is not None
        assert retrieved.context_window == 64000

    def test_list_models_all(self):
        """Test listing all models."""
        models = ModelRegistry.list_models()
        assert len(models) > 0

    def test_list_models_by_provider(self):
        """Test listing models by provider."""
        openai_models = ModelRegistry.list_models(provider="openai")
        assert len(openai_models) > 0
        assert all(m.provider == "openai" for m in openai_models)

    def test_list_models_with_tools(self):
        """Test listing models with tool support."""
        models = ModelRegistry.list_models(supports_tools=True)
        assert len(models) > 0
        assert all(m.supports_tools for m in models)

    def test_list_models_with_vision(self):
        """Test listing models with vision support."""
        models = ModelRegistry.list_models(supports_vision=True)
        assert len(models) > 0
        assert all(m.supports_vision for m in models)

    def test_list_models_min_context(self):
        """Test listing models with minimum context."""
        models = ModelRegistry.list_models(min_context=1000000)
        assert len(models) > 0
        assert all(m.context_window >= 1000000 for m in models)

    def test_list_providers(self):
        """Test listing all providers."""
        providers = ModelRegistry.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers

    def test_get_context_limit(self):
        """Test getting context limit."""
        limit = ModelRegistry.get_context_limit("gpt-4o")
        assert limit == 128000

    def test_get_context_limit_unknown(self):
        """Test getting context limit for unknown model."""
        limit = ModelRegistry.get_context_limit("unknown", default=32000)
        assert limit == 32000

    def test_estimate_cost(self):
        """Test cost estimation."""
        cost = ModelRegistry.estimate_cost(
            model="gpt-4o",
            input_tokens=1000000,
            output_tokens=500000,
        )
        assert cost is not None
        # GPT-4o: $2.50/1M input + $10.00/1M output * 0.5 = $2.50 + $5.00 = $7.50
        assert abs(cost - 7.50) < 0.01

    def test_estimate_cost_with_cache(self):
        """Test cost estimation with cached tokens."""
        cost = ModelRegistry.estimate_cost(
            model="gpt-4o",
            input_tokens=1000000,
            output_tokens=0,
            cached_tokens=500000,  # Half cached
        )
        assert cost is not None
        # 500K regular at $2.50/1M + 500K cached at $1.25/1M
        # = $1.25 + $0.625 = $1.875
        assert abs(cost - 1.875) < 0.01

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        cost = ModelRegistry.estimate_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost is None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_model_info(self):
        """Test get_model_info function."""
        info = get_model_info("gpt-4o")
        assert info is not None
        assert info.name == "gpt-4o"

    def test_list_models(self):
        """Test list_models function."""
        models = list_models(provider="anthropic")
        assert len(models) > 0

    def test_register_model(self):
        """Test register_model function."""
        info = register_model(
            "test-function-model",
            provider="test",
            context_window=16000,
        )
        assert info.name == "test-function-model"


class TestBuiltInModels:
    """Tests for built-in model data."""

    def test_gpt4o_info(self):
        """Test GPT-4o model info."""
        info = get_model_info("gpt-4o")
        assert info.provider == "openai"
        assert info.context_window == 128000
        assert info.supports_tools is True
        assert info.supports_vision is True
        assert info.input_cost_per_1m == 2.50
        assert info.output_cost_per_1m == 10.00

    def test_o1_info(self):
        """Test o1 model info."""
        info = get_model_info("o1")
        assert info.provider == "openai"
        assert info.context_window == 200000  # 200K context
        assert info.max_output_tokens == 100000  # 100K output

    def test_claude_info(self):
        """Test Claude model info."""
        info = get_model_info("claude-3-5-sonnet-20241022")
        assert info.provider == "anthropic"
        assert info.context_window == 200000
        assert info.cached_input_cost_per_1m == 0.30  # 90% cache discount

    def test_gemini_info(self):
        """Test Gemini model info."""
        info = get_model_info("gemini-1.5-pro")
        assert info.provider == "google"
        assert info.context_window == 2000000  # 2M tokens!

    def test_llama_info(self):
        """Test Llama model info."""
        info = get_model_info("llama-3.1-8b")
        assert info.provider == "meta"
        assert info.context_window == 128000
        assert info.tokenizer_backend == "huggingface"

    def test_mistral_info(self):
        """Test Mistral model info."""
        info = get_model_info("mistral-large")
        assert info.provider == "mistral"
        assert info.supports_tools is True
