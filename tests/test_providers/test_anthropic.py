"""Tests for Anthropic provider."""

import pytest


class TestAnthropicTokenCounting:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_count_text_fallback(self, anthropic_provider):
        # Without API client, should use tiktoken fallback
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        count = counter.count_text("Hello world")
        assert count > 0

    def test_count_messages_basic(self, anthropic_provider):
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        messages = [{"role": "user", "content": "Hello"}]
        count = counter.count_messages(messages)
        assert count > 0


class TestAnthropicModelLimits:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_get_context_limit_claude_sonnet(self, anthropic_provider):
        limit = anthropic_provider.get_context_limit("claude-3-5-sonnet-20241022")
        assert limit == 200000

    def test_get_context_limit_claude_opus(self, anthropic_provider):
        limit = anthropic_provider.get_context_limit("claude-3-opus-20240229")
        assert limit == 200000

    def test_supports_model_known(self, anthropic_provider):
        assert anthropic_provider.supports_model("claude-3-5-sonnet-20241022")

    def test_supports_model_prefix(self, anthropic_provider):
        assert anthropic_provider.supports_model("claude-3-5-sonnet-latest")


class TestAnthropicCostEstimation:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_estimate_cost_basic(self, anthropic_provider):
        cost = anthropic_provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=0,
            model="claude-3-5-sonnet-20241022",
        )
        # $3.00 per 1M input
        assert cost == pytest.approx(3.00, rel=0.1)
