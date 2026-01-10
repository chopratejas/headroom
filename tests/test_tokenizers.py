"""Tests for the pluggable tokenizer system."""

from __future__ import annotations

import pytest

from headroom.tokenizers import (
    BaseTokenizer,
    CharacterCounter,
    EstimatingTokenCounter,
    TiktokenCounter,
    TokenCounter,
    TokenizerRegistry,
    get_mistral_tokenizer,
    get_tokenizer,
    is_mistral_tokenizer_available,
    list_supported_models,
    register_tokenizer,
)


class TestTiktokenCounter:
    """Tests for TiktokenCounter."""

    def test_init_default_model(self):
        """Test initialization with default model."""
        counter = TiktokenCounter()
        assert counter.model == "gpt-4o"
        assert counter.encoding_name == "o200k_base"

    def test_init_gpt4_model(self):
        """Test initialization with GPT-4."""
        counter = TiktokenCounter("gpt-4")
        assert counter.model == "gpt-4"
        assert counter.encoding_name == "cl100k_base"

    def test_count_text_empty(self):
        """Test counting empty text."""
        counter = TiktokenCounter()
        assert counter.count_text("") == 0

    def test_count_text_simple(self):
        """Test counting simple text."""
        counter = TiktokenCounter()
        count = counter.count_text("Hello, world!")
        assert count > 0
        assert count < 10  # Should be a few tokens

    def test_count_text_unicode(self):
        """Test counting text with unicode."""
        counter = TiktokenCounter()
        count = counter.count_text("Hello, 世界!")
        assert count > 0

    def test_count_messages_single(self):
        """Test counting single message."""
        counter = TiktokenCounter()
        messages = [{"role": "user", "content": "Hello!"}]
        count = counter.count_messages(messages)
        assert count > 0

    def test_count_messages_with_tool_calls(self):
        """Test counting messages with tool calls."""
        counter = TiktokenCounter()
        messages = [
            {"role": "user", "content": "Search for Python"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "Python"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Results...",
            },
        ]
        count = counter.count_messages(messages)
        assert count > 0

    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        counter = TiktokenCounter()
        text = "Hello, world!"
        tokens = counter.encode(text)
        decoded = counter.decode(tokens)
        assert decoded == text

    def test_repr(self):
        """Test string representation."""
        counter = TiktokenCounter("gpt-4o")
        assert "TiktokenCounter" in repr(counter)
        assert "gpt-4o" in repr(counter)


class TestEstimatingTokenCounter:
    """Tests for EstimatingTokenCounter."""

    def test_init_default(self):
        """Test initialization with defaults."""
        counter = EstimatingTokenCounter()
        assert counter._fixed_ratio is None

    def test_init_fixed_ratio(self):
        """Test initialization with fixed ratio."""
        counter = EstimatingTokenCounter(chars_per_token=3.5)
        assert counter._fixed_ratio == 3.5

    def test_count_text_empty(self):
        """Test counting empty text."""
        counter = EstimatingTokenCounter()
        assert counter.count_text("") == 0

    def test_count_text_simple(self):
        """Test counting simple text."""
        counter = EstimatingTokenCounter()
        text = "Hello, world!"
        count = counter.count_text(text)
        assert count > 0
        # Rough estimate: 13 chars / 4 chars per token ≈ 3-4 tokens
        assert 2 <= count <= 6

    def test_count_text_fixed_ratio(self):
        """Test counting with fixed ratio."""
        counter = EstimatingTokenCounter(chars_per_token=5.0)
        text = "x" * 50  # 50 chars
        count = counter.count_text(text)
        assert count == 10  # 50 / 5 = 10

    def test_count_text_minimum_one(self):
        """Test minimum of 1 token."""
        counter = EstimatingTokenCounter()
        assert counter.count_text("x") >= 1

    def test_count_messages(self):
        """Test counting messages."""
        counter = EstimatingTokenCounter()
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = counter.count_messages(messages)
        assert count > 0

    def test_json_detection(self):
        """Test JSON content detection."""
        counter = EstimatingTokenCounter()
        json_text = '{"name": "test", "value": 123}'
        # Should use JSON ratio
        count = counter.count_text(json_text)
        assert count > 0

    def test_code_detection(self):
        """Test code content detection."""
        counter = EstimatingTokenCounter()
        code_text = """
def hello():
    return "Hello, world!"
"""
        count = counter.count_text(code_text)
        assert count > 0

    def test_repr(self):
        """Test string representation."""
        counter = EstimatingTokenCounter()
        assert "EstimatingTokenCounter" in repr(counter)


class TestCharacterCounter:
    """Tests for CharacterCounter."""

    def test_init_default(self):
        """Test initialization with default ratio."""
        counter = CharacterCounter()
        assert counter.chars_per_token == 4.0

    def test_init_custom_ratio(self):
        """Test initialization with custom ratio."""
        counter = CharacterCounter(chars_per_token=3.5)
        assert counter.chars_per_token == 3.5

    def test_count_text(self):
        """Test counting text."""
        counter = CharacterCounter(chars_per_token=4.0)
        text = "x" * 40  # 40 chars
        count = counter.count_text(text)
        assert count == 10  # 40 / 4 = 10

    def test_count_text_empty(self):
        """Test counting empty text."""
        counter = CharacterCounter()
        assert counter.count_text("") == 0


class TestTokenizerRegistry:
    """Tests for TokenizerRegistry."""

    def test_get_openai_model(self):
        """Test getting tokenizer for OpenAI model."""
        tokenizer = get_tokenizer("gpt-4o")
        assert isinstance(tokenizer, TiktokenCounter)

    def test_get_anthropic_model(self):
        """Test getting tokenizer for Anthropic model."""
        tokenizer = get_tokenizer("claude-3-sonnet")
        assert isinstance(tokenizer, EstimatingTokenCounter)

    def test_get_unknown_model_fallback(self):
        """Test fallback for unknown model."""
        tokenizer = get_tokenizer("unknown-model-xyz")
        assert isinstance(tokenizer, EstimatingTokenCounter)

    def test_get_with_specific_backend(self):
        """Test forcing specific backend."""
        tokenizer = get_tokenizer("any-model", backend="estimation")
        assert isinstance(tokenizer, EstimatingTokenCounter)

    def test_register_custom_tokenizer(self):
        """Test registering custom tokenizer."""
        custom = EstimatingTokenCounter(chars_per_token=3.0)
        register_tokenizer("my-custom-model", tokenizer=custom)
        retrieved = get_tokenizer("my-custom-model")
        assert retrieved is custom

    def test_list_supported_models(self):
        """Test listing supported models."""
        models = list_supported_models()
        assert isinstance(models, dict)
        assert "gpt-4o" in str(models) or "^gpt-4o" in str(models)

    def test_clear_cache(self):
        """Test clearing tokenizer cache."""
        # Get a tokenizer to populate cache
        get_tokenizer("gpt-4o")
        # Clear cache
        TokenizerRegistry.clear_cache()
        # Should still work after clearing
        tokenizer = get_tokenizer("gpt-4o")
        assert tokenizer is not None


class TestTokenCounterProtocol:
    """Tests for TokenCounter protocol."""

    def test_tiktoken_implements_protocol(self):
        """Test TiktokenCounter implements protocol."""
        counter = TiktokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_estimating_implements_protocol(self):
        """Test EstimatingTokenCounter implements protocol."""
        counter = EstimatingTokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_character_implements_protocol(self):
        """Test CharacterCounter implements protocol."""
        counter = CharacterCounter()
        assert isinstance(counter, TokenCounter)


class TestBaseTokenizer:
    """Tests for BaseTokenizer base class."""

    def test_message_overhead_constant(self):
        """Test message overhead constant."""
        assert BaseTokenizer.MESSAGE_OVERHEAD == 4

    def test_reply_overhead_constant(self):
        """Test reply overhead constant."""
        assert BaseTokenizer.REPLY_OVERHEAD == 3


class TestMistralTokenizer:
    """Tests for Mistral tokenizer using official mistral-common."""

    def test_is_available(self):
        """Test availability check."""
        result = is_mistral_tokenizer_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_get_mistral_tokenizer_class(self):
        """Test getting MistralTokenizer class."""
        MistralTokenizer = get_mistral_tokenizer()
        assert MistralTokenizer is not None
        assert hasattr(MistralTokenizer, "count_text")

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_init_default_model(self):
        """Test initialization with default model."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        assert counter.model == "mistral-large"
        assert counter.version == "v3"

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_init_mixtral_model(self):
        """Test initialization with Mixtral model (uses v1)."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer("mixtral-8x7b")
        assert counter.version == "v1"

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_text_empty(self):
        """Test counting empty text."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        assert counter.count_text("") == 0

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_text_simple(self):
        """Test counting simple text."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        count = counter.count_text("Hello, world!")
        assert count > 0
        assert count < 10

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_text_unicode(self):
        """Test counting text with unicode."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        count = counter.count_text("Hello, 世界!")
        assert count > 0

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_messages(self):
        """Test counting messages."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = counter.count_messages(messages)
        assert count > 0

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_messages_with_system(self):
        """Test counting messages with system prompt."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        count = counter.count_messages(messages)
        assert count > 0

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        text = "Hello, world!"
        tokens = counter.encode(text)
        decoded = counter.decode(tokens)
        assert decoded == text

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_implements_protocol(self):
        """Test MistralTokenizer implements TokenCounter protocol."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        assert isinstance(counter, TokenCounter)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_repr(self):
        """Test string representation."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer("mistral-large")
        assert "MistralTokenizer" in repr(counter)
        assert "mistral-large" in repr(counter)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_registry_returns_mistral_for_mistral_models(self):
        """Test registry returns Mistral tokenizer for Mistral models."""
        tokenizer = get_tokenizer("mistral-large")
        MistralTokenizer = get_mistral_tokenizer()
        assert isinstance(tokenizer, MistralTokenizer)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_registry_returns_mistral_for_mixtral(self):
        """Test registry returns Mistral tokenizer for Mixtral models."""
        tokenizer = get_tokenizer("mixtral-8x7b")
        MistralTokenizer = get_mistral_tokenizer()
        assert isinstance(tokenizer, MistralTokenizer)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_registry_returns_mistral_for_codestral(self):
        """Test registry returns Mistral tokenizer for Codestral models."""
        tokenizer = get_tokenizer("codestral")
        MistralTokenizer = get_mistral_tokenizer()
        assert isinstance(tokenizer, MistralTokenizer)
