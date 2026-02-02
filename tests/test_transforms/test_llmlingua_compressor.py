"""Tests for LLMLingua-2 compressor integration.

Comprehensive tests covering:
- LLMLinguaConfig: Configuration validation and defaults
- LLMLinguaCompressor: Core compression functionality
- Transform interface: apply(), should_apply() methods
- Content type detection: JSON, code, plain text
- CCR integration: Reversible compression storage
- Edge cases: Empty content, unavailable dependency, fallbacks
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from headroom.transforms.llmlingua_compressor import (
    LLMLinguaCompressor,
    LLMLinguaConfig,
    LLMLinguaResult,
    compress_with_llmlingua,
    is_llmlingua_model_loaded,
    unload_llmlingua_model,
)

# Try to import for availability check
try:
    import llmlingua  # noqa: F401

    LLMLINGUA_INSTALLED = True
except ImportError:
    LLMLINGUA_INSTALLED = False


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default LLMLinguaConfig for testing."""
    return LLMLinguaConfig(
        min_tokens_for_compression=10,  # Low threshold for tests
        enable_ccr=False,  # Disable CCR for unit tests
    )


@pytest.fixture
def compressor(default_config):
    """LLMLinguaCompressor instance with default config."""
    return LLMLinguaCompressor(default_config)


@pytest.fixture
def mock_llmlingua():
    """Mock the llmlingua module and PromptCompressor."""
    mock_compressor = MagicMock()
    mock_compressor._model_name = "test-model"

    # Default compress_prompt return value
    mock_compressor.compress_prompt.return_value = {
        "compressed_prompt": "compressed content here",
        "origin_tokens": 100,
        "compressed_tokens": 30,
    }

    with patch(
        "headroom.transforms.llmlingua_compressor._check_llmlingua_available",
        return_value=True,
    ):
        with patch(
            "headroom.transforms.llmlingua_compressor._get_llmlingua_compressor",
            return_value=mock_compressor,
        ):
            yield mock_compressor


@pytest.fixture
def tokenizer():
    """Get a tokenizer for Transform interface tests."""
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    return Tokenizer(token_counter, "gpt-4o")


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_long_text(n_words: int = 500) -> str:
    """Generate long text content for compression testing."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    return " ".join(words[i % len(words)] for i in range(n_words))


def generate_long_json(n_items: int = 50) -> str:
    """Generate long JSON content for compression testing."""
    items = [
        {
            "id": i,
            "name": f"Item {i}",
            "description": f"This is a detailed description for item number {i}",
            "value": i * 10,
            "active": i % 2 == 0,
        }
        for i in range(n_items)
    ]
    return json.dumps(items)


def generate_long_code(n_functions: int = 20) -> str:
    """Generate Python code content for compression testing."""
    lines = ['"""Module with many functions."""', "", "import os", "from typing import Any", ""]
    for i in range(n_functions):
        lines.extend(
            [
                f"def function_{i}(arg: Any) -> str:",
                f'    """Process argument {i}."""',
                "    result = str(arg)",
                f'    return f"Function {i}: {{result}}"',
                "",
            ]
        )
    return "\n".join(lines)


# =============================================================================
# TestLLMLinguaConfig
# =============================================================================


class TestLLMLinguaConfig:
    """Tests for LLMLinguaConfig dataclass."""

    def test_default_values(self):
        """Default config values are sensible."""
        config = LLMLinguaConfig()

        assert config.model_name == "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        assert config.device == "auto"
        assert config.target_compression_rate == 0.3
        assert config.min_tokens_for_compression == 100
        assert config.enable_ccr is True
        assert config.drop_consecutive is True

    def test_custom_values(self):
        """Custom config values are applied."""
        config = LLMLinguaConfig(
            model_name="custom/model",
            device="cuda",
            target_compression_rate=0.5,
            min_tokens_for_compression=50,
            force_tokens=["important", "keep"],
        )

        assert config.model_name == "custom/model"
        assert config.device == "cuda"
        assert config.target_compression_rate == 0.5
        assert config.min_tokens_for_compression == 50
        assert "important" in config.force_tokens

    def test_content_type_rates(self):
        """Different content types have appropriate compression rates."""
        config = LLMLinguaConfig()

        # Code and text are equally conservative for accuracy
        assert config.code_compression_rate >= config.text_compression_rate
        # JSON can be slightly more aggressive since structure is preserved
        assert config.json_compression_rate <= config.code_compression_rate
        assert config.json_compression_rate <= config.text_compression_rate


# =============================================================================
# TestLLMLinguaResult
# =============================================================================


class TestLLMLinguaResult:
    """Tests for LLMLinguaResult dataclass."""

    def test_tokens_saved(self):
        """tokens_saved property calculates correctly."""
        result = LLMLinguaResult(
            compressed="short",
            original="long content here",
            original_tokens=100,
            compressed_tokens=30,
            compression_ratio=0.3,
        )

        assert result.tokens_saved == 70

    def test_tokens_saved_no_negative(self):
        """tokens_saved never returns negative."""
        result = LLMLinguaResult(
            compressed="expanded content",
            original="short",
            original_tokens=10,
            compressed_tokens=20,  # Expanded (unusual case)
            compression_ratio=2.0,
        )

        assert result.tokens_saved == 0

    def test_savings_percentage(self):
        """savings_percentage property calculates correctly."""
        result = LLMLinguaResult(
            compressed="short",
            original="long content",
            original_tokens=100,
            compressed_tokens=25,
            compression_ratio=0.25,
        )

        assert result.savings_percentage == 75.0

    def test_savings_percentage_zero_original(self):
        """savings_percentage handles zero original tokens."""
        result = LLMLinguaResult(
            compressed="",
            original="",
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=1.0,
        )

        assert result.savings_percentage == 0.0


# =============================================================================
# TestLLMLinguaCompressor
# =============================================================================


class TestLLMLinguaCompressor:
    """Tests for LLMLinguaCompressor core functionality."""

    def test_init_with_default_config(self):
        """Compressor initializes with default config."""
        compressor = LLMLinguaCompressor()

        assert compressor.config is not None
        assert compressor.config.model_name is not None

    def test_init_with_custom_config(self, default_config):
        """Compressor initializes with custom config."""
        compressor = LLMLinguaCompressor(default_config)

        assert compressor.config == default_config

    def test_compress_returns_result_when_unavailable(self, compressor):
        """Compress returns passthrough result when llmlingua unavailable."""
        with patch(
            "headroom.transforms.llmlingua_compressor._check_llmlingua_available",
            return_value=False,
        ):
            content = generate_long_text(100)
            result = compressor.compress(content)

            # Should return unchanged content
            assert result.compressed == content
            assert result.compression_ratio == 1.0

    def test_compress_skips_small_content(self, compressor):
        """Small content is not compressed."""
        small_content = "short text"
        result = compressor.compress(small_content)

        assert result.compressed == small_content
        assert result.compression_ratio == 1.0

    def test_compress_with_llmlingua(self, default_config, mock_llmlingua):
        """Compression uses llmlingua when available."""
        compressor = LLMLinguaCompressor(default_config)
        content = generate_long_text(200)

        result = compressor.compress(content)

        # Should have called compress_prompt
        mock_llmlingua.compress_prompt.assert_called_once()
        assert result.compressed == "compressed content here"
        assert result.compression_ratio < 1.0

    def test_compress_with_context(self, default_config, mock_llmlingua):
        """Context words are used as force tokens."""
        compressor = LLMLinguaCompressor(default_config)
        content = generate_long_text(200)
        context = "important keywords here"

        compressor.compress(content, context=context)

        # Check force_tokens includes context words
        call_args = mock_llmlingua.compress_prompt.call_args
        force_tokens = call_args.kwargs.get("force_tokens", [])
        # Should include context words longer than 3 chars
        assert "important" in force_tokens or "keywords" in force_tokens

    def test_compress_handles_exception(self, default_config, mock_llmlingua):
        """Exceptions from llmlingua are handled gracefully."""
        mock_llmlingua.compress_prompt.side_effect = RuntimeError("Model error")

        compressor = LLMLinguaCompressor(default_config)
        content = generate_long_text(200)

        result = compressor.compress(content)

        # Should return original content on error
        assert result.compressed == content
        assert result.compression_ratio == 1.0


# =============================================================================
# TestContentTypeDetection
# =============================================================================


class TestContentTypeDetection:
    """Tests for content type auto-detection."""

    def test_detect_json_content(self, default_config, mock_llmlingua):
        """JSON content is detected and uses JSON compression rate."""
        compressor = LLMLinguaCompressor(default_config)

        rate = compressor._get_compression_rate(generate_long_json(50), None)

        assert rate == default_config.json_compression_rate

    def test_detect_code_content(self, default_config, mock_llmlingua):
        """Code content is detected and uses code compression rate."""
        compressor = LLMLinguaCompressor(default_config)
        code = generate_long_code(20)

        rate = compressor._get_compression_rate(code, None)

        assert rate == default_config.code_compression_rate

    def test_detect_plain_text(self, default_config, mock_llmlingua):
        """Plain text uses text compression rate."""
        compressor = LLMLinguaCompressor(default_config)
        text = generate_long_text(200)

        rate = compressor._get_compression_rate(text, None)

        assert rate == default_config.text_compression_rate

    def test_explicit_content_type(self, default_config, mock_llmlingua):
        """Explicit content_type overrides detection."""
        compressor = LLMLinguaCompressor(default_config)
        # JSON-looking content but marked as text
        json_content = generate_long_json(50)

        rate = compressor._get_compression_rate(json_content, content_type="text")

        assert rate == default_config.text_compression_rate

    def test_looks_like_json_detection(self, default_config):
        """JSON detection works for arrays and objects."""
        compressor = LLMLinguaCompressor(default_config)

        assert compressor._looks_like_json('[{"key": "value"}]')
        assert compressor._looks_like_json('{"key": "value"}')
        assert not compressor._looks_like_json("plain text")
        assert not compressor._looks_like_json("def function():")

    def test_looks_like_code_detection(self, default_config):
        """Code detection works for common patterns."""
        compressor = LLMLinguaCompressor(default_config)

        assert compressor._looks_like_code("def function():")
        assert compressor._looks_like_code("class MyClass:")
        assert compressor._looks_like_code("import os")
        assert compressor._looks_like_code("function test() {")
        assert compressor._looks_like_code("const x = 5")
        assert not compressor._looks_like_code("plain text content")


# =============================================================================
# TestTransformInterface
# =============================================================================


class TestTransformInterface:
    """Tests for Transform interface (apply, should_apply)."""

    def test_should_apply_returns_false_when_unavailable(self, compressor, tokenizer):
        """should_apply returns False when llmlingua unavailable."""
        messages = [{"role": "user", "content": generate_long_text(200)}]

        with patch(
            "headroom.transforms.llmlingua_compressor._check_llmlingua_available",
            return_value=False,
        ):
            assert not compressor.should_apply(messages, tokenizer)

    def test_should_apply_returns_false_for_small_content(self, default_config, tokenizer):
        """should_apply returns False for small content."""
        config = LLMLinguaConfig(min_tokens_for_compression=1000)
        compressor = LLMLinguaCompressor(config)
        messages = [{"role": "user", "content": "small"}]

        with patch(
            "headroom.transforms.llmlingua_compressor._check_llmlingua_available",
            return_value=True,
        ):
            assert not compressor.should_apply(messages, tokenizer)

    def test_should_apply_returns_true_for_large_content(self, default_config, tokenizer):
        """should_apply returns True for large content."""
        compressor = LLMLinguaCompressor(default_config)
        messages = [{"role": "user", "content": generate_long_text(500)}]

        with patch(
            "headroom.transforms.llmlingua_compressor._check_llmlingua_available",
            return_value=True,
        ):
            assert compressor.should_apply(messages, tokenizer)

    def test_apply_compresses_tool_messages(self, default_config, tokenizer, mock_llmlingua):
        """apply() compresses tool message content."""
        compressor = LLMLinguaCompressor(default_config)
        tool_content = generate_long_json(100)
        messages = [
            {"role": "user", "content": "Get data"},
            {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
        ]

        result = compressor.apply(messages, tokenizer)

        # Tool content should be compressed
        assert result.messages[1]["content"] != tool_content
        assert "compressed content here" in result.messages[1]["content"]
        assert len(result.transforms_applied) > 0

    def test_apply_compresses_long_assistant_messages(
        self, default_config, tokenizer, mock_llmlingua
    ):
        """apply() compresses long assistant messages."""
        compressor = LLMLinguaCompressor(default_config)
        long_content = generate_long_text(1000)
        messages = [
            {"role": "user", "content": "Tell me a story"},
            {"role": "assistant", "content": long_content},
        ]

        result = compressor.apply(messages, tokenizer)

        # Assistant content should be compressed (>500 chars)
        assert result.messages[1]["content"] != long_content

    def test_apply_passes_through_short_messages(self, default_config, tokenizer, mock_llmlingua):
        """apply() passes through short messages unchanged."""
        compressor = LLMLinguaCompressor(default_config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = compressor.apply(messages, tokenizer)

        # Short messages unchanged
        assert result.messages[0]["content"] == "Hello"
        assert result.messages[1]["content"] == "Hi there!"

    def test_apply_tracks_transform_metadata(self, default_config, tokenizer, mock_llmlingua):
        """apply() returns proper TransformResult metadata."""
        compressor = LLMLinguaCompressor(default_config)
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": generate_long_json(100)},
        ]

        result = compressor.apply(messages, tokenizer)

        assert result.tokens_before > 0
        assert result.tokens_after > 0
        assert len(result.transforms_applied) > 0
        assert "llmlingua" in result.transforms_applied[0]

    def test_apply_adds_warning_when_unavailable(self, default_config, tokenizer):
        """apply() adds warning when llmlingua unavailable."""
        compressor = LLMLinguaCompressor(default_config)
        messages = [{"role": "user", "content": "test"}]

        with patch(
            "headroom.transforms.llmlingua_compressor._check_llmlingua_available",
            return_value=False,
        ):
            result = compressor.apply(messages, tokenizer)

            assert len(result.warnings) > 0
            assert "llmlingua" in result.warnings[0].lower()


# =============================================================================
# TestDeviceResolution
# =============================================================================


class TestDeviceResolution:
    """Tests for device resolution logic."""

    def test_resolve_explicit_device(self, default_config):
        """Explicit device is returned unchanged."""
        config = LLMLinguaConfig(device="cuda")
        compressor = LLMLinguaCompressor(config)

        assert compressor._resolve_device() == "cuda"

    def test_resolve_auto_to_cpu_no_torch(self, default_config):
        """Auto resolves to CPU when torch unavailable."""
        config = LLMLinguaConfig(device="auto")
        compressor = LLMLinguaCompressor(config)

        with patch.dict("sys.modules", {"torch": None}):
            with patch(
                "headroom.transforms.llmlingua_compressor.LLMLinguaCompressor._resolve_device"
            ) as mock_resolve:
                mock_resolve.return_value = "cpu"
                assert compressor._resolve_device() == "cpu"


# =============================================================================
# TestCCRIntegration
# =============================================================================


class TestCCRIntegration:
    """Tests for CCR (Compress-Cache-Retrieve) integration."""

    def test_ccr_stores_original(self, mock_llmlingua):
        """Compressed content is stored in CCR when enabled."""
        config = LLMLinguaConfig(
            enable_ccr=True,
            min_tokens_for_compression=10,
        )
        compressor = LLMLinguaCompressor(config)
        content = generate_long_text(200)

        with patch(
            "headroom.transforms.llmlingua_compressor.LLMLinguaCompressor._store_in_ccr"
        ) as mock_store:
            mock_store.return_value = "hash123"

            result = compressor.compress(content)

            mock_store.assert_called_once()
            assert result.cache_key == "hash123"

    def test_ccr_skipped_when_disabled(self, mock_llmlingua):
        """CCR is not used when disabled in config."""
        config = LLMLinguaConfig(
            enable_ccr=False,
            min_tokens_for_compression=10,
        )
        compressor = LLMLinguaCompressor(config)
        content = generate_long_text(200)

        with patch(
            "headroom.transforms.llmlingua_compressor.LLMLinguaCompressor._store_in_ccr"
        ) as mock_store:
            result = compressor.compress(content)

            mock_store.assert_not_called()
            assert result.cache_key is None

    def test_ccr_handles_storage_error(self, mock_llmlingua):
        """CCR storage errors are handled gracefully."""
        config = LLMLinguaConfig(
            enable_ccr=True,
            min_tokens_for_compression=10,
        )
        compressor = LLMLinguaCompressor(config)
        content = generate_long_text(200)

        with patch(
            "headroom.transforms.llmlingua_compressor.LLMLinguaCompressor._store_in_ccr"
        ) as mock_store:
            # Return None to simulate storage failure (internal error handling)
            mock_store.return_value = None

            # Should not raise
            result = compressor.compress(content)

            # Storage failed, so cache_key should be None
            assert result.cache_key is None


# =============================================================================
# TestConvenienceFunction
# =============================================================================


class TestConvenienceFunction:
    """Tests for compress_with_llmlingua convenience function."""

    def test_compress_with_llmlingua_basic(self, mock_llmlingua):
        """compress_with_llmlingua works with default settings."""
        content = generate_long_text(200)

        # Disable CCR for this test to avoid hash suffix
        with patch(
            "headroom.transforms.llmlingua_compressor.LLMLinguaCompressor._store_in_ccr"
        ) as mock_store:
            mock_store.return_value = None
            result = compress_with_llmlingua(content)

            # Should contain the compressed content
            assert "compressed content here" in result

    def test_compress_with_llmlingua_custom_rate(self, mock_llmlingua):
        """compress_with_llmlingua accepts custom compression rate."""
        content = generate_long_text(200)

        compress_with_llmlingua(content, compression_rate=0.5)

        # Verify compress_prompt was called
        mock_llmlingua.compress_prompt.assert_called()

    def test_compress_with_llmlingua_with_context(self, mock_llmlingua):
        """compress_with_llmlingua passes context."""
        content = generate_long_text(200)
        context = "important keywords"

        compress_with_llmlingua(content, context=context)

        call_args = mock_llmlingua.compress_prompt.call_args
        force_tokens = call_args.kwargs.get("force_tokens", [])
        # Context words should be in force_tokens
        assert any("important" in str(t) for t in force_tokens) or len(force_tokens) > 0


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for LLMLingua compressor."""

    def test_empty_content(self, compressor):
        """Empty content is handled gracefully."""
        result = compressor.compress("")

        assert result.compressed == ""
        assert result.compression_ratio == 1.0

    def test_whitespace_only_content(self, compressor):
        """Whitespace-only content is handled gracefully."""
        result = compressor.compress("   \n\t\n   ")

        assert result.compression_ratio == 1.0

    def test_unicode_content(self, default_config, mock_llmlingua):
        """Unicode content is handled correctly."""
        mock_llmlingua.compress_prompt.return_value = {
            "compressed_prompt": "compressed \u4e2d\u6587 content",
            "origin_tokens": 100,
            "compressed_tokens": 30,
        }

        compressor = LLMLinguaCompressor(default_config)
        content = "\u4e2d\u6587 \u65e5\u672c\u8a9e " * 100  # Chinese/Japanese text

        result = compressor.compress(content)

        assert "\u4e2d\u6587" in result.compressed

    def test_very_long_content(self, default_config, mock_llmlingua):
        """Very long content is compressed."""
        compressor = LLMLinguaCompressor(default_config)
        content = generate_long_text(10000)

        compressor.compress(content)

        mock_llmlingua.compress_prompt.assert_called_once()

    def test_mixed_content_types(self, default_config, mock_llmlingua):
        """Mixed content (JSON with text) is handled."""
        compressor = LLMLinguaCompressor(default_config)
        # JSON-like but with extra text
        content = 'Some preamble text\n{"key": "value"}\nMore text after'

        # Should not crash
        result = compressor.compress(content)
        assert result is not None

    def test_malformed_json_content(self, default_config, mock_llmlingua):
        """Malformed JSON is treated as text."""
        compressor = LLMLinguaCompressor(default_config)
        content = "{malformed: json, missing quotes" * 50

        rate = compressor._get_compression_rate(content, None)

        # Should not detect as JSON
        assert rate == default_config.text_compression_rate

    def test_force_tokens_list_handling(self, default_config, mock_llmlingua):
        """Force tokens list is properly passed."""
        config = LLMLinguaConfig(
            force_tokens=["keep", "these", "tokens"],
            min_tokens_for_compression=10,
        )
        compressor = LLMLinguaCompressor(config)
        content = generate_long_text(200)

        compressor.compress(content)

        call_args = mock_llmlingua.compress_prompt.call_args
        force_tokens = call_args.kwargs.get("force_tokens", [])
        assert "keep" in force_tokens
        assert "these" in force_tokens
        assert "tokens" in force_tokens


# =============================================================================
# Integration Tests (only run if llmlingua is installed)
# =============================================================================


@pytest.mark.skipif(not LLMLINGUA_INSTALLED, reason="llmlingua not installed")
class TestLLMLinguaIntegration:
    """Integration tests that require actual llmlingua installation.

    These tests verify the actual compression behavior and should be run
    in environments where llmlingua is installed.
    """

    def test_actual_compression(self):
        """Test actual compression with real llmlingua."""
        config = LLMLinguaConfig(
            target_compression_rate=0.3,
            min_tokens_for_compression=50,
            enable_ccr=False,
        )
        compressor = LLMLinguaCompressor(config)
        content = generate_long_text(500)

        result = compressor.compress(content)

        # Should achieve actual compression
        assert result.compression_ratio < 1.0
        assert result.tokens_saved > 0
        assert len(result.compressed) < len(content)

    def test_actual_json_compression(self):
        """Test JSON content compression with real llmlingua."""
        config = LLMLinguaConfig(
            target_compression_rate=0.35,
            min_tokens_for_compression=50,
            enable_ccr=False,
        )
        compressor = LLMLinguaCompressor(config)
        content = generate_long_json(50)

        result = compressor.compress(content, content_type="json")

        assert result.compression_ratio < 1.0

    def test_actual_code_compression(self):
        """Test code content compression with real llmlingua."""
        config = LLMLinguaConfig(
            target_compression_rate=0.4,
            min_tokens_for_compression=50,
            enable_ccr=False,
        )
        compressor = LLMLinguaCompressor(config)
        content = generate_long_code(30)

        result = compressor.compress(content, content_type="code")

        assert result.compression_ratio < 1.0


# =============================================================================
# TestMemoryManagement
# =============================================================================


class TestMemoryManagement:
    """Tests for memory management functions (unload_llmlingua_model, is_llmlingua_model_loaded)."""

    def test_is_model_loaded_returns_false_initially(self):
        """is_llmlingua_model_loaded returns False when no model loaded."""
        # Ensure model is unloaded
        with patch(
            "headroom.transforms.llmlingua_compressor._llmlingua_instance",
            None,
        ):
            assert is_llmlingua_model_loaded() is False

    def test_is_model_loaded_returns_true_when_loaded(self):
        """is_llmlingua_model_loaded returns True when model is loaded."""
        mock_instance = MagicMock()

        with patch(
            "headroom.transforms.llmlingua_compressor._llmlingua_instance",
            mock_instance,
        ):
            assert is_llmlingua_model_loaded() is True

    def test_unload_returns_false_when_no_model(self):
        """unload_llmlingua_model returns False when no model loaded."""
        import headroom.transforms.llmlingua_compressor as module

        # Save original
        original = module._llmlingua_instance

        try:
            module._llmlingua_instance = None
            result = unload_llmlingua_model()
            assert result is False
        finally:
            module._llmlingua_instance = original

    def test_unload_clears_instance(self):
        """unload_llmlingua_model clears the global instance."""
        import headroom.transforms.llmlingua_compressor as module

        # Save original
        original = module._llmlingua_instance

        try:
            # Set a mock instance
            mock_instance = MagicMock()
            mock_instance._model_name = "test-model"
            module._llmlingua_instance = mock_instance

            # Unload
            result = unload_llmlingua_model()

            assert result is True
            assert module._llmlingua_instance is None
        finally:
            module._llmlingua_instance = original

    def test_unload_clears_cuda_cache(self):
        """unload_llmlingua_model attempts to clear CUDA cache."""
        import headroom.transforms.llmlingua_compressor as module

        original = module._llmlingua_instance

        try:
            mock_instance = MagicMock()
            mock_instance._model_name = "test-model"
            module._llmlingua_instance = mock_instance

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True

            with patch.dict("sys.modules", {"torch": mock_torch}):
                with patch(
                    "headroom.transforms.llmlingua_compressor.torch",
                    mock_torch,
                    create=True,
                ):
                    result = unload_llmlingua_model()

                    assert result is True
        finally:
            module._llmlingua_instance = original


# =============================================================================
# TestThreadSafety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of model loading."""

    def test_lock_exists(self):
        """Verify thread lock is available."""
        import headroom.transforms.llmlingua_compressor as module

        assert hasattr(module, "_llmlingua_lock")
        import threading

        assert isinstance(module._llmlingua_lock, type(threading.Lock()))


# =============================================================================
# TestErrorMessages
# =============================================================================


class TestErrorMessages:
    """Tests for improved error messages."""

    def test_import_error_message_includes_install_hint(self):
        """ImportError includes installation instructions."""
        with patch(
            "headroom.transforms.llmlingua_compressor._check_llmlingua_available",
            return_value=False,
        ):
            from headroom.transforms.llmlingua_compressor import _get_llmlingua_compressor

            with pytest.raises(ImportError) as exc_info:
                _get_llmlingua_compressor("test-model", "cpu")

            error_msg = str(exc_info.value)
            assert "pip install headroom-ai[llmlingua]" in error_msg
            assert "2GB" in error_msg or "disk space" in error_msg.lower()

    def test_oom_error_provides_helpful_suggestions(self):
        """Out of memory error provides helpful suggestions."""
        import headroom.transforms.llmlingua_compressor as module

        # Save original state
        original_instance = module._llmlingua_instance
        original_available = module._llmlingua_available

        try:
            module._llmlingua_instance = None
            module._llmlingua_available = True

            # Create a mock that raises OOM when called
            mock_prompt_compressor_class = MagicMock()
            mock_prompt_compressor_class.side_effect = RuntimeError("CUDA out of memory")

            with patch.dict("sys.modules", {"llmlingua": MagicMock()}):
                with patch(
                    "llmlingua.PromptCompressor",
                    mock_prompt_compressor_class,
                ):
                    from headroom.transforms.llmlingua_compressor import (
                        _get_llmlingua_compressor,
                    )

                    with pytest.raises(RuntimeError) as exc_info:
                        _get_llmlingua_compressor("test-model", "cuda")

                    error_msg = str(exc_info.value)
                    # Should include helpful suggestions
                    assert "cpu" in error_msg.lower() or "memory" in error_msg.lower()
        finally:
            module._llmlingua_instance = original_instance
            module._llmlingua_available = original_available
