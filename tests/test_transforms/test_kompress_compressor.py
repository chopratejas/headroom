"""Tests for Kompress compressor.

Covers:
- Lazy imports: module importable without torch installed
- is_kompress_available(): correct detection of [ml] extra
- KompressConfig / KompressResult: dataclass defaults
- KompressCompressor: passthrough for short content, fallback on error
- Transform interface: apply() method
"""

from unittest.mock import MagicMock, patch

# ── Import safety (the whole point of the fix) ─────────────────────────


class TestLazyImports:
    """The module must be importable without torch/transformers."""

    def test_is_kompress_available_importable(self) -> None:
        """is_kompress_available can be imported even without torch."""
        from headroom.transforms.kompress_compressor import is_kompress_available

        # Should return bool (True or False depending on environment)
        result = is_kompress_available()
        assert isinstance(result, bool)

    def test_module_import_without_torch(self) -> None:
        """Importing the module with torch blocked should not raise."""
        import sys

        # Block torch imports
        with patch.dict(sys.modules, {"torch": None, "torch.nn": None}):
            # Force re-evaluation of is_kompress_available
            from headroom.transforms.kompress_compressor import is_kompress_available

            # Should gracefully return False, not crash
            assert is_kompress_available() is False

    def test_dataclasses_importable_without_torch(self) -> None:
        """KompressConfig, KompressResult, KompressCompressor are importable without torch."""
        from headroom.transforms.kompress_compressor import (
            KompressCompressor,  # noqa: F401
            KompressConfig,
            KompressResult,
        )

        # These don't need torch to instantiate
        config = KompressConfig()
        assert config.device == "auto"
        assert config.enable_ccr is True

        result = KompressResult(
            compressed="hello",
            original="hello world",
            original_tokens=2,
            compressed_tokens=1,
            compression_ratio=0.5,
        )
        assert result.tokens_saved == 1
        assert result.savings_percentage == 50.0


# ── KompressResult ──────────────────────────────────────────────────────


class TestKompressResult:
    def test_tokens_saved(self) -> None:
        from headroom.transforms.kompress_compressor import KompressResult

        r = KompressResult(
            compressed="a b",
            original="a b c d",
            original_tokens=4,
            compressed_tokens=2,
            compression_ratio=0.5,
        )
        assert r.tokens_saved == 2

    def test_tokens_saved_no_negative(self) -> None:
        from headroom.transforms.kompress_compressor import KompressResult

        r = KompressResult(
            compressed="a b c d e",
            original="a b c",
            original_tokens=3,
            compressed_tokens=5,
            compression_ratio=1.67,
        )
        assert r.tokens_saved == 0

    def test_savings_percentage_zero_tokens(self) -> None:
        from headroom.transforms.kompress_compressor import KompressResult

        r = KompressResult(
            compressed="",
            original="",
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=1.0,
        )
        assert r.savings_percentage == 0.0

    def test_default_model(self) -> None:
        from headroom.transforms.kompress_compressor import HF_MODEL_ID, KompressResult

        r = KompressResult(
            compressed="x",
            original="x y",
            original_tokens=2,
            compressed_tokens=1,
            compression_ratio=0.5,
        )
        assert r.model_used == HF_MODEL_ID


# ── KompressCompressor (without model) ──────────────────────────────────


class TestKompressCompressorPassthrough:
    """Test compressor behavior that doesn't require the actual model."""

    def test_short_content_passthrough(self) -> None:
        """Content under 10 words should pass through unchanged."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        result = compressor.compress("hello world")
        assert result.compressed == "hello world"
        assert result.compression_ratio == 1.0
        assert result.original_tokens == 2
        assert result.compressed_tokens == 2

    def test_empty_content_passthrough(self) -> None:
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        result = compressor.compress("")
        assert result.compressed == ""
        assert result.compression_ratio == 1.0

    def test_fallback_on_model_error(self) -> None:
        """If _load_kompress fails, compress should return passthrough."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        long_text = " ".join(f"word{i}" for i in range(20))

        with patch(
            "headroom.transforms.kompress_compressor._load_kompress",
            side_effect=RuntimeError("no model"),
        ):
            result = compressor.compress(long_text)
            assert result.compressed == long_text
            assert result.compression_ratio == 1.0


# ── Transform interface ─────────────────────────────────────────────────


class TestKompressTransformInterface:
    def test_apply_short_messages_unchanged(self) -> None:
        """Messages with <10 words should pass through apply() unchanged."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "short"},
        ]
        tokenizer = MagicMock()
        tokenizer.count_text = MagicMock(return_value=5)

        result = compressor.apply(messages, tokenizer)
        assert len(result.messages) == 2
        assert result.messages[0]["content"] == "hello"
        assert result.messages[1]["content"] == "short"

    def test_apply_preserves_user_messages(self) -> None:
        """User messages should never be compressed."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        long_text = " ".join(f"word{i}" for i in range(50))
        messages = [{"role": "user", "content": long_text}]
        tokenizer = MagicMock()
        tokenizer.count_text = MagicMock(return_value=50)

        with patch(
            "headroom.transforms.kompress_compressor._load_kompress",
            side_effect=RuntimeError("should not be called"),
        ):
            result = compressor.apply(messages, tokenizer)
            assert result.messages[0]["content"] == long_text


# ── unload_kompress_model ───────────────────────────────────────────────


class TestUnloadKompressModel:
    def test_unload_when_no_model(self) -> None:
        import headroom.transforms.kompress_compressor as kmod
        from headroom.transforms.kompress_compressor import unload_kompress_model

        # Ensure no model is loaded (previous tests may have set the global)
        kmod._kompress_model = None
        kmod._kompress_tokenizer = None

        # Should return False when no model is loaded
        assert unload_kompress_model() is False
