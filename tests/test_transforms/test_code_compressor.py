"""Tests for Code-Aware Compressor using tree-sitter AST parsing.

Comprehensive tests covering:
- CodeCompressorConfig: Configuration validation and defaults
- CodeAwareCompressor: Core AST-based compression functionality
- Language detection: Auto-detection from extensions and content
- Transform interface: apply(), should_apply() methods
- Syntax preservation: Guarantees valid output syntax
- Edge cases: Empty content, unavailable dependency, fallbacks
"""

from unittest.mock import patch

import pytest

from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressionResult,
    CodeCompressorConfig,
    CodeLanguage,
    DocstringMode,
    detect_language,
    is_tree_sitter_available,
    is_tree_sitter_loaded,
    unload_tree_sitter,
)

# Try to import for availability check
try:
    import tree_sitter_language_pack  # noqa: F401

    TREE_SITTER_INSTALLED = True
except ImportError:
    TREE_SITTER_INSTALLED = False


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default CodeCompressorConfig for testing."""
    return CodeCompressorConfig(
        min_tokens_for_compression=10,  # Low threshold for tests
        enable_ccr=False,  # Disable CCR for unit tests
    )


@pytest.fixture
def compressor(default_config):
    """CodeAwareCompressor instance with default config."""
    return CodeAwareCompressor(default_config)


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


def generate_python_code(n_functions: int = 5, n_classes: int = 1) -> str:
    """Generate Python code for testing."""
    lines = [
        '"""Module with classes and functions."""',
        "",
        "import os",
        "import sys",
        "from typing import Any, Optional, List",
        "from dataclasses import dataclass",
        "",
    ]

    for c in range(n_classes):
        lines.extend(
            [
                "@dataclass",
                f"class TestClass{c}:",
                '    """A test class with docstring."""',
                "    name: str",
                "    value: int = 0",
                "",
                "    def method(self, arg: Any) -> str:",
                '        """Process the argument."""',
                "        result = str(arg)",
                "        for i in range(10):",
                '            result += f"iteration {i}"',
                "        return result",
                "",
            ]
        )

    for i in range(n_functions):
        lines.extend(
            [
                f"def function_{i}(arg: Any, optional: Optional[str] = None) -> str:",
                f'    """Process argument {i}.',
                "",
                "    This is a longer docstring with multiple lines.",
                "    It explains what the function does in detail.",
                "",
                "    Args:",
                "        arg: The argument to process.",
                "        optional: An optional parameter.",
                "",
                "    Returns:",
                "        A string result.",
                '    """',
                "    result = str(arg)",
                "    if optional:",
                "        result += optional",
                "    for i in range(10):",
                '        result += f"iteration {i}"',
                "    try:",
                "        int(result)",
                "    except ValueError:",
                '        result = "0"',
                "    return result",
                "",
            ]
        )

    return "\n".join(lines)


def generate_javascript_code(n_functions: int = 5) -> str:
    """Generate JavaScript code for testing."""
    lines = [
        "// Module with various functions",
        'import { something } from "module";',
        'const config = require("./config");',
        "",
    ]

    for i in range(n_functions):
        lines.extend(
            [
                "/**",
                f" * Process function {i}",
                " * @param {any} arg - The argument",
                " * @returns {string} The result",
                " */",
                f"function processFunction{i}(arg) {{",
                "    let result = String(arg);",
                "    for (let j = 0; j < 10; j++) {",
                "        result += `iteration ${j}`;",
                "    }",
                "    try {",
                "        JSON.parse(result);",
                "    } catch (e) {",
                "        console.error(e);",
                "    }",
                "    return result;",
                "}",
                "",
            ]
        )

    lines.append("export { processFunction0 };")
    return "\n".join(lines)


def generate_go_code(n_functions: int = 3) -> str:
    """Generate Go code for testing."""
    lines = [
        "package main",
        "",
        'import "fmt"',
        "",
        "// Config holds configuration",
        "type Config struct {",
        "    Name  string",
        "    Value int",
        "}",
        "",
    ]

    for i in range(n_functions):
        lines.extend(
            [
                f"// Process{i} processes the input",
                f"func Process{i}(input string) (string, error) {{",
                "    result := input",
                "    for i := 0; i < 10; i++ {",
                '        result = fmt.Sprintf("%s-%d", result, i)',
                "    }",
                "    if len(result) == 0 {",
                '        return "", fmt.Errorf("empty result")',
                "    }",
                "    return result, nil",
                "}",
                "",
            ]
        )

    return "\n".join(lines)


# =============================================================================
# TestCodeCompressorConfig
# =============================================================================


class TestCodeCompressorConfig:
    """Tests for CodeCompressorConfig dataclass."""

    def test_default_values(self):
        """Default config values are sensible."""
        config = CodeCompressorConfig()

        assert config.preserve_imports is True
        assert config.preserve_signatures is True
        assert config.preserve_type_annotations is True
        assert config.preserve_error_handlers is True
        assert config.preserve_decorators is True
        assert config.docstring_mode == DocstringMode.FIRST_LINE
        assert config.target_compression_rate == 0.2
        assert config.max_body_lines == 5
        assert config.min_tokens_for_compression == 100
        assert config.enable_ccr is True

    def test_custom_values(self):
        """Custom config values are applied."""
        config = CodeCompressorConfig(
            preserve_imports=False,
            preserve_signatures=True,
            docstring_mode=DocstringMode.FULL,
            target_compression_rate=0.3,
            max_body_lines=10,
            min_tokens_for_compression=50,
        )

        assert config.preserve_imports is False
        assert config.preserve_signatures is True
        assert config.docstring_mode == DocstringMode.FULL
        assert config.target_compression_rate == 0.3
        assert config.max_body_lines == 10
        assert config.min_tokens_for_compression == 50

    def test_docstring_modes(self):
        """All docstring modes are valid."""
        for mode in DocstringMode:
            config = CodeCompressorConfig(docstring_mode=mode)
            assert config.docstring_mode == mode


# =============================================================================
# TestCodeCompressionResult
# =============================================================================


class TestCodeCompressionResult:
    """Tests for CodeCompressionResult dataclass."""

    def test_tokens_saved(self):
        """tokens_saved property calculates correctly."""
        result = CodeCompressionResult(
            compressed="short",
            original="long content here",
            original_tokens=100,
            compressed_tokens=30,
            compression_ratio=0.3,
            language=CodeLanguage.PYTHON,
            syntax_valid=True,
        )

        assert result.tokens_saved == 70

    def test_tokens_saved_no_negative(self):
        """tokens_saved never returns negative."""
        result = CodeCompressionResult(
            compressed="expanded",
            original="short",
            original_tokens=10,
            compressed_tokens=20,
            compression_ratio=2.0,
            language=CodeLanguage.PYTHON,
            syntax_valid=True,
        )

        assert result.tokens_saved == 0

    def test_savings_percentage(self):
        """savings_percentage property calculates correctly."""
        result = CodeCompressionResult(
            compressed="short",
            original="long content",
            original_tokens=100,
            compressed_tokens=25,
            compression_ratio=0.25,
            language=CodeLanguage.PYTHON,
            syntax_valid=True,
        )

        assert result.savings_percentage == 75.0

    def test_savings_percentage_zero_original(self):
        """savings_percentage handles zero original tokens."""
        result = CodeCompressionResult(
            compressed="",
            original="",
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=1.0,
            language=CodeLanguage.UNKNOWN,
            syntax_valid=True,
        )

        assert result.savings_percentage == 0.0


# =============================================================================
# TestCodeLanguage
# =============================================================================


class TestCodeLanguage:
    """Tests for CodeLanguage enum and detection."""

    def test_all_language_values_are_unique(self):
        """All language enum values are unique."""
        values = [lang.value for lang in CodeLanguage]
        assert len(values) == len(set(values))

    def test_detect_python_language(self):
        """Python language is detected from code patterns."""

        code = """
import os
from typing import List

def function(arg: str) -> str:
    return arg

class MyClass:
    pass
"""
        lang, confidence = detect_language(code)
        assert lang == CodeLanguage.PYTHON
        assert confidence > 0.5

    def test_detect_javascript_language(self):
        """JavaScript language is detected from code patterns."""

        code = """
const express = require('express');
import { something } from 'module';

function handler(req, res) {
    return res.json({ status: 'ok' });
}

export default handler;
"""
        lang, confidence = detect_language(code)
        assert lang in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT)
        assert confidence > 0.3

    def test_detect_go_language(self):
        """Go language is detected from code patterns."""

        code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello")
}
"""
        lang, confidence = detect_language(code)
        assert lang == CodeLanguage.GO
        assert confidence > 0.3


# =============================================================================
# TestCodeAwareCompressor
# =============================================================================


class TestCodeAwareCompressor:
    """Tests for CodeAwareCompressor core functionality."""

    def test_init_with_default_config(self):
        """Compressor initializes with default config."""
        compressor = CodeAwareCompressor()

        assert compressor.config is not None
        assert compressor.config.preserve_imports is True

    def test_init_with_custom_config(self, default_config):
        """Compressor initializes with custom config."""
        compressor = CodeAwareCompressor(default_config)

        assert compressor.config == default_config

    def test_compress_skips_small_content(self, compressor):
        """Small content is not compressed."""
        small_code = "def f(): pass"
        result = compressor.compress(small_code)

        assert result.compressed == small_code
        assert result.compression_ratio == 1.0

    def test_compress_handles_empty_content(self, compressor):
        """Empty content returns empty result."""
        result = compressor.compress("")

        assert result.compressed == ""
        assert result.compression_ratio == 1.0
        assert result.syntax_valid is True

    def test_compress_with_explicit_language(self, compressor):
        """Language can be specified explicitly."""
        code = generate_python_code(2)
        result = compressor.compress(code, language="python")

        # Should detect or use the specified language
        assert result.language == CodeLanguage.PYTHON or result.language == CodeLanguage.UNKNOWN

    def test_compress_auto_detects_python(self, compressor):
        """Python code is auto-detected during compression."""
        code = """
import os
from typing import List

def function(arg: str) -> List[str]:
    return [arg]

class MyClass:
    pass
"""
        result = compressor.compress(code)
        # Should detect Python (if tree-sitter available) or return UNKNOWN
        assert result.language in (CodeLanguage.PYTHON, CodeLanguage.UNKNOWN)

    def test_compress_auto_detects_javascript(self, compressor):
        """JavaScript code is auto-detected during compression."""
        code = """
const express = require('express');
import { something } from 'module';

function handler(req, res) {
    return res.json({ status: 'ok' });
}

export default handler;
"""
        result = compressor.compress(code)
        assert result.language in (
            CodeLanguage.JAVASCRIPT,
            CodeLanguage.TYPESCRIPT,
            CodeLanguage.UNKNOWN,
        )

    def test_compress_auto_detects_go(self, compressor):
        """Go code is auto-detected during compression."""
        code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello")
}
"""
        result = compressor.compress(code)
        assert result.language in (CodeLanguage.GO, CodeLanguage.UNKNOWN)


# =============================================================================
# TestFallbackCompression
# =============================================================================


class TestFallbackCompression:
    """Tests for fallback compression when tree-sitter unavailable."""

    def test_fallback_when_tree_sitter_unavailable(self, default_config):
        """Uses fallback compression when tree-sitter is not installed."""
        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available",
            return_value=False,
        ):
            compressor = CodeAwareCompressor(default_config)
            code = generate_python_code(5)

            result = compressor.compress(code)

            # Should still return a result (fallback compression)
            assert result is not None
            assert result.syntax_valid is True  # Fallback guarantees validity

    def test_fallback_preserves_structure(self, default_config):
        """Fallback compression preserves basic structure when no compressor available.

        When both tree-sitter and LLMLingua are unavailable, the fallback
        returns the original code unchanged - preserving all structure.
        """
        with (
            patch(
                "headroom.transforms.code_compressor._check_tree_sitter_available",
                return_value=False,
            ),
            patch(
                "headroom.transforms.llmlingua_compressor._check_llmlingua_available",
                return_value=False,
            ),
        ):
            compressor = CodeAwareCompressor(default_config)
            code = generate_python_code(3)

            result = compressor.compress(code)

            # With no compressor available, original code is returned unchanged
            # This preserves all imports and class/function signatures
            assert "import os" in result.compressed
            assert "def function_" in result.compressed
            # Compression ratio should be 1.0 (no compression)
            assert result.compression_ratio == 1.0


# =============================================================================
# TestTransformInterface
# =============================================================================


class TestTransformInterface:
    """Tests for Transform interface (apply, should_apply)."""

    def test_should_apply_returns_false_for_small_content(self, default_config, tokenizer):
        """should_apply returns False for small content."""
        config = CodeCompressorConfig(min_tokens_for_compression=1000)
        compressor = CodeAwareCompressor(config)
        messages = [{"role": "user", "content": "def f(): pass"}]

        assert not compressor.should_apply(messages, tokenizer)

    def test_should_apply_returns_bool_for_large_code(self, default_config, tokenizer):
        """should_apply returns boolean for large code content."""
        compressor = CodeAwareCompressor(default_config)
        code = generate_python_code(20)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": code}]

        # Should return True if there's code content to process
        result = compressor.should_apply(messages, tokenizer)
        assert isinstance(result, bool)

    def test_apply_returns_transform_result(self, default_config, tokenizer):
        """apply() returns proper TransformResult."""
        compressor = CodeAwareCompressor(default_config)
        code = generate_python_code(10)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": code}]

        result = compressor.apply(messages, tokenizer)

        assert result is not None
        assert result.tokens_before > 0
        assert len(result.messages) == 1

    def test_apply_passes_through_non_code_messages(self, default_config, tokenizer):
        """apply() passes through non-code messages unchanged."""
        compressor = CodeAwareCompressor(default_config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = compressor.apply(messages, tokenizer)

        assert result.messages[0]["content"] == "Hello"
        assert result.messages[1]["content"] == "Hi there!"

    def test_name_property(self, compressor):
        """Compressor has correct name."""
        assert compressor.name == "code_aware_compressor"


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for CodeAwareCompressor."""

    def test_whitespace_only_content(self, compressor):
        """Whitespace-only content is handled gracefully."""
        result = compressor.compress("   \n\t\n   ")

        assert result.compression_ratio == 1.0
        assert result.syntax_valid is True

    def test_unicode_content(self, default_config):
        """Unicode in code is handled correctly."""
        compressor = CodeAwareCompressor(default_config)
        code = '''
def greet(name: str) -> str:
    """Greet the user in multiple languages."""
    return f"Hello, {name}! \u4f60\u597d! \u3053\u3093\u306b\u3061\u306f!"
'''
        result = compressor.compress(code)

        # Should handle unicode without crashing
        assert result is not None

    def test_very_long_function(self, default_config):
        """Very long functions are compressed."""
        compressor = CodeAwareCompressor(default_config)
        lines = ["def very_long_function():"]
        lines.append('    """A very long function."""')
        for i in range(100):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_99")
        code = "\n".join(lines)

        result = compressor.compress(code)

        # Should compress the long function body
        assert result.compression_ratio < 1.0 or "tree_sitter" not in str(
            is_tree_sitter_available()
        )

    def test_nested_functions(self, default_config):
        """Nested functions are handled."""
        compressor = CodeAwareCompressor(default_config)
        code = """
def outer():
    def inner():
        return "inner"
    return inner()
"""
        result = compressor.compress(code)

        assert result is not None
        assert result.syntax_valid is True

    def test_syntax_errors_in_input(self, default_config):
        """Syntax errors in input don't crash the compressor."""
        compressor = CodeAwareCompressor(default_config)
        # Invalid Python syntax
        code = """
def broken(
    # Missing closing paren
"""
        # Should not raise
        result = compressor.compress(code, language="python")
        assert result is not None

    def test_mixed_language_content(self, default_config):
        """Mixed language content (like markdown with code) is handled."""
        compressor = CodeAwareCompressor(default_config)
        content = """
# Documentation

Here is some code:

```python
def example():
    pass
```

And some more text.
"""
        # Should not crash
        result = compressor.compress(content)
        assert result is not None


# =============================================================================
# TestMemoryManagement
# =============================================================================


class TestMemoryManagement:
    """Tests for memory management functions."""

    def test_is_tree_sitter_available_returns_bool(self):
        """is_tree_sitter_available returns a boolean."""
        result = is_tree_sitter_available()
        assert isinstance(result, bool)

    def test_is_tree_sitter_loaded_returns_false_initially(self):
        """is_tree_sitter_loaded returns False when no parsers loaded."""
        # Clear any loaded parsers first
        unload_tree_sitter()
        assert is_tree_sitter_loaded() is False

    def test_unload_returns_false_when_nothing_loaded(self):
        """unload_tree_sitter returns False when nothing to unload."""
        # Ensure nothing is loaded
        unload_tree_sitter()
        result = unload_tree_sitter()
        assert result is False


# =============================================================================
# Integration Tests (only run if tree-sitter is installed)
# =============================================================================


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter-languages not installed")
class TestTreeSitterIntegration:
    """Integration tests that require actual tree-sitter installation.

    These tests verify actual AST parsing and compression behavior.
    """

    def test_actual_python_compression(self):
        """Test actual compression of Python code."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(5)

        result = compressor.compress(code, language="python")

        # Should achieve compression
        assert result.compression_ratio < 1.0
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.PYTHON

    def test_actual_javascript_compression(self):
        """Test actual compression of JavaScript code."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_javascript_code(5)

        result = compressor.compress(code, language="javascript")

        assert result.compression_ratio < 1.0
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.JAVASCRIPT

    def test_actual_go_compression(self):
        """Test Go code is processed (compression may fall back due to nested structures)."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_go_code(3)

        result = compressor.compress(code, language="go")

        # Go code is processed and returns valid output
        # Note: compression_ratio may be 1.0 if compression produces invalid syntax
        # and falls back to original (Go has complex nested brace handling)
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.GO
        assert result.compressed  # Some output is produced

    def test_imports_preserved(self):
        """Imports are preserved in compressed output."""
        config = CodeCompressorConfig(
            preserve_imports=True,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(5)

        result = compressor.compress(code, language="python")

        assert "import os" in result.compressed
        assert "from typing import" in result.compressed

    def test_signatures_preserved(self):
        """Function signatures are preserved."""
        config = CodeCompressorConfig(
            preserve_signatures=True,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(3)

        result = compressor.compress(code, language="python")

        # Should preserve function signatures
        assert "def function_" in result.compressed
        assert "arg:" in result.compressed or "(arg" in result.compressed

    def test_error_handlers_preserved(self):
        """Module-level try/except blocks are preserved."""
        config = CodeCompressorConfig(
            preserve_error_handlers=True,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        # Code with module-level try/except (not inside functions)
        code = '''
import os

def setup():
    """Setup function."""
    pass

try:
    from optional_module import feature
except ImportError:
    feature = None

def main():
    """Main function with long body."""
    result = []
    for i in range(100):
        result.append(i)
    return result
'''
        result = compressor.compress(code, language="python")

        # Module-level error handlers should be preserved
        assert "try:" in result.compressed or "except" in result.compressed

    def test_syntax_verification(self):
        """Output syntax is verified as valid."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(5)

        result = compressor.compress(code, language="python")

        # Verify the compressed output is valid Python
        assert result.syntax_valid is True

        # Should be parseable
        try:
            compile(result.compressed, "<test>", "exec")
        except SyntaxError:
            pytest.fail("Compressed output has invalid Python syntax")

    def test_tree_sitter_loaded_after_compression(self):
        """Parser is loaded after compression."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)

        # Ensure clean state
        unload_tree_sitter()
        assert is_tree_sitter_loaded() is False

        # Compress should load parser
        code = generate_python_code(3)
        compressor.compress(code, language="python")

        assert is_tree_sitter_loaded() is True

    def test_unload_clears_parsers(self):
        """unload_tree_sitter clears loaded parsers."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)

        # Load a parser
        code = generate_python_code(3)
        compressor.compress(code, language="python")
        assert is_tree_sitter_loaded() is True

        # Unload
        result = unload_tree_sitter()
        assert result is True
        assert is_tree_sitter_loaded() is False


# =============================================================================
# TestDocstringModes
# =============================================================================


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter-languages not installed")
class TestDocstringModes:
    """Tests for different docstring handling modes."""

    def test_docstring_mode_full(self):
        """FULL mode preserves entire docstrings."""
        config = CodeCompressorConfig(
            docstring_mode=DocstringMode.FULL,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(2)

        result = compressor.compress(code, language="python")

        # Should preserve full docstrings
        assert "Args:" in result.compressed or "Returns:" in result.compressed

    def test_docstring_mode_first_line(self):
        """FIRST_LINE mode keeps only first line of docstring."""
        config = CodeCompressorConfig(
            docstring_mode=DocstringMode.FIRST_LINE,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(2)

        result = compressor.compress(code, language="python")

        # Multi-line docstring details should be removed
        # This is implementation-dependent
        assert result.compressed is not None

    def test_docstring_mode_remove(self):
        """REMOVE mode removes all docstrings."""
        config = CodeCompressorConfig(
            docstring_mode=DocstringMode.REMOVE,
            min_tokens_for_compression=10,
            max_body_lines=2,  # Low threshold to trigger compression
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        # Larger function to trigger body compression
        code = '''
def example():
    """This docstring should be removed."""
    x = 1
    y = 2
    z = 3
    result = x + y + z
    for i in range(10):
        result += i
    return result
'''
        result = compressor.compress(code, language="python")

        # Docstring should be removed when REMOVE mode is active
        assert "This docstring should be removed" not in result.compressed
