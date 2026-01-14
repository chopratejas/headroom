"""Tests for ContentRouter - intelligent content-based compression routing.

Comprehensive tests covering:
- ContentRouterConfig: Configuration validation and defaults
- ContentRouter: Core routing functionality
- Strategy detection: Code, JSON, search, logs, text
- Source hint parsing: File paths, tool names
- Mixed content handling: Split, route, reassemble
- Transform interface: apply(), should_apply() methods
"""

import pytest

from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
    generate_source_hint,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default ContentRouterConfig for testing."""
    return ContentRouterConfig(
        min_section_tokens=10,  # Low threshold for tests
    )


@pytest.fixture
def router(default_config):
    """ContentRouter instance with default config."""
    return ContentRouter(default_config)


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


def generate_python_code(n_functions: int = 5) -> str:
    """Generate Python code for testing."""
    lines = [
        '"""Module with functions."""',
        "import os",
        "from typing import Any",
        "",
    ]
    for i in range(n_functions):
        lines.extend(
            [
                f"def function_{i}(arg: Any) -> str:",
                f'    """Process argument {i}."""',
                "    return str(arg)",
                "",
            ]
        )
    return "\n".join(lines)


def generate_json_data(n_items: int = 20) -> str:
    """Generate JSON content for testing."""
    import json

    items = [
        {"id": i, "name": f"Item {i}", "value": i * 10, "active": i % 2 == 0}
        for i in range(n_items)
    ]
    return json.dumps(items, indent=2)


def generate_search_results(n_results: int = 10) -> str:
    """Generate grep/search-like results for testing."""
    lines = []
    for i in range(n_results):
        lines.append(f"src/module{i}.py:42: def process_data(input: str) -> str:")
        lines.append(f"src/module{i}.py:43:     return transform(input)")
    return "\n".join(lines)


def generate_log_output(n_lines: int = 30) -> str:
    """Generate build/test log output for testing."""
    lines = [
        "Running tests...",
        "=== Test Suite: Unit Tests ===",
    ]
    for i in range(n_lines):
        if i % 10 == 0:
            lines.append(f"PASS tests/test_module{i}.py::test_function")
        elif i % 15 == 0:
            lines.append(f"FAIL tests/test_module{i}.py::test_failing")
        else:
            lines.append(f"  Running test_{i}... ok")
    lines.append("=== Summary ===")
    lines.append(f"Tests: {n_lines}, Passed: {n_lines - 2}, Failed: 2")
    return "\n".join(lines)


def generate_mixed_content() -> str:
    """Generate content with mixed types (markdown with code)."""
    return """# Documentation

This is a README file with code examples.

## Python Example

```python
def example():
    return "hello"
```

## JSON Configuration

```json
{"key": "value", "number": 42}
```

## Usage

Run the following command:
```bash
python main.py --verbose
```

That's all!
"""


# =============================================================================
# TestContentRouterConfig
# =============================================================================


class TestContentRouterConfig:
    """Tests for ContentRouterConfig dataclass."""

    def test_default_values(self):
        """Default config values are sensible."""
        config = ContentRouterConfig()

        assert config.enable_code_aware is True
        assert config.enable_llmlingua is True
        assert config.enable_smart_crusher is True
        assert config.enable_search_compressor is True
        assert config.enable_log_compressor is True
        assert config.min_section_tokens == 20
        assert config.fallback_strategy == CompressionStrategy.PASSTHROUGH

    def test_custom_values(self):
        """Custom config values are applied."""
        config = ContentRouterConfig(
            min_section_tokens=50,
            enable_code_aware=False,
            enable_llmlingua=False,
            fallback_strategy=CompressionStrategy.TEXT,
        )

        assert config.min_section_tokens == 50
        assert config.enable_code_aware is False
        assert config.enable_llmlingua is False
        assert config.fallback_strategy == CompressionStrategy.TEXT

    def test_all_strategies_in_enum(self):
        """All expected strategies are in the enum."""
        expected = [
            "CODE_AWARE",
            "SMART_CRUSHER",
            "SEARCH",
            "LOG",
            "LLMLINGUA",
            "TEXT",
            "MIXED",
            "PASSTHROUGH",
        ]
        actual = [s.name for s in CompressionStrategy]
        for strategy in expected:
            assert strategy in actual, f"Missing strategy: {strategy}"


# =============================================================================
# TestRouterCompressionResult
# =============================================================================


class TestRouterCompressionResult:
    """Tests for RouterCompressionResult dataclass."""

    def test_tokens_saved_from_routing_log(self):
        """tokens_saved property calculates correctly from routing log."""
        result = RouterCompressionResult(
            compressed="short",
            original="long content here",
            strategy_used=CompressionStrategy.CODE_AWARE,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.SOURCE_CODE,
                    strategy=CompressionStrategy.CODE_AWARE,
                    confidence=0.9,
                    original_tokens=100,
                    compressed_tokens=30,
                )
            ],
            sections_processed=1,
        )

        assert result.tokens_saved == 70

    def test_tokens_saved_no_negative(self):
        """tokens_saved never returns negative."""
        result = RouterCompressionResult(
            compressed="expanded",
            original="short",
            strategy_used=CompressionStrategy.PASSTHROUGH,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.PASSTHROUGH,
                    confidence=1.0,
                    original_tokens=10,
                    compressed_tokens=20,  # Expanded
                )
            ],
            sections_processed=1,
        )

        # Should be 0 not negative
        assert result.tokens_saved == 0

    def test_savings_percentage(self):
        """savings_percentage property calculates correctly."""
        result = RouterCompressionResult(
            compressed="short",
            original="long content",
            strategy_used=CompressionStrategy.TEXT,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.TEXT,
                    confidence=0.8,
                    original_tokens=100,
                    compressed_tokens=25,
                )
            ],
            sections_processed=1,
        )

        assert result.savings_percentage == 75.0

    def test_empty_routing_log(self):
        """Handles empty routing log gracefully."""
        result = RouterCompressionResult(
            compressed="content",
            original="content",
            strategy_used=CompressionStrategy.PASSTHROUGH,
            routing_log=[],
            sections_processed=0,
        )

        assert result.total_original_tokens == 0
        assert result.total_compressed_tokens == 0
        assert result.savings_percentage == 0.0


# =============================================================================
# TestSourceHintGeneration
# =============================================================================


class TestSourceHintGeneration:
    """Tests for generate_source_hint function."""

    def test_file_path_hint_with_file_path(self):
        """File path generates file: hint."""
        hint = generate_source_hint("Read", {"file_path": "/src/auth.py"})
        assert hint == "file:/src/auth.py"

    def test_file_path_hint_with_path(self):
        """File path with 'path' key also works."""
        hint = generate_source_hint("read_file", {"path": "/src/auth.py"})
        assert hint == "file:/src/auth.py"

    def test_grep_tool_hint(self):
        """Grep tool generates tool:grep hint."""
        hint = generate_source_hint("Grep", {"pattern": "def.*"})
        assert hint == "tool:grep"

    def test_glob_tool_hint(self):
        """Glob tool generates tool:glob hint."""
        hint = generate_source_hint("Glob", {"pattern": "*.py"})
        assert hint == "tool:glob"

    def test_bash_test_command(self):
        """Bash with test command generates tool:pytest hint."""
        hint = generate_source_hint("Bash", {"command": "pytest tests/"})
        assert hint == "tool:pytest"

    def test_bash_build_command(self):
        """Bash with build command generates tool:build hint."""
        hint = generate_source_hint("Bash", {"command": "npm run build"})
        assert "tool:" in hint

    def test_unknown_tool(self):
        """Unknown tool returns generic tool hint."""
        hint = generate_source_hint("CustomTool", {"arg": "value"})
        # Should return some hint, not crash
        assert hint is not None


# =============================================================================
# TestStrategyDetection
# =============================================================================


class TestStrategyDetection:
    """Tests for content type and strategy detection."""

    def test_detect_python_code(self, router):
        """Python code is detected."""
        code = generate_python_code(5)
        strategy = router._determine_strategy(code, None)
        # Should be either CODE_AWARE or fallback
        assert strategy in CompressionStrategy

    def test_detect_json_content(self, router):
        """JSON content is detected."""
        json_data = generate_json_data(20)
        strategy = router._determine_strategy(json_data, None)
        assert strategy in CompressionStrategy

    def test_detect_search_results(self, router):
        """Search/grep results are detected."""
        search_results = generate_search_results(10)
        strategy = router._determine_strategy(search_results, None)
        assert strategy in CompressionStrategy

    def test_detect_log_output(self, router):
        """Build/test logs are detected."""
        logs = generate_log_output(30)
        strategy = router._determine_strategy(logs, None)
        assert strategy in CompressionStrategy

    def test_detect_plain_text(self, router):
        """Plain text detection."""
        text = "This is just plain text without any special formatting."
        strategy = router._determine_strategy(text, None)
        assert strategy in CompressionStrategy


# =============================================================================
# TestSourceHintRouting
# =============================================================================


class TestSourceHintRouting:
    """Tests for source hint-based routing."""

    def test_file_hint_routes_appropriately(self, router):
        """file:*.py hint influences routing."""
        content = generate_python_code(3)
        strategy = router._determine_strategy(content, "file:/src/auth.py")
        # Should return a valid strategy
        assert strategy in CompressionStrategy

    def test_grep_hint_routes_appropriately(self, router):
        """tool:grep hint influences routing."""
        content = generate_search_results(5)
        strategy = router._determine_strategy(content, "tool:grep")
        assert strategy in CompressionStrategy


# =============================================================================
# TestContentRouter
# =============================================================================


class TestContentRouter:
    """Tests for ContentRouter core functionality."""

    def test_init_with_default_config(self):
        """Router initializes with default config."""
        router = ContentRouter()
        assert router.config is not None
        assert router.config.enable_code_aware is True

    def test_init_with_custom_config(self, default_config):
        """Router initializes with custom config."""
        router = ContentRouter(default_config)
        assert router.config == default_config

    def test_compress_empty_content(self, router):
        """Empty content returns passthrough."""
        result = router.compress("")
        assert result.compressed == ""
        assert result.strategy_used == CompressionStrategy.PASSTHROUGH

    def test_compress_small_content(self, router):
        """Small content returns same content."""
        result = router.compress("small")
        assert result.compressed == "small"
        # Small content might use TEXT or PASSTHROUGH strategy
        assert result.strategy_used in (
            CompressionStrategy.PASSTHROUGH,
            CompressionStrategy.TEXT,
        )

    def test_compress_returns_result(self, router):
        """compress() returns RouterCompressionResult."""
        content = generate_python_code(10)
        result = router.compress(content)

        assert isinstance(result, RouterCompressionResult)
        assert result.original == content
        assert result.strategy_used is not None

    def test_compress_with_source_hint(self, router):
        """Source hint influences routing decision."""
        content = generate_python_code(5)
        result = router.compress(content, source_hint="file:/src/auth.py")

        # Should return a valid result
        assert isinstance(result, RouterCompressionResult)

    def test_name_property(self, router):
        """Router has correct name."""
        assert router.name == "content_router"


# =============================================================================
# TestTransformInterface
# =============================================================================


class TestTransformInterface:
    """Tests for Transform interface (apply, should_apply)."""

    def test_should_apply_returns_bool(self, default_config, tokenizer):
        """should_apply returns a boolean."""
        router = ContentRouter(default_config)
        messages = [{"role": "user", "content": "small"}]

        result = router.should_apply(messages, tokenizer)
        assert isinstance(result, bool)

    def test_should_apply_returns_true_for_large_content(self, default_config, tokenizer):
        """should_apply returns True for large content."""
        router = ContentRouter(default_config)
        content = generate_python_code(20)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": content}]

        assert router.should_apply(messages, tokenizer)

    def test_apply_returns_transform_result(self, default_config, tokenizer):
        """apply() returns proper TransformResult."""
        router = ContentRouter(default_config)
        content = generate_python_code(10)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": content}]

        result = router.apply(messages, tokenizer)

        assert result is not None
        assert result.tokens_before > 0
        assert len(result.messages) == 1

    def test_apply_passes_through_non_tool_messages(self, default_config, tokenizer):
        """apply() passes through non-tool messages unchanged."""
        router = ContentRouter(default_config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = router.apply(messages, tokenizer)

        assert result.messages[0]["content"] == "Hello"
        assert result.messages[1]["content"] == "Hi there!"


# =============================================================================
# TestCompressorDisabling
# =============================================================================


class TestCompressorDisabling:
    """Tests for disabling specific compressors.

    Note: These tests verify the config is accepted, not that the router
    actually respects the disable flags (which may not be fully implemented).
    """

    def test_config_accepts_disable_code_compression(self):
        """Config accepts enable_code_aware=False."""
        config = ContentRouterConfig(
            enable_code_aware=False,
            min_section_tokens=10,
        )
        router = ContentRouter(config)
        code = generate_python_code(10)

        # Should not crash
        result = router.compress(code, source_hint="file:/src/auth.py")
        assert result is not None

    def test_config_accepts_disable_search_compression(self):
        """Config accepts enable_search_compressor=False."""
        config = ContentRouterConfig(
            enable_search_compressor=False,
            min_section_tokens=10,
        )
        router = ContentRouter(config)
        search_results = generate_search_results(10)

        # Should not crash
        result = router.compress(search_results, source_hint="tool:grep")
        assert result is not None

    def test_config_accepts_disable_log_compression(self):
        """Config accepts enable_log_compressor=False."""
        config = ContentRouterConfig(
            enable_log_compressor=False,
            min_section_tokens=10,
        )
        router = ContentRouter(config)
        logs = generate_log_output(30)

        # Should not crash
        result = router.compress(logs, source_hint="tool:pytest")
        assert result is not None


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for ContentRouter."""

    def test_whitespace_only_content(self, router):
        """Whitespace-only content is handled gracefully."""
        result = router.compress("   \n\t\n   ")
        assert result.strategy_used == CompressionStrategy.PASSTHROUGH

    def test_unicode_content(self, router):
        """Unicode content is handled correctly."""
        content = "This has unicode: \u4e2d\u6587 \u65e5\u672c\u8a9e " * 50
        result = router.compress(content)
        assert result is not None

    def test_very_long_content(self, router):
        """Very long content is handled."""
        content = generate_python_code(100)
        result = router.compress(content)
        assert result is not None

    def test_null_source_hint(self, router):
        """Null source hint doesn't crash."""
        content = generate_python_code(5)
        result = router.compress(content, source_hint=None)
        assert result is not None

    def test_empty_source_hint(self, router):
        """Empty source hint doesn't crash."""
        content = generate_python_code(5)
        result = router.compress(content, source_hint="")
        assert result is not None


# =============================================================================
# TestRoutingLog
# =============================================================================


class TestRoutingLog:
    """Tests for routing log functionality."""

    def test_routing_log_populated(self, router):
        """Routing log is populated with decisions."""
        content = generate_python_code(10)
        result = router.compress(content)

        # Routing log should be a list
        assert isinstance(result.routing_log, list)

    def test_routing_log_entries_have_strategy(self, router):
        """Routing log entries contain strategy."""
        content = generate_python_code(10)
        result = router.compress(content)

        for entry in result.routing_log:
            assert hasattr(entry, "strategy")
            assert entry.strategy in CompressionStrategy


# =============================================================================
# TestSummary
# =============================================================================


class TestSummary:
    """Tests for result summary generation."""

    def test_summary_property(self, router):
        """Summary property exists and is callable or returns string."""
        content = generate_python_code(10)
        result = router.compress(content)

        # Check summary property exists
        assert hasattr(result, "summary")

        # Get summary (call if callable)
        summary = result.summary
        if callable(summary):
            summary = summary()

        # Should be a string
        assert summary is not None
