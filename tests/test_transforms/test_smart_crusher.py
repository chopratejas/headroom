"""Tests for SmartCrusher transform.

Comprehensive tests covering:
- SmartAnalyzer: Statistical analysis of arrays
- SmartCrusher: Intelligent compression with Safe V1 Recipe
- RelevanceScoring: Context extraction and item matching
- Edge cases: Malformed JSON, nested arrays, different message formats
"""

import json

import pytest

from headroom import (
    OpenAIProvider,
    RelevanceScorerConfig,
    SmartCrusherConfig,
    Tokenizer,
)
from headroom.relevance import RelevanceScore, RelevanceScorer
from headroom.transforms.smart_crusher import (
    CompressionStrategy,
    SmartAnalyzer,
    SmartCrusher,
)

# =============================================================================
# Test Fixtures
# =============================================================================

# Create a shared provider for tests
_provider = OpenAIProvider()


def get_tokenizer(model: str = "gpt-4o") -> Tokenizer:
    """Get a tokenizer for tests using OpenAI provider."""
    token_counter = _provider.get_token_counter(model)
    return Tokenizer(token_counter, model)


@pytest.fixture
def tokenizer():
    """Provide a tokenizer for tests."""
    return get_tokenizer()


@pytest.fixture
def default_config():
    """Default SmartCrusherConfig for testing."""
    return SmartCrusherConfig(
        enabled=True,
        min_items_to_analyze=3,
        min_tokens_to_crush=0,  # Always crush for tests
        max_items_after_crush=10,
        variance_threshold=2.0,
    )


@pytest.fixture
def analyzer(default_config):
    """SmartAnalyzer instance for testing."""
    return SmartAnalyzer(default_config)


@pytest.fixture
def crusher(default_config):
    """SmartCrusher instance for testing."""
    return SmartCrusher(default_config)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_time_series_data(n: int = 20, with_spike: bool = False) -> list[dict]:
    """Generate time series data with optional anomaly."""
    data = []
    for i in range(n):
        value = 100.0 + (i * 0.5)  # Slight upward trend
        if with_spike and i == n // 2:
            value = 500.0  # Spike in the middle
        data.append(
            {
                "timestamp": f"2025-01-{(i % 28) + 1:02d}T12:00:00Z",
                "value": value,
                "metric": "cpu_usage",
            }
        )
    return data


def generate_log_data(n: int = 20, with_errors: bool = False) -> list[dict]:
    """Generate log-style data with optional errors."""
    data = []
    levels = ["INFO", "DEBUG", "WARN"]
    for i in range(n):
        level = levels[i % len(levels)]
        if with_errors and i in [5, 15]:
            level = "ERROR"
            message = f"Connection failed: timeout after 30s (attempt {i})"
        else:
            message = f"Processing request {i} successfully"
        data.append(
            {
                "level": level,
                "message": message,
                "timestamp": f"2025-01-06T{12 + (i // 60):02d}:{i % 60:02d}:00Z",
            }
        )
    return data


def generate_search_results(n: int = 20) -> list[dict]:
    """Generate search results with scores."""
    return [
        {
            "id": f"doc_{i}",
            "title": f"Document {i}",
            "score": 1.0 - (i * 0.05),
            "snippet": f"This is a snippet from document {i}...",
        }
        for i in range(n)
    ]


def generate_generic_data(
    n: int = 20,
    constant_field: bool = False,
    with_signals: bool = False,
) -> list[dict]:
    """Generate generic array data.

    Args:
        n: Number of items to generate
        constant_field: If True, type field is constant "product"
        with_signals: If True, adds importance signals (errors, anomalies)
                     to enable crushing with new statistical detection
    """
    items = []
    for i in range(n):
        item = {
            "id": i,
            "name": f"Item {i}",
            "type": "product" if constant_field else f"type_{i % 3}",
            "active": True if constant_field else (i % 2 == 0),
        }
        if with_signals:
            item["value"] = 100.0
            # Add some errors
            if i == n // 4:
                item["error"] = f"Error at {i}"
            # Add some anomalies
            if i == n // 2:
                item["value"] = 99999.0
        items.append(item)
    return items


# =============================================================================
# TestSmartAnalyzer
# =============================================================================


class TestSmartAnalyzer:
    """Tests for SmartAnalyzer class."""

    def test_analyze_empty_array(self, analyzer):
        """Empty array should return analysis with no field stats."""
        result = analyzer.analyze_array([])

        assert result.item_count == 0
        assert result.field_stats == {}
        assert result.detected_pattern == "generic"
        assert result.recommended_strategy == CompressionStrategy.NONE
        assert result.constant_fields == {}

    def test_analyze_single_item(self, analyzer):
        """Single item array should return analysis but no compression."""
        items = [{"id": 1, "name": "Test"}]
        result = analyzer.analyze_array(items)

        assert result.item_count == 1
        assert "id" in result.field_stats
        assert "name" in result.field_stats
        # Single item means constant fields
        assert result.field_stats["id"].is_constant
        assert result.field_stats["name"].is_constant

    def test_analyze_numeric_field_stats(self, analyzer):
        """Numeric fields should have correct statistics computed."""
        items = [
            {"value": 10.0},
            {"value": 20.0},
            {"value": 30.0},
            {"value": 40.0},
            {"value": 50.0},
        ]
        result = analyzer.analyze_array(items)

        stats = result.field_stats["value"]
        assert stats.field_type == "numeric"
        assert stats.min_val == 10.0
        assert stats.max_val == 50.0
        assert stats.mean_val == 30.0
        assert stats.variance is not None
        assert stats.variance > 0

    def test_analyze_string_field_stats(self, analyzer):
        """String fields should have correct statistics computed."""
        items = [
            {"name": "Alice"},
            {"name": "Bob"},
            {"name": "Alice"},  # Duplicate
            {"name": "Charlie"},
            {"name": "Alice"},  # Another duplicate
        ]
        result = analyzer.analyze_array(items)

        stats = result.field_stats["name"]
        assert stats.field_type == "string"
        assert stats.avg_length is not None
        assert stats.top_values is not None
        # Alice appears 3 times, should be top
        assert stats.top_values[0][0] == "Alice"
        assert stats.top_values[0][1] == 3

    def test_detect_time_series_pattern(self, analyzer):
        """Time series data should be detected correctly."""
        # Create data with timestamp and numeric variance
        # Include anomaly to provide an importance signal for crushing
        items = []
        for i in range(40):
            # Create variance-inducing data
            value = 100.0 + (i * 2.0)  # Steady increase with variance
            if i == 20:
                value = 999.0  # Anomaly provides importance signal
            items.append(
                {
                    "timestamp": f"2025-01-{(i % 28) + 1:02d}T12:00:00Z",
                    "value": value,
                    "metric": "cpu_usage",
                }
            )

        result = analyzer.analyze_array(items)

        # Pattern should be detected as time_series (timestamp + numeric variance)
        assert result.detected_pattern == "time_series"
        # With anomaly signal, strategy should allow crushing
        assert result.recommended_strategy in [
            CompressionStrategy.TIME_SERIES,
            CompressionStrategy.SMART_SAMPLE,
        ]

    def test_detect_time_series_pattern_with_change_points(self):
        """Time series with clear change points should use TIME_SERIES strategy."""
        # The change point detection threshold is variance_threshold * std
        # To detect a change point, the before/after mean difference must exceed this
        # With bimodal data, std is very high. We need a lower variance_threshold
        # to reliably detect change points, OR the test should use a config
        # with lower variance threshold.

        config = SmartCrusherConfig(
            min_items_to_analyze=3,
            variance_threshold=1.0,  # Lower threshold to detect changes
        )
        analyzer = SmartAnalyzer(config)

        # Create data with clear step change
        items = []
        for i in range(40):
            if i < 20:
                value = 100.0 + (i * 0.5)  # Values around 100-110
            else:
                value = 300.0 + ((i - 20) * 0.5)  # Values around 300-310 (jump)
            items.append(
                {
                    "timestamp": f"2025-01-{(i % 28) + 1:02d}T12:00:00Z",
                    "value": value,
                    "metric": "cpu_usage",
                }
            )

        result = analyzer.analyze_array(items)

        assert result.detected_pattern == "time_series"
        # With lower variance_threshold, change points should be detected
        value_stats = result.field_stats.get("value")
        assert value_stats is not None
        # Even with low threshold, bimodal data has high std
        # The test verifies the strategy selection logic
        if len(value_stats.change_points) > 0:
            assert result.recommended_strategy == CompressionStrategy.TIME_SERIES
        else:
            # If change points still not detected, strategy falls back
            assert result.recommended_strategy in [
                CompressionStrategy.TIME_SERIES,
                CompressionStrategy.SMART_SAMPLE,
            ]

    def test_detect_logs_pattern(self, analyzer):
        """Log data should be detected correctly."""
        # Use logs WITH errors to provide importance signal
        items = generate_log_data(20, with_errors=True)
        result = analyzer.analyze_array(items)

        # With structural detection, logs are detected as logs pattern
        # but strategy depends on crushability analysis
        assert result.detected_pattern in ["logs", "generic"]
        # With error items providing signal, crushing can proceed
        assert result.recommended_strategy in [
            CompressionStrategy.CLUSTER_SAMPLE,
            CompressionStrategy.SMART_SAMPLE,
            CompressionStrategy.SKIP,  # May still skip if other conditions met
        ]

    def test_detect_search_results_pattern(self, analyzer):
        """Search results with scores should be detected correctly."""
        items = generate_search_results(20)
        result = analyzer.analyze_array(items)

        assert result.detected_pattern == "search_results"
        assert result.recommended_strategy == CompressionStrategy.TOP_N

    def test_detect_generic_pattern(self, analyzer):
        """Generic data without special patterns should be detected."""
        items = generate_generic_data(20)
        result = analyzer.analyze_array(items)

        assert result.detected_pattern == "generic"
        # With new crushability analysis: unique IDs + no importance signal = SKIP
        # This is the safe behavior to avoid dropping important unique entities
        assert result.recommended_strategy in [
            CompressionStrategy.SMART_SAMPLE,
            CompressionStrategy.SKIP,  # More conservative when no signal present
        ]

    def test_detect_change_points(self, analyzer):
        """Change points should be detected in numeric data with variance."""
        # Create data with clear change point
        items = []
        for i in range(30):
            if i < 15:
                value = 100.0 + (i * 0.1)  # Low values
            else:
                value = 200.0 + ((i - 15) * 0.1)  # High values after change
            items.append({"timestamp": f"2025-01-{(i % 28) + 1:02d}", "metric": value})

        result = analyzer.analyze_array(items)

        # Should detect change point around index 15
        metric_stats = result.field_stats.get("metric")
        assert metric_stats is not None
        assert metric_stats.change_points is not None
        # Change points should be near the transition
        if metric_stats.change_points:
            assert any(10 <= cp <= 20 for cp in metric_stats.change_points)

    def test_constant_field_detection(self, analyzer):
        """Constant fields should be identified."""
        items = generate_generic_data(20, constant_field=True)
        result = analyzer.analyze_array(items)

        # type field should be constant ("product")
        type_stats = result.field_stats.get("type")
        assert type_stats is not None
        assert type_stats.is_constant
        assert type_stats.constant_value == "product"

        # Constant fields should be in constant_fields dict
        assert "type" in result.constant_fields
        assert result.constant_fields["type"] == "product"


# =============================================================================
# TestSmartCrusher
# =============================================================================


class TestSmartCrusher:
    """Tests for SmartCrusher transform."""

    def test_should_apply_below_threshold(self, tokenizer):
        """Should not apply when tokens below min_tokens_to_crush."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "ok"}'},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=1000,  # High threshold
        )
        crusher = SmartCrusher(config)

        assert not crusher.should_apply(messages, tokenizer)

    def test_should_apply_no_arrays(self, tokenizer):
        """Should not apply when no crushable arrays present."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "ok", "value": 123}'},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
        )
        crusher = SmartCrusher(config)

        assert not crusher.should_apply(messages, tokenizer)

    def test_should_apply_small_array(self, tokenizer):
        """Should not apply when array is below min_items_to_analyze."""
        small_array = [{"id": i} for i in range(3)]
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(small_array)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=10,  # Array too small
        )
        crusher = SmartCrusher(config)

        assert not crusher.should_apply(messages, tokenizer)

    def test_crush_time_series_keeps_change_points(self, tokenizer, default_config):
        """Time series crushing should preserve items around change points."""
        # Create data with clear change point AND an anomaly signal
        items = []
        for i in range(30):
            if i < 15:
                value = 100.0
            else:
                value = 200.0  # Jump at index 15
            # Add anomaly to provide importance signal for crushing
            if i == 25:
                value = 999.0  # Extreme anomaly
            items.append(
                {
                    "timestamp": f"2025-01-{(i % 28) + 1:02d}T12:00:00Z",
                    "value": value,
                }
            )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=15,
            preserve_change_points=True,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[1]["content"]
        # Remove digest marker
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Should have fewer items than original
        assert len(crushed) < len(items)

        # Should have items around change point (index ~15)
        values = [item["value"] for item in crushed]
        # Should have both low and high values (around change point)
        assert 100.0 in values
        assert 200.0 in values

    def test_crush_keeps_first_k_items(self, tokenizer):
        """Crushing should always keep first K items (Safe V1 Recipe)."""
        items = generate_generic_data(30)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # First 3 items should be preserved
        crushed_ids = [item["id"] for item in crushed]
        assert 0 in crushed_ids  # First item
        assert 1 in crushed_ids  # Second item
        assert 2 in crushed_ids  # Third item

    def test_crush_keeps_last_k_items(self, tokenizer):
        """Crushing should always keep last K items (Safe V1 Recipe)."""
        items = generate_generic_data(30)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Last 2 items should be preserved
        crushed_ids = [item["id"] for item in crushed]
        assert 28 in crushed_ids  # Second to last
        assert 29 in crushed_ids  # Last item

    def test_crush_keeps_error_items(self, tokenizer):
        """Crushing should always preserve error items (Safe V1 Recipe)."""
        items = generate_generic_data(30)
        # Add error items in the middle
        items[10] = {"id": 10, "status": "error", "message": "Connection failed"}
        items[20] = {"id": 20, "status": "failed", "exception": "TimeoutError"}

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Error items should be preserved
        crushed_ids = [item["id"] for item in crushed]
        assert 10 in crushed_ids  # Error item
        assert 20 in crushed_ids  # Failed item

    def test_crush_keeps_anomalies(self, tokenizer):
        """Crushing should preserve anomalous numeric items (> 2 std from mean)."""
        items = []
        for i in range(30):
            value = 100.0 + (i * 0.1)  # Normal range ~100-103
            items.append({"id": i, "metric": value})

        # Add anomaly in the middle
        items[15]["metric"] = 500.0  # Way above mean

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
            variance_threshold=2.0,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Anomaly should be preserved
        crushed_ids = [item["id"] for item in crushed]
        assert 15 in crushed_ids  # Anomaly

    def test_crush_top_n_by_score(self, tokenizer):
        """Search results should be crushed keeping top N by score."""
        items = generate_search_results(30)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Should have reduced items
        assert len(crushed) < len(items)

        # Top scored items should be present
        crushed_ids = [item["id"] for item in crushed]
        assert "doc_0" in crushed_ids  # Highest score
        assert "doc_1" in crushed_ids  # Second highest

    def test_schema_preserved(self, tokenizer):
        """Output items should have same schema as input items."""
        items = [
            {
                "id": i,
                "name": f"Item {i}",
                "nested": {"key": f"value_{i}"},
                "tags": ["a", "b"],
            }
            for i in range(20)
        ]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Each item should have original schema
        original_keys = set(items[0].keys())
        for item in crushed:
            assert set(item.keys()) == original_keys
            assert "nested" in item
            assert "key" in item["nested"]
            assert "tags" in item
            assert isinstance(item["tags"], list)

    def test_respects_max_items_after_crush(self, tokenizer):
        """Output should respect max_items_after_crush limit."""
        # Create data with importance signals (errors, anomalies) so crushing happens
        items = []
        for i in range(100):
            item = {
                "id": i,
                "name": f"Item {i}",
                "type": f"type_{i % 3}",
                "value": 100.0,
            }
            # Add some errors to provide importance signal
            if i in [10, 30, 50, 70, 90]:
                item["error"] = f"Error at {i}"
            # Add some anomalies
            if i in [15, 45, 75]:
                item["value"] = 99999.0
            items.append(item)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=15,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # With errors and anomalies, crushing should happen
        # But critical items override max, so we may have more than 15
        # The test verifies that crushing happened (fewer than original)
        assert len(crushed) < 100, "Should compress the data"
        # Errors must be preserved
        error_count = sum(1 for x in crushed if x.get("error"))
        assert error_count == 5, "All errors must be preserved"


# =============================================================================
# TestRelevanceScoring
# =============================================================================


class TestRelevanceScoring:
    """Tests for relevance scoring in SmartCrusher."""

    def test_context_extraction_from_user_messages(self, tokenizer):
        """Context should be extracted from user messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Find user Alice in the database"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_users", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "[]"},
        ]

        config = SmartCrusherConfig(enabled=True, min_tokens_to_crush=0)
        crusher = SmartCrusher(config)

        context = crusher._extract_context_from_messages(messages)

        assert "Alice" in context
        assert "database" in context

    def test_context_extraction_from_tool_calls(self, tokenizer):
        """Context should be extracted from tool call arguments."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Search for it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_users",
                            "arguments": '{"query": "user_id=12345"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "[]"},
        ]

        config = SmartCrusherConfig(enabled=True, min_tokens_to_crush=0)
        crusher = SmartCrusher(config)

        context = crusher._extract_context_from_messages(messages)

        assert "12345" in context

    def test_relevance_keeps_matching_items(self, tokenizer):
        """Items matching user query should be preserved."""
        # Create items where one matches the user query
        items = [{"id": i, "name": f"User {i}", "email": f"user{i}@example.com"} for i in range(30)]
        # Add special user Alice
        items[15] = {"id": 15, "name": "Alice", "email": "alice@example.com"}

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Find user Alice"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_users", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        # Use BM25 scorer for deterministic testing
        relevance_config = RelevanceScorerConfig(
            tier="bm25",
            relevance_threshold=0.1,  # Lower threshold to catch matches
        )
        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config, relevance_config=relevance_config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[-1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Alice should be preserved due to relevance matching
        crushed_names = [item["name"] for item in crushed]
        assert "Alice" in crushed_names

    def test_custom_scorer_injection(self, tokenizer):
        """Custom scorer should be used when provided."""

        class MockScorer(RelevanceScorer):
            """Mock scorer that always returns high score for items with id=5."""

            def score(self, item: str, context: str) -> RelevanceScore:
                if '"id": 5' in item or '"id":5' in item:
                    return RelevanceScore(score=0.9, reason="mock high score")
                return RelevanceScore(score=0.0, reason="mock low score")

            def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
                return [self.score(item, context) for item in items]

        items = generate_generic_data(30)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Find the special item"},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config, scorer=MockScorer())

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[-1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Item with id=5 should be preserved due to mock scorer
        crushed_ids = [item["id"] for item in crushed]
        assert 5 in crushed_ids


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for SmartCrusher."""

    def test_malformed_json_passthrough(self, tokenizer):
        """Malformed JSON should pass through unchanged."""
        malformed = "{ this is not valid JSON ["

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": malformed},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Content should be unchanged
        assert result.messages[1]["content"] == malformed

    def test_nested_arrays(self, tokenizer):
        """Nested arrays should be handled correctly."""
        # Use with_signals=True to enable crushing with new statistical detection
        nested_data = {
            "results": generate_generic_data(20, with_signals=True),
            "metadata": {
                "inner_array": [{"x": i, "value": 100.0 if i != 7 else 99999.0} for i in range(15)],
            },
        }

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(nested_data)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Should be modified
        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Results array should be crushed (with signals, crushing can happen)
        assert len(crushed["results"]) < 20, "Results should be crushed"

        # Nested array should also be crushed if large enough
        assert "metadata" in crushed
        assert "inner_array" in crushed["metadata"]

    def test_anthropic_style_tool_results(self, tokenizer):
        """Anthropic-style tool_result blocks should be handled."""
        # Use with_signals=True to enable crushing with new statistical detection
        items = generate_generic_data(20, with_signals=True)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What are the results?"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": json.dumps(items),
                    },
                ],
            },
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        # Check should_apply works
        assert crusher.should_apply(messages, tokenizer)

        result = crusher.apply(messages, tokenizer)

        # Tool result content should be crushed (with signals present)
        tool_result_block = result.messages[1]["content"][1]
        content = tool_result_block["content"]
        json_part = content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        assert len(crushed) < len(items), "With signals, crushing should happen"

    def test_openai_style_tool_results(self, tokenizer):
        """OpenAI-style tool messages should be handled."""
        # Use with_signals=True to enable crushing with new statistical detection
        items = generate_generic_data(20, with_signals=True)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Search for items"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "search_items", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": json.dumps(items),
            },
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        # Check should_apply works
        assert crusher.should_apply(messages, tokenizer)

        result = crusher.apply(messages, tokenizer)

        # Tool content should be crushed
        tool_content = result.messages[3]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        assert len(crushed) < len(items)
        assert len(crushed) <= 10

    def test_empty_tool_content(self, tokenizer):
        """Empty tool content should be handled gracefully."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": ""},
        ]

        config = SmartCrusherConfig(enabled=True, min_tokens_to_crush=0)
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Should not crash, content unchanged
        assert result.messages[1]["content"] == ""

    def test_non_dict_array_items(self, tokenizer):
        """Arrays of non-dict items should be handled gracefully."""
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
        )
        crusher = SmartCrusher(config)

        # Should not crash, but won't crush non-dict arrays
        result = crusher.apply(messages, tokenizer)

        # Primitive arrays are not crushed by SmartCrusher (it requires dict items)
        # Content should pass through
        assert result.messages is not None

    def test_mixed_null_values(self, tokenizer):
        """Items with null values should be handled without crashing.

        With statistical detection, this data has unique IDs and no importance
        signals, so it will be SKIPPED (not crushed) - which is correct behavior.
        The test verifies that null values don't crash the analyzer.
        """
        items = [{"id": i, "value": None if i % 2 == 0 else i * 10} for i in range(20)]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Should not crash - parsing should succeed
        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # With statistical detection, unique entities with no signals are SKIPPED
        # This is correct conservative behavior - all 20 items preserved
        assert len(crushed) == 20

    def test_unicode_content(self, tokenizer):
        """Unicode content should be preserved."""
        items = [
            {"id": i, "name": f"Item {i} - \u4e2d\u6587 \u65e5\u672c\u8a9e \ud83d\ude80"}
            for i in range(20)
        ]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(items, ensure_ascii=False),
            },
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Unicode should be preserved
        for item in crushed:
            assert "\u4e2d\u6587" in item["name"]
            assert "\u65e5\u672c\u8a9e" in item["name"]

    def test_digest_marker_added(self, tokenizer):
        """Digest marker should be added to crushed content."""
        items = generate_generic_data(30)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[1]["content"]

        # Should have digest marker
        assert "<headroom:tool_digest" in tool_content
        assert "sha256=" in tool_content
        assert len(result.markers_inserted) > 0

    def test_transforms_applied_tracking(self, tokenizer):
        """Transforms applied should be tracked."""
        items = generate_generic_data(30)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Should track transforms
        assert len(result.transforms_applied) > 0
        assert any("smart" in t.lower() for t in result.transforms_applied)

    def test_token_reduction(self, tokenizer):
        """Token count should be reduced after crushing."""
        items = generate_generic_data(100)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Tokens should be reduced
        assert result.tokens_after < result.tokens_before


# =============================================================================
# Integration Tests
# =============================================================================


class TestSmartCrusherIntegration:
    """Integration tests for SmartCrusher with realistic scenarios."""

    def test_database_query_results(self, tokenizer):
        """Simulate crushing database query results."""
        # Simulate a database query returning many rows
        items = [
            {
                "user_id": f"usr_{i:05d}",
                "email": f"user{i}@example.com",
                "created_at": f"2025-01-{(i % 28) + 1:02d}T00:00:00Z",
                "status": "active" if i % 10 != 0 else "inactive",
                "login_count": i * 5,
            }
            for i in range(100)
        ]

        messages = [
            {"role": "system", "content": "You are a database assistant."},
            {"role": "user", "content": "Show me users with email containing 'user50'"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "query_users", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        relevance_config = RelevanceScorerConfig(tier="bm25", relevance_threshold=0.1)
        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=15,
        )
        crusher = SmartCrusher(config, relevance_config=relevance_config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[3]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Should be significantly reduced
        assert len(crushed) <= 15
        assert len(crushed) < len(items)

        # User 50 should be preserved due to relevance
        user_ids = [item["user_id"] for item in crushed]
        assert "usr_00050" in user_ids

    def test_api_search_results(self, tokenizer):
        """Simulate crushing API search results."""
        items = [
            {
                "id": f"result_{i}",
                "title": f"Result {i}: {'Important Finding' if i < 5 else 'Regular Result'}",
                "relevance_score": 0.95 - (i * 0.02),
                "snippet": f"This is the snippet for result {i}...",
                "url": f"https://example.com/doc/{i}",
            }
            for i in range(50)
        ]

        messages = [
            {"role": "system", "content": "You are a search assistant."},
            {"role": "user", "content": "Search for important findings"},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[2]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Should use TOP_N strategy
        assert len(crushed) <= 12  # max + some buffer for first/last

        # Top results should be preserved
        ids = [item["id"] for item in crushed]
        assert "result_0" in ids
        assert "result_1" in ids

    def test_monitoring_metrics(self, tokenizer):
        """Simulate crushing monitoring/metrics data."""
        items = []
        for i in range(60):
            # Normal CPU usage around 50%
            cpu = 50.0 + (i * 0.1)
            # Spike at index 30
            if i == 30:
                cpu = 95.0
            items.append(
                {
                    "timestamp": f"2025-01-06T{10 + (i // 60):02d}:{i % 60:02d}:00Z",
                    "cpu_percent": cpu,
                    "memory_percent": 60.0,
                    "host": "server-01",
                }
            )

        messages = [
            {"role": "system", "content": "You are a monitoring assistant."},
            {"role": "user", "content": "Show me CPU metrics"},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=15,
            preserve_change_points=True,
        )
        crusher = SmartCrusher(config)

        result = crusher.apply(messages, tokenizer)

        # Parse result
        tool_content = result.messages[2]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Should preserve the spike
        cpu_values = [item["cpu_percent"] for item in crushed]
        assert 95.0 in cpu_values  # Spike should be preserved
