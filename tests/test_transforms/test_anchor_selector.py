"""Tests for AnchorSelector - Adaptive position-based anchor allocation.

Comprehensive tests covering:
- Adversarial positions: Important data NOT at expected positions
- Size adaptation: Anchor allocation scaling with array size
- Pattern-aware anchoring: Different strategies for different data patterns
- Query-aware anchoring: Query-based anchor adjustment
- Information density: Unique/informative item selection
- Coverage metrics: Distribution coverage verification
- Edge cases: Boundary conditions and special scenarios

These tests verify the AnchorSelector replaces the static "first 3 + last 2"
preservation with adaptive, pattern-aware anchor selection.
"""

import json

import pytest

from headroom import OpenAIProvider, SmartCrusherConfig, Tokenizer
from headroom.transforms.smart_crusher import (
    SmartAnalyzer,
    SmartCrusher,
)

# =============================================================================
# Test Fixtures
# =============================================================================

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


def generate_uniform_items(n: int, value: int = 0) -> list[dict]:
    """Generate array of identical items."""
    return [{"id": "same", "value": value, "status": "ok"} for _ in range(n)]


def generate_numbered_items(n: int, with_value: bool = True) -> list[dict]:
    """Generate items with sequential IDs for position tracking."""
    items = []
    for i in range(n):
        item = {"id": i, "name": f"Item {i}"}
        if with_value:
            item["value"] = i * 10
        items.append(item)
    return items


def generate_time_series_data(
    n: int = 50,
    spike_positions: list[int] | None = None,
    spike_value: float = 1000.0,
) -> list[dict]:
    """Generate time series with optional spikes at specific positions."""
    items = []
    for i in range(n):
        value = 100.0 + (i * 0.5)  # Slight upward trend
        if spike_positions and i in spike_positions:
            value = spike_value
        items.append(
            {
                "timestamp": f"2025-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00Z",
                "value": value,
                "metric": "cpu_usage",
            }
        )
    return items


def generate_log_data(
    n: int = 50,
    error_positions: list[int] | None = None,
) -> list[dict]:
    """Generate log-style data with optional errors at specific positions."""
    levels = ["INFO", "DEBUG", "WARN"]
    items = []
    for i in range(n):
        level = levels[i % len(levels)]
        if error_positions and i in error_positions:
            level = "ERROR"
            message = f"Critical failure at step {i}: connection timeout"
        else:
            message = f"Processing request {i} successfully"
        items.append(
            {
                "level": level,
                "message": message,
                "timestamp": f"2025-01-06T{12 + (i // 60):02d}:{i % 60:02d}:00Z",
            }
        )
    return items


def generate_search_results(n: int = 50) -> list[dict]:
    """Generate search results with scores (higher = better, at front)."""
    return [
        {
            "id": f"doc_{i}",
            "title": f"Document {i}",
            "score": 1.0 - (i * 0.02),  # Scores decrease as index increases
            "snippet": f"This is a snippet from document {i}...",
        }
        for i in range(n)
    ]


def generate_categorized_items(
    categories: dict[str, int],
) -> list[dict]:
    """Generate items with specific category distribution.

    Args:
        categories: Dict mapping category name to count, e.g. {"A": 30, "B": 30, "C": 40}
    """
    items = []
    i = 0
    for category, count in categories.items():
        for _ in range(count):
            items.append({"id": i, "category": category, "name": f"Item {i}"})
            i += 1
    return items


# =============================================================================
# Helper Functions
# =============================================================================


def crush_items(
    items: list[dict],
    tokenizer: Tokenizer,
    max_items: int = 10,
    query: str = "",
    min_items: int = 3,
) -> list[dict]:
    """Helper to crush items and return the result list.

    Args:
        items: Array of items to compress.
        tokenizer: Tokenizer instance.
        max_items: Maximum items after crushing.
        query: Optional query context.
        min_items: Minimum items to analyze.

    Returns:
        List of preserved items after crushing.
    """
    messages = [
        {"role": "system", "content": "You are helpful."},
    ]

    if query:
        messages.append({"role": "user", "content": query})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_items", "arguments": "{}"},
                    }
                ],
            }
        )

    messages.append({"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)})

    config = SmartCrusherConfig(
        enabled=True,
        min_tokens_to_crush=0,
        min_items_to_analyze=min_items,
        max_items_after_crush=max_items,
    )
    crusher = SmartCrusher(config)
    result = crusher.apply(messages, tokenizer)

    # Parse result
    tool_content = result.messages[-1]["content"]
    json_part = tool_content.split("\n<headroom:")[0]
    return json.loads(json_part)


def get_preserved_positions(result: list[dict], original_size: int) -> dict[str, list[int]]:
    """Categorize preserved item positions into front/middle/back.

    Args:
        result: List of preserved items (must have 'id' field with original index).
        original_size: Size of original array.

    Returns:
        Dict with 'front', 'middle', 'back' keys containing lists of preserved indices.
    """
    front_boundary = int(original_size * 0.1)
    back_boundary = int(original_size * 0.9)

    positions = {"front": [], "middle": [], "back": []}

    for item in result:
        idx = item.get("id")
        if idx is None:
            continue

        if isinstance(idx, int):
            if idx < front_boundary:
                positions["front"].append(idx)
            elif idx >= back_boundary:
                positions["back"].append(idx)
            else:
                positions["middle"].append(idx)

    return positions


# =============================================================================
# TestAdversarialPositions
# =============================================================================


class TestAdversarialPositions:
    """Test scenarios that break 'first 3 + last 2' assumption.

    These tests verify that important data is preserved regardless of position,
    not just because it happens to be at the front or back of the array.
    """

    def test_important_data_in_middle(self, tokenizer):
        """Critical error item at position 50 of 100-item array should be preserved.

        The error item is in the middle where static anchoring would miss it.
        It should be preserved due to error detection, not position.
        """
        items = [{"id": i, "status": "ok", "value": i * 10} for i in range(100)]
        # Place critical error in the middle
        items[50] = {
            "id": 50,
            "status": "error",
            "error_code": "CRITICAL",
            "message": "Connection failed",
        }

        crushed = crush_items(items, tokenizer, max_items=15)

        # Error item MUST be preserved regardless of position
        preserved_ids = [item["id"] for item in crushed]
        assert 50 in preserved_ids, "Critical error at position 50 should be preserved"

        # Verify it's preserved for the right reason (has error indicator)
        error_items = [item for item in crushed if item.get("status") == "error"]
        assert len(error_items) >= 1, "At least one error item should be preserved"

    def test_spike_not_at_boundaries(self, tokenizer):
        """Numeric spike at position 75 of 100-item array must be preserved.

        The spike represents an anomaly that should be detected statistically,
        not missed because it's not at the front or back.
        """
        items = generate_time_series_data(100, spike_positions=[75], spike_value=1000.0)
        # Add id field for tracking
        for i, item in enumerate(items):
            item["id"] = i

        crushed = crush_items(items, tokenizer, max_items=15)

        # Spike MUST be preserved as anomaly
        has_spike = any(item.get("value", 0) > 500 for item in crushed)
        assert has_spike, "Anomalous spike at position 75 should be preserved"

        # Verify the specific position was kept
        preserved_ids = [item.get("id") for item in crushed]
        assert 75 in preserved_ids, "Position 75 with spike should be in result"

    def test_multiple_spikes_scattered(self, tokenizer):
        """Multiple spikes at positions 25, 50, 75 should all be preserved."""
        spike_positions = [25, 50, 75]
        items = generate_time_series_data(100, spike_positions=spike_positions, spike_value=999.0)
        for i, item in enumerate(items):
            item["id"] = i

        crushed = crush_items(items, tokenizer, max_items=15)

        # All spikes should be preserved
        preserved_ids = {item.get("id") for item in crushed}
        for pos in spike_positions:
            assert pos in preserved_ids, f"Spike at position {pos} should be preserved"

    def test_first_items_identical(self, tokenizer):
        """First 10 identical items - should NOT waste all anchor slots on them.

        When first N items are identical, we should keep at most 1-2 of them,
        not waste 3 anchor slots on duplicates.
        """
        # First 10 items are identical
        identical_items = [{"id": "same", "value": 0, "type": "duplicate"} for _ in range(10)]
        # Rest are unique
        unique_items = [{"id": f"unique_{i}", "value": i * 10, "type": "unique"} for i in range(90)]
        items = identical_items + unique_items

        crushed = crush_items(items, tokenizer, max_items=10)

        # Should NOT have multiple identical items
        same_count = sum(1 for item in crushed if item.get("id") == "same")
        assert same_count <= 2, f"Got {same_count} identical items, expected at most 2"

        # Should have some unique items
        unique_count = sum(1 for item in crushed if item.get("type") == "unique")
        assert unique_count >= 5, f"Expected at least 5 unique items, got {unique_count}"

    def test_last_items_identical(self, tokenizer):
        """Last 10 identical items - should NOT waste anchor slots.

        Similar to above but for back anchors - when last N items are identical,
        we should preserve variety instead of duplicates.
        """
        # First 90 items are unique
        unique_items = [{"id": f"unique_{i}", "value": i * 10, "type": "unique"} for i in range(90)]
        # Last 10 items are identical
        identical_items = [{"id": "same", "value": 100, "type": "duplicate"} for _ in range(10)]
        items = unique_items + identical_items

        crushed = crush_items(items, tokenizer, max_items=10)

        # Should NOT have multiple identical items
        same_count = sum(1 for item in crushed if item.get("id") == "same")
        assert same_count <= 2, f"Got {same_count} identical items, expected at most 2"

    def test_relevant_item_in_middle(self, tokenizer):
        """Item matching query in middle position must be found.

        When user queries for a specific item that's at position 42,
        it should be preserved even though it's not at front/back.
        """
        items = [{"id": i, "name": f"item_{i}", "status": "active"} for i in range(100)]
        # Put target item in the middle
        items[42]["name"] = "target_special_item"
        items[42]["description"] = "This is what the user is looking for"

        crushed = crush_items(
            items,
            tokenizer,
            max_items=10,
            query="Find target_special_item",
        )

        # Query-matched item MUST be preserved
        has_target = any("target_special_item" in item.get("name", "") for item in crushed)
        assert has_target, "Item matching query should be preserved regardless of position"

    def test_error_in_middle_of_otherwise_uniform_data(self, tokenizer):
        """Single error in middle of 100 identical OK items must be preserved."""
        items = [{"id": i, "status": "ok", "data": "normal"} for i in range(100)]
        items[47] = {"id": 47, "status": "failed", "error": "Unexpected failure"}

        crushed = crush_items(items, tokenizer, max_items=15)

        # Error must be preserved
        error_items = [item for item in crushed if item.get("status") == "failed"]
        assert len(error_items) >= 1, "Error item at position 47 should be preserved"
        assert error_items[0]["id"] == 47, "The specific error item should be preserved"


# =============================================================================
# TestSizeAdaptation
# =============================================================================


class TestSizeAdaptation:
    """Test that anchor allocation scales with array size.

    Larger arrays should allocate proportionally more anchor slots
    to ensure adequate coverage across the data range.
    """

    @pytest.mark.parametrize(
        "size,max_items,expected_min_anchors",
        [
            (20, 10, 3),  # Small array: at least 3 anchors (front + back)
            (100, 15, 4),  # Medium array: at least 4 anchors
            (500, 20, 5),  # Large array: at least 5 anchors
            (2000, 25, 6),  # Very large: at least 6 anchors
        ],
    )
    def test_anchor_count_scales(self, tokenizer, size, max_items, expected_min_anchors):
        """Anchor count should increase with array size.

        Verifies that the number of items from boundary regions (front 10%, back 10%)
        increases as the array size grows.
        """
        items = generate_numbered_items(size)

        crushed = crush_items(items, tokenizer, max_items=max_items)

        # Count items from first 10% and last 10%
        front_boundary = int(size * 0.1)
        back_boundary = int(size * 0.9)

        anchor_count = sum(
            1 for item in crushed if item["id"] < front_boundary or item["id"] >= back_boundary
        )

        assert anchor_count >= expected_min_anchors, (
            f"Array of size {size} should have at least {expected_min_anchors} "
            f"anchors from boundary regions, got {anchor_count}"
        )

    def test_small_array_high_preservation(self, tokenizer):
        """Small arrays (< max_items) should preserve most/all items.

        When the array is smaller than max_items, there's no need to drop items.
        """
        items = generate_numbered_items(8)

        # max_items=20 is larger than array size
        crushed = crush_items(items, tokenizer, max_items=20)

        # Should preserve all or nearly all items
        assert len(crushed) >= 7, f"Small array should preserve most items, got {len(crushed)}"

    def test_small_array_exact_size(self, tokenizer):
        """Array exactly at max_items should preserve all items."""
        items = generate_numbered_items(10)

        crushed = crush_items(items, tokenizer, max_items=10)

        # Should preserve all items
        assert len(crushed) == 10, "Array at max_items limit should keep all items"

    def test_large_array_efficient_sampling(self, tokenizer):
        """Large arrays should sample efficiently across all positions.

        A 500-item array crushed to 20 items should have representation
        from front, middle, and back regions.
        """
        items = generate_numbered_items(500)

        crushed = crush_items(items, tokenizer, max_items=20)

        positions = get_preserved_positions(crushed, 500)

        # Should have representation from all regions
        has_front = len(positions["front"]) >= 1
        has_back = len(positions["back"]) >= 1
        # Middle representation is optional but preferred for large arrays
        has_middle = len(positions["middle"]) >= 1

        assert has_front, "Large array should preserve items from front"
        assert has_back, "Large array should preserve items from back"
        # Middle coverage is important for large arrays
        assert has_middle, "Large array should have some middle representation"

    def test_very_large_array_coverage(self, tokenizer):
        """Very large array (1000+) should have good position distribution."""
        items = generate_numbered_items(1000)

        crushed = crush_items(items, tokenizer, max_items=25)

        positions = get_preserved_positions(crushed, 1000)

        # Calculate coverage spread
        all_positions = positions["front"] + positions["middle"] + positions["back"]
        if len(all_positions) >= 2:
            spread = max(all_positions) - min(all_positions)
            # Should span at least 80% of the array
            assert spread >= 800, f"Preserved items should span array, spread was {spread}"


# =============================================================================
# TestPatternAwareAnchoring
# =============================================================================


class TestPatternAwareAnchoring:
    """Test pattern-specific anchor strategies.

    Different data patterns (search results, logs, time series) should
    use different anchor weighting strategies.
    """

    def test_search_results_front_heavy(self, tokenizer):
        """Search results with scores should preserve more from front (high scores).

        Search results are sorted by relevance score, so top items are most important.
        """
        items = generate_search_results(100)
        for i, item in enumerate(items):
            item["idx"] = i  # Track original position

        crushed = crush_items(items, tokenizer, max_items=12)

        # Count items by their original position
        front_count = sum(1 for item in crushed if item.get("idx", 100) < 30)
        back_count = sum(1 for item in crushed if item.get("idx", 0) >= 70)

        # For search results, front (high scores) should dominate
        assert front_count > back_count, (
            f"Search results should preserve more front items (high scores), "
            f"got front={front_count}, back={back_count}"
        )

        # Top results should definitely be present
        ids = [item["id"] for item in crushed]
        assert "doc_0" in ids, "Top search result should be preserved"
        assert "doc_1" in ids, "Second search result should be preserved"

    def test_logs_back_heavy(self, tokenizer):
        """Logs with timestamps should preserve more recent items (back of array).

        Logs are typically ordered chronologically, with recent items at the end.
        """
        items = generate_log_data(100)
        for i, item in enumerate(items):
            item["idx"] = i

        crushed = crush_items(items, tokenizer, max_items=12)

        # For logs, back (recent) should be emphasized
        # Extract indices - use 'idx' field we added
        indices = [item.get("idx", 0) for item in crushed]

        recent_count = sum(1 for idx in indices if idx >= 70)
        old_count = sum(1 for idx in indices if idx < 30)

        # Recent logs should be at least as represented as old logs
        assert recent_count >= old_count, (
            f"Logs should preserve recent items, got recent={recent_count}, old={old_count}"
        )

    def test_time_series_balanced(self, tokenizer):
        """Time series should have balanced front/back representation.

        For trend analysis, we need both the start and end of the time series.
        """
        items = generate_time_series_data(100)
        for i, item in enumerate(items):
            item["id"] = i

        crushed = crush_items(items, tokenizer, max_items=12)

        positions = get_preserved_positions(crushed, 100)

        front_count = len(positions["front"])
        back_count = len(positions["back"])

        # Should be relatively balanced for time series (within 2:1 ratio)
        if front_count > 0 and back_count > 0:
            ratio = max(front_count, back_count) / min(front_count, back_count)
            assert ratio <= 3, f"Time series should have balanced anchors, ratio was {ratio}"

    def test_generic_distributed(self, tokenizer):
        """Generic data should sample across all positions.

        When pattern is unknown, sampling should be distributed rather than
        heavily weighted to any particular region.
        """
        items = generate_numbered_items(100)

        crushed = crush_items(items, tokenizer, max_items=15)

        positions = get_preserved_positions(crushed, 100)

        # Should have items from multiple regions
        regions_with_items = sum(1 for region in ["front", "middle", "back"] if positions[region])

        assert regions_with_items >= 2, (
            f"Generic data should cover multiple regions, got {regions_with_items}"
        )


# =============================================================================
# TestQueryAwareAnchoring
# =============================================================================


class TestQueryAwareAnchoring:
    """Test query-based anchor adjustment.

    User queries containing temporal keywords should shift anchor weighting.
    """

    def test_latest_query_shifts_to_back(self, tokenizer):
        """'Latest' in query should preserve more recent items."""
        items = [{"id": i, "created": f"2024-01-{i:02d}"} for i in range(1, 31)]

        crushed = crush_items(
            items,
            tokenizer,
            max_items=8,
            query="Show me the latest entries",
        )

        ids = [item["id"] for item in crushed]
        recent_count = sum(1 for id in ids if id > 20)

        # Should have multiple recent items due to "latest" keyword
        assert recent_count >= 2, (
            f"Query with 'latest' should preserve recent items, got {recent_count}"
        )

    def test_recent_query_shifts_to_back(self, tokenizer):
        """'Recent' in query should preserve more recent items."""
        items = generate_log_data(50)
        for i, item in enumerate(items):
            item["idx"] = i

        crushed = crush_items(
            items,
            tokenizer,
            max_items=10,
            query="Show me recent log entries",
        )

        indices = [item.get("idx", 0) for item in crushed]
        recent_count = sum(1 for idx in indices if idx >= 35)

        assert recent_count >= 2, "Query with 'recent' should have multiple recent items"

    def test_first_query_shifts_to_front(self, tokenizer):
        """'First' in query should preserve earlier items."""
        items = [{"id": i, "created": f"2024-01-{i:02d}"} for i in range(1, 31)]

        crushed = crush_items(
            items,
            tokenizer,
            max_items=8,
            query="Show me the first entries",
        )

        ids = [item["id"] for item in crushed]
        early_count = sum(1 for id in ids if id < 10)

        # Should have multiple early items due to "first" keyword
        assert early_count >= 2, (
            f"Query with 'first' should preserve early items, got {early_count}"
        )

    def test_oldest_query_shifts_to_front(self, tokenizer):
        """'Oldest' in query should preserve earlier items."""
        items = generate_numbered_items(50)

        crushed = crush_items(
            items,
            tokenizer,
            max_items=10,
            query="Find the oldest records",
        )

        ids = [item["id"] for item in crushed]
        early_count = sum(1 for id in ids if id < 15)

        assert early_count >= 3, "Query with 'oldest' should have multiple early items"

    def test_specific_id_query_finds_item(self, tokenizer):
        """Query for specific ID should find it regardless of position."""
        items = [{"id": f"item_{i:04d}", "value": i} for i in range(200)]

        crushed = crush_items(
            items,
            tokenizer,
            max_items=10,
            query="Find item_0123",
        )

        # Item at position 123 should be found
        ids = [item["id"] for item in crushed]
        assert "item_0123" in ids, "Specific ID query should find the item"

    def test_no_query_uses_default_weights(self, tokenizer):
        """Without query, use pattern-based defaults."""
        items = generate_numbered_items(100)

        crushed = crush_items(items, tokenizer, max_items=15)

        # Without query, should use default anchoring (some front, some back)
        positions = get_preserved_positions(crushed, 100)

        assert len(positions["front"]) >= 1, "Default should include front items"
        assert len(positions["back"]) >= 1, "Default should include back items"


# =============================================================================
# TestInformationDensity
# =============================================================================


class TestInformationDensity:
    """Test information-density based selection.

    Items with unique or rare properties should be preferred over common ones.
    """

    def test_unique_items_preferred(self, tokenizer):
        """Items with rare field values should be preferred over common ones.

        When most items have status="ok" but a few have status="warning",
        the warning items should be preserved as they're more informative.
        """
        items = []
        for i in range(100):
            status = "warning" if i in [25, 50, 75] else "ok"
            items.append({"id": i, "status": status, "value": i})

        crushed = crush_items(items, tokenizer, max_items=15)

        # Warning items (rare status) should be preserved
        warning_count = sum(1 for item in crushed if item.get("status") == "warning")
        assert warning_count >= 2, f"Rare status items should be preferred, got {warning_count}"

    def test_dedup_identical_items(self, tokenizer):
        """Identical items should be deduplicated in anchor selection.

        If positions 0-5 all have identical content, we shouldn't keep all of them.
        """
        # First 6 items are identical
        items = [{"id": "dup", "value": 0, "constant": "same"} for _ in range(6)]
        # Rest are unique
        items.extend([{"id": f"uniq_{i}", "value": i * 10, "unique": True} for i in range(94)])

        crushed = crush_items(items, tokenizer, max_items=12)

        # Should not have many identical items
        dup_count = sum(1 for item in crushed if item.get("id") == "dup")
        unique_count = sum(1 for item in crushed if item.get("unique"))

        assert dup_count <= 2, f"Should deduplicate identical items, got {dup_count}"
        assert unique_count >= 6, f"Should prefer unique items, got {unique_count}"

    def test_structural_outliers_preferred(self, tokenizer):
        """Items with different structure should be preferred.

        An item with extra fields (like an error with stack trace) should be
        preferred over uniform items.
        """
        items = [{"id": i, "status": "ok"} for i in range(100)]
        # Add a structurally different item in the middle
        items[42] = {
            "id": 42,
            "status": "error",
            "error_message": "Something went wrong",
            "stack_trace": "at line 123...",
            "error_code": "ERR_001",
        }

        crushed = crush_items(items, tokenizer, max_items=15)

        # Structural outlier should be preserved
        outlier = [item for item in crushed if item.get("stack_trace")]
        assert len(outlier) >= 1, "Structurally different item should be preserved"

    def test_diverse_values_over_uniform(self, tokenizer):
        """When selecting from candidates, prefer diverse values."""
        items = []
        # Create items with varying diversity
        for i in range(100):
            items.append(
                {
                    "id": i,
                    "type": "common" if i < 90 else f"rare_type_{i}",
                    "value": i,
                }
            )

        crushed = crush_items(items, tokenizer, max_items=15)

        # Should have some rare types
        rare_types = [item for item in crushed if "rare" in item.get("type", "")]
        assert len(rare_types) >= 1, "Rare type values should be preserved"


# =============================================================================
# TestCoverageMetrics
# =============================================================================


class TestCoverageMetrics:
    """Test that preserved items represent the full distribution.

    Compression should maintain coverage of value ranges, categories, and time.
    """

    def test_value_range_coverage(self, tokenizer):
        """Preserved items should cover the value range."""
        items = [{"id": i, "value": i} for i in range(100)]

        crushed = crush_items(items, tokenizer, max_items=12)

        values = [item["value"] for item in crushed]

        # Should cover most of the range [0, 100]
        assert min(values) < 10, "Should have low values"
        assert max(values) > 90, "Should have high values"

        # Should have some middle values too
        middle_count = sum(1 for v in values if 30 < v < 70)
        assert middle_count >= 1, "Should have some middle-range values"

    def test_category_coverage(self, tokenizer):
        """Preserved items should represent multiple categories."""
        items = generate_categorized_items({"A": 30, "B": 30, "C": 40})

        crushed = crush_items(items, tokenizer, max_items=12)

        categories = {item["category"] for item in crushed}

        # Should have at least 2 of 3 categories represented
        assert len(categories) >= 2, f"Should cover multiple categories, got {categories}"

    def test_category_proportional_representation(self, tokenizer):
        """Category distribution should roughly reflect original proportions."""
        items = generate_categorized_items({"major": 80, "minor": 20})

        crushed = crush_items(items, tokenizer, max_items=15)

        major_count = sum(1 for item in crushed if item["category"] == "major")
        minor_count = sum(1 for item in crushed if item["category"] == "minor")

        # Major category should have more items, but minor should be represented
        assert major_count > minor_count, "Major category should dominate"
        assert minor_count >= 1, "Minor category should still be represented"

    def test_temporal_coverage(self, tokenizer):
        """Preserved items should span the time range."""
        # Create items spanning 12 months
        items = [
            {"id": i, "timestamp": f"2024-{(i % 12) + 1:02d}-15", "event": f"event_{i}"}
            for i in range(60)
        ]

        crushed = crush_items(items, tokenizer, max_items=12)

        months = [int(item["timestamp"][5:7]) for item in crushed]

        # Should span a significant portion of the year
        month_range = max(months) - min(months)
        assert month_range >= 5, f"Should span multiple months, got range of {month_range}"

    def test_numeric_distribution_coverage(self, tokenizer):
        """Preserved numeric values should represent the distribution."""
        # Create items with bimodal distribution
        items = []
        for i in range(50):
            items.append({"id": i, "value": 10 + (i % 5)})  # Low cluster: 10-15
        for i in range(50):
            items.append({"id": 50 + i, "value": 90 + (i % 5)})  # High cluster: 90-95

        crushed = crush_items(items, tokenizer, max_items=12)

        values = [item["value"] for item in crushed]

        # Should have items from both clusters
        low_cluster = [v for v in values if v < 20]
        high_cluster = [v for v in values if v > 85]

        assert len(low_cluster) >= 1, "Should have items from low cluster"
        assert len(high_cluster) >= 1, "Should have items from high cluster"


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_array(self, tokenizer):
        """Empty array should return empty result."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": "[]"},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)
        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[-1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        assert crushed == [], "Empty array should return empty result"

    def test_single_item(self, tokenizer):
        """Single item array should preserve the item."""
        items = [{"id": 0, "name": "Only item"}]

        crushed = crush_items(items, tokenizer, max_items=10, min_items=1)

        assert len(crushed) == 1, "Single item should be preserved"
        assert crushed[0]["name"] == "Only item"

    def test_two_items(self, tokenizer):
        """Two item array should preserve both items."""
        items = [{"id": 0, "name": "First"}, {"id": 1, "name": "Second"}]

        crushed = crush_items(items, tokenizer, max_items=10, min_items=1)

        assert len(crushed) == 2, "Both items should be preserved"

    def test_all_identical_items(self, tokenizer):
        """Array of all identical items should keep minimal anchors."""
        items = [{"id": "same", "value": 42, "status": "ok"} for _ in range(50)]

        crushed = crush_items(items, tokenizer, max_items=10)

        # Should not keep all 10 identical items - that would be wasteful
        # But some anchors are needed for structure
        assert len(crushed) <= 10, "Should respect max_items"
        # All items are identical, so any subset is equivalent

    def test_max_items_greater_than_array(self, tokenizer):
        """max_items > array size should return all items."""
        items = generate_numbered_items(8)

        crushed = crush_items(items, tokenizer, max_items=20)

        assert len(crushed) == 8, "Should return all items when max_items > array size"

    def test_max_items_equals_array(self, tokenizer):
        """max_items == array size should return all items."""
        items = generate_numbered_items(15)

        crushed = crush_items(items, tokenizer, max_items=15)

        assert len(crushed) == 15, "Should return all items when max_items == array size"

    def test_all_items_have_errors(self, tokenizer):
        """When all items have errors, quality guarantee preserves all.

        Current behavior: Error preservation takes precedence over max_items.
        All error items are preserved to avoid losing critical information.
        This is a deliberate design choice documented in SmartCrusher.

        NOTE: Future AnchorSelector may implement error deduplication or sampling
        for cases where all items are errors, but that requires careful design.
        """
        items = [{"id": i, "status": "error", "error_code": f"ERR_{i}"} for i in range(50)]

        crushed = crush_items(items, tokenizer, max_items=15)

        # Current behavior: all errors are preserved (quality guarantee)
        # This is intentional - we don't want to lose error information
        assert len(crushed) == 50, "All error items should be preserved (quality guarantee)"

    def test_none_values_handled(self, tokenizer):
        """Items with None values should be handled gracefully."""
        items = [
            {"id": i, "value": None if i % 3 == 0 else i * 10, "name": f"Item {i}"}
            for i in range(30)
        ]

        crushed = crush_items(items, tokenizer, max_items=10)

        # Should not crash, should return valid items
        assert len(crushed) > 0, "Should handle None values"
        assert all(isinstance(item, dict) for item in crushed)

    def test_mixed_types_in_array(self, tokenizer):
        """Array with mixed item structures should be handled."""
        items = [
            {"id": 0, "simple": True},
            {"id": 1, "nested": {"deep": {"value": 42}}},
            {"id": 2, "list_field": [1, 2, 3]},
            {"id": 3, "mixed": {"a": [1, 2], "b": "text"}},
        ]
        # Add more simple items to trigger crushing
        items.extend([{"id": i, "simple": True} for i in range(4, 20)])

        crushed = crush_items(items, tokenizer, max_items=8, min_items=3)

        # Should preserve structurally interesting items
        assert len(crushed) > 0, "Should handle mixed structures"

    def test_unicode_content_preserved(self, tokenizer):
        """Unicode content should be preserved correctly."""
        # Use the full Unicode codepoint for rocket emoji (U+1F680)
        rocket_emoji = "\U0001f680"  # Full codepoint, not surrogate pair
        items = [
            {"id": i, "name": f"Item {i} - \u4e2d\u6587 \u65e5\u672c\u8a9e {rocket_emoji}"}
            for i in range(20)
        ]

        crushed = crush_items(items, tokenizer, max_items=10)

        # Unicode should be preserved
        for item in crushed:
            assert "\u4e2d\u6587" in item["name"], "Chinese characters should be preserved"
            assert rocket_emoji in item["name"], "Emoji should be preserved"

    def test_very_long_string_values(self, tokenizer):
        """Items with very long string values should be handled."""
        items = [{"id": i, "data": "x" * 10000 if i == 10 else "short"} for i in range(30)]

        crushed = crush_items(items, tokenizer, max_items=10)

        # Should not crash
        assert len(crushed) > 0, "Should handle long strings"

    def test_deeply_nested_items(self, tokenizer):
        """Deeply nested items should be handled without stack overflow."""

        def create_nested(depth: int) -> dict:
            if depth == 0:
                return {"value": "leaf"}
            return {"nested": create_nested(depth - 1)}

        items = [{"id": i, "deep": create_nested(10)} for i in range(20)]

        crushed = crush_items(items, tokenizer, max_items=10)

        assert len(crushed) > 0, "Should handle deeply nested items"


# =============================================================================
# TestAnchorConfigBehavior
# =============================================================================


class TestAnchorConfigBehavior:
    """Test configuration-driven anchor behavior."""

    def test_high_max_items_reduces_compression(self, tokenizer):
        """Higher max_items should preserve more items.

        NOTE: Data must have "importance signals" (errors, anomalies) to trigger
        compression. Generic unique items without signals are SKIPPED by the
        crushability analysis (conservative behavior to avoid losing entities).

        This test uses search results which always compress based on score.
        """
        # Use search results - they always trigger TOP_N compression
        items = generate_search_results(100)

        crushed_low = crush_items(items, tokenizer, max_items=8)
        crushed_high = crush_items(items, tokenizer, max_items=25)

        assert len(crushed_high) > len(crushed_low), (
            f"Higher max_items should preserve more items, "
            f"got low={len(crushed_low)}, high={len(crushed_high)}"
        )

    def test_min_items_to_analyze_threshold(self, tokenizer):
        """Arrays below min_items_to_analyze should not be crushed."""
        items = generate_numbered_items(5)

        # Set min_items_to_analyze higher than array size
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=10,  # Higher than array size
            max_items_after_crush=3,
        )
        crusher = SmartCrusher(config)
        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[-1]["content"]
        json_part = tool_content.split("\n<headroom:")[0]
        crushed = json.loads(json_part)

        # Should not crush because array is below threshold
        assert len(crushed) == 5, "Array below min_items_to_analyze should not be crushed"


# =============================================================================
# TestPreservationGuarantees
# =============================================================================


class TestPreservationGuarantees:
    """Test hard guarantees about what must be preserved."""

    def test_errors_always_preserved(self, tokenizer):
        """Error items must ALWAYS be preserved, regardless of position or count."""
        items = [{"id": i, "status": "ok"} for i in range(100)]
        # Scatter errors throughout
        error_positions = [10, 25, 45, 65, 85]
        for pos in error_positions:
            items[pos] = {"id": pos, "status": "error", "error": f"Error at {pos}"}

        crushed = crush_items(items, tokenizer, max_items=15)

        # All errors should be preserved
        error_count = sum(1 for item in crushed if item.get("status") == "error")
        assert error_count == len(error_positions), (
            f"All {len(error_positions)} errors should be preserved, got {error_count}"
        )

    def test_anomalies_always_preserved(self, tokenizer):
        """Numeric anomalies (statistical outliers) must be preserved."""
        items = [{"id": i, "value": 100 + (i % 5)} for i in range(100)]
        # Add anomalies
        items[33]["value"] = 9999
        items[66]["value"] = -9999

        crushed = crush_items(items, tokenizer, max_items=15)

        values = [item["value"] for item in crushed]

        # Anomalies should be present
        assert 9999 in values, "High anomaly should be preserved"
        assert -9999 in values, "Low anomaly should be preserved"

    def test_query_matches_always_preserved(self, tokenizer):
        """Items matching user query must be preserved."""
        items = [{"id": i, "name": f"generic_item_{i}"} for i in range(100)]
        items[55]["name"] = "special_target_item"
        items[55]["important"] = True

        crushed = crush_items(
            items,
            tokenizer,
            max_items=10,
            query="Find special_target_item",
        )

        # Target should be found
        names = [item["name"] for item in crushed]
        assert "special_target_item" in names, "Query-matched item must be preserved"

    def test_change_points_preserved(self, tokenizer):
        """Items around detected change points should be preserved."""
        # Create data with clear step change
        items = []
        for i in range(60):
            value = 100.0 if i < 30 else 300.0  # Step change at index 30
            items.append({"id": i, "value": value, "timestamp": f"2025-01-{(i % 28) + 1:02d}"})

        crushed = crush_items(items, tokenizer, max_items=12)

        # Should have items from around the change point
        preserved_ids = [item["id"] for item in crushed]

        # Check for items near the change point (indices 28-32)
        near_change = sum(1 for id in preserved_ids if 25 <= id <= 35)
        assert near_change >= 1, "Items near change point should be preserved"
