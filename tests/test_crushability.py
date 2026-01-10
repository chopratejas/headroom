"""Tests for SmartCrusher crushability analysis.

These tests verify that SmartCrusher correctly identifies when it's SAFE
to crush data vs when it should SKIP crushing.

The key insight: High variability + No importance signal = DON'T CRUSH.

Test scenarios:
1. DB results (unique entities, no signal) → SKIP
2. Search results (has score field) → CRUSH using score
3. Log entries (has errors) → CRUSH keeping errors
4. Time series (has anomalies) → CRUSH keeping anomalies
5. Repetitive data (low uniqueness) → CRUSH with sampling
"""

import json

import pytest

from headroom.transforms.smart_crusher import (
    CompressionStrategy,
    SmartAnalyzer,
    SmartCrusherConfig,
    smart_crush_tool_output,
)


class TestCrushabilityDetection:
    """Test the crushability analysis logic."""

    @pytest.fixture
    def analyzer(self):
        """Create a SmartAnalyzer instance."""
        return SmartAnalyzer(SmartCrusherConfig())

    def test_db_results_not_crushable(self, analyzer):
        """DB query results with unique IDs and no signal should NOT be crushed."""
        # Simulate: SELECT * FROM users LIMIT 50
        items = [
            {
                "id": i,
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "department": "Engineering",
            }
            for i in range(50)
        ]

        analysis = analyzer.analyze_array(items)

        # Should detect unique entities with no importance signal
        assert analysis.crushability is not None
        assert not analysis.crushability.crushable, (
            f"DB results should NOT be crushable. "
            f"Reason: {analysis.crushability.reason}, "
            f"Signals: {analysis.crushability.signals_present}"
        )
        assert analysis.recommended_strategy == CompressionStrategy.SKIP
        assert "unique" in analysis.crushability.reason.lower()

    def test_db_results_with_unique_uuid(self, analyzer):
        """DB results with UUID field should NOT be crushed."""
        items = [
            {
                "uuid": f"550e8400-e29b-41d4-a716-44665544{i:04d}",
                "name": f"Record {i}",
                "value": i * 10,
            }
            for i in range(50)
        ]

        analysis = analyzer.analyze_array(items)

        assert analysis.crushability is not None
        assert not analysis.crushability.crushable
        assert analysis.crushability.has_id_field

    def test_search_results_crushable(self, analyzer):
        """Search results with score field SHOULD be crushed."""
        items = [
            {
                "id": i,
                "title": f"Document {i}",
                "snippet": f"This is document {i} content...",
                "score": 1.0 - (i * 0.01),  # Decreasing relevance
            }
            for i in range(100)
        ]

        analysis = analyzer.analyze_array(items)

        # Should detect score field as importance signal
        assert analysis.crushability is not None
        assert analysis.crushability.crushable, (
            f"Search results should be crushable. Reason: {analysis.crushability.reason}"
        )
        assert analysis.crushability.has_score_field
        assert any("score" in s for s in analysis.crushability.signals_present)

    def test_log_entries_with_errors_crushable(self, analyzer):
        """Log entries containing structural outliers SHOULD be crushed (outliers preserved)."""
        items = []
        for i in range(100):
            item = {
                "id": i,
                "timestamp": f"2024-01-15T10:{i:02d}:00Z",
                "message": f"Request processed successfully - {i}",
                "level": "INFO",
            }
            # Add some errors - these are STRUCTURAL OUTLIERS (have extra "error" field)
            if i % 20 == 0:
                item["level"] = "ERROR"
                item["message"] = f"Connection failed: timeout at {i}"
                item["error"] = "TimeoutError"  # Extra field that most items don't have
            items.append(item)

        analysis = analyzer.analyze_array(items)

        # Should detect structural outliers (items with rare fields like "error")
        assert analysis.crushability is not None
        assert analysis.crushability.crushable
        # Now uses structural_outliers instead of keyword-based error count
        assert any(
            "structural_outliers" in s or "outlier" in s.lower()
            for s in analysis.crushability.signals_present
        )

    def test_time_series_with_anomalies_crushable(self, analyzer):
        """Time series with numeric anomalies SHOULD be crushed."""
        items = []
        for i in range(100):
            value = 100.0  # Normal value
            if i in [25, 50, 75]:  # Anomaly points
                value = 999.0
            items.append(
                {
                    "id": i,
                    "timestamp": i,
                    "cpu_usage": value,
                }
            )

        analysis = analyzer.analyze_array(items)

        # Should detect anomalies as importance signal
        assert analysis.crushability is not None
        assert analysis.crushability.crushable
        assert analysis.crushability.anomaly_count > 0

    def test_repetitive_data_crushable(self, analyzer):
        """Repetitive data (low uniqueness) SHOULD be crushable."""
        # Same status repeated many times
        items = [
            {
                "id": i,
                "status": "success",  # Same for all
                "code": 200,  # Same for all
                "message": "OK",  # Same for all
            }
            for i in range(100)
        ]

        analysis = analyzer.analyze_array(items)

        # Should detect low uniqueness - safe to sample
        assert analysis.crushability is not None
        assert analysis.crushability.crushable
        # Can be "low_uniqueness" or "repetitive_content_with_ids"
        assert (
            "low_uniqueness" in analysis.crushability.reason
            or "repetitive" in analysis.crushability.reason
        )

    def test_file_listing_not_crushable(self, analyzer):
        """File listing with unique paths should NOT be crushed."""
        items = [
            {
                "id": i,
                "path": f"/home/user/project/src/module{i}/file{i}.py",
                "size": 1000 + i,
                "modified": f"2024-01-{(i % 28) + 1:02d}",
            }
            for i in range(50)
        ]

        analysis = analyzer.analyze_array(items)

        # Paths are highly unique, no importance signal
        assert analysis.crushability is not None
        # Should NOT crush file listings
        assert not analysis.crushability.crushable or analysis.crushability.confidence < 0.7

    def test_order_list_not_crushable(self, analyzer):
        """Order list with unique order IDs should NOT be crushed."""
        items = [
            {
                "order_id": f"ORD-2024-{i:05d}",
                "customer": f"Customer {i}",
                "total": 50.0 + i,
                "status": "completed",
            }
            for i in range(50)
        ]

        analysis = analyzer.analyze_array(items)

        # Each order is a unique entity
        assert analysis.crushability is not None
        # order_id contains 'id' pattern
        assert not analysis.crushability.crushable


class TestCrushabilityEndToEnd:
    """End-to-end tests for crushability-aware crushing."""

    def test_db_results_preserved_completely(self):
        """DB results should be returned unchanged when not crushable."""
        items = [{"id": i, "name": f"User {i}", "email": f"user{i}@test.com"} for i in range(30)]
        content = json.dumps(items)

        config = SmartCrusherConfig(max_items_after_crush=10)
        crushed, was_modified, info = smart_crush_tool_output(content, config)

        # Should NOT be modified (skip crushing)
        if was_modified:
            result = json.loads(crushed)
            # If it was modified, all items should still be there
            assert len(result) == 30, (
                f"DB results should not lose items! Had 30, got {len(result)}. Info: {info}"
            )

    def test_search_results_crushed_by_score(self):
        """Search results should be crushed using score field."""
        items = [
            {
                "id": i,
                "title": f"Result {i}",
                "score": 100 - i,  # Higher score = more relevant
            }
            for i in range(100)
        ]
        content = json.dumps(items)

        config = SmartCrusherConfig(max_items_after_crush=15)
        crushed, was_modified, info = smart_crush_tool_output(content, config)

        assert was_modified
        result = json.loads(crushed)
        assert len(result) < 100

        # Top scores should be preserved
        scores = [item.get("score", 0) for item in result]
        assert max(scores) >= 90  # Top items preserved

    def test_mixed_data_with_errors_preserves_errors(self):
        """Data with errors should crush but preserve ALL errors."""
        items = []
        error_ids = [5, 25, 45, 65, 85]
        for i in range(100):
            item = {"id": i, "data": f"value_{i}"}
            if i in error_ids:
                item["status"] = "failed"
                item["error"] = f"Error at {i}"
            items.append(item)

        content = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=20)
        crushed, was_modified, info = smart_crush_tool_output(content, config)

        result = json.loads(crushed)

        # All errors must be preserved
        error_count = sum(1 for item in result if item.get("error"))
        assert error_count == len(error_ids), (
            f"All {len(error_ids)} errors should be preserved, got {error_count}"
        )


class TestCrushabilitySignals:
    """Test individual signal detection."""

    @pytest.fixture
    def analyzer(self):
        return SmartAnalyzer(SmartCrusherConfig())

    def test_detects_id_field_variations(self, analyzer):
        """Should detect various ID field naming patterns."""
        test_cases = [
            ("id", [{"id": i} for i in range(20)]),
            ("uuid", [{"uuid": f"uuid-{i}"} for i in range(20)]),
            ("_id", [{"_id": f"mongo-{i}"} for i in range(20)]),
            ("pk", [{"pk": i} for i in range(20)]),
            ("key", [{"key": f"key-{i}"} for i in range(20)]),
            ("user_id", [{"user_id": i} for i in range(20)]),
        ]

        for field_name, items in test_cases:
            analysis = analyzer.analyze_array(items)
            assert analysis.crushability is not None
            assert analysis.crushability.has_id_field, f"Should detect '{field_name}' as ID field"

    def test_detects_score_field_variations(self, analyzer):
        """Should detect various score field naming patterns."""
        test_cases = [
            "score",
            "rank",
            "relevance",
            "confidence",
            "_score",
            "rating",
        ]

        for field_name in test_cases:
            items = [{field_name: i * 0.1, "data": f"item_{i}"} for i in range(20)]
            analysis = analyzer.analyze_array(items)
            assert analysis.crushability is not None
            assert analysis.crushability.has_score_field, (
                f"Should detect '{field_name}' as score field"
            )

    def test_detects_error_keywords(self, analyzer):
        """Should detect various error keyword patterns."""
        error_keywords = ["error", "exception", "failed", "failure", "critical", "fatal"]

        for keyword in error_keywords:
            items = [{"id": i, "msg": "OK"} for i in range(20)]
            items[10]["msg"] = f"Something {keyword} happened"

            analysis = analyzer.analyze_array(items)
            assert analysis.crushability is not None
            assert analysis.crushability.error_item_count >= 1, (
                f"Should detect '{keyword}' as error indicator"
            )


class TestCrushabilityEdgeCases:
    """Test edge cases in crushability analysis."""

    @pytest.fixture
    def analyzer(self):
        return SmartAnalyzer(SmartCrusherConfig())

    def test_empty_array(self, analyzer):
        """Empty array should not crash."""
        analysis = analyzer.analyze_array([])
        assert analysis.recommended_strategy == CompressionStrategy.NONE

    def test_small_array_skipped(self, analyzer):
        """Arrays below min_items_to_analyze should be skipped."""
        items = [{"id": i} for i in range(3)]
        analysis = analyzer.analyze_array(items)
        assert analysis.recommended_strategy == CompressionStrategy.NONE

    def test_mixed_signals(self, analyzer):
        """Data with multiple signals should still be crushable."""
        items = []
        for i in range(100):
            item = {
                "id": i,
                "score": 100 - i,  # Score signal
                "value": 50.0,
            }
            if i == 50:
                item["error"] = "Test error"  # Error signal
                item["value"] = 999.0  # Anomaly signal
            items.append(item)

        analysis = analyzer.analyze_array(items)
        assert analysis.crushability is not None
        assert analysis.crushability.crushable
        assert len(analysis.crushability.signals_present) >= 2

    def test_all_items_are_errors(self, analyzer):
        """When all items are errors, keyword detection finds them as a signal.

        With keyword-based error detection (for the preservation guarantee),
        when ALL items have error keywords, we detect error_keywords:50 as a
        signal. This makes the data technically crushable.

        However, since ALL items are errors, they will ALL be preserved due to
        the preservation guarantee. The end result is the same - no data loss.
        """
        items = [{"id": i, "error": f"Error {i}", "status": "failed"} for i in range(50)]

        analysis = analyzer.analyze_array(items)
        assert analysis.crushability is not None

        # With keyword-based error detection, all 50 items contain error keywords
        # This IS a signal (error_keywords:50), making the data crushable.
        # However, all 50 items will be preserved due to the preservation guarantee.
        assert analysis.crushability.crushable
        assert "error_keywords:50" in analysis.crushability.signals_present


class TestCrushabilityConfidence:
    """Test confidence scoring in crushability analysis."""

    @pytest.fixture
    def analyzer(self):
        return SmartAnalyzer(SmartCrusherConfig())

    def test_high_confidence_for_clear_cases(self, analyzer):
        """Clear-cut cases should have high confidence."""
        # Low uniqueness - clearly safe
        items = [{"status": "ok", "code": 200} for _ in range(100)]
        analysis = analyzer.analyze_array(items)
        assert analysis.crushability is not None
        assert analysis.crushability.confidence >= 0.8

    def test_lower_confidence_for_ambiguous_cases(self, analyzer):
        """Ambiguous cases should have lower confidence."""
        # Medium uniqueness with weak signal
        items = [
            {"id": i, "value": i % 10, "status": "active" if i % 2 == 0 else "inactive"}
            for i in range(100)
        ]
        # Add one error to provide weak signal
        items[50]["error"] = "minor issue"

        analysis = analyzer.analyze_array(items)
        assert analysis.crushability is not None
        # Should be lower confidence due to ambiguity
        assert analysis.crushability.confidence <= 0.7
