"""Regression tests for SmartCrusher bugs.

Bug 1: _crush_number_array mixes types (string summary + numbers),
       violating the schema-preserving guarantee.
Bug 2: _current_field_semantics is shared instance state, creating
       a race condition when crushing concurrently.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from headroom import SmartCrusherConfig
from headroom.transforms.smart_crusher import SmartCrusher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_crusher(max_items: int = 10, min_items: int = 3) -> SmartCrusher:
    config = SmartCrusherConfig(
        enabled=True,
        min_items_to_analyze=min_items,
        min_tokens_to_crush=0,
        max_items_after_crush=max_items,
        variance_threshold=2.0,
    )
    return SmartCrusher(config=config)


# ---------------------------------------------------------------------------
# Bug 1: Number array type mixing
# ---------------------------------------------------------------------------


class TestNumberArraySchemaPreservation:
    """_crush_number_array must return only original numeric values.

    Previously it prepended a stats summary string, producing
    [string, int, int, ...] which violates the schema-preserving
    guarantee and breaks type-aware JSON consumers.
    """

    def test_crushed_number_array_contains_only_numbers(self) -> None:
        """Every element of the crushed array must be int or float."""
        crusher = _make_crusher(max_items=10)
        numbers = list(range(50))  # 0..49, well above the n<=8 passthrough
        crushed, strategy = crusher._crush_number_array(numbers)

        for i, item in enumerate(crushed):
            assert isinstance(item, int | float), (
                f"Item {i} is {type(item).__name__} = {item!r}, expected int/float. "
                f"Schema-preserving guarantee violated."
            )

    def test_crushed_number_array_subset_of_original(self) -> None:
        """Every value in the crushed array must exist in the original."""
        crusher = _make_crusher(max_items=10)
        numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        crushed, _ = crusher._crush_number_array(numbers)

        original_set = set(numbers)
        for item in crushed:
            assert item in original_set, (
                f"Value {item!r} not in original array — generated content detected"
            )

    def test_stats_summary_in_strategy_not_in_array(self) -> None:
        """Statistics should be communicated via strategy string, not array content."""
        crusher = _make_crusher(max_items=5)
        numbers = list(range(100))
        crushed, strategy = crusher._crush_number_array(numbers)

        # Strategy should contain stats info
        assert "number:" in strategy

        # Array should not contain any strings
        strings_in_result = [x for x in crushed if isinstance(x, str)]
        assert strings_in_result == [], f"Found string(s) in numeric array: {strings_in_result}"

    def test_number_array_passthrough_for_small(self) -> None:
        """Arrays with n <= 8 should pass through unchanged."""
        crusher = _make_crusher()
        small = [1, 2, 3, 4, 5]
        crushed, strategy = crusher._crush_number_array(small)
        assert crushed == small
        assert strategy == "number:passthrough"

    def test_number_array_preserves_outliers(self) -> None:
        """Outlier values should be preserved in the crushed output."""
        crusher = _make_crusher(max_items=10)
        # Normal range + extreme outlier
        numbers = [10] * 20 + [10000]
        crushed, strategy = crusher._crush_number_array(numbers)
        assert 10000 in crushed, "Outlier value 10000 was dropped"

    def test_number_array_preserves_boundaries(self) -> None:
        """First and last values should always be kept."""
        crusher = _make_crusher(max_items=5)
        numbers = list(range(100))
        crushed, strategy = crusher._crush_number_array(numbers)
        assert crushed[0] == 0, "First value not preserved"
        assert numbers[-1] in crushed, "Last value not preserved"

    def test_non_finite_passthrough(self) -> None:
        """All-NaN/Inf arrays should return unchanged."""
        crusher = _make_crusher()
        nans = [float("nan")] * 10
        crushed, strategy = crusher._crush_number_array(nans)
        assert strategy == "number:no_finite"
        assert len(crushed) == 10

    def test_full_crush_pipeline_number_array_types(self) -> None:
        """End-to-end: crushing a JSON number array via the public API."""
        crusher = _make_crusher(max_items=10)
        content = json.dumps(list(range(50)))
        result, was_modified, info = crusher._smart_crush_content(content)

        if was_modified:
            parsed = json.loads(result)
            assert isinstance(parsed, list)
            for item in parsed:
                assert isinstance(item, int | float), (
                    f"Public API returned non-numeric item {item!r} in number array"
                )


# ---------------------------------------------------------------------------
# Bug 2: Race condition on _current_field_semantics
# ---------------------------------------------------------------------------


class TestFieldSemanticsThreadSafety:
    """_current_field_semantics must not leak between concurrent crushes.

    Previously it was stored as instance state (self._current_field_semantics)
    which created a race condition when the same SmartCrusher instance
    was used from multiple threads.
    """

    def test_concurrent_crushes_no_cross_contamination(self) -> None:
        """Two concurrent crushes must not share field_semantics state."""
        crusher = _make_crusher(max_items=5)

        # Two different array payloads
        payload_a = json.dumps([{"name": f"item_{i}", "value": i} for i in range(20)])
        payload_b = json.dumps([{"key": f"k_{i}", "score": i * 0.1} for i in range(20)])

        results: dict[str, str] = {}
        errors: list[Exception] = []

        def crush_task(label: str, content: str) -> None:
            try:
                result, modified, info = crusher._smart_crush_content(content)
                results[label] = result
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            # Run many concurrent crushes to increase race probability
            for i in range(20):
                futures.append(executor.submit(crush_task, f"a_{i}", payload_a))
                futures.append(executor.submit(crush_task, f"b_{i}", payload_b))
            for f in as_completed(futures):
                f.result()  # Re-raise exceptions

        assert not errors, f"Concurrent crushes raised errors: {errors}"

        # After all crushes, thread-local state must be clean
        tl = getattr(crusher, "_thread_local", None)
        if tl is not None:
            semantics = getattr(tl, "field_semantics", None)
            assert semantics is None, f"field_semantics leaked in thread-local: {semantics}"


# ---------------------------------------------------------------------------
# Issue 7: Recursion depth limit
# ---------------------------------------------------------------------------


class TestRecursionDepthLimit:
    """_process_value must not crash on deeply nested JSON."""

    def test_deeply_nested_json_does_not_crash(self) -> None:
        """Nesting deeper than _MAX_PROCESS_DEPTH should return value unchanged."""
        crusher = _make_crusher()
        # Build a 100-level nested structure
        nested: dict = {"leaf": "value"}
        for _i in range(100):
            nested = {"level": nested}

        content = json.dumps(nested)
        result, was_modified, info = crusher._smart_crush_content(content)
        # Should not raise RecursionError
        parsed = json.loads(result)
        # The deep structure should be preserved (returned as-is past depth limit)
        assert isinstance(parsed, dict)

    def test_deeply_nested_list_does_not_crash(self) -> None:
        """Deeply nested lists should also be handled safely."""
        crusher = _make_crusher()
        nested: list = ["leaf"]
        for _i in range(100):
            nested = [nested]

        content = json.dumps(nested)
        result, was_modified, info = crusher._smart_crush_content(content)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
