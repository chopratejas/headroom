"""Tests for universal JSON compression (all array types).

Verifies that SmartCrusher handles arrays of dicts, strings, numbers,
mixed types, and nested arrays — with consistent safety guarantees
across all types.
"""

from __future__ import annotations

import json

import pytest

from headroom.transforms.smart_crusher import (
    ArrayType,
    SmartCrusher,
    SmartCrusherConfig,
    _classify_array,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def crusher():
    """SmartCrusher configured for testing."""
    return SmartCrusher(
        config=SmartCrusherConfig(
            min_items_to_analyze=5,
            min_tokens_to_crush=0,  # Always crush
            max_items_after_crush=15,
        )
    )


@pytest.fixture
def crusher_large_k():
    """SmartCrusher with higher max items for larger test arrays."""
    return SmartCrusher(
        config=SmartCrusherConfig(
            min_items_to_analyze=5,
            min_tokens_to_crush=0,
            max_items_after_crush=50,
        )
    )


# =====================================================================
# Type Classification
# =====================================================================


class TestClassifyArray:
    def test_dict_array(self):
        assert _classify_array([{"a": 1}, {"b": 2}]) == ArrayType.DICT_ARRAY

    def test_string_array(self):
        assert _classify_array(["hello", "world", "foo"]) == ArrayType.STRING_ARRAY

    def test_number_array_int(self):
        assert _classify_array([1, 2, 3]) == ArrayType.NUMBER_ARRAY

    def test_number_array_float(self):
        assert _classify_array([1.0, 2.5, 3.7]) == ArrayType.NUMBER_ARRAY

    def test_number_array_mixed_int_float(self):
        assert _classify_array([1, 2.5, 3]) == ArrayType.NUMBER_ARRAY

    def test_bool_array(self):
        assert _classify_array([True, False, True]) == ArrayType.BOOL_ARRAY

    def test_nested_array(self):
        assert _classify_array([[1, 2], [3, 4]]) == ArrayType.NESTED_ARRAY

    def test_mixed_array(self):
        assert _classify_array([{"a": 1}, "string", 42]) == ArrayType.MIXED_ARRAY

    def test_empty(self):
        assert _classify_array([]) == ArrayType.EMPTY

    def test_single_dict(self):
        assert _classify_array([{"key": "val"}]) == ArrayType.DICT_ARRAY

    def test_single_string(self):
        assert _classify_array(["only"]) == ArrayType.STRING_ARRAY

    def test_none_values(self):
        # Arrays with None mixed in are MIXED
        assert _classify_array([1, None, 3]) == ArrayType.MIXED_ARRAY

    def test_bool_not_confused_with_int(self):
        # Python's True/False are int subclasses — we handle this
        assert _classify_array([True, False]) == ArrayType.BOOL_ARRAY
        # But mixed bools and ints should be MIXED or NUMBER depending on impl
        result = _classify_array([True, 42])
        assert result in (ArrayType.MIXED_ARRAY, ArrayType.NUMBER_ARRAY)


# =====================================================================
# String Array Compression
# =====================================================================


class TestCrushStringArray:
    def test_basic_compression(self, crusher):
        strings = [f"item_{i}" for i in range(100)]
        crushed, strategy = crusher._crush_string_array(strings)
        assert len(crushed) < len(strings)
        assert "string:adaptive" in strategy

    def test_errors_always_preserved(self, crusher):
        strings = ["ok"] * 50 + ["error: connection timeout", "failed: auth denied"] + ["ok"] * 48
        crushed, strategy = crusher._crush_string_array(strings)
        assert any("error" in s for s in crushed)
        assert any("failed" in s for s in crushed)

    def test_first_last_kept(self, crusher):
        strings = [f"item_{i}" for i in range(50)]
        crushed, strategy = crusher._crush_string_array(strings)
        # First item always present
        assert strings[0] in crushed
        # Last item always present
        assert strings[-1] in crushed

    def test_dedup_reduces_output(self, crusher):
        # 95 identical + 5 unique
        strings = ["repeated_value"] * 95 + [f"unique_{i}" for i in range(5)]
        crushed, strategy = crusher._crush_string_array(strings)
        # Should massively reduce — not keep 95 copies
        assert len(crushed) < 20
        # All 5 unique values should survive (they have high info value)
        for i in range(5):
            assert f"unique_{i}" in crushed

    def test_below_threshold_passthrough(self, crusher):
        strings = ["a", "b", "c"]  # Below min_items_to_analyze=5
        # Direct method call — should passthrough since <= 8
        crushed, strategy = crusher._crush_string_array(strings)
        assert crushed == strings
        assert "passthrough" in strategy

    def test_empty_strings_handled(self, crusher):
        strings = [""] * 20
        crushed, strategy = crusher._crush_string_array(strings)
        # Should not crash
        assert isinstance(crushed, list)

    def test_unicode_strings(self, crusher):
        strings = [f"日本語テスト_{i}" for i in range(50)]
        crushed, strategy = crusher._crush_string_array(strings)
        assert len(crushed) < len(strings)
        assert all(isinstance(s, str) for s in crushed)

    def test_length_anomalies_preserved(self, crusher_large_k):
        # Most strings are short, one is very long
        strings = ["short"] * 95 + ["x" * 10000] + ["short"] * 4
        crushed, strategy = crusher_large_k._crush_string_array(strings)
        assert any(len(s) > 1000 for s in crushed)


# =====================================================================
# Number Array Compression
# =====================================================================


class TestCrushNumberArray:
    def test_basic_compression(self, crusher):
        numbers = [42.0 + i * 0.1 for i in range(100)]
        crushed, strategy = crusher._crush_number_array(numbers)
        assert len(crushed) < len(numbers)
        assert "number:adaptive" in strategy

    def test_summary_prepended(self, crusher):
        numbers = list(range(100))
        crushed, strategy = crusher._crush_number_array(numbers)
        # First element should be the stats summary string
        assert isinstance(crushed[0], str)
        assert "numbers:" in crushed[0]
        assert "min=" in crushed[0]
        assert "max=" in crushed[0]

    def test_outliers_preserved(self, crusher):
        # Normal values around 50 with one extreme outlier
        numbers = [50.0 + i * 0.01 for i in range(100)] + [999.9]
        crushed, strategy = crusher._crush_number_array(numbers)
        assert 999.9 in crushed
        assert "outliers" in strategy

    def test_all_identical(self, crusher):
        numbers = [42.0] * 100
        crushed, strategy = crusher._crush_number_array(numbers)
        # With all identical, should compress heavily
        # Summary + a few representatives
        numeric_values = [v for v in crushed if isinstance(v, (int, float))]
        assert all(v == 42.0 for v in numeric_values)

    def test_first_last_kept(self, crusher):
        numbers = list(range(50))
        crushed, strategy = crusher._crush_number_array(numbers)
        numeric_values = [v for v in crushed if isinstance(v, (int, float))]
        assert 0 in numeric_values  # First
        assert 49 in numeric_values  # Last

    def test_change_point_preserved(self, crusher_large_k):
        # Stable at 10, then jumps to 100
        numbers = [10.0] * 50 + [100.0] * 50
        crushed, strategy = crusher_large_k._crush_number_array(numbers)
        numeric_values = [v for v in crushed if isinstance(v, (int, float))]
        # Both 10.0 and 100.0 should be present
        assert 10.0 in numeric_values
        assert 100.0 in numeric_values

    def test_nan_inf_filtered(self, crusher):
        numbers = [1.0, 2.0, float("nan"), float("inf"), 3.0] * 10
        crushed, strategy = crusher._crush_number_array(numbers)
        # Should not crash; stats should be based on finite values
        assert isinstance(crushed[0], str)

    def test_integers_preserved_as_int(self, crusher):
        numbers = list(range(50))
        crushed, strategy = crusher._crush_number_array(numbers)
        numeric_values = [v for v in crushed if isinstance(v, (int, float))]
        # Integers should remain integers (not converted to float)
        assert any(isinstance(v, int) for v in numeric_values)

    def test_statistics_accuracy(self, crusher):
        numbers = list(range(1, 101))  # 1 to 100
        crushed, strategy = crusher._crush_number_array(numbers)
        summary = crushed[0]
        assert "min=1" in summary
        assert "max=100" in summary
        assert "mean=50.5" in summary


# =====================================================================
# Mixed Array Compression
# =====================================================================


class TestCrushMixedArray:
    def test_basic_compression(self, crusher_large_k):
        mixed = [{"id": i} for i in range(30)] + [f"msg_{i}" for i in range(30)]
        crushed, strategy = crusher_large_k._crush_mixed_array(mixed)
        assert len(crushed) < len(mixed)
        assert "mixed:adaptive" in strategy

    def test_small_groups_kept(self, crusher):
        # 50 dicts + 3 strings (below threshold)
        mixed = [{"id": i} for i in range(50)] + ["rare_1", "rare_2", "rare_3"]
        crushed, strategy = crusher._crush_mixed_array(mixed)
        # All 3 rare strings should be kept (below min_items threshold)
        assert "rare_1" in crushed
        assert "rare_2" in crushed
        assert "rare_3" in crushed

    def test_errors_across_types(self, crusher_large_k):
        mixed = (
            [{"status": "ok"}] * 30
            + [{"status": "error: timeout"}]
            + ["error: auth failed"]
            + [f"ok_{i}" for i in range(30)]
        )
        crushed, strategy = crusher_large_k._crush_mixed_array(mixed)
        crushed_str = json.dumps(crushed)
        assert "error: timeout" in crushed_str
        assert "error: auth failed" in crushed_str

    def test_original_order_preserved(self, crusher_large_k):
        mixed = [{"id": i} for i in range(20)] + [f"str_{i}" for i in range(20)]
        crushed, strategy = crusher_large_k._crush_mixed_array(mixed)
        # Dicts should come before strings (original order)
        first_str_idx = next(
            (i for i, item in enumerate(crushed) if isinstance(item, str)), len(crushed)
        )
        last_dict_idx = max(
            (i for i, item in enumerate(crushed) if isinstance(item, dict)), default=-1
        )
        assert last_dict_idx < first_str_idx

    def test_passthrough_small(self, crusher):
        mixed = [1, "two", {"three": 3}]
        crushed, strategy = crusher._crush_mixed_array(mixed)
        assert crushed == mixed
        assert "passthrough" in strategy


# =====================================================================
# Adaptive K
# =====================================================================


class TestAdaptiveK:
    def test_scales_with_n(self, crusher):
        """K grows sublinearly with collection size."""
        small = [f"item_{i}" for i in range(20)]
        large = [f"item_{i}" for i in range(500)]

        k_small = crusher._compute_k_split(small)[0]
        k_large = crusher._compute_k_split(large)[0]

        assert k_large > k_small
        # Should be sublinear: ratio of K should be much less than ratio of n
        assert k_large / k_small < 500 / 20

    def test_respects_max_items(self, crusher):
        items = [f"item_{i}" for i in range(1000)]
        k_total, _, _, _ = crusher._compute_k_split(items)
        assert k_total <= crusher.config.max_items_after_crush

    def test_first_last_fractions(self, crusher):
        items = [f"item_{i}" for i in range(100)]
        k_total, k_first, k_last, k_importance = crusher._compute_k_split(items)
        # First and last should be roughly the configured fractions
        assert k_first >= 1
        assert k_last >= 1
        assert k_first + k_last + k_importance == k_total

    def test_homogeneous_vs_diverse(self, crusher_large_k):
        """Homogeneous data should produce smaller K than diverse data."""
        homogeneous = ["same value"] * 100
        diverse = [f"unique_value_{i}_{'x' * (i * 10)}" for i in range(100)]

        k_homo = crusher_large_k._compute_k_split(homogeneous)[0]
        k_diverse = crusher_large_k._compute_k_split(diverse)[0]

        # Diverse should need more items (or at least equal)
        assert k_diverse >= k_homo


# =====================================================================
# Safety Guarantees (parametrized across types)
# =====================================================================


class TestSafetyGuarantees:
    @pytest.mark.parametrize(
        "items,item_type",
        [
            ([{"status": "ok"}] * 50 + [{"status": "error: timeout"}], "dict"),
            (["ok"] * 50 + ["error: connection failed"], "string"),
            (
                [{"id": i} for i in range(30)]
                + ["error: auth denied"]
                + [f"ok_{i}" for i in range(30)],
                "mixed",
            ),
        ],
        ids=["dict_array", "string_array", "mixed_array"],
    )
    def test_errors_never_dropped(self, crusher_large_k, items, item_type):
        """Error items must be preserved regardless of array type."""
        if item_type == "dict":
            crushed, _ = crusher_large_k._crush_string_array([json.dumps(i) for i in items])
            crushed_text = " ".join(crushed)
        elif item_type == "string":
            crushed, _ = crusher_large_k._crush_string_array(items)
            crushed_text = " ".join(crushed)
        elif item_type == "mixed":
            crushed, _ = crusher_large_k._crush_mixed_array(items)
            crushed_text = json.dumps(crushed)

        assert "error" in crushed_text.lower()

    @pytest.mark.parametrize(
        "items",
        [
            [f"item_{i}" for i in range(50)],
            list(range(50)),
        ],
        ids=["string_array", "number_array"],
    )
    def test_first_last_always_present(self, crusher, items):
        """First and last items must be present in output."""
        if isinstance(items[0], str):
            crushed, _ = crusher._crush_string_array(items)
            assert items[0] in crushed
            assert items[-1] in crushed
        else:
            crushed, _ = crusher._crush_number_array(items)
            numeric = [v for v in crushed if isinstance(v, (int, float))]
            assert items[0] in numeric
            assert items[-1] in numeric

    @pytest.mark.parametrize(
        "items",
        [
            ["a", "b", "c"],
            [1, 2, 3],
            [{"x": 1}, "y", 2],
        ],
        ids=["string_below", "number_below", "mixed_below"],
    )
    def test_passthrough_below_min_items(self, crusher, items):
        """Arrays below min_items_to_analyze pass through unchanged."""
        if all(isinstance(i, str) for i in items):
            crushed, strategy = crusher._crush_string_array(items)
        elif all(isinstance(i, (int, float)) for i in items):
            crushed, strategy = crusher._crush_number_array(items)
        else:
            crushed, strategy = crusher._crush_mixed_array(items)
        assert "passthrough" in strategy


# =====================================================================
# Integration: Full pipeline
# =====================================================================


class TestFullPipelineIntegration:
    """Test that new types work through the compress() API."""

    def test_string_array_via_compress(self):
        from headroom import compress

        strings = [f"log line {i}: GET /api 200" for i in range(100)]
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Show logs"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_logs", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(strings)},
        ]
        result = compress(messages)
        assert result.tokens_saved > 0
        assert result.compression_ratio > 0

    def test_number_array_via_compress(self):
        from headroom import compress

        numbers = [42.0 + i * 0.1 for i in range(200)]
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Show metrics"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_metrics", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(numbers)},
        ]
        result = compress(messages)
        assert result.tokens_saved > 0

    def test_dict_array_unchanged(self):
        """Verify dict arrays still work (regression test)."""
        from headroom import compress

        data = [{"id": i, "name": f"user_{i}", "status": "active"} for i in range(100)]
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "List users"},
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
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        result = compress(messages)
        assert result.tokens_saved > 0
        assert result.compression_ratio > 0
