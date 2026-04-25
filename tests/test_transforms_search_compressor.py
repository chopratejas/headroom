from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.transforms.search_compressor import (
    FileMatches,
    SearchCompressionResult,
    SearchCompressor,
    SearchCompressorConfig,
    SearchMatch,
)


def test_parse_score_select_and_format_search_results(monkeypatch: pytest.MonkeyPatch) -> None:
    compressor = SearchCompressor(
        SearchCompressorConfig(
            max_matches_per_file=3,
            max_total_matches=4,
            max_files=2,
            context_keywords=["auth"],
        )
    )
    content = "\n".join(
        [
            "src/auth.py:10:ERROR auth failed",
            "src/auth.py-11-warning auth retry",
            "src/auth.py:12:plain auth line",
            "src/db.py:2:warning token expired",
            "not a match",
        ]
    )
    parsed = compressor._parse_search_results(content)
    assert set(parsed) == {"src/auth.py", "src/db.py"}
    assert parsed["src/auth.py"].first == SearchMatch(
        file="src/auth.py", line_number=10, content="ERROR auth failed"
    )
    assert parsed["src/auth.py"].last.line_number == 12

    compressor._score_matches(parsed, "find auth error")
    assert parsed["src/auth.py"].matches[0].score == 1.0
    assert parsed["src/db.py"].matches[0].score > 0

    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.transforms.adaptive_sizer",
        SimpleNamespace(compute_optimal_k=lambda items, **kwargs: 4),
    )
    selected = compressor._select_matches(parsed, bias=1.2)
    assert list(selected) == ["src/auth.py", "src/db.py"]
    assert [m.line_number for m in selected["src/auth.py"].matches] == [10, 11, 12]

    formatted, summaries = compressor._format_output(
        selected,
        {
            **parsed,
            "src/db.py": FileMatches(
                file="src/db.py",
                matches=[
                    SearchMatch(file="src/db.py", line_number=2, content="warning token expired"),
                    SearchMatch(file="src/db.py", line_number=3, content="another line"),
                ],
            ),
        },
    )
    assert "src/auth.py:10:ERROR auth failed" in formatted
    assert summaries["src/db.py"] == "[... and 1 more matches in src/db.py]"


def test_search_compressor_compress_paths_and_ccr(monkeypatch: pytest.MonkeyPatch) -> None:
    compressor = SearchCompressor(
        SearchCompressorConfig(enable_ccr=True, min_matches_for_ccr=2, context_keywords=["auth"])
    )
    no_match = compressor.compress("plain text only")
    assert no_match.original_match_count == 0
    assert no_match.compressed == "plain text only"

    parsed = {
        "src/auth.py": FileMatches(
            file="src/auth.py",
            matches=[
                SearchMatch(file="src/auth.py", line_number=1, content="auth error"),
                SearchMatch(file="src/auth.py", line_number=2, content="auth ok"),
            ],
        )
    }
    monkeypatch.setattr(compressor, "_parse_search_results", lambda content: parsed)
    monkeypatch.setattr(compressor, "_score_matches", lambda file_matches, context: None)
    monkeypatch.setattr(compressor, "_select_matches", lambda file_matches, bias=1.0: parsed)
    monkeypatch.setattr(
        compressor,
        "_format_output",
        lambda selected, original: ("short", {"src/auth.py": "summary"}),
    )
    monkeypatch.setattr(compressor, "_store_in_ccr", lambda original, compressed, count: "abc123")

    result = compressor.compress("raw search", context="auth", bias=0.8)
    assert result.original_match_count == 2
    assert result.compressed_match_count == 2
    assert result.cache_key == "abc123"
    assert result.summaries == {"src/auth.py": "summary"}
    assert result.compressed.endswith("[2 matches compressed to 2. Retrieve more: hash=abc123]")

    monkeypatch.setattr(compressor, "_store_in_ccr", lambda original, compressed, count: None)
    no_cache = compressor.compress("raw search", context="auth")
    assert no_cache.cache_key is None
    assert no_cache.compressed == "short"


def test_store_in_ccr_and_result_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    compressor = SearchCompressor()
    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.cache.compression_store",
        SimpleNamespace(
            get_compression_store=lambda: SimpleNamespace(
                store=lambda original, compressed, original_item_count=0: "stored-key"
            )
        ),
    )
    assert compressor._store_in_ccr("orig", "comp", 5) == "stored-key"

    def broken_store():
        raise RuntimeError("boom")

    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.cache.compression_store",
        SimpleNamespace(get_compression_store=broken_store),
    )
    assert compressor._store_in_ccr("orig", "comp", 5) is None

    result = SearchCompressionResult(
        compressed="tiny",
        original="this is a much longer original string",
        original_match_count=10,
        compressed_match_count=4,
        files_affected=2,
        compression_ratio=0.3,
    )
    assert result.tokens_saved_estimate > 0
    assert result.matches_omitted == 6
