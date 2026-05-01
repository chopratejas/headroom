from __future__ import annotations

from types import SimpleNamespace

from headroom.perf import analyzer
from headroom.perf.analyzer import (
    PerfRecord,
    PerfReport,
    RouterRecord,
    ToinRecord,
    TransformRecord,
    _generate_recommendations,
    _get_list_price,
    _litellm_cost,
    _parse_kv,
    _resolve_model,
    format_report,
    parse_log_files,
)


def test_litellm_helpers_and_kv_parsing(monkeypatch) -> None:
    analyzer._resolved_model_cache.clear()
    monkeypatch.setattr(analyzer, "_LITELLM_AVAILABLE", False)
    assert _resolve_model("claude-sonnet") == "claude-sonnet"
    assert _litellm_cost("claude-sonnet", 10) is None
    assert _get_list_price("claude-sonnet") is None

    fake_litellm = SimpleNamespace(
        model_cost={
            "anthropic/claude-sonnet": {"input_cost_per_token": 0.000002},
            "gpt-4o": {"input_cost_per_token": 0.000005},
        },
        cost_per_token=lambda **kwargs: (0.123, 0.0),
    )
    monkeypatch.setattr(analyzer, "_LITELLM_AVAILABLE", True)
    monkeypatch.setattr(analyzer, "_litellm", fake_litellm)
    analyzer._resolved_model_cache.clear()

    assert _resolve_model("gpt-4o") == "gpt-4o"
    assert _resolve_model("claude-sonnet") == "anthropic/claude-sonnet"
    assert _litellm_cost("claude-sonnet", 100, cache_read_tokens=10, cache_write_tokens=5) == 0.123
    assert _get_list_price("claude-sonnet") == 2.0

    monkeypatch.setattr(
        analyzer,
        "_litellm",
        SimpleNamespace(
            model_cost=fake_litellm.model_cost,
            cost_per_token=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
    )
    assert _litellm_cost("claude-sonnet", 100) is None

    parsed = _parse_kv(
        "model=claude msgs=3 tok_before=100 tok_after=50 transforms=router:excluded:tool*2 read_lifecycle:stale*1"
    )
    assert parsed == {
        "model": "claude",
        "msgs": "3",
        "tok_before": "100",
        "tok_after": "50",
        "transforms": "router:excluded:tool*2 read_lifecycle:stale*1",
    }


def test_parse_log_files_collects_perf_router_transform_and_toin_records(
    tmp_path, monkeypatch
) -> None:
    log_one = tmp_path / "proxy.log"
    log_one.write_text(
        "\n".join(
            [
                "2026-03-07 13:38:31,009 - headroom.proxy - INFO - [req-1] PERF model=claude msgs=3 tok_before=100 tok_after=40 tok_saved=60 cache_read=10 cache_write=5 cache_hit_pct=66 opt_ms=123 transforms=router:excluded:tool*2 read_lifecycle:stale*1",
                "2026-03-07 13:38:31,100 - headroom.proxy - INFO - content_router: 5 msgs — 2 compressed, 1 excluded, 1 skipped, 1 unchanged, 3 content blocks",
                "2026-03-07 13:38:31,200 - headroom.proxy - INFO - Transform content_router: 100 -> 40 tokens (saved 60)",
                "2026-03-07 13:38:31,300 - headroom.proxy - INFO - TOIN: 105 patterns, 3837 compressions, 0 retrievals, 0.0% retrieval rate",
            ]
        ),
        encoding="utf-8",
    )
    log_two = tmp_path / "proxy.log.1"
    log_two.write_text(
        "2026-03-07 13:39:31,009 - headroom.proxy - INFO - [req-2] PERF model=gpt-4o msgs=2 tok_before=50 tok_after=25 tok_saved=25 cache_read=0 cache_write=0 cache_hit_pct=0 opt_ms=50 transforms=cache_aligner,content_router",
        encoding="utf-8",
    )

    monkeypatch.setattr(analyzer, "LOG_DIR", tmp_path)
    report = parse_log_files()

    assert report.log_files_read == 2
    assert report.total_lines_parsed == 5
    assert len(report.perf_records) == 2
    assert report.perf_records[0].transforms == ["router:excluded:tool", "read_lifecycle:stale"]
    assert report.perf_records[1].transforms == ["cache_aligner", "content_router"]

    router = report.router_records[0]
    assert router.num_messages == 5
    assert router.compressed == 2
    assert router.excluded == 1
    assert router.skipped == 1
    assert router.unchanged == 1
    assert router.content_blocks == 3

    transform = report.transform_records[0]
    assert transform.name == "content_router"
    assert transform.tokens_saved == 60

    toin = report.toin_records[0]
    assert toin.patterns == 105
    assert toin.compressions == 3837
    assert toin.retrieval_rate == 0.0


def test_format_report_handles_empty_report() -> None:
    text = format_report(PerfReport())
    assert "No performance data found" in text
    assert "headroom proxy" in text


def test_format_report_and_recommendations_cover_major_sections(monkeypatch) -> None:
    monkeypatch.setattr(
        analyzer, "_get_list_price", lambda model: 2.5 if model == "claude" else None
    )

    perf_records = [
        PerfRecord(
            timestamp=f"2026-03-07 13:38:{i:02d},000",
            request_id=f"req-{i}",
            model="claude",
            num_messages=10 + i,
            tokens_before=1000,
            tokens_after=400,
            tokens_saved=600,
            cache_read=50 if i < 5 else 300,
            cache_write=600 if i < 5 else 20,
            optimization_ms=700 if i < 3 else 100,
            transforms=["content_router"],
        )
        for i in range(10)
    ]
    report = PerfReport(
        perf_records=perf_records,
        router_records=[
            RouterRecord(
                timestamp="2026-03-07 13:40:00,000",
                compressed=2,
                excluded=10,
                skipped=1,
                unchanged=1,
            )
        ],
        transform_records=[
            TransformRecord(
                timestamp="2026-03-07 13:40:00,000",
                name="content_router",
                tokens_before=1000,
                tokens_after=400,
                tokens_saved=600,
            ),
            TransformRecord(
                timestamp="2026-03-07 13:41:00,000",
                name="cache_aligner",
                tokens_before=100,
                tokens_after=95,
                tokens_saved=5,
            ),
        ],
        toin_records=[
            ToinRecord(
                timestamp="2026-03-07 13:42:00,000",
                patterns=120,
                compressions=500,
                retrievals=0,
                retrieval_rate=0.0,
            )
        ],
        log_files_read=2,
        total_lines_parsed=100,
    )

    text = format_report(report)
    recs = _generate_recommendations(report)

    assert "Headroom Performance Report" in text
    assert "Per-Model Breakdown" in text
    assert "~$0.01 at list price" in text
    assert "Cache Performance" in text
    assert "-> Cache stabilizing over conversation lifetime" in text
    assert "Optimization Overhead" in text
    assert ">500ms:   3 requests" in text
    assert "Conversation Size" in text
    assert "Transform Effectiveness" in text
    assert "Content Router Routing" in text
    assert "TOIN Learning" in text
    assert "Recommendations" in text
    assert "Log files: 2 | Lines parsed: 100" in text

    assert any("Cache prefix unstable" in rec for rec in recs)
    assert any("First 5 turns have very low cache hit ratio" in rec for rec in recs)
    assert any("took >500ms for optimization" in rec for rec in recs)
    assert any("Read/Glob outputs are majority" in rec for rec in recs)
    assert any("TOIN has 0% retrieval rate" in rec for rec in recs)
    assert any("cache_aligner saving <10 tokens" in rec for rec in recs)
