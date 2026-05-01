from __future__ import annotations

import json

from headroom.evals.reports.report_card import (
    BenchmarkRunResult,
    SuiteResult,
    generate_html,
    generate_json,
    generate_markdown,
    save_reports,
)


def _make_suite_result() -> SuiteResult:
    return SuiteResult(
        model="gpt-4o",
        tiers_run=[1, 2],
        timestamp="2026-04-24T00:00:00",
        total_cost_usd=1.2345,
        total_duration_seconds=98.7,
        benchmarks=[
            BenchmarkRunResult(
                name="mmlu",
                category="knowledge",
                tier=1,
                baseline_score=0.8,
                headroom_score=0.79,
                delta=-0.01,
                avg_compression_ratio=0.42,
                tokens_saved=1200,
                n_samples=50,
                model="gpt-4o",
                metric_name="acc",
                duration_seconds=10.5,
                passed=True,
            ),
            BenchmarkRunResult(
                name="tool_outputs",
                category="tool_use",
                tier=2,
                accuracy_rate=0.96,
                avg_compression_ratio=0.73,
                tokens_saved=2500,
                n_samples=20,
                model="gpt-4o",
                metric_name="accuracy",
                duration_seconds=20.2,
                passed=False,
                error="regression",
            ),
        ],
    )


def test_report_card_models_and_renderers(tmp_path) -> None:
    suite = _make_suite_result()
    empty = SuiteResult(model="gpt-4o-mini", tiers_run=[])

    assert suite.standard_benchmarks[0].name == "mmlu"
    assert suite.compression_benchmarks[0].name == "tool_outputs"
    assert suite.all_passed is False
    assert suite.pass_rate == 0.5
    assert suite.avg_delta == -0.01
    assert suite.avg_compression == (0.42 + 0.73) / 2
    assert suite.total_tokens_saved == 3700
    assert empty.pass_rate == 0.0
    assert empty.avg_delta == 0.0
    assert empty.avg_compression == 0.0
    assert empty.total_tokens_saved == 0

    benchmark_dict = suite.benchmarks[1].to_dict()
    assert benchmark_dict["accuracy_rate"] == 0.96
    assert benchmark_dict["error"] == "regression"

    suite_dict = suite.to_dict()
    assert suite_dict["summary"]["failed"] == 1
    assert suite_dict["summary"]["total_tokens_saved"] == 3700

    markdown = generate_markdown(suite)
    assert "## Headroom Accuracy Report Card" in markdown
    assert "Standard Benchmarks" in markdown
    assert "Compression Benchmarks" in markdown
    assert "**VERDICT: 1/2 PASS**" in markdown

    empty_markdown = generate_markdown(empty)
    assert "VERDICT: 0/0 PASS" in empty_markdown

    rendered_json = generate_json(suite)
    assert json.loads(rendered_json)["summary"]["passed"] == 1

    html = generate_html(suite)
    assert "<h1>Headroom Accuracy Report Card</h1>" in html
    assert "mmlu" in html
    assert "tool_outputs" in html
    assert 'class="badge fail"' in html

    empty_html = generate_html(empty)
    assert "0/0 PASS" in empty_html

    paths = save_reports(suite, tmp_path)
    assert set(paths) == {"markdown", "json", "html"}
    assert paths["markdown"].read_text() == markdown
    assert json.loads(paths["json"].read_text())["model"] == "gpt-4o"
    assert paths["html"].read_text() == html
