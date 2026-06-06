"""Tests for compression analytics CLI exports."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from headroom.cli.main import main
from headroom.proxy.savings_tracker import SavingsTracker


def _populate_savings(path: Path) -> None:
    tracker = SavingsTracker(path=str(path))
    tracker.record_request(
        model="gpt-4o",
        input_tokens=120,
        tokens_saved=40,
        total_input_cost_usd=0.24,
        timestamp="2026-03-27T09:15:00Z",
    )
    tracker.record_request(
        model="gpt-4o",
        input_tokens=80,
        tokens_saved=20,
        total_input_cost_usd=0.16,
        timestamp="2026-03-28T10:30:00Z",
    )


def test_analytics_json_export_reads_savings_history(tmp_path: Path) -> None:
    savings_path = tmp_path / "proxy_savings.json"
    _populate_savings(savings_path)

    result = CliRunner().invoke(
        main,
        [
            "analytics",
            "--format",
            "json",
            "--history-mode",
            "full",
            "--savings-path",
            str(savings_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["storage_path"] == str(savings_path)
    assert payload["lifetime"]["requests"] == 2
    assert payload["lifetime"]["tokens_saved"] == 60
    assert payload["lifetime"]["total_input_tokens"] == 200
    assert payload["history_summary"]["mode"] == "full"
    assert len(payload["history"]) == 2


def test_analytics_csv_export_uses_selected_rollup_series(tmp_path: Path) -> None:
    savings_path = tmp_path / "proxy_savings.json"
    _populate_savings(savings_path)

    result = CliRunner().invoke(
        main,
        [
            "analytics",
            "--format",
            "csv",
            "--series",
            "daily",
            "--savings-path",
            str(savings_path),
        ],
    )

    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == (
        "timestamp,tokens_saved,compression_savings_usd_delta,total_tokens_saved,"
        "compression_savings_usd,total_input_tokens_delta,total_input_tokens,"
        "total_input_cost_usd_delta,total_input_cost_usd"
    )
    assert "2026-03-27T00:00:00Z,40" in lines[1]
    assert "2026-03-28T00:00:00Z,20" in lines[2]


def test_analytics_export_infers_markdown_from_file_extension(tmp_path: Path) -> None:
    savings_path = tmp_path / "proxy_savings.json"
    output_path = tmp_path / "reports" / "analytics.md"
    _populate_savings(savings_path)

    result = CliRunner().invoke(
        main,
        [
            "analytics",
            "--series",
            "history",
            "--savings-path",
            str(savings_path),
            "--export",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Wrote analytics report to {output_path}" in result.output
    report = output_path.read_text(encoding="utf-8")
    assert "# Headroom Compression Analytics" in report
    assert "- Requests: 2" in report
    assert "- Tokens saved: 60" in report
    assert "## History Series" in report
    assert "| timestamp | total_tokens_saved |" in report


def test_analytics_rejects_output_and_export_together(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "analytics",
            "--output",
            str(tmp_path / "one.json"),
            "--export",
            str(tmp_path / "two.json"),
        ],
    )

    assert result.exit_code != 0
    assert "Use either --output or --export, not both." in result.output
