"""Compression analytics export CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from .main import main

_FORMATS = ("markdown", "json", "csv")
_SERIES = ("history", "hourly", "daily", "weekly", "monthly")
_HISTORY_MODES = ("compact", "full", "none")


def _infer_output_format(output_format: str | None, output_path: Path | None) -> str:
    if output_format:
        return output_format
    if output_path is None:
        return "markdown"

    suffix = output_path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return "markdown"


def _format_usd(value: Any) -> str:
    try:
        return f"${float(value):.6f}"
    except (TypeError, ValueError):
        return "$0.000000"


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _lifetime_savings_percent(lifetime: dict[str, Any]) -> float:
    tokens_saved = int(lifetime.get("tokens_saved") or 0)
    total_input_tokens = int(lifetime.get("total_input_tokens") or 0)
    total_before = tokens_saved + total_input_tokens
    if total_before <= 0:
        return 0.0
    return round(tokens_saved / total_before * 100, 2)


def _markdown_table(rows: list[dict[str, Any]], *, series: str) -> list[str]:
    if not rows:
        return ["No history points recorded yet."]

    if series == "history":
        headers = [
            "timestamp",
            "total_tokens_saved",
            "compression_savings_usd",
            "total_input_tokens",
            "total_input_cost_usd",
        ]
    else:
        headers = [
            "timestamp",
            "tokens_saved",
            "compression_savings_usd_delta",
            "total_tokens_saved",
            "total_input_tokens_delta",
            "total_input_tokens",
        ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_markdown_report(
    data: dict[str, Any], rows: list[dict[str, Any]], *, series: str
) -> str:
    lifetime = data.get("lifetime", {})
    display_session = data.get("display_session", {})
    history_summary = data.get("history_summary", {})

    lines = [
        "# Headroom Compression Analytics",
        "",
        f"- Storage: `{data.get('storage_path', '')}`",
        f"- Generated: `{data.get('generated_at', '')}`",
        f"- History points: {history_summary.get('returned_points', 0)} returned / "
        f"{history_summary.get('stored_points', 0)} stored",
        "",
        "## Lifetime",
        "",
        f"- Requests: {_format_int(lifetime.get('requests'))}",
        f"- Tokens saved: {_format_int(lifetime.get('tokens_saved'))}",
        f"- Input tokens tracked: {_format_int(lifetime.get('total_input_tokens'))}",
        f"- Compression savings: {_format_usd(lifetime.get('compression_savings_usd'))}",
        f"- Input spend tracked: {_format_usd(lifetime.get('total_input_cost_usd'))}",
        f"- Savings percent: {_lifetime_savings_percent(lifetime):.2f}%",
        "",
        "## Current Display Session",
        "",
        f"- Requests: {_format_int(display_session.get('requests'))}",
        f"- Tokens saved: {_format_int(display_session.get('tokens_saved'))}",
        f"- Input tokens tracked: {_format_int(display_session.get('total_input_tokens'))}",
        f"- Savings percent: {float(display_session.get('savings_percent') or 0.0):.2f}%",
        "",
        f"## {series.title()} Series",
        "",
        *_markdown_table(rows, series=series),
    ]
    return "\n".join(lines) + "\n"


@main.command("analytics")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(_FORMATS),
    default=None,
    help="Output format. Inferred from --output/--export when omitted.",
)
@click.option(
    "--series",
    type=click.Choice(_SERIES),
    default="daily",
    show_default=True,
    help="History series used for CSV and Markdown output.",
)
@click.option(
    "--history-mode",
    type=click.Choice(_HISTORY_MODES),
    default="compact",
    show_default=True,
    help="History detail level for JSON output.",
)
@click.option(
    "--savings-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Read analytics from a specific proxy_savings.json file.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the report to a file instead of stdout.",
)
@click.option(
    "--export",
    "export_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Alias for --output.",
)
def analytics(
    output_format: str | None,
    series: str,
    history_mode: str,
    savings_path: Path | None,
    output_path: Path | None,
    export_path: Path | None,
) -> None:
    """Export compression analytics from local proxy savings history.

    \b
    Examples:
        headroom analytics
        headroom analytics --format json
        headroom analytics --series weekly --export savings.csv
    """
    from headroom.proxy.savings_tracker import SavingsTracker

    if output_path is not None and export_path is not None:
        raise click.UsageError("Use either --output or --export, not both.")

    target_path = output_path or export_path
    resolved_format = _infer_output_format(output_format, target_path)
    tracker = SavingsTracker(path=str(savings_path) if savings_path is not None else None)

    if resolved_format == "json":
        payload = json.dumps(tracker.history_response(history_mode=history_mode), indent=2)
    elif resolved_format == "csv":
        payload = tracker.export_csv(series=series)
    else:
        data = tracker.history_response(history_mode=history_mode)
        rows = tracker.export_rows(series=series)
        payload = _render_markdown_report(data, rows, series=series)

    if target_path is None:
        click.echo(payload.rstrip("\n"))
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(payload, encoding="utf-8")
    click.echo(f"Wrote analytics report to {target_path}")
