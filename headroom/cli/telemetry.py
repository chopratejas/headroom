"""Telemetry transparency CLI commands."""

from __future__ import annotations

import json

import click

from .main import main


@main.group("telemetry")
def telemetry_group() -> None:
    """Inspect Headroom telemetry and observability surfaces."""


@telemetry_group.command("list")
@click.option("--json", "json_output", is_flag=True, help="Emit JSON instead of a table.")
def telemetry_list_cmd(json_output: bool) -> None:
    """List every known telemetry/observability surface."""

    from rich.console import Console
    from rich.table import Table

    from headroom.telemetry.surfaces import collect_telemetry_surface_dicts

    rows = collect_telemetry_surface_dicts()
    if json_output:
        click.echo(json.dumps(rows, indent=2))
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Surface", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Leaves host")
    table.add_column("Prompt content")
    table.add_column("Target")
    table.add_column("Controls")

    for row in rows:
        controls = ", ".join(str(control) for control in row["controls"])
        table.add_row(
            str(row["name"]),
            str(row["status"]),
            str(row["leaves_host"]),
            str(row["prompt_content"]),
            str(row["target"]),
            controls,
        )

    Console(width=180).print(table)
