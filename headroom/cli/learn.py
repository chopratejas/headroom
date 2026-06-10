"""CLI commands for Headroom Learn — offline failure learning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from ..learn.base import LearnPlugin

from .main import main


class _AgentChoice(click.ParamType):
    """Dynamic Click type that validates against the plugin registry."""

    name = "agent"

    def get_metavar(self, param: click.Parameter, ctx: click.Context | None = None) -> str | None:
        return "[auto|<agent>]"

    def convert(
        self,
        value: str,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> str:
        if value == "auto":
            return value
        from ..learn.registry import get_registry

        reg = get_registry()
        if value.lower() not in reg:
            available = ", ".join(sorted(reg.keys()))
            self.fail(f"Unknown agent: {value}. Available: auto, {available}", param, ctx)
        return value.lower()

    def shell_complete(
        self,
        ctx: click.Context,
        param: click.Parameter,
        incomplete: str,
    ) -> list[click.shell_completion.CompletionItem]:
        from ..learn.registry import available_agent_names

        names = ["auto"] + available_agent_names()
        return [click.shell_completion.CompletionItem(n) for n in names if n.startswith(incomplete)]


_AGENT_HELP = """Which coding agent to analyze. Auto-detects by default.

\b
Built-in: claude, codex, gemini.
External plugins register via 'headroom.learn_plugin' entry point.
Use 'auto' (default) to scan all detected agents."""


@main.command()
@click.option(
    "--project",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Project directory to analyze. Defaults to current directory.",
)
@click.option(
    "--all",
    "analyze_all",
    is_flag=True,
    default=False,
    help="Analyze all discovered projects.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Write recommendations to context/memory files (default: dry-run).",
)
@click.option(
    "--agent",
    type=_AgentChoice(),
    default="auto",
    help=_AGENT_HELP,
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model for analysis (e.g., claude-sonnet-4-6, gpt-4o, gemini/gemini-flash-latest). "
    "Auto-detected from API keys if not specified.",
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=None,
    help="Parallel workers for session scanning. "
    "Default: auto (min of CPU count, 8). Use 1 for serial.",
)
def learn(
    project: Path | None,
    analyze_all: bool,
    apply: bool,
    agent: str,
    model: str | None,
    workers: int | None,
) -> None:
    """Learn from past tool call failures to prevent future ones.

    Analyzes conversation history using an LLM to find failure patterns
    (wrong paths, missing modules, stubborn retries) and generates context
    that prevents them from recurring.

    Supports multiple coding agents via a plugin architecture. Built-in
    support for Claude Code, Codex, and Gemini CLI. External plugins can
    be installed via pip (entry point: headroom.learn_plugin).

    \b
    Examples:
        headroom learn                        # Auto-detect agent & model
        headroom learn --apply                # Write recommendations
        headroom learn --model gpt-4o         # Use GPT-4o for analysis
        headroom learn --all                  # Analyze all projects
        headroom learn --agent codex --all    # Analyze all Codex sessions
    """
    import os

    from ..learn.analyzer import SessionAnalyzer, _detect_default_model
    from ..learn.registry import auto_detect_plugins, get_plugin

    max_workers = workers if workers is not None else min(os.cpu_count() or 4, 8)

    # Resolve model early to fail fast with a clear message
    try:
        resolved_model = model or _detect_default_model()
    except RuntimeError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from None

    analyzer = SessionAnalyzer(model=resolved_model)

    # Determine which agents to scan
    agent_configs: list[tuple[str, LearnPlugin]] = []

    if agent == "auto":
        detected = auto_detect_plugins()
        if not detected:
            click.echo("No coding agent data found.")
            return
        click.echo(f"Detected agents: {', '.join(p.display_name for p in detected)}")
        agent_configs = [(p.name, p) for p in detected]
    else:
        selected = get_plugin(agent)
        agent_configs = [(selected.name, selected)]

    total_projects = 0
    total_failures = 0
    total_recommendations = 0
    matched_projects = 0
    available_projects: list[tuple[str, Path]] = []

    for agent_name, plugin in agent_configs:
        writer = plugin.create_writer()
        all_projects = plugin.discover_projects()
        if not all_projects:
            continue
        available_projects.extend((agent_name, proj.project_path) for proj in all_projects)

        # Filter to target project(s)
        if analyze_all:
            targets = all_projects
        elif project:
            resolved = project.resolve()
            targets = [p for p in all_projects if p.project_path == resolved]
            if not targets:
                continue
        else:
            cwd = Path.cwd().resolve()
            targets = [p for p in all_projects if p.project_path == cwd]
            if not targets:
                for parent in cwd.parents:
                    targets = [p for p in all_projects if p.project_path == parent]
                    if targets:
                        break
            if not targets and len(agent_configs) == 1:
                click.echo(f"No {agent_name} project data found for {cwd}")
                click.echo("Try: headroom learn --all  or  headroom learn --project <path>")
                click.echo(f"\nAvailable {agent_name} projects:")
                for proj_info in all_projects[:10]:
                    click.echo(f"  {proj_info.name:30s} {proj_info.project_path}")
                return

        for proj in targets:
            matched_projects += 1
            click.echo(f"\n{'=' * 60}")
            click.echo(f"[{agent_name}] {proj.name}")
            click.echo(f"Path: {proj.project_path}")
            click.echo(f"{'=' * 60}")

            sessions = plugin.scan_project(proj, max_workers=max_workers)
            if not sessions:
                click.echo("  No conversation data found.")
                continue

            click.echo(f"  Analyzing with {resolved_model}...")
            result_data = analyzer.analyze(proj, sessions)
            total_projects += 1
            total_failures += result_data.total_failures

            click.echo(
                f"\n  Sessions: {result_data.total_sessions}  |  "
                f"Calls: {result_data.total_calls}  |  "
                f"Failures: {result_data.total_failures} ({result_data.failure_rate:.1%})"
            )

            if result_data.failure_rate == 0 and not result_data.recommendations:
                click.echo("  No failures or patterns found.")
                continue

            recommendations = result_data.recommendations
            if not recommendations:
                click.echo("  No actionable patterns found.")
                continue

            total_recommendations += len(recommendations)
            click.echo(f"  Recommendations: {len(recommendations)}")

            try:
                result = writer.write(recommendations, proj, dry_run=not apply)
            except OSError as e:
                click.echo(
                    f"  Warning: failed to write recommendations for {proj.project_path}: {e}"
                )
                continue

            for file_path, content in result.content_by_file.items():
                click.echo(f"\n  {'[WOULD WRITE]' if result.dry_run else '[WROTE]'} {file_path}")
                click.echo(f"  {'─' * 50}")
                for line in content.split("\n"):
                    if line.startswith("<!-- headroom"):
                        continue
                    click.echo(f"  {line}")
                click.echo(f"  {'─' * 50}")

            if result.dry_run:
                click.echo("\n  Dry run — use --apply to write.")

    if project and matched_projects == 0:
        click.echo(f"No project data found for {project.resolve()}")
        if available_projects:
            click.echo("\nAvailable discovered projects:")
            for agent_name, project_path in available_projects[:10]:
                click.echo(f"  [{agent_name}] {project_path}")
        return

    # Summary
    if total_projects > 1:
        click.echo(f"\n{'=' * 60}")
        click.echo(
            f"Total: {total_projects} projects, {total_failures} failures, "
            f"{total_recommendations} recommendations"
        )
