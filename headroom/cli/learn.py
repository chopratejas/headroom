"""CLI commands for Headroom Learn — offline failure learning."""

from __future__ import annotations

from pathlib import Path

import click

from .main import main


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
    help="Write recommendations to CLAUDE.md / MEMORY.md (default: dry-run).",
)
@click.option(
    "--claude-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to .claude directory. Defaults to ~/.claude.",
)
def learn(
    project: Path | None,
    analyze_all: bool,
    apply: bool,
    claude_dir: Path | None,
) -> None:
    """Learn from past tool call failures to prevent future ones.

    Analyzes conversation history to find failure patterns (wrong paths,
    missing modules, stubborn retries) and generates context that prevents
    them from recurring.

    \b
    Examples:
        headroom learn                    # Analyze current project (dry-run)
        headroom learn --apply            # Write recommendations
        headroom learn --all              # Analyze all projects
        headroom learn --project ~/myapp  # Analyze specific project
    """
    from ..learn.analyzer import FailureAnalyzer
    from ..learn.scanner import ClaudeCodeScanner
    from ..learn.writer import ClaudeCodeWriter, Recommender

    scanner = ClaudeCodeScanner(claude_dir=claude_dir)
    analyzer = FailureAnalyzer()
    recommender = Recommender()
    writer = ClaudeCodeWriter()

    # Discover projects
    all_projects = scanner.discover_projects()

    if not all_projects:
        click.echo("No projects found in ~/.claude/projects/")
        return

    # Filter to target project(s)
    if analyze_all:
        targets = all_projects
    elif project:
        resolved = project.resolve()
        targets = [p for p in all_projects if p.project_path == resolved]
        if not targets:
            click.echo(f"Project not found: {resolved}")
            click.echo(f"Available projects: {', '.join(p.name for p in all_projects)}")
            return
    else:
        # Auto-detect from cwd
        cwd = Path.cwd().resolve()
        targets = [p for p in all_projects if p.project_path == cwd]
        if not targets:
            # Try parent directories
            for parent in cwd.parents:
                targets = [p for p in all_projects if p.project_path == parent]
                if targets:
                    break
        if not targets:
            click.echo(f"No project data found for {cwd}")
            click.echo("Try: headroom learn --project <path>  or  headroom learn --all")
            click.echo("\nAvailable projects:")
            for p in all_projects[:10]:
                click.echo(f"  {p.name:30s} {p.project_path}")
            return

    # Analyze each target
    for proj in targets:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Project: {proj.name}")
        click.echo(f"Path:    {proj.project_path}")
        click.echo(f"{'=' * 60}")

        sessions = scanner.scan_project(proj)
        if not sessions:
            click.echo("  No conversation data found.")
            continue

        report = analyzer.analyze(proj, sessions)

        # Print summary
        click.echo(f"\n  Sessions analyzed: {report.total_sessions}")
        click.echo(f"  Total tool calls:  {report.total_calls}")
        click.echo(f"  Failed calls:      {report.total_failures} ({report.failure_rate:.1%})")
        click.echo(f"  Waste bytes:       {report.waste_bytes / 1024:.0f} KB")

        if report.failure_rate == 0:
            click.echo("\n  No failures found. Nothing to learn.")
            continue

        # Generate recommendations
        recommendations = recommender.recommend(report)

        if not recommendations:
            click.echo("\n  No actionable patterns found.")
            continue

        click.echo(f"\n  Recommendations: {len(recommendations)}")

        # Write (or dry-run)
        result = writer.write(recommendations, proj, dry_run=not apply)

        for file_path, content in result.content_by_file.items():
            click.echo(f"\n  {'[WOULD WRITE]' if result.dry_run else '[WROTE]'} {file_path}")
            click.echo(f"  {'─' * 50}")
            # Show content preview (indented)
            for line in content.split("\n"):
                if line.startswith("<!-- headroom"):
                    continue  # Skip markers in display
                click.echo(f"  {line}")
            click.echo(f"  {'─' * 50}")

        if result.dry_run:
            click.echo("\n  Dry run — no files modified. Use --apply to write.")
