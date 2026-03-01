"""CLI commands for Headroom Learn — offline failure learning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from ..learn.scanner import ConversationScanner
    from ..learn.writer import ContextWriter

from .main import main

_AGENT_HELP = """Which coding agent to analyze. Auto-detects by default.

\b
Supported agents:
  claude   Claude Code (~/.claude/)
  codex    OpenAI Codex CLI (~/.codex/)
  gemini   Google Gemini CLI (~/.gemini/)
  auto     Auto-detect (check all, default)
"""


def _get_scanner_writer(agent: str) -> tuple[ConversationScanner, ContextWriter]:
    """Get the appropriate scanner and writer for an agent type."""
    from ..learn.scanner import ClaudeCodeScanner, CodexScanner
    from ..learn.writer import ClaudeCodeWriter, CodexWriter

    scanners = {
        "claude": (ClaudeCodeScanner, ClaudeCodeWriter),
        "codex": (CodexScanner, CodexWriter),
        # Gemini scanner not yet implemented (protobuf sessions)
        # Cursor scanner not yet implemented (SQLite blobs)
    }

    if agent in scanners:
        scanner_cls, writer_cls = scanners[agent]
        return scanner_cls(), writer_cls()

    raise click.BadParameter(f"Unknown agent: {agent}. Supported: claude, codex")


def _auto_detect_agents() -> list[tuple[str, ConversationScanner, ContextWriter]]:
    """Auto-detect which agents have data on this machine."""
    from ..learn.scanner import ClaudeCodeScanner, CodexScanner
    from ..learn.writer import ClaudeCodeWriter, CodexWriter

    agents: list[tuple[str, ConversationScanner, ContextWriter]] = []

    # Claude Code
    claude_dir = Path.home() / ".claude" / "projects"
    if claude_dir.exists() and any(claude_dir.iterdir()):
        agents.append(("claude", ClaudeCodeScanner(), ClaudeCodeWriter()))

    # Codex
    codex_dir = Path.home() / ".codex" / "sessions"
    if codex_dir.exists() and any(codex_dir.glob("*.json")):
        agents.append(("codex", CodexScanner(), CodexWriter()))

    return agents


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
    type=click.Choice(["auto", "claude", "codex", "gemini"], case_sensitive=False),
    default="auto",
    help=_AGENT_HELP,
)
def learn(
    project: Path | None,
    analyze_all: bool,
    apply: bool,
    agent: str,
) -> None:
    """Learn from past tool call failures to prevent future ones.

    Analyzes conversation history to find failure patterns (wrong paths,
    missing modules, stubborn retries) and generates context that prevents
    them from recurring.

    Supports multiple coding agents: Claude Code, Codex, Gemini CLI.
    Auto-detects which agents have data on your machine by default.

    \b
    Examples:
        headroom learn                        # Auto-detect agent, analyze current project
        headroom learn --apply                # Write recommendations
        headroom learn --all                  # Analyze all projects
        headroom learn --agent codex --all    # Analyze all Codex sessions
        headroom learn --agent claude --project ~/myapp
    """
    from ..learn.analyzer import FailureAnalyzer

    analyzer = FailureAnalyzer()

    # Determine which agents to scan
    if agent == "auto":
        agent_configs = _auto_detect_agents()
        if not agent_configs:
            click.echo("No coding agent data found. Checked: ~/.claude/, ~/.codex/")
            return
        click.echo(f"Detected agents: {', '.join(name for name, _, _ in agent_configs)}")
    else:
        scanner, writer = _get_scanner_writer(agent)
        agent_configs = [(agent, scanner, writer)]

    total_projects = 0
    total_failures = 0
    total_recommendations = 0

    for agent_name, scanner, writer in agent_configs:
        all_projects = scanner.discover_projects()
        if not all_projects:
            continue

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
                for p in all_projects[:10]:
                    click.echo(f"  {p.name:30s} {p.project_path}")
                return

        for proj in targets:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"[{agent_name}] {proj.name}")
            click.echo(f"Path: {proj.project_path}")
            click.echo(f"{'=' * 60}")

            sessions = scanner.scan_project(proj)
            if not sessions:
                click.echo("  No conversation data found.")
                continue

            from ..learn.writer import Recommender

            recommender = Recommender()
            report = analyzer.analyze(proj, sessions)
            total_projects += 1
            total_failures += report.total_failures

            click.echo(
                f"\n  Sessions: {report.total_sessions}  |  "
                f"Calls: {report.total_calls}  |  "
                f"Failures: {report.total_failures} ({report.failure_rate:.1%})  |  "
                f"Corrections: {len(report.corrections)}"
            )

            if report.failure_rate == 0:
                click.echo("  No failures found.")
                continue

            recommendations = recommender.recommend(report)
            if not recommendations:
                click.echo("  No actionable patterns found.")
                continue

            total_recommendations += len(recommendations)
            click.echo(f"  Recommendations: {len(recommendations)}")

            result = writer.write(recommendations, proj, dry_run=not apply)

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

    # Summary
    if total_projects > 1:
        click.echo(f"\n{'=' * 60}")
        click.echo(
            f"Total: {total_projects} projects, {total_failures} failures, "
            f"{total_recommendations} recommendations"
        )
