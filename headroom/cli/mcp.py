"""MCP (Model Context Protocol) CLI commands for Claude Code integration.

Provides commands to configure and run the Headroom MCP server, enabling
Claude Code subscription users to use CCR (Compress-Cache-Retrieve) without
needing API key access.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from .main import main

# Default paths
CLAUDE_CONFIG_DIR = Path.home() / ".claude"
MCP_CONFIG_PATH = CLAUDE_CONFIG_DIR / "mcp.json"
DEFAULT_PROXY_URL = "http://127.0.0.1:8787"


def get_headroom_command() -> list[str]:
    """Get the command to run headroom MCP server.

    Returns the most reliable way to invoke headroom based on installation.
    """
    # Check if headroom is in PATH
    headroom_path = shutil.which("headroom")
    if headroom_path:
        return ["headroom", "mcp", "serve"]

    # Fall back to python -m
    return [sys.executable, "-m", "headroom.ccr.mcp_server"]


def load_mcp_config() -> dict[str, Any]:
    """Load existing MCP config or return empty structure."""
    if MCP_CONFIG_PATH.exists():
        try:
            with open(MCP_CONFIG_PATH) as f:
                result: dict[str, Any] = json.load(f)
                return result
        except (json.JSONDecodeError, OSError):
            return {"mcpServers": {}}
    return {"mcpServers": {}}


def save_mcp_config(config: dict) -> None:
    """Save MCP config, creating directory if needed."""
    CLAUDE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(MCP_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")  # Trailing newline


@main.group()
def mcp() -> None:
    """MCP server for Claude Code integration.

    \b
    The MCP server exposes headroom_retrieve as a tool that Claude Code
    can use to retrieve compressed content. This enables CCR (Compress-
    Cache-Retrieve) for subscription users who don't have API access.

    \b
    Quick Start:
        headroom mcp install    # Configure Claude Code
        headroom proxy          # Start the proxy (in another terminal)
        ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude

    \b
    The MCP server provides on-demand tools (compress, retrieve, stats).
    For automatic compression of ALL traffic, also set ANTHROPIC_BASE_URL
    to route through the proxy.

    \b
    How it works:
        1. ANTHROPIC_BASE_URL routes all requests through the proxy
        2. The proxy compresses large tool outputs (file listings, search results)
        3. Claude sees compressed summaries with hash markers
        4. When Claude needs full details, it calls headroom_retrieve
        5. The MCP server fetches original content from the proxy
    """
    pass


@mcp.command("install")
@click.option(
    "--proxy-url",
    default=DEFAULT_PROXY_URL,
    help=f"Headroom proxy URL (default: {DEFAULT_PROXY_URL})",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing headroom config",
)
def mcp_install(proxy_url: str, force: bool) -> None:
    """Install Headroom MCP server into Claude Code config.

    \b
    This registers headroom with Claude Code so it can use the
    headroom_retrieve tool for CCR (Compress-Cache-Retrieve).

    \b
    Example:
        headroom mcp install
        headroom mcp install --proxy-url http://localhost:9000
    """
    # Check for MCP SDK
    try:
        import mcp  # noqa: F401
    except ImportError:
        click.echo("Error: MCP SDK not installed.", err=True)
        click.echo("Install with: pip install 'headroom-ai[mcp]'", err=True)
        raise SystemExit(1) from None

    command = get_headroom_command()
    env: dict[str, str] = {}
    if proxy_url != DEFAULT_PROXY_URL:
        env["HEADROOM_PROXY_URL"] = proxy_url

    # Prefer `claude mcp add` (Claude Code CLI ≥2.x stores servers in
    # ~/.claude/.claude.json, which is what `claude mcp list` reads).
    claude_cli = shutil.which("claude")
    used_claude_cli = False
    if claude_cli:
        # Check if already registered
        result = subprocess.run(
            [claude_cli, "mcp", "get", "headroom"],
            capture_output=True,
            text=True,
        )
        already_registered = result.returncode == 0

        if already_registered and not force:
            click.echo("Headroom MCP is already configured in Claude Code.")
            click.echo("Use --force to overwrite, or 'headroom mcp uninstall' first.")
            raise SystemExit(0)

        if already_registered and force:
            subprocess.run(
                [claude_cli, "mcp", "remove", "headroom", "-s", "user"],
                capture_output=True,
            )

        add_cmd = [claude_cli, "mcp", "add", "headroom", "-s", "user"]
        for k, v in env.items():
            add_cmd += ["-e", f"{k}={v}"]
        add_cmd += ["--", *command]

        result = subprocess.run(add_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            used_claude_cli = True
        else:
            click.echo(
                f"Warning: 'claude mcp add' failed ({result.stderr.strip()}), "
                "falling back to mcp.json.",
                err=True,
            )

    if not used_claude_cli:
        # Fallback: write ~/.claude/mcp.json (used by older Claude Code versions
        # and the Claude.ai desktop app).
        config = load_mcp_config()

        if "headroom" in config.get("mcpServers", {}) and not force:
            click.echo("Headroom MCP is already configured in Claude Code.")
            click.echo("Use --force to overwrite, or 'headroom mcp uninstall' first.")
            raise SystemExit(0)

        server_config: dict = {"command": command[0], "args": command[1:]}
        if env:
            server_config["env"] = env

        if "mcpServers" not in config:
            config["mcpServers"] = {}
        config["mcpServers"]["headroom"] = server_config
        save_mcp_config(config)

    config_note = (
        "Registered via: claude mcp add (scope: user)"
        if used_claude_cli
        else f"Configuration written to: {MCP_CONFIG_PATH}"
    )

    click.echo(f"""
✓ Headroom MCP server installed!

{config_note}

Next steps:
  1. Start the Headroom proxy (if not running):
     headroom proxy

  2. Start Claude Code WITH the proxy base URL:
     ANTHROPIC_BASE_URL={proxy_url} claude

  3. Claude Code now has:
     - All requests compressed through the proxy (saves tokens & cost)
     - Access to headroom_retrieve tool for CCR retrieval
     - Stats visible at {proxy_url}/stats

NOTE: The MCP server provides on-demand compression tools
(headroom_compress, headroom_retrieve, headroom_stats). For automatic
compression of ALL traffic, also set ANTHROPIC_BASE_URL as shown above.

Proxy URL: {proxy_url}
""")


@mcp.command("uninstall")
def mcp_uninstall() -> None:
    """Remove Headroom MCP server from Claude Code config.

    \b
    Removes headroom from both the claude CLI registry (Claude Code CLI >=2.x)
    and ~/.claude/mcp.json if present. Other MCP servers are preserved.
    """
    removed = False

    # Remove from claude CLI registry (Claude Code CLI >=2.x)
    claude_cli = shutil.which("claude")
    if claude_cli:
        check = subprocess.run(
            [claude_cli, "mcp", "get", "headroom"],
            capture_output=True,
        )
        if check.returncode == 0:
            rm = subprocess.run(
                [claude_cli, "mcp", "remove", "headroom", "-s", "user"],
                capture_output=True,
                text=True,
            )
            if rm.returncode == 0:
                click.echo("✓ Headroom MCP server removed (via claude mcp remove)")
                removed = True
            else:
                click.echo(
                    f"Warning: 'claude mcp remove' failed ({rm.stderr.strip()}).",
                    err=True,
                )

    # Also remove from mcp.json fallback config if present
    if MCP_CONFIG_PATH.exists():
        config = load_mcp_config()
        if "headroom" in config.get("mcpServers", {}):
            del config["mcpServers"]["headroom"]
            save_mcp_config(config)
            click.echo(f"✓ Headroom MCP server removed from {MCP_CONFIG_PATH}")
            removed = True

    if not removed:
        if MCP_CONFIG_PATH.exists():
            click.echo("Headroom MCP is not configured. Nothing to uninstall.")
        else:
            click.echo("No MCP config found. Nothing to uninstall.")


@mcp.command("status")
def mcp_status() -> None:
    """Check Headroom MCP configuration status.

    \b
    Shows whether headroom is configured in Claude Code and if
    the proxy is reachable.
    """
    click.echo("Headroom MCP Status")
    click.echo("=" * 40)

    # Check MCP SDK
    try:
        import mcp  # noqa: F401

        click.echo("MCP SDK:        ✓ Installed")
    except ImportError:
        click.echo("MCP SDK:        ✗ Not installed")
        click.echo("                pip install 'headroom-ai[mcp]'")

    # Check config
    if MCP_CONFIG_PATH.exists():
        config = load_mcp_config()
        if "headroom" in config.get("mcpServers", {}):
            server_config = config["mcpServers"]["headroom"]
            click.echo("Claude Config:  ✓ Configured")
            click.echo(f"                {MCP_CONFIG_PATH}")

            # Show proxy URL
            env = server_config.get("env", {})
            proxy_url = env.get("HEADROOM_PROXY_URL", DEFAULT_PROXY_URL)
            click.echo(f"Proxy URL:      {proxy_url}")
        else:
            click.echo("Claude Config:  ✗ Not configured")
            click.echo("                Run: headroom mcp install")
    else:
        click.echo("Claude Config:  ✗ No config file")
        click.echo("                Run: headroom mcp install")

    # Check proxy connectivity
    try:
        import httpx

        config = load_mcp_config()
        env = config.get("mcpServers", {}).get("headroom", {}).get("env", {})
        proxy_url = env.get("HEADROOM_PROXY_URL", DEFAULT_PROXY_URL)

        try:
            response = httpx.get(f"{proxy_url}/health", timeout=2.0)
            if response.status_code == 200:
                click.echo(f"Proxy Status:   ✓ Running at {proxy_url}")
            else:
                click.echo(f"Proxy Status:   ✗ Unhealthy (status {response.status_code})")
        except httpx.ConnectError:
            click.echo("Proxy Status:   ✗ Not running")
            click.echo("                Run: headroom proxy")
        except httpx.TimeoutException:
            click.echo("Proxy Status:   ✗ Timeout")
    except ImportError:
        click.echo("Proxy Status:   ? (httpx not installed)")


@mcp.command("serve")
@click.option(
    "--proxy-url",
    default=None,
    envvar="HEADROOM_PROXY_URL",
    help=f"Headroom proxy URL (default: {DEFAULT_PROXY_URL})",
)
@click.option(
    "--direct",
    is_flag=True,
    help="Use direct CompressionStore access (same process as proxy)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def mcp_serve(proxy_url: str | None, direct: bool, debug: bool) -> None:
    """Start the MCP server (called by Claude Code).

    \b
    This command is typically invoked by Claude Code via the MCP config,
    not run directly. It starts the MCP server with stdio transport.

    \b
    For manual testing:
        headroom mcp serve --debug
    """
    import asyncio
    import logging

    # Check for MCP SDK
    try:
        from headroom.ccr.mcp_server import create_ccr_mcp_server
    except ImportError as e:
        click.echo(f"Error: MCP dependencies not installed: {e}", err=True)
        click.echo("Install with: pip install 'headroom-ai[mcp]'", err=True)
        raise SystemExit(1) from None

    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # Minimal logging for MCP (stdout is used for protocol)
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )

    # Use default if not specified
    effective_proxy_url = proxy_url or DEFAULT_PROXY_URL

    server = create_ccr_mcp_server(
        proxy_url=effective_proxy_url,
        direct_mode=direct,
    )

    async def run() -> None:
        try:
            await server.run_stdio()
        finally:
            await server.cleanup()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C
