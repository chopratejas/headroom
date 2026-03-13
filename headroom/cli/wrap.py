"""Wrap CLI commands to run through Headroom proxy.

Usage:
    headroom wrap claude                    # Start proxy + rtk + claude
    headroom wrap claude --no-rtk           # Without rtk hooks
    headroom wrap claude --port 9999        # Custom proxy port
    headroom wrap claude -- --model opus    # Pass args to claude
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click

from .main import main

# Proxy health check (reused from evals/suite_runner.py pattern)


def _check_proxy(port: int) -> bool:
    """Check if Headroom proxy is running on given port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(("127.0.0.1", port))
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def _start_proxy(port: int) -> subprocess.Popen:
    """Start Headroom proxy as a background subprocess."""
    cmd = [sys.executable, "-m", "headroom.cli", "proxy", "--port", str(port)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for proxy to be ready (up to 15 seconds)
    for _i in range(15):
        time.sleep(1)
        if _check_proxy(port):
            return proc
        # Check if process died
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"Proxy exited with code {proc.returncode}: {stderr[:500]}")

    proc.kill()
    raise RuntimeError(f"Proxy failed to start on port {port} within 15 seconds")


def _setup_rtk(verbose: bool = False) -> Path | None:
    """Ensure rtk is installed and hooks are registered."""
    from headroom.rtk import get_rtk_path
    from headroom.rtk.installer import ensure_rtk, register_claude_hooks

    rtk_path = get_rtk_path()

    if rtk_path:
        if verbose:
            click.echo(f"  rtk found at {rtk_path}")
    else:
        click.echo("  Downloading rtk (Rust Token Killer)...")
        rtk_path = ensure_rtk()
        if rtk_path:
            click.echo(f"  rtk installed at {rtk_path}")
        else:
            click.echo("  rtk download failed — continuing without it")
            return None

    # Register hooks (idempotent)
    if register_claude_hooks(rtk_path):
        if verbose:
            click.echo("  rtk hooks registered in Claude Code")
    else:
        click.echo("  rtk hook registration failed — continuing without it")

    return rtk_path


@main.group()
def wrap() -> None:
    """Wrap CLI tools to run through Headroom.

    \b
    Starts a Headroom proxy, configures the environment, and launches
    the target tool so all API calls route through Headroom automatically.

    \b
    Example:
        headroom wrap claude              # Proxy + rtk + Claude Code
        headroom wrap claude --no-rtk     # Proxy only, no rtk hooks
        headroom wrap claude -- -p        # Pass --print flag to claude
    """


@wrap.command()
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and hook registration")
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def claude(port: int, no_rtk: bool, no_proxy: bool, verbose: bool, claude_args: tuple) -> None:
    """Launch Claude Code through Headroom proxy.

    \b
    This command:
      1. Starts a Headroom optimization proxy on localhost
      2. Installs rtk and registers Claude Code hooks (compresses CLI output)
      3. Sets ANTHROPIC_API_URL to route through the proxy
      4. Launches 'claude' with your arguments

    \b
    All API calls from Claude Code flow through Headroom, which:
      - Compresses tool outputs (SmartCrusher, Kompress, CodeCompressor)
      - Preserves prefix cache (frozen message optimization)
      - Tracks token savings and cache hit rates

    \b
    Examples:
        headroom wrap claude                # Start everything
        headroom wrap claude -- -p          # Claude in print mode
        headroom wrap claude --port 9999    # Custom proxy port
        headroom wrap claude --no-rtk       # Skip rtk (proxy only)
    """
    # Check that claude CLI is available
    claude_bin = shutil.which("claude")
    if not claude_bin:
        click.echo("Error: 'claude' not found in PATH.")
        click.echo("Install Claude Code: https://docs.anthropic.com/en/docs/claude-code")
        raise SystemExit(1)

    proxy_proc: subprocess.Popen | None = None

    def cleanup(signum: int | None = None, frame: Any = None) -> None:
        """Clean up proxy on exit."""
        if proxy_proc and proxy_proc.poll() is None:
            proxy_proc.terminate()
            try:
                proxy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_proc.kill()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        click.echo()
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo("  ║            HEADROOM WRAP: CLAUDE              ║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        # Step 1: Start proxy
        if not no_proxy:
            if _check_proxy(port):
                click.echo(f"  Proxy already running on port {port}")
            else:
                click.echo(f"  Starting Headroom proxy on port {port}...")
                try:
                    proxy_proc = _start_proxy(port)
                    click.echo(f"  Proxy ready on http://127.0.0.1:{port}")
                except RuntimeError as e:
                    click.echo(f"  Error: {e}")
                    raise SystemExit(1) from e
        else:
            if not _check_proxy(port):
                click.echo(f"  Warning: No proxy detected on port {port}")

        # Step 2: Setup rtk
        if not no_rtk:
            click.echo("  Setting up rtk...")
            _setup_rtk(verbose=verbose)
        else:
            if verbose:
                click.echo("  Skipping rtk (--no-rtk)")

        # Step 3: Launch claude
        click.echo()
        click.echo("  Launching Claude Code (API routed through Headroom)...")
        click.echo(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:{port}")
        if claude_args:
            click.echo(f"  Extra args: {' '.join(claude_args)}")
        click.echo()

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

        # Run claude — this blocks until claude exits
        result = subprocess.run(
            [claude_bin, *claude_args],
            env=env,
        )

        raise SystemExit(result.returncode)

    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  Error: {e}")
        raise SystemExit(1) from e
    finally:
        cleanup()
