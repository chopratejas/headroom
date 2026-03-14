"""Wrap CLI commands to run through Headroom proxy.

Usage:
    headroom wrap claude                    # Start proxy + rtk + claude
    headroom wrap codex                     # Start proxy + OpenAI Codex CLI
    headroom wrap aider                     # Start proxy + aider
    headroom wrap cursor                    # Start proxy + print Cursor config instructions
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


def _get_log_path() -> Path:
    """Get path for proxy log file."""
    log_dir = Path.home() / ".headroom" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "proxy.log"


def _start_proxy(port: int) -> subprocess.Popen:
    """Start Headroom proxy as a background subprocess.

    Logs are written to ~/.headroom/logs/proxy.log to avoid pipe buffer
    deadlocks (macOS pipe buffer is ~64KB — a busy proxy fills it quickly,
    blocking the process).
    """
    cmd = [sys.executable, "-m", "headroom.cli", "proxy", "--port", str(port)]

    log_path = _get_log_path()
    log_file = open(log_path, "a")  # noqa: SIM115

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
    )

    # Wait for proxy to be ready (up to 15 seconds)
    for _i in range(15):
        time.sleep(1)
        if _check_proxy(port):
            click.echo(f"  Logs: {log_path}")
            return proc
        # Check if process died
        if proc.poll() is not None:
            log_file.close()
            # Read last few lines of log for error context
            try:
                tail = log_path.read_text()[-500:]
            except Exception:
                tail = "(no log output)"
            raise RuntimeError(f"Proxy exited with code {proc.returncode}: {tail}")

    proc.kill()
    log_file.close()
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


# rtk instructions for tools without hook support (Codex, Cursor, Aider).
# These get injected into AGENTS.md / .cursorrules so the LLM voluntarily
# uses rtk-prefixed commands. Kept concise to minimize instruction overhead.
RTK_INSTRUCTIONS_BLOCK = """\
<!-- headroom:rtk-instructions -->
# RTK (Rust Token Killer) - Token-Optimized Commands

When running shell commands, **always prefix with `rtk`**. This reduces context
usage by 60-90% with zero behavior change. If rtk has no filter for a command,
it passes through unchanged — so it is always safe to use.

## Key Commands
```bash
# Git (59-80% savings)
rtk git status          rtk git diff            rtk git log

# Files & Search (60-75% savings)
rtk ls <path>           rtk read <file>         rtk grep <pattern>
rtk find <pattern>      rtk diff <file>

# Test (90-99% savings) — shows failures only
rtk pytest tests/       rtk cargo test          rtk test <cmd>

# Build & Lint (80-90% savings) — shows errors only
rtk tsc                 rtk lint                rtk cargo build
rtk prettier --check    rtk mypy                rtk ruff check

# Analysis (70-90% savings)
rtk err <cmd>           rtk log <file>          rtk json <file>
rtk summary <cmd>       rtk deps                rtk env

# GitHub (26-87% savings)
rtk gh pr view <n>      rtk gh run list         rtk gh issue list

# Infrastructure (85% savings)
rtk docker ps           rtk kubectl get         rtk docker logs <c>

# Package managers (70-90% savings)
rtk pip list            rtk pnpm install        rtk npm run <script>
```

## Rules
- In command chains, prefix each segment: `rtk git add . && rtk git commit -m "msg"`
- For debugging, use raw command without rtk prefix
- `rtk proxy <cmd>` runs command without filtering but tracks usage
<!-- /headroom:rtk-instructions -->
"""

# Marker used to detect if instructions are already injected
_RTK_MARKER = "<!-- headroom:rtk-instructions -->"


def _ensure_rtk_binary(verbose: bool = False) -> Path | None:
    """Ensure rtk binary is installed (download if needed). No hook registration."""
    from headroom.rtk import get_rtk_path
    from headroom.rtk.installer import ensure_rtk

    rtk_path = get_rtk_path()

    if rtk_path:
        if verbose:
            click.echo(f"  rtk found at {rtk_path}")
        return rtk_path

    click.echo("  Downloading rtk (Rust Token Killer)...")
    rtk_path = ensure_rtk()
    if rtk_path:
        click.echo(f"  rtk installed at {rtk_path}")
        return rtk_path

    click.echo("  rtk download failed — continuing without it")
    return None


def _inject_rtk_instructions(file_path: Path, verbose: bool = False) -> bool:
    """Inject rtk instructions into a file (AGENTS.md, .cursorrules, etc.).

    Idempotent — skips if marker already present. Appends to existing content.
    Returns True if instructions were written.
    """
    if file_path.exists():
        existing = file_path.read_text()
        if _RTK_MARKER in existing:
            if verbose:
                click.echo(f"  rtk instructions already in {file_path.name}")
            return True
        # Append to existing file
        with open(file_path, "a") as f:
            f.write("\n\n" + RTK_INSTRUCTIONS_BLOCK)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(RTK_INSTRUCTIONS_BLOCK)

    click.echo(f"  rtk instructions injected into {file_path}")
    return True


def _ensure_proxy(port: int, no_proxy: bool) -> subprocess.Popen | None:
    """Start or verify proxy. Returns process handle if we started it."""
    if not no_proxy:
        if _check_proxy(port):
            click.echo(f"  Proxy already running on port {port}")
            return None
        else:
            click.echo(f"  Starting Headroom proxy on port {port}...")
            try:
                proc = _start_proxy(port)
                click.echo(f"  Proxy ready on http://127.0.0.1:{port}")
                return proc
            except RuntimeError as e:
                click.echo(f"  Error: {e}")
                raise SystemExit(1) from e
    else:
        if not _check_proxy(port):
            click.echo(f"  Warning: No proxy detected on port {port}")
        return None


def _make_cleanup(proxy_proc_holder: list) -> Any:
    """Create a cleanup function that terminates the proxy on exit."""

    def cleanup(signum: int | None = None, frame: Any = None) -> None:
        proc = proxy_proc_holder[0] if proxy_proc_holder else None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    return cleanup


def _launch_tool(
    binary: str,
    args: tuple,
    env: dict[str, str],
    port: int,
    no_proxy: bool,
    tool_label: str,
    env_vars_display: list[str],
) -> None:
    """Common logic: start proxy, launch tool, clean up."""
    proxy_holder: list[subprocess.Popen | None] = [None]
    cleanup = _make_cleanup(proxy_holder)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        click.echo()
        padded = f"HEADROOM WRAP: {tool_label}".center(47)
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo(f"  ║{padded}║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        proxy_holder[0] = _ensure_proxy(port, no_proxy)

        click.echo()
        click.echo(f"  Launching {tool_label} (API routed through Headroom)...")
        for var in env_vars_display:
            click.echo(f"  {var}")
        if args:
            click.echo(f"  Extra args: {' '.join(args)}")
        click.echo()

        result = subprocess.run([binary, *args], env=env)
        raise SystemExit(result.returncode)

    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  Error: {e}")
        raise SystemExit(1) from e
    finally:
        cleanup()


@main.group()
def wrap() -> None:
    """Wrap CLI tools to run through Headroom.

    \b
    Starts a Headroom proxy, configures the environment, and launches
    the target tool so all API calls route through Headroom automatically.

    \b
    Supported tools:
        headroom wrap claude              # Claude Code (Anthropic)
        headroom wrap codex               # OpenAI Codex CLI
        headroom wrap aider               # Aider
        headroom wrap cursor              # Cursor (prints config instructions)
    """


# =============================================================================
# Claude Code
# =============================================================================


@wrap.command()
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and hook registration")
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def claude(port: int, no_rtk: bool, no_proxy: bool, verbose: bool, claude_args: tuple) -> None:
    """Launch Claude Code through Headroom proxy.

    \b
    Sets ANTHROPIC_BASE_URL to route all Anthropic API calls through Headroom.

    \b
    Examples:
        headroom wrap claude                # Start everything
        headroom wrap claude -- -p          # Claude in print mode
        headroom wrap claude --port 9999    # Custom proxy port
        headroom wrap claude --no-rtk       # Skip rtk (proxy only)
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        click.echo("Error: 'claude' not found in PATH.")
        click.echo("Install Claude Code: https://docs.anthropic.com/en/docs/claude-code")
        raise SystemExit(1)

    # Setup rtk before launching (Claude-specific)
    proxy_holder: list[subprocess.Popen | None] = [None]
    cleanup = _make_cleanup(proxy_holder)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        click.echo()
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo("  ║            HEADROOM WRAP: CLAUDE              ║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        proxy_holder[0] = _ensure_proxy(port, no_proxy)

        if not no_rtk:
            click.echo("  Setting up rtk...")
            _setup_rtk(verbose=verbose)
        elif verbose:
            click.echo("  Skipping rtk (--no-rtk)")

        click.echo()
        click.echo("  Launching Claude Code (API routed through Headroom)...")
        click.echo(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:{port}")
        if claude_args:
            click.echo(f"  Extra args: {' '.join(claude_args)}")
        click.echo()

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

        result = subprocess.run([claude_bin, *claude_args], env=env)
        raise SystemExit(result.returncode)

    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  Error: {e}")
        raise SystemExit(1) from e
    finally:
        cleanup()


# =============================================================================
# OpenAI Codex CLI
# =============================================================================


@wrap.command()
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and AGENTS.md injection")
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("codex_args", nargs=-1, type=click.UNPROCESSED)
def codex(port: int, no_rtk: bool, no_proxy: bool, verbose: bool, codex_args: tuple) -> None:
    """Launch OpenAI Codex CLI through Headroom proxy.

    \b
    Sets OPENAI_BASE_URL to route all OpenAI API calls through Headroom.
    Installs rtk and injects instructions into AGENTS.md so Codex uses
    token-optimized commands (60-90% savings on shell output).

    \b
    Examples:
        headroom wrap codex                         # Start proxy + rtk + codex
        headroom wrap codex -- "fix the bug"        # Pass prompt to codex
        headroom wrap codex --no-rtk                # Skip rtk setup
        headroom wrap codex --port 9999             # Custom proxy port
    """
    codex_bin = shutil.which("codex")
    if not codex_bin:
        click.echo("Error: 'codex' not found in PATH.")
        click.echo("Install Codex CLI: npm install -g @openai/codex")
        raise SystemExit(1)

    # Setup rtk for Codex (binary + AGENTS.md instructions, no hooks)
    if not no_rtk:
        click.echo("  Setting up rtk for Codex...")
        rtk_path = _ensure_rtk_binary(verbose=verbose)
        if rtk_path:
            # Inject into project AGENTS.md (Codex reads this automatically)
            agents_md = Path.cwd() / "AGENTS.md"
            _inject_rtk_instructions(agents_md, verbose=verbose)

            # Also inject into global ~/.codex/AGENTS.md
            global_agents = Path.home() / ".codex" / "AGENTS.md"
            _inject_rtk_instructions(global_agents, verbose=verbose)

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"

    _launch_tool(
        binary=codex_bin,
        args=codex_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="CODEX",
        env_vars_display=[f"OPENAI_BASE_URL=http://127.0.0.1:{port}/v1"],
    )


# =============================================================================
# Aider
# =============================================================================


@wrap.command()
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and conventions injection")
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("aider_args", nargs=-1, type=click.UNPROCESSED)
def aider(port: int, no_rtk: bool, no_proxy: bool, verbose: bool, aider_args: tuple) -> None:
    """Launch aider through Headroom proxy.

    \b
    Sets OPENAI_API_BASE to route all API calls through Headroom.
    Installs rtk and injects instructions into .aider.conf.yml conventions
    so aider uses token-optimized commands.

    \b
    Examples:
        headroom wrap aider                              # Start proxy + rtk + aider
        headroom wrap aider -- --model gpt-4o            # Use GPT-4o
        headroom wrap aider -- --model claude-sonnet-4   # Use Claude
        headroom wrap aider --no-rtk                     # Skip rtk setup
    """
    aider_bin = shutil.which("aider")
    if not aider_bin:
        click.echo("Error: 'aider' not found in PATH.")
        click.echo("Install aider: pip install aider-chat")
        raise SystemExit(1)

    # Setup rtk for aider (binary + CONVENTIONS.md instructions)
    if not no_rtk:
        click.echo("  Setting up rtk for aider...")
        rtk_path = _ensure_rtk_binary(verbose=verbose)
        if rtk_path:
            # aider reads CONVENTIONS.md from project root
            conventions = Path.cwd() / "CONVENTIONS.md"
            _inject_rtk_instructions(conventions, verbose=verbose)

    env = os.environ.copy()
    env["OPENAI_API_BASE"] = f"http://127.0.0.1:{port}/v1"
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

    _launch_tool(
        binary=aider_bin,
        args=aider_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="AIDER",
        env_vars_display=[
            f"OPENAI_API_BASE=http://127.0.0.1:{port}/v1",
            f"ANTHROPIC_BASE_URL=http://127.0.0.1:{port}",
        ],
    )


# =============================================================================
# Cursor
# =============================================================================


@wrap.command()
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and .cursorrules injection")
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def cursor(port: int, no_rtk: bool, no_proxy: bool, verbose: bool) -> None:
    """Start Headroom proxy for use with Cursor.

    \b
    Cursor reads its API configuration from its settings UI, not from
    environment variables. This command starts the proxy, installs rtk
    with .cursorrules instructions, and prints the Cursor settings.

    \b
    After running this command, open Cursor and configure:
        Settings > Models > OpenAI API Key > Advanced > Override Base URL

    \b
    Example:
        headroom wrap cursor                # Start proxy + rtk + instructions
        headroom wrap cursor --no-rtk       # Proxy only, no rtk
        headroom wrap cursor --port 9999    # Custom proxy port
    """
    proxy_holder: list[subprocess.Popen | None] = [None]
    cleanup = _make_cleanup(proxy_holder)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        click.echo()
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo("  ║            HEADROOM WRAP: CURSOR              ║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        proxy_holder[0] = _ensure_proxy(port, no_proxy)

        # Setup rtk for Cursor (binary + .cursorrules instructions)
        if not no_rtk:
            click.echo("  Setting up rtk for Cursor...")
            rtk_path = _ensure_rtk_binary(verbose=verbose)
            if rtk_path:
                cursorrules = Path.cwd() / ".cursorrules"
                _inject_rtk_instructions(cursorrules, verbose=verbose)

        click.echo()
        click.echo("  Headroom proxy is running. Configure Cursor:")
        click.echo()
        click.echo("  For OpenAI models:")
        click.echo(f"    Base URL:  http://127.0.0.1:{port}/v1")
        click.echo("    API Key:   your-openai-api-key")
        click.echo()
        click.echo("  For Anthropic models:")
        click.echo(f"    Base URL:  http://127.0.0.1:{port}")
        click.echo("    API Key:   your-anthropic-api-key")
        click.echo()
        click.echo("  In Cursor:")
        click.echo("    Settings > Models > OpenAI API Key > Override OpenAI Base URL")
        click.echo(f"    Set to: http://127.0.0.1:{port}/v1")
        if not no_rtk:
            click.echo()
            click.echo("  rtk instructions injected into .cursorrules")
            click.echo("  Cursor will use token-optimized commands automatically.")
        click.echo()
        click.echo("  Press Ctrl+C to stop the proxy.")
        click.echo()

        # Block until Ctrl+C
        try:
            while True:
                time.sleep(1)
                proc = proxy_holder[0]
                if proc and proc.poll() is not None:
                    click.echo("  Proxy process exited unexpectedly.")
                    raise SystemExit(1)
        except KeyboardInterrupt:
            click.echo("\n  Shutting down...")

    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  Error: {e}")
        raise SystemExit(1) from e
    finally:
        cleanup()
