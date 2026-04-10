"""Wrap CLI commands to run through Headroom proxy.

Usage:
    headroom wrap claude                    # Start proxy + rtk + claude
    headroom wrap codex                     # Start proxy + OpenAI Codex CLI
    headroom wrap aider                     # Start proxy + aider
    headroom wrap cursor                    # Start proxy + print Cursor config instructions
    headroom wrap openclaw                  # Install + configure OpenClaw plugin
    headroom wrap claude --no-rtk           # Without rtk hooks
    headroom wrap claude --port 9999        # Custom proxy port
    headroom wrap claude -- --model opus    # Pass args to claude
"""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Fix Windows cp1252 encoding — box-drawing characters require UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import click

from .main import main


def _print_telemetry_notice() -> None:
    """Print a telemetry notice when anonymous telemetry is enabled.

    Respects the HEADROOM_TELEMETRY and HEADROOM_TELEMETRY_WARN feature flags.
    Does nothing when telemetry or warnings are disabled.
    """
    from headroom.telemetry.beacon import format_telemetry_notice

    notice = format_telemetry_notice(prefix="  ")
    if notice:
        click.echo(notice)


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


def _start_proxy(
    port: int,
    *,
    learn: bool = False,
    agent_type: str = "unknown",
    backend: str | None = None,
    anyllm_provider: str | None = None,
    region: str | None = None,
) -> subprocess.Popen:
    """Start Headroom proxy as a background subprocess.

    Logs are written to ~/.headroom/logs/proxy.log to avoid pipe buffer
    deadlocks (macOS pipe buffer is ~64KB — a busy proxy fills it quickly,
    blocking the process).
    """
    cmd = [sys.executable, "-m", "headroom.cli", "proxy", "--port", str(port)]

    # Forward HEADROOM_MODE env var so the proxy respects the user's mode choice
    headroom_mode = os.environ.get("HEADROOM_MODE")
    if headroom_mode:
        cmd.extend(["--mode", headroom_mode])

    # Forward --learn flag to proxy subprocess
    if learn:
        cmd.append("--learn")

    # Forward backend configuration to proxy subprocess
    _backend = backend or os.environ.get("HEADROOM_BACKEND")
    if _backend:
        cmd.extend(["--backend", _backend])

    _anyllm = anyllm_provider or os.environ.get("HEADROOM_ANYLLM_PROVIDER")
    if _anyllm:
        cmd.extend(["--anyllm-provider", _anyllm])

    _region = region or os.environ.get("HEADROOM_REGION")
    if _region:
        cmd.extend(["--region", _region])

    log_path = _get_log_path()
    log_file = open(log_path, "a")  # noqa: SIM115

    # Ensure proxy subprocess uses UTF-8 (Windows defaults to cp1252)
    proxy_env = os.environ.copy()
    proxy_env["PYTHONIOENCODING"] = "utf-8"

    # Tell the proxy which agent is being wrapped (for traffic learning output)
    if agent_type != "unknown":
        proxy_env["HEADROOM_AGENT_TYPE"] = agent_type

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        env=proxy_env,
    )

    # Wait for proxy to be ready (up to 45 seconds).
    # ML components (Kompress, Magika, Tree-sitter) load synchronously before
    # uvicorn binds the port. On slower machines this can take 20-30 seconds.
    for _i in range(45):
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
    raise RuntimeError(f"Proxy failed to start on port {port} within 45 seconds")


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


def _inject_codex_provider_config(port: int) -> None:
    """Inject a Headroom model provider into Codex's config.toml.

    Codex ignores OPENAI_BASE_URL for WebSocket transport unless a custom
    provider declares ``supports_websockets = true``.  This writes a
    ``[model_providers.headroom]`` section that routes both HTTP and WS
    through the proxy, and sets ``model_provider = "headroom"``.

    Safe to call multiple times — only writes if the section is missing
    or the port changed.
    """
    config_dir = Path.home() / ".codex"
    config_file = config_dir / "config.toml"

    headroom_section = (
        f"\n# --- Headroom proxy (auto-injected by headroom wrap codex) ---\n"
        f'model_provider = "headroom"\n'
        f"\n"
        f"[model_providers.headroom]\n"
        f'name = "OpenAI via Headroom proxy"\n'
        f'base_url = "http://127.0.0.1:{port}/v1"\n'
        f'env_key = "OPENAI_API_KEY"\n'
        f"requires_openai_auth = true\n"
        f"supports_websockets = true\n"
        f"# --- end Headroom ---\n"
    )

    marker = "# --- Headroom proxy (auto-injected by headroom wrap codex) ---"
    end_marker = "# --- end Headroom ---"

    try:
        config_dir.mkdir(parents=True, exist_ok=True)

        if config_file.exists():
            content = config_file.read_text()
            if marker in content:
                # Replace existing section
                start = content.index(marker)
                end = content.index(end_marker) + len(end_marker)
                content = content[:start].rstrip() + headroom_section + content[end:].lstrip("\n")
            else:
                content = content.rstrip() + "\n" + headroom_section
        else:
            content = headroom_section

        config_file.write_text(content)
        click.echo(f"  Codex config: injected Headroom provider (WS + HTTP) into {config_file}")
    except Exception as e:
        click.echo(f"  Warning: could not update Codex config: {e}")


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


def _ensure_proxy(
    port: int,
    no_proxy: bool,
    *,
    learn: bool = False,
    agent_type: str = "unknown",
    backend: str | None = None,
    anyllm_provider: str | None = None,
    region: str | None = None,
) -> subprocess.Popen | None:
    """Start or verify proxy. Returns process handle if we started it."""
    if not no_proxy:
        if _check_proxy(port):
            click.echo(f"  Proxy already running on port {port}")
            return None
        else:
            click.echo(f"  Starting Headroom proxy on port {port}...")
            try:
                proc = _start_proxy(
                    port,
                    learn=learn,
                    agent_type=agent_type,
                    backend=backend,
                    anyllm_provider=anyllm_provider,
                    region=region,
                )
                click.echo(f"  Proxy ready on http://127.0.0.1:{port}")
                return proc
            except RuntimeError as e:
                click.echo(f"  Error: {e}")
                raise SystemExit(1) from e
    else:
        if not _check_proxy(port):
            click.echo(f"  Warning: No proxy detected on port {port}")
        return None


def _make_cleanup(proxy_proc_holder: list, port: int = 8787) -> Any:
    """Create a cleanup function that terminates the proxy on exit.

    Only kills the proxy if no other headroom-wrapped clients are using it.
    Checks by looking for other processes with ANTHROPIC_BASE_URL or
    OPENAI_BASE_URL pointing at our port.
    """

    def _other_clients_exist() -> bool:
        """Check if other processes are using this proxy."""
        try:
            # Count headroom wrap processes (excluding ourselves)
            result = subprocess.run(
                ["pgrep", "-f", f"127.0.0.1:{port}"],
                capture_output=True,
                text=True,
            )
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            my_pid = str(os.getpid())
            other_pids = [p for p in pids if p != my_pid]
            return len(other_pids) > 0
        except Exception:
            return False  # If we can't check, assume no others

    def cleanup(signum: int | None = None, frame: Any = None) -> None:
        proc = proxy_proc_holder[0] if proxy_proc_holder else None
        if proc and proc.poll() is None:
            if _other_clients_exist():
                # Other clients still using the proxy — leave it running
                return
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
    *,
    learn: bool = False,
    agent_type: str = "unknown",
    backend: str | None = None,
    anyllm_provider: str | None = None,
    region: str | None = None,
) -> None:
    """Common logic: start proxy, launch tool, clean up."""
    proxy_holder: list[subprocess.Popen | None] = [None]
    cleanup = _make_cleanup(proxy_holder, port)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        click.echo()
        padded = f"HEADROOM WRAP: {tool_label}".center(47)
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo(f"  ║{padded}║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        proxy_holder[0] = _ensure_proxy(
            port,
            no_proxy,
            learn=learn,
            agent_type=agent_type,
            backend=backend,
            anyllm_provider=anyllm_provider,
            region=region,
        )

        click.echo()
        click.echo(f"  Launching {tool_label} (API routed through Headroom)...")
        for var in env_vars_display:
            click.echo(f"  {var}")
        if args:
            click.echo(f"  Extra args: {' '.join(args)}")
        _print_telemetry_notice()
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


def _run_checked(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    action: str,
) -> subprocess.CompletedProcess[str]:
    """Run subprocess and raise a ClickException with actionable context on failure."""
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as e:
        raise click.ClickException(f"{action} failed: command not found: {cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        details = stderr or stdout or f"exit code {e.returncode}"
        raise click.ClickException(f"{action} failed: {details}") from e


def _resolve_openclaw_extensions_dir(openclaw_bin: str) -> Path:
    """Resolve OpenClaw extension root from active config file path."""
    result = _run_checked([openclaw_bin, "config", "file"], action="openclaw config file")
    lines = result.stdout.strip().splitlines()
    config_path_str = lines[-1].strip() if lines else ""
    if not config_path_str:
        raise click.ClickException(
            "Unable to resolve OpenClaw config path from `openclaw config file`."
        )
    config_path = Path(config_path_str).expanduser()
    return config_path.parent / "extensions"


def _normalize_openclaw_gateway_provider_ids(provider_ids: tuple[str, ...] | None) -> list[str]:
    """Normalize configured OpenClaw provider ids, defaulting to openai-codex."""
    values = provider_ids or ()
    seen: set[str] = set()
    normalized: list[str] = []

    for entry in values:
        provider_id = entry.strip()
        if not provider_id or provider_id in seen:
            continue
        seen.add(provider_id)
        normalized.append(provider_id)

    if normalized:
        return normalized
    return ["openai-codex"]


def _read_openclaw_config_value(openclaw_bin: str, path: str) -> Any | None:
    """Read an OpenClaw config value when present, returning None on missing paths."""
    result = subprocess.run(
        [openclaw_bin, "config", "get", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    if not output:
        return None

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


def _build_openclaw_plugin_entry(
    *,
    existing_entry: Any,
    proxy_port: int,
    startup_timeout_ms: int,
    python_path: str | None,
    no_auto_start: bool,
    gateway_provider_ids: tuple[str, ...] | None,
    enabled: bool,
) -> dict[str, object]:
    """Merge managed Headroom plugin settings with any existing entry payload."""
    base_entry = existing_entry if isinstance(existing_entry, dict) else {}
    existing_config = base_entry.get("config")
    next_config = dict(existing_config) if isinstance(existing_config, dict) else {}

    next_config["proxyPort"] = proxy_port
    next_config["autoStart"] = not no_auto_start
    next_config["startupTimeoutMs"] = startup_timeout_ms
    next_config["gatewayProviderIds"] = _normalize_openclaw_gateway_provider_ids(
        gateway_provider_ids
    )

    if python_path:
        next_config["pythonPath"] = python_path
    else:
        next_config.pop("pythonPath", None)

    return {
        **base_entry,
        "enabled": enabled,
        "config": next_config,
    }


def _write_openclaw_plugin_entry(openclaw_bin: str, entry: dict[str, object]) -> None:
    """Persist the Headroom plugin config entry."""
    _run_checked(
        [
            openclaw_bin,
            "config",
            "set",
            "plugins.entries.headroom",
            json.dumps(entry, separators=(",", ":")),
            "--strict-json",
        ],
        action="openclaw config set plugins.entries.headroom",
    )


def _set_openclaw_context_engine_slot(openclaw_bin: str, engine_id: str) -> None:
    """Persist the selected OpenClaw context engine slot."""
    _run_checked(
        [
            openclaw_bin,
            "config",
            "set",
            "plugins.slots.contextEngine",
            json.dumps(engine_id),
            "--strict-json",
        ],
        action="openclaw config set plugins.slots.contextEngine",
    )


def _restart_or_start_openclaw_gateway(openclaw_bin: str) -> tuple[str, str]:
    """Restart the gateway when running, otherwise start it."""
    restart_result = subprocess.run(
        [openclaw_bin, "gateway", "restart"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if restart_result.returncode == 0:
        output = restart_result.stdout.strip() or restart_result.stderr.strip()
        return "restarted", output

    start_result = _run_checked(
        [openclaw_bin, "gateway", "start"],
        action="openclaw gateway start",
    )
    output = start_result.stdout.strip() or start_result.stderr.strip()
    return "started", output


def _copy_openclaw_plugin_into_extensions(
    *,
    plugin_dir: Path,
    openclaw_bin: str,
) -> Path:
    """Fallback install path when `openclaw plugins install` is blocked on linked source."""
    dist_dir = plugin_dir / "dist"
    if not dist_dir.exists():
        raise click.ClickException(
            f"Plugin dist folder missing at {dist_dir}. Build the plugin first."
        )
    hook_shim_dir = plugin_dir / "hook-shim"
    if not hook_shim_dir.exists():
        raise click.ClickException(
            f"Plugin hook-shim folder missing at {hook_shim_dir}. Build the plugin first."
        )

    extensions_dir = _resolve_openclaw_extensions_dir(openclaw_bin)
    target_dir = extensions_dir / "headroom"
    target_dist = target_dir / "dist"
    target_hook_shim = target_dir / "hook-shim"
    target_dir.mkdir(parents=True, exist_ok=True)
    if target_dist.exists():
        shutil.rmtree(target_dist)
    if target_hook_shim.exists():
        shutil.rmtree(target_hook_shim)
    shutil.copytree(dist_dir, target_dist)
    shutil.copytree(hook_shim_dir, target_hook_shim)

    for filename in ("openclaw.plugin.json", "package.json", "README.md"):
        source = plugin_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)

    return target_dir


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
        headroom wrap openclaw            # OpenClaw plugin bootstrap
    """


@main.group()
def unwrap() -> None:
    """Undo durable Headroom wrapping for supported tools."""


# =============================================================================
# Claude Code
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and hook registration")
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--learn", is_flag=True, help="Enable live traffic learning (patterns saved to MEMORY.md)"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def claude(
    port: int, no_rtk: bool, no_proxy: bool, learn: bool, verbose: bool, claude_args: tuple
) -> None:
    """Launch Claude Code through Headroom proxy.

    \b
    Sets ANTHROPIC_BASE_URL to route all Anthropic API calls through Headroom.
    All unknown flags are passed through to claude (e.g. --resume, --model).

    \b
    Examples:
        headroom wrap claude                # Start everything
        headroom wrap claude --resume <id>  # Resume a session
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
    cleanup = _make_cleanup(proxy_holder, port)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        click.echo()
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo("  ║            HEADROOM WRAP: CLAUDE              ║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        proxy_holder[0] = _ensure_proxy(port, no_proxy, learn=learn, agent_type="claude")

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
        _print_telemetry_notice()
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


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and AGENTS.md injection")
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--learn", is_flag=True, help="Enable live traffic learning (patterns saved to AGENTS.md)"
)
@click.option(
    "--backend",
    default=None,
    help="API backend for the proxy: 'anthropic', 'anyllm', 'litellm-vertex', etc. (env: HEADROOM_BACKEND)",
)
@click.option(
    "--anyllm-provider",
    default=None,
    help="Provider for any-llm backend: openai, mistral, groq, etc. (env: HEADROOM_ANYLLM_PROVIDER)",
)
@click.option(
    "--region", default=None, help="Cloud region for Bedrock/Vertex (env: HEADROOM_REGION)"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("codex_args", nargs=-1, type=click.UNPROCESSED)
def codex(
    port: int,
    no_rtk: bool,
    no_proxy: bool,
    learn: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    codex_args: tuple,
) -> None:
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
        headroom wrap codex --backend anyllm --anyllm-provider groq
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

    # Inject Headroom provider into Codex config so WebSocket traffic also
    # routes through the proxy.  Codex ignores OPENAI_BASE_URL for its WS
    # transport unless a custom provider declares supports_websockets = true.
    _inject_codex_provider_config(port)

    _launch_tool(
        binary=codex_bin,
        args=codex_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="CODEX",
        env_vars_display=[f"OPENAI_BASE_URL=http://127.0.0.1:{port}/v1"],
        learn=learn,
        agent_type="codex",
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
    )


# =============================================================================
# Aider
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and conventions injection")
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option(
    "--backend", default=None, help="API backend: 'anthropic', 'anyllm', 'litellm-vertex', etc."
)
@click.option("--anyllm-provider", default=None, help="Provider for any-llm backend")
@click.option("--region", default=None, help="Cloud region for Bedrock/Vertex")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("aider_args", nargs=-1, type=click.UNPROCESSED)
def aider(
    port: int,
    no_rtk: bool,
    no_proxy: bool,
    learn: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    aider_args: tuple,
) -> None:
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
        headroom wrap aider --backend litellm-vertex --region us-central1
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
        learn=learn,
        agent_type="aider",
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
    )


# =============================================================================
# Cursor
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and .cursorrules injection")
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--learn", is_flag=True, help="Enable live traffic learning (patterns saved to .cursor/rules/)"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def cursor(port: int, no_rtk: bool, no_proxy: bool, learn: bool, verbose: bool) -> None:
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
    cleanup = _make_cleanup(proxy_holder, port)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        click.echo()
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo("  ║            HEADROOM WRAP: CURSOR              ║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        proxy_holder[0] = _ensure_proxy(port, no_proxy, learn=learn, agent_type="cursor")

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


# =============================================================================
# OpenClaw
# =============================================================================


@wrap.command("openclaw")
@click.option(
    "--plugin-path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Path to local OpenClaw plugin source directory (advanced/dev override)",
)
@click.option(
    "--plugin-spec",
    default="headroom-ai/openclaw",
    show_default=True,
    help="NPM plugin spec for OpenClaw install (used when --plugin-path is omitted)",
)
@click.option(
    "--skip-build",
    is_flag=True,
    help="Skip npm install/build in local source mode (--plugin-path)",
)
@click.option(
    "--copy",
    is_flag=True,
    help="Install by copying plugin path instead of using --link",
)
@click.option("--proxy-port", default=8787, type=int, help="Headroom proxy port")
@click.option("--startup-timeout-ms", default=20000, type=int, help="Proxy startup timeout")
@click.option(
    "--gateway-provider-id",
    "gateway_provider_ids",
    multiple=True,
    help="OpenClaw provider id to route through Headroom (repeatable; default: openai-codex)",
)
@click.option(
    "--python-path",
    default=None,
    help="Optional Python executable for proxy launcher fallback",
)
@click.option(
    "--no-auto-start",
    is_flag=True,
    help="Disable plugin auto-start of local headroom proxy",
)
@click.option(
    "--no-restart",
    is_flag=True,
    help="Do not restart OpenClaw gateway at the end",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def openclaw(
    plugin_path: Path | None,
    plugin_spec: str,
    skip_build: bool,
    copy: bool,
    proxy_port: int,
    startup_timeout_ms: int,
    gateway_provider_ids: tuple[str, ...],
    python_path: str | None,
    no_auto_start: bool,
    no_restart: bool,
    verbose: bool,
) -> None:
    """Install and configure Headroom OpenClaw plugin in one command.

    \b
    What this command does:
      1. Installs OpenClaw plugin from npm (or local --plugin-path)
      2. Builds plugin source if --plugin-path is used
      3. Writes minimal plugin config and sets contextEngine slot
      4. Validates config
      5. Restarts OpenClaw gateway (unless --no-restart)

    \b
    Example:
      headroom wrap openclaw
      headroom wrap openclaw --plugin-path C:\\git\\headroom\\plugins\\openclaw
    """
    openclaw_bin = shutil.which("openclaw")
    if not openclaw_bin:
        raise click.ClickException("'openclaw' not found in PATH. Install OpenClaw CLI first.")

    plugin_dir = plugin_path.resolve() if plugin_path else None
    local_source_mode = plugin_dir is not None
    if plugin_dir:
        if not plugin_dir.exists():
            raise click.ClickException(f"Plugin path not found: {plugin_dir}.")
        if not (plugin_dir / "package.json").exists():
            raise click.ClickException(f"Invalid plugin path (missing package.json): {plugin_dir}")
        if not (plugin_dir / "openclaw.plugin.json").exists():
            raise click.ClickException(
                f"Invalid plugin path (missing openclaw.plugin.json): {plugin_dir}"
            )

    npm_bin = shutil.which("npm")
    if local_source_mode and not skip_build and not npm_bin:
        raise click.ClickException(
            "'npm' not found in PATH. Install Node/npm or rerun with --skip-build."
        )

    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║           HEADROOM WRAP: OPENCLAW             ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()
    if local_source_mode:
        click.echo(f"  Plugin source: local ({plugin_dir})")
    else:
        click.echo(f"  Plugin source: npm ({plugin_spec})")

    if local_source_mode and not skip_build:
        click.echo("  Building OpenClaw plugin (npm install + npm run build)...")
        _run_checked([npm_bin or "npm", "install"], cwd=plugin_dir, action="npm install")
        _run_checked([npm_bin or "npm", "run", "build"], cwd=plugin_dir, action="npm run build")
    elif not local_source_mode and skip_build:
        click.echo("  Skipping build: npm install mode does not build local source.")

    effective_python_path = python_path
    if effective_python_path is None and not no_auto_start and sys.executable:
        effective_python_path = sys.executable

    existing_entry = _read_openclaw_config_value(openclaw_bin, "plugins.entries.headroom")
    entry = _build_openclaw_plugin_entry(
        existing_entry=existing_entry,
        proxy_port=proxy_port,
        startup_timeout_ms=startup_timeout_ms,
        python_path=effective_python_path,
        no_auto_start=no_auto_start,
        gateway_provider_ids=gateway_provider_ids,
        enabled=True,
    )

    click.echo("  Writing plugin configuration...")
    _write_openclaw_plugin_entry(openclaw_bin, entry)

    install_cmd = [
        openclaw_bin,
        "plugins",
        "install",
        "--dangerously-force-unsafe-install",
    ]
    if local_source_mode:
        if copy:
            install_cmd.append(str(plugin_dir))
            install_cwd = None
        else:
            install_cmd.extend(["--link", "."])
            install_cwd = plugin_dir
    else:
        install_cmd.append(plugin_spec)
        install_cwd = None

    click.echo("  Installing OpenClaw plugin with required unsafe-install flag...")
    install_result = subprocess.run(
        install_cmd,
        cwd=str(install_cwd) if install_cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if install_result.returncode != 0:
        combined_error = "\n".join(
            x for x in [install_result.stderr.strip(), install_result.stdout.strip()] if x
        )
        plugin_already_exists = "plugin already exists" in combined_error.lower()
        linked_install_bug = (
            "also not a valid hook pack" in combined_error.lower()
            and "--dangerously-force-unsafe-install" in " ".join(install_cmd)
        )
        if plugin_already_exists:
            click.echo("  Plugin already installed; continuing with configuration/update steps.")
        elif linked_install_bug and local_source_mode and plugin_dir is not None:
            click.echo(
                "  OpenClaw linked-path install bug detected; applying extension-path fallback..."
            )
            target_dir = _copy_openclaw_plugin_into_extensions(
                plugin_dir=plugin_dir,
                openclaw_bin=openclaw_bin,
            )
            click.echo(f"  Fallback plugin copy completed: {target_dir}")
        else:
            details = combined_error or f"exit code {install_result.returncode}"
            raise click.ClickException(f"openclaw plugins install failed: {details}")
    elif verbose and install_result.stdout.strip():
        click.echo(install_result.stdout.strip())

    _set_openclaw_context_engine_slot(openclaw_bin, "headroom")
    _run_checked(
        [openclaw_bin, "config", "validate"],
        action="openclaw config validate",
    )

    if no_restart:
        click.echo("  Skipping gateway restart (--no-restart).")
        click.echo(
            "  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply plugin changes."
        )
    else:
        click.echo("  Applying plugin changes to OpenClaw gateway...")
        gateway_action, gateway_output = _restart_or_start_openclaw_gateway(openclaw_bin)
        click.echo(f"  Gateway {gateway_action}.")
        if verbose and gateway_output:
            click.echo(gateway_output)

    inspect_result = _run_checked(
        [openclaw_bin, "plugins", "inspect", "headroom"],
        action="openclaw plugins inspect headroom",
    )
    if verbose and inspect_result.stdout.strip():
        click.echo(inspect_result.stdout.strip())

    click.echo()
    click.echo("✓ OpenClaw is configured to use Headroom context compression.")
    click.echo("  Plugin: headroom")
    click.echo("  Slot:   plugins.slots.contextEngine = headroom")
    click.echo()


@unwrap.command("openclaw")
@click.option("--no-restart", is_flag=True, help="Do not restart OpenClaw gateway at the end")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def unwrap_openclaw(no_restart: bool, verbose: bool) -> None:
    """Disable the Headroom OpenClaw plugin and restore the legacy engine slot."""
    openclaw_bin = shutil.which("openclaw")
    if not openclaw_bin:
        raise click.ClickException("'openclaw' not found in PATH. Install OpenClaw CLI first.")

    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║          HEADROOM UNWRAP: OPENCLAW            ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()
    click.echo("  Disabling Headroom plugin and removing engine mapping...")

    existing_entry = _read_openclaw_config_value(openclaw_bin, "plugins.entries.headroom")
    existing_config = {}
    if isinstance(existing_entry, dict) and isinstance(existing_entry.get("config"), dict):
        existing_config = {
            key: value
            for key, value in existing_entry["config"].items()
            if key
            not in {
                "gatewayProviderIds",
                "proxyUrl",
                "proxyPort",
                "autoStart",
                "startupTimeoutMs",
                "pythonPath",
            }
        }

    entry = {"enabled": False, "config": existing_config}
    _write_openclaw_plugin_entry(openclaw_bin, entry)
    _set_openclaw_context_engine_slot(openclaw_bin, "legacy")
    _run_checked(
        [openclaw_bin, "config", "validate"],
        action="openclaw config validate",
    )

    if no_restart:
        click.echo("  Skipping gateway restart (--no-restart).")
        click.echo(
            "  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply unwrap changes."
        )
    else:
        click.echo("  Applying unwrap changes to OpenClaw gateway...")
        gateway_action, gateway_output = _restart_or_start_openclaw_gateway(openclaw_bin)
        click.echo(f"  Gateway {gateway_action}.")
        if verbose and gateway_output:
            click.echo(gateway_output)

    if verbose:
        inspect_result = _run_checked(
            [openclaw_bin, "plugins", "inspect", "headroom"],
            action="openclaw plugins inspect headroom",
        )
        if inspect_result.stdout.strip():
            click.echo(inspect_result.stdout.strip())

    click.echo()
    click.echo("✓ OpenClaw Headroom wrap removed.")
    click.echo("  Plugin: headroom (installed, disabled)")
    click.echo("  Slot:   plugins.slots.contextEngine = legacy")
    click.echo()
