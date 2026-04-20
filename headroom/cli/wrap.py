"""Wrap CLI commands to run through Headroom proxy.

Usage:
    headroom wrap claude                    # Start proxy + rtk + claude
    headroom wrap copilot -- --model ...    # Start proxy + launch GitHub Copilot CLI
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
import urllib.error
import urllib.request
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
    from headroom import paths as _paths

    log_dir = _paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "proxy.log"


def _start_proxy(
    port: int,
    *,
    learn: bool = False,
    memory: bool = False,
    agent_type: str = "unknown",
    code_graph: bool = False,
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

    # Forward --memory flag to proxy subprocess
    if memory:
        cmd.append("--memory")

    # Forward --code-graph flag to proxy subprocess (live file watcher)
    if code_graph:
        cmd.append("--code-graph")

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


_CBM_MCP_SERVER_NAME = "codebase-memory-mcp"


def _register_cbm_mcp_server(cbm_bin: str) -> None:
    """Register codebase-memory-mcp as an MCP server in Claude Code.

    Uses ``claude mcp add`` so the tools appear in ``/mcp`` automatically.
    Idempotent — skips if already registered.
    """
    claude_cli = shutil.which("claude")
    if not claude_cli:
        return

    # Check if already registered
    check = subprocess.run(
        [claude_cli, "mcp", "get", _CBM_MCP_SERVER_NAME],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return  # Already registered

    result = subprocess.run(
        [claude_cli, "mcp", "add", _CBM_MCP_SERVER_NAME, "-s", "user", "--", cbm_bin],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        click.echo(f"  Code graph: registered {_CBM_MCP_SERVER_NAME} MCP server")
    else:
        pass  # Non-critical — tools won't appear in /mcp but graph still works


def _setup_code_graph(verbose: bool = False) -> bool:
    """Ensure codebase-memory-mcp is installed, registered as MCP server, and project is indexed.

    codebase-memory-mcp builds a knowledge graph of the codebase using
    tree-sitter, enabling the LLM to query code structure (call chains,
    function definitions, impact analysis) instead of reading entire files.

    Steps:
    1. Download the binary if not already present.
    2. Register as an MCP server in Claude Code (``claude mcp add``).
    3. Index the current project (fast, idempotent).

    With Claude Code's MCP Tool Search, the 14 graph tools add ~200 tokens
    overhead per request (not the full ~1,915) — they're lazy-loaded.

    Returns True if graph is ready, False if setup failed.
    """
    from headroom.graph.installer import ensure_cbm, get_cbm_path

    cbm_path = get_cbm_path()
    if not cbm_path:
        click.echo("  Code graph: downloading codebase-memory-mcp...")
        cbm_path = ensure_cbm()
        if cbm_path:
            click.echo(f"  Code graph: installed at {cbm_path}")
        else:
            click.echo("  Code graph: download failed — skipping")
            return False

    cbm_bin = str(cbm_path)

    # Register as MCP server so tools appear in /mcp
    _register_cbm_mcp_server(cbm_bin)

    # Index current project (fast — ~1s for most repos, idempotent)
    project_dir = str(Path.cwd())
    try:
        result = subprocess.run(
            [
                cbm_bin,
                "cli",
                "index_repository",
                json.dumps({"repo_path": project_dir, "mode": "fast"}),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Parse node/edge counts from output
            for line in result.stdout.splitlines():
                if '"nodes"' in line:
                    try:
                        # Parse the JSON response to extract node/edge counts
                        import re

                        m_nodes = re.search(r'"nodes":(\d+)', line)
                        m_edges = re.search(r'"edges":(\d+)', line)
                        if m_nodes and m_edges:
                            nodes = int(m_nodes.group(1))
                            edges = int(m_edges.group(1))
                            click.echo(
                                f"  Code graph: indexed ({nodes:,} symbols, "
                                f"{edges:,} relationships)"
                            )
                        else:
                            click.echo("  Code graph: indexed")
                    except (ValueError, AttributeError):
                        click.echo("  Code graph: indexed")
                    return True
            click.echo("  Code graph: indexed")
            return True
        else:
            if verbose:
                click.echo(f"  Code graph: indexing failed ({result.stderr[:100]})")
            return False
    except subprocess.TimeoutExpired:
        click.echo("  Code graph: indexing timed out (will complete in background)")
        return False
    except Exception as e:
        if verbose:
            click.echo(f"  Code graph: setup failed ({e})")
        return False


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

# Memory MCP markers
_MEMORY_MCP_MARKER = "# --- Headroom memory MCP (auto-injected) ---"
_MEMORY_MCP_END = "# --- end Headroom memory ---"
_MEMORY_AGENTS_MARKER = "<!-- headroom:memory-instructions -->"


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


def _prepare_wrap_rtk(verbose: bool = False, *, label: str | None = None) -> Path | None:
    """Ensure rtk is present for host-bridged wrap flows without host-specific setup."""
    if label:
        click.echo(f"  Preparing rtk for {label}...")
    return _ensure_rtk_binary(verbose=verbose)


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

    # model_provider must be a top-level TOML key (before any [section]).
    # The [model_providers.headroom] table can go at the end.
    top_level_marker = "# --- Headroom proxy (auto-injected by headroom wrap codex) ---"
    top_level_block = f'{top_level_marker}\nmodel_provider = "headroom"\n'
    provider_section = (
        f"\n[model_providers.headroom]\n"
        f'name = "OpenAI via Headroom proxy"\n'
        f'base_url = "http://127.0.0.1:{port}/v1"\n'
        f'env_key = "OPENAI_API_KEY"\n'
        f"requires_openai_auth = true\n"
        f"supports_websockets = true\n"
        f"# --- end Headroom ---\n"
    )

    marker = top_level_marker
    end_marker = "# --- end Headroom ---"

    try:
        config_dir.mkdir(parents=True, exist_ok=True)

        if config_file.exists():
            content = config_file.read_text()
            if marker in content:
                # Remove existing Headroom blocks entirely
                start = content.index(marker)
                end = content.index(end_marker) + len(end_marker)
                content = content[:start].rstrip("\n") + content[end:].lstrip("\n")

            # Strip any stale top-level model_provider left behind
            import re

            content = re.sub(r'\nmodel_provider\s*=\s*"headroom"\n', "\n", content)

            # Place top-level key at the very beginning, provider table at the end
            content = top_level_block + "\n" + content.strip() + "\n" + provider_section
        else:
            content = top_level_block + "\n" + provider_section

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


def _inject_memory_mcp_config(db_path: str, user_id: str) -> None:
    """Register headroom memory as an MCP server in Codex's config.toml.

    Idempotent — replaces existing section if present.
    """
    import sys

    config_dir = Path.home() / ".codex"
    config_file = config_dir / "config.toml"

    # Use forward slashes in TOML paths (works on all platforms, avoids
    # backslash escaping issues on Windows)
    python_bin = sys.executable.replace("\\", "/")
    db_path_toml = db_path.replace("\\", "/")
    mcp_section = (
        f"\n{_MEMORY_MCP_MARKER}\n"
        f"[mcp_servers.headroom_memory]\n"
        f'command = "{python_bin}"\n'
        f'args = ["-m", "headroom.memory.mcp_server", "--db", "{db_path_toml}", "--user", "{user_id}"]\n'
        f"startup_timeout_sec = 30\n"
        f"tool_timeout_sec = 30\n"
        f"{_MEMORY_MCP_END}\n"
    )

    try:
        config_dir.mkdir(parents=True, exist_ok=True)

        if config_file.exists():
            content = config_file.read_text()
            if _MEMORY_MCP_MARKER in content:
                start = content.index(_MEMORY_MCP_MARKER)
                end = content.index(_MEMORY_MCP_END) + len(_MEMORY_MCP_END)
                content = content[:start].rstrip("\n") + mcp_section + content[end:].lstrip("\n")
            else:
                content = content.rstrip() + "\n" + mcp_section
        else:
            content = mcp_section

        config_file.write_text(content)
        click.echo(f"  Memory MCP: registered in {config_file}")
    except Exception as e:
        click.echo(f"  Warning: could not register memory MCP: {e}")


def _inject_memory_agents_md(file_path: Path) -> bool:
    """Inject memory usage guidance into AGENTS.md.

    Idempotent — skips if marker already present.
    """
    memory_block = (
        f"{_MEMORY_AGENTS_MARKER}\n"
        "## Memory\n\n"
        "Use the `headroom_memory` MCP server for persistent cross-session knowledge.\n\n"
        "**Before** answering questions about prior decisions, conventions, project context,\n"
        "architecture, user preferences, org info, codenames, debugging history, or anything\n"
        "from past sessions — call `memory_search` first.\n\n"
        "**After** making durable decisions, discovering conventions, or learning important\n"
        "facts — call `memory_save` to persist them for future sessions.\n\n"
        "Memory is your first source of truth for anything not visible in the current conversation.\n"
    )

    if file_path.exists():
        existing = file_path.read_text()
        if _MEMORY_AGENTS_MARKER in existing:
            return True  # Already injected
        with open(file_path, "a") as f:
            f.write("\n\n" + memory_block)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(memory_block)

    click.echo(f"  Memory guidance injected into {file_path.name}")
    return True


def _resolve_copilot_provider_type(backend: str | None, provider_type: str) -> str:
    """Resolve Copilot BYOK provider type for the current proxy backend."""
    if provider_type != "auto":
        return provider_type

    effective_backend = backend or os.environ.get("HEADROOM_BACKEND") or "anthropic"
    return "anthropic" if effective_backend == "anthropic" else "openai"


def _query_proxy_config(port: int) -> dict[str, Any] | None:
    """Query the running proxy's feature configuration via /health.

    Returns a dict with keys like backend, optimize, cache, rate_limit,
    memory, learn, code_graph, pid.  Returns None if unreachable or the
    response lacks a config block.
    """
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return None

    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    return config


def _detect_running_proxy_backend(port: int) -> str | None:
    """Read the backend of an already-running proxy from its health endpoint."""
    config = _query_proxy_config(port)
    if config is None:
        return None
    backend = config.get("backend")
    return backend if isinstance(backend, str) else None


def _kill_proxy_by_pid(pid: int, port: int) -> bool:
    """Terminate a proxy process by PID and wait for the port to free up.

    Sends SIGTERM first, falls back to SIGKILL after 5 seconds.
    Returns True if the port is free afterwards, False otherwise.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # Already gone
    except PermissionError:
        click.echo(f"  Warning: No permission to kill proxy PID {pid}")
        return False

    # Wait for port to free (up to 5 seconds)
    for _ in range(50):
        time.sleep(0.1)
        if not _check_proxy(port):
            return True

    # SIGTERM didn't work — escalate to SIGKILL (Unix) or terminate (Windows)
    try:
        _kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        os.kill(pid, _kill_signal)
    except (ProcessLookupError, PermissionError):
        pass

    for _ in range(20):
        time.sleep(0.1)
        if not _check_proxy(port):
            return True

    return False


def _find_persistent_manifest(port: int) -> Any:
    """Return a matching persistent deployment manifest for the requested port."""
    from headroom.install.state import list_manifests

    manifests = [manifest for manifest in list_manifests() if manifest.port == port]
    manifests.sort(key=lambda manifest: (manifest.profile != "default", manifest.profile))
    return manifests[0] if manifests else None


def _recover_persistent_proxy(port: int) -> bool:
    """Start or recover a matching persistent deployment for the requested port."""
    from headroom.install.health import probe_ready
    from headroom.install.models import InstallPreset, SupervisorKind
    from headroom.install.runtime import start_detached_agent, start_persistent_docker, wait_ready
    from headroom.install.supervisors import start_supervisor

    manifest = _find_persistent_manifest(port)
    if manifest is None:
        return False

    if probe_ready(manifest.health_url):
        click.echo(f"  Reusing persistent deployment '{manifest.profile}' on port {port}")
        return True

    if manifest.supervisor_kind == SupervisorKind.TASK.value:
        click.echo(
            f"  Warning: task-based deployment '{manifest.profile}' cannot be auto-recovered via wrap"
        )
        return False

    click.echo(f"  Recovering persistent deployment '{manifest.profile}' on port {port}...")
    try:
        if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value:
            start_persistent_docker(manifest)
        elif manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            start_supervisor(manifest)
        else:
            start_detached_agent(manifest.profile)
    except Exception as exc:
        click.echo(
            f"  Warning: could not recover persistent deployment '{manifest.profile}': {exc}"
        )
        return False

    if wait_ready(manifest, timeout_seconds=45):
        click.echo(f"  Recovered persistent deployment '{manifest.profile}' on port {port}")
        return True

    click.echo(f"  Warning: persistent deployment '{manifest.profile}' did not become ready")
    return False


def _copilot_model_configured(copilot_args: tuple[str, ...], env: dict[str, str]) -> bool:
    """Return True when Copilot BYOK model selection is configured."""
    if env.get("COPILOT_MODEL") or env.get("COPILOT_PROVIDER_MODEL_ID"):
        return True

    for idx, arg in enumerate(copilot_args):
        if arg == "--model" and idx + 1 < len(copilot_args):
            return True
        if arg.startswith("--model="):
            return True

    return False


def _ensure_proxy(
    port: int,
    no_proxy: bool,
    *,
    learn: bool = False,
    memory: bool = False,
    agent_type: str = "unknown",
    code_graph: bool = False,
    backend: str | None = None,
    anyllm_provider: str | None = None,
    region: str | None = None,
) -> subprocess.Popen | None:
    """Start or verify proxy. Returns process handle if we started it."""
    if not no_proxy:
        manifest = _find_persistent_manifest(port)
        if manifest is not None:
            from headroom.install.health import probe_ready

            if probe_ready(manifest.health_url):
                click.echo(f"  Proxy already running on port {port}")
                return None
            if _recover_persistent_proxy(port):
                return None
            raise click.ClickException(
                f"Persistent deployment '{manifest.profile}' on port {port} is not healthy."
            )

        if _check_proxy(port):
            # Proxy is running — check if it has the features we need
            needs_restart = False
            running_config = _query_proxy_config(port)

            if running_config is not None:
                missing = []
                if memory and not running_config.get("memory"):
                    missing.append("memory")
                if learn and not running_config.get("learn"):
                    missing.append("learn")
                if code_graph and not running_config.get("code_graph"):
                    missing.append("code_graph")

                if missing:
                    needs_restart = True
                    flags_str = ", ".join(f"--{f.replace('_', '-')}" for f in missing)
                    click.echo(f"  Proxy on port {port} is missing: {flags_str}")
                    click.echo("  Restarting proxy with upgraded configuration...")

                    # Merge: keep features the running proxy already has
                    memory = memory or bool(running_config.get("memory"))
                    learn = learn or bool(running_config.get("learn"))
                    code_graph = code_graph or bool(running_config.get("code_graph"))

                    proxy_pid = running_config.get("pid")
                    if proxy_pid is not None:
                        if not _kill_proxy_by_pid(int(proxy_pid), port):
                            raise click.ClickException(
                                f"Failed to stop existing proxy (PID {proxy_pid}) on port {port}. "
                                "Stop it manually and retry."
                            )
                    else:
                        click.echo(
                            "  Warning: Running proxy does not expose PID. "
                            "Cannot restart automatically."
                        )
                        click.echo(
                            f"  Please stop the proxy on port {port} manually "
                            f"and rerun with {flags_str}."
                        )
                        return None

            if not needs_restart:
                click.echo(f"  Proxy already running on port {port}")
                return None

        # Start (or restart) the proxy with the requested flags
        click.echo(f"  Starting Headroom proxy on port {port}...")
        try:
            proc = _start_proxy(
                port,
                learn=learn,
                memory=memory,
                agent_type=agent_type,
                code_graph=code_graph,
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
    memory: bool = False,
    agent_type: str = "unknown",
    code_graph: bool = False,
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
            memory=memory,
            agent_type=agent_type,
            code_graph=code_graph,
            backend=backend,
            anyllm_provider=anyllm_provider,
            region=region,
        )

        if code_graph:
            _setup_code_graph(verbose=False)

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


def _decode_openclaw_entry_json(raw_value: str | None) -> Any | None:
    """Decode a JSON payload captured from `openclaw config get` when available."""
    if not raw_value:
        return None

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


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


def _build_openclaw_unwrap_entry(existing_entry: Any) -> dict[str, object]:
    """Disable the managed plugin while preserving unrelated user config."""
    base_entry = existing_entry if isinstance(existing_entry, dict) else {}
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

    return {**base_entry, "enabled": False, "config": existing_config}


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
        headroom wrap copilot -- --model claude-sonnet-4-20250514
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
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option(
    "--learn", is_flag=True, help="Enable live traffic learning (patterns saved to MEMORY.md)"
)
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def claude(
    port: int,
    no_rtk: bool,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    prepare_only: bool,
    claude_args: tuple,
) -> None:
    """Launch Claude Code through Headroom proxy.

    \b
    Sets ANTHROPIC_BASE_URL to route all Anthropic API calls through Headroom.
    All unknown flags are passed through to claude (e.g. --resume, --model).

    \b
    Examples:
        headroom wrap claude                    # Start everything
        headroom wrap claude --memory           # With persistent memory
        headroom wrap claude --resume <id>      # Resume a session
        headroom wrap claude -- -p              # Claude in print mode
        headroom wrap claude --code-graph        # With code graph intelligence
        headroom wrap claude --no-rtk           # Skip rtk (proxy only)
    """
    if prepare_only:
        if not no_rtk:
            _prepare_wrap_rtk(verbose=verbose, label="Claude")
        return

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

    # Memory sync BEFORE proxy startup — sync headroom DB ↔ Claude's files
    if memory:
        try:
            import subprocess as _sp

            mem_dir = Path.cwd() / ".headroom"
            mem_dir.mkdir(parents=True, exist_ok=True)
            _sync_db = str(mem_dir / "memory.db")
            _sync_user = os.environ.get("USER", os.environ.get("USERNAME", "default"))

            click.echo(f"  Syncing memory (user={_sync_user})...")
            sync_result = _sp.run(
                [
                    sys.executable,
                    "-m",
                    "headroom.memory.sync",
                    "--db",
                    _sync_db,
                    "--user",
                    _sync_user,
                    "--agent",
                    "claude",
                    "--force",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if sync_result.returncode == 0 and sync_result.stdout.strip():
                import json as _json

                stats = _json.loads(sync_result.stdout.strip().split("\n")[-1])
                imp, exp, ms = stats["imported"], stats["exported"], stats["ms"]
                if imp or exp:
                    click.echo(f"  Memory synced: {imp} imported, {exp} exported ({ms}ms)")
                else:
                    click.echo(f"  Memory: up to date ({ms}ms)")
            elif sync_result.returncode != 0:
                click.echo(f"  Warning: memory sync error: {sync_result.stderr[-200:]}")
        except Exception as e:
            click.echo(f"  Warning: memory sync failed: {e}")

    try:
        click.echo()
        click.echo("  ╔═══════════════════════════════════════════════╗")
        click.echo("  ║            HEADROOM WRAP: CLAUDE              ║")
        click.echo("  ╚═══════════════════════════════════════════════╝")
        click.echo()

        proxy_holder[0] = _ensure_proxy(
            port, no_proxy, learn=learn, memory=memory, agent_type="claude", code_graph=code_graph
        )

        if not no_rtk:
            click.echo("  Setting up rtk...")
            _setup_rtk(verbose=verbose)
        elif verbose:
            click.echo("  Skipping rtk (--no-rtk)")

        if code_graph:
            _setup_code_graph(verbose=verbose)

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
# GitHub Copilot CLI
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option(
    "--no-rtk",
    is_flag=True,
    help="Skip rtk installation and Copilot instructions injection",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
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
@click.option(
    "--provider-type",
    type=click.Choice(["auto", "anthropic", "openai"]),
    default="auto",
    show_default=True,
    help="Copilot BYOK provider mode. 'auto' uses anthropic for the default proxy backend and openai for translated backends.",
)
@click.option(
    "--wire-api",
    type=click.Choice(["completions", "responses"]),
    default=None,
    help="OpenAI-compatible Copilot wire API. Defaults to 'completions' when provider-type resolves to openai.",
)
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.argument("copilot_args", nargs=-1, type=click.UNPROCESSED)
def copilot(
    port: int,
    no_rtk: bool,
    no_proxy: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    provider_type: str,
    wire_api: str | None,
    memory: bool,
    verbose: bool,
    copilot_args: tuple[str, ...],
) -> None:
    """Launch GitHub Copilot CLI through Headroom proxy.

    \b
    Configures Copilot CLI BYOK provider variables so Copilot routes through
    the local Headroom proxy. In auto mode, the wrapper uses Anthropic-style
    routing for the stock proxy backend and OpenAI-compatible routing for
    translated backends such as any-llm and LiteLLM.

    \b
    Examples:
        headroom wrap copilot -- --model claude-sonnet-4-20250514
        headroom wrap copilot --backend anyllm --anyllm-provider groq -- --model gpt-4o
        headroom wrap copilot --provider-type openai --wire-api responses -- --model gpt-5.4
        headroom wrap copilot --no-rtk -- --prompt "explain this file"
    """
    copilot_bin = shutil.which("copilot")
    if not copilot_bin:
        click.echo("Error: 'copilot' not found in PATH.")
        click.echo(
            "Install GitHub Copilot CLI: "
            "https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli"
        )
        raise SystemExit(1)

    effective_backend = backend or os.environ.get("HEADROOM_BACKEND")
    if _check_proxy(port):
        running_backend = _detect_running_proxy_backend(port)
        if effective_backend and running_backend and effective_backend != running_backend:
            raise click.ClickException(
                f"Proxy already running on port {port} with backend '{running_backend}'. "
                f"Stop it or rerun with --backend {running_backend}."
            )
        effective_backend = running_backend or effective_backend

    effective_provider_type = _resolve_copilot_provider_type(effective_backend, provider_type)
    if effective_provider_type == "anthropic" and wire_api is not None:
        raise click.ClickException(
            "--wire-api is only valid when Copilot is using the openai provider type."
        )
    if wire_api == "responses" and effective_backend not in (None, "anthropic"):
        raise click.ClickException(
            "--wire-api responses is not supported with translated backends; use completions."
        )

    if not no_rtk:
        click.echo("  Setting up rtk for Copilot...")
        rtk_path = _ensure_rtk_binary(verbose=verbose)
        if rtk_path:
            copilot_instructions = Path.cwd() / ".github" / "copilot-instructions.md"
            _inject_rtk_instructions(copilot_instructions, verbose=verbose)

    env = os.environ.copy()
    env["COPILOT_PROVIDER_TYPE"] = effective_provider_type
    env.pop("COPILOT_PROVIDER_WIRE_API", None)

    # Copilot BYOK requires COPILOT_PROVIDER_API_KEY — propagate from the
    # user's existing provider key so they don't have to set it twice.
    # Note: `headroom wrap copilot` uses Copilot's BYOK mode, which bypasses
    # GitHub's Copilot API and talks directly to the model provider through
    # the Headroom proxy. This requires the provider's own API key — a GitHub
    # Copilot subscription alone is not sufficient for BYOK mode.
    if not env.get("COPILOT_PROVIDER_API_KEY"):
        if effective_provider_type == "anthropic":
            _key = env.get("ANTHROPIC_API_KEY", "")
        else:
            _key = env.get("OPENAI_API_KEY", "")
        if _key:
            env["COPILOT_PROVIDER_API_KEY"] = _key

    env_vars_display: list[str]
    if effective_provider_type == "anthropic":
        env["COPILOT_PROVIDER_BASE_URL"] = f"http://127.0.0.1:{port}"
        env_vars_display = [
            "COPILOT_PROVIDER_TYPE=anthropic",
            f"COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:{port}",
        ]
    else:
        effective_wire_api = wire_api or "completions"
        env["COPILOT_PROVIDER_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["COPILOT_PROVIDER_WIRE_API"] = effective_wire_api
        env_vars_display = [
            "COPILOT_PROVIDER_TYPE=openai",
            f"COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:{port}/v1",
            f"COPILOT_PROVIDER_WIRE_API={effective_wire_api}",
        ]

    if not env.get("COPILOT_PROVIDER_API_KEY"):
        src = "ANTHROPIC_API_KEY" if effective_provider_type == "anthropic" else "OPENAI_API_KEY"
        click.echo(
            f"\n  Error: Copilot BYOK mode requires a provider API key.\n"
            f"  `headroom wrap copilot` uses Copilot's BYOK mode, which bypasses GitHub's\n"
            f"  Copilot API and routes requests directly to the model provider through the\n"
            f"  Headroom proxy. A GitHub Copilot subscription alone is not sufficient.\n\n"
            f"  Set one of:\n"
            f"    export {src}=sk-...          # recommended\n"
            f"    export COPILOT_PROVIDER_API_KEY=sk-...  # also works\n"
        )
        raise SystemExit(1)

    if not _copilot_model_configured(copilot_args, env):
        click.echo(
            "  Note: Copilot BYOK requires a model. Pass `--model <name>` "
            "or set `COPILOT_MODEL` / `COPILOT_PROVIDER_MODEL_ID`."
        )

    _launch_tool(
        binary=copilot_bin,
        args=copilot_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="COPILOT",
        env_vars_display=env_vars_display,
        learn=False,
        memory=memory,
        agent_type="copilot",
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
    )


# =============================================================================
# OpenAI Codex CLI
# =============================================================================


@wrap.command(context_settings={"ignore_unknown_options": True})
@click.option("--port", "-p", default=8787, type=int, help="Proxy port (default: 8787)")
@click.option("--no-rtk", is_flag=True, help="Skip rtk installation and AGENTS.md injection")
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
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
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("codex_args", nargs=-1, type=click.UNPROCESSED)
def codex(
    port: int,
    no_rtk: bool,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
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

    # Setup memory MCP server for Codex (native tool integration)
    if memory:
        click.echo("  Setting up memory for Codex...")
        mem_dir = Path.cwd() / ".headroom"
        mem_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(mem_dir / "memory.db")
        mem_user = os.environ.get("USER", os.environ.get("USERNAME", "default"))

        # Register MCP server in Codex config
        _inject_memory_mcp_config(db_path, mem_user)

        # Inject memory guidance into project AGENTS.md
        agents_md = Path.cwd() / "AGENTS.md"
        _inject_memory_agents_md(agents_md)

        # Sync Claude's memories → DB so MCP search finds them
        try:
            import asyncio

            from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
            from headroom.memory.sync import sync_import
            from headroom.memory.sync_adapters.claude_code import (
                ClaudeCodeAdapter,
                get_claude_memory_dir,
            )

            claude_memory_dir = get_claude_memory_dir()

            async def _import_claude_memories() -> int:
                config = LocalBackendConfig(db_path=db_path)
                backend = LocalBackend(config)
                await backend._ensure_initialized()
                adapter = ClaudeCodeAdapter(claude_memory_dir)
                count = await sync_import(backend, adapter, mem_user)
                await backend.close()
                return count

            imported = asyncio.run(_import_claude_memories())
            if imported:
                click.echo(f"  Memory: imported {imported} memories from Claude")
        except Exception as e:
            click.echo(f"  Warning: Claude memory import failed: {e}")

    if prepare_only:
        _inject_codex_provider_config(port)
        return

    codex_bin = shutil.which("codex")
    if not codex_bin:
        click.echo("Error: 'codex' not found in PATH.")
        click.echo("Install Codex CLI: npm install -g @openai/codex")
        raise SystemExit(1)

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"

    # Inject Headroom provider into Codex config so WebSocket traffic also
    # routes through the proxy.  Codex ignores OPENAI_BASE_URL for its WS
    # transport unless a custom provider declares supports_websockets = true.
    # NOTE: this must run BEFORE _inject_memory_mcp_config because it rewrites
    # the config file.  Re-inject MCP config after if memory is enabled.
    _inject_codex_provider_config(port)
    if memory:
        mem_dir = Path.cwd() / ".headroom"
        _inject_memory_mcp_config(
            str(mem_dir / "memory.db"),
            os.environ.get("USER", os.environ.get("USERNAME", "default")),
        )

    _launch_tool(
        binary=codex_bin,
        args=codex_args,
        env=env,
        port=port,
        no_proxy=no_proxy,
        tool_label="CODEX",
        env_vars_display=[f"OPENAI_BASE_URL=http://127.0.0.1:{port}/v1"],
        learn=learn,
        memory=memory,
        agent_type="codex",
        code_graph=code_graph,
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
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph indexing via codebase-memory-mcp (optional)",
)
@click.option("--no-proxy", is_flag=True, help="Skip proxy startup (use existing proxy)")
@click.option("--learn", is_flag=True, help="Enable live traffic learning")
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option(
    "--backend", default=None, help="API backend: 'anthropic', 'anyllm', 'litellm-vertex', etc."
)
@click.option("--anyllm-provider", default=None, help="Provider for any-llm backend")
@click.option("--region", default=None, help="Cloud region for Bedrock/Vertex")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.argument("aider_args", nargs=-1, type=click.UNPROCESSED)
def aider(
    port: int,
    no_rtk: bool,
    code_graph: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    backend: str | None,
    anyllm_provider: str | None,
    region: str | None,
    verbose: bool,
    prepare_only: bool,
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
    # Setup rtk for aider (binary + CONVENTIONS.md instructions)
    if not no_rtk:
        click.echo("  Setting up rtk for aider...")
        rtk_path = _ensure_rtk_binary(verbose=verbose)
        if rtk_path:
            # aider reads CONVENTIONS.md from project root
            conventions = Path.cwd() / "CONVENTIONS.md"
            _inject_rtk_instructions(conventions, verbose=verbose)

    if prepare_only:
        return

    aider_bin = shutil.which("aider")
    if not aider_bin:
        click.echo("Error: 'aider' not found in PATH.")
        click.echo("Install aider: pip install aider-chat")
        raise SystemExit(1)

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
        memory=memory,
        agent_type="aider",
        code_graph=code_graph,
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
@click.option("--memory", is_flag=True, help="Enable persistent cross-session memory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--prepare-only", is_flag=True, hidden=True)
def cursor(
    port: int,
    no_rtk: bool,
    no_proxy: bool,
    learn: bool,
    memory: bool,
    verbose: bool,
    prepare_only: bool,
) -> None:
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
    if not no_rtk:
        click.echo("  Setting up rtk for Cursor...")
        rtk_path = _ensure_rtk_binary(verbose=verbose)
        if rtk_path:
            cursorrules = Path.cwd() / ".cursorrules"
            _inject_rtk_instructions(cursorrules, verbose=verbose)

    if prepare_only:
        return

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

        proxy_holder[0] = _ensure_proxy(
            port, no_proxy, learn=learn, memory=memory, agent_type="cursor"
        )

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
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.option("--existing-entry-json", default=None, hidden=True)
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
    prepare_only: bool,
    existing_entry_json: str | None,
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
    if prepare_only:
        entry = _build_openclaw_plugin_entry(
            existing_entry=_decode_openclaw_entry_json(existing_entry_json),
            proxy_port=proxy_port,
            startup_timeout_ms=startup_timeout_ms,
            python_path=python_path,
            no_auto_start=no_auto_start,
            gateway_provider_ids=gateway_provider_ids,
            enabled=True,
        )
        click.echo(json.dumps(entry, separators=(",", ":")))
        return

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
@click.option("--prepare-only", is_flag=True, hidden=True)
@click.option("--existing-entry-json", default=None, hidden=True)
def unwrap_openclaw(
    no_restart: bool,
    verbose: bool,
    prepare_only: bool,
    existing_entry_json: str | None,
) -> None:
    """Disable the Headroom OpenClaw plugin and restore the legacy engine slot."""
    if prepare_only:
        click.echo(
            json.dumps(
                _build_openclaw_unwrap_entry(_decode_openclaw_entry_json(existing_entry_json)),
                separators=(",", ":"),
            )
        )
        return

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
    entry = _build_openclaw_unwrap_entry(existing_entry)
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
