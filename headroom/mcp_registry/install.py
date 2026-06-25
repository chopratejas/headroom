"""Top-level orchestration: register Headroom MCP across detected agents."""

from __future__ import annotations

from collections.abc import Iterable

from .agy import AgyRegistrar
from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec
from .claude import ClaudeRegistrar
from .codex import CodexRegistrar
from .opencode import OpencodeRegistrar

#: Default proxy URL used when none is given.
DEFAULT_PROXY_URL = "http://127.0.0.1:8787"


def get_all_registrars() -> list[MCPRegistrar]:
    """Return one instance of every registrar implemented today.

    The list grows as we add adapters for Cursor, Continue, Cline, etc.
    """
    return [ClaudeRegistrar(), CodexRegistrar(), AgyRegistrar(), OpencodeRegistrar()]


def build_headroom_spec(proxy_url: str = DEFAULT_PROXY_URL) -> ServerSpec:
    """Construct the canonical :class:`ServerSpec` for the headroom server.

    The spec is identical across agents — every JSON/TOML registrar
    serializes the same shape into its own format.
    """
    env: dict[str, str] = {}
    if proxy_url and proxy_url != DEFAULT_PROXY_URL:
        env["HEADROOM_PROXY_URL"] = proxy_url
    return ServerSpec(
        name="headroom",
        command="headroom",
        args=("mcp", "serve"),
        env=env,
    )


def build_lean_ctx_spec(lean_ctx_path: str, data_dir: str) -> ServerSpec:
    """Construct the canonical lean-ctx context-tool MCP server spec.

    ``lean-ctx init --agent antigravity-cli`` writes a *bare* entry
    (``command: lean-ctx`` with no ``args``).  Bare ``lean-ctx`` does run as a
    stdio MCP server, but we author the spec explicitly here — with the
    ``mcp`` subcommand and an explicit ``LEAN_CTX_DATA_DIR`` — so the
    registered entry is unambiguous and reproducible rather than dependent on
    lean-ctx's init output.  ``data_dir`` selects the lean-ctx data directory
    (which carries the ``tool_profile``).
    """
    return ServerSpec(
        name="lean-ctx",
        command=lean_ctx_path,
        args=("mcp",),
        env={"LEAN_CTX_DATA_DIR": data_dir},
    )


def build_serena_spec(context: str) -> ServerSpec:
    """Construct the canonical Serena MCP server spec for an agent context.

    ``--open-web-dashboard False`` suppresses Serena's browser popup on
    startup. Headroom installs Serena by default, so without this flag every
    wrapped session opens the Serena dashboard tab even for users who never
    opted into Serena or created a ``~/.serena/serena_config.yml``. The flag
    overrides Serena's own config at startup (it sets
    ``web_dashboard_open_on_launch=False``), so it works regardless of the
    user's local config. The dashboard backend still runs and remains
    reachable at http://localhost:24282/dashboard/ for anyone who wants it —
    only the automatic browser-open is disabled.
    """
    return ServerSpec(
        name="serena",
        command="uvx",
        args=(
            "--from",
            "git+https://github.com/oraios/serena",
            "serena",
            "start-mcp-server",
            "--project-from-cwd",
            "--context",
            context,
            "--open-web-dashboard",
            "False",
        ),
    )


def build_codegraph_spec(cbm_bin: str) -> ServerSpec:
    """Construct the canonical codebase-memory-mcp server spec for agy.

    ``command`` is the resolved cbm binary path; no extra args are needed
    (the binary exposes a stdio MCP server when invoked bare, matching how
    ``_register_cbm_mcp_server`` invokes it via ``claude mcp add <name> -- <bin>``).

    The server name ``"codebase-memory-mcp"`` matches the constant
    ``_CBM_MCP_SERVER_NAME`` in ``headroom.cli.wrap``; kept here as a literal
    to avoid a circular import (wrap.py imports from mcp_registry at call time).
    """
    return ServerSpec(
        name="codebase-memory-mcp",
        command=cbm_bin,
        args=(),
        env={},
    )


def install_everywhere(
    proxy_url: str = DEFAULT_PROXY_URL,
    *,
    agents: Iterable[str] | None = None,
    force: bool = False,
    registrars: Iterable[MCPRegistrar] | None = None,
) -> dict[str, RegisterResult]:
    """Install the headroom MCP server into every detected agent.

    Args:
        proxy_url: URL the MCP server should contact for retrieval.
        agents: If given, only install into agents whose ``name`` matches.
        force: Pass through to each registrar — overwrites mismatched config.
        registrars: Inject a custom registrar list (test seam).

    Returns:
        Dict keyed by registrar name. Includes :attr:`RegisterStatus.NOT_DETECTED`
        entries for agents we know about that aren't installed locally.
    """
    spec = build_headroom_spec(proxy_url)
    selected = list(registrars) if registrars is not None else get_all_registrars()

    if agents is not None:
        agent_set = set(agents)
        selected = [r for r in selected if r.name in agent_set]

    results: dict[str, RegisterResult] = {}
    for registrar in selected:
        if not registrar.detect():
            results[registrar.name] = RegisterResult(
                RegisterStatus.NOT_DETECTED,
                f"{registrar.display_name} not found on this system",
            )
            continue
        results[registrar.name] = registrar.register_server(spec, force=force)

    return results
