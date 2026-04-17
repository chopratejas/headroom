"""Proxy server CLI commands."""

import os

import click

from headroom.proxy.modes import PROXY_MODE_TOKEN, normalize_proxy_mode

from .main import main


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    envvar="HEADROOM_HOST",
    help="Host to bind to (default: 127.0.0.1, env: HEADROOM_HOST)",
)
@click.option(
    "--port",
    "-p",
    default=8787,
    type=int,
    envvar="HEADROOM_PORT",
    help="Port to bind to (default: 8787, env: HEADROOM_PORT)",
)
@click.option(
    "--mode",
    default=None,
    type=click.Choice(
        [
            "token",
            "cache",
            "token_mode",
            "cache_mode",
            "token_savings",
            "cost_savings",
            "token_headroom",
        ]
    ),
    help=(
        "Optimization mode: token (prioritize compression) or cache "
        "(freeze prior turns for prefix-cache stability). "
        "Legacy aliases are accepted. Default: token. Env: HEADROOM_MODE"
    ),
)
@click.option("--no-optimize", is_flag=True, help="Disable optimization (passthrough mode)")
@click.option("--no-cache", is_flag=True, help="Disable semantic caching")
@click.option("--no-rate-limit", is_flag=True, help="Disable rate limiting")
@click.option(
    "--retry-max-attempts",
    type=int,
    default=None,
    help="Maximum upstream retry attempts for connect/read/5xx failures (default: 3)",
)
@click.option(
    "--connect-timeout-seconds",
    type=int,
    default=None,
    help="Upstream connection timeout in seconds (default: 10)",
)
@click.option(
    "--anthropic-pre-upstream-concurrency",
    type=int,
    default=None,
    envvar="HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY",
    help=(
        "Cap the number of Anthropic HTTP requests that may run pre-upstream work "
        "(request parse / deep-copy / first compression stage / memory context / upstream connect) "
        "concurrently. Prevents cold-start replay storms from starving /livez and new Codex WS opens. "
        "Default: max(2, min(8, os.cpu_count() or 4)). "
        "Set to 0 or negative to disable (unbounded). "
        "Env: HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY."
    ),
)
@click.option("--log-file", default=None, help="Path to JSONL log file")
@click.option(
    "--budget",
    type=float,
    default=None,
    envvar="HEADROOM_BUDGET",
    help="Daily budget limit in USD (env: HEADROOM_BUDGET)",
)
# Code graph: indexes project + watches files for live reindex via codebase-memory-mcp
@click.option(
    "--code-graph",
    is_flag=True,
    help="Enable code graph intelligence (indexes project, watches files for live reindex via codebase-memory-mcp)",
)
# Read lifecycle (ON by default: compresses stale/superseded Read outputs)
@click.option(
    "--no-read-lifecycle",
    is_flag=True,
    help="Disable Read lifecycle management (stale/superseded Read compression)",
)
# Intelligent Context Management (ON by default)
@click.option(
    "--no-intelligent-context",
    is_flag=True,
    help="Disable IntelligentContextManager (fall back to RollingWindow)",
)
@click.option(
    "--no-intelligent-scoring",
    is_flag=True,
    help="Disable multi-factor importance scoring (use position-based)",
)
@click.option(
    "--no-compress-first",
    is_flag=True,
    help="Disable trying deeper compression before dropping messages",
)
# Memory System (Multi-Provider Support)
@click.option(
    "--memory",
    is_flag=True,
    help="Enable persistent user memory. Auto-detects provider and uses appropriate tool format. "
    "Set x-headroom-user-id header for per-user memory (defaults to 'default').",
)
@click.option(
    "--memory-db-path",
    default="",
    help="Path to memory database file (default: {cwd}/.headroom/memory.db)",
)
@click.option("--no-memory-tools", is_flag=True, help="Disable automatic memory tool injection")
@click.option(
    "--no-memory-context", is_flag=True, help="Disable automatic memory context injection"
)
@click.option(
    "--memory-top-k",
    type=int,
    default=10,
    help="Number of memories to inject as context (default: 10)",
)
# Traffic Learning (live pattern extraction from proxy traffic)
@click.option(
    "--learn",
    is_flag=True,
    help="Enable live traffic learning: extract error→recovery patterns, environment facts, "
    "and user preferences from proxy traffic. Implies --memory. "
    "Learned patterns are saved to agent-native memory files (MEMORY.md, .cursor/rules, AGENTS.md).",
)
@click.option(
    "--no-learn",
    is_flag=True,
    help="Explicitly disable traffic learning even when --memory is set.",
)
# Backend configuration
@click.option(
    "--backend",
    default="anthropic",
    help=(
        "API backend: 'anthropic' (direct), 'bedrock' (AWS), 'openrouter' (OpenRouter), "
        "'anyllm' (any-llm), or 'litellm-<provider>' (e.g., litellm-vertex)"
    ),
)
@click.option(
    "--anyllm-provider",
    default="openai",
    help="Provider for any-llm backend: openai, mistral, groq, ollama, etc. (default: openai)",
)
@click.option(
    "--anthropic-api-url",
    default=None,
    help="Custom Anthropic API URL for passthrough endpoints (env: ANTHROPIC_TARGET_API_URL)",
)
@click.option(
    "--openai-api-url",
    default=None,
    help="Custom OpenAI API URL for passthrough endpoints (env: OPENAI_TARGET_API_URL)",
)
@click.option(
    "--gemini-api-url",
    default=None,
    help="Custom Gemini API URL for passthrough endpoints (env: GEMINI_TARGET_API_URL)",
)
@click.option(
    "--region",
    default="us-west-2",
    help="Cloud region for Bedrock/Vertex/etc (default: us-west-2)",
)
@click.option(
    "--bedrock-region",
    default=None,
    help="(deprecated, use --region) AWS region for Bedrock",
)
@click.option(
    "--bedrock-profile",
    default=None,
    help="AWS profile name for Bedrock (default: use default credentials)",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    help="Disable anonymous usage telemetry (env: HEADROOM_TELEMETRY=off)",
)
@click.option(
    "--stateless",
    is_flag=True,
    help="Disable all filesystem writes — run purely in-memory. "
    "For containerized / read-only / load-balanced deployments. "
    "(env: HEADROOM_STATELESS=true)",
)
@click.pass_context
def proxy(
    ctx: click.Context,
    mode: str | None,
    host: str,
    port: int,
    no_optimize: bool,
    no_cache: bool,
    no_rate_limit: bool,
    retry_max_attempts: int | None,
    connect_timeout_seconds: int | None,
    anthropic_pre_upstream_concurrency: int | None,
    log_file: str | None,
    budget: float | None,
    code_graph: bool,
    no_read_lifecycle: bool,
    no_intelligent_context: bool,
    no_intelligent_scoring: bool,
    no_compress_first: bool,
    memory: bool,
    memory_db_path: str,
    no_memory_tools: bool,
    no_memory_context: bool,
    memory_top_k: int,
    learn: bool,
    no_learn: bool,
    backend: str,
    anyllm_provider: str,
    anthropic_api_url: str | None,
    openai_api_url: str | None,
    gemini_api_url: str | None,
    region: str,
    bedrock_region: str | None,
    bedrock_profile: str | None,
    no_telemetry: bool,
    stateless: bool,
) -> None:
    """Start the optimization proxy server.

    \b
    Examples:
        headroom proxy                    Start proxy on port 8787
        headroom proxy --port 8080        Start proxy on port 8080
        headroom proxy --no-optimize      Passthrough mode (no optimization)

    \b
    Usage with Claude Code:
        ANTHROPIC_BASE_URL=http://localhost:8787 claude

    \b
    Usage with OpenAI-compatible clients:
        OPENAI_BASE_URL=http://localhost:8787/v1 your-app
    """
    # Import here to avoid slow startup
    try:
        from headroom.proxy.server import ProxyConfig, run_server
    except ImportError as e:
        click.echo("Error: Proxy dependencies not installed. Run: pip install headroom[proxy]")
        click.echo(f"Details: {e}")
        raise SystemExit(1) from None

    # Resolve API URL overrides: CLI flag > env var > None
    effective_anthropic_api_url = anthropic_api_url or os.environ.get("ANTHROPIC_TARGET_API_URL")
    effective_openai_api_url = openai_api_url or os.environ.get("OPENAI_TARGET_API_URL")
    effective_gemini_api_url = gemini_api_url or os.environ.get("GEMINI_TARGET_API_URL")

    # Resolve anyllm provider: env var takes precedence over CLI default (matches argparse path)
    effective_anyllm_provider = os.environ.get("HEADROOM_ANYLLM_PROVIDER") or anyllm_provider

    # Resolve mode: CLI flag > env var > default
    effective_mode: str = normalize_proxy_mode(
        mode or os.environ.get("HEADROOM_MODE") or PROXY_MODE_TOKEN
    )

    # Stateless mode: CLI flag or env var
    is_stateless = stateless or os.environ.get("HEADROOM_STATELESS", "").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    # Telemetry opt-out: --no-telemetry flag sets the env var
    if no_telemetry:
        os.environ["HEADROOM_TELEMETRY"] = "off"

    # Stateless mode: suppress TOIN filesystem persistence
    if is_stateless:
        os.environ["HEADROOM_TOIN_BACKEND"] = "none"

    # License key for managed/enterprise deployments (optional)
    license_key = os.environ.get("HEADROOM_LICENSE_KEY")

    config = ProxyConfig(
        host=host,
        port=port,
        anthropic_api_url=effective_anthropic_api_url,
        openai_api_url=effective_openai_api_url,
        gemini_api_url=effective_gemini_api_url,
        mode=effective_mode,
        optimize=not no_optimize,
        cache_enabled=not no_cache,
        rate_limit_enabled=not no_rate_limit,
        retry_max_attempts=retry_max_attempts if retry_max_attempts is not None else 3,
        connect_timeout_seconds=connect_timeout_seconds
        if connect_timeout_seconds is not None
        else 10,
        log_file=None if is_stateless else log_file,
        budget_limit_usd=budget,
        # Code graph: live file watcher for incremental reindexing
        code_graph_watcher=code_graph,
        # Read lifecycle: ON by default (use --no-read-lifecycle to disable)
        read_lifecycle=not no_read_lifecycle,
        # Intelligent Context: ON by default (use --no-intelligent-context to disable)
        intelligent_context=not no_intelligent_context,
        intelligent_context_scoring=not no_intelligent_scoring,
        intelligent_context_compress_first=not no_compress_first,
        # Memory System (Multi-Provider with auto-detection)
        # --learn implies --memory (need backend for storing patterns)
        # Stateless mode disables memory (requires SQLite on disk)
        memory_enabled=False if is_stateless else (memory or (learn and not no_learn)),
        memory_db_path=memory_db_path,
        memory_inject_tools=not no_memory_tools,
        memory_inject_context=not no_memory_context,
        memory_top_k=memory_top_k,
        # Traffic Learning: only with --learn, never with --no-learn
        # Stateless mode disables learning (requires filesystem)
        traffic_learning_enabled=False if is_stateless else (learn and not no_learn),
        traffic_learning_agent_type=os.environ.get("HEADROOM_AGENT_TYPE", "unknown"),
        # Backend (Anthropic direct, Bedrock, LiteLLM, or any-llm)
        backend=backend,
        bedrock_region=bedrock_region or region,
        bedrock_profile=bedrock_profile,
        anyllm_provider=effective_anyllm_provider,
        # License / Usage Reporting (managed/enterprise)
        license_key=license_key,
        # Stateless mode: disable all filesystem writes
        stateless=is_stateless,
        # Unit 4: bounded pre-upstream concurrency on the Anthropic HTTP
        # path. ``None`` -> HeadroomProxy computes ``max(2, min(8,
        # os.cpu_count() or 4))``; ``<= 0`` -> disabled (unbounded).
        # Precedence: CLI > env > auto-compute (click's ``envvar``
        # handles the env-var fallback).
        anthropic_pre_upstream_concurrency=anthropic_pre_upstream_concurrency,
    )

    memory_status = "DISABLED"
    if config.memory_enabled:
        memory_status = "ENABLED (multi-provider)"

    license_status = "OSS (no license key)"
    if license_key:
        license_status = f"MANAGED (key={license_key[:8]}...)"

    anthropic_url = config.anthropic_api_url or "https://api.anthropic.com"
    openai_url = config.openai_api_url or "https://api.openai.com"
    backend_section = ""

    if config.backend == "anyllm" or config.backend.startswith("anyllm-"):
        # any-llm backend
        backend_section = """
  Set credentials for your provider (e.g., OPENAI_API_KEY, MISTRAL_API_KEY)
  Providers: https://mozilla-ai.github.io/any-llm/providers/
"""
    elif config.backend != "anthropic":
        # LiteLLM backend
        from headroom.backends.litellm import get_provider_config

        provider = config.backend.replace("litellm-", "")
        provider_config = get_provider_config(provider)

        # Build usage instructions from provider config
        env_vars_str = (
            ", ".join(provider_config.env_vars) if provider_config.env_vars else "See docs"
        )
        backend_section = f"""
IMPORTANT for {provider_config.display_name} users:
  1. Set credentials: {env_vars_str}
  2. Set a dummy Anthropic key: ANTHROPIC_API_KEY="sk-ant-dummy"
     (Headroom ignores this - it uses your {provider_config.display_name} credentials)
  3. Set base URL: ANTHROPIC_BASE_URL=http://{config.host}:{config.port}"""
        if provider_config.model_format_hint:
            backend_section += f"\n  4. Use model names: {provider_config.model_format_hint}"
        backend_section += "\n"

    # Build memory section if enabled
    memory_section = ""
    if config.memory_enabled:
        memory_section = f"""
Memory (Multi-Provider):
  - Auto-detects provider from request (Anthropic, OpenAI, Gemini, etc.)
  - Anthropic: Uses native memory tool (memory_20250818) - subscription safe
  - OpenAI/Gemini/Others: Uses function calling format
  - All providers share the same semantic vector store backend
  - Set x-headroom-user-id header for per-user memory (defaults to 'default')
  - Tools: {"ENABLED" if config.memory_inject_tools else "DISABLED"}
  - Context injection: {"ENABLED" if config.memory_inject_context else "DISABLED"}
  - Database: {config.memory_db_path}
"""

    # Stateless mode warning
    stateless_line = ""
    if is_stateless:
        stateless_line = (
            "  Stateless:    YES (no filesystem writes — memory, logs, TOIN disabled)\n"
        )

    from headroom.telemetry.beacon import is_telemetry_enabled

    # Build telemetry section for the startup banner
    if is_telemetry_enabled():
        telemetry_line = (
            "  Telemetry:    ENABLED (anonymous aggregate stats)\n"
            "                Disable: HEADROOM_TELEMETRY=off or headroom proxy --no-telemetry"
        )
    else:
        telemetry_line = "  Telemetry:    DISABLED"

    click.echo(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                         HEADROOM PROXY                                 ║
║           The Context Optimization Layer for LLM Applications          ║
╚═══════════════════════════════════════════════════════════════════════╝

Starting proxy server...

  URL:          http://{config.host}:{config.port}
  Mode:         {config.mode}
  Optimization: {"ENABLED" if config.optimize else "DISABLED"}
  Caching:      {"ENABLED" if config.cache_enabled else "DISABLED"}
  Rate Limit:   {"ENABLED" if config.rate_limit_enabled else "DISABLED"}
  Memory:       {memory_status}
  License:      {license_status}
{stateless_line}{telemetry_line}
{backend_section}
Routing:
  /v1/messages         → {anthropic_url}
  /v1/chat/completions → {openai_url}
  /v1/responses        → {openai_url}  (HTTP + WebSocket)

Usage:
  Claude Code:   ANTHROPIC_BASE_URL=http://{config.host}:{config.port} claude
  Codex / OpenAI: OPENAI_BASE_URL=http://{config.host}:{config.port}/v1 your-app
{memory_section}
Endpoints:
  GET  /livez      Process liveness
  GET  /readyz     Traffic readiness
  GET  /health     Aggregate health
  GET  /stats      Detailed statistics
  GET  /stats-history Durable compression history + display session
  GET  /metrics    Prometheus metrics

Press Ctrl+C to stop.
""")

    try:
        run_server(config)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
