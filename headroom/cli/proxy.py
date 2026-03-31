"""Proxy server CLI commands."""

import os

import click

from .main import main


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
@click.option("--port", "-p", default=8787, type=int, help="Port to bind to (default: 8787)")
@click.option(
    "--mode",
    default=None,
    type=click.Choice(["cost_savings", "token_headroom"]),
    help="Optimization mode: token_headroom (compress for session extension) or cost_savings (preserve prefix cache). Default: token_headroom. Env: HEADROOM_MODE",
)
@click.option("--no-optimize", is_flag=True, help="Disable optimization (passthrough mode)")
@click.option("--no-cache", is_flag=True, help="Disable semantic caching")
@click.option("--no-rate-limit", is_flag=True, help="Disable rate limiting")
@click.option("--log-file", default=None, help="Path to JSONL log file")
@click.option("--budget", type=float, default=None, help="Daily budget limit in USD")
# Code-aware compression (ON by default if installed)
@click.option("--no-code-aware", is_flag=True, help="Disable AST-based code compression")
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
    default="headroom_memory.db",
    help="Path to memory database file (default: headroom_memory.db)",
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
@click.pass_context
def proxy(
    ctx: click.Context,
    mode: str | None,
    host: str,
    port: int,
    no_optimize: bool,
    no_cache: bool,
    no_rate_limit: bool,
    log_file: str | None,
    budget: float | None,
    no_code_aware: bool,
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
    effective_mode: str = mode or os.environ.get("HEADROOM_MODE") or "token_headroom"

    # Telemetry opt-out: --no-telemetry flag sets the env var
    if no_telemetry:
        os.environ["HEADROOM_TELEMETRY"] = "off"

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
        log_file=log_file,
        budget_limit_usd=budget,
        # Code-aware: ON by default (use --no-code-aware to disable)
        code_aware_enabled=not no_code_aware,
        # Read lifecycle: ON by default (use --no-read-lifecycle to disable)
        read_lifecycle=not no_read_lifecycle,
        # Intelligent Context: ON by default (use --no-intelligent-context to disable)
        intelligent_context=not no_intelligent_context,
        intelligent_context_scoring=not no_intelligent_scoring,
        intelligent_context_compress_first=not no_compress_first,
        # Memory System (Multi-Provider with auto-detection)
        # --learn implies --memory (need backend for storing patterns)
        memory_enabled=memory or (learn and not no_learn),
        memory_db_path=memory_db_path,
        memory_inject_tools=not no_memory_tools,
        memory_inject_context=not no_memory_context,
        memory_top_k=memory_top_k,
        # Traffic Learning: only with --learn, never with --no-learn
        traffic_learning_enabled=learn and not no_learn,
        # Backend (Anthropic direct, Bedrock, LiteLLM, or any-llm)
        backend=backend,
        bedrock_region=bedrock_region or region,
        bedrock_profile=bedrock_profile,
        anyllm_provider=effective_anyllm_provider,
        # License / Usage Reporting (managed/enterprise)
        license_key=license_key,
    )

    memory_status = "DISABLED"
    if config.memory_enabled:
        memory_status = "ENABLED (multi-provider)"

    license_status = "OSS (no license key)"
    if license_key:
        license_status = f"MANAGED (key={license_key[:8]}...)"

    effective_region = bedrock_region or region
    backend_status = "Anthropic (direct API)"
    backend_section = ""

    if config.backend == "anyllm" or config.backend.startswith("anyllm-"):
        # any-llm backend
        backend_status = f"{effective_anyllm_provider.title()} via any-llm"
        backend_section = """
  Set credentials for your provider (e.g., OPENAI_API_KEY, MISTRAL_API_KEY)
  Providers: https://mozilla-ai.github.io/any-llm/providers/
"""
    elif config.backend != "anthropic":
        # LiteLLM backend
        from headroom.backends.litellm import get_provider_config

        provider = config.backend.replace("litellm-", "")
        provider_config = get_provider_config(provider)

        # Build backend status
        if provider_config.uses_region:
            backend_status = (
                f"{provider_config.display_name} via LiteLLM (region={effective_region})"
            )
        else:
            backend_status = f"{provider_config.display_name} via LiteLLM"

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

    click.echo(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                         HEADROOM PROXY                                 ║
║           The Context Optimization Layer for LLM Applications          ║
╚═══════════════════════════════════════════════════════════════════════╝

Starting proxy server...

  URL:          http://{config.host}:{config.port}
  Backend:      {backend_status}
  Mode:         {config.mode}
  Optimization: {"ENABLED" if config.optimize else "DISABLED"}
  Caching:      {"ENABLED" if config.cache_enabled else "DISABLED"}
  Rate Limit:   {"ENABLED" if config.rate_limit_enabled else "DISABLED"}
  Memory:       {memory_status}
  License:      {license_status}
{backend_section}
Usage with Claude Code:
  ANTHROPIC_BASE_URL=http://{config.host}:{config.port} claude

Usage with OpenAI-compatible clients:
  OPENAI_BASE_URL=http://{config.host}:{config.port}/v1 your-app
{memory_section}
Endpoints:
  GET  /health     Health check
  GET  /stats      Detailed statistics
  GET  /metrics    Prometheus metrics
  POST /v1/messages           Anthropic API
  POST /v1/chat/completions   OpenAI API

Press Ctrl+C to stop.
""")

    try:
        run_server(config)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
