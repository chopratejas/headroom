"""Proxy server CLI commands."""

import click

from .main import main


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
@click.option("--port", "-p", default=8787, type=int, help="Port to bind to (default: 8787)")
@click.option("--no-optimize", is_flag=True, help="Disable optimization (passthrough mode)")
@click.option("--no-cache", is_flag=True, help="Disable semantic caching")
@click.option("--no-rate-limit", is_flag=True, help="Disable rate limiting")
@click.option("--log-file", default=None, help="Path to JSONL log file")
@click.option("--budget", type=float, default=None, help="Daily budget limit in USD")
# LLMLingua ML-based compression (ON by default if installed)
@click.option("--no-llmlingua", is_flag=True, help="Disable LLMLingua-2 ML-based compression")
@click.option(
    "--llmlingua-device",
    type=click.Choice(["auto", "cuda", "cpu", "mps"]),
    default="auto",
    help="Device for LLMLingua model (default: auto)",
)
@click.option(
    "--llmlingua-rate",
    type=float,
    default=0.3,
    help="LLMLingua compression rate 0.0-1.0 (default: 0.3 = keep 30%)",
)
# Code-aware compression (ON by default if installed)
@click.option("--no-code-aware", is_flag=True, help="Disable AST-based code compression")
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
# Memory System
@click.option(
    "--memory",
    is_flag=True,
    help="Enable persistent user memory (uses x-headroom-user-id header if set, otherwise 'default')",
)
@click.option(
    "--memory-backend",
    type=click.Choice(["local", "qdrant-neo4j"]),
    default="local",
    help="Memory storage backend: local (SQLite+HNSW) or qdrant-neo4j (default: local)",
)
@click.option(
    "--memory-db-path",
    default="headroom_memory.db",
    help="Path to memory database file for local backend (default: headroom_memory.db)",
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
# Backend configuration
@click.option(
    "--backend",
    default="anthropic",
    help=(
        "API backend: 'anthropic' (direct), 'bedrock' (AWS), "
        "or 'litellm-<provider>' (e.g., litellm-bedrock, litellm-vertex)"
    ),
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
@click.pass_context
def proxy(
    ctx: click.Context,
    host: str,
    port: int,
    no_optimize: bool,
    no_cache: bool,
    no_rate_limit: bool,
    log_file: str | None,
    budget: float | None,
    no_llmlingua: bool,
    llmlingua_device: str,
    llmlingua_rate: float,
    no_code_aware: bool,
    no_intelligent_context: bool,
    no_intelligent_scoring: bool,
    no_compress_first: bool,
    memory: bool,
    memory_backend: str,
    memory_db_path: str,
    no_memory_tools: bool,
    no_memory_context: bool,
    memory_top_k: int,
    backend: str,
    region: str,
    bedrock_region: str | None,
    bedrock_profile: str | None,
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

    config = ProxyConfig(
        host=host,
        port=port,
        optimize=not no_optimize,
        cache_enabled=not no_cache,
        rate_limit_enabled=not no_rate_limit,
        log_file=log_file,
        budget_limit_usd=budget,
        # LLMLingua: ON by default (use --no-llmlingua to disable)
        llmlingua_enabled=not no_llmlingua,
        llmlingua_device=llmlingua_device,
        llmlingua_target_rate=llmlingua_rate,
        # Code-aware: ON by default (use --no-code-aware to disable)
        code_aware_enabled=not no_code_aware,
        # Intelligent Context: ON by default (use --no-intelligent-context to disable)
        intelligent_context=not no_intelligent_context,
        intelligent_context_scoring=not no_intelligent_scoring,
        intelligent_context_compress_first=not no_compress_first,
        # Memory System
        memory_enabled=memory,
        memory_backend=memory_backend,  # type: ignore[arg-type]
        memory_db_path=memory_db_path,
        memory_inject_tools=not no_memory_tools,
        memory_inject_context=not no_memory_context,
        memory_top_k=memory_top_k,
        # Backend (Anthropic direct, Bedrock, or LiteLLM)
        backend=backend,
        bedrock_region=bedrock_region or region,
        bedrock_profile=bedrock_profile,
    )

    memory_status = "DISABLED"
    if config.memory_enabled:
        memory_status = f"ENABLED ({config.memory_backend})"

    effective_region = bedrock_region or region
    backend_status = "Anthropic (direct API)"
    if config.backend != "anthropic":
        # Normalize: "bedrock" -> "litellm-bedrock"
        backend_name = config.backend
        if not backend_name.startswith("litellm-"):
            backend_name = f"litellm-{backend_name}"
        provider = backend_name.replace("litellm-", "")
        backend_status = f"{provider.upper()} via LiteLLM (region={effective_region})"

    # Build memory section if enabled
    memory_section = ""
    if config.memory_enabled:
        memory_section = f"""
Memory:
  - Memories are scoped per user. Set x-headroom-user-id header (defaults to 'default').
  - Tools: {"ENABLED" if config.memory_inject_tools else "DISABLED"}  Context: {"ENABLED" if config.memory_inject_context else "DISABLED"}
"""
        if config.memory_inject_tools:
            memory_section += "  - NOTE: Memory tools require ANTHROPIC_API_KEY (Claude Code subscription credentials have restrictions).\n"

    # Build usage note for Bedrock
    bedrock_section = ""
    if config.backend != "anthropic":
        provider = config.backend.replace("litellm-", "").upper()
        bedrock_section = f"""
IMPORTANT for {provider} users (Claude Code / VS Code):
  1. Set AWS credentials: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
  2. Set a dummy Anthropic key: ANTHROPIC_API_KEY="sk-ant-dummy"
     (Headroom ignores this - it uses your AWS credentials for {provider})
  3. Set base URL: ANTHROPIC_BASE_URL=http://{config.host}:{config.port}
  4. Do NOT set: CLAUDE_CODE_USE_BEDROCK=1 (Headroom handles this)
"""

    click.echo(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                         HEADROOM PROXY                                 ║
║           The Context Optimization Layer for LLM Applications          ║
╚═══════════════════════════════════════════════════════════════════════╝

Starting proxy server...

  URL:          http://{config.host}:{config.port}
  Backend:      {backend_status}
  Optimization: {"ENABLED" if config.optimize else "DISABLED"}
  Caching:      {"ENABLED" if config.cache_enabled else "DISABLED"}
  Rate Limit:   {"ENABLED" if config.rate_limit_enabled else "DISABLED"}
  Memory:       {memory_status}
{bedrock_section}
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
