#!/usr/bin/env python3
"""Headroom CLI - The Context Optimization Layer for LLM Applications.

Usage:
    headroom proxy [OPTIONS]     Start the optimization proxy server
    headroom --version           Show version
    headroom --help              Show this help message

Examples:
    # Start proxy on default port (8787)
    headroom proxy

    # Start proxy on custom port
    headroom proxy --port 8080

    # Start with optimization disabled (passthrough mode)
    headroom proxy --no-optimize

    # Use with Claude Code
    ANTHROPIC_BASE_URL=http://localhost:8787 claude
"""

from __future__ import annotations

import argparse
import sys


def get_version() -> str:
    """Get the current version."""
    try:
        from headroom import __version__

        return __version__
    except ImportError:
        return "unknown"


def cmd_proxy(args: argparse.Namespace) -> int:
    """Start the proxy server."""
    try:
        from headroom.proxy.server import ProxyConfig, run_server
    except ImportError as e:
        print("Error: Proxy dependencies not installed. Run: pip install headroom[proxy]")
        print(f"Details: {e}")
        return 1

    config = ProxyConfig(
        host=args.host,
        port=args.port,
        optimize=not args.no_optimize,
        cache_enabled=not args.no_cache,
        rate_limit_enabled=not args.no_rate_limit,
        log_file=args.log_file,
        budget_limit_usd=args.budget,
    )

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                         HEADROOM PROXY                                 ║
║           The Context Optimization Layer for LLM Applications          ║
╚═══════════════════════════════════════════════════════════════════════╝

Starting proxy server...

  URL:          http://{config.host}:{config.port}
  Optimization: {"ENABLED" if config.optimize else "DISABLED"}
  Caching:      {"ENABLED" if config.cache_enabled else "DISABLED"}
  Rate Limit:   {"ENABLED" if config.rate_limit_enabled else "DISABLED"}

Usage with Claude Code:
  ANTHROPIC_BASE_URL=http://{config.host}:{config.port} claude

Usage with OpenAI-compatible clients:
  OPENAI_BASE_URL=http://{config.host}:{config.port}/v1 your-app

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
        print("\nShutting down...")
        return 0

    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Print version information."""
    print(f"headroom {get_version()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="headroom",
        description="The Context Optimization Layer for LLM Applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  headroom proxy                    Start proxy on port 8787
  headroom proxy --port 8080        Start proxy on port 8080
  headroom proxy --no-optimize      Passthrough mode (no optimization)

Environment Variables:
  ANTHROPIC_API_KEY    Your Anthropic API key (for proxying)
  OPENAI_API_KEY       Your OpenAI API key (for proxying)

Documentation: https://github.com/headroom-sdk/headroom
        """,
    )

    parser.add_argument(
        "--version",
        "-V",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Proxy command
    proxy_parser = subparsers.add_parser(
        "proxy",
        help="Start the optimization proxy server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    proxy_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    proxy_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8787,
        help="Port to bind to (default: 8787)",
    )
    proxy_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable optimization (passthrough mode)",
    )
    proxy_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable semantic caching",
    )
    proxy_parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="Disable rate limiting",
    )
    proxy_parser.add_argument(
        "--log-file",
        help="Path to JSONL log file",
    )
    proxy_parser.add_argument(
        "--budget",
        type=float,
        help="Daily budget limit in USD",
    )
    proxy_parser.set_defaults(func=cmd_proxy)

    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
