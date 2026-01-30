#!/usr/bin/env python3
"""Headroom CLI - The Context Optimization Layer for LLM Applications.

Usage:
    headroom proxy [OPTIONS]        Start the optimization proxy server
    headroom memory-eval [OPTIONS]  Run LoCoMo memory evaluation
    headroom --version              Show version
    headroom --help                 Show this help message

Examples:
    # Start proxy on default port (8787)
    headroom proxy

    # Start proxy on custom port
    headroom proxy --port 8080

    # Start with optimization disabled (passthrough mode)
    headroom proxy --no-optimize

    # Use with Claude Code
    ANTHROPIC_BASE_URL=http://localhost:8787 claude

    # Run memory evaluation on 3 conversations
    headroom memory-eval -n 3

    # Run memory evaluation with LLM answering and judging
    headroom memory-eval --answer-model gpt-4o --llm-judge
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable


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
        # LLMLingua: ON by default (use --no-llmlingua to disable)
        llmlingua_enabled=not args.no_llmlingua,
        llmlingua_device=args.llmlingua_device,
        llmlingua_target_rate=args.llmlingua_rate,
        # Code-aware: ON by default (use --no-code-aware to disable)
        code_aware_enabled=not args.no_code_aware,
        # Intelligent Context: ON by default (use --no-intelligent-context to disable)
        intelligent_context=not args.no_intelligent_context,
        intelligent_context_scoring=not args.no_intelligent_scoring,
        intelligent_context_compress_first=not args.no_compress_first,
        # Memory System
        memory_enabled=args.memory,
        memory_backend=args.memory_backend,
        memory_db_path=args.memory_db_path,
        memory_inject_tools=not args.no_memory_tools,
        memory_inject_context=not args.no_memory_context,
        memory_top_k=args.memory_top_k,
        # Backend (Anthropic direct, Bedrock, or LiteLLM)
        backend=args.backend,
        bedrock_region=args.bedrock_region or args.region,
        bedrock_profile=args.bedrock_profile,
    )

    memory_status = "DISABLED"
    if config.memory_enabled:
        memory_status = f"ENABLED ({config.memory_backend})"

    region = args.bedrock_region or args.region
    backend_status = "Anthropic (direct API)"
    if config.backend != "anthropic":
        # Normalize: "bedrock" -> "litellm-bedrock"
        backend = config.backend
        if not backend.startswith("litellm-"):
            backend = f"litellm-{backend}"
        provider = backend.replace("litellm-", "")
        backend_status = f"{provider.upper()} via LiteLLM (region={region})"

    print(f"""
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

Usage with Claude Code:
  {"unset CLAUDE_CODE_USE_BEDROCK  # Important!" if config.backend == "bedrock" else ""}
  ANTHROPIC_BASE_URL=http://{config.host}:{config.port} claude

Usage with OpenAI-compatible clients:
  OPENAI_BASE_URL=http://{config.host}:{config.port}/v1 your-app
{
        ""
        if not config.memory_enabled
        else f'''
Memory:
  - Memories are scoped per user. Set x-headroom-user-id header (defaults to 'default').
  - Tools: {"ENABLED" if config.memory_inject_tools else "DISABLED"}  Context: {"ENABLED" if config.memory_inject_context else "DISABLED"}
'''
        + (
            "  - NOTE: Memory tools require ANTHROPIC_API_KEY (Claude Code subscription credentials have restrictions)."
            + chr(10)
            if config.memory_inject_tools
            else ""
        )
    }
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


def cmd_memory_eval(args: argparse.Namespace) -> int:
    """Run LoCoMo memory evaluation."""
    # Suppress noisy pydantic warnings from litellm
    import warnings

    warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    try:
        from headroom.evals.memory import (
            LoCoMoEvaluator,
            MemoryEvalConfig,
            create_anthropic_judge,
            create_litellm_judge,
            create_openai_judge,
            simple_judge,
        )
        from headroom.memory import MemoryConfig
    except ImportError as e:
        print("Error: Memory eval dependencies not installed.")
        print("Run: pip install headroom[memory,evals]")
        print(f"Details: {e}")
        return 1

    import asyncio

    # Build configuration
    categories = None
    if args.categories:
        categories = [int(c) for c in args.categories.split(",")]

    memory_config = MemoryConfig()

    eval_config = MemoryEvalConfig(
        n_conversations=args.n_conversations,
        categories=categories,
        skip_adversarial=not args.include_adversarial,
        top_k_memories=args.top_k,
        llm_judge_enabled=args.llm_judge,
        llm_judge_model=args.judge_model,
        memory_config=memory_config,
        f1_threshold=args.f1_threshold,
        extract_memories=not args.no_extract,
        extraction_model=args.extraction_model,
        pass_all_memories=args.pass_all,
        parallel_workers=args.parallel,
        debug=args.debug,
    )

    # Create answer function based on provider
    answer_fn = None
    if args.answer_model:
        try:
            import litellm

            def answer_fn(question: str, memories: list[str]) -> str:
                if not memories:
                    return "I don't have information about that."

                # Format memories - use all if pass_all, else top 10
                context = "\n".join(f"- {m}" for m in memories)

                prompt = f"""You are answering questions about a conversation between two people based on extracted memories/facts.

## Memories from the conversation:
{context}

## Question: {question}

## Instructions:
1. Find the specific fact(s) in the memories that answer this question
2. Answer with JUST the key information requested - be concise
3. For "when" questions: give the specific date if mentioned (e.g., "7 May 2023", "2022")
4. For "what" questions: give the specific thing/action
5. For "who" questions: give the name
6. If the exact answer is in the memories, use those exact words/dates
7. If you cannot find the answer, say "Information not found"

## Answer (be concise - just the facts):"""

                response = litellm.completion(
                    model=args.answer_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                return response.choices[0].message.content or ""

        except ImportError:
            print("Error: litellm required for --answer-model. Run: pip install litellm")
            return 1

    # Create LLM judge if enabled
    llm_judge_fn: Callable[[str, str, str], tuple[float, str]] | None = None
    if args.llm_judge:
        # Use answer model for judge if not explicitly set
        judge_model = args.judge_model
        if args.answer_model and args.judge_model == "gpt-4o":
            judge_model = args.answer_model  # Match the answer model

        if args.judge_provider == "simple":
            llm_judge_fn = simple_judge
        elif args.judge_provider == "openai":
            llm_judge_fn = create_openai_judge(model=judge_model)
        elif args.judge_provider == "anthropic":
            llm_judge_fn = create_anthropic_judge(model=judge_model)
        else:
            llm_judge_fn = create_litellm_judge(model=judge_model)

    # Determine judge info for display
    judge_info = "DISABLED"
    if args.llm_judge:
        if args.judge_provider == "simple":
            judge_info = "ENABLED (rule-based F1)"
        else:
            jm = args.judge_model
            if args.answer_model and args.judge_model == "gpt-4o":
                jm = args.answer_model
            judge_info = f"ENABLED ({args.judge_provider}: {jm})"

    extract_info = (
        f"ENABLED ({args.extraction_model})" if not args.no_extract else "DISABLED (raw dialogue)"
    )
    retrieval_info = "ALL memories (Path A)" if args.pass_all else f"Top-{args.top_k} retrieval"

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    HEADROOM MEMORY EVALUATION                          ║
║                         LoCoMo Benchmark                               ║
╚═══════════════════════════════════════════════════════════════════════╝

Configuration:
  Conversations:    {args.n_conversations or "all"}
  Categories:       {categories or "[1,2,3,4]"}
  Retrieval:        {retrieval_info}
  Memory Extract:   {extract_info}
  Answer Model:     {args.answer_model or "default (retrieval)"}
  LLM Judge:        {judge_info}
  Parallelism:      {args.parallel} workers
  Debug:            {"ENABLED" if args.debug else "DISABLED"}

Running evaluation...
""")

    # Run evaluation
    evaluator = LoCoMoEvaluator(
        answer_fn=answer_fn,
        llm_judge_fn=llm_judge_fn,
        config=eval_config,
    )

    try:
        result = asyncio.run(evaluator.run())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
        return 1

    # Print results
    print(result.summary())

    # Save results if output path specified
    if args.output:
        result.save(args.output)
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_memory_eval_v2(args: argparse.Namespace) -> int:
    """Run LoCoMo V2 memory evaluation (LLM-controlled tools)."""
    # Suppress noisy pydantic warnings from litellm
    import warnings

    warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    try:
        from headroom.evals.memory import (
            LoCoMoEvaluatorV2,
            MemoryEvalConfigV2,
        )
    except ImportError as e:
        print("Error: Memory eval V2 dependencies not installed.")
        print("Run: pip install headroom[memory,evals]")
        print(f"Details: {e}")
        return 1

    import asyncio

    # Build configuration
    categories = None
    if args.categories:
        categories = [int(c) for c in args.categories.split(",")]

    eval_config = MemoryEvalConfigV2(
        n_conversations=args.n_conversations,
        categories=categories,
        skip_adversarial=not args.include_adversarial,
        llm_judge_enabled=args.llm_judge,
        llm_judge_model=args.judge_model,
        f1_threshold=args.f1_threshold,
        parallel_workers=args.parallel,
        debug=args.debug,
        save_model=args.save_model,
        answer_model=args.answer_model,
        max_search_results=args.max_results,
        include_graph_expansion=not args.no_graph,
    )

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                   HEADROOM MEMORY EVALUATION V2                        ║
║              LLM-Controlled Memory Architecture                        ║
╚═══════════════════════════════════════════════════════════════════════╝

Configuration:
  Conversations:    {args.n_conversations or "all"}
  Categories:       {categories or "[1,2,3,4]"}
  Save Model:       {args.save_model}
  Answer Model:     {args.answer_model}
  Max Results:      {args.max_results}
  Graph Expansion:  {"DISABLED" if args.no_graph else "ENABLED"}
  LLM Judge:        {"ENABLED" if args.llm_judge else "DISABLED"}
  Parallelism:      {args.parallel} workers
  Debug:            {"ENABLED" if args.debug else "DISABLED"}

Key Differences from V1:
  - LLM decides WHAT to save (memory_save tool)
  - LLM decides HOW to search (memory_search tool)
  - Graph expansion enables multi-hop reasoning

Running evaluation...
""")

    # Run evaluation
    evaluator = LoCoMoEvaluatorV2(
        answer_model=args.answer_model,
        config=eval_config,
    )

    try:
        result = asyncio.run(evaluator.run())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
        return 1

    # Print results
    print(result.summary())

    # Save results if output path specified
    if args.output:
        result.save(args.output)
        print(f"\nResults saved to: {args.output}")

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
    # LLMLingua ML-based compression (ON by default if installed)
    proxy_parser.add_argument(
        "--no-llmlingua",
        action="store_true",
        help="Disable LLMLingua-2 ML-based compression",
    )
    proxy_parser.add_argument(
        "--llmlingua-device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device for LLMLingua model (default: auto)",
    )
    proxy_parser.add_argument(
        "--llmlingua-rate",
        type=float,
        default=0.3,
        help="LLMLingua compression rate 0.0-1.0 (default: 0.3 = keep 30%%)",
    )
    # Code-aware compression (ON by default if installed)
    proxy_parser.add_argument(
        "--no-code-aware",
        action="store_true",
        help="Disable AST-based code compression",
    )
    # Intelligent Context Management (ON by default)
    proxy_parser.add_argument(
        "--no-intelligent-context",
        action="store_true",
        help="Disable IntelligentContextManager (fall back to RollingWindow)",
    )
    proxy_parser.add_argument(
        "--no-intelligent-scoring",
        action="store_true",
        help="Disable multi-factor importance scoring (use position-based)",
    )
    proxy_parser.add_argument(
        "--no-compress-first",
        action="store_true",
        help="Disable trying deeper compression before dropping messages",
    )
    # Memory System
    proxy_parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable persistent user memory (uses x-headroom-user-id header if set, otherwise 'default')",
    )
    proxy_parser.add_argument(
        "--memory-backend",
        choices=["local", "qdrant-neo4j"],
        default="local",
        help="Memory storage backend: local (SQLite+HNSW) or qdrant-neo4j (default: local)",
    )
    proxy_parser.add_argument(
        "--memory-db-path",
        default="headroom_memory.db",
        help="Path to memory database file for local backend (default: headroom_memory.db)",
    )
    proxy_parser.add_argument(
        "--no-memory-tools",
        action="store_true",
        help="Disable automatic memory tool injection",
    )
    proxy_parser.add_argument(
        "--no-memory-context",
        action="store_true",
        help="Disable automatic memory context injection",
    )
    proxy_parser.add_argument(
        "--memory-top-k",
        type=int,
        default=10,
        help="Number of memories to inject as context (default: 10)",
    )
    # Backend configuration
    proxy_parser.add_argument(
        "--backend",
        default="anthropic",
        help=(
            "API backend: 'anthropic' (direct), 'bedrock' (AWS), "
            "or 'litellm-<provider>' (e.g., litellm-bedrock, litellm-vertex)"
        ),
    )
    proxy_parser.add_argument(
        "--region",
        default="us-west-2",
        help="Cloud region for Bedrock/Vertex/etc (default: us-west-2)",
    )
    proxy_parser.add_argument(
        "--bedrock-region",
        default=None,
        help="(deprecated, use --region) AWS region for Bedrock",
    )
    proxy_parser.add_argument(
        "--bedrock-profile",
        help="AWS profile name for Bedrock (default: use default credentials)",
    )
    proxy_parser.set_defaults(func=cmd_proxy)

    # Memory eval command
    eval_parser = subparsers.add_parser(
        "memory-eval",
        help="Run LoCoMo memory evaluation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Run the LoCoMo memory benchmark to evaluate the Headroom memory system.

LoCoMo (Long-term Conversational Memory) tests memory across:
- Single-hop questions (simple fact recall)
- Temporal questions (time-based)
- Multi-hop questions (reasoning across memories)
- Open-domain questions (interpretation required)

Example:
  headroom memory-eval --n-conversations 3
  headroom memory-eval --answer-model gpt-4o --llm-judge
        """,
    )
    eval_parser.add_argument(
        "--n-conversations",
        "-n",
        type=int,
        help="Number of conversations to evaluate (default: all 10)",
    )
    eval_parser.add_argument(
        "--categories",
        help="Comma-separated list of categories 1-5 (default: 1,2,3,4)",
    )
    eval_parser.add_argument(
        "--include-adversarial",
        action="store_true",
        help="Include category 5 (unanswerable questions)",
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of memories to retrieve per question (default: 10)",
    )
    eval_parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.5,
        help="F1 score threshold for 'correct' (default: 0.5)",
    )
    eval_parser.add_argument(
        "--answer-model",
        help="LLM model for generating answers (e.g., gpt-4o, claude-sonnet-4-20250514)",
    )
    eval_parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use LLM-as-judge scoring",
    )
    eval_parser.add_argument(
        "--judge-provider",
        choices=["openai", "anthropic", "litellm", "simple"],
        default="litellm",
        help="LLM judge provider (default: litellm - uses same model as answer-model)",
    )
    eval_parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="Model for LLM judge (default: gpt-4o)",
    )
    eval_parser.add_argument(
        "--output",
        "-o",
        help="Path to save JSON results",
    )
    eval_parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Disable LLM memory extraction (store raw dialogue instead)",
    )
    eval_parser.add_argument(
        "--extraction-model",
        default="gpt-4o-mini",
        help="Model for memory extraction (default: gpt-4o-mini)",
    )
    eval_parser.add_argument(
        "--pass-all",
        action="store_true",
        help="Pass ALL memories to LLM (Path A: no retrieval bottleneck)",
    )
    eval_parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Number of parallel workers for LLM calls (default: 10)",
    )
    eval_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (saved to results JSON)",
    )
    eval_parser.set_defaults(func=cmd_memory_eval)

    # Memory eval V2 command (LLM-controlled tools)
    eval_v2_parser = subparsers.add_parser(
        "memory-eval-v2",
        help="Run LoCoMo V2 evaluation with LLM-controlled memory tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Run the LoCoMo V2 memory benchmark with LLM-controlled memory architecture.

This evaluator tests the new architecture where:
- LLM decides what to save (memory_save tool)
- LLM decides when to search (memory_search tool)
- Graph relationships enable multi-hop reasoning

Example:
  headroom memory-eval-v2 --n-conversations 3
  headroom memory-eval-v2 --answer-model gpt-4o --save-model gpt-4o-mini
        """,
    )
    eval_v2_parser.add_argument(
        "--n-conversations",
        "-n",
        type=int,
        help="Number of conversations to evaluate (default: all 10)",
    )
    eval_v2_parser.add_argument(
        "--categories",
        help="Comma-separated list of categories 1-5 (default: 1,2,3,4)",
    )
    eval_v2_parser.add_argument(
        "--include-adversarial",
        action="store_true",
        help="Include category 5 (unanswerable questions)",
    )
    eval_v2_parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.5,
        help="F1 score threshold for 'correct' (default: 0.5)",
    )
    eval_v2_parser.add_argument(
        "--save-model",
        default="gpt-4o-mini",
        help="LLM model for deciding what to save (default: gpt-4o-mini)",
    )
    eval_v2_parser.add_argument(
        "--answer-model",
        default="gpt-4o",
        help="LLM model for answering questions (default: gpt-4o)",
    )
    eval_v2_parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum memories to retrieve per search (default: 10)",
    )
    eval_v2_parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph expansion in search",
    )
    eval_v2_parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use LLM-as-judge scoring",
    )
    eval_v2_parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="Model for LLM judge (default: gpt-4o)",
    )
    eval_v2_parser.add_argument(
        "--output",
        "-o",
        help="Path to save JSON results",
    )
    eval_v2_parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Number of parallel workers for LLM calls (default: 5)",
    )
    eval_v2_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (saved to results JSON)",
    )
    eval_v2_parser.set_defaults(func=cmd_memory_eval_v2)

    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    if args.command is None:
        parser.print_help()
        return 0

    result = args.func(args)
    return int(result) if result is not None else 0


if __name__ == "__main__":
    sys.exit(main())
