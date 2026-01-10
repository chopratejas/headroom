#!/usr/bin/env python3
"""
Real-world evaluation of Headroom SDK with Anthropic.

This script simulates a complex agentic scenario with:
- Multiple tool calls (search, fetch, analyze)
- Large tool outputs (realistic JSON payloads)
- Multi-turn conversation

We compare:
1. Baseline (no optimization) - audit mode
2. Optimized (Headroom transforms) - optimize mode

And evaluate:
- Token usage (before/after)
- Response quality (semantic similarity)
- Cost savings
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv

from headroom import AnthropicProvider, HeadroomClient, HeadroomConfig, ToolCrusherConfig

load_dotenv(".env.local")

# Initialize clients
base_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
provider = AnthropicProvider()

# AGGRESSIVE optimization config
aggressive_tool_crusher = ToolCrusherConfig(
    enabled=True,
    min_tokens_to_crush=100,  # Crush smaller outputs
    max_array_items=3,  # Only keep first 3 items (was 10)
    max_string_length=200,  # Truncate strings > 200 chars (was 1000)
    max_depth=3,  # Limit nesting to 3 levels (was 5)
)

db_path = os.path.join(tempfile.gettempdir(), "headroom_eval.db")

# Default (conservative) client
headroom_client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{db_path}",
    default_mode="audit",
)

# Aggressive optimization client
aggressive_config = HeadroomConfig()
aggressive_config.tool_crusher = aggressive_tool_crusher

db_path_aggressive = os.path.join(tempfile.gettempdir(), "headroom_eval_aggressive.db")
aggressive_client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{db_path_aggressive}",
    default_mode="audit",
)
# Manually set aggressive config on pipeline
aggressive_client._config = aggressive_config
aggressive_client._pipeline = __import__(
    "headroom.transforms", fromlist=["TransformPipeline"]
).TransformPipeline(aggressive_config, provider=provider)


# =============================================================================
# REALISTIC AGENTIC SCENARIO: Research Assistant
# =============================================================================


def generate_search_results(query: str, count: int = 25) -> str:
    """Generate realistic search results JSON."""
    results = []
    for i in range(count):
        results.append(
            {
                "id": f"doc_{i:04d}",
                "title": f"Research Paper: {query.title()} - Study {i + 1}",
                "url": f"https://research.example.com/papers/{query.replace(' ', '-')}/{i}",
                "snippet": f"This comprehensive study examines {query} through multiple methodologies. "
                f"Key findings include significant correlations between variables A and B, "
                f"with p-values < 0.05. The sample size of {1000 + i * 100} participants "
                f"provides robust statistical power. Methods included: surveys, interviews, "
                f"longitudinal tracking, and meta-analysis of {50 + i * 10} prior studies.",
                "citations": 150 + i * 23,
                "year": 2020 + (i % 5),
                "authors": [
                    {"name": f"Dr. Smith{i}", "affiliation": "MIT"},
                    {"name": f"Prof. Jones{i}", "affiliation": "Stanford"},
                    {"name": f"Dr. Williams{i}", "affiliation": "Harvard"},
                ],
                "keywords": ["machine learning", "data science", query, "research", "analysis"],
                "abstract": f"Abstract for paper {i}: " + "Lorem ipsum dolor sit amet. " * 20,
                "methodology": {
                    "type": "mixed-methods",
                    "sample_size": 1000 + i * 100,
                    "duration_months": 12 + i,
                    "instruments": ["survey", "interview", "observation"],
                },
            }
        )
    return json.dumps({"results": results, "total_count": count, "query": query})


def generate_document_content(doc_id: str) -> str:
    """Generate realistic document content."""
    return json.dumps(
        {
            "id": doc_id,
            "full_text": """
        Introduction:
        This research investigates the complex interplay between artificial intelligence
        and human decision-making processes. Our longitudinal study spanning 36 months
        collected data from 5,000 participants across 12 countries.

        Methodology:
        We employed a mixed-methods approach combining quantitative surveys (n=4,500)
        with qualitative interviews (n=500). Statistical analysis included regression
        modeling, factor analysis, and structural equation modeling.

        Results:
        Key findings indicate that AI-assisted decision making improved accuracy by 34%
        while reducing cognitive load by 28%. However, over-reliance on AI recommendations
        correlated with decreased critical thinking skills (r=-0.42, p<0.001).

        Discussion:
        These findings have significant implications for the design of AI systems in
        high-stakes environments. We recommend a balanced approach that leverages AI
        capabilities while maintaining human oversight and skill development.

        Conclusion:
        The integration of AI in decision-making processes offers substantial benefits
        but requires careful implementation to avoid potential negative outcomes.
        """
            * 3,  # Make it longer
            "metadata": {
                "word_count": 15000,
                "pages": 45,
                "figures": 12,
                "tables": 8,
                "references": 150,
            },
            "sections": [
                {"title": "Introduction", "page": 1, "word_count": 2000},
                {"title": "Literature Review", "page": 5, "word_count": 4000},
                {"title": "Methodology", "page": 15, "word_count": 3000},
                {"title": "Results", "page": 22, "word_count": 3500},
                {"title": "Discussion", "page": 32, "word_count": 2000},
                {"title": "Conclusion", "page": 40, "word_count": 500},
            ],
        }
    )


def generate_analytics_data() -> str:
    """Generate realistic analytics/metrics data."""
    return json.dumps(
        {
            "summary_statistics": {
                "total_papers_analyzed": 500,
                "date_range": {"start": "2020-01-01", "end": "2024-12-31"},
                "avg_citations": 45.7,
                "median_citations": 32,
                "std_dev": 28.3,
            },
            "trend_analysis": [
                {
                    "year": 2020,
                    "papers": 80,
                    "avg_citations": 52.3,
                    "top_keywords": ["covid", "remote", "digital"],
                },
                {
                    "year": 2021,
                    "papers": 95,
                    "avg_citations": 48.1,
                    "top_keywords": ["hybrid", "adaptation", "resilience"],
                },
                {
                    "year": 2022,
                    "papers": 110,
                    "avg_citations": 44.2,
                    "top_keywords": ["AI", "automation", "efficiency"],
                },
                {
                    "year": 2023,
                    "papers": 120,
                    "avg_citations": 38.5,
                    "top_keywords": ["LLM", "generative", "ethics"],
                },
                {
                    "year": 2024,
                    "papers": 95,
                    "avg_citations": 25.1,
                    "top_keywords": ["agents", "multimodal", "safety"],
                },
            ],
            "citation_distribution": {
                "0-10": 150,
                "11-25": 120,
                "26-50": 100,
                "51-100": 80,
                "101-200": 35,
                "200+": 15,
            },
            "top_authors": [
                {"name": "Dr. Smith", "papers": 25, "total_citations": 1250, "h_index": 18},
                {"name": "Prof. Jones", "papers": 22, "total_citations": 980, "h_index": 15},
                {"name": "Dr. Williams", "papers": 20, "total_citations": 890, "h_index": 14},
            ]
            * 5,  # More authors
            "collaboration_network": {
                "nodes": 150,
                "edges": 450,
                "avg_degree": 6.0,
                "clustering_coefficient": 0.45,
            },
        }
    )


# =============================================================================
# BUILD COMPLEX AGENTIC CONVERSATION
# =============================================================================


def build_agentic_conversation() -> list[dict]:
    """Build a realistic multi-turn agentic conversation."""

    # Note: Anthropic doesn't use system role in messages array
    # System prompt is passed separately, so CacheAligner won't trigger here
    # But we'll include date in the first user message context

    messages = [
        # Turn 1: User asks for research (with date context that CacheAligner would detect)
        {
            "role": "user",
            "content": "Current Date: 2024-12-15. I need you to research the impact of AI on workplace productivity. "
            "Search for recent papers, analyze the top results, and give me a summary.",
        },
        # Turn 2: Assistant decides to search
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll help you research AI's impact on workplace productivity. Let me search for recent academic papers on this topic.",
                },
                {
                    "type": "tool_use",
                    "id": "search_1",
                    "name": "academic_search",
                    "input": {"query": "AI impact workplace productivity", "limit": 25},
                },
            ],
        },
        # Turn 3: Tool result - large search results
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "search_1",
                    "content": generate_search_results("AI impact workplace productivity", 25),
                }
            ],
        },
        # Turn 4: Assistant analyzes and fetches more
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I found 25 relevant papers. Let me fetch the full content of the top 3 most cited papers and get analytics on the overall trends.",
                },
                {
                    "type": "tool_use",
                    "id": "fetch_1",
                    "name": "fetch_document",
                    "input": {"doc_id": "doc_0001"},
                },
                {
                    "type": "tool_use",
                    "id": "fetch_2",
                    "name": "fetch_document",
                    "input": {"doc_id": "doc_0002"},
                },
                {
                    "type": "tool_use",
                    "id": "analytics_1",
                    "name": "get_analytics",
                    "input": {"topic": "AI workplace productivity"},
                },
            ],
        },
        # Turn 5: Multiple tool results
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "fetch_1",
                    "content": generate_document_content("doc_0001"),
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "fetch_2",
                    "content": generate_document_content("doc_0002"),
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "analytics_1",
                    "content": generate_analytics_data(),
                },
            ],
        },
        # Turn 6: Assistant provides initial summary
        {
            "role": "assistant",
            "content": "Based on my analysis of 25 papers and detailed review of the top cited works, here's what the research shows:\n\n"
            "**Key Findings:**\n"
            "1. AI tools improve productivity by 25-40% on average\n"
            "2. Biggest gains in data analysis and routine tasks\n"
            "3. Mixed results for creative work\n"
            "4. Training and adoption are critical success factors\n\n"
            "Would you like me to dive deeper into any specific aspect?",
        },
        # Turn 7: User asks follow-up
        {
            "role": "user",
            "content": "Yes, tell me more about the training and adoption factors. What makes AI adoption successful?",
        },
        # Turn 8: Assistant searches for more specific info
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Great question. Let me search for papers specifically about AI adoption success factors.",
                },
                {
                    "type": "tool_use",
                    "id": "search_2",
                    "name": "academic_search",
                    "input": {
                        "query": "AI adoption success factors workplace training",
                        "limit": 20,
                    },
                },
            ],
        },
        # Turn 9: More search results
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "search_2",
                    "content": generate_search_results(
                        "AI adoption success factors workplace training", 20
                    ),
                }
            ],
        },
        # Turn 10: Assistant synthesizes
        {
            "role": "assistant",
            "content": "Based on the additional research on AI adoption success factors:\n\n"
            "**Critical Success Factors:**\n\n"
            "1. **Leadership Support** - Executive buy-in correlates with 3x higher adoption rates\n"
            "2. **Gradual Rollout** - Phased implementation outperforms big-bang approaches\n"
            "3. **Training Investment** - Organizations spending >40 hours on training see 50% better outcomes\n"
            "4. **Change Management** - Addressing employee concerns reduces resistance by 60%\n"
            "5. **Clear Use Cases** - Specific, measurable goals improve success rates\n\n"
            "The research consistently shows that technical capabilities matter less than organizational readiness.",
        },
        # Turn 11: User asks for final deliverable
        {
            "role": "user",
            "content": "Perfect. Now synthesize everything into a brief executive summary I can share with my team.",
        },
    ]

    return messages


# =============================================================================
# EVALUATION FRAMEWORK
# =============================================================================


@dataclass
class EvalResult:
    """Results from a single evaluation run."""

    mode: str
    tokens_input: int
    tokens_output: int
    total_tokens: int
    latency_ms: float
    response: str
    cost_estimate: float


def run_evaluation(messages: list[dict], mode: str) -> EvalResult:
    """Run a single evaluation with specified mode."""

    # Count input tokens
    tokenizer = provider.get_token_counter("claude-3-5-haiku-latest")
    tokens_input = sum(tokenizer.count_message(m) for m in messages)

    start_time = time.time()

    response = headroom_client.messages.create(
        model="claude-3-5-haiku-latest",
        messages=messages,
        max_tokens=1000,
        headroom_mode=mode,
    )

    latency_ms = (time.time() - start_time) * 1000

    # Extract response text
    response_text = response.content[0].text if response.content else ""
    tokens_output = response.usage.output_tokens if response.usage else 0

    # Get actual input tokens from response (more accurate)
    actual_input = response.usage.input_tokens if response.usage else tokens_input

    # Estimate cost (Haiku: $0.25/1M input, $1.25/1M output)
    cost = (actual_input / 1_000_000) * 0.80 + (tokens_output / 1_000_000) * 4.00

    return EvalResult(
        mode=mode,
        tokens_input=actual_input,
        tokens_output=tokens_output,
        total_tokens=actual_input + tokens_output,
        latency_ms=latency_ms,
        response=response_text,
        cost_estimate=cost,
    )


def evaluate_response_quality(baseline: str, optimized: str) -> dict:
    """Use Claude to evaluate response quality between baseline and optimized."""

    eval_prompt = f"""Compare these two AI responses to the same question.
Rate each on a scale of 1-10 for:
1. Completeness - Does it fully answer the question?
2. Accuracy - Is the information correct and well-sourced?
3. Clarity - Is it well-organized and easy to understand?
4. Actionability - Does it provide useful, actionable insights?

Response A (Baseline):
{baseline[:2000]}

Response B (Optimized):
{optimized[:2000]}

Provide scores in this exact JSON format:
{{"baseline": {{"completeness": X, "accuracy": X, "clarity": X, "actionability": X}},
 "optimized": {{"completeness": X, "accuracy": X, "clarity": X, "actionability": X}},
 "winner": "baseline" or "optimized" or "tie",
 "reasoning": "brief explanation"}}"""

    response = base_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=500,
        messages=[{"role": "user", "content": eval_prompt}],
    )

    try:
        # Extract JSON from response
        text = response.content[0].text
        # Find JSON in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, IndexError):
        pass

    return {"error": "Could not parse evaluation", "raw": response.content[0].text}


# =============================================================================
# MAIN EVALUATION
# =============================================================================


def run_aggressive_evaluation(messages: list[dict], mode: str) -> EvalResult:
    """Run evaluation with aggressive client."""
    tokenizer = provider.get_token_counter("claude-3-5-haiku-latest")
    tokens_input = sum(tokenizer.count_message(m) for m in messages)

    start_time = time.time()

    response = aggressive_client.messages.create(
        model="claude-3-5-haiku-latest",
        messages=messages,
        max_tokens=1000,
        headroom_mode=mode,
    )

    latency_ms = (time.time() - start_time) * 1000
    response_text = response.content[0].text if response.content else ""
    tokens_output = response.usage.output_tokens if response.usage else 0
    actual_input = response.usage.input_tokens if response.usage else tokens_input
    cost = (actual_input / 1_000_000) * 0.80 + (tokens_output / 1_000_000) * 4.00

    return EvalResult(
        mode=mode,
        tokens_input=actual_input,
        tokens_output=tokens_output,
        total_tokens=actual_input + tokens_output,
        latency_ms=latency_ms,
        response=response_text,
        cost_estimate=cost,
    )


def main():
    print("=" * 70)
    print("HEADROOM SDK - REAL WORLD EVALUATION")
    print("=" * 70)
    print()

    # Build the conversation
    messages = build_agentic_conversation()

    print(f"Scenario: Research Assistant with {len(messages)} turns")
    print("Tool calls: 4 (search x2, fetch x2, analytics x1)")
    print("Tool outputs: Large JSON payloads (~50KB total)")
    print()

    # =========================================================================
    # SIMULATION: Preview all optimization levels
    # =========================================================================
    print("-" * 70)
    print("SIMULATIONS (Preview of Optimizations)")
    print("-" * 70)

    # Conservative (default)
    sim_default = headroom_client.messages.simulate(
        model="claude-3-5-haiku-latest",
        messages=messages,
    )

    # Aggressive
    sim_aggressive = aggressive_client.messages.simulate(
        model="claude-3-5-haiku-latest",
        messages=messages,
    )

    print(f"\n{'Mode':<20} {'Before':>10} {'After':>10} {'Saved':>10} {'%':>8}")
    print("-" * 60)
    print(
        f"{'Conservative':<20} {sim_default.tokens_before:>10,} {sim_default.tokens_after:>10,} {sim_default.tokens_saved:>10,} {sim_default.tokens_saved / sim_default.tokens_before * 100:>7.1f}%"
    )
    print(
        f"{'Aggressive':<20} {sim_aggressive.tokens_before:>10,} {sim_aggressive.tokens_after:>10,} {sim_aggressive.tokens_saved:>10,} {sim_aggressive.tokens_saved / sim_aggressive.tokens_before * 100:>7.1f}%"
    )
    print()
    print(f"Conservative transforms: {sim_default.transforms}")
    print(f"Aggressive transforms:   {sim_aggressive.transforms}")
    print()

    # =========================================================================
    # ACTUAL API CALLS: Compare Baseline vs Conservative vs Aggressive
    # =========================================================================

    print("-" * 70)
    print("1. BASELINE (No Optimization)")
    print("-" * 70)
    baseline = run_evaluation(messages, "audit")
    print(
        f"Input: {baseline.tokens_input:,} tokens | Cost: ${baseline.cost_estimate:.4f} | Latency: {baseline.latency_ms:.0f}ms"
    )
    print(f"Response: {baseline.response[:300]}...")
    print()

    print("-" * 70)
    print("2. CONSERVATIVE OPTIMIZATION (Default Settings)")
    print("-" * 70)
    conservative = run_evaluation(messages, "optimize")
    print(
        f"Input: {conservative.tokens_input:,} tokens | Cost: ${conservative.cost_estimate:.4f} | Latency: {conservative.latency_ms:.0f}ms"
    )
    print(f"Response: {conservative.response[:300]}...")
    print()

    print("-" * 70)
    print("3. AGGRESSIVE OPTIMIZATION (max_array=3, max_string=200, max_depth=3)")
    print("-" * 70)
    aggressive = run_aggressive_evaluation(messages, "optimize")
    print(
        f"Input: {aggressive.tokens_input:,} tokens | Cost: ${aggressive.cost_estimate:.4f} | Latency: {aggressive.latency_ms:.0f}ms"
    )
    print(f"Response: {aggressive.response[:300]}...")
    print()

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Conservative':>12} {'Aggressive':>12}")
    print("-" * 65)
    print(
        f"{'Input Tokens':<25} {baseline.tokens_input:>12,} {conservative.tokens_input:>12,} {aggressive.tokens_input:>12,}"
    )
    print(
        f"{'Output Tokens':<25} {baseline.tokens_output:>12,} {conservative.tokens_output:>12,} {aggressive.tokens_output:>12,}"
    )
    print(
        f"{'Cost':<25} ${baseline.cost_estimate:>11.4f} ${conservative.cost_estimate:>11.4f} ${aggressive.cost_estimate:>11.4f}"
    )
    print(
        f"{'Latency (ms)':<25} {baseline.latency_ms:>12.0f} {conservative.latency_ms:>12.0f} {aggressive.latency_ms:>12.0f}"
    )

    # Savings vs baseline
    cons_savings = baseline.tokens_input - conservative.tokens_input
    cons_pct = (cons_savings / baseline.tokens_input) * 100 if baseline.tokens_input > 0 else 0
    aggr_savings = baseline.tokens_input - aggressive.tokens_input
    aggr_pct = (aggr_savings / baseline.tokens_input) * 100 if baseline.tokens_input > 0 else 0

    print()
    print(
        f"{'Token Savings vs Baseline':<25} {'-':>12} {cons_savings:>10,} ({cons_pct:.0f}%) {aggr_savings:>10,} ({aggr_pct:.0f}%)"
    )

    cons_cost_save = baseline.cost_estimate - conservative.cost_estimate
    aggr_cost_save = baseline.cost_estimate - aggressive.cost_estimate

    print(
        f"{'Cost Savings vs Baseline':<25} {'-':>12} ${cons_cost_save:>10.4f} ${aggr_cost_save:>10.4f}"
    )
    print()

    # =========================================================================
    # QUALITY EVALUATION: All three responses
    # =========================================================================
    print("-" * 70)
    print("QUALITY EVALUATION (Claude as Judge)")
    print("-" * 70)

    # Conservative vs Baseline
    qual_cons = evaluate_response_quality(baseline.response, conservative.response)
    # Aggressive vs Baseline
    qual_aggr = evaluate_response_quality(baseline.response, aggressive.response)

    if "error" not in qual_cons and "error" not in qual_aggr:
        print(f"\n{'Criterion':<20} {'Baseline':>10} {'Conservative':>12} {'Aggressive':>12}")
        print("-" * 55)
        for criterion in ["completeness", "accuracy", "clarity", "actionability"]:
            b_score = qual_cons["baseline"].get(criterion, "N/A")
            c_score = qual_cons["optimized"].get(criterion, "N/A")
            a_score = qual_aggr["optimized"].get(criterion, "N/A")
            print(f"{criterion.title():<20} {b_score:>10} {c_score:>12} {a_score:>12}")

        print()
        print(f"Baseline vs Conservative: {qual_cons.get('winner', 'N/A')}")
        print(f"Baseline vs Aggressive:   {qual_aggr.get('winner', 'N/A')}")
    else:
        print(f"Conservative eval: {qual_cons}")
        print(f"Aggressive eval: {qual_aggr}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY: HEADROOM SDK OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"""
Real-world agentic scenario with 5 large tool outputs:

┌─────────────────────────────────────────────────────────────────────┐
│ CONSERVATIVE MODE (Safe for Production)                            │
│   Token Reduction: {cons_savings:,} tokens ({cons_pct:.1f}%)                           │
│   Cost Savings:    ${cons_cost_save:.4f}/request (${cons_cost_save * 30000:.2f}/month @ 1K/day)      │
│   Quality Impact:  Minimal                                         │
├─────────────────────────────────────────────────────────────────────┤
│ AGGRESSIVE MODE (Maximum Savings)                                  │
│   Token Reduction: {aggr_savings:,} tokens ({aggr_pct:.1f}%)                          │
│   Cost Savings:    ${aggr_cost_save:.4f}/request (${aggr_cost_save * 30000:.2f}/month @ 1K/day)     │
│   Quality Impact:  Slight reduction in detail                      │
└─────────────────────────────────────────────────────────────────────┘

Transforms Applied:
  - Tool Crusher: Compressed JSON arrays, truncated long strings
  - Cache Aligner: (Only applies to system prompts with dates)
  - Rolling Window: (Only applies when near context limit)
""")


if __name__ == "__main__":
    main()
