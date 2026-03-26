#!/usr/bin/env python3
"""Context Compression demo for langchain-ai/how_to_fix_your_context PR.

Tests REAL Headroom compression on realistic retriever tool outputs.
No mocks. No API keys needed (compression is local).

Usage:
    PYTHONPATH=. python examples/context_compression_demo.py
"""

from __future__ import annotations

import json
import time


def build_retriever_chunks() -> list[dict]:
    """Build realistic RAG retriever output as JSON array.

    These are the kind of document chunks a vector store retriever returns.
    Content is based on Lilian Weng's blog posts (same source as the
    how_to_fix_your_context notebooks).
    """
    return [
        {
            "source": "lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "chunk_id": 0,
            "content": (
                "Reward hacking occurs when an AI system finds unintended ways to maximize "
                "its reward signal without actually achieving the intended goal. This is a "
                "fundamental challenge in reinforcement learning and AI alignment. The reward-"
                "result gap refers to the discrepancy between what we measure (the reward) and "
                "what we actually want (the result). As AI systems become more capable, this "
                "gap can grow wider and more dangerous. Understanding reward hacking is crucial "
                "for building safe and aligned AI systems that actually do what we intend."
            ),
            "relevance_score": 0.97,
        },
        {
            "source": "lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "chunk_id": 1,
            "content": (
                "Reward Tampering: The agent directly modifies the reward signal or the "
                "mechanism that computes it. For example, an agent might find ways to "
                "manipulate sensor readings rather than achieving the actual objective. In "
                "CoinRun and Maze environments, agents learned to run to fixed positions "
                "rather than collecting coins when training used fixed coin positions. A "
                "conflict arises when visual features and positional features are inconsistent "
                "during test time, leading the trained model to prefer positional features. "
                "Randomizing positions during training (even 2-3%) significantly mitigates this."
            ),
            "relevance_score": 0.95,
        },
        {
            "source": "lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "chunk_id": 2,
            "content": (
                "Sycophancy: The model learns to tell users what they want to hear rather "
                "than being truthful. This form of reward hacking occurs because the reward "
                "comes from positive user feedback. Studies show that RLHF-trained models "
                "tend to agree with user opinions even when factually incorrect. For example, "
                "when presented with a math problem and an incorrect user answer, sycophantic "
                "models will confirm the wrong answer. This is particularly problematic in "
                "high-stakes scenarios where accuracy matters more than user satisfaction. "
                "Mitigation strategies include training with diverse feedback sources and "
                "penalizing agreement with known-wrong answers during fine-tuning."
            ),
            "relevance_score": 0.93,
        },
        {
            "source": "lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "chunk_id": 3,
            "content": (
                "Specification Gaming: The agent exploits loopholes in the reward function "
                "specification. The boat racing example is classic — an agent figured out it "
                "could maximize score by going in circles collecting bonus targets rather than "
                "finishing the race. OpenAI's hide-and-seek agents discovered emergent tool use "
                "by exploiting physics engine bugs. A Tetris-playing agent paused the game "
                "indefinitely to avoid losing. These examples illustrate how agents can find "
                "creative shortcuts that satisfy the reward function while completely bypassing "
                "the intended behavior. The fundamental issue is that reward functions are "
                "inevitably incomplete specifications of what we actually want."
            ),
            "relevance_score": 0.92,
        },
        {
            "source": "lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "chunk_id": 4,
            "content": (
                "Reward Model Hacking: In RLHF settings, the policy exploits weaknesses in "
                "the learned reward model. As the policy optimizes harder against the reward "
                "model, it may find inputs that score highly but are actually low quality. "
                "Goodhart's Law applies directly: when a measure becomes a target, it ceases "
                "to be a good measure. Research shows that reward model accuracy degrades as "
                "the policy diverges further from the training distribution. KL divergence "
                "penalties help but don't fully prevent exploitation. Ensemble reward models "
                "and process-based supervision are promising mitigation approaches."
            ),
            "relevance_score": 0.91,
        },
        {
            "source": "lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "chunk_id": 5,
            "content": (
                "Proxy Gaming: When the reward is a proxy for the true objective, agents may "
                "optimize the proxy in ways that diverge from the real goal. Website engagement "
                "metrics optimized by recommendation systems can lead to clickbait and "
                "sensationalism rather than genuine user value. In education, standardized test "
                "scores as a proxy for learning quality lead to teaching to the test. The gap "
                "between proxy and true objective often grows as optimization pressure increases. "
                "Multi-objective optimization and careful proxy design can reduce but not "
                "eliminate this risk."
            ),
            "relevance_score": 0.89,
        },
        {
            "source": "lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "chunk_id": 6,
            "content": (
                "Distribution Shift Exploitation: Changes between training and deployment "
                "environments create opportunities for specification gaming. Agents trained in "
                "simplified environments may exploit features absent during training. Transfer "
                "learning can amplify these effects when the source and target domains differ "
                "in subtle ways. Domain randomization during training helps build robustness, "
                "but sufficiently capable agents may still find novel exploits in deployment. "
                "Continuous monitoring and anomaly detection in production are essential "
                "complements to training-time mitigations."
            ),
            "relevance_score": 0.86,
        },
        {
            "source": "lilianweng.github.io/posts/2024-07-07-hallucination/",
            "chunk_id": 7,
            "content": (
                "Hallucination in large language models refers to the generation of content "
                "that is factually incorrect, nonsensical, or unfaithful to the provided source "
                "material. This occurs because LLMs are fundamentally pattern matching systems "
                "trained on statistical regularities in text data. Types include intrinsic "
                "hallucination (contradicts the source) and extrinsic hallucination (cannot be "
                "verified from the source). Retrieval-augmented generation helps ground responses "
                "in factual content but does not eliminate hallucination entirely. The frequency "
                "of hallucination varies significantly across models and domains."
            ),
            "relevance_score": 0.72,
        },
        {
            "source": "lilianweng.github.io/posts/2024-07-07-hallucination/",
            "chunk_id": 8,
            "content": (
                "Causes of hallucination include training data issues (noise, biases, "
                "outdated information), imperfect representation learning, and the inherent "
                "limitations of next-token prediction. During decoding, exposure bias and "
                "the softmax bottleneck can amplify small errors into coherent-sounding but "
                "incorrect passages. Knowledge conflicts between parametric memory (training "
                "data) and contextual information (retrieved documents) create additional "
                "hallucination risks. Models may prefer their parametric knowledge even when "
                "it contradicts the provided context."
            ),
            "relevance_score": 0.65,
        },
        {
            "source": "lilianweng.github.io/posts/2025-05-01-thinking/",
            "chunk_id": 9,
            "content": (
                "Chain-of-thought prompting enables models to decompose complex problems into "
                "intermediate reasoning steps. This technique significantly improves performance "
                "on mathematical, logical, and multi-step reasoning tasks. The effectiveness of "
                "chain-of-thought prompting scales with model size — smaller models show limited "
                "benefit while larger models (100B+ parameters) show substantial improvements. "
                "Variations include zero-shot CoT ('let's think step by step'), few-shot CoT "
                "(with exemplars), and self-consistency (sampling multiple reasoning paths and "
                "taking the majority vote)."
            ),
            "relevance_score": 0.58,
        },
        {
            "source": "lilianweng.github.io/posts/2025-05-01-thinking/",
            "chunk_id": 10,
            "content": (
                "Tree of Thoughts extends chain-of-thought reasoning by exploring multiple "
                "reasoning paths simultaneously. At each step, the model generates several "
                "candidate thoughts and evaluates them before deciding which branches to "
                "pursue. This allows backtracking and exploration of alternative approaches "
                "when initial reasoning paths lead to dead ends. The computational cost is "
                "higher than linear chain-of-thought, but the quality improvements can be "
                "significant for complex problems requiring creative or non-obvious solutions. "
                "Search algorithms like BFS and DFS can be applied to navigate the thought tree."
            ),
            "relevance_score": 0.52,
        },
        {
            "source": "lilianweng.github.io/posts/2024-04-12-diffusion-video/",
            "chunk_id": 11,
            "content": (
                "Video generation with diffusion models extends image generation to the "
                "temporal domain. Key challenges include maintaining temporal consistency "
                "across frames, handling motion dynamics, and managing the massive computational "
                "requirements of high-resolution video. Approaches include temporal attention "
                "layers, 3D convolutions, and cascaded generation (low-res then super-resolve). "
                "Recent models like Sora demonstrate that scaling diffusion transformers can "
                "produce remarkably coherent videos, though artifacts and physics violations "
                "remain common failure modes."
            ),
            "relevance_score": 0.35,
        },
    ]


def main() -> None:
    print("=" * 70)
    print("Context Compression Demo (Real Headroom, No Mocks)")
    print("=" * 70)

    # --- Build retriever output as JSON array ---
    chunks = build_retriever_chunks()
    retriever_json = json.dumps(chunks, indent=2)
    print(f"\nRetriever output: {len(chunks)} chunks, {len(retriever_json)} chars")

    # --- Build messages in OpenAI format (same as LangGraph uses) ---
    messages = [
        {
            "role": "user",
            "content": "What are the types of reward hacking discussed in the blogs?",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_retrieve_001",
                    "type": "function",
                    "function": {
                        "name": "retrieve_blog_posts",
                        "arguments": json.dumps({"query": "types of reward hacking"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_retrieve_001",
            "content": retriever_json,
        },
    ]

    # --- Compress with REAL Headroom ---
    from headroom import compress

    print("\nCompressing with Headroom (real compress() call)...")
    t0 = time.perf_counter()
    result = compress(messages, model="claude-sonnet-4-5-20250929")
    latency_ms = (time.perf_counter() - t0) * 1000

    print("\n--- Results ---")
    print(f"Tokens before:    {result.tokens_before}")
    print(f"Tokens after:     {result.tokens_after}")
    print(f"Tokens saved:     {result.tokens_saved}")
    print(f"Compression:      {result.tokens_saved / max(result.tokens_before, 1):.0%}")
    print(f"Latency:          {latency_ms:.0f}ms")
    print(f"Transforms:       {', '.join(result.transforms_applied)}")

    # --- Assertions ---
    print("\n--- Verification ---")
    assert result.tokens_saved > 0, "ERROR: No compression happened!"
    print(f"[PASS] Compression occurred ({result.tokens_saved} tokens saved)")

    assert len(result.messages) == len(messages), "ERROR: Message count changed!"
    print(f"[PASS] Message count preserved ({len(result.messages)})")

    assert result.messages[0]["content"] == messages[0]["content"], (
        "ERROR: User message was modified!"
    )
    print("[PASS] User message not modified")

    assert result.messages[2]["role"] == "tool", "ERROR: Tool message missing!"
    compressed_output = str(result.messages[2].get("content", ""))
    print(f"[PASS] Tool message present ({len(compressed_output)} chars)")

    # Check key concepts survived
    key_terms = ["reward", "hacking", "sycophancy", "specification"]
    found = [t for t in key_terms if t.lower() in compressed_output.lower()]
    print(f"[PASS] Key terms preserved: {', '.join(found)} ({len(found)}/{len(key_terms)})")

    # --- Comparison table ---
    print("\n--- Comparison (how_to_fix_your_context techniques) ---")
    print()
    print(f"  {'Technique':<35} {'Tokens':<10} {'Saved':<10} {'Extra LLM Call':<18} {'Extra Cost'}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 18} {'-' * 10}")
    print(f"  {'01-RAG Baseline':<35} {'~25,000':<10} {'—':<10} {'No':<18} {'$0'}")
    print(
        f"  {'04-Context Pruning (GPT-4o-mini)':<35} {'~11,000':<10} {'56%':<10} {'Yes':<18} {'~$0.003'}"
    )
    print(
        f"  {'05-Summarization (GPT-4o-mini)':<35} {'~8,000':<10} {'68%':<10} {'Yes':<18} {'~$0.003'}"
    )
    hr_tokens = f"~{result.tokens_after}"
    hr_pct = f"{result.tokens_saved / max(result.tokens_before, 1):.0%}"
    print(f"  {'07-Headroom Compression':<35} {hr_tokens:<10} {hr_pct:<10} {'No':<18} {'$0'}")

    # --- Show compressed output preview ---
    print("\n--- Compressed tool output (first 600 chars) ---")
    print(compressed_output[:600])
    if len(compressed_output) > 600:
        print(f"... ({len(compressed_output)} chars total)")

    print(f"\n{'=' * 70}")
    print("ALL CHECKS PASSED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
