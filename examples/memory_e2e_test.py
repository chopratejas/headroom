#!/usr/bin/env python3
"""End-to-end memory system test with LLM-as-judge evaluation.

This program tests the memory extraction and retrieval system with:
1. Multi-turn conversations containing embedded memory nuggets
2. Real OpenAI API calls for extraction and conversation
3. LLM-as-judge evaluation of memory quality

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/memory_e2e_test.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from headroom.memory import with_memory
from headroom.memory.store import SQLiteMemoryStore

# =============================================================================
# Test Scenarios - Conversations with embedded memory nuggets
# =============================================================================


@dataclass
class MemoryNugget:
    """A fact that should be remembered from the conversation."""

    content: str
    category: str  # preference, fact, context
    importance: float  # 0.0-1.0
    turn_index: int  # Which turn contains this nugget


@dataclass
class TestScenario:
    """A test scenario with conversation and expected memories."""

    name: str
    description: str
    entity_id: str  # user_id or agent_id
    conversation: list[tuple[str, str]]  # List of (user_msg, expected_response_topic)
    expected_nuggets: list[MemoryNugget]
    retrieval_queries: list[tuple[str, list[str]]]  # (query, expected_keywords_in_memory)


# Scenario 1: Software Developer User
DEVELOPER_SCENARIO = TestScenario(
    name="software_developer",
    description="A software developer discussing their preferences and projects",
    entity_id="dev_alice",
    conversation=[
        # Turn 0 - Preference nugget
        (
            "Hi! I'm starting a new backend project. I strongly prefer Python over "
            "JavaScript for backend work because of its cleaner syntax.",
            "backend_project_advice",
        ),
        # Turn 1 - Fact nugget
        (
            "Good point. I work at a fintech startup called PayFlow where we handle "
            "high-volume payment processing.",
            "fintech_architecture",
        ),
        # Turn 2 - Context nugget
        (
            "We're currently migrating from a monolith to microservices. It's been "
            "challenging but necessary for scale.",
            "migration_advice",
        ),
        # Turn 3 - Preference nugget
        (
            "For databases, I always use PostgreSQL. I've tried MongoDB but found "
            "relational databases more reliable for financial data.",
            "database_choice",
        ),
        # Turn 4 - Casual (no nugget expected)
        ("Thanks for all the help today!", "closing"),
    ],
    expected_nuggets=[
        MemoryNugget("Prefers Python over JavaScript for backend", "preference", 0.8, 0),
        MemoryNugget("Works at fintech startup PayFlow", "fact", 0.9, 1),
        MemoryNugget("Handles high-volume payment processing", "fact", 0.7, 1),
        MemoryNugget("Migrating from monolith to microservices", "context", 0.8, 2),
        MemoryNugget("Prefers PostgreSQL over MongoDB", "preference", 0.8, 3),
        MemoryNugget("Works with financial data", "fact", 0.7, 3),
    ],
    retrieval_queries=[
        # FTS5 is keyword-based, so queries must contain matching words
        ("Python backend", ["Python", "backend"]),
        ("PostgreSQL database", ["PostgreSQL", "database"]),
        ("PayFlow fintech", ["PayFlow", "fintech"]),
        ("microservices migration", ["microservices", "monolith"]),
    ],
)

# Scenario 2: AI Research Agent
AGENT_SCENARIO = TestScenario(
    name="research_agent",
    description="An AI agent discussing its capabilities and constraints",
    entity_id="agent_researcher",
    conversation=[
        # Turn 0 - Capability fact
        (
            "I'm Agent-7, specialized in scientific literature analysis. I can process "
            "up to 50 papers per hour and identify cross-domain connections.",
            "agent_intro",
        ),
        # Turn 1 - Constraint context
        (
            "My knowledge cutoff is March 2025, so I may not have the latest preprints. "
            "I work best with structured abstracts.",
            "limitations",
        ),
        # Turn 2 - Preference
        (
            "When summarizing papers, I prefer to use the IMRaD structure - Introduction, "
            "Methods, Results, and Discussion. It's more systematic.",
            "summary_format",
        ),
        # Turn 3 - Configuration fact
        (
            "I'm currently configured to prioritize papers from Nature, Science, and Cell "
            "journals, with a citation threshold of 10+.",
            "configuration",
        ),
        # Turn 4 - Context about ongoing task
        (
            "Right now I'm tracking the emerging field of mechanistic interpretability "
            "in neural networks. It's my primary research focus.",
            "current_focus",
        ),
    ],
    expected_nuggets=[
        MemoryNugget("Agent-7 specialized in scientific literature", "fact", 0.9, 0),
        MemoryNugget("Can process 50 papers per hour", "fact", 0.7, 0),
        MemoryNugget("Knowledge cutoff March 2025", "context", 0.8, 1),
        MemoryNugget("Prefers IMRaD structure for summaries", "preference", 0.8, 2),
        MemoryNugget("Prioritizes Nature, Science, Cell journals", "fact", 0.7, 3),
        MemoryNugget("Citation threshold of 10+", "fact", 0.6, 3),
        MemoryNugget("Focus on mechanistic interpretability", "context", 0.9, 4),
    ],
    retrieval_queries=[
        # FTS5 keyword-based queries
        ("scientific papers analysis", ["papers", "scientific", "literature"]),
        ("IMRaD summary structure", ["IMRaD", "structure"]),
        ("Nature Science Cell journals", ["Nature", "Science", "Cell"]),
        ("mechanistic interpretability neural", ["mechanistic", "interpretability"]),
    ],
)

# Scenario 3: Multi-session customer
CUSTOMER_SCENARIO = TestScenario(
    name="returning_customer",
    description="A customer across multiple support interactions",
    entity_id="customer_bob",
    conversation=[
        # Turn 0 - Account fact
        (
            "Hi, I'm Bob Chen, account number AC-789456. I've been a premium member since 2021.",
            "account_lookup",
        ),
        # Turn 1 - Preference
        (
            "Please always contact me via email at bob.chen@email.com, never by phone. "
            "I work odd hours as a night shift nurse.",
            "contact_preference",
        ),
        # Turn 2 - Issue context
        (
            "I've had recurring issues with billing - this is the third time this month "
            "I've been double-charged.",
            "billing_issue",
        ),
        # Turn 3 - Product preference
        (
            "I mainly use your enterprise plan for the API access. The dashboard features "
            "I never touch.",
            "usage_pattern",
        ),
    ],
    expected_nuggets=[
        MemoryNugget("Bob Chen, account AC-789456", "fact", 0.9, 0),
        MemoryNugget("Premium member since 2021", "fact", 0.7, 0),
        MemoryNugget("Prefers email contact, never phone", "preference", 0.9, 1),
        MemoryNugget("Works as night shift nurse", "fact", 0.6, 1),
        MemoryNugget("Recurring billing/double-charge issues", "context", 0.8, 2),
        MemoryNugget("Uses enterprise plan for API access", "fact", 0.7, 3),
    ],
    retrieval_queries=[
        # FTS5 keyword-based queries
        ("Bob Chen account premium", ["Bob", "account", "premium"]),
        ("email contact phone", ["email", "phone"]),
        ("billing double charged", ["billing", "charged"]),
        ("enterprise API plan", ["API", "enterprise"]),
    ],
)

ALL_SCENARIOS = [DEVELOPER_SCENARIO, AGENT_SCENARIO, CUSTOMER_SCENARIO]


# =============================================================================
# Conversation Simulator
# =============================================================================


class ConversationSimulator:
    """Simulates realistic conversations using OpenAI."""

    def __init__(self, client: OpenAI):
        self.client = client
        self.model = "gpt-4o-mini"

    def generate_response(self, user_message: str, topic_hint: str) -> str:
        """Generate a realistic assistant response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Respond naturally and briefly "
                        "to the user's message. Keep responses under 100 words."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content


# =============================================================================
# LLM-as-Judge Evaluator
# =============================================================================


@dataclass
class EvaluationResult:
    """Result of LLM judge evaluation."""

    scenario_name: str
    extraction_score: float  # 0-1, how many expected nuggets were captured
    retrieval_score: float  # 0-1, how well queries retrieved relevant memories
    overall_score: float
    extracted_memories: list[dict]
    missing_nuggets: list[str]
    retrieval_results: list[dict]
    judge_reasoning: str


class LLMJudge:
    """Uses LLM to evaluate memory extraction and retrieval quality."""

    def __init__(self, client: OpenAI):
        self.client = client
        self.model = "gpt-4o"  # Use stronger model for judging

    def evaluate_extraction(
        self,
        scenario: TestScenario,
        extracted_memories: list[dict],
    ) -> tuple[float, list[str], str]:
        """Evaluate if extracted memories capture expected nuggets."""

        prompt = f"""You are evaluating a memory extraction system.

The system processed this conversation and extracted memories.

## Expected Information to Remember:
{json.dumps([{"content": n.content, "category": n.category, "importance": n.importance} for n in scenario.expected_nuggets], indent=2)}

## Actually Extracted Memories:
{json.dumps(extracted_memories, indent=2)}

## Evaluation Task:
1. For each expected nugget, determine if it was captured (exact match not required - semantic similarity counts)
2. Calculate what percentage of expected nuggets were captured
3. Identify which nuggets were MISSING

Return a JSON object:
{{
  "captured_count": <number of expected nuggets that were captured>,
  "total_expected": {len(scenario.expected_nuggets)},
  "score": <0.0 to 1.0>,
  "missing_nuggets": ["list of expected nuggets that were not captured"],
  "reasoning": "Brief explanation of the evaluation"
}}

Return ONLY valid JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result = json.loads(response.choices[0].message.content)
        return (
            result.get("score", 0.0),
            result.get("missing_nuggets", []),
            result.get("reasoning", ""),
        )

    def evaluate_retrieval(
        self,
        query: str,
        expected_keywords: list[str],
        retrieved_memories: list[dict],
    ) -> tuple[float, str]:
        """Evaluate if retrieval returned relevant memories."""

        prompt = f"""You are evaluating a memory retrieval system.

## Query: "{query}"

## Expected Keywords in Results: {expected_keywords}

## Retrieved Memories:
{json.dumps(retrieved_memories, indent=2)}

## Evaluation Task:
Determine if the retrieved memories are relevant to the query and contain the expected information.

Return a JSON object:
{{
  "score": <0.0 to 1.0>,
  "keywords_found": ["list of expected keywords that appeared in results"],
  "reasoning": "Brief explanation"
}}

Return ONLY valid JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("score", 0.0), result.get("reasoning", "")


# =============================================================================
# Main Test Runner
# =============================================================================


class MemoryE2ETest:
    """End-to-end test runner for the memory system."""

    def __init__(self, api_key: str):
        self.raw_client = OpenAI(api_key=api_key)
        self.simulator = ConversationSimulator(self.raw_client)
        self.judge = LLMJudge(self.raw_client)
        self.results: list[EvaluationResult] = []

    def run_scenario(self, scenario: TestScenario, db_path: Path) -> EvaluationResult:
        """Run a complete test scenario."""
        print(f"\n{'=' * 60}")
        print(f"Running scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"{'=' * 60}")

        # Create memory-wrapped client
        store = SQLiteMemoryStore(db_path)
        memory_client = with_memory(
            self.raw_client,
            user_id=scenario.entity_id,
            db_path=db_path,
            _store=store,
        )

        # Run conversation through memory-wrapped client
        print(f"\n--- Running {len(scenario.conversation)} conversation turns ---")
        for i, (user_msg, _topic_hint) in enumerate(scenario.conversation):
            print(f"\nTurn {i + 1}:")
            print(f"  User: {user_msg[:80]}...")

            # Send through memory-wrapped client - this:
            # 1. Retrieves relevant memories (if any)
            # 2. Injects them into user message
            # 3. Calls the actual API
            # 4. Queues extraction of (original_query, response) in background
            response = memory_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": user_msg},
                ],
            )
            assistant_response = response.choices[0].message.content
            print(f"  Assistant: {assistant_response[:80]}...")

            # Small delay to allow background extraction
            time.sleep(0.5)

        # Flush all extractions (force immediate processing)
        print("\n--- Flushing background extractions ---")
        start_time = time.time()
        success = memory_client.flush_extractions(timeout=60.0)
        elapsed = time.time() - start_time

        if success:
            print(f"  All extractions complete in {elapsed:.1f}s")
        else:
            print(f"  WARNING: Flush timed out after {elapsed:.1f}s")
            pending = store.get_pending_extractions(limit=10)
            still_pending = [p for p in pending if p.status == "pending"]
            if still_pending:
                print(f"  {len(still_pending)} extractions still pending")
                for p in still_pending[:2]:
                    print(f"    - Query: {p.query[:50]}...")

        # If no memories extracted, try direct extraction for debugging
        all_memories = store.get_all(scenario.entity_id)
        if not all_memories and scenario.conversation:
            print("\n  DEBUG: Attempting direct extraction for first turn...")
            from headroom.memory.extractor import MemoryExtractor

            extractor = MemoryExtractor(self.raw_client)
            first_query, _ = scenario.conversation[0]
            test_response = "Acknowledged, I understand."
            direct_memories = extractor.extract(first_query, test_response)
            print(f"  DEBUG: Direct extraction got {len(direct_memories)} memories")
            for m in direct_memories[:3]:
                print(f"    - [{m.category}] {m.content[:50]}...")

        # Get all extracted memories (refresh)
        extracted = [
            {"content": m.content, "category": m.category, "importance": m.importance}
            for m in all_memories
        ]
        print(f"\n--- Extracted {len(extracted)} memories ---")
        for m in extracted:
            print(f"  [{m['category']}] {m['content'][:60]}...")

        # Evaluate extraction quality
        print("\n--- Evaluating extraction quality ---")
        extraction_score, missing, extraction_reasoning = self.judge.evaluate_extraction(
            scenario, extracted
        )
        print(f"  Extraction Score: {extraction_score:.2f}")
        if missing:
            print(f"  Missing nuggets: {len(missing)}")
            for m in missing[:3]:
                print(f"    - {m[:60]}...")

        # Test retrieval queries
        print("\n--- Testing retrieval queries ---")
        retrieval_results = []
        retrieval_scores = []

        for query, expected_keywords in scenario.retrieval_queries:
            results = store.search(scenario.entity_id, query, top_k=5)
            retrieved = [{"content": m.content, "category": m.category} for m in results]

            score, reasoning = self.judge.evaluate_retrieval(query, expected_keywords, retrieved)
            retrieval_scores.append(score)

            retrieval_results.append(
                {
                    "query": query,
                    "expected_keywords": expected_keywords,
                    "retrieved_count": len(retrieved),
                    "score": score,
                    "reasoning": reasoning,
                }
            )
            print(f"  Query: '{query[:40]}...' -> Score: {score:.2f}, Found: {len(retrieved)}")

        avg_retrieval_score = (
            sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
        )

        # Calculate overall score
        overall_score = (extraction_score * 0.6) + (avg_retrieval_score * 0.4)

        result = EvaluationResult(
            scenario_name=scenario.name,
            extraction_score=extraction_score,
            retrieval_score=avg_retrieval_score,
            overall_score=overall_score,
            extracted_memories=extracted,
            missing_nuggets=missing,
            retrieval_results=retrieval_results,
            judge_reasoning=extraction_reasoning,
        )

        print("\n--- Scenario Complete ---")
        print(f"  Extraction Score: {extraction_score:.2f}")
        print(f"  Retrieval Score:  {avg_retrieval_score:.2f}")
        print(f"  Overall Score:    {overall_score:.2f}")

        return result

    def run_all_scenarios(self) -> list[EvaluationResult]:
        """Run all test scenarios."""
        print("\n" + "=" * 60)
        print("MEMORY SYSTEM END-TO-END TEST")
        print("=" * 60)
        print(f"Running {len(ALL_SCENARIOS)} scenarios with LLM-as-judge evaluation")

        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for scenario in ALL_SCENARIOS:
                db_path = Path(tmpdir) / f"{scenario.name}.db"
                result = self.run_scenario(scenario, db_path)
                results.append(result)

        self.results = results
        return results

    def print_summary(self):
        """Print summary of all test results."""
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)

        total_extraction = 0
        total_retrieval = 0
        total_overall = 0

        for r in self.results:
            print(f"\n{r.scenario_name}:")
            print(f"  Extraction: {r.extraction_score:.2f}")
            print(f"  Retrieval:  {r.retrieval_score:.2f}")
            print(f"  Overall:    {r.overall_score:.2f}")
            if r.missing_nuggets:
                print(f"  Missing:    {len(r.missing_nuggets)} nuggets")

            total_extraction += r.extraction_score
            total_retrieval += r.retrieval_score
            total_overall += r.overall_score

        n = len(self.results)
        print(f"\n{'=' * 60}")
        print("AGGREGATE SCORES")
        print(f"{'=' * 60}")
        print(f"  Avg Extraction: {total_extraction / n:.2f}")
        print(f"  Avg Retrieval:  {total_retrieval / n:.2f}")
        print(f"  Avg Overall:    {total_overall / n:.2f}")

        # Overall assessment
        avg_overall = total_overall / n
        if avg_overall >= 0.8:
            verdict = "EXCELLENT - Memory system working well"
        elif avg_overall >= 0.6:
            verdict = "GOOD - Memory system functional with room for improvement"
        elif avg_overall >= 0.4:
            verdict = "FAIR - Memory system needs tuning"
        else:
            verdict = "POOR - Memory system needs significant work"

        print(f"\nVERDICT: {verdict}")
        print("=" * 60)

        return avg_overall


def main():
    """Main entry point."""
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Usage: export OPENAI_API_KEY='sk-...' && python examples/memory_e2e_test.py")
        sys.exit(1)

    # Run tests
    tester = MemoryE2ETest(api_key)
    tester.run_all_scenarios()
    avg_score = tester.print_summary()

    # Exit with appropriate code
    sys.exit(0 if avg_score >= 0.5 else 1)


if __name__ == "__main__":
    main()
