"""Session analysis via LLM — replaces all regex/heuristic analysis.

Pipeline: Scanner (events) → Digest Builder → LLM → Recommendations

No regex patterns, no static lookback windows, no hardcoded heuristics.
A single LLM call understands the full conversation context and produces
structured recommendations for CLAUDE.md / MEMORY.md.

Supports any LLM provider via LiteLLM: Anthropic, OpenAI, Google, Bedrock,
Ollama, and 100+ others. Auto-detects the best available model from env vars.
"""

from __future__ import annotations

import json
import logging
import os

from .models import (
    AnalysisResult,
    ProjectInfo,
    Recommendation,
    RecommendationTarget,
    SessionData,
    SessionEvent,
    ToolCall,
)

logger = logging.getLogger(__name__)

# Default models by provider (checked in order)
_MODEL_DEFAULTS: list[tuple[str, str]] = [
    ("ANTHROPIC_API_KEY", "claude-sonnet-4-6"),
    ("OPENAI_API_KEY", "gpt-4o"),
    ("GEMINI_API_KEY", "gemini/gemini-2.0-flash"),
]

_MAX_DIGEST_TOKENS = 80_000  # Budget for the digest (leave room for prompt + output)


def _detect_default_model() -> str:
    """Pick the best available model based on which API keys are set."""
    for env_var, model in _MODEL_DEFAULTS:
        if os.environ.get(env_var):
            return model
    raise RuntimeError(
        "No LLM API key found. headroom learn needs one of:\n"
        "  export ANTHROPIC_API_KEY=sk-ant-...   → uses claude-sonnet-4-6\n"
        "  export OPENAI_API_KEY=sk-...          → uses gpt-4o\n"
        "  export GEMINI_API_KEY=...             → uses gemini-2.0-flash\n"
        "Or specify a model directly: headroom learn --model <litellm-model-name>"
    )


class SessionAnalyzer:
    """Analyzes session data via LLM to produce actionable recommendations.

    Uses LiteLLM for provider-agnostic access to 100+ models.
    Auto-detects the best available model from environment API keys.
    """

    def __init__(self, model: str | None = None):
        self.model = model

    def analyze(
        self, project: ProjectInfo, sessions: list[SessionData]
    ) -> AnalysisResult:
        """Analyze sessions and produce recommendations via LLM."""
        all_calls = [tc for s in sessions for tc in s.tool_calls]
        failed_calls = [tc for tc in all_calls if tc.is_error]

        result = AnalysisResult(
            project=project,
            total_sessions=len(sessions),
            total_calls=len(all_calls),
            total_failures=len(failed_calls),
        )

        if not failed_calls and not any(s.events for s in sessions):
            return result

        # Build compact digest of all sessions
        digest = _build_digest(project, sessions)

        # Resolve model (auto-detect if not specified)
        model = self.model or _detect_default_model()

        # Call LLM for analysis
        try:
            raw = _call_llm(digest, model)
            result.recommendations = _parse_llm_response(raw)
        except Exception as e:
            logger.warning("LLM analysis failed: %s", e)
            # Return result with stats but no recommendations

        return result


# =============================================================================
# Digest Builder — compact text representation of session events
# =============================================================================


def _build_digest(project: ProjectInfo, sessions: list[SessionData]) -> str:
    """Build a token-efficient text digest of all session events.

    The digest includes:
    - Project context
    - Per-session summaries with condensed event streams
    - Error outputs (truncated), success indicators, user messages
    """
    lines: list[str] = []

    # Project header
    lines.append(f"Project: {project.name} ({project.project_path})")
    total_calls = sum(len(s.tool_calls) for s in sessions)
    total_failures = sum(s.failure_count for s in sessions)
    total_tokens_in = sum(s.total_input_tokens for s in sessions)
    total_tokens_out = sum(s.total_output_tokens for s in sessions)
    lines.append(
        f"Total: {len(sessions)} sessions, {total_calls} tool calls, "
        f"{total_failures} failures ({total_failures / total_calls:.1%})"
        if total_calls
        else f"Total: {len(sessions)} sessions, 0 tool calls"
    )
    if total_tokens_in:
        lines.append(f"Tokens used: {total_tokens_in:,} in / {total_tokens_out:,} out")
    lines.append("")

    # Budget tracking — stop adding events when we approach the limit
    # Rough estimate: 4 chars per token
    char_budget = _MAX_DIGEST_TOKENS * 4
    chars_used = sum(len(ln) for ln in lines)

    for session in sessions:
        if chars_used > char_budget:
            lines.append(f"... (remaining {len(sessions) - sessions.index(session)} sessions truncated)")
            break

        session_header = (
            f"=== Session {session.session_id[:12]} "
            f"({len(session.tool_calls)} calls, {session.failure_count} failures"
        )
        if session.total_input_tokens:
            session_header += f", {session.total_input_tokens:,} input tokens"
        session_header += ") ==="
        lines.append(session_header)
        chars_used += len(session_header)

        # Use events if available (richer context), fall back to tool_calls
        if session.events:
            for event in session.events:
                if chars_used > char_budget:
                    lines.append("  ... (remaining events truncated)")
                    break
                event_line = _format_event(event)
                if event_line:
                    lines.append(event_line)
                    chars_used += len(event_line)
        else:
            for tc in session.tool_calls:
                if chars_used > char_budget:
                    lines.append("  ... (remaining calls truncated)")
                    break
                tc_line = _format_tool_call(tc)
                lines.append(tc_line)
                chars_used += len(tc_line)

        lines.append("")

    return "\n".join(lines)


def _format_event(event: SessionEvent) -> str | None:
    """Format a single event into a compact digest line."""

    if event.type == "tool_call" and event.tool_call:
        return _format_tool_call(event.tool_call)

    if event.type == "user_message" and event.text.strip():
        text = event.text.strip()[:300]
        return f"  [{event.msg_index}] USER: \"{text}\""

    if event.type == "interruption":
        return f"  [{event.msg_index}] INTERRUPTED: {event.text[:150]}"

    if event.type == "agent_summary":
        return (
            f"  [{event.msg_index}] SUBAGENT: {event.agent_tool_count} tool calls, "
            f"{event.agent_tokens:,} tokens, {event.agent_duration_ms / 1000:.1f}s "
            f"— prompt: \"{event.agent_prompt[:100]}\""
        )

    return None


def _format_tool_call(tc: ToolCall) -> str:
    """Format a single tool call into a compact digest line."""
    status = "ERROR" if tc.is_error else "OK"
    error_cat = f"({tc.error_category.value})" if tc.is_error else ""

    # Input summary
    input_str = tc.input_summary[:120]

    if tc.is_error:
        # Include truncated error output for failures
        output_preview = tc.output[:200].replace("\n", " ").strip()
        return f"  [{tc.msg_index}] {tc.name}: {input_str} → {status}{error_cat}: {output_preview}"
    else:
        # Just indicate success with size
        size = f"({tc.output_bytes} bytes)" if tc.output_bytes > 0 else ""
        return f"  [{tc.msg_index}] {tc.name}: {input_str} → {status} {size}"


# =============================================================================
# LLM Call — Sonnet 4.6 with structured output
# =============================================================================

_SYSTEM_PROMPT = """\
You are an expert at analyzing coding agent sessions to extract actionable patterns.

You will receive a digest of tool call sessions from a coding agent (Claude Code, Codex, etc.).
Your job is to identify patterns that, if documented, would PREVENT TOKEN WASTE in future sessions.

Focus on:
1. **Environment rules** — what runtime commands work vs fail (e.g., "use uv run python, not python3")
2. **File structure facts** — known large files, correct paths, search scopes
3. **User preferences** — things the user corrected, rejected, or explicitly requested
4. **Failure patterns** — repeated failures that could be prevented with upfront knowledge
5. **Workflow rules** — subagent guidance, command execution preferences
6. **Token waste hotspots** — patterns that waste the most tokens (re-reads, wrong paths, retries)

Rules:
- Only include patterns with CLEAR evidence from the data (2+ occurrences or explicit user direction)
- Every recommendation must be specific and actionable (not "be careful" but "use X instead of Y")
- Estimate tokens saved per recommendation (how many tokens would be saved per session if this rule existed)
- Separate stable project facts (CONTEXT_FILE) from evolving preferences (MEMORY_FILE)
- CONTEXT_FILE rules go in CLAUDE.md/AGENTS.md — they are project-level, stable facts
- MEMORY_FILE rules go in MEMORY.md — they are session-level, evolving preferences
- Keep recommendations concise — each should be 1-3 lines of markdown
- Do NOT produce tautological rules (e.g., "use python3 not python3")
- Do NOT produce rules about things that only happened once (transient errors)

Return ONLY valid JSON matching this schema — no other text:
{
  "context_file_rules": [
    {
      "section": "string — section heading (e.g., 'Environment', 'File Paths', 'Commands')",
      "content": "string — markdown content, 1-3 bullet points",
      "estimated_tokens_saved": "integer — tokens saved per session if rule existed",
      "evidence_count": "integer — number of occurrences supporting this rule"
    }
  ],
  "memory_file_rules": [
    {
      "section": "string — section heading",
      "content": "string — markdown content, 1-3 bullet points",
      "estimated_tokens_saved": "integer",
      "evidence_count": "integer"
    }
  ]
}
"""


def _call_llm(digest: str, model: str) -> dict:
    """Call LLM with the session digest and return parsed JSON.

    Uses LiteLLM for provider-agnostic access. The model string determines
    the provider: "claude-*" → Anthropic, "gpt-*" → OpenAI, "gemini/*" → Google, etc.
    """
    import litellm

    # Suppress LiteLLM's verbose logging
    litellm.suppress_debug_info = True

    # For Anthropic models, bypass ANTHROPIC_BASE_URL which may point to
    # the user's local headroom proxy
    api_base = None
    if model.startswith("claude"):
        api_base = "https://api.anthropic.com"

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze these coding agent sessions and return JSON recommendations:\n\n"
                    + digest
                ),
            },
        ],
        max_tokens=4096,
        api_base=api_base,
    )

    # Extract text from response
    text = response.choices[0].message.content or ""

    # Parse JSON — handle both raw JSON and ```json fenced blocks
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    result: dict = json.loads(text)
    return result


# =============================================================================
# Response Parser — LLM JSON → Recommendation list
# =============================================================================


def _parse_llm_response(raw: dict) -> list[Recommendation]:
    """Convert LLM structured output into Recommendation objects."""
    recommendations: list[Recommendation] = []

    for rule in raw.get("context_file_rules", []):
        if not isinstance(rule, dict):
            continue
        section = rule.get("section", "").strip()
        content = rule.get("content", "").strip()
        if not section or not content:
            continue
        recommendations.append(
            Recommendation(
                target=RecommendationTarget.CONTEXT_FILE,
                section=section,
                content=content,
                confidence=0.9,
                evidence_count=_safe_int(rule.get("evidence_count", 1)),
                estimated_tokens_saved=_safe_int(rule.get("estimated_tokens_saved", 0)),
            )
        )

    for rule in raw.get("memory_file_rules", []):
        if not isinstance(rule, dict):
            continue
        section = rule.get("section", "").strip()
        content = rule.get("content", "").strip()
        if not section or not content:
            continue
        recommendations.append(
            Recommendation(
                target=RecommendationTarget.MEMORY_FILE,
                section=section,
                content=content,
                confidence=0.7,
                evidence_count=_safe_int(rule.get("evidence_count", 1)),
                estimated_tokens_saved=_safe_int(rule.get("estimated_tokens_saved", 0)),
            )
        )

    # Sort by estimated token savings
    recommendations.sort(key=lambda r: r.estimated_tokens_saved, reverse=True)

    return recommendations


def _safe_int(val: object) -> int:
    """Safely convert a value to int."""
    if isinstance(val, int):
        return val
    if isinstance(val, (float, str)):
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0
    return 0


# =============================================================================
# Legacy compatibility alias
# =============================================================================


class FailureAnalyzer:
    """Legacy alias for SessionAnalyzer — used by existing CLI code."""

    def __init__(self) -> None:
        self._analyzer = SessionAnalyzer()

    def analyze(
        self, project: ProjectInfo, sessions: list[SessionData]
    ) -> AnalysisResult:
        return self._analyzer.analyze(project, sessions)
