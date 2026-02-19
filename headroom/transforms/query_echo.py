"""Query Echo — re-injects the user's question after compressed tool outputs.

After Headroom compresses tool outputs, the user's original question may be
thousands of tokens away from where the LLM generates its answer. Attention
to the question decays with distance (the "lost in the middle" problem).

Query Echo solves this by appending a brief reminder of the user's question
after the last compressed content block. This:

1. Improves answer quality from compressed data (fresh attention on the question)
2. Makes CCR retrieval triggers more accurate (LLM is more aware of what's missing)
3. Feeds cleaner signals to TOIN (more accurate retrievals → better learning)

The echo is:
- Cache-safe: always placed after the cache boundary (in the "new tokens" region)
- Compression-ratio-proportional: only triggers when compression was >30%
- Provider-agnostic: appends to content strings, works for all providers
- Cheap: ~50 tokens overhead, vs thousands of tokens saved by compression
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def extract_user_query(messages: list[dict]) -> str:
    """Extract the last substantive user question from messages.

    Scans backwards through messages to find the last user message
    that contains a real question (not just "yes", "continue", etc.).

    Works for both Anthropic and OpenAI message formats.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", "")

        # Anthropic: content can be a list of blocks
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text", "")).strip()
                    if len(text) > 10:  # Skip trivial messages
                        return text
            continue

        # OpenAI/generic: content is a string
        if isinstance(content, str):
            text = content.strip()
            if len(text) > 10:
                return text

    return ""


def inject_query_echo(
    messages: list[dict],
    query: str,
    tokens_saved: int,
    original_tokens: int,
    min_compression_ratio: float = 0.3,
    max_query_chars: int = 200,
) -> bool:
    """Inject a query reminder after the last compressed tool output.

    Args:
        messages: The optimized messages list (will be modified in-place).
        query: The user's question to echo.
        tokens_saved: Tokens saved by compression.
        original_tokens: Original token count before compression.
        min_compression_ratio: Minimum compression ratio to trigger echo (0.0–1.0).
        max_query_chars: Maximum characters of query to include in echo.

    Returns:
        True if echo was injected, False if skipped.
    """
    if not query or original_tokens == 0:
        return False

    compression_ratio = tokens_saved / original_tokens
    if compression_ratio < min_compression_ratio:
        return False

    # Truncate and sanitize query
    safe_query = query[:max_query_chars].replace("\n", " ").replace('"', "'").strip()
    if not safe_query:
        return False

    echo = f"\n\n[Recall: User asked: '{safe_query}']"

    # Walk backwards through messages, find last compressed content
    for msg in reversed(messages):
        content = msg.get("content", "")

        # String content (OpenAI tool messages, Anthropic string content)
        if isinstance(content, str) and "compressed" in content:
            msg["content"] = content + echo
            logger.debug(
                "Query echo injected (ratio=%.1f%%, query=%s...)",
                compression_ratio * 100,
                safe_query[:50],
            )
            return True

        # List of content blocks (Anthropic format)
        if isinstance(content, list):
            for block in reversed(content):
                if not isinstance(block, dict):
                    continue
                block_content = block.get("content", "")
                if isinstance(block_content, str) and "compressed" in block_content:
                    block["content"] = block_content + echo
                    logger.debug(
                        "Query echo injected in content block (ratio=%.1f%%)",
                        compression_ratio * 100,
                    )
                    return True

    return False
