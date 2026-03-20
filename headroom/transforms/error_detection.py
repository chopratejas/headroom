"""Centralized error/importance detection for all transforms.

Design principle: Keywords serve as a FALLBACK safety net for error detection.
When TOIN field semantics are available, they take priority over keywords.

This module prevents each transform from maintaining its own hardcoded keyword
list, ensuring consistency and a single place to evolve detection logic.
"""

from __future__ import annotations

import re

# ─── Canonical keyword sets ──────────────────────────────────────────────────
# These are the FALLBACK when TOIN semantics aren't available yet.
# They are intentionally broad to avoid missing errors.

ERROR_KEYWORDS: frozenset[str] = frozenset(
    {
        "error",
        "exception",
        "failed",
        "failure",
        "critical",
        "fatal",
        "crash",
        "panic",
        "abort",
        "timeout",
        "denied",
        "rejected",
    }
)

# Broader importance keywords (for line-level scoring, not item preservation)
IMPORTANCE_KEYWORDS: frozenset[str] = frozenset(
    ERROR_KEYWORDS
    | {
        "warning",
        "warn",
        "todo",
        "fixme",
        "hack",
        "xxx",
        "bug",
        "fix",
        "important",
        "note",
    }
)

# Security-related keywords (for diff/search prioritization)
SECURITY_KEYWORDS: frozenset[str] = frozenset(
    {
        "security",
        "auth",
        "password",
        "secret",
        "token",
    }
)

# ─── Compiled patterns (for line-level matching) ────────────────────────────
# Shared across text_compressor, diff_compressor, search_compressor

ERROR_PATTERN: re.Pattern[str] = re.compile(
    r"\b(error|exception|fail(?:ed|ure)?|fatal|critical|crash|panic)\b",
    re.IGNORECASE,
)

WARNING_PATTERN: re.Pattern[str] = re.compile(
    r"\b(warn(?:ing)?)\b",
    re.IGNORECASE,
)

IMPORTANCE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(important|note|todo|fixme|hack|xxx|bug|fix)\b",
    re.IGNORECASE,
)

SECURITY_PATTERN: re.Pattern[str] = re.compile(
    r"\b(security|auth|password|secret|token)\b",
    re.IGNORECASE,
)

# Pre-built pattern lists for each compressor context
PRIORITY_PATTERNS_SEARCH: list[re.Pattern[str]] = [
    ERROR_PATTERN,
    WARNING_PATTERN,
    IMPORTANCE_PATTERN,
]

PRIORITY_PATTERNS_DIFF: list[re.Pattern[str]] = [
    ERROR_PATTERN,
    IMPORTANCE_PATTERN,
    SECURITY_PATTERN,
]

PRIORITY_PATTERNS_TEXT: list[re.Pattern[str]] = [
    ERROR_PATTERN,
    IMPORTANCE_PATTERN,
    re.compile(r"^#+\s"),  # Markdown headers
    re.compile(r"^\*\*"),  # Bold text
    re.compile(r"^>\s"),  # Quotes
]

# ─── Quick check for message-level error indicators ─────────────────────────
# Used by intelligent_context.py for message signature creation

ERROR_INDICATOR_KEYWORDS: tuple[str, ...] = (
    "error",
    "fail",
    "exception",
    "traceback",
    "fatal",
    "panic",
    "crash",
)


def content_has_error_indicators(text: str) -> bool:
    """Check if text contains error indicators (fast keyword check).

    Used for message signature creation and quick triage, NOT for
    compression decisions (those should use TOIN when available).
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in ERROR_INDICATOR_KEYWORDS)
