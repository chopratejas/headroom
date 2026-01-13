"""Content type detection for multi-format compression.

This module detects the type of tool output content to route it to the
appropriate compressor. SmartCrusher handles JSON arrays, but coding tasks
produce many other formats that need specialized handling.

Supported content types:
- JSON_ARRAY: Structured JSON data (existing SmartCrusher)
- SOURCE_CODE: Python, JavaScript, TypeScript, Go, etc.
- SEARCH_RESULTS: grep/ripgrep output (file:line:content)
- BUILD_OUTPUT: Compiler, test, lint logs
- GIT_DIFF: Unified diff format
- PLAIN_TEXT: Generic text (fallback)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Types of content that can be compressed."""

    JSON_ARRAY = "json_array"  # Existing SmartCrusher handles this
    SOURCE_CODE = "source_code"  # Python, JS, TS, Go, Rust, etc.
    SEARCH_RESULTS = "search"  # grep/ripgrep output
    BUILD_OUTPUT = "build"  # Compiler, test, lint logs
    GIT_DIFF = "diff"  # Unified diff format
    PLAIN_TEXT = "text"  # Fallback


@dataclass
class DetectionResult:
    """Result of content type detection."""

    content_type: ContentType
    confidence: float  # 0.0 to 1.0
    metadata: dict  # Type-specific metadata (e.g., language for code)


# Patterns for detection
_SEARCH_RESULT_PATTERN = re.compile(
    r"^[^\s:]+:\d+:"  # file:line: format (grep -n style)
)

_DIFF_HEADER_PATTERN = re.compile(r"^(diff --git|--- a/|@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@)")

_DIFF_CHANGE_PATTERN = re.compile(r"^[+-][^+-]")

# Code patterns by language
_CODE_PATTERNS = {
    "python": [
        re.compile(r"^\s*(def|class|import|from|async def)\s+\w+"),
        re.compile(r"^\s*@\w+"),  # decorators
        re.compile(r'^\s*"""'),  # docstrings
        re.compile(r"^\s*if __name__\s*=="),
    ],
    "javascript": [
        re.compile(r"^\s*(function|const|let|var|class|import|export)\s+"),
        re.compile(r"^\s*(async\s+function|=>\s*\{)"),
        re.compile(r"^\s*module\.exports"),
    ],
    "typescript": [
        re.compile(r"^\s*(interface|type|enum|namespace)\s+\w+"),
        re.compile(r":\s*(string|number|boolean|any|void)\b"),
    ],
    "go": [
        re.compile(r"^\s*(func|type|package|import)\s+"),
        re.compile(r"^\s*func\s+\([^)]+\)\s+\w+"),  # method
    ],
    "rust": [
        re.compile(r"^\s*(fn|struct|enum|impl|mod|use|pub)\s+"),
        re.compile(r"^\s*#\["),  # attributes
    ],
    "java": [
        re.compile(r"^\s*(public|private|protected)\s+(class|interface|enum)"),
        re.compile(r"^\s*@\w+"),  # annotations
        re.compile(r"^\s*package\s+[\w.]+;"),
    ],
}

# Log/build output patterns
_LOG_PATTERNS = [
    re.compile(r"\b(ERROR|FAIL|FAILED|FATAL|CRITICAL)\b", re.IGNORECASE),
    re.compile(r"\b(WARN|WARNING)\b", re.IGNORECASE),
    re.compile(r"\b(INFO|DEBUG|TRACE)\b", re.IGNORECASE),
    re.compile(r"^\s*\d{4}-\d{2}-\d{2}"),  # timestamp
    re.compile(r"^\s*\[\d{2}:\d{2}:\d{2}\]"),  # time format
    re.compile(r"^={3,}|^-{3,}"),  # separators
    re.compile(r"^\s*PASSED|^\s*FAILED|^\s*SKIPPED"),  # test results
    re.compile(r"^npm ERR!|^yarn error|^cargo error"),  # build tools
    re.compile(r"Traceback \(most recent call last\)"),  # Python traceback
    re.compile(r"^\s*at\s+[\w.$]+\("),  # JS/Java stack trace
]


def detect_content_type(content: str) -> DetectionResult:
    """Detect the type of content for appropriate compression.

    Args:
        content: The content to analyze.

    Returns:
        DetectionResult with type, confidence, and metadata.

    Examples:
        >>> result = detect_content_type('[{"id": 1}, {"id": 2}]')
        >>> result.content_type
        ContentType.JSON_ARRAY

        >>> result = detect_content_type('src/main.py:42:def process():')
        >>> result.content_type
        ContentType.SEARCH_RESULTS
    """
    if not content or not content.strip():
        return DetectionResult(ContentType.PLAIN_TEXT, 0.0, {})

    # 1. Try JSON first (highest priority for SmartCrusher compatibility)
    json_result = _try_detect_json(content)
    if json_result:
        return json_result

    # 2. Check for diff (very distinctive patterns)
    diff_result = _try_detect_diff(content)
    if diff_result and diff_result.confidence >= 0.7:
        return diff_result

    # 3. Check for search results (file:line: format)
    search_result = _try_detect_search(content)
    if search_result and search_result.confidence >= 0.6:
        return search_result

    # 4. Check for build/log output
    log_result = _try_detect_log(content)
    if log_result and log_result.confidence >= 0.5:
        return log_result

    # 5. Check for source code
    code_result = _try_detect_code(content)
    if code_result and code_result.confidence >= 0.5:
        return code_result

    # 6. Fallback to plain text
    return DetectionResult(ContentType.PLAIN_TEXT, 0.5, {})


def _try_detect_json(content: str) -> DetectionResult | None:
    """Try to detect JSON array content."""
    content = content.strip()

    # Quick check: must start with [ for array
    if not content.startswith("["):
        return None

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            # Check if it's a list of dicts (SmartCrusher compatible)
            if parsed and all(isinstance(item, dict) for item in parsed):
                return DetectionResult(
                    ContentType.JSON_ARRAY,
                    1.0,
                    {"item_count": len(parsed), "is_dict_array": True},
                )
            # It's a list but not of dicts
            return DetectionResult(
                ContentType.JSON_ARRAY,
                0.8,
                {"item_count": len(parsed), "is_dict_array": False},
            )
    except json.JSONDecodeError:
        pass

    return None


def _try_detect_diff(content: str) -> DetectionResult | None:
    """Try to detect git diff format."""
    lines = content.split("\n")[:50]  # Check first 50 lines

    header_matches = 0
    change_matches = 0

    for line in lines:
        if _DIFF_HEADER_PATTERN.match(line):
            header_matches += 1
        if _DIFF_CHANGE_PATTERN.match(line):
            change_matches += 1

    if header_matches == 0:
        return None

    # High confidence if we see diff headers
    confidence = min(1.0, 0.5 + (header_matches * 0.2) + (change_matches * 0.05))

    return DetectionResult(
        ContentType.GIT_DIFF,
        confidence,
        {"header_matches": header_matches, "change_lines": change_matches},
    )


def _try_detect_search(content: str) -> DetectionResult | None:
    """Try to detect grep/ripgrep search results."""
    lines = content.split("\n")[:100]  # Check first 100 lines
    if not lines:
        return None

    matching_lines = 0
    for line in lines:
        if line.strip() and _SEARCH_RESULT_PATTERN.match(line):
            matching_lines += 1

    if matching_lines == 0:
        return None

    # Calculate confidence based on proportion of matching lines
    non_empty_lines = sum(1 for line in lines if line.strip())
    if non_empty_lines == 0:
        return None

    ratio = matching_lines / non_empty_lines

    # Need at least 30% of lines to match the pattern
    if ratio < 0.3:
        return None

    confidence = min(1.0, 0.4 + (ratio * 0.6))

    return DetectionResult(
        ContentType.SEARCH_RESULTS,
        confidence,
        {"matching_lines": matching_lines, "total_lines": non_empty_lines},
    )


def _try_detect_log(content: str) -> DetectionResult | None:
    """Try to detect build/log output."""
    lines = content.split("\n")[:200]  # Check first 200 lines
    if not lines:
        return None

    pattern_matches = 0
    error_matches = 0

    for line in lines:
        for i, pattern in enumerate(_LOG_PATTERNS):
            if pattern.search(line):
                pattern_matches += 1
                if i < 2:  # ERROR or WARN patterns
                    error_matches += 1
                break  # One pattern per line is enough

    if pattern_matches == 0:
        return None

    non_empty_lines = sum(1 for line in lines if line.strip())
    if non_empty_lines == 0:
        return None

    ratio = pattern_matches / non_empty_lines

    # Need at least 10% of lines to match log patterns
    if ratio < 0.1:
        return None

    confidence = min(1.0, 0.3 + (ratio * 0.5) + (error_matches * 0.05))

    return DetectionResult(
        ContentType.BUILD_OUTPUT,
        confidence,
        {
            "pattern_matches": pattern_matches,
            "error_matches": error_matches,
            "total_lines": non_empty_lines,
        },
    )


def _try_detect_code(content: str) -> DetectionResult | None:
    """Try to detect source code and identify language."""
    lines = content.split("\n")[:100]  # Check first 100 lines
    if not lines:
        return None

    language_scores: dict[str, int] = {}

    for line in lines:
        for lang, patterns in _CODE_PATTERNS.items():
            for pattern in patterns:
                if pattern.match(line):
                    language_scores[lang] = language_scores.get(lang, 0) + 1
                    break  # One pattern per language per line

    if not language_scores:
        return None

    # Find best matching language
    best_lang = max(language_scores, key=lambda k: language_scores[k])
    best_score = language_scores[best_lang]

    # Need at least 3 pattern matches to be confident
    if best_score < 3:
        return None

    non_empty_lines = sum(1 for line in lines if line.strip())
    ratio = best_score / max(non_empty_lines, 1)

    confidence = min(1.0, 0.4 + (ratio * 0.4) + (best_score * 0.02))

    return DetectionResult(
        ContentType.SOURCE_CODE,
        confidence,
        {"language": best_lang, "pattern_matches": best_score},
    )


def is_json_array_of_dicts(content: str) -> bool:
    """Quick check if content is a JSON array of dictionaries.

    This is the format SmartCrusher can handle natively.

    Args:
        content: The content to check.

    Returns:
        True if content is a JSON array where all items are dicts.
    """
    result = detect_content_type(content)
    return result.content_type == ContentType.JSON_ARRAY and result.metadata.get(
        "is_dict_array", False
    )
