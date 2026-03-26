"""Protect workflow/custom XML tags from text compression.

LLM workflows use XML-style tags (<system-reminder>, <tool_call>, <thinking>)
as structural markers.  Text compressors (Kompress) treat these as
droppable noise and silently remove them, breaking downstream tools.

This module detects custom tags (anything NOT standard HTML), replaces entire
blocks with placeholders before compression, and restores them after.

Standard HTML tags (<div>, <p>, <span>) are left alone — HTMLExtractor
handles those via trafilatura.

Usage:
    from headroom.transforms.tag_protector import protect_tags, restore_tags

    cleaned, protected = protect_tags(text)
    compressed = kompress.compress(cleaned)
    result = restore_tags(compressed, protected)
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML5 Living Standard element names (https://html.spec.whatwg.org/)
# These are the ONLY tags treated as "HTML".  Everything else is a custom
# workflow tag and gets protected from compression.
# ---------------------------------------------------------------------------
KNOWN_HTML_TAGS: frozenset[str] = frozenset(
    {
        # Main root
        "html",
        # Document metadata
        "base",
        "head",
        "link",
        "meta",
        "style",
        "title",
        # Sectioning root
        "body",
        # Content sectioning
        "address",
        "article",
        "aside",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "main",
        "nav",
        "section",
        "search",
        # Text content
        "blockquote",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "hr",
        "li",
        "menu",
        "ol",
        "p",
        "pre",
        "ul",
        # Inline text semantics
        "a",
        "abbr",
        "b",
        "bdi",
        "bdo",
        "br",
        "cite",
        "code",
        "data",
        "dfn",
        "em",
        "i",
        "kbd",
        "mark",
        "q",
        "rp",
        "rt",
        "ruby",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "time",
        "u",
        "var",
        "wbr",
        # Image and multimedia
        "area",
        "audio",
        "img",
        "map",
        "track",
        "video",
        # Embedded content
        "embed",
        "iframe",
        "object",
        "param",
        "picture",
        "portal",
        "source",
        # SVG and MathML
        "svg",
        "math",
        # Scripting
        "canvas",
        "noscript",
        "script",
        # Demarcating edits
        "del",
        "ins",
        # Table content
        "caption",
        "col",
        "colgroup",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        # Forms
        "button",
        "datalist",
        "fieldset",
        "form",
        "input",
        "label",
        "legend",
        "meter",
        "optgroup",
        "option",
        "output",
        "progress",
        "select",
        "textarea",
        # Interactive
        "details",
        "dialog",
        "summary",
        # Web Components
        "slot",
        "template",
    }
)

# Placeholder format — unlikely to appear in real content
_PLACEHOLDER_PREFIX = "{{HEADROOM_TAG_"
_PLACEHOLDER_SUFFIX = "}}"

# Match opening tags: <tag>, <tag attr="val">, and self-closing <tag/>
_OPEN_TAG_RE = re.compile(r"<([a-zA-Z_][\w.-]*)((?:\s+[^>]*?)?)(/?)>")

# Match closing tags: </tag>
_CLOSE_TAG_RE = re.compile(r"</([a-zA-Z_][\w.-]*)>")


def _is_html_tag(tag_name: str) -> bool:
    """Check if a tag name is a known HTML element (case-insensitive)."""
    return tag_name.lower() in KNOWN_HTML_TAGS


def protect_tags(
    text: str,
    compress_tagged_content: bool = False,
) -> tuple[str, list[tuple[str, str]]]:
    """Protect custom/workflow XML tags from text compression.

    Scans for XML-style tags, classifies each as HTML or custom.
    Custom tags (and optionally their content) are replaced with
    placeholders that survive compression.

    Args:
        text: Input text potentially containing XML tags.
        compress_tagged_content: If False (default), protect entire
            ``<tag>content</tag>`` block verbatim.  If True, only
            protect the tag markers; content between them can be
            compressed.

    Returns:
        Tuple of (cleaned_text, protected_blocks) where protected_blocks
        is a list of (placeholder, original_text) pairs for restoration.
    """
    if not text or "<" not in text:
        return text, []

    protected: list[tuple[str, str]] = []
    counter = 0

    if not compress_tagged_content:
        # Protect entire <tag>...</tag> blocks for custom tags.
        # Process from innermost out by iterating until stable.
        result = text
        changed = True
        max_iterations = 50  # Safety limit for deeply nested tags

        while changed and max_iterations > 0:
            changed = False
            max_iterations -= 1

            # Find blocks: <tag ...>...</tag> where tag is NOT known HTML
            # Use non-greedy match for content between tags
            pattern = re.compile(
                r"<([a-zA-Z_][\w.-]*)((?:\s+[^>]*?)?)>"
                r"(.*?)"
                r"</\1>",
                re.DOTALL,
            )

            for match in pattern.finditer(result):
                tag_name = match.group(1)
                if _is_html_tag(tag_name):
                    continue

                original_block = match.group(0)
                placeholder = f"{_PLACEHOLDER_PREFIX}{counter}{_PLACEHOLDER_SUFFIX}"
                protected.append((placeholder, original_block))
                result = result.replace(original_block, placeholder, 1)
                counter += 1
                changed = True
                break  # Restart iteration after each replacement

        # Also protect self-closing custom tags: <tag/>  <tag attr="x"/>
        for match in _OPEN_TAG_RE.finditer(result):
            tag_name = match.group(1)
            is_self_closing = match.group(3) == "/"
            if is_self_closing and not _is_html_tag(tag_name):
                original = match.group(0)
                if original in result:  # Not already replaced
                    placeholder = f"{_PLACEHOLDER_PREFIX}{counter}{_PLACEHOLDER_SUFFIX}"
                    protected.append((placeholder, original))
                    result = result.replace(original, placeholder, 1)
                    counter += 1

        return result, protected

    else:
        # compress_tagged_content=True: only protect tag markers, not content
        result = text

        # Protect opening custom tags
        for match in _OPEN_TAG_RE.finditer(text):
            tag_name = match.group(1)
            if not _is_html_tag(tag_name):
                original = match.group(0)
                placeholder = f"{_PLACEHOLDER_PREFIX}{counter}{_PLACEHOLDER_SUFFIX}"
                protected.append((placeholder, original))
                result = result.replace(original, placeholder, 1)
                counter += 1

        # Protect closing custom tags
        for match in _CLOSE_TAG_RE.finditer(text):
            tag_name = match.group(1)
            if not _is_html_tag(tag_name):
                original = match.group(0)
                if original in result:
                    placeholder = f"{_PLACEHOLDER_PREFIX}{counter}{_PLACEHOLDER_SUFFIX}"
                    protected.append((placeholder, original))
                    result = result.replace(original, placeholder, 1)
                    counter += 1

        return result, protected


def restore_tags(
    text: str,
    protected_blocks: list[tuple[str, str]],
) -> str:
    """Restore protected tag blocks after compression.

    Args:
        text: Compressed text with placeholders.
        protected_blocks: List of (placeholder, original_text) pairs
            from ``protect_tags()``.

    Returns:
        Text with all placeholders replaced by original blocks.
    """
    if not protected_blocks:
        return text

    result = text
    for placeholder, original in protected_blocks:
        if placeholder in result:
            result = result.replace(placeholder, original)
        else:
            # Placeholder was lost during compression — append the block
            logger.warning(
                "Tag placeholder lost during compression, appending: %s",
                original[:80],
            )
            result = result + "\n" + original

    return result
