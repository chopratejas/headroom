"""TextCrusher: fast deterministic extractive text compressor (Phase 2, #1171).

The cold-start-large latency catastrophe comes from running ModernBERT (kompress)
ONNX inference O(tokens) on the request thread. TextCrusher is a heuristic
*extractive* scorer -- it SELECTS the most salient original segments and drops
the rest, in one O(n) pass with no neural forward -- so large text compresses in
~milliseconds instead of minutes while keeping the kept text byte-verbatim.

It is intentionally simple and deterministic (no sampling, no model): each
segment is scored by recency, query relevance (BM25-lite over the user context),
and structural salience, near-duplicate segments are suppressed, and the
highest-scoring segments are kept (in original order) up to a target ratio.

This is the Python reference implementation. Quality is validated against the
kompress baseline via ``headroom/evals`` before it is defaulted on; a Rust port
behind the same interface is a later speed optimization.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# Split into sentence/line segments: on newlines, and on sentence terminators
# followed by whitespace. Keeps segments byte-faithful (we re-join the originals).
_SEG_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z0-9_]+")
# Tokens that signal a segment carries specific, hard-to-reconstruct information.
_SALIENT = re.compile(
    r"\b(?:error|exception|fail(?:ed|ure)?|warning|traceback|assert|"
    r"todo|fixme|null|none|true|false)\b|[0-9]|[A-Z]{2,}|[{}\[\]()=<>]|"
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_]"
)


@dataclass
class TextCrusherResult:
    compressed: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    kept_segments: int
    total_segments: int
    model_used: str = "text_crusher_v1"
    cache_key: str | None = None


@dataclass
class TextCrusherConfig:
    target_ratio: float = 0.5  # keep ~50% of characters by default
    w_recency: float = 1.0
    w_relevance: float = 2.0
    w_salience: float = 1.5
    min_segment_chars: int = 12  # very short segments rarely carry standalone info
    near_dup_threshold: float = 0.85  # Jaccard over word-shingles
    min_segments_for_crush: int = 6  # below this, passthrough (nothing to gain)


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _shingles(words: list[str], k: int = 3) -> frozenset[tuple[str, ...]]:
    if len(words) < k:
        return frozenset([tuple(words)]) if words else frozenset()
    return frozenset(tuple(words[i : i + k]) for i in range(len(words) - k + 1))


class TextCrusher:
    """Heuristic extractive compressor. ``compress`` returns a
    ``TextCrusherResult`` whose ``compressed`` text is a byte-verbatim subset of
    the kept input segments, joined in original order."""

    def __init__(self, config: TextCrusherConfig | None = None) -> None:
        self.config = config or TextCrusherConfig()

    def _passthrough(self, content: str, n_segments: int) -> TextCrusherResult:
        n = len(content.split())
        return TextCrusherResult(
            compressed=content,
            original_tokens=n,
            compressed_tokens=n,
            compression_ratio=1.0,
            kept_segments=n_segments,
            total_segments=n_segments,
        )

    def compress(
        self,
        content: str,
        context: str = "",
        target_ratio: float | None = None,
    ) -> TextCrusherResult:
        cfg = self.config
        ratio = cfg.target_ratio if target_ratio is None else target_ratio
        ratio = min(max(ratio, 0.05), 1.0)

        raw_segments = [s for s in _SEG_SPLIT.split(content) if s and s.strip()]
        if len(raw_segments) < cfg.min_segments_for_crush:
            return self._passthrough(content, len(raw_segments))

        total_chars = sum(len(s) for s in raw_segments)
        target_chars = int(total_chars * ratio)
        n = len(raw_segments)

        # Corpus document frequency for a BM25-lite relevance term weighting.
        seg_tokens = [_tokens(s) for s in raw_segments]
        df: Counter[str] = Counter()
        for toks in seg_tokens:
            for term in set(toks):
                df[term] += 1
        query_terms = set(_tokens(context))

        def idf(term: str) -> float:
            # Smoothed inverse document frequency.
            return math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))

        scores: list[float] = []
        for i, (seg, toks) in enumerate(zip(raw_segments, seg_tokens, strict=True)):
            recency = (i + 1) / n  # later segments slightly favored (recent context)
            # Relevance: idf-weighted overlap with the user query terms.
            relevance = 0.0
            if query_terms and toks:
                tf = Counter(toks)
                matched = set(toks) & query_terms
                hit = sum(tf[t] * idf(t) for t in matched)
                relevance = hit / (len(toks) + 1)
            # Salience: fraction of WORDS carrying specific, hard-to-reconstruct
            # information (errors, numbers, identifiers, ALLCAPS). Word-based and
            # bounded to [0, 1] so a single number cannot dominate the score.
            words = seg.split()
            salient_words = sum(1 for w in words if _SALIENT.search(w))
            salience = salient_words / (len(words) + 1)
            score = (
                cfg.w_recency * recency
                + cfg.w_relevance * relevance
                + cfg.w_salience * salience
            )
            if len(seg) < cfg.min_segment_chars:
                score *= 0.25  # de-prioritize tiny fragments
            scores.append(score)

        # Greedily keep highest-scoring segments until the target character
        # budget is reached, suppressing near-duplicates via a global shingle
        # index: a candidate whose word-shingles are mostly already covered by
        # kept segments is skipped. This is O(total shingles) rather than
        # O(kept) per candidate, so it stays linear on huge inputs.
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        kept: set[int] = set()
        seen_shingles: set[tuple[str, ...]] = set()
        kept_chars = 0
        for i in order:
            if kept_chars >= target_chars:
                break
            sh = _shingles(seg_tokens[i])
            if sh:
                covered = sum(1 for s in sh if s in seen_shingles) / len(sh)
                if covered >= cfg.near_dup_threshold:
                    continue  # near-duplicate: most shingles already kept
            kept.add(i)
            seen_shingles |= sh
            kept_chars += len(raw_segments[i])

        if not kept:
            return self._passthrough(content, n)

        compressed = "\n".join(raw_segments[i] for i in sorted(kept))
        orig_tok = len(content.split())
        comp_tok = len(compressed.split())
        return TextCrusherResult(
            compressed=compressed,
            original_tokens=orig_tok,
            compressed_tokens=comp_tok,
            compression_ratio=(comp_tok / orig_tok if orig_tok else 1.0),
            kept_segments=len(kept),
            total_segments=n,
        )
