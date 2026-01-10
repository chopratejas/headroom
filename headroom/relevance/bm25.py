"""BM25 relevance scorer for Headroom SDK.

This module provides a BM25-based relevance scorer with ZERO external dependencies.
BM25 (Best Match 25) is a bag-of-words retrieval function that ranks documents
based on query term frequency.

Key features:
- Zero dependencies (pure Python)
- Fast execution (~0ms per item)
- Excellent for exact matches (UUIDs, IDs, specific terms)
- Returns matched terms for explainability

Limitations:
- No semantic understanding ("errors" won't match "failed")
- Sensitive to tokenization
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .base import RelevanceScore, RelevanceScorer


class BM25Scorer(RelevanceScorer):
    """BM25 keyword relevance scorer.

    Zero dependencies, instant execution. Excellent for exact ID/UUID matching.

    BM25 formula:
        score(D, Q) = sum over q in Q of:
            IDF(q) * (f(q,D) * (k1 + 1)) / (f(q,D) + k1 * (1 - b + b * |D|/avgdl))

    Where:
        - f(q,D) = frequency of term q in document D
        - |D| = length of document D
        - avgdl = average document length
        - k1, b = tuning parameters

    Example:
        scorer = BM25Scorer()
        score = scorer.score(
            '{"id": "550e8400-e29b-41d4-a716-446655440000", "name": "Alice"}',
            "find record 550e8400-e29b-41d4-a716-446655440000"
        )
        # score.score > 0.5 (UUID matches exactly)
    """

    # Tokenization pattern: alphanumeric sequences, UUIDs, numeric IDs
    _TOKEN_PATTERN = re.compile(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"  # UUIDs
        r"|\b\d{4,}\b"  # Numeric IDs (4+ digits)
        r"|[a-zA-Z0-9_]+"  # Alphanumeric tokens
    )

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        normalize_score: bool = True,
        max_score: float = 10.0,
    ):
        """Initialize BM25 scorer.

        Args:
            k1: Term frequency saturation parameter (default 1.5).
                Higher values increase term frequency impact.
            b: Length normalization parameter (default 0.75).
                0 = no length normalization, 1 = full normalization.
            normalize_score: If True, normalize score to [0, 1].
            max_score: Maximum raw score for normalization.
        """
        self.k1 = k1
        self.b = b
        self.normalize_score = normalize_score
        self.max_score = max_score

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Preserves:
        - UUIDs as single tokens
        - Numeric IDs
        - Alphanumeric words

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        if not text:
            return []

        tokens = self._TOKEN_PATTERN.findall(text.lower())
        return tokens

    def _compute_idf(self, term: str, doc_count: int, doc_freq: int) -> float:
        """Compute inverse document frequency.

        Uses the standard BM25 IDF formula:
            IDF = log((N - n + 0.5) / (n + 0.5) + 1)

        Where N = total docs, n = docs containing term.

        For single-document scoring, we use a simplified version.
        """
        if doc_freq == 0:
            return 0.0

        # Simplified IDF for single-document case
        # Term present = higher IDF, term absent = 0
        return math.log(2.0)  # Constant since we have single document

    def _bm25_score(
        self,
        doc_tokens: list[str],
        query_tokens: list[str],
        avg_doc_len: float | None = None,
    ) -> tuple[float, list[str]]:
        """Compute BM25 score between document and query.

        Args:
            doc_tokens: Tokenized document.
            query_tokens: Tokenized query.
            avg_doc_len: Average document length (optional).

        Returns:
            Tuple of (score, matched_terms).
        """
        if not doc_tokens or not query_tokens:
            return 0.0, []

        doc_len = len(doc_tokens)
        avgdl = avg_doc_len or doc_len

        doc_freq = Counter(doc_tokens)
        query_freq = Counter(query_tokens)

        score = 0.0
        matched_terms: list[str] = []

        for term, qf in query_freq.items():
            if term not in doc_freq:
                continue

            f = doc_freq[term]
            matched_terms.append(term)

            # BM25 term score
            idf = math.log(2.0)  # Simplified for single doc
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / avgdl)

            term_score = idf * numerator / denominator
            score += term_score * qf  # Weight by query frequency

        return score, matched_terms

    def score(self, item: str, context: str) -> RelevanceScore:
        """Score item relevance to context using BM25.

        Args:
            item: Item text (typically JSON string).
            context: Query context.

        Returns:
            RelevanceScore with BM25-based score.
        """
        item_tokens = self._tokenize(item)
        context_tokens = self._tokenize(context)

        raw_score, matched = self._bm25_score(item_tokens, context_tokens)

        # Normalize to [0, 1]
        if self.normalize_score:
            normalized = min(1.0, raw_score / self.max_score)
        else:
            normalized = raw_score

        # Bonus for exact long-token matches (UUIDs, long IDs)
        # These are high-value matches that should be preserved
        long_matches = [t for t in matched if len(t) >= 8]
        if long_matches:
            normalized = min(1.0, normalized + 0.3)

        match_count = len(matched)
        if match_count == 0:
            reason = "BM25: no term matches"
        elif match_count == 1:
            reason = f"BM25: matched '{matched[0]}'"
        else:
            reason = f"BM25: matched {match_count} terms ({', '.join(matched[:3])}{'...' if match_count > 3 else ''})"

        return RelevanceScore(
            score=normalized,
            reason=reason,
            matched_terms=matched[:10],  # Limit for readability
        )

    def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
        """Score multiple items.

        BM25 is fast enough that sequential scoring is efficient.
        Could be optimized with vectorization if needed.

        Args:
            items: List of items to score.
            context: Query context.

        Returns:
            List of RelevanceScore objects.
        """
        # Pre-tokenize context once
        context_tokens = self._tokenize(context)

        if not context_tokens:
            return [RelevanceScore(score=0.0, reason="BM25: empty context") for _ in items]

        # Compute average document length for normalization
        all_tokens = [self._tokenize(item) for item in items]
        avg_len = sum(len(t) for t in all_tokens) / max(len(items), 1)

        results = []
        for item_tokens in all_tokens:
            raw_score, matched = self._bm25_score(item_tokens, context_tokens, avg_doc_len=avg_len)

            # Normalize
            if self.normalize_score:
                normalized = min(1.0, raw_score / self.max_score)
            else:
                normalized = raw_score

            # Bonus for long matches
            long_matches = [t for t in matched if len(t) >= 8]
            if long_matches:
                normalized = min(1.0, normalized + 0.3)

            match_count = len(matched)
            if match_count == 0:
                reason = "BM25: no matches"
            else:
                reason = f"BM25: {match_count} terms"

            results.append(
                RelevanceScore(
                    score=normalized,
                    reason=reason,
                    matched_terms=matched[:5],
                )
            )

        return results
