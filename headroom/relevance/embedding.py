"""Embedding-based relevance scorer for Headroom SDK.

This module provides semantic relevance scoring using sentence embeddings.
Requires the optional `sentence-transformers` dependency.

Key features:
- Semantic understanding ("errors" matches "failed", "issues")
- Handles paraphrases and synonyms
- Uses lightweight all-MiniLM-L6-v2 model by default (22M params)
- Batch encoding for efficiency

Install with: pip install headroom[relevance]

Limitations:
- Requires ~500MB for model download on first use
- ~5-10ms per batch (slower than BM25)
- May miss exact ID matches that BM25 catches
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import RelevanceScore, RelevanceScorer

# numpy is an optional dependency - import lazily
_numpy = None


def _get_numpy():
    """Lazily import numpy."""
    global _numpy
    if _numpy is None:
        try:
            import numpy as np

            _numpy = np
        except ImportError as e:
            raise ImportError(
                "numpy is required for EmbeddingScorer. "
                "Install with: pip install headroom[relevance]"
            ) from e
    return _numpy


if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (numpy array).
        b: Second vector (numpy array).

    Returns:
        Cosine similarity in range [-1, 1], clamped to [0, 1].
    """
    np = _get_numpy()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [0, 1] since we only care about positive similarity
    return max(0.0, min(1.0, similarity))


class EmbeddingScorer(RelevanceScorer):
    """Semantic relevance scorer using sentence embeddings.

    Uses sentence-transformers to compute dense embeddings and cosine similarity.
    The default model (all-MiniLM-L6-v2) offers a good balance of speed and quality.

    Example:
        scorer = EmbeddingScorer()
        score = scorer.score(
            '{"status": "failed", "error": "connection refused"}',
            "show me the errors"
        )
        # score.score > 0.5 (semantic match between "failed"/"error" and "errors")

    Note:
        Requires sentence-transformers: pip install headroom[relevance]
    """

    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        cache_model: bool = True,
    ):
        """Initialize embedding scorer.

        Args:
            model_name: Sentence transformer model name.
                Recommended models:
                - "all-MiniLM-L6-v2": Fast, good quality (default)
                - "all-mpnet-base-v2": Best quality, slower
                - "paraphrase-MiniLM-L6-v2": Good for paraphrase detection
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto).
            cache_model: If True, cache loaded models across instances.
        """
        self.model_name = model_name
        self.device = device
        self.cache_model = cache_model
        self._model: SentenceTransformer | None = None
        self._available: bool | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if sentence-transformers is installed.

        Returns:
            True if the package is available.
        """
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_model(self) -> SentenceTransformer:
        """Get or load the sentence transformer model.

        Returns:
            Loaded SentenceTransformer model.

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """
        if self._model is not None:
            return self._model

        if not self.is_available():
            raise RuntimeError(
                "EmbeddingScorer requires sentence-transformers. "
                "Install with: pip install headroom[relevance]"
            )

        # Check cache
        if self.cache_model and self.model_name in self._model_cache:
            self._model = self._model_cache[self.model_name]
            return self._model

        # Load model
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name, device=self.device)

        if self.cache_model:
            self._model_cache[self.model_name] = self._model

        return self._model

    def _encode(self, texts: list[str]):
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode.

        Returns:
            Array of embeddings, shape (len(texts), embedding_dim).
        """
        model = self._get_model()
        # normalize_embeddings=True ensures unit vectors for fast cosine via dot product
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def score(self, item: str, context: str) -> RelevanceScore:
        """Score item relevance to context using embeddings.

        Args:
            item: Item text.
            context: Query context.

        Returns:
            RelevanceScore with embedding-based similarity.
        """
        if not item or not context:
            return RelevanceScore(score=0.0, reason="Embedding: empty input")

        embeddings = self._encode([item, context])
        similarity = _cosine_similarity(embeddings[0], embeddings[1])

        return RelevanceScore(
            score=similarity,
            reason=f"Embedding: semantic similarity {similarity:.2f}",
        )

    def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
        """Score multiple items efficiently using batch encoding.

        This is much faster than scoring items individually since:
        1. Context is encoded only once
        2. Items are encoded in a single batch

        Args:
            items: List of items to score.
            context: Query context.

        Returns:
            List of RelevanceScore objects.
        """
        if not items:
            return []

        if not context:
            return [RelevanceScore(score=0.0, reason="Embedding: empty context") for _ in items]

        # Encode all texts in one batch
        all_texts = items + [context]
        embeddings = self._encode(all_texts)

        # Last embedding is the context
        context_emb = embeddings[-1]
        item_embs = embeddings[:-1]

        # Compute similarities
        results = []
        for emb in item_embs:
            similarity = _cosine_similarity(emb, context_emb)
            results.append(
                RelevanceScore(
                    score=similarity,
                    reason=f"Embedding: {similarity:.2f}",
                )
            )

        return results


# Convenience function for checking availability without instantiation
def embedding_available() -> bool:
    """Check if embedding scorer is available.

    Returns:
        True if sentence-transformers is installed.
    """
    return EmbeddingScorer.is_available()
