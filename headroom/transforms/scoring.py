"""Message importance scoring for intelligent context management.

This module provides semantic-aware scoring of messages to determine
which ones are most important to keep when context limits are exceeded.

Design principle: NO HARDCODED PATTERNS
- Error detection: Uses TOIN field_semantics.inferred_type == "error_indicator"
- Importance: Derived from TOIN retrieval_rate and field patterns
- Relevance: Computed via embedding similarity, not keyword matching
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..config import ScoringWeights
    from ..telemetry.toin import ToolIntelligenceNetwork

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


@dataclass
class MessageScore:
    """Importance score for a single message.

    All scores are in range [0.0, 1.0] where higher = more important.
    """

    message_index: int
    total_score: float

    # Component scores
    recency_score: float = 0.0
    semantic_score: float = 0.0
    toin_score: float = 0.0
    error_score: float = 0.0
    reference_score: float = 0.0
    density_score: float = 0.0

    # Metadata
    tokens: int = 0
    is_protected: bool = False
    drop_safe: bool = True  # Can be dropped without orphaning tool responses

    # Debug info
    score_breakdown: dict[str, float] = field(default_factory=dict)


class MessageScorer:
    """Scores messages by semantic importance using learned patterns.

    This scorer uses TOIN-learned patterns and computed metrics to determine
    message importance. It does NOT use hardcoded keyword matching.

    Scoring factors:
    1. Recency: Exponential decay from conversation end
    2. Semantic similarity: Embedding cosine similarity to recent context
    3. TOIN importance: Learned field importance from retrieval patterns
    4. Error indicators: TOIN-learned error_indicator field types
    5. Forward references: Messages referenced by later messages
    6. Token density: Information density (unique tokens / total tokens)
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        toin: ToolIntelligenceNetwork | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        recency_decay_rate: float = 0.1,
    ):
        """Initialize scorer.

        Args:
            weights: Scoring weights for each factor
            toin: ToolIntelligenceNetwork instance for learned patterns
            embedding_provider: Optional embedding provider for semantic similarity
            recency_decay_rate: Lambda for exponential recency decay
        """
        # Import here to avoid circular imports
        from ..config import ScoringWeights

        self.weights = (weights or ScoringWeights()).normalized()
        self.toin = toin
        self.embedding_provider = embedding_provider
        self.recency_decay_rate = recency_decay_rate

        # Cache for embeddings
        self._embedding_cache: dict[int, list[float]] = {}

    def score_messages(
        self,
        messages: list[dict[str, Any]],
        protected_indices: set[int],
        tool_unit_indices: set[int],
    ) -> list[MessageScore]:
        """Score all messages by importance.

        Args:
            messages: List of messages to score
            protected_indices: Indices that should never be dropped
            tool_unit_indices: Indices that are part of tool units

        Returns:
            List of MessageScore objects, one per message
        """
        scores: list[MessageScore] = []

        # Pre-compute forward references
        forward_refs = self._compute_forward_references(messages)

        # Pre-compute recent context embedding (last 3 messages)
        recent_embedding = self._compute_recent_context_embedding(messages)

        for i, msg in enumerate(messages):
            score = self._score_message(
                msg=msg,
                index=i,
                total_messages=len(messages),
                protected=i in protected_indices,
                in_tool_unit=i in tool_unit_indices,
                forward_refs=forward_refs,
                recent_embedding=recent_embedding,
            )
            scores.append(score)

        return scores

    def _score_message(
        self,
        msg: dict[str, Any],
        index: int,
        total_messages: int,
        protected: bool,
        in_tool_unit: bool,
        forward_refs: dict[int, int],
        recent_embedding: list[float] | None,
    ) -> MessageScore:
        """Score a single message."""
        # Compute individual scores
        recency = self._compute_recency_score(index, total_messages)
        semantic = self._compute_semantic_score(msg, index, recent_embedding)
        toin_importance = self._compute_toin_score(msg)
        error = self._compute_error_score(msg)
        reference = self._compute_reference_score(index, forward_refs)
        density = self._compute_density_score(msg)

        # Weighted combination
        w = self.weights
        total = (
            w.recency * recency
            + w.semantic_similarity * semantic
            + w.toin_importance * toin_importance
            + w.error_indicator * error
            + w.forward_reference * reference
            + w.token_density * density
        )

        # Estimate tokens (simple heuristic)
        content = msg.get("content", "")
        if isinstance(content, str):
            tokens = len(content) // 4  # Rough estimate
        else:
            tokens = 100  # Default for complex content

        return MessageScore(
            message_index=index,
            total_score=total,
            recency_score=recency,
            semantic_score=semantic,
            toin_score=toin_importance,
            error_score=error,
            reference_score=reference,
            density_score=density,
            tokens=tokens,
            is_protected=protected,
            drop_safe=not in_tool_unit or not protected,
            score_breakdown={
                "recency": recency,
                "semantic": semantic,
                "toin": toin_importance,
                "error": error,
                "reference": reference,
                "density": density,
            },
        )

    def _compute_recency_score(self, index: int, total: int) -> float:
        """Compute recency score using exponential decay."""
        if total <= 1:
            return 1.0

        # Position from end (0 = last message)
        position_from_end = total - 1 - index

        # Exponential decay: score = e^(-λ * position)
        score = math.exp(-self.recency_decay_rate * position_from_end)

        return score

    def _compute_semantic_score(
        self,
        msg: dict[str, Any],
        index: int,
        recent_embedding: list[float] | None,
    ) -> float:
        """Compute semantic similarity to recent context."""
        if self.embedding_provider is None or recent_embedding is None:
            return 0.5  # Neutral score if no embedding available

        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            return 0.5

        try:
            # Get or compute embedding for this message
            if index not in self._embedding_cache:
                self._embedding_cache[index] = self.embedding_provider.embed(content)

            msg_embedding = self._embedding_cache[index]

            # Cosine similarity
            return self._cosine_similarity(msg_embedding, recent_embedding)
        except Exception as e:
            logger.debug(f"Embedding computation failed: {e}")
            return 0.5

    def _compute_toin_score(self, msg: dict[str, Any]) -> float:
        """Compute importance score from TOIN patterns.

        This uses TOIN-learned retrieval patterns to determine importance.
        Higher retrieval rate = more important (users needed this data).
        """
        if self.toin is None:
            return 0.5  # Neutral if no TOIN

        # Only tool messages have TOIN patterns
        if msg.get("role") != "tool":
            return 0.5

        content = msg.get("content", "")
        if not content:
            return 0.5

        try:
            # Try to parse as JSON and get tool signature
            from ..telemetry.models import ToolSignature

            data = json.loads(content) if isinstance(content, str) else content
            if not isinstance(data, (list, dict)):
                return 0.5

            # Get tool signature
            items = data if isinstance(data, list) else [data]
            if not items:
                return 0.5

            signature = ToolSignature.from_items(items)
            pattern = self.toin.get_pattern(signature.structure_hash)

            if pattern is None or pattern.confidence < 0.3:
                return 0.5

            # Score based on retrieval rate (high retrieval = important)
            # Note: We use retrieval_rate as the primary importance signal from TOIN.
            # The commonly_retrieved_fields comparison is not used here because
            # ToolSignature only tracks structure_hash, not individual field hashes.
            score = 0.5 + (pattern.retrieval_rate * 0.5)

            # Boost slightly if pattern has commonly retrieved fields (indicates
            # this tool type has learned importance patterns)
            if pattern.commonly_retrieved_fields:
                boost = min(0.1, 0.02 * len(pattern.commonly_retrieved_fields))
                score = min(1.0, score + boost)

            return score

        except Exception as e:
            logger.debug(f"TOIN scoring failed: {e}")
            return 0.5

    def _compute_error_score(self, msg: dict[str, Any]) -> float:
        """Compute error indicator score using TOIN-learned patterns.

        This does NOT use hardcoded keyword matching. Instead, it uses
        TOIN's learned field_semantics to identify error_indicator fields.
        """
        if self.toin is None:
            return 0.0

        # Only tool messages have field semantics
        if msg.get("role") != "tool":
            return 0.0

        content = msg.get("content", "")
        if not content:
            return 0.0

        try:
            from ..telemetry.models import ToolSignature

            data = json.loads(content) if isinstance(content, str) else content
            if not isinstance(data, (list, dict)):
                return 0.0

            items = data if isinstance(data, list) else [data]
            if not items:
                return 0.0

            signature = ToolSignature.from_items(items)
            pattern = self.toin.get_pattern(signature.structure_hash)

            if pattern is None:
                return 0.0

            # Check field_semantics for error_indicator type
            error_field_count = 0
            high_confidence_errors = 0

            for _field_hash, field_sem in pattern.field_semantics.items():
                if field_sem.inferred_type == "error_indicator":
                    error_field_count += 1
                    if field_sem.confidence >= 0.7:
                        high_confidence_errors += 1

            if error_field_count == 0:
                return 0.0

            # Score based on presence and confidence of error fields
            base_score = min(1.0, 0.3 * error_field_count)
            confidence_boost = min(0.5, 0.2 * high_confidence_errors)

            return base_score + confidence_boost

        except Exception as e:
            logger.debug(f"Error scoring failed: {e}")
            return 0.0

    def _compute_reference_score(
        self,
        index: int,
        forward_refs: dict[int, int],
    ) -> float:
        """Compute score based on forward references.

        Messages that are referenced by later messages are more important.
        """
        ref_count = forward_refs.get(index, 0)

        if ref_count == 0:
            return 0.0

        # Logarithmic scaling for reference count
        return min(1.0, 0.3 + 0.2 * math.log(ref_count + 1))

    def _compute_density_score(self, msg: dict[str, Any]) -> float:
        """Compute information density score.

        Higher unique token ratio = more information dense = more important.
        """
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < 10:
            return 0.5

        # Simple token density: unique tokens / total tokens
        tokens = content.lower().split()
        if len(tokens) < 3:
            return 0.5

        unique_tokens = len(set(tokens))
        density = unique_tokens / len(tokens)

        # Normalize to [0, 1] range (typical density is 0.3-0.8)
        return min(1.0, max(0.0, (density - 0.2) / 0.6))

    def _compute_forward_references(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[int, int]:
        """Compute which messages are referenced by later messages.

        Returns dict mapping message index to reference count.
        """
        refs: dict[int, int] = {}

        # Track tool_call_id references
        tool_call_ids: dict[str, int] = {}  # tool_call_id -> assistant message index

        for i, msg in enumerate(messages):
            role = msg.get("role")

            # Track assistant tool calls
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    tc_id = tc.get("id")
                    if tc_id:
                        tool_call_ids[tc_id] = i

            # Tool responses reference assistant messages
            elif role == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id and tc_id in tool_call_ids:
                    ref_idx = tool_call_ids[tc_id]
                    refs[ref_idx] = refs.get(ref_idx, 0) + 1

        return refs

    def _compute_recent_context_embedding(
        self,
        messages: list[dict[str, Any]],
        num_recent: int = 3,
    ) -> list[float] | None:
        """Compute average embedding of recent messages."""
        if self.embedding_provider is None:
            return None

        recent_texts: list[str] = []
        for msg in messages[-num_recent:]:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                recent_texts.append(content)

        if not recent_texts:
            return None

        try:
            # Combine recent texts and embed
            combined = " ".join(recent_texts)
            return self.embedding_provider.embed(combined)
        except Exception as e:
            logger.debug(f"Recent context embedding failed: {e}")
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
