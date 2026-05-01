"""Direct branch tests for scoring helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.transforms import scoring as scoring_mod
from headroom.transforms.scoring import MessageScorer


class _EmbeddingProvider:
    def __init__(self, response):
        self.response = response
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_compute_semantic_score_uses_cache_and_handles_blank_content() -> None:
    provider = _EmbeddingProvider([1.0, 0.0])
    scorer = MessageScorer(embedding_provider=provider)

    blank_score = scorer._compute_semantic_score({"content": "   "}, 0, [1.0, 0.0])
    first_score = scorer._compute_semantic_score({"content": "hello"}, 1, [1.0, 0.0])
    second_score = scorer._compute_semantic_score({"content": "hello"}, 1, [1.0, 0.0])

    assert blank_score == 0.5
    assert first_score == 1.0
    assert second_score == 1.0
    assert provider.calls == ["hello"]


def test_compute_semantic_score_returns_neutral_on_embedding_failure() -> None:
    scorer = MessageScorer(embedding_provider=_EmbeddingProvider(RuntimeError("boom")))

    assert scorer._compute_semantic_score({"content": "hello"}, 0, [1.0, 0.0]) == 0.5


def test_compute_toin_score_handles_non_tool_invalid_content_and_low_confidence(
    monkeypatch,
) -> None:
    scorer = MessageScorer(toin=SimpleNamespace(get_pattern=lambda _: None))

    assert scorer._compute_toin_score({"role": "user", "content": "{}"}) == 0.5
    assert scorer._compute_toin_score({"role": "tool", "content": ""}) == 0.5
    assert scorer._compute_toin_score({"role": "tool", "content": "1"}) == 0.5
    assert scorer._compute_toin_score({"role": "tool", "content": "[]"}) == 0.5

    monkeypatch.setattr(
        "headroom.telemetry.models.ToolSignature.from_items",
        lambda items: SimpleNamespace(structure_hash="sig"),
    )
    scorer = MessageScorer(
        toin=SimpleNamespace(
            get_pattern=lambda _: SimpleNamespace(
                confidence=0.2,
                retrieval_rate=0.9,
                commonly_retrieved_fields=["a"],
            )
        )
    )

    assert scorer._compute_toin_score({"role": "tool", "content": '{"ok": true}'}) == 0.5


def test_compute_toin_score_applies_retrieval_and_field_boost(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.telemetry.models.ToolSignature.from_items",
        lambda items: SimpleNamespace(structure_hash="sig"),
    )
    scorer = MessageScorer(
        toin=SimpleNamespace(
            get_pattern=lambda _: SimpleNamespace(
                confidence=0.9,
                retrieval_rate=0.8,
                commonly_retrieved_fields=["a", "b", "c"],
            )
        )
    )

    score = scorer._compute_toin_score({"role": "tool", "content": '{"ok": true}'})

    assert score == pytest.approx(0.96)


def test_compute_toin_score_returns_neutral_on_signature_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.telemetry.models.ToolSignature.from_items",
        lambda items: (_ for _ in ()).throw(RuntimeError("sig fail")),
    )
    scorer = MessageScorer(toin=SimpleNamespace(get_pattern=lambda _: None))

    assert scorer._compute_toin_score({"role": "tool", "content": '{"ok": true}'}) == 0.5


def test_compute_error_score_handles_no_pattern_and_high_confidence_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.telemetry.models.ToolSignature.from_items",
        lambda items: SimpleNamespace(structure_hash="sig"),
    )

    no_pattern = MessageScorer(toin=SimpleNamespace(get_pattern=lambda _: None))
    assert no_pattern._compute_error_score({"role": "tool", "content": '{"ok": true}'}) == 0.0

    pattern = SimpleNamespace(
        field_semantics={
            "err1": SimpleNamespace(inferred_type="error_indicator", confidence=0.9),
            "err2": SimpleNamespace(inferred_type="error_indicator", confidence=0.6),
            "other": SimpleNamespace(inferred_type="metadata", confidence=0.9),
        }
    )
    scorer = MessageScorer(toin=SimpleNamespace(get_pattern=lambda _: pattern))

    assert scorer._compute_error_score(
        {"role": "tool", "content": '{"ok": true}'}
    ) == pytest.approx(0.8)


def test_compute_error_score_returns_zero_for_non_tool_and_failures(monkeypatch) -> None:
    scorer = MessageScorer(toin=SimpleNamespace(get_pattern=lambda _: None))
    assert scorer._compute_error_score({"role": "user", "content": "{}"}) == 0.0
    assert scorer._compute_error_score({"role": "tool", "content": ""}) == 0.0
    assert scorer._compute_error_score({"role": "tool", "content": "1"}) == 0.0
    assert scorer._compute_error_score({"role": "tool", "content": "[]"}) == 0.0

    monkeypatch.setattr(
        "headroom.telemetry.models.ToolSignature.from_items",
        lambda items: (_ for _ in ()).throw(RuntimeError("sig fail")),
    )
    assert scorer._compute_error_score({"role": "tool", "content": '{"ok": true}'}) == 0.0


def test_compute_density_score_short_content_and_reference_score_zero() -> None:
    scorer = MessageScorer()

    assert scorer._compute_density_score({"content": "short"}) == 0.5
    assert scorer._compute_reference_score(0, {}) == 0.0


def test_recent_context_embedding_and_cosine_zero_norm() -> None:
    scorer = MessageScorer()
    assert scorer._compute_recent_context_embedding([{"content": "hello"}]) is None

    failing = MessageScorer(embedding_provider=_EmbeddingProvider(RuntimeError("boom")))
    assert failing._compute_recent_context_embedding([{"content": "hello"}]) is None

    provider = _EmbeddingProvider([0.0, 1.0])
    working = MessageScorer(embedding_provider=provider)
    assert working._compute_recent_context_embedding(
        [{"content": ""}, {"content": "hello"}, {"content": "world"}]
    ) == [0.0, 1.0]
    assert provider.calls == ["hello world"]

    assert scoring_mod.MessageScorer._cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
