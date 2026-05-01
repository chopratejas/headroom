from __future__ import annotations

import numpy as np

from headroom.prediction.feature_extractor import (
    ComplexityLevel,
    DomainType,
    EmbeddingExtractor,
    EmbeddingFeatures,
    MetaExtractor,
    MetaFeatures,
    PromptFeatureExtractor,
    PromptFeatures,
    PromptFormat,
    SemanticExtractor,
    SemanticFeatures,
    StructuralExtractor,
    StructuralFeatures,
    TaskType,
    TextStatisticsExtractor,
    TextStatisticsFeatures,
    extract_features,
    get_feature_vector,
)


class _Tokenizer:
    def __init__(self, count: int, *, raises: bool = False) -> None:
        self._count = count
        self._raises = raises

    def count_text(self, text: str) -> int:
        if self._raises:
            raise RuntimeError("tokenizer unavailable")
        return self._count


def test_feature_dataclasses_align_vectors_and_names() -> None:
    text = TextStatisticsFeatures(token_count_exact=12)
    assert len(text.to_vector()) == len(TextStatisticsFeatures.feature_names())

    structural = StructuralFeatures(question_types=["what"], code_languages_detected=["python"])
    assert len(structural.to_vector()) == len(StructuralFeatures.feature_names())

    semantic = SemanticFeatures(
        primary_task_type=TaskType.CODE,
        primary_domain=DomainType.CODE,
        complexity_level=ComplexityLevel.COMPLEX,
        prompt_format=PromptFormat.TEMPLATE,
    )
    assert len(semantic.to_vector()) == len(SemanticFeatures.feature_names())

    embedding = EmbeddingFeatures(raw_embedding=[0.1, 0.2], embedding_dim=2)
    assert len(embedding.to_vector(include_raw=True)) == len(
        EmbeddingFeatures.feature_names(include_raw=True, embedding_dim=2)
    )

    meta = MetaFeatures(model_family="custom", model_size_category="xl")
    assert len(meta.to_vector()) == len(MetaFeatures.feature_names())

    prompt = PromptFeatures(
        text_statistics=text,
        structural=structural,
        semantic=semantic,
        embedding=embedding,
        meta=meta,
    )
    assert len(prompt.to_vector(include_raw_embedding=True)) == len(
        PromptFeatures.feature_names(include_raw_embedding=True, embedding_dim=2)
    )

    payload = prompt.to_dict()
    assert payload["semantic"]["primary_task_type"] == "code"
    assert "raw_embedding" not in payload["embedding"]


def test_text_statistics_extractor_computes_metrics_and_fails_open(monkeypatch) -> None:
    extractor = TextStatisticsExtractor(tokenizer=_Tokenizer(42))
    text = "HELLO world 123!\n\nHELLO again. Maybe make code."

    features = extractor.extract(text)

    assert features.char_count == len(text)
    assert features.word_count == 8
    assert features.token_count_exact == 42
    assert features.sentence_count >= 3
    assert features.paragraph_count == 2
    assert features.unique_word_count >= 5
    assert features.uppercase_ratio > 0
    assert features.digit_ratio > 0
    assert features.compression_ratio > 0
    assert features.entropy_estimate > 0
    assert features.repetition_score >= 0
    assert 0.0 <= features.flesch_reading_ease <= 100.0
    assert features.flesch_kincaid_grade >= 0.0

    fail_open = TextStatisticsExtractor(tokenizer=_Tokenizer(0, raises=True))
    assert fail_open.extract("short text").token_count_exact is None

    monkeypatch.setattr(
        "headroom.prediction.feature_extractor.gzip.compress",
        lambda data: (_ for _ in ()).throw(OSError("gzip failed")),
    )
    assert extractor._calculate_compression_ratio("abc") == 1.0
    assert extractor._calculate_entropy("") == 0.0
    assert extractor._calculate_repetition_score("tiny") == 0.0
    assert extractor._count_syllables("make code") >= 2
    assert extractor._flesch_reading_ease(0, 1, 1) == 0.0
    assert extractor._flesch_kincaid_grade(0, 1, 1) == 0.0


def test_structural_extractor_detects_rich_prompt_patterns() -> None:
    text = """# Header
System: Keep it concise.
User: What is this?
Assistant: Here is context.

1. First item
  - nested bullet
* second bullet

For example: use this.
You are an expert and must respond in JSON format.
Think step by step.
Context: <root><item>v</item></root>
---
```python
print("hi")
```
Inline `code` and ![alt](img.png) plus [docs](https://example.com)
| a | b |
| 1 | 2 |
> quote
{"key":"value"}
"""
    features = StructuralExtractor().extract(text)

    assert features.is_question is True
    assert features.has_multiple_questions is False
    assert "what" in features.question_types
    assert features.numbered_list_count == 1
    assert features.bullet_list_count >= 1
    assert features.has_nested_lists is True
    assert features.code_block_count == 1
    assert features.code_languages_detected == ["python"]
    assert features.total_code_lines >= 2
    assert features.inline_code_count >= 1
    assert features.code_to_text_ratio > 0
    assert features.header_count == 1
    assert features.link_count >= 1
    assert features.image_reference_count == 1
    assert features.table_count >= 1
    assert features.blockquote_count == 1
    assert features.xml_tag_count >= 2
    assert features.json_object_count == 1
    assert features.has_structured_template is True
    assert features.has_role_markers is True
    assert features.turn_count == 3
    assert features.has_system_prompt_marker is True
    assert features.has_examples is True
    assert features.has_constraints is True
    assert features.has_output_format_spec is True
    assert features.has_chain_of_thought is True
    assert features.has_persona_definition is True
    assert features.has_context_window is True


def test_semantic_extractor_detects_intent_and_format_variants(monkeypatch) -> None:
    extractor = SemanticExtractor(use_ner=True)
    monkeypatch.setattr(
        extractor,
        "_extract_entities",
        lambda text: [("Python", "LANGUAGE"), ("OpenAI", "ORG")],
    )

    text = (
        "Analyze this Python API bug and compare two fixes. "
        "Please provide 3 items in JSON format urgent now. "
        "First explain the cause, then evaluate the options."
    )
    features = extractor.extract(text)

    assert features.primary_task_type in {TaskType.ANALYZE, TaskType.DEBUG}
    assert (
        TaskType.ANALYZE in features.secondary_task_types
        or features.primary_task_type == TaskType.ANALYZE
    )
    assert features.primary_domain == DomainType.CODE
    assert features.complexity_level in {ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX}
    assert features.reasoning_depth_estimate >= 2
    assert features.specificity_score > 0.5
    assert features.has_specific_entities is True
    assert features.named_entity_count == 2
    assert features.requires_reasoning is True
    assert features.requires_structured_output is True
    assert features.explicit_length_request == "items"
    assert features.requested_item_count == 3
    assert features.urgency_indicators >= 1
    assert "python" in features.top_keywords
    assert features.keyword_density > 0
    assert features.prompt_format == PromptFormat.INSTRUCTION

    assert extractor._detect_format("What is this?") == PromptFormat.QUESTION
    assert extractor._detect_format("User: hi\nAssistant: hello") == PromptFormat.MULTI_TURN
    assert extractor._detect_format("Name: {name}\nRole: <role>") == PromptFormat.TEMPLATE
    assert (
        extractor._detect_format("value = 1;\nvalue += 2;\ncall();\nother();")
        == PromptFormat.RAW_DATA
    )
    assert extractor._detect_format("Explain the system behavior.") == PromptFormat.INSTRUCTION

    failure_extractor = SemanticExtractor(use_ner=True)
    failure_extractor._nlp = object()
    assert failure_extractor._extract_entities("Acme Corp") == []


def test_embedding_extractor_supports_stubbed_models_and_fail_open(monkeypatch) -> None:
    unavailable = EmbeddingExtractor()
    monkeypatch.setattr(unavailable, "is_available", lambda: False)
    try:
        unavailable._get_model()
    except RuntimeError as exc:
        assert "sentence-transformers" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when sentence-transformers is unavailable")

    class _Model:
        @staticmethod
        def encode(*args, **kwargs):
            return np.array([1.0, 0.0], dtype=float)

    extractor = EmbeddingExtractor(
        cluster_centers={
            "short_response": [1.0, 0.0],
            "code": [0.0, 1.0],
            "bad_dim": [1.0],
        }
    )
    monkeypatch.setattr(extractor, "is_available", lambda: True)
    monkeypatch.setattr(extractor, "_get_model", lambda: _Model())
    features = extractor.extract("hello world")

    assert features.raw_embedding == [1.0, 0.0]
    assert features.embedding_dim == 2
    assert features.embedding_norm > 0
    assert features.embedding_entropy >= 0
    assert features.similarity_to_short_response_cluster == 1.0
    assert features.similarity_to_code_cluster == 0.0

    broken = EmbeddingExtractor()
    monkeypatch.setattr(broken, "is_available", lambda: True)
    monkeypatch.setattr(broken, "_get_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    failed = broken.extract("hello world")
    assert failed.embedding_dim == 0
    assert failed.raw_embedding is None


def test_meta_and_prompt_feature_extractors_cover_cache_batch_and_utilities() -> None:
    meta = MetaExtractor(tokenizer=_Tokenizer(40)).extract(
        text="Explain the model behavior",
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=120,
        top_p=0.8,
        system_prompt="Please keep responses brief and no more than 10 words.",
        conversation_turn=2,
        cumulative_tokens=500,
    )

    assert meta.model_family == "gpt"
    assert meta.model_size_category == "small"
    assert meta.model_context_limit == 128000
    assert meta.prompt_context_ratio > 0
    assert meta.available_output_tokens == 120
    assert len(meta.prompt_hash) == 16
    assert meta.prompt_signature == "B"
    assert meta.is_first_turn is False
    assert meta.system_prompt_length > 0
    assert meta.has_output_constraints_in_system is True

    extractor = PromptFeatureExtractor(use_embeddings=False)
    first = extractor.extract("What is machine learning?", model="gpt-4o")
    second = extractor.extract("What is machine learning?", model="gpt-4o")
    assert first is second

    extractor._cache_max_size = 2
    extractor.extract("prompt one")
    extractor.extract("prompt two")
    extractor.extract("prompt three")
    assert len(extractor._cache) <= 2
    extractor.clear_cache()
    assert extractor._cache == {}

    class _BatchModel:
        @staticmethod
        def encode(prompts, **kwargs):
            return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)

    class _BatchEmbeddingExtractor:
        def _get_model(self):
            return _BatchModel()

        def extract(self, prompt: str) -> EmbeddingFeatures:
            return EmbeddingFeatures(raw_embedding=[9.0], embedding_dim=1)

    batch_extractor = PromptFeatureExtractor(use_embeddings=False)
    batch_extractor.use_embeddings = True
    batch_extractor.embedding_extractor = _BatchEmbeddingExtractor()
    batch = batch_extractor.extract_batch(["first prompt", "second prompt"])

    assert [item.embedding.raw_embedding for item in batch] == [[0.1, 0.2], [0.3, 0.4]]

    single = extract_features("List 2 items", model="claude-3-sonnet", conversation_turn=1)
    vector = get_feature_vector("List 2 items", model="claude-3-sonnet")
    assert isinstance(single, PromptFeatures)
    assert len(vector) == len(single.to_vector())
