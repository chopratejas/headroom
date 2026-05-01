from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import headroom.evals.html_extraction as html_eval


def test_html_eval_results_and_suite_summary() -> None:
    result = html_eval.HTMLEvalResult(
        case_id="case-1",
        category="news",
        original_html_length=1000,
        extracted_length=250,
        compression_ratio=0.25,
        answer_from_original="orig",
        answer_from_extracted="extr",
        extracted_score=4.2,
        extracted_reasoning="good",
        baseline_score=3.5,
    )
    assert result.information_preserved is True
    assert result.extraction_wins is True

    no_baseline = html_eval.HTMLEvalResult(
        case_id="case-2",
        category="docs",
        original_html_length=100,
        extracted_length=50,
        compression_ratio=0.5,
        answer_from_original="orig",
        answer_from_extracted="extr",
    )
    assert no_baseline.information_preserved is False
    assert no_baseline.extraction_wins is None

    suite = html_eval.HTMLEvalSuiteResult(total_cases=2, results=[result, no_baseline])
    assert suite.avg_extraction_score == (4.2 + 0.0) / 2
    assert suite.avg_baseline_score == 3.5
    assert suite.information_preservation_rate == 50.0
    assert suite.extraction_win_rate == 100.0
    assert suite.avg_compression_ratio == 0.375
    summary = suite.summary()
    assert summary["by_category"]["news"]["count"] == 1
    assert summary["by_category"]["docs"]["preservation_rate"] == 0.0

    empty_suite = html_eval.HTMLEvalSuiteResult(total_cases=0, results=[])
    assert empty_suite.avg_extraction_score == 0.0
    assert empty_suite.avg_baseline_score is None
    assert empty_suite.extraction_win_rate is None
    assert empty_suite.avg_compression_ratio == 0.0


def test_html_extraction_evaluator_lazy_loading_and_provider_paths(monkeypatch) -> None:
    evaluator = html_eval.HTMLExtractionEvaluator(compare_baseline=True, provider="openai")

    def fake_extractor_cls():
        return SimpleNamespace(extract=lambda html, url=None: None)

    monkeypatch.setitem(
        sys.modules,
        "headroom.transforms.html_extractor",
        types.SimpleNamespace(HTMLExtractor=fake_extractor_cls),
    )
    assert evaluator.extractor.extract("x") is None

    def fake_kompress_cls():
        return SimpleNamespace(compress=lambda html: None)

    monkeypatch.setitem(
        sys.modules,
        "headroom.transforms.kompress_compressor",
        types.SimpleNamespace(KompressCompressor=fake_kompress_cls),
    )
    assert evaluator.kompress.compress("x") is None

    evaluator_no_baseline = html_eval.HTMLExtractionEvaluator(compare_baseline=False)
    assert evaluator_no_baseline.kompress is None

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(
            OpenAI=lambda: SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kwargs: SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content="Reasoning: ok\nScore: 4")
                                )
                            ]
                        )
                    )
                )
            )
        ),
    )
    score, reasoning = evaluator._create_judge()("q", "g", "p")
    assert score == 4.0
    assert reasoning == "ok"

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(
            Anthropic=lambda: SimpleNamespace(
                messages=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        content=[SimpleNamespace(text="Reasoning: yes\nScore: 5")]
                    )
                )
            )
        ),
    )
    anthropic_eval = html_eval.HTMLExtractionEvaluator(provider="anthropic")
    assert anthropic_eval._create_judge()("q", "g", "p")[0] == 5.0

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(
            completion=lambda **kwargs: SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="Reasoning: maybe\nScore: 2"))
                ]
            )
        ),
    )
    litellm_eval = html_eval.HTMLExtractionEvaluator(provider="litellm")
    assert litellm_eval._create_judge()("q", "g", "p")[0] == 2.0

    assert evaluator._parse_judge_response("Reasoning: detail\nScore: 9") == (5.0, "detail")
    assert evaluator._parse_judge_response("Reasoning: low\nScore: 0") == (1.0, "low")
    assert evaluator._parse_judge_response("unstructured") == (3.0, "unstructured")

    openai_answer = evaluator._get_answer("content", "question")
    assert openai_answer == "Reasoning: ok\nScore: 4"
    anthropic_answer = anthropic_eval._get_answer("content", "question")
    assert anthropic_answer == "Reasoning: yes\nScore: 5"
    litellm_answer = litellm_eval._get_answer("content", "question")
    assert litellm_answer == "Reasoning: maybe\nScore: 2"


def test_html_extraction_evaluator_case_and_suite_execution(monkeypatch) -> None:
    evaluator = html_eval.HTMLExtractionEvaluator(compare_baseline=True)
    evaluator._extractor = SimpleNamespace(
        extract=lambda html, url=None: SimpleNamespace(
            extracted="extracted text", compression_ratio=0.3
        )
    )
    evaluator._kompress = SimpleNamespace(
        compress=lambda html: SimpleNamespace(compressed="baseline text")
    )
    answers = {
        ("extracted text", "What?"): "answer extracted",
        ("<html>body</html>", "What?"): "answer original",
        ("baseline text", "What?"): "answer baseline",
    }
    evaluator._get_answer = lambda content, question: answers[(content, question)]
    evaluator._judge_fn = lambda q, g, prediction: (
        (4.5, "great") if prediction == "answer extracted" else (3.0, "baseline")
    )

    case = html_eval.HTMLEvalCase(
        id="case-1",
        html="<html>body</html>",
        url="https://example.test",
        question="What?",
        ground_truth="truth",
        category="news",
    )
    result = evaluator.evaluate_case(case)
    assert result.answer_from_original == "answer original"
    assert result.answer_from_baseline == "answer baseline"
    assert result.baseline_score == 3.0
    assert result.extracted_score == 4.5

    evaluator._kompress = SimpleNamespace(
        compress=lambda html: (_ for _ in ()).throw(RuntimeError("baseline failed"))
    )
    baseline_failure = evaluator.evaluate_case(case)
    assert baseline_failure.baseline_score is None

    original_evaluate_case = evaluator.evaluate_case

    def fake_evaluate_case(current_case):
        if current_case.id == "bad":
            raise RuntimeError("boom")
        return original_evaluate_case(current_case)

    evaluator.evaluate_case = fake_evaluate_case
    suite = evaluator.evaluate(
        [
            case,
            html_eval.HTMLEvalCase(
                id="bad",
                html="<html></html>",
                url=None,
                question="What?",
                ground_truth="truth",
            ),
        ]
    )
    assert suite.total_cases == 2
    assert len(suite.results) == 1

    samples = html_eval.get_sample_eval_cases()
    assert len(samples) >= 2
    assert samples[0].category == "news"
    assert "Authentication" in samples[1].html
