from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import headroom.evals.html_oss_benchmarks as hob


class _FakeDataset:
    def __init__(self, items):
        self._items = list(items)

    def select(self, indices):
        return _FakeDataset([self._items[i] for i in indices])

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, item):
        return self._items[item]


def test_html_oss_metrics_and_summary() -> None:
    assert hob.tokenize("Hello, World!") == ["hello", "world"]
    assert hob.compute_f1("a b", "a b c") == (1.0, 2 / 3, pytest.approx(0.8))
    assert hob.compute_f1("", "abc") == (0.0, 0.0, 0.0)
    assert hob.compute_f1("x", "y") == (0.0, 0.0, 0.0)
    assert hob.compute_exact_match("Hello,   world!", "hello world") is True

    extraction = hob.ExtractionBenchmarkResult(
        total_samples=2,
        avg_precision=0.9,
        avg_recall=0.8,
        avg_f1=0.95,
        avg_compression_ratio=0.4,
    )
    assert extraction.matches_baseline is True
    assert extraction.beats_baseline is False
    assert extraction.summary()["avg_f1"] == 0.95

    qa = hob.QAAccuracyResult(
        total_questions=2,
        accuracy_original_html=0.9,
        accuracy_extracted=0.91,
        accuracy_preserved=True,
        avg_f1_original=0.9,
        avg_f1_extracted=0.91,
        exact_match_original=0.5,
        exact_match_extracted=1.0,
    )
    assert qa.summary()["accuracy_delta"] == 0.01

    suite = hob.HTMLExtractorBenchmarkSuite(extraction_result=extraction, qa_result=qa)
    assert suite.all_passed is True
    assert "extraction" in suite.summary()


def test_evaluate_scrapinghub_and_qa_preservation(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            load_dataset=lambda *args, **kwargs: {
                "train": _FakeDataset(
                    [
                        {
                            "html": "<html>alpha beta</html>",
                            "articleBody": "alpha beta",
                            "url": "u1",
                        },
                        {
                            "html": "<html>gamma delta</html>",
                            "articleBody": "gamma delta",
                            "url": "u2",
                        },
                    ]
                )
            }
        ),
    )
    extractor = SimpleNamespace(
        extract=lambda html, url=None: SimpleNamespace(
            extracted=html.replace("<html>", "").replace("</html>", ""),
            compression_ratio=0.5,
        )
    )
    extraction_result = hob.evaluate_scrapinghub_benchmark(extractor=extractor, max_samples=1)
    assert extraction_result.total_samples == 1
    assert extraction_result.avg_f1 == 1.0

    squad_samples = _FakeDataset(
        [
            {
                "question": "Who?",
                "context": "Alice went home.",
                "answers": {"text": ["Alice"]},
            },
            {
                "question": "Skip?",
                "context": "",
                "answers": {"text": [""]},
            },
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: squad_samples),
    )

    def answer_fn(context, question):
        return "Alice" if "Alice" in context else "Unknown"

    qa_result = hob.evaluate_qa_accuracy_preservation(
        answer_fn=answer_fn,
        extractor=SimpleNamespace(
            extract=lambda html: SimpleNamespace(extracted=html, compression_ratio=0.5)
        ),
        max_questions=2,
        dataset_name="squad",
    )
    assert qa_result.total_questions == 1
    assert qa_result.accuracy_preserved is True

    hotpot_samples = _FakeDataset(
        [
            {
                "question": "Where?",
                "context": {"sentences": ["Paris is in France."]},
                "answer": "Paris",
            }
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: hotpot_samples),
    )
    hotpot_result = hob.evaluate_qa_accuracy_preservation(
        answer_fn=lambda context, question: "Paris",
        extractor=SimpleNamespace(
            extract=lambda html: SimpleNamespace(extracted=html, compression_ratio=0.4)
        ),
        max_questions=1,
        dataset_name="hotpotqa",
    )
    assert hotpot_result.total_questions == 1

    with pytest.raises(ValueError):
        hob.evaluate_qa_accuracy_preservation(
            answer_fn=answer_fn,
            extractor=extractor,
            dataset_name="unknown",
        )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: _FakeDataset([])),
    )
    with pytest.raises(ValueError):
        hob.evaluate_qa_accuracy_preservation(
            answer_fn=lambda *_args: "x",
            extractor=extractor,
            max_questions=0,
            dataset_name="squad",
        )


def test_run_full_html_oss_benchmark_suite(monkeypatch) -> None:
    extractor = SimpleNamespace()
    extraction = hob.ExtractionBenchmarkResult(
        total_samples=1,
        avg_precision=1.0,
        avg_recall=1.0,
        avg_f1=0.95,
        avg_compression_ratio=0.4,
    )
    qa = hob.QAAccuracyResult(
        total_questions=1,
        accuracy_original_html=1.0,
        accuracy_extracted=1.0,
        accuracy_preserved=True,
        avg_f1_original=1.0,
        avg_f1_extracted=1.0,
        exact_match_original=1.0,
        exact_match_extracted=1.0,
    )
    monkeypatch.setattr(hob, "evaluate_scrapinghub_benchmark", lambda **kwargs: extraction)
    monkeypatch.setattr(hob, "evaluate_qa_accuracy_preservation", lambda **kwargs: qa)
    suite = hob.run_full_benchmark_suite(
        extractor=extractor,
        answer_fn=lambda *_args: "x",
        extraction_samples=5,
        qa_questions=5,
    )
    assert suite.extraction_result is extraction
    assert suite.qa_result is qa

    monkeypatch.setattr(
        hob,
        "evaluate_scrapinghub_benchmark",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("extract fail")),
    )
    monkeypatch.setattr(
        hob,
        "evaluate_qa_accuracy_preservation",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("qa fail")),
    )
    failure_suite = hob.run_full_benchmark_suite(extractor=extractor, answer_fn=lambda *_args: "x")
    assert failure_suite.extraction_result is None
    assert failure_suite.qa_result is None
