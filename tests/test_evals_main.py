from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

import headroom.evals.__main__ as eval_main


class _QuickResults:
    def __init__(self, accuracy_preservation_rate: float) -> None:
        self.accuracy_preservation_rate = accuracy_preservation_rate
        self.saved_to = None

    def save(self, path: str) -> None:
        self.saved_to = path


class _BenchResult:
    def __init__(
        self,
        *,
        suite_name: str = "suite",
        total_cases: int = 10,
        passed_cases: int = 9,
        total_original_tokens: int = 1000,
        total_compressed_tokens: int = 400,
    ) -> None:
        self.suite_name = suite_name
        self.total_cases = total_cases
        self.passed_cases = passed_cases
        self.total_original_tokens = total_original_tokens
        self.total_compressed_tokens = total_compressed_tokens
        self.avg_f1_score = 0.95
        self.avg_compression_ratio = 0.6

    def summary(self) -> str:
        return f"summary:{self.suite_name}"

    def to_dict(self) -> dict:
        return {"suite_name": self.suite_name}


def test_cmd_quick_and_list(monkeypatch, capsys) -> None:
    quick_module = types.ModuleType("headroom.evals.runners.before_after")
    quick_module.run_quick_eval = lambda **kwargs: _QuickResults(0.95)
    monkeypatch.setitem(sys.modules, "headroom.evals.runners.before_after", quick_module)

    args = Namespace(n=5, provider="anthropic", model="claude", output="results.json")
    eval_main.cmd_quick(args)
    assert "Results saved to: results.json" in capsys.readouterr().out

    quick_module.run_quick_eval = lambda **kwargs: _QuickResults(0.5)
    with pytest.raises(SystemExit):
        eval_main.cmd_quick(args)

    datasets_module = types.ModuleType("headroom.evals.datasets")
    datasets_module.DATASET_REGISTRY = {
        "tool_outputs": {"default_n": 5, "description": "tool dataset"},
        "squad": {"default_n": 10, "description": "qa dataset"},
    }
    datasets_module.list_available_datasets = lambda: {
        "tool_use": ["tool_outputs"],
        "qa": ["squad"],
    }
    monkeypatch.setitem(sys.modules, "headroom.evals.datasets", datasets_module)

    eval_main.cmd_list(Namespace())
    output = capsys.readouterr().out
    assert "AVAILABLE EVALUATION DATASETS" in output
    assert "tool_outputs" in output
    assert "qa dataset" in output


def test_cmd_benchmark_paths(monkeypatch, tmp_path: Path, capsys) -> None:
    datasets_module = types.ModuleType("headroom.evals.datasets")
    datasets_module.DATASET_REGISTRY = {
        "tool_outputs": {"default_n": 5, "description": "tool dataset"},
        "squad": {"default_n": 10, "description": "qa dataset"},
        "bfcl": {"default_n": 10, "description": "bfcl dataset"},
    }
    datasets_module.list_available_datasets = lambda: {"qa": ["squad"], "tool_use": ["bfcl"]}
    datasets_module.load_tool_output_samples = lambda: ["tool-case"]
    datasets_module.load_dataset_by_name = lambda name, n: [f"{name}-{n}"]
    monkeypatch.setitem(sys.modules, "headroom.evals.datasets", datasets_module)

    runner_module = types.ModuleType("headroom.evals.runners.before_after")

    class LLMConfig:
        def __init__(self, provider: str, model: str) -> None:
            self.provider = provider
            self.model = model

    class BeforeAfterRunner:
        def __init__(self, llm_config, use_semantic_similarity: bool) -> None:
            self.llm_config = llm_config
            self.use_semantic_similarity = use_semantic_similarity

        def run(self, suite, progress_callback):
            progress_callback(
                1,
                1,
                types.SimpleNamespace(accuracy_preserved=True, f1_score=1.0, compression_ratio=0.5),
            )
            return _BenchResult(suite_name=str(suite[0]))

    runner_module.LLMConfig = LLMConfig
    runner_module.BeforeAfterRunner = BeforeAfterRunner
    monkeypatch.setitem(sys.modules, "headroom.evals.runners.before_after", runner_module)

    args = Namespace(
        n=3,
        dataset="quick",
        provider="anthropic",
        model="claude",
        semantic=True,
        output=str(tmp_path / "benchmark.json"),
    )
    eval_main.cmd_benchmark(args)
    output = capsys.readouterr().out
    assert "OVERALL RESULTS" in output
    assert Path(args.output).exists()
    assert json.loads(Path(args.output).read_text())["totals"]["cases"] == 30

    args.dataset = "qa"
    eval_main.cmd_benchmark(args)
    assert "Loaded 1 cases" in capsys.readouterr().out

    args.dataset = "unknown"
    with pytest.raises(SystemExit):
        eval_main.cmd_benchmark(args)

    datasets_module.load_dataset_by_name = lambda name, n: (_ for _ in ()).throw(
        ImportError("missing extras")
    )
    args.dataset = "squad"
    with pytest.raises(SystemExit):
        eval_main.cmd_benchmark(Namespace(**{**args.__dict__, "output": None}))
    missing_output = capsys.readouterr().out
    assert "Install with: pip install headroom-ai[evals]" in missing_output

    datasets_module.load_dataset_by_name = lambda name, n: (_ for _ in ()).throw(
        RuntimeError("broken")
    )
    with pytest.raises(SystemExit):
        eval_main.cmd_benchmark(Namespace(**{**args.__dict__, "output": None}))
    assert "Error loading squad: broken" in capsys.readouterr().out


def test_cmd_suite_report_and_main(monkeypatch, tmp_path: Path, capsys) -> None:
    report_module = types.ModuleType("headroom.evals.reports.report_card")
    report_module.save_reports = lambda result, output: {"html": Path(output) / "report.html"}
    monkeypatch.setitem(sys.modules, "headroom.evals.reports.report_card", report_module)

    suite_module = types.ModuleType("headroom.evals.suite_runner")

    class SuiteRunner:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def run(self):
            return types.SimpleNamespace(
                all_passed=False,
                benchmarks=[types.SimpleNamespace(name="bench-1", passed=False)],
            )

    suite_module.SuiteRunner = SuiteRunner
    monkeypatch.setitem(sys.modules, "headroom.evals.suite_runner", suite_module)

    args = Namespace(
        tier=2,
        model="gpt-4o-mini",
        budget=12.5,
        port=8787,
        no_proxy=False,
        output=str(tmp_path),
        ci=True,
    )
    with pytest.raises(SystemExit):
        eval_main.cmd_suite(args)
    ci_output = capsys.readouterr().out
    assert "CI FAILURE" in ci_output
    assert "Reports saved:" in ci_output

    suite_module.SuiteRunner = lambda **kwargs: types.SimpleNamespace(
        run=lambda: types.SimpleNamespace(
            all_passed=True,
            benchmarks=[types.SimpleNamespace(name="bench-1", passed=True)],
        )
    )
    eval_main.cmd_suite(Namespace(**{**args.__dict__, "ci": False, "output": None}))
    assert "All benchmarks PASSED." in capsys.readouterr().out

    input_path = tmp_path / "results.json"
    input_path.write_text(
        json.dumps(
            {
                "totals": {
                    "cases": 2,
                    "accuracy_rate": 1.0,
                    "tokens_original": 100,
                    "tokens_compressed": 40,
                },
                "suites": [
                    {
                        "suite_name": "suite-a",
                        "passed_cases": 1,
                        "total_cases": 1,
                        "avg_f1_score": 1.0,
                        "avg_compression_ratio": 0.5,
                        "results": [
                            {
                                "case_id": "1",
                                "accuracy_preserved": True,
                                "f1_score": 1.0,
                                "compression_ratio": 0.5,
                            }
                        ],
                    }
                ],
            }
        )
    )
    eval_main.cmd_report(Namespace(input=str(input_path), output=None))
    html_path = tmp_path / "results.html"
    assert html_path.exists()
    assert "Headroom Evaluation Report" in html_path.read_text()

    with pytest.raises(SystemExit):
        eval_main.cmd_report(Namespace(input=str(tmp_path / "missing.json"), output=None))

    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)

    called = {}
    monkeypatch.setattr(sys, "argv", ["prog", "list"])
    monkeypatch.setattr(eval_main, "cmd_list", lambda args: called.setdefault("list", True))
    eval_main.main()
    assert called["list"] is True

    monkeypatch.delitem(sys.modules, "dotenv", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        eval_main.main()
