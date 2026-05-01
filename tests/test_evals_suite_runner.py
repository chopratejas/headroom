from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import headroom.evals.suite_runner as suite_runner


def test_suite_runner_env_and_proxy_helpers(monkeypatch, tmp_path: Path) -> None:
    env_calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(load_dotenv=lambda path: env_calls.append(path)),
    )
    env_path = tmp_path / ".env"
    env_path.write_text("X=1")

    monkeypatch.setattr(suite_runner.os.path, "abspath", lambda path: str(tmp_path / "x" / "y"))
    monkeypatch.setattr(suite_runner.os.path, "dirname", os.path.dirname)
    monkeypatch.setattr(
        suite_runner.os.path,
        "exists",
        lambda path: path == str(env_path),
    )
    suite_runner._load_env()
    assert env_calls == [str(env_path)]

    class _SocketOK:
        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, addr):
            self.addr = addr

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(suite_runner.socket, "socket", lambda *args: _SocketOK())
    assert suite_runner._check_proxy(8787) is True

    class _SocketFail(_SocketOK):
        def connect(self, addr):
            raise ConnectionRefusedError("nope")

    monkeypatch.setattr(suite_runner.socket, "socket", lambda *args: _SocketFail())
    assert suite_runner._check_proxy(8787) is False

    class _Proc:
        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    monkeypatch.setattr(suite_runner.time, "sleep", lambda _secs: None)
    monkeypatch.setattr(suite_runner, "_check_proxy", lambda port: True)
    monkeypatch.setattr(suite_runner.subprocess, "Popen", lambda *args, **kwargs: _Proc())
    assert isinstance(suite_runner._start_proxy(8787), _Proc)

    proc = _Proc()
    monkeypatch.setattr(suite_runner, "_check_proxy", lambda port: False)
    monkeypatch.setattr(suite_runner.subprocess, "Popen", lambda *args, **kwargs: proc)
    assert suite_runner._start_proxy(8787) is None
    assert proc.killed is True

    monkeypatch.setattr(
        suite_runner.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("spawn failed")),
    )
    assert suite_runner._start_proxy(8787) is None


def test_suite_runner_benchmark_helpers(monkeypatch) -> None:
    monkeypatch.setattr(suite_runner, "_load_env", lambda: None)
    runner = suite_runner.SuiteRunner(model="gpt-4o-mini", tiers=[1], auto_start_proxy=False)

    comparisons = [
        SimpleNamespace(
            baseline_score=0.9, headroom_score=0.89, delta=-0.01, accuracy_preserved=True
        )
    ]
    monkeypatch.setitem(
        sys.modules,
        "headroom.evals.comprehensive_benchmark",
        types.SimpleNamespace(
            run_baseline_benchmark=lambda **kwargs: {"baseline": True},
            run_headroom_benchmark=lambda **kwargs: {"headroom": True},
            compare_results=lambda baseline, headroom: comparisons,
        ),
    )
    lm_spec = suite_runner.BenchmarkSpec(
        name="Task",
        category="reasoning",
        tier=1,
        runner_type="lm_eval",
        sample_size=5,
        lm_eval_tasks=["gsm8k"],
    )
    lm_result = runner._run_lm_eval_benchmark(lm_spec, tracker=None)
    assert lm_result["baseline_score"] == 0.9
    assert lm_result["passed"] is True

    monkeypatch.setitem(
        sys.modules,
        "headroom.evals.core",
        types.SimpleNamespace(
            EvalMode=SimpleNamespace(GROUND_TRUTH="GROUND_TRUTH", BEFORE_AFTER="BEFORE_AFTER")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "headroom.evals.datasets",
        types.SimpleNamespace(
            load_dataset_by_name=lambda name, n: [f"{name}-{n}"],
            load_tool_output_samples=lambda: ["tool"],
        ),
    )

    class _BeforeAfterRunner:
        def __init__(self, llm_config, use_semantic_similarity):
            self.llm_config = llm_config

        def run(self, suite, progress_callback, mode):
            progress_callback(
                1,
                1,
                SimpleNamespace(accuracy_preserved=True, f1_score=1.0, contains_ground_truth=True),
            )
            return SimpleNamespace(
                accuracy_preservation_rate=0.91,
                avg_compression_ratio=0.6,
                total_tokens_saved=100,
                total_cases=1,
                duration_seconds=2.5,
            )

    monkeypatch.setitem(
        sys.modules,
        "headroom.evals.runners.before_after",
        types.SimpleNamespace(
            BeforeAfterRunner=_BeforeAfterRunner,
            LLMConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setattr(suite_runner, "_check_proxy", lambda port: True)
    before_after_spec = suite_runner.BenchmarkSpec(
        name="Tool Outputs",
        category="agent",
        tier=1,
        runner_type="before_after",
        sample_size=3,
        dataset_name="tool_outputs",
        provider="openai",
        eval_mode="ground_truth",
    )
    before_after = runner._run_before_after_benchmark(before_after_spec, tracker=None)
    assert before_after["accuracy_rate"] == 0.91
    assert before_after["tokens_saved"] == 100

    class _CompressionOnlyRunner:
        def generate_ccr_test_cases(self, n):
            return [n]

        def evaluate_ccr_lossless(self, cases):
            return SimpleNamespace(
                accuracy_rate=1.0,
                avg_compression_ratio=0.5,
                total_tokens_saved=50,
                passed=True,
                total_cases=len(cases),
                duration_seconds=1.0,
            )

        def generate_info_retention_cases(self, n):
            return [n]

        def evaluate_information_retention(self, cases):
            return SimpleNamespace(
                accuracy_rate=0.8,
                avg_compression_ratio=0.3,
                total_tokens_saved=20,
                passed=False,
                total_cases=len(cases),
                duration_seconds=1.5,
            )

    monkeypatch.setitem(
        sys.modules,
        "headroom.evals.runners.compression_only",
        types.SimpleNamespace(CompressionOnlyRunner=_CompressionOnlyRunner),
    )
    ccr_spec = suite_runner.BenchmarkSpec(
        name="CCR Round-trip",
        category="lossless",
        tier=1,
        runner_type="compression_only",
        sample_size=4,
    )
    assert runner._run_compression_only_benchmark(ccr_spec)["passed"] is True

    info_spec = suite_runner.BenchmarkSpec(
        name="Info Retention",
        category="compression",
        tier=1,
        runner_type="compression_only",
        sample_size=4,
    )
    assert runner._run_compression_only_benchmark(info_spec)["passed"] is False

    with pytest.raises(ValueError):
        runner._run_compression_only_benchmark(
            suite_runner.BenchmarkSpec(
                name="Unknown",
                category="compression",
                tier=1,
                runner_type="compression_only",
                sample_size=1,
            )
        )


def test_suite_runner_run_and_cleanup(monkeypatch, capsys) -> None:
    monkeypatch.setattr(suite_runner, "_load_env", lambda: None)

    report_card = __import__("headroom.evals.reports.report_card", fromlist=["dummy"])
    monkeypatch.setattr(report_card, "generate_markdown", lambda result: "MARKDOWN")

    class _Tracker:
        def __init__(self, budget_usd):
            self.budget_usd = budget_usd
            self.spent_usd = 1.25
            self.remaining_usd = 0.05

        def can_afford(self, model, sample_size, avg_input_tokens):
            return sample_size < 5

        def print_summary(self):
            print("TRACKER SUMMARY")

    monkeypatch.setitem(
        sys.modules, "headroom.evals.cost_tracker", types.SimpleNamespace(CostTracker=_Tracker)
    )

    specs = [
        suite_runner.BenchmarkSpec(
            name="LM Eval",
            category="reasoning",
            tier=1,
            runner_type="lm_eval",
            sample_size=2,
        ),
        suite_runner.BenchmarkSpec(
            name="Budget Skip",
            category="qa",
            tier=1,
            runner_type="before_after",
            sample_size=10,
        ),
        suite_runner.BenchmarkSpec(
            name="BeforeAfter",
            category="qa",
            tier=1,
            runner_type="before_after",
            sample_size=2,
        ),
        suite_runner.BenchmarkSpec(
            name="Compression",
            category="lossless",
            tier=1,
            runner_type="compression_only",
            sample_size=2,
        ),
    ]
    monkeypatch.setattr(suite_runner, "BENCHMARK_SUITE", specs)

    runner = suite_runner.SuiteRunner(model="gpt-4o-mini", tiers=[1], auto_start_proxy=False)
    cleaned = {"called": False}
    monkeypatch.setattr(runner, "_ensure_proxy", lambda: False)
    monkeypatch.setattr(
        runner, "_run_lm_eval_benchmark", lambda spec, tracker: {"passed": True, "n_samples": 2}
    )
    monkeypatch.setattr(
        runner,
        "_run_before_after_benchmark",
        lambda spec, tracker: {
            "accuracy_rate": 0.95,
            "avg_compression_ratio": 0.4,
            "tokens_saved": 30,
            "passed": True,
            "n_samples": 2,
            "duration_seconds": 1.0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_run_compression_only_benchmark",
        lambda spec: (_ for _ in ()).throw(RuntimeError("compression failed")),
    )
    monkeypatch.setattr(runner, "_cleanup_proxy", lambda: cleaned.__setitem__("called", True))

    result = runner.run()
    assert cleaned["called"] is True
    assert len(result.benchmarks) == 4
    assert result.benchmarks[0].error == "Proxy not available"
    assert result.benchmarks[1].error == "Budget exceeded"
    assert result.benchmarks[2].passed is True
    assert result.benchmarks[3].error == "compression failed"

    output = capsys.readouterr().out
    assert "WARNING: Headroom proxy not available" in output
    assert "TRACKER SUMMARY" in output
    assert "MARKDOWN" in output
