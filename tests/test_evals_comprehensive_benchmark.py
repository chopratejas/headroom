from __future__ import annotations

import json
import socket
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import headroom.evals.comprehensive_benchmark as cb


def test_benchmark_result_models_and_comparisons(capsys) -> None:
    bench = cb.BenchmarkResult(
        task="gsm8k",
        metric="exact_match",
        score=0.91,
        stderr=0.02,
        samples=5,
        duration_seconds=12.3,
    )
    assert bench.to_dict()["score"] == 0.91

    comparison = cb.ComparisonResult(
        task="gsm8k",
        metric="exact_match",
        baseline_score=0.9,
        headroom_score=0.89,
    )
    assert comparison.delta == pytest.approx(-0.01)
    assert comparison.accuracy_preserved is True
    assert comparison.accuracy_improved is False
    assert comparison.to_dict()["preserved"] is True

    suite = cb.BenchmarkSuiteResult(
        model="gpt-4o-mini",
        baseline_results=[bench],
        headroom_results=[bench],
        comparisons=[comparison],
        total_duration_seconds=20.5,
    )
    assert suite.all_preserved is True
    assert suite.avg_delta == pytest.approx(-0.01)
    assert suite.summary()["model"] == "gpt-4o-mini"
    suite.print_summary()
    output = capsys.readouterr().out
    assert "HEADROOM COMPREHENSIVE BENCHMARK RESULTS" in output
    assert "gsm8k" in output

    headroom_only = cb.BenchmarkSuiteResult(model="gpt-4o-mini", headroom_results=[bench])
    headroom_only.print_summary()
    assert "Headroom results only" in capsys.readouterr().out


def test_run_lm_eval_parse_and_compare(monkeypatch, tmp_path: Path) -> None:
    results_dir = tmp_path / "lm_eval"
    results_dir.mkdir()
    result_file = results_dir / "results_123.json"
    result_file.write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,flexible-extract": 0.91,
                        "exact_match_stderr,flexible-extract": 0.01,
                    },
                    "mmlu": {"acc,none": 0.72},
                }
            }
        )
    )

    recorded = {}

    def fake_run(cmd, capture_output, text, env):
        recorded["cmd"] = cmd
        recorded["env"] = env
        return SimpleNamespace(returncode=0, stderr="", stdout="stdout")

    monkeypatch.setattr(cb.subprocess, "run", fake_run)
    time_calls = {"count": 0}

    def fake_time():
        time_calls["count"] += 1
        if time_calls["count"] == 1:
            return 100.0
        if time_calls["count"] == 2:
            return 106.0
        return 200.0

    monkeypatch.setattr(cb.time, "time", fake_time)

    raw = cb.run_lm_eval(
        model="openai-chat-completions",
        model_args="model=gpt-4o-mini",
        tasks=["gsm8k", "humaneval"],
        num_fewshot=2,
        limit=5,
        output_path=str(results_dir),
        base_url="http://localhost:8787",
    )
    assert "--confirm_run_unsafe_code" in recorded["cmd"]
    assert raw["_duration_seconds"] == 6.0

    parsed = cb.parse_lm_eval_results(raw)
    assert parsed[0].task == "gsm8k"
    assert parsed[0].metric == "exact_match_flexible-extract"
    assert parsed[1].metric == "acc"

    fallback = cb.parse_lm_eval_results(
        {
            "_duration_seconds": 4.0,
            "results": {"truthfulqa_gen": {"rouge1_acc,none": 0.3}, "other": {"metric,none": 0.5}},
        }
    )
    assert fallback[0].task == "truthfulqa_gen"
    assert fallback[1].metric == "metric"

    comparisons = cb.compare_results(
        [
            cb.BenchmarkResult(task="gsm8k", metric="acc", score=0.9),
            cb.BenchmarkResult(task="mmlu", metric="acc", score=0.7),
        ],
        [cb.BenchmarkResult(task="gsm8k", metric="acc", score=0.91)],
    )
    assert len(comparisons) == 1
    assert comparisons[0].task == "gsm8k"

    monkeypatch.setattr(
        cb.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="bad run", stdout=""),
    )
    monkeypatch.setattr(cb.logger, "error", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError):
        cb.run_lm_eval(output_path=str(results_dir))

    monkeypatch.setattr(
        cb.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="", stdout="stdout-only"),
    )
    stdout_only = cb.run_lm_eval(output_path=str(tmp_path / "missing"))
    assert stdout_only["_stdout"] == "stdout-only"


def test_comprehensive_benchmark_wrappers_and_main(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        cb,
        "run_lm_eval",
        lambda **kwargs: {
            "_duration_seconds": 4.0,
            "results": {"gsm8k": {"exact_match,flexible-extract": 0.91}},
        },
    )
    baseline = cb.run_baseline_benchmark(model="gpt-4o-mini", tasks=["gsm8k"], limit=5)
    assert baseline[0].task == "gsm8k"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        cb.run_headroom_benchmark()

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    headroom = cb.run_headroom_benchmark(
        model="gpt-4o-mini", tasks=["gsm8k"], limit=5, headroom_port=9999
    )
    assert headroom[0].task == "gsm8k"

    monkeypatch.setattr(
        cb,
        "run_baseline_benchmark",
        lambda **kwargs: [cb.BenchmarkResult(task="gsm8k", metric="acc", score=0.9)],
    )
    monkeypatch.setattr(
        cb,
        "run_headroom_benchmark",
        lambda **kwargs: [cb.BenchmarkResult(task="gsm8k", metric="acc", score=0.91)],
    )
    suite = cb.run_comprehensive_benchmark(compare_baseline=True)
    assert suite.comparisons[0].accuracy_improved is True

    monkeypatch.setattr(cb, "run_headroom_benchmark", lambda **kwargs: [])
    headroom_only_suite = cb.run_comprehensive_benchmark(compare_baseline=False)
    assert headroom_only_suite.baseline_results == []

    class _SocketOK:
        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, addr):
            self.addr = addr

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(socket, "socket", lambda *args: _SocketOK())
    assert cb.check_headroom_proxy(8787) is True

    class _SocketFail(_SocketOK):
        def connect(self, addr):
            raise TimeoutError("bad")

    monkeypatch.setattr(socket, "socket", lambda *args: _SocketFail())
    assert cb.check_headroom_proxy(8787) is False

    dotenv_module = types.SimpleNamespace(load_dotenv=lambda: None)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)
    cb._load_env()

    output_path = tmp_path / "results.json"
    fake_suite = SimpleNamespace(
        all_preserved=True,
        summary=lambda: {"ok": True},
        print_summary=lambda: print("SUMMARY"),
    )
    monkeypatch.setattr(cb, "_load_env", lambda: None)
    monkeypatch.setattr(cb, "check_headroom_proxy", lambda port: True)
    monkeypatch.setattr(cb, "run_comprehensive_benchmark", lambda **kwargs: fake_suite)
    monkeypatch.setattr(sys, "argv", ["prog", "--quick", "--compare", "--output", str(output_path)])
    with pytest.raises(SystemExit) as ok_exit:
        cb.main()
    assert ok_exit.value.code == 0
    assert json.loads(output_path.read_text()) == {"ok": True}
    assert "Results saved to" in capsys.readouterr().out

    monkeypatch.setattr(cb, "check_headroom_proxy", lambda port: False)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit) as proxy_exit:
        cb.main()
    assert proxy_exit.value.code == 1

    monkeypatch.setattr(cb, "check_headroom_proxy", lambda port: True)
    monkeypatch.setattr(
        cb,
        "run_comprehensive_benchmark",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit) as fail_exit:
        cb.main()
    assert fail_exit.value.code == 1

    fail_suite = SimpleNamespace(
        all_preserved=False,
        summary=lambda: {"ok": False},
        print_summary=lambda: None,
    )
    monkeypatch.setattr(cb, "run_comprehensive_benchmark", lambda **kwargs: fail_suite)
    monkeypatch.setattr(sys, "argv", ["prog", "--compare"])
    with pytest.raises(SystemExit) as compare_exit:
        cb.main()
    assert compare_exit.value.code == 1
