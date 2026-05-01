from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_runner_v3(monkeypatch):
    locomo_module = types.ModuleType("headroom.evals.memory.locomo")
    locomo_module.LOCOMO_CATEGORIES = {1: "profile", 2: "event"}
    locomo_module.LoCoMoCase = object
    locomo_module.LoCoMoConversation = object
    locomo_module.load_locomo = lambda **kwargs: []

    class LocalBackend:
        pass

    local_module = types.ModuleType("headroom.memory.backends.local")
    local_module.LocalBackend = LocalBackend
    local_module.LocalBackendConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    class Mem0Backend:
        pass

    mem0_module = types.ModuleType("headroom.memory.backends.mem0")
    mem0_module.Mem0Backend = Mem0Backend
    mem0_module.Mem0Config = lambda **kwargs: SimpleNamespace(**kwargs)

    models_module = types.ModuleType("headroom.memory.models")
    models_module.Memory = lambda **kwargs: SimpleNamespace(**kwargs)

    ports_module = types.ModuleType("headroom.memory.ports")
    ports_module.MemorySearchResult = object
    ports_module.VectorSearchResult = object

    litellm_module = types.ModuleType("litellm")
    litellm_module.completion = lambda **kwargs: None

    memory_pkg = types.ModuleType("headroom.evals.memory")
    memory_pkg.__path__ = []  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "headroom.evals.memory", memory_pkg)
    monkeypatch.setitem(sys.modules, "headroom.evals.memory.locomo", locomo_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.backends.local", local_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.backends.mem0", mem0_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.models", models_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.ports", ports_module)
    monkeypatch.setitem(sys.modules, "litellm", litellm_module)
    monkeypatch.delitem(sys.modules, "headroom.evals.memory.runner_v3", raising=False)
    module_path = (
        Path(__file__).resolve().parents[1] / "headroom" / "evals" / "memory" / "runner_v3.py"
    )
    spec = importlib.util.spec_from_file_location("headroom.evals.memory.runner_v3", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.evals.memory.runner_v3"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_runner_v3_aggregate_summary_and_save(monkeypatch, tmp_path: Path) -> None:
    runner_v3 = _load_runner_v3(monkeypatch)
    case1 = SimpleNamespace(
        question="Q1", answer="A1", category=1, category_name="profile", evidence=["t1"]
    )
    case2 = SimpleNamespace(
        question="Q2", answer="A2", category=2, category_name="event", evidence=["t2"]
    )
    result1 = runner_v3.CaseResultV3(
        case=case1,
        retrieved_turn_ids=["t1"],
        evidence_turn_ids=["t1"],
        retrieval_recall=1.0,
        retrieval_rank=1,
        predicted_answer="A1",
        is_correct=True,
        judge_score=0.9,
    )
    result2 = runner_v3.CaseResultV3(
        case=case2,
        retrieved_turn_ids=["x"],
        evidence_turn_ids=["t2"],
        retrieval_recall=0.0,
        retrieval_rank=None,
        predicted_answer="wrong",
        is_correct=False,
        judge_score=0.1,
    )

    evaluator = runner_v3.LoCoMoEvaluatorV3()
    aggregated = evaluator._aggregate_results([result1, result2], 12.5)
    assert aggregated.avg_retrieval_recall == 0.5
    assert aggregated.avg_mrr == 0.5
    assert aggregated.accuracy == 0.5
    assert "LoCoMo V3 Evaluation Results" in aggregated.summary()

    empty = evaluator._aggregate_results([], 3.0)
    assert empty.total_cases == 0
    assert empty.avg_mrr == 0.0

    save_path = tmp_path / "locomo.json"
    aggregated.save(save_path)
    assert "predicted" in save_path.read_text()


@pytest.mark.asyncio
async def test_runner_v3_generation_evaluation_and_wrappers(monkeypatch, tmp_path: Path) -> None:
    runner_v3 = _load_runner_v3(monkeypatch)
    completion_calls = []

    def fake_completion(**kwargs):
        completion_calls.append(kwargs)
        if len(completion_calls) == 1:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Predicted"))]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"score": 0.8, "reasoning": "good"}')
                )
            ]
        )

    monkeypatch.setattr(runner_v3.litellm, "completion", fake_completion)
    evaluator = runner_v3.LoCoMoEvaluatorV3(runner_v3.EvalConfigV3(use_llm_judge=True))
    predicted, judge_score, judge_reasoning = await evaluator._generate_answer(
        "Question?",
        "Ground truth",
        "Context",
        "Alice",
        "Bob",
    )
    assert predicted == "Predicted"
    assert judge_score == 0.8
    assert judge_reasoning == "good"

    monkeypatch.setattr(
        runner_v3.litellm,
        "completion",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    predicted, judge_score, judge_reasoning = await evaluator._generate_answer(
        "Question?",
        "Ground truth",
        "Context",
        "Alice",
        "Bob",
    )
    assert predicted == "Error generating answer"
    assert judge_score is None

    case = SimpleNamespace(
        question="Question?",
        answer="Ground truth",
        evidence=["t1"],
        category=1,
        category_name="profile",
    )
    conv = SimpleNamespace(sample_id="conv1", speaker_a="Alice", speaker_b="Bob")
    memory = SimpleNamespace(content="Stored context", metadata={"dia_id": "t1"})
    result_item = SimpleNamespace(memory=memory)

    async def search_memories(**kwargs):
        return [result_item]

    backend = SimpleNamespace(search_memories=search_memories)
    evaluator = runner_v3.LoCoMoEvaluatorV3(runner_v3.EvalConfigV3(use_llm_judge=False, top_k=3))
    evaluator._backend = backend

    async def fake_generate_answer(*args):
        return "Predicted", 0.75, "reasoned"

    evaluator._generate_answer = fake_generate_answer
    evaluated = await evaluator._evaluate_case(case, conv)
    assert evaluated.retrieval_rank == 1
    assert evaluated.is_correct is True

    conv_store = SimpleNamespace(
        sample_id="conv1",
        sessions=[
            SimpleNamespace(
                datetime="2024-01-01",
                session_num=1,
                dialogues=[
                    SimpleNamespace(dia_id="d1", speaker="Alice", text="Hello"),
                    SimpleNamespace(dia_id="d2", speaker="Bob", text="Hi"),
                ],
            )
        ],
        qa_cases=[],
    )
    saved = []

    async def save_memory(**kwargs):
        saved.append(kwargs)

    evaluator._backend = SimpleNamespace(save_memory=save_memory)
    assert await evaluator._store_dialogue_turns(conv_store) == 2

    async def backend_run(self):
        return runner_v3.EvalResultV3(
            total_cases=1,
            avg_retrieval_recall=1.0,
            avg_mrr=1.0,
            retrieval_by_category={},
            accuracy=1.0,
            accuracy_by_category={},
        )

    monkeypatch.setattr(runner_v3.LoCoMoEvaluatorV3, "run", backend_run)
    output_path = tmp_path / "out.json"
    result = await runner_v3.run_locomo_eval_v3(output_path=output_path)
    assert result.total_cases == 1
    assert output_path.exists()

    monkeypatch.setattr(
        runner_v3.asyncio,
        "run",
        lambda coro: ("sync", coro.close(), result)[2],
    )
    assert runner_v3.run_locomo_eval_v3_sync().total_cases == 1
