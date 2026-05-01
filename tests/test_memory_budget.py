from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest


def _load_budget(monkeypatch):
    @dataclass
    class MemoryEntry:
        content: str
        importance: float
        created_at: float
        access_count: int = 0
        entity_refs: list[str] = field(default_factory=list)
        score: float = 0.0

    writers_base = types.ModuleType("headroom.memory.writers.base")
    writers_base.MemoryEntry = MemoryEntry
    writers_base._estimate_tokens = lambda text: len(text.split())
    monkeypatch.setitem(sys.modules, "headroom.memory.writers.base", writers_base)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "budget.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.budget", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.budget", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.budget"] = module
    spec.loader.exec_module(module)
    return module, MemoryEntry


def test_budget_manager_optimize_decay_staleness_and_merge(monkeypatch, tmp_path: Path) -> None:
    budget_module, MemoryEntry = _load_budget(monkeypatch)
    monkeypatch.setattr(budget_module.time, "time", lambda: 1_000_000.0)

    config = budget_module.BudgetConfig(
        agent_budgets={"generic": 14}, similarity_merge_threshold=0.4
    )
    manager = budget_module.MemoryBudgetManager(project_path=tmp_path, config=config)

    kept_path = tmp_path / "keep.txt"
    kept_path.write_text("ok")
    stale_abs_path = tmp_path / "missing.txt"

    fresh = MemoryEntry(
        content="keep this memory",
        importance=0.9,
        created_at=1_000_000.0,
        access_count=2,
        entity_refs=[f"./{kept_path.name}"],
        score=0.95,
    )
    stale = MemoryEntry(
        content=f"references `{stale_abs_path.as_posix()}`",
        importance=0.8,
        created_at=1_000_000.0,
        entity_refs=[f"./{stale_abs_path.name}"],
        score=0.8,
    )
    similar_a = MemoryEntry(
        content="alpha beta gamma",
        importance=0.7,
        created_at=1_000_000.0,
        entity_refs=["one"],
        score=0.7,
    )
    similar_b = MemoryEntry(
        content="alpha beta delta",
        importance=0.6,
        created_at=1_000_000.0,
        access_count=3,
        entity_refs=["two"],
        score=0.6,
    )
    old_low = MemoryEntry(
        content="old and weak",
        importance=0.1,
        created_at=1_000_000.0 - 86400 * 100,
        score=0.1,
    )
    over_budget = MemoryEntry(
        content="this entry gets pruned by budget",
        importance=0.4,
        created_at=1_000_000.0,
        score=0.1,
    )

    monkeypatch.setattr(manager, "_get_git_files", lambda: {kept_path.name})
    optimized, report = manager.optimize([fresh, stale, similar_a, similar_b, old_low, over_budget])

    assert [entry.content for entry in optimized] == ["keep this memory"]
    assert report.total_memories == 6
    assert report.kept == 1
    assert report.pruned_staleness == 1
    assert report.pruned_budget == 2
    assert report.merged == 1
    assert report.tokens_before > report.tokens_after
    assert report.tokens_saved == report.tokens_before - report.tokens_after
    assert fresh.importance > 0.9
    merged_entry = next(
        entry
        for entry in [fresh, similar_a, similar_b, over_budget]
        if set(entry.entity_refs) == {"one", "two"}
    )
    assert merged_entry.access_count == 3


def test_budget_manager_helpers_git_cache_and_similarity(monkeypatch, tmp_path: Path) -> None:
    budget_module, MemoryEntry = _load_budget(monkeypatch)
    manager = budget_module.MemoryBudgetManager(project_path=tmp_path)

    files_result = types.SimpleNamespace(returncode=0, stdout="a.txt\nb.txt\n")
    monkeypatch.setattr(budget_module.subprocess, "run", lambda *args, **kwargs: files_result)
    assert manager._get_git_files() == {"a.txt", "b.txt"}
    monkeypatch.setattr(
        budget_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache should be reused")),
    )
    assert manager._get_git_files() == {"a.txt", "b.txt"}

    failed = budget_module.MemoryBudgetManager(project_path=tmp_path)
    monkeypatch.setattr(
        budget_module.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=1, stdout=""),
    )
    assert failed._get_git_files() == set()

    timed_out = budget_module.MemoryBudgetManager(project_path=tmp_path)

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(budget_module.subprocess, "run", raise_timeout)
    assert timed_out._get_git_files() == set()

    no_git = budget_module.MemoryBudgetManager(project_path=tmp_path)

    def raise_missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(budget_module.subprocess, "run", raise_missing)
    assert no_git._get_git_files() == set()

    assert budget_module.MemoryBudgetManager._text_similarity(
        "alpha beta", "alpha gamma"
    ) == pytest.approx(1 / 3)
    assert budget_module.MemoryBudgetManager._text_similarity("", "alpha") == 0.0

    singleton = MemoryEntry(content="solo", importance=0.5, created_at=0, score=0.5)
    assert manager._merge_similar([singleton]) == [singleton]
