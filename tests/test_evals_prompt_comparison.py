from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import headroom.evals.prompt_comparison as pc


def test_prompt_comparison_result_and_parser(tmp_path: Path) -> None:
    result = pc.PromptComparisonResult(
        original="orig",
        headroom_modified="mod",
        are_equivalent=False,
        confidence="HIGH",
        differences=["tone", "detail"],
        reasoning="meaning changed",
        concern_level="CRITICAL",
        judge_model="gpt-4o-mini",
        timestamp="2026-04-24T00:00:00",
        raw_judge_response="raw",
        metadata={"x": 1},
    )
    restored = pc.PromptComparisonResult.from_dict(result.to_dict())
    assert restored == result
    assert result.is_concerning() is True
    assert "Prompt Comparison Result: FAIL" in result.summary()

    parsed = pc._parse_judge_response(
        "EQUIVALENT: YES\nCONFIDENCE: LOW\nDIFFERENCES: tone, wording\nREASONING: same meaning\nCONCERN_LEVEL: LOW"
    )
    assert parsed["are_equivalent"] is True
    assert parsed["differences"] == ["tone", "wording"]
    assert parsed["reasoning"] == "same meaning"

    parsed_none = pc._parse_judge_response("EQUIVALENT: NO\nDIFFERENCES: None")
    assert parsed_none["differences"] == []

    comparer = pc.PromptComparer(judge_model="gpt-4o", log_dir=tmp_path / "logs")
    result_ok = pc.PromptComparisonResult(
        original="a",
        headroom_modified="a",
        are_equivalent=True,
        concern_level="NONE",
    )
    comparer.comparison_history = [result_ok, result]
    assert comparer.get_concerning_results() == [result]
    summary = comparer.summary()
    assert "Total comparisons: 2" in summary
    assert "Concerning comparisons:" in summary

    empty = pc.PromptComparer()
    assert empty.summary() == "No comparisons performed yet."

    comparer.log_result(result_ok)
    comparer.log_result(result)
    log_file = tmp_path / "logs" / "comparisons.jsonl"
    assert log_file.exists()
    assert len(log_file.read_text().strip().splitlines()) == 2

    export_path = tmp_path / "exports" / "history.json"
    comparer.export_history(export_path)
    exported = json.loads(export_path.read_text())
    assert exported["summary"]["concerning_count"] == 1


def test_compare_prompts_and_messages(monkeypatch) -> None:
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "EQUIVALENT: YES\nCONFIDENCE: HIGH\nDIFFERENCES: None\n"
                        "REASONING: same\nCONCERN_LEVEL: NONE"
                    )
                )
            )
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: fake_response))
    )
    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=lambda api_key=None: fake_client)
    )

    compared = pc.compare_prompts(
        original_prompt="What is 2+2?",
        headroom_modified_prompt="What is 2+2?",
        api_key="secret",
        metadata={"kind": "unit"},
    )
    assert compared.are_equivalent is True
    assert compared.metadata["kind"] == "unit"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        pc.compare_prompts("a", "b")

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError):
        pc.compare_prompts("a", "b", api_key="secret")

    calls = []

    def fake_compare_prompts(**kwargs):
        calls.append(kwargs)
        return pc.PromptComparisonResult(
            original=kwargs["original_prompt"],
            headroom_modified=kwargs["headroom_modified_prompt"],
            are_equivalent=True,
            metadata=kwargs.get("metadata", {}),
        )

    monkeypatch.setattr(pc, "compare_prompts", fake_compare_prompts)
    comparer = pc.PromptComparer(api_key="secret")
    result = comparer.compare("orig", "mod", metadata={"id": 1})
    assert comparer.comparison_history == [result]

    compare_and_log_calls = []
    monkeypatch.setattr(comparer, "log_result", lambda result: compare_and_log_calls.append(result))
    comparer.compare_and_log("orig2", "mod2")
    assert compare_and_log_calls

    message_result = pc.compare_messages(
        original_messages=[
            {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url"}]},
            {"role": "assistant", "content": "answer"},
        ],
        modified_messages=[{"role": "user", "content": "hello [IMAGE]"}],
        api_key="secret",
    )
    assert message_result.metadata["comparison_type"] == "messages"
    assert calls[-1]["original_prompt"].startswith("[user]: hello [IMAGE]")


def test_batch_compare_and_verify(monkeypatch) -> None:
    monkeypatch.setattr(
        pc,
        "compare_prompts",
        lambda **kwargs: pc.PromptComparisonResult(
            original=kwargs["original_prompt"],
            headroom_modified=kwargs["headroom_modified_prompt"],
            are_equivalent=kwargs["original_prompt"] == kwargs["headroom_modified_prompt"],
            differences=["changed"]
            if kwargs["original_prompt"] != kwargs["headroom_modified_prompt"]
            else [],
            reasoning="checked",
            metadata=kwargs.get("metadata", {}),
        ),
    )

    batch = pc.batch_compare_prompts(
        [("a", "a"), ("b", "c"), ("d", "d")],
        max_concurrent=2,
    )
    assert [r.metadata["batch_index"] for r in batch] == [0, 1, 2]
    assert batch[1].are_equivalent is False

    verified = pc.verify_headroom_preservation(
        [{"role": "user", "content": "same"}],
        [{"role": "user", "content": "same"}],
    )
    assert verified.are_equivalent is True

    with pytest.raises(ValueError):
        pc.verify_headroom_preservation(
            [{"role": "user", "content": "orig"}],
            [{"role": "user", "content": "changed"}],
            fail_on_difference=True,
        )
