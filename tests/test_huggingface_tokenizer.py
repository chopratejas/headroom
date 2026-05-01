from __future__ import annotations

import sys
import types

import pytest

from headroom.tokenizers import huggingface


class _FakeTokenizer:
    def __init__(self, *, template_error: bool = False) -> None:
        self.template_error = template_error

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(text), int(add_special_tokens)]

    def decode(self, tokens: list[int]) -> str:
        return "-".join(str(token) for token in tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        if self.template_error:
            raise RuntimeError("bad template")
        assert tokenize is True
        assert add_generation_prompt is True
        return list(range(len(messages) + 2))


def test_get_tokenizer_name_and_cached_loader(monkeypatch) -> None:
    huggingface._load_tokenizer.cache_clear()
    fake_module = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda name, trust_remote_code=True: {
                "name": name,
                "trust_remote_code": trust_remote_code,
            }
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    loaded = huggingface._load_tokenizer("meta-llama/Meta-Llama-3-8B")
    cached = huggingface._load_tokenizer("meta-llama/Meta-Llama-3-8B")
    assert loaded is cached
    assert loaded["trust_remote_code"] is True

    assert huggingface.get_tokenizer_name("llama-3-8b") == "meta-llama/Meta-Llama-3-8B"
    assert huggingface.get_tokenizer_name("mistral-7b-custom") == "mistralai/Mistral-7B-v0.1"
    assert huggingface.get_tokenizer_name("custom/model") == "custom/model"


def test_loader_failure_and_availability(monkeypatch) -> None:
    huggingface._load_tokenizer.cache_clear()
    fake_module = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    assert huggingface._load_tokenizer("broken/model") is None

    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert huggingface.HuggingFaceTokenizer.is_available() is False
    assert "llama-3" in huggingface.HuggingFaceTokenizer.list_supported_models()


def test_huggingface_tokenizer_counting_and_fallback(monkeypatch) -> None:
    huggingface._load_tokenizer.cache_clear()
    tokenizer = huggingface.HuggingFaceTokenizer("llama-3-8b")
    monkeypatch.setattr(huggingface, "_load_tokenizer", lambda _name: _FakeTokenizer())
    assert tokenizer.tokenizer is not None
    assert tokenizer.count_text("") == 0
    assert tokenizer.count_text("hello") == 2
    assert tokenizer.count_messages([{"role": "user", "content": "hi"}]) == 3
    assert tokenizer.encode("abc") == [3, 0]
    assert tokenizer.decode([1, 2, 3]) == "1-2-3"
    assert "llama-3-8b" in repr(tokenizer)

    template_fallback = huggingface.HuggingFaceTokenizer("mistral")
    template_fallback._tokenizer = _FakeTokenizer(template_error=True)
    assert template_fallback.count_messages([{"role": "user", "content": "abcd"}]) > 0

    unavailable = huggingface.HuggingFaceTokenizer("custom-model")
    monkeypatch.setattr(huggingface, "_load_tokenizer", lambda _name: None)
    assert unavailable.count_text("abcdef") == 2
    with pytest.raises(NotImplementedError):
        unavailable.encode("abc")
    with pytest.raises(NotImplementedError):
        unavailable.decode([1, 2, 3])
