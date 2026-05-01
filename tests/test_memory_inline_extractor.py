from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_inline_extractor():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "inline_extractor.py"
    spec = importlib.util.spec_from_file_location("headroom.memory.inline_extractor", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.inline_extractor"] = module
    spec.loader.exec_module(module)
    return module


def test_inject_memory_instruction_preserves_original_and_handles_system_message() -> None:
    inline = _load_inline_extractor()
    original = [{"role": "user", "content": "hi"}]
    injected = inline.inject_memory_instruction(original)
    assert original == [{"role": "user", "content": "hi"}]
    assert injected[0]["role"] == "system"
    assert inline.MEMORY_INSTRUCTION_SHORT.strip() in injected[0]["content"]

    with_system = [{"role": "system", "content": "base"}, {"role": "user", "content": "hi"}]
    expanded = inline.inject_memory_instruction(with_system, short=False)
    assert expanded[0]["content"].endswith(inline.MEMORY_INSTRUCTION)


def test_parse_response_with_memory_variants() -> None:
    inline = _load_inline_extractor()
    parsed = inline.parse_response_with_memory(
        'Hello there\n<memory>{"memories":[{"content":"User likes Python"}]}</memory>'
    )
    assert isinstance(parsed, inline.ParsedResponse)
    assert parsed.content == "Hello there"
    assert parsed.memories == [{"content": "User likes Python"}]

    empty = inline.parse_response_with_memory("No memory block here")
    assert empty.content == "No memory block here"
    assert empty.memories == []

    invalid = inline.parse_response_with_memory("Hi<memory>{bad json}</memory>")
    assert invalid.content == "Hi"
    assert invalid.memories == []


def test_inline_memory_wrapper_chat_and_chat_with_response() -> None:
    inline = _load_inline_extractor()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='Sure!<memory>{"memories":[{"content":"Prefers Python"}]}</memory>'
                )
            )
        ]
    )
    completions = SimpleNamespace(create=lambda **kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    wrapper = inline.InlineMemoryWrapper(client)

    content, memories = wrapper.chat(
        [{"role": "user", "content": "I prefer Python"}], temperature=0
    )
    assert content == "Sure!"
    assert memories == [{"content": "Prefers Python"}]

    response2 = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='Answer<memory>{"memories":[]}</memory>')
            )
        ]
    )
    client2 = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response2))
    )
    wrapper2 = inline.InlineMemoryWrapper(client2)
    full_response, clean_content, clean_memories = wrapper2.chat_with_response(
        [{"role": "user", "content": "Hello"}],
        model="gpt-test",
    )
    assert full_response is response2
    assert clean_content == "Answer"
    assert clean_memories == []
    assert response2.choices[0].message.content == "Answer"
