from __future__ import annotations

import builtins
import sys
import types

import pytest

import headroom.transforms.code_compressor as code_compressor
from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressionResult,
    CodeCompressorConfig,
    CodeLanguage,
    CodeStructure,
    DocstringMode,
    LangConfig,
    _count_error_nodes,
    _detect_indent,
    _get_body_limit,
    _get_definition_name,
    _get_node_text,
    _get_parser,
    _has_syntax_issues,
    _is_public_symbol,
    _make_omitted_comment,
    _SymbolAnalysis,
    compress_code,
    detect_language,
)


class _Node:
    def __init__(
        self,
        node_type: str,
        children: list[_Node] | None = None,
        *,
        is_missing: bool = False,
        text: bytes | str | None = None,
        start_byte: int = 0,
        end_byte: int = 0,
        start_point: tuple[int, int] = (0, 0),
        end_point: tuple[int, int] = (0, 0),
        is_named: bool = True,
    ):
        self.type = node_type
        self.children = children or []
        self.is_missing = is_missing
        self.child_count = len(self.children)
        self.text = text if text is not None else node_type
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.start_point = start_point
        self.end_point = end_point
        self.is_named = is_named


def test_get_parser_caches_results_and_wraps_errors(monkeypatch) -> None:
    code_compressor._tree_sitter_languages.clear()
    original_import = builtins.__import__

    calls: list[str] = []

    def fake_get_parser(language: str):
        calls.append(language)
        return {"language": language}

    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: True)
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda name, *args, **kwargs: (
            types.SimpleNamespace(get_parser=fake_get_parser)
            if name == "tree_sitter_language_pack"
            else original_import(name, *args, **kwargs)
        ),
    )
    try:
        first = _get_parser("python")
        second = _get_parser("python")
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)

    assert first == {"language": "python"}
    assert second is first
    assert calls == ["python"]

    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: False)
    with pytest.raises(ImportError):
        _get_parser("python")

    code_compressor._tree_sitter_languages.clear()

    def failing_import(name, *args, **kwargs):
        if name == "tree_sitter_language_pack":
            return types.SimpleNamespace(
                get_parser=lambda language: (_ for _ in ()).throw(RuntimeError("bad parser"))
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: True)
    monkeypatch.setattr(builtins, "__import__", failing_import)
    try:
        with pytest.raises(ValueError, match="Language 'python'"):
            _get_parser("python")
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)


def test_count_error_nodes_recurses_over_missing_and_error_children() -> None:
    tree = _Node(
        "root",
        [
            _Node("ERROR"),
            _Node("ok", [_Node("child", is_missing=True)]),
            _Node("ok"),
        ],
    )

    assert _count_error_nodes(tree) == 2


def test_detect_language_covers_empty_prefilter_fallback_and_tree_sitter_paths(monkeypatch) -> None:
    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: False)
    assert detect_language("") == (CodeLanguage.UNKNOWN, 0.0)
    assert detect_language("plain prose without code markers") == (CodeLanguage.UNKNOWN, 0.0)

    ts_code = (
        "interface User {}\nconst value: string = 'x';\nexport function use(): void { return; }\n"
    )
    lang, confidence = detect_language(ts_code)
    assert lang == CodeLanguage.TYPESCRIPT
    assert confidence >= 0.3

    cpp_code = "#include <vector>\nnamespace demo { int main() { return 0; } }\n"
    lang, confidence = detect_language(cpp_code)
    assert lang == CodeLanguage.CPP
    assert confidence >= 0.3

    class _Parser:
        def __init__(self, root: _Node) -> None:
            self._root = root

        def parse(self, code_bytes: bytes):
            return types.SimpleNamespace(root_node=self._root)

    def fake_get_parser(language: str):
        if language == "python":
            return _Parser(_Node("module", [_Node("def"), _Node("class")]))
        return _Parser(_Node("module", [_Node("ERROR")]))

    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: True)
    monkeypatch.setattr(code_compressor, "_get_parser", fake_get_parser)
    lang, confidence = detect_language("def run(x):\n    return x\nclass Demo:\n    pass\n")
    assert lang == CodeLanguage.PYTHON
    assert 0.3 <= confidence <= 1.0


def test_code_compression_result_summary_includes_semantic_counts() -> None:
    result = CodeCompressionResult(
        compressed="short",
        original="longer content",
        original_tokens=100,
        compressed_tokens=20,
        compression_ratio=0.2,
        language=CodeLanguage.PYTHON,
        syntax_valid=True,
        preserved_imports=2,
        preserved_signatures=3,
        compressed_bodies=4,
        symbol_scores={"important": 0.9, "discard": 0.05, "middle": 0.3},
    )

    summary = result.summary
    assert "Compressed python code" in summary
    assert "2 imports" in summary
    assert "3 signatures" in summary
    assert "4 bodies" in summary
    assert "1 high-importance" in summary
    assert "1 low-importance" in summary


def test_private_helper_functions_cover_symbol_and_syntax_paths() -> None:
    assert _get_node_text(_Node("slice", start_byte=2, end_byte=5), "0123456789") == "234"

    definition = _Node(
        "function_definition", [_Node("keyword"), _Node("identifier", text=b"runner")]
    )
    assert _get_definition_name(definition) == "runner"
    assert _get_definition_name(_Node("function_definition", [_Node("keyword")])) is None

    assert not _is_public_symbol("", CodeLanguage.PYTHON)
    assert _is_public_symbol("Runner", CodeLanguage.GO)
    assert not _is_public_symbol("runner", CodeLanguage.GO)
    assert not _is_public_symbol("_hidden", CodeLanguage.PYTHON)
    assert _is_public_symbol("visible", CodeLanguage.PYTHON)

    assert _get_body_limit("runner", {"runner": 9}, 4) == 4
    assert _get_body_limit("missing", {"runner": 3}, 5) == 5

    analysis = _SymbolAnalysis(calls={"Worker.runner": {"a", "b", "c", "d", "e", "f"}})
    comment = _make_omitted_comment("runner", 7, "  ", "#", analysis)
    assert comment == "  # [7 lines omitted; calls: a, b, c, d, e +1 more]"

    assert _detect_indent(["", "   first()", " second()"]) == "   "
    assert _detect_indent(["", ""]) == "    "

    assert _has_syntax_issues(_Node("root", [_Node("ok", [_Node("child", is_missing=True)])]))
    assert not _has_syntax_issues(_Node("root", [_Node("ok")]))


def test_verify_syntax_and_store_in_ccr_paths(monkeypatch) -> None:
    compressor = CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False))

    class _Parser:
        def __init__(self, root: _Node) -> None:
            self.root = root

        def parse(self, code_bytes: bytes):
            return types.SimpleNamespace(root_node=self.root)

    monkeypatch.setattr(
        code_compressor,
        "_get_parser",
        lambda language: _Parser(_Node("module", [_Node("identifier")])),
    )
    assert compressor._verify_syntax("def run(): pass", CodeLanguage.PYTHON) is True

    monkeypatch.setattr(
        code_compressor,
        "_get_parser",
        lambda language: _Parser(_Node("module", [_Node("ERROR")])),
    )
    assert compressor._verify_syntax("def run(:", CodeLanguage.PYTHON) is False

    monkeypatch.setattr(
        code_compressor,
        "_get_parser",
        lambda language: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert compressor._verify_syntax("def run(): pass", CodeLanguage.PYTHON) is False

    class _Store:
        def store(self, *args, **kwargs):
            return "ccr-123"

    monkeypatch.setitem(
        sys.modules,
        "headroom.cache.compression_store",
        types.SimpleNamespace(get_compression_store=lambda: _Store()),
    )
    assert compressor._store_in_ccr("original", "compressed", 20) == "ccr-123"

    class _ExplodingStore:
        def store(self, *args, **kwargs):
            raise RuntimeError("nope")

    monkeypatch.setitem(
        sys.modules,
        "headroom.cache.compression_store",
        types.SimpleNamespace(get_compression_store=lambda: _ExplodingStore()),
    )
    assert compressor._store_in_ccr("original", "compressed", 20) is None


def test_fallback_compress_and_convenience_wrapper(monkeypatch) -> None:
    compressor = CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False))

    class _KompressResult:
        compressed = "shrunk"
        original_tokens = 40
        compressed_tokens = 10
        compression_ratio = 0.25

    class _KompressCompressor:
        def compress(self, code: str):
            assert code == "long code"
            return _KompressResult()

    monkeypatch.setitem(
        sys.modules,
        "headroom.transforms.kompress_compressor",
        types.SimpleNamespace(
            KompressCompressor=_KompressCompressor,
            is_kompress_available=lambda: True,
        ),
    )
    fallback = compressor._fallback_compress("long code", 40)
    assert fallback.compressed == "shrunk"
    assert fallback.syntax_valid is False
    assert fallback.language == CodeLanguage.UNKNOWN

    monkeypatch.setitem(
        sys.modules,
        "headroom.transforms.kompress_compressor",
        types.SimpleNamespace(
            KompressCompressor=_KompressCompressor,
            is_kompress_available=lambda: False,
        ),
    )
    no_fallback = compressor._fallback_compress("long code", 40)
    assert no_fallback.compressed == "long code"
    assert no_fallback.compression_ratio == 1.0

    captured: dict[str, object] = {}

    class _FakeCompressor:
        def __init__(self, config):
            captured["config"] = config

        def compress(self, code: str, language: str | None = None, context: str = ""):
            captured["call"] = (code, language, context)
            return types.SimpleNamespace(compressed="delegated")

    monkeypatch.setattr(code_compressor, "CodeAwareCompressor", _FakeCompressor)
    assert (
        compress_code("print('x')", language="python", target_rate=0.4, context="ctx")
        == "delegated"
    )
    assert captured["call"] == ("print('x')", "python", "ctx")
    assert captured["config"].target_compression_rate == 0.4
    assert captured["config"].language_hint == "python"


def test_apply_try_compress_text_and_should_apply(monkeypatch) -> None:
    compressor = CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False))
    tokenizer = types.SimpleNamespace(count_text=lambda text: len(text))

    class _ContentType:
        SOURCE_CODE = "source"
        TEXT = "text"

    def detect_content_type(text: str):
        is_code = "def " in text or "code" in text
        return types.SimpleNamespace(
            content_type=_ContentType.SOURCE_CODE if is_code else _ContentType.TEXT,
            metadata={"language": "python"},
        )

    monkeypatch.setitem(
        sys.modules,
        "headroom.transforms.content_detector",
        types.SimpleNamespace(ContentType=_ContentType, detect_content_type=detect_content_type),
    )

    def fake_compress(text: str, language: str | None = None, context: str = "", tokenizer=None):
        return CodeCompressionResult(
            compressed="COMPRESSED",
            original=text,
            original_tokens=100,
            compressed_tokens=50,
            compression_ratio=0.5,
            language=CodeLanguage.PYTHON,
        )

    monkeypatch.setattr(compressor, "compress", fake_compress)

    transforms_applied: list[str] = []
    assert compressor._try_compress_text("", "ctx", tokenizer, transforms_applied) == ""
    assert (
        compressor._try_compress_text(
            "def run():\n    return 1", "ctx", tokenizer, transforms_applied
        )
        == "COMPRESSED"
    )
    assert transforms_applied == ["code_aware:python:0.50"]
    assert (
        compressor._try_compress_text("plain text", "ctx", tokenizer, transforms_applied)
        == "plain text"
    )

    def fake_try(text: str, context: str, tokenizer, applied: list[str]) -> str:
        if "code" in text:
            applied.append("code_aware:python:0.50")
            return text.upper()
        return text

    monkeypatch.setattr(compressor, "_try_compress_text", fake_try)
    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: False)

    result = compressor.apply(
        [
            {"role": "user", "content": "code sample"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "nested code"}, {"type": "tool_use"}],
            },
            {"role": "system", "content": 123},
        ],
        tokenizer,
        context="ctx",
    )
    assert result.messages[0]["content"] == "CODE SAMPLE"
    assert result.messages[1]["content"][0]["text"] == "NESTED CODE"
    assert result.messages[1]["content"][1] == {"type": "tool_use"}
    assert result.messages[2]["content"] == 123
    assert result.transforms_applied == ["code_aware:python:0.50", "code_aware:python:0.50"]
    assert result.warnings == [
        "tree-sitter not installed. Install with: pip install headroom-ai[code]"
    ]

    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: True)
    assert compressor.should_apply([{"content": "plain text"}], tokenizer) is False
    assert compressor.should_apply([{"content": "def run():\n    return 1"}], tokenizer) is True
    assert (
        compressor.should_apply(
            [{"content": [{"type": "text", "text": "nested code"}, {"type": "image"}]}],
            tokenizer,
        )
        is True
    )


def test_symbol_analysis_and_budget_allocation_cover_semantic_paths() -> None:
    compressor = CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False, semantic_analysis=True))
    code = (
        "class Worker:\n"
        "    def keep(self):\n"
        "        helper()\n"
        "        helper()\n"
        "\n"
        "def helper():\n"
        "    keep()\n"
        "\n"
        "@decorator\n"
        "def __magic__():\n"
        "    helper()\n"
    )

    helper_node = _Node(
        "function_definition",
        [_Node("identifier", text=b"helper"), _Node("identifier", text=b"keep")],
        start_byte=44,
        end_byte=72,
    )
    keep_node = _Node(
        "function_definition",
        [_Node("identifier", text=b"keep"), _Node("identifier", text=b"helper")],
        start_byte=18,
        end_byte=43,
    )
    class_node = _Node(
        "class_definition",
        [_Node("identifier", text=b"Worker"), keep_node],
        start_byte=0,
        end_byte=43,
    )
    magic_function = _Node(
        "function_definition",
        [_Node("identifier", text=b"__magic__"), _Node("identifier", text=b"helper")],
        start_byte=84,
        end_byte=len(code),
    )
    decorated_magic = _Node("decorated_definition", [_Node("decorator"), magic_function])
    root = _Node("module", [class_node, helper_node, decorated_magic])

    analysis = compressor._analyze_symbol_importance(
        root,
        code,
        CodeLanguage.PYTHON,
        context="keep helper critical __magic__",
    )
    assert set(analysis.scores) == {"Worker", "Worker.keep", "helper", "__magic__"}
    assert analysis.bare_names["Worker.keep"] == "keep"
    assert analysis.calls["Worker.keep"] == {"helper"}
    assert analysis.calls["helper"] == {"keep"}
    assert analysis.ref_counts["helper"] >= 2
    assert all(0.0 <= score <= 1.0 for score in analysis.scores.values())

    body_limits = compressor._allocate_body_budget(analysis, code)
    assert "Worker.keep" in body_limits
    assert "keep" in body_limits
    assert body_limits["keep"] == body_limits["Worker.keep"]
    assert body_limits["helper"] <= compressor.config.max_body_lines

    disabled = CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False, semantic_analysis=False))
    assert disabled._analyze_symbol_importance(root, code, CodeLanguage.PYTHON) == _SymbolAnalysis()
    assert (
        compressor._analyze_symbol_importance(root, code, CodeLanguage.UNKNOWN) == _SymbolAnalysis()
    )
    assert compressor._allocate_body_budget(_SymbolAnalysis(), code) == {}
    assert (
        compressor._allocate_body_budget(
            _SymbolAnalysis(scores={"helper": 0.9}, body_line_counts={"helper": 0}),
            code,
        )
        == {}
    )


def test_compress_covers_ccr_unknown_and_failure_guards(monkeypatch) -> None:
    compressor = CodeAwareCompressor(
        CodeCompressorConfig(
            enable_ccr=True,
            ccr_ttl=120,
            min_tokens_for_compression=1,
            fallback_to_kompress=False,
        )
    )

    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: True)
    monkeypatch.setattr(
        compressor,
        "_compress_with_ast",
        lambda code, language, context, tokenizer=None: (
            "compressed body",
            CodeStructure(
                imports=["import os"],
                function_signatures=["def keep(): ..."],
                function_bodies=[("keep", "body", 1)],
            ),
            {"helper": 0.7},
        ),
    )
    monkeypatch.setattr(compressor, "_verify_syntax", lambda code, language: True)
    monkeypatch.setattr(
        compressor, "_store_in_ccr", lambda original, compressed, original_tokens: "cache-123"
    )
    monkeypatch.setattr(
        compressor,
        "_estimate_tokens",
        lambda text, tokenizer=None: 100 if text == "original code" else 50,
    )
    monkeypatch.setitem(
        sys.modules,
        "headroom.transforms.compression_summary",
        types.SimpleNamespace(
            summarize_compressed_code=lambda bodies, count: f"{count} bodies summarized"
        ),
    )
    result = compressor.compress("original code", language="python", context="ctx")
    assert result.language == CodeLanguage.PYTHON
    assert result.cache_key == "cache-123"
    assert result.compression_ratio == 0.5
    assert "Retrieve more: hash=cache-123." in result.compressed
    assert "Expires in 2m." in result.compressed
    assert result.preserved_imports == 1
    assert result.preserved_signatures == 1
    assert result.compressed_bodies == 1

    compressor = CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=1, fallback_to_kompress=False)
    )
    monkeypatch.setattr(
        code_compressor, "detect_language", lambda code: (CodeLanguage.UNKNOWN, 0.0)
    )
    unknown = compressor.compress("unknown language")
    assert unknown.language == CodeLanguage.UNKNOWN
    assert unknown.compression_ratio == 1.0

    compressor = CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=1, fallback_to_kompress=True)
    )
    monkeypatch.setattr(
        code_compressor, "detect_language", lambda code: (CodeLanguage.UNKNOWN, 0.0)
    )
    monkeypatch.setattr(
        compressor,
        "_fallback_compress",
        lambda code, original_tokens: CodeCompressionResult(
            compressed="fallback",
            original=code,
            original_tokens=original_tokens,
            compressed_tokens=5,
            compression_ratio=0.5,
            language=CodeLanguage.UNKNOWN,
            syntax_valid=False,
        ),
    )
    assert compressor.compress("unknown language").compressed == "fallback"

    compressor = CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=1, fallback_to_kompress=False)
    )
    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: False)
    available = compressor.compress("def keep(): pass", language="python")
    assert available.language == CodeLanguage.PYTHON
    assert available.compression_ratio == 1.0

    compressor = CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=1, fallback_to_kompress=False)
    )
    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: True)
    monkeypatch.setattr(
        compressor,
        "_compress_with_ast",
        lambda code, language, context, tokenizer=None: ("bad", CodeStructure(), {}),
    )
    monkeypatch.setattr(compressor, "_verify_syntax", lambda code, language: False)
    invalid = compressor.compress("def keep(): pass", language="python")
    assert invalid.compressed == "def keep(): pass"

    compressor = CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=1, fallback_to_kompress=False)
    )
    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: True)
    monkeypatch.setattr(
        compressor,
        "_compress_with_ast",
        lambda code, language, context, tokenizer=None: ("tiny", CodeStructure(), {}),
    )
    monkeypatch.setattr(compressor, "_verify_syntax", lambda code, language: True)
    monkeypatch.setattr(
        compressor,
        "_estimate_tokens",
        lambda text, tokenizer=None: 100 if text == "def keep(): pass" else 1,
    )
    too_aggressive = compressor.compress("def keep(): pass", language="python")
    assert too_aggressive.compression_ratio == 1.0

    compressor = CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=1, fallback_to_kompress=False)
    )
    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: True)
    monkeypatch.setattr(
        compressor,
        "_compress_with_ast",
        lambda code, language, context, tokenizer=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    failed = compressor.compress("def keep(): pass", language="python")
    assert failed.compressed == "def keep(): pass"


def test_compress_with_ast_extract_structure_and_assembly_paths(monkeypatch) -> None:
    compressor = CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False))

    class _Parser:
        def parse(self, code_bytes: bytes):
            return types.SimpleNamespace(root_node=_Node("module"))

    monkeypatch.setattr(code_compressor, "_get_parser", lambda language: _Parser())
    monkeypatch.setattr(
        compressor,
        "_analyze_symbol_importance",
        lambda root, code, language, context="": _SymbolAnalysis(
            scores={"Worker.keep": 0.8, "keep": 0.6},
            bare_names={"Worker.keep": "keep", "keep": "keep"},
            body_line_counts={"Worker.keep": 4},
        ),
    )
    monkeypatch.setattr(compressor, "_allocate_body_budget", lambda analysis, code: {"keep": 2})
    monkeypatch.setattr(
        compressor,
        "_extract_structure",
        lambda root, code, language, lang_config, body_limits, analysis: CodeStructure(
            imports=["import os"],
            function_signatures=["def keep():\n    pass"],
            function_bodies=[("keep", "body", 1)],
        ),
    )
    compressed, structure, symbol_scores = compressor._compress_with_ast(
        "code", CodeLanguage.PYTHON, "ctx"
    )
    assert compressed == "import os\n\ndef keep():\n    pass"
    assert structure.imports == ["import os"]
    assert symbol_scores == {"keep": 0.8}

    monkeypatch.setattr(
        compressor,
        "_extract_generic_structure",
        lambda root, code: CodeStructure(other=["misc", ""]),
    )
    compressed, _, symbol_scores = compressor._compress_with_ast(
        "misc", CodeLanguage.UNKNOWN, "ctx"
    )
    assert compressed == "misc"
    assert symbol_scores == {"keep": 0.8}

    generic = CodeAwareCompressor(
        CodeCompressorConfig(enable_ccr=False)
    )._extract_generic_structure(_Node("module"), "a\nb")
    assert generic.other == ["a", "b"]
    assembled = compressor._assemble_compressed(
        CodeStructure(
            imports=["import os"],
            type_definitions=["type Alias = int"],
            class_definitions=["class A:\n    pass"],
            function_signatures=["def run():\n    pass"],
            top_level_code=["X = 1"],
            other=["misc", ""],
        ),
        CodeLanguage.PYTHON,
    )
    assert assembled.endswith("misc")
    assert "type Alias = int" in assembled


def test_extract_structure_function_and_class_compression_paths(monkeypatch) -> None:
    compressor = CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False))

    code = (
        "package main\n"
        "import os\n"
        "export function wrapped() {}\n"
        "export { value }\n"
        "@decorator\n"
        "def decorated():\n"
        "    pass\n"
        "def plain():\n"
        "    pass\n"
        "class Klass:\n"
        "    pass\n"
        "type Alias = int\n"
        "x = 1\n"
    )

    def span(snippet: str, start: int = 0) -> tuple[int, int]:
        index = code.index(snippet, start)
        return index, index + len(snippet)

    export_func_text = "export function wrapped() {}"
    function_text = "function wrapped() {}"
    export_func_start, export_func_end = span(export_func_text)
    function_start, function_end = span(function_text)
    decorator_start, decorator_end = span("@decorator")
    decorated_start, decorated_end = span("@decorator\n" + "def decorated():\n    pass")
    plain_start, plain_end = span("def plain():\n    pass")
    class_start, class_end = span("class Klass:\n    pass")
    type_start, type_end = span("type Alias = int")
    top_start, top_end = span("x = 1")

    root = _Node(
        "module",
        [
            _Node("package_clause", start_byte=0, end_byte=len("package main")),
            _Node(
                "import_statement",
                start_byte=len("package main\n"),
                end_byte=len("package main\nimport os"),
            ),
            _Node(
                "export_statement",
                [
                    _Node(
                        "function_declaration",
                        [_Node("identifier", text=b"wrapped")],
                        start_byte=function_start,
                        end_byte=function_end,
                    )
                ],
                start_byte=export_func_start,
                end_byte=export_func_end,
            ),
            _Node(
                "export_statement",
                [_Node("identifier", text=b"value")],
                start_byte=code.index("export { value }"),
                end_byte=code.index("export { value }") + len("export { value }"),
            ),
            _Node(
                "decorated_definition",
                [
                    _Node("decorator", start_byte=decorator_start, end_byte=decorator_end),
                    _Node(
                        "function_definition",
                        [_Node("identifier", text=b"decorated")],
                        start_byte=code.index("def decorated():"),
                        end_byte=decorated_end,
                    ),
                ],
                start_byte=decorated_start,
                end_byte=decorated_end,
            ),
            _Node(
                "function_definition",
                [_Node("identifier", text=b"plain")],
                start_byte=plain_start,
                end_byte=plain_end,
            ),
            _Node(
                "class_definition",
                [_Node("identifier", text=b"Klass")],
                start_byte=class_start,
                end_byte=class_end,
            ),
            _Node("type_alias_statement", start_byte=type_start, end_byte=type_end),
            _Node("assignment", start_byte=top_start, end_byte=top_end),
        ],
    )
    lang_config = LangConfig(
        import_nodes=frozenset({"import_statement"}),
        function_nodes=frozenset({"function_definition", "function_declaration"}),
        class_nodes=frozenset({"class_definition"}),
        type_nodes=frozenset({"type_alias_statement"}),
        body_node_types=frozenset({"block"}),
        decorator_node="decorated_definition",
        comment_prefix="#",
        uses_colon_after_signature=True,
        package_node="package_clause",
    )
    monkeypatch.setattr(
        compressor,
        "_compress_function_ast",
        lambda node, code, language, lang_config, body_limits, analysis: (
            f"FUNC<{_get_definition_name(node)}>"
        ),
    )
    monkeypatch.setattr(
        compressor,
        "_compress_class_ast",
        lambda node, code, language, lang_config, body_limits, analysis: "CLASS<Klass>",
    )
    structure = compressor._extract_structure(
        root, code, CodeLanguage.PYTHON, lang_config, {"plain": 1}, _SymbolAnalysis()
    )
    assert structure.imports[0] == "package main"
    assert "import os" in structure.imports
    assert "export { value }" in structure.imports
    assert "export FUNC<wrapped>" in structure.function_signatures
    assert "@decorator\nFUNC<decorated>" in structure.function_signatures
    assert "FUNC<plain>" in structure.function_signatures
    assert structure.class_definitions == ["CLASS<Klass>"]
    assert structure.type_definitions == ["type Alias = int"]
    assert structure.top_level_code == ["x = 1"]


def test_compress_function_and_class_ast_cover_truncation_paths(monkeypatch) -> None:
    compressor = CodeAwareCompressor(
        CodeCompressorConfig(
            enable_ccr=False, max_body_lines=1, docstring_mode=DocstringMode.FIRST_LINE
        )
    )
    analysis = _SymbolAnalysis(calls={"sample": {"helper", "other"}})
    python_code = (
        "def sample():\n"
        '    """Doc first line\n'
        "    Doc second line\n"
        '    """\n'
        "    first()\n"
        "    second()\n"
        "    third()\n"
    )
    docstring_node = _Node(
        "expression_statement",
        [_Node("string")],
        start_point=(1, 0),
        end_point=(3, 7),
    )
    body_node = _Node(
        "block",
        [
            docstring_node,
            _Node("expression_statement", start_point=(4, 0), end_point=(4, 11)),
            _Node("expression_statement", start_point=(5, 0), end_point=(5, 12)),
            _Node("expression_statement", start_point=(6, 0), end_point=(6, 11)),
        ],
        start_point=(1, 0),
        end_point=(6, 11),
    )
    function_node = _Node(
        "function_definition",
        [_Node("identifier", text=b"sample"), body_node],
        start_point=(0, 0),
        end_point=(6, 11),
    )
    py_lang = code_compressor._LANG_CONFIGS[CodeLanguage.PYTHON]
    compressed = compressor._compress_function_ast(
        function_node,
        python_code,
        CodeLanguage.PYTHON,
        py_lang,
        {"sample": 1},
        analysis,
    )
    assert '"""Doc first line"""' in compressed
    assert "first()" in compressed
    assert "[2 lines omitted; calls: helper, other]" in compressed
    assert compressed.rstrip().endswith("pass")

    js_code = "function keep(){\n  one();\n  two();\n}\n"
    js_body = _Node(
        "statement_block",
        [
            _Node("{", start_point=(0, 15), end_point=(0, 15), is_named=False),
            _Node("expression_statement", start_point=(1, 0), end_point=(1, 7)),
            _Node("expression_statement", start_point=(2, 0), end_point=(2, 7)),
            _Node("}", start_point=(3, 0), end_point=(3, 0), is_named=False),
        ],
        start_point=(0, 15),
        end_point=(3, 0),
    )
    js_node = _Node(
        "function_declaration",
        [_Node("identifier", text=b"keep"), js_body],
        start_point=(0, 0),
        end_point=(3, 0),
    )
    js_lang = code_compressor._LANG_CONFIGS[CodeLanguage.JAVASCRIPT]
    compressed_js = compressor._compress_function_ast(
        js_node,
        js_code,
        CodeLanguage.JAVASCRIPT,
        js_lang,
        {"keep": 1},
        _SymbolAnalysis(),
    )
    assert compressed_js.splitlines()[0] == "function keep(){"
    assert compressed_js.splitlines()[-1] == "}"

    class_code = (
        "class Outer:\n"
        "    attr = 1\n"
        "    def method(self):\n"
        "        run()\n"
        "    @decorator\n"
        "    def decorated(self):\n"
        "        run()\n"
        "    class Inner:\n"
        "        pass\n"
    )
    outer_body = _Node(
        "block",
        [
            _Node("expression_statement", start_point=(1, 0), end_point=(1, 11)),
            _Node("function_definition", start_point=(2, 0), end_point=(3, 13)),
            _Node(
                "decorated_definition",
                [
                    _Node(
                        "decorator",
                        start_byte=class_code.index("    @decorator"),
                        end_byte=class_code.index("    @decorator") + len("    @decorator"),
                    ),
                    _Node("function_definition", start_point=(5, 0), end_point=(6, 13)),
                ],
                start_point=(4, 0),
                end_point=(6, 13),
            ),
            _Node(
                "class_definition",
                [_Node("identifier", text=b"Inner")],
                start_point=(7, 0),
                end_point=(8, 11),
            ),
        ],
        start_point=(1, 0),
        end_point=(8, 11),
    )
    outer_node = _Node(
        "class_definition",
        [_Node("identifier", text=b"Outer"), outer_body],
        start_point=(0, 0),
        end_point=(8, 11),
    )
    monkeypatch.setattr(
        compressor,
        "_compress_function_ast",
        lambda node, code, language, lang_config, body_limits, analysis: "COMPRESSED_METHOD",
    )
    original_class_ast = compressor._compress_class_ast
    monkeypatch.setattr(
        compressor,
        "_compress_class_ast",
        lambda node, code, language, lang_config, body_limits, analysis: (
            "COMPRESSED_INNER"
            if _get_definition_name(node) == "Inner"
            else original_class_ast(node, code, language, lang_config, body_limits, analysis)
        ),
    )
    compressed_class = compressor._compress_class_ast(
        outer_node,
        class_code,
        CodeLanguage.PYTHON,
        py_lang,
        {"method": 1},
        _SymbolAnalysis(),
    )
    assert "class Outer:" in compressed_class
    assert "attr = 1" in compressed_class
    assert "@decorator" in compressed_class
    assert "COMPRESSED_METHOD" in compressed_class
    assert "COMPRESSED_INNER" in compressed_class
