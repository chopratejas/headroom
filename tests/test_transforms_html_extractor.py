from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from types import SimpleNamespace

_trafilatura_module = types.ModuleType("trafilatura")
_trafilatura_module.__spec__ = importlib.machinery.ModuleSpec("trafilatura", loader=None)
_trafilatura_module.extract = lambda *_args, **_kwargs: None
_trafilatura_module.extract_metadata = lambda *_args, **_kwargs: None
_trafilatura_settings_module = types.ModuleType("trafilatura.settings")
_trafilatura_settings_module.__spec__ = importlib.machinery.ModuleSpec(
    "trafilatura.settings", loader=None
)
_trafilatura_settings_module.use_config = lambda: None
sys.modules.setdefault("trafilatura", _trafilatura_module)
sys.modules.setdefault("trafilatura.settings", _trafilatura_settings_module)

html_extractor = importlib.import_module("headroom.transforms.html_extractor")


class _FakeConfig:
    def __init__(self) -> None:
        self.values = {}

    def set(self, section: str, key: str, value: str) -> None:
        self.values[(section, key)] = value


def test_html_extraction_result_reduction_percent() -> None:
    result = html_extractor.HTMLExtractionResult(
        extracted="hello",
        original="<p>hello</p>",
        original_length=10,
        extracted_length=5,
        compression_ratio=0.5,
    )
    assert result.reduction_percent == 50.0

    zero = html_extractor.HTMLExtractionResult(
        extracted="",
        original="",
        original_length=0,
        extracted_length=0,
        compression_ratio=0.0,
    )
    assert zero.reduction_percent == 0.0


def test_html_extractor_extract_and_batch(monkeypatch) -> None:
    fake_config = _FakeConfig()
    monkeypatch.setattr(html_extractor, "use_config", lambda: fake_config)
    monkeypatch.setattr(
        html_extractor.trafilatura,
        "extract",
        lambda html, **kwargs: "# Title\ncontent" if "good" in html else None,
    )
    monkeypatch.setattr(
        html_extractor.trafilatura,
        "extract_metadata",
        lambda html, default_url=None: (
            SimpleNamespace(
                title="Doc",
                author="Ada",
                date="2024-01-01",
                sitename="Headroom",
                description="desc",
                categories=["docs"],
                tags=["tag"],
            )
            if "good" in html
            else None
        ),
    )

    extractor = html_extractor.HTMLExtractor(
        html_extractor.HTMLExtractorConfig(
            output_format="text",
            include_links=False,
            include_images=True,
            include_tables=False,
            include_comments=True,
            include_formatting=False,
            favor_precision=True,
            favor_recall=False,
            extract_metadata=True,
        )
    )
    assert fake_config.values[("DEFAULT", "FAVOR_PRECISION")] == "True"
    assert fake_config.values[("DEFAULT", "FAVOR_RECALL")] == "False"

    result = extractor.extract("<html>good</html>", url="https://example.test")
    assert result.extracted == "# Title\ncontent"
    assert result.title == "Doc"
    assert result.metadata["sitename"] == "Headroom"
    assert result.extracted_length == len("# Title\ncontent")

    batch = extractor.extract_batch(
        [("<html>good</html>", "https://example.test"), ("<html>bad</html>", None)]
    )
    assert batch[0].title == "Doc"
    assert batch[1].extracted == ""


def test_html_extractor_empty_input_and_html_detection(monkeypatch) -> None:
    monkeypatch.setattr(html_extractor, "use_config", lambda: _FakeConfig())
    extractor = html_extractor.HTMLExtractor(
        html_extractor.HTMLExtractorConfig(extract_metadata=False)
    )

    empty = extractor.extract("   ")
    assert empty.extracted == ""
    assert empty.metadata == {}

    assert html_extractor.is_html_content("<!DOCTYPE html><html><body></body></html>") is True
    assert html_extractor.is_html_content("<div>hello</div><script>x</script>") is True
    assert html_extractor.is_html_content("plain text only") is False
    assert html_extractor.is_html_content("") is False
