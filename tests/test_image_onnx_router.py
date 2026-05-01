from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from headroom.image.onnx_router import OnnxTechniqueRouter
from headroom.image.trained_router import ImageSignals, Technique


class _FakeEncoding:
    ids = [101, 102, 103]
    attention_mask = [1, 1, 1]


class _FakeTokenizer:
    def __init__(self) -> None:
        self.truncation = None
        self.padding = None

    def encode(self, query: str) -> _FakeEncoding:
        assert query
        return _FakeEncoding()

    def enable_truncation(self, *, max_length: int) -> None:
        self.truncation = max_length

    def enable_padding(self, *, length: int) -> None:
        self.padding = length


class _FakeImage:
    def convert(self, mode: str) -> _FakeImage:
        assert mode == "RGB"
        return self

    def resize(self, size: tuple[int, int], _resample) -> _FakeImage:
        assert size == (224, 224)
        return self

    def __array__(self, dtype=None, copy=None):
        return np.ones((224, 224, 3), dtype=dtype or np.float32) * 255.0


def _install_classifier_stubs(monkeypatch, tmp_path: Path, logits: list[float]) -> None:
    model_path = tmp_path / "model_quantized.onnx"
    tokenizer_path = tmp_path / "tokenizer.json"
    config_path = tmp_path / "config.json"
    model_path.write_bytes(b"onnx")
    tokenizer_path.write_text("{}")
    config_path.write_text(
        json.dumps({"id2label": {"0": "preserve", "1": "transcode", "2": "full_low"}})
    )

    paths = {
        "model_quantized.onnx": str(model_path),
        "tokenizer.json": str(tokenizer_path),
        "config.json": str(config_path),
    }

    class _FakeSession:
        def __init__(self, path: str, _session_options, providers: list[str]) -> None:
            self.path = path
            self.providers = providers

        def run(self, _outputs, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert set(inputs) == {"input_ids", "attention_mask", "token_type_ids"}
            return [np.array([logits], dtype=np.float32)]

    fake_tokenizer = _FakeTokenizer()
    fake_ort = types.SimpleNamespace(
        InferenceSession=_FakeSession,
        SessionOptions=type("SessionOptions", (), {}),
    )
    fake_hf = types.SimpleNamespace(hf_hub_download=lambda _repo, filename: paths[filename])
    fake_tokenizers = types.SimpleNamespace(
        Tokenizer=types.SimpleNamespace(from_file=lambda path: fake_tokenizer)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)


def _install_siglip_stubs(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "image_encoder_int8.onnx"
    embeddings_path = tmp_path / "text_embeddings.npz"
    model_path.write_bytes(b"onnx")
    np.savez(
        embeddings_path,
        has_text=np.array([[1.0, 0.0]], dtype=np.float32),
        is_document=np.array([[-1.0, 0.0]], dtype=np.float32),
        has_small_details=np.array([[0.2, 0.8]], dtype=np.float32),
    )
    paths = {
        "image_encoder_int8.onnx": str(model_path),
        "text_embeddings.npz": str(embeddings_path),
    }

    class _FakeSession:
        def __init__(self, path: str, _session_options, providers: list[str]) -> None:
            self.path = path
            self.providers = providers

        def run(self, _outputs, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert "pixel_values" in inputs
            return [np.array([[1.0, 0.0]], dtype=np.float32)]

    fake_ort = types.SimpleNamespace(
        InferenceSession=_FakeSession,
        SessionOptions=type("SessionOptions", (), {}),
    )
    fake_hf = types.SimpleNamespace(hf_hub_download=lambda _repo, filename: paths[filename])
    fake_image_module = types.SimpleNamespace(
        open=lambda _bytes: _FakeImage(),
        Resampling=types.SimpleNamespace(LANCZOS="lanczos"),
    )
    fake_pil = types.SimpleNamespace(Image=fake_image_module)

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)


def test_classifier_loading_and_query_classification(monkeypatch, tmp_path: Path) -> None:
    _install_classifier_stubs(monkeypatch, tmp_path, [0.2, 3.0, 0.1])
    router = OnnxTechniqueRouter()

    technique, confidence = router.classify_query("read the screenshot")
    assert technique is Technique.TRANSCODE
    assert 0.8 < confidence < 1.0
    assert router._tokenizer.truncation == 64
    assert router._tokenizer.padding == 64

    first_session = router._classifier_session
    router._load_classifier()
    assert router._classifier_session is first_session


def test_siglip_loading_and_image_analysis(monkeypatch, tmp_path: Path) -> None:
    _install_siglip_stubs(monkeypatch, tmp_path)
    router = OnnxTechniqueRouter()

    signals = router.analyze_image(b"fake-image-bytes")
    assert isinstance(signals, ImageSignals)
    assert signals.has_text > 0.9
    assert signals.is_document < 0.1
    assert signals.is_complex == 0.5
    assert signals.has_small_details > 0.7

    first_session = router._siglip_session
    router._load_siglip()
    assert router._siglip_session is first_session


def test_image_analysis_failure_and_classify_adjustments(monkeypatch, tmp_path: Path) -> None:
    router = OnnxTechniqueRouter(use_siglip=False)
    assert router.analyze_image(b"ignored") is None

    _install_siglip_stubs(monkeypatch, tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        types.SimpleNamespace(
            Image=types.SimpleNamespace(
                open=lambda _bytes: (_ for _ in ()).throw(RuntimeError("bad image")),
                Resampling=types.SimpleNamespace(LANCZOS="lanczos"),
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "PIL.Image",
        types.SimpleNamespace(
            open=lambda _bytes: (_ for _ in ()).throw(RuntimeError("bad image")),
            Resampling=types.SimpleNamespace(LANCZOS="lanczos"),
        ),
    )
    assert OnnxTechniqueRouter().analyze_image(b"bad-image") is None

    router = OnnxTechniqueRouter()
    monkeypatch.setattr(
        router,
        "classify_query",
        lambda _query: (Technique.TRANSCODE, 0.8),
    )
    monkeypatch.setattr(
        router,
        "analyze_image",
        lambda _data: ImageSignals(0.2, 0.2, 0.1, 0.1),
    )
    result = router.classify(b"x", "q")
    assert result.confidence == pytest.approx(0.64)
    assert "low text in image" in result.reason

    monkeypatch.setattr(router, "classify_query", lambda _query: (Technique.FULL_LOW, 0.7))
    monkeypatch.setattr(
        router,
        "analyze_image",
        lambda _data: ImageSignals(0.5, 0.5, 0.5, 0.8),
    )
    assert "fine details detected" in router.classify(b"x", "q").reason

    monkeypatch.setattr(router, "classify_query", lambda _query: (Technique.PRESERVE, 0.95))
    monkeypatch.setattr(
        router,
        "analyze_image",
        lambda _data: ImageSignals(0.5, 0.5, 0.6, 0.6),
    )
    preserve = router.classify(b"x", "q")
    assert preserve.confidence == 1.0
    assert "confirmed: complex/detailed" in preserve.reason

    monkeypatch.setattr(router, "analyze_image", lambda _data: None)
    plain = router.classify(b"x", "q")
    assert plain.image_signals is None
    assert plain.confidence == 0.95
