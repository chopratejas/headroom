from __future__ import annotations

import builtins
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_embedders(monkeypatch):
    config_module = types.ModuleType("headroom.models.config")
    config_module.ML_MODEL_DEFAULTS = SimpleNamespace(sentence_transformer="mini-model")
    onnx_runtime_module = types.ModuleType("headroom.onnx_runtime")
    onnx_runtime_module.create_cpu_session_options = lambda ort, **kwargs: ("opts", kwargs)

    monkeypatch.setitem(sys.modules, "headroom.models.config", config_module)
    monkeypatch.setitem(sys.modules, "headroom.onnx_runtime", onnx_runtime_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "adapters" / "embedders.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.adapters.embedders", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.adapters.embedders", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.adapters.embedders"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_local_embedder_and_normalizers(monkeypatch) -> None:
    embedders = _load_embedders(monkeypatch)

    np.testing.assert_allclose(
        embedders._normalize_embedding(np.array([3.0, 4.0])),
        np.array([0.6, 0.8], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        embedders._normalize_embedding(np.array([0.0, 0.0])),
        np.array([0.0, 0.0], dtype=np.float32),
    )
    batch_norm = embedders._normalize_embeddings_batch(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(batch_norm[0], np.array([0.6, 0.8], dtype=np.float32))
    np.testing.assert_array_equal(batch_norm[1], np.array([0.0, 0.0], dtype=np.float32))

    fake_model = SimpleNamespace(
        encode=lambda texts, convert_to_numpy=True, normalize_embeddings=False: (
            np.array([3.0, 4.0], dtype=np.float32)
            if isinstance(texts, str)
            else np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)
        ),
        get_sentence_embedding_dimension=lambda: 2,
    )
    registry_module = types.ModuleType("headroom.models.ml_models")
    registry_module.MLModelRegistry = SimpleNamespace(
        get_sentence_transformer=lambda model_name, device: fake_model
    )
    monkeypatch.setitem(sys.modules, "headroom.models.ml_models", registry_module)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.ModuleType("sentence_transformers")
    )
    torch_cuda = SimpleNamespace(is_available=lambda: False)
    torch_mps = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=torch_cuda, backends=SimpleNamespace(mps=torch_mps)),
    )

    local = embedders.LocalEmbedder()
    assert local.dimension == embedders.LocalEmbedder.DEFAULT_DIMENSION
    assert local.model_name == "mini-model"
    assert local.max_tokens == embedders.LocalEmbedder.DEFAULT_MAX_TOKENS
    assert local._detect_device() == "mps"
    local._load_model()
    assert local._device == "mps"
    assert local.dimension == 2
    np.testing.assert_allclose(await local.embed("hello"), np.array([0.6, 0.8], dtype=np.float32))
    np.testing.assert_array_equal(await local.embed("  "), np.zeros(2, dtype=np.float32))
    batch = await local.embed_batch(["hello", "", "world"])
    np.testing.assert_allclose(batch[0], np.array([0.6, 0.8], dtype=np.float32))
    np.testing.assert_array_equal(batch[1], np.zeros(2, dtype=np.float32))
    assert len(batch) == 3
    assert await local.close() is None

    requested = embedders.LocalEmbedder(model_name="custom", device="cpu")
    requested._load_model()
    assert requested._device == "cpu"
    assert requested.model_name == "custom"

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "sentence_transformers":
            raise ImportError("missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        embedders.LocalEmbedder()._check_dependencies()


@pytest.mark.asyncio
async def test_onnx_local_embedder(monkeypatch) -> None:
    embedders = _load_embedders(monkeypatch)

    class FakeSession:
        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
                SimpleNamespace(name="token_type_ids"),
            ]

        def run(self, _outputs, feeds):
            assert "input_ids" in feeds
            return [np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)]

    class FakeInferenceSession:
        def __new__(cls, model_path, sess_options, providers):
            assert providers == ["CPUExecutionProvider"]
            return FakeSession()

    class FakeTokenizer:
        @staticmethod
        def from_file(path):
            tok = FakeTokenizer()
            tok.path = path
            return tok

        def enable_truncation(self, max_length):
            self.max_length = max_length

        def enable_padding(self, length):
            self.padding = length

        def encode(self, text):
            return SimpleNamespace(ids=[1, 2], attention_mask=[1, 1])

        def encode_batch(self, texts):
            return [self.encode(text) for text in texts]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeInferenceSession),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda repo, name: f"C:\\tmp\\{name}"),
    )
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizer))

    onnx = embedders.OnnxLocalEmbedder(max_length=8)
    assert onnx.dimension == 384
    assert onnx.model_name == "all-MiniLM-L6-v2-onnx"
    assert onnx.max_tokens == 8
    vector = await onnx.embed("hello")
    assert vector.shape == (2,)
    empty = onnx._embed_single("   ")
    assert empty.shape == (384,)
    batch = await onnx.embed_batch(["hello", "world"])
    assert len(batch) == 2
    await onnx.close()
    assert onnx._session is None
    assert onnx._tokenizer is None


@pytest.mark.asyncio
async def test_openai_embedder_paths(monkeypatch) -> None:
    embedders = _load_embedders(monkeypatch)

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class FakeAsyncEmbeddings:
        def __init__(self, responses):
            self.responses = list(responses)

        async def create(self, model, input):
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    class FakeAsyncOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.embeddings = None
            self.closed = False

        async def close(self):
            self.closed = True

    openai_module = types.ModuleType("openai")
    openai_module.APIConnectionError = APIConnectionError
    openai_module.APITimeoutError = APITimeoutError
    openai_module.RateLimitError = RateLimitError
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    async def _no_sleep(_delay):
        return None

    with pytest.raises(ValueError):
        embedders.OpenAIEmbedder(api_key=None)

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    embedder = embedders.OpenAIEmbedder(model_name="embed-model", max_retries=2)
    assert embedder.model_name == "embed-model"
    assert embedder.dimension == 1536
    assert embedder.max_tokens == embedders.OpenAIEmbedder.DEFAULT_MAX_TOKENS

    success_client = FakeAsyncOpenAI("env-key")
    success_client.embeddings = FakeAsyncEmbeddings(
        [
            APIConnectionError("retry"),
            SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0])]),
        ]
    )
    monkeypatch.setattr(embedders.OpenAIEmbedder, "_async_client", success_client)
    monkeypatch.setattr(embedders.asyncio, "sleep", _no_sleep)
    result = await embedder._embed_with_retry(["hello"])
    np.testing.assert_array_equal(result[0], np.array([3.0, 4.0], dtype=np.float32))
    success_client.embeddings = FakeAsyncEmbeddings(
        [SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0])])]
    )
    np.testing.assert_allclose(
        await embedder.embed("hello"), np.array([0.6, 0.8], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        await embedder.embed(" "), np.zeros(embedder.dimension, dtype=np.float32)
    )

    batch_client = FakeAsyncOpenAI("env-key")
    batch_client.embeddings = FakeAsyncEmbeddings(
        [
            SimpleNamespace(
                data=[SimpleNamespace(embedding=[3.0, 4.0]), SimpleNamespace(embedding=[5.0, 12.0])]
            ),
            SimpleNamespace(data=[SimpleNamespace(embedding=[8.0, 15.0])]),
        ]
    )
    monkeypatch.setattr(embedders.OpenAIEmbedder, "_async_client", batch_client)
    monkeypatch.setattr(embedders.OpenAIEmbedder, "MAX_BATCH_SIZE", 2)
    batch = await embedder.embed_batch(["a", "", "bb", "ccc"])
    assert len(batch) == 4
    np.testing.assert_array_equal(batch[1], np.zeros(embedder.dimension, dtype=np.float32))
    assert await embedder.embed_batch([]) == []
    await embedder.close()
    assert "_async_client" not in embedder.__dict__

    failure_client = FakeAsyncOpenAI("env-key")
    failure_client.embeddings = FakeAsyncEmbeddings([Exception("bad")])
    monkeypatch.setattr(embedders.OpenAIEmbedder, "_async_client", failure_client)
    with pytest.raises(ConnectionError):
        await embedders.OpenAIEmbedder(api_key="k", max_retries=1)._embed_with_retry(["x"])

    retry_client = FakeAsyncOpenAI("env-key")
    retry_client.embeddings = FakeAsyncEmbeddings(
        [APIConnectionError("x"), APIConnectionError("y")]
    )
    monkeypatch.setattr(embedders.OpenAIEmbedder, "_async_client", retry_client)
    with pytest.raises(ConnectionError):
        await embedders.OpenAIEmbedder(api_key="k", max_retries=2)._embed_with_retry(["x"])

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("missing openai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        embedders.OpenAIEmbedder(api_key="k")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.mark.asyncio
async def test_ollama_embedder_paths(monkeypatch) -> None:
    embedders = _load_embedders(monkeypatch)

    class ConnectError(Exception):
        pass

    class TimeoutException(Exception):
        pass

    class HTTPStatusError(Exception):
        pass

    class FakeResponse:
        def __init__(self, embedding):
            self._embedding = embedding

        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": self._embedding}

    class FakeAsyncClient:
        def __init__(self, responses, base_url=None, timeout=None):
            self.responses = list(responses)
            self.base_url = base_url
            self.timeout = timeout
            self.closed = False

        async def post(self, path, json):
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        async def aclose(self):
            self.closed = True

    httpx_module = types.ModuleType("httpx")
    httpx_module.ConnectError = ConnectError
    httpx_module.TimeoutException = TimeoutException
    httpx_module.HTTPStatusError = HTTPStatusError
    monkeypatch.setitem(sys.modules, "httpx", httpx_module)

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(embedders.asyncio, "sleep", _no_sleep)

    client_holder = {}

    def async_client_factory(base_url=None, timeout=None):
        client = FakeAsyncClient(
            [
                ConnectError("retry"),
                FakeResponse([3.0, 4.0]),
                FakeResponse([5.0, 12.0]),
                FakeResponse([8.0, 15.0]),
            ],
            base_url=base_url,
            timeout=timeout,
        )
        client_holder["client"] = client
        return client

    httpx_module.AsyncClient = async_client_factory

    ollama = embedders.OllamaEmbedder(
        model_name="nomic-embed-text", base_url="http://ollama/", max_retries=2
    )
    client = await ollama._get_client()
    assert client.base_url == "http://ollama"
    assert ollama.dimension == 768
    assert ollama.model_name == "nomic-embed-text"
    assert ollama.max_tokens == embedders.OllamaEmbedder.DEFAULT_MAX_TOKENS
    np.testing.assert_allclose(await ollama.embed("hello"), np.array([0.6, 0.8], dtype=np.float32))
    assert ollama.dimension == 2
    np.testing.assert_array_equal(await ollama.embed(" "), np.zeros(2, dtype=np.float32))
    batch = await ollama.embed_batch(["a", "", "bb"])
    assert len(batch) == 3
    np.testing.assert_array_equal(batch[1], np.zeros(2, dtype=np.float32))
    assert await ollama.embed_batch([]) == []
    await ollama.close()
    assert client_holder["client"].closed is True

    explicit = embedders.OllamaEmbedder(model_name="unknown", dimension=123)
    assert explicit.dimension == 123
    default_dim = embedders.OllamaEmbedder(model_name="unknown")
    assert default_dim.dimension == embedders.OllamaEmbedder.DEFAULT_DIMENSION

    failing = embedders.OllamaEmbedder(model_name="nomic-embed-text", max_retries=1)
    failing._client = FakeAsyncClient([Exception("bad")])
    with pytest.raises(ConnectionError):
        await failing._embed_single_with_retry("x")

    exhausted = embedders.OllamaEmbedder(model_name="nomic-embed-text", max_retries=2)
    exhausted._client = FakeAsyncClient([ConnectError("x"), TimeoutException("y")])
    with pytest.raises(ConnectionError):
        await exhausted._embed_single_with_retry("x")

    async with embedders.OllamaEmbedder(model_name="nomic-embed-text", dimension=4) as ctx:
        assert ctx.model_name == "nomic-embed-text"

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "httpx":
            raise ImportError("missing httpx")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        embedders.OllamaEmbedder()
