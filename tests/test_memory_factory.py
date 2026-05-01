from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


def _load_factory_module(monkeypatch):
    class EmbedderBackend:
        LOCAL = "local"
        ONNX = "onnx"
        OPENAI = "openai"
        OLLAMA = "ollama"

    class StoreBackend:
        SQLITE = "sqlite"
        EXTERNAL = "external"

    class VectorBackend:
        AUTO = "auto"
        SQLITE_VEC = "sqlite_vec"
        HNSW = "hnsw"
        EXTERNAL = "external"

    class TextBackend:
        FTS5 = "fts5"
        EXTERNAL = "external"

    @dataclass
    class MemoryConfig:
        db_path: Path = Path("memory.db")
        cache_enabled: bool = True
        cache_max_size: int = 100
        store_backend: str = StoreBackend.SQLITE
        store_backend_name: str | None = None
        embedder_backend: str = EmbedderBackend.LOCAL
        embedder_model: str = "mini"
        openai_api_key: str | None = None
        ollama_base_url: str = "http://ollama"
        vector_backend: str = VectorBackend.AUTO
        vector_backend_name: str | None = None
        vector_db_path: Path | None = None
        vector_dimension: int = 384
        vector_cache_size_kb: int = 1024
        hnsw_ef_construction: int = 50
        hnsw_m: int = 16
        hnsw_ef_search: int = 20
        hnsw_max_entries: int = 500
        text_backend: str = TextBackend.FTS5
        text_backend_name: str | None = None

    config_module = types.ModuleType("headroom.memory.config")
    config_module.EmbedderBackend = EmbedderBackend
    config_module.StoreBackend = StoreBackend
    config_module.VectorBackend = VectorBackend
    config_module.TextBackend = TextBackend
    config_module.MemoryConfig = MemoryConfig
    monkeypatch.setitem(sys.modules, "headroom.memory.config", config_module)

    adapters_module = types.ModuleType("headroom.memory.adapters")
    adapters_module.SQLITE_VEC_AVAILABLE = False
    adapters_module.HNSW_AVAILABLE = False
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters", adapters_module)

    sqlite_module = types.ModuleType("headroom.memory.adapters.sqlite")
    sqlite_module.SQLiteMemoryStore = lambda db_path: ("sqlite-store", db_path)
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.sqlite", sqlite_module)

    embedders_module = types.ModuleType("headroom.memory.adapters.embedders")
    embedders_module.LocalEmbedder = lambda model_name: ("local-embedder", model_name)
    embedders_module.OnnxLocalEmbedder = lambda: ("onnx-embedder",)
    embedders_module.OpenAIEmbedder = lambda api_key, model_name: (
        "openai-embedder",
        api_key,
        model_name,
    )
    embedders_module.OllamaEmbedder = lambda base_url, model_name: (
        "ollama-embedder",
        base_url,
        model_name,
    )
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.embedders", embedders_module)

    sqlite_vector_module = types.ModuleType("headroom.memory.adapters.sqlite_vector")
    sqlite_vector_module.SQLiteVectorIndex = lambda dimension, db_path, page_cache_size_kb: (
        "sqlite-vec",
        dimension,
        db_path,
        page_cache_size_kb,
    )
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.sqlite_vector", sqlite_vector_module)

    hnsw_module = types.ModuleType("headroom.memory.adapters.hnsw")
    hnsw_module.HNSWVectorIndex = lambda **kwargs: ("hnsw", kwargs)
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.hnsw", hnsw_module)

    fts5_module = types.ModuleType("headroom.memory.adapters.fts5")
    fts5_module.FTS5TextIndex = lambda db_path: ("fts5", db_path)
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.fts5", fts5_module)

    cache_module = types.ModuleType("headroom.memory.adapters.cache")
    cache_module.LRUMemoryCache = lambda max_size: ("cache", max_size)
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.cache", cache_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "factory.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.factory", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.factory", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.factory"] = module
    spec.loader.exec_module(module)
    return (
        module,
        MemoryConfig,
        EmbedderBackend,
        StoreBackend,
        VectorBackend,
        TextBackend,
        adapters_module,
    )


def test_memory_factory_branching(monkeypatch) -> None:
    factory, MemoryConfig, EmbedderBackend, StoreBackend, VectorBackend, TextBackend, adapters = (
        _load_factory_module(monkeypatch)
    )

    class FakeEntryPoint:
        def __init__(self, name):
            self.name = name

        def load(self):
            return lambda config: ("external", self.name, config.db_path)

    monkeypatch.setattr(factory, "entry_points", lambda group: [FakeEntryPoint("custom")])
    assert factory._load_external_backend("g", "custom", "field", MemoryConfig()) == (
        "external",
        "custom",
        Path("memory.db"),
    )

    for fn in (
        lambda: factory._load_external_backend("g", None, "field", MemoryConfig()),
        lambda: factory._load_external_backend("g", "missing", "field", MemoryConfig()),
    ):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    assert factory._create_store(MemoryConfig(store_backend=StoreBackend.SQLITE)) == (
        "sqlite-store",
        Path("memory.db"),
    )
    assert (
        factory._create_store(
            MemoryConfig(store_backend=StoreBackend.EXTERNAL, store_backend_name="custom")
        )[0]
        == "external"
    )
    try:
        factory._create_store(MemoryConfig(store_backend="bad"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected store error")

    assert factory._create_embedder(MemoryConfig(embedder_backend=EmbedderBackend.LOCAL)) == (
        "local-embedder",
        "mini",
    )
    assert factory._create_embedder(MemoryConfig(embedder_backend=EmbedderBackend.ONNX)) == (
        "onnx-embedder",
    )
    try:
        factory._create_embedder(MemoryConfig(embedder_backend=EmbedderBackend.OPENAI))
    except ValueError:
        pass
    else:
        raise AssertionError("expected openai key error")
    assert factory._create_embedder(
        MemoryConfig(embedder_backend=EmbedderBackend.OPENAI, openai_api_key="k")
    ) == ("openai-embedder", "k", "mini")
    assert factory._create_embedder(MemoryConfig(embedder_backend=EmbedderBackend.OLLAMA)) == (
        "ollama-embedder",
        "http://ollama",
        "mini",
    )
    try:
        factory._create_embedder(MemoryConfig(embedder_backend="bad"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected embedder error")

    assert (
        factory._create_vector_index(
            MemoryConfig(vector_backend=VectorBackend.EXTERNAL, vector_backend_name="custom")
        )[0]
        == "external"
    )
    try:
        factory._create_vector_index(MemoryConfig(vector_backend=VectorBackend.AUTO))
    except ValueError:
        pass
    else:
        raise AssertionError("expected auto error")

    adapters.SQLITE_VEC_AVAILABLE = True
    sqlite_vec = factory._create_vector_index(MemoryConfig(vector_backend=VectorBackend.AUTO))
    assert sqlite_vec[0] == "sqlite-vec"
    assert sqlite_vec[2] == Path("memory_vectors.db")

    adapters.SQLITE_VEC_AVAILABLE = False
    try:
        factory._create_vector_index(MemoryConfig(vector_backend=VectorBackend.SQLITE_VEC))
    except ValueError:
        pass
    else:
        raise AssertionError("expected sqlite-vec error")

    adapters.SQLITE_VEC_AVAILABLE = True
    explicit_sqlite = factory._create_vector_index(
        MemoryConfig(vector_backend=VectorBackend.SQLITE_VEC, vector_db_path=Path("custom.db"))
    )
    assert explicit_sqlite[2] == Path("custom.db")

    adapters.SQLITE_VEC_AVAILABLE = False
    adapters.HNSW_AVAILABLE = True
    hnsw = factory._create_vector_index(MemoryConfig(vector_backend=VectorBackend.HNSW))
    assert hnsw[0] == "hnsw"
    assert hnsw[1]["save_path"] == Path("memory_hnsw")
    try:
        adapters.HNSW_AVAILABLE = False
        factory._create_vector_index(MemoryConfig(vector_backend=VectorBackend.HNSW))
    except ValueError:
        pass
    else:
        raise AssertionError("expected hnsw error")
    try:
        factory._create_vector_index(MemoryConfig(vector_backend="bad"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected vector error")

    assert factory._create_text_index(MemoryConfig(text_backend=TextBackend.FTS5)) == (
        "fts5",
        Path("memory.db"),
    )
    assert (
        factory._create_text_index(
            MemoryConfig(text_backend=TextBackend.EXTERNAL, text_backend_name="custom")
        )[0]
        == "external"
    )
    try:
        factory._create_text_index(MemoryConfig(text_backend="bad"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected text error")

    assert factory._create_cache(MemoryConfig(cache_max_size=77)) == ("cache", 77)


@pytest.mark.asyncio
async def test_create_memory_system_wires_components(monkeypatch) -> None:
    factory, MemoryConfig, EmbedderBackend, StoreBackend, VectorBackend, TextBackend, adapters = (
        _load_factory_module(monkeypatch)
    )
    adapters.SQLITE_VEC_AVAILABLE = True
    config = MemoryConfig(
        embedder_backend=EmbedderBackend.LOCAL,
        store_backend=StoreBackend.SQLITE,
        vector_backend=VectorBackend.AUTO,
        text_backend=TextBackend.FTS5,
        cache_enabled=True,
    )
    store, vector, text, embedder, cache = await factory.create_memory_system(config)
    assert store[0] == "sqlite-store"
    assert vector[0] == "sqlite-vec"
    assert text[0] == "fts5"
    assert embedder[0] == "local-embedder"
    assert cache[0] == "cache"

    _, _, _, _, no_cache = await factory.create_memory_system(
        MemoryConfig(cache_enabled=False, vector_backend=VectorBackend.AUTO)
    )
    assert no_cache is None
