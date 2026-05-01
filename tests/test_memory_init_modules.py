from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _stub_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_memory_package_init_lazy_exports(monkeypatch) -> None:
    _stub_module(
        monkeypatch,
        "headroom.memory.adapters.graph",
        InMemoryGraphStore=type("InMemoryGraphStore", (), {}),
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.backends.local",
        LocalBackend=type("LocalBackend", (), {}),
        LocalBackendConfig=type("LocalBackendConfig", (), {}),
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.bridge",
        ImportStats=type("ImportStats", (), {}),
        MemoryBridge=type("MemoryBridge", (), {}),
        SyncStats=type("SyncStats", (), {}),
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.bridge_config",
        BridgeConfig=type("BridgeConfig", (), {}),
        MarkdownFormat=type("MarkdownFormat", (), {}),
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.config",
        EmbedderBackend=type("EmbedderBackend", (), {}),
        MemoryConfig=type("MemoryConfig", (), {}),
        StoreBackend=type("StoreBackend", (), {}),
        TextBackend=type("TextBackend", (), {}),
        VectorBackend=type("VectorBackend", (), {}),
    )
    _stub_module(
        monkeypatch, "headroom.memory.core", HierarchicalMemory=type("HierarchicalMemory", (), {})
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.easy",
        Memory=type("Memory", (), {}),
        MemoryResult=type("MemoryResult", (), {}),
    )
    _stub_module(
        monkeypatch, "headroom.memory.factory", create_memory_system=lambda config=None: config
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.models",
        Memory=type("MemoryModel", (), {}),
        ScopeLevel=type("ScopeLevel", (), {}),
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.ports",
        Embedder=type("Embedder", (), {}),
        Entity=type("Entity", (), {}),
        GraphStore=type("GraphStore", (), {}),
        MemoryCache=type("MemoryCache", (), {}),
        MemoryFilter=type("MemoryFilter", (), {}),
        MemorySearchResult=type("MemorySearchResult", (), {}),
        MemoryStore=type("MemoryStore", (), {}),
        Relationship=type("Relationship", (), {}),
        Subgraph=type("Subgraph", (), {}),
        TextFilter=type("TextFilter", (), {}),
        TextIndex=type("TextIndex", (), {}),
        TextSearchResult=type("TextSearchResult", (), {}),
        VectorFilter=type("VectorFilter", (), {}),
        VectorIndex=type("VectorIndex", (), {}),
        VectorSearchResult=type("VectorSearchResult", (), {}),
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.system",
        MemoryBackend=type("MemoryBackend", (), {}),
        MemorySystem=type("MemorySystem", (), {}),
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.tools",
        MEMORY_TOOLS=["basic"],
        MEMORY_TOOLS_OPTIMIZED=["optimized"],
        get_memory_tools=lambda: ["basic"],
        get_memory_tools_optimized=lambda: ["optimized"],
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.wrapper",
        MemoryWrapper=type("MemoryWrapper", (), {}),
        with_memory=lambda client, **kwargs: client,
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.wrapper_tools",
        MemoryToolsWrapper=type("MemoryToolsWrapper", (), {}),
        with_memory_tools=lambda client, **kwargs: client,
    )
    mem0_backend = type("Mem0Backend", (), {})
    mem0_config = type("Mem0Config", (), {})
    direct_adapter = type("DirectMem0Adapter", (), {})
    _stub_module(
        monkeypatch,
        "headroom.memory.backends.mem0",
        Mem0Backend=mem0_backend,
        Mem0Config=mem0_config,
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.backends.direct_mem0",
        DirectMem0Adapter=direct_adapter,
        Mem0Config=type("DirectMem0Config", (), {}),
    )

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "__init__.py"
    monkeypatch.delitem(sys.modules, "headroom.memory", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory"] = module
    spec.loader.exec_module(module)

    assert module.Memory.__name__ == "Memory"
    assert module.__getattr__("Mem0Backend") is mem0_backend
    assert module.__getattr__("Mem0Config") is mem0_config
    assert module.__getattr__("DirectMem0Adapter") is direct_adapter
    assert module.__getattr__("DirectMem0Config").__name__ == "DirectMem0Config"
    assert module.__getattr__("DirectMem0Adapter") is direct_adapter
    assert "with_memory_tools" in module.__all__
    assert "MemoryModel" in module.__all__

    try:
        module.__getattr__("MissingThing")
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")


def test_memory_adapters_init_lazy_exports(monkeypatch) -> None:
    cache = type("LRUMemoryCache", (), {})
    fts5 = type("FTS5TextIndex", (), {})
    graph = type("InMemoryGraphStore", (), {})
    sqlite = type("SQLiteMemoryStore", (), {})
    sqlite_graph = type("SQLiteGraphStore", (), {})
    _stub_module(monkeypatch, "headroom.memory.adapters.cache", LRUMemoryCache=cache)
    _stub_module(monkeypatch, "headroom.memory.adapters.fts5", FTS5TextIndex=fts5)
    _stub_module(monkeypatch, "headroom.memory.adapters.graph", InMemoryGraphStore=graph)
    _stub_module(monkeypatch, "headroom.memory.adapters.sqlite", SQLiteMemoryStore=sqlite)
    _stub_module(
        monkeypatch, "headroom.memory.adapters.sqlite_graph", SQLiteGraphStore=sqlite_graph
    )

    hnsw_cls = type("HNSWVectorIndex", (), {})
    sqlite_vec_cls = type("SQLiteVectorIndex", (), {})
    local_embedder = type("LocalEmbedder", (), {})
    openai_embedder = type("OpenAIEmbedder", (), {})
    ollama_embedder = type("OllamaEmbedder", (), {})
    _stub_module(
        monkeypatch,
        "headroom.memory.adapters.hnsw",
        HNSWVectorIndex=hnsw_cls,
        _check_hnswlib_available=lambda: True,
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.adapters.sqlite_vector",
        SQLiteVectorIndex=sqlite_vec_cls,
        is_sqlite_vec_available=lambda: False,
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.adapters.embedders",
        LocalEmbedder=local_embedder,
        OpenAIEmbedder=openai_embedder,
        OllamaEmbedder=ollama_embedder,
    )

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "adapters" / "__init__.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.adapters", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.adapters", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.adapters"] = module
    spec.loader.exec_module(module)

    assert module.LRUMemoryCache is cache
    assert module.FTS5TextIndex is fts5
    assert module.HNSW_AVAILABLE is True
    assert module.HNSW_AVAILABLE is True
    assert module.SQLITE_VEC_AVAILABLE is False
    assert module.HNSWVectorIndex is hnsw_cls
    assert module.SQLiteVectorIndex is sqlite_vec_cls
    assert module.LocalEmbedder is local_embedder
    assert module.OpenAIEmbedder is openai_embedder
    assert module.OllamaEmbedder is ollama_embedder
    assert "SQLITE_VEC_AVAILABLE" in module.__all__

    try:
        module.__getattr__("MissingThing")
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")


def test_memory_backends_init_lazy_exports(monkeypatch) -> None:
    local_backend = type("LocalBackend", (), {})
    local_config = type("LocalBackendConfig", (), {})
    mem0_backend = type("Mem0Backend", (), {})
    mem0_config = type("Mem0Config", (), {})
    mem0_system = type("Mem0SystemAdapter", (), {})
    direct_adapter = type("DirectMem0Adapter", (), {})
    direct_config = type("DirectMem0Config", (), {})

    _stub_module(
        monkeypatch,
        "headroom.memory.backends.local",
        LocalBackend=local_backend,
        LocalBackendConfig=local_config,
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.backends.mem0",
        Mem0Backend=mem0_backend,
        Mem0Config=mem0_config,
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.backends.mem0_system_adapter",
        Mem0SystemAdapter=mem0_system,
    )
    _stub_module(
        monkeypatch,
        "headroom.memory.backends.direct_mem0",
        DirectMem0Adapter=direct_adapter,
        Mem0Config=direct_config,
    )

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "backends" / "__init__.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.backends", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.backends", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.backends"] = module
    spec.loader.exec_module(module)

    assert module.LocalBackend is local_backend
    assert module.LocalBackendConfig is local_config
    assert module.__getattr__("Mem0Backend") is mem0_backend
    assert module.__getattr__("Mem0Config") is mem0_config
    assert module.__getattr__("Mem0SystemAdapter") is mem0_system
    assert module.__getattr__("DirectMem0Adapter") is direct_adapter
    assert module.__getattr__("DirectMem0Config") is direct_config
    assert module.__getattr__("DirectMem0Adapter") is direct_adapter
    assert "DirectMem0Config" in module.__all__

    try:
        module.__getattr__("MissingThing")
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")


def test_memory_tools_exports_are_copied() -> None:
    from headroom.memory import tools

    copied_tools = tools.get_memory_tools()
    copied_optimized = tools.get_memory_tools_optimized()

    assert copied_tools == tools.MEMORY_TOOLS
    assert copied_optimized == tools.MEMORY_TOOLS_OPTIMIZED
    assert copied_tools is not tools.MEMORY_TOOLS
    assert copied_optimized is not tools.MEMORY_TOOLS_OPTIMIZED
    assert tools.get_tool_names() == [
        "memory_save",
        "memory_search",
        "memory_update",
        "memory_delete",
    ]
    assert (
        tools.MEMORY_SAVE_OPTIMIZED["function"]["parameters"]["properties"]["background"]["type"]
        == "boolean"
    )
