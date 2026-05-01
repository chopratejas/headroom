from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_module(monkeypatch, module_name: str, relative_parts: list[str]):
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root.joinpath(*relative_parts)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_graph_models_round_trip_and_subgraph_helpers(monkeypatch) -> None:
    graph_models = _load_module(
        monkeypatch,
        "headroom.memory.adapters.graph_models",
        ["headroom", "memory", "adapters", "graph_models.py"],
    )

    assert graph_models.RelationshipDirection.OUTGOING.value == "outgoing"
    assert graph_models.RelationshipDirection.INCOMING.value == "incoming"
    assert graph_models.RelationshipDirection.BOTH.value == "both"

    entity = graph_models.Entity(
        id="e1",
        user_id="u1",
        name="Alice",
        entity_type="person",
        description="Team lead",
        properties={"team": "platform"},
        metadata={"source": "test"},
    )
    entity_round_trip = graph_models.Entity.from_dict(entity.to_dict())
    assert entity_round_trip.id == "e1"
    assert entity_round_trip.properties == {"team": "platform"}
    assert entity_round_trip.metadata == {"source": "test"}

    minimal_entity = graph_models.Entity.from_dict(
        {
            "id": "e2",
            "user_id": "u1",
            "name": "Project X",
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
        }
    )
    assert minimal_entity.entity_type == "unknown"
    assert minimal_entity.description is None
    assert minimal_entity.properties == {}
    assert minimal_entity.metadata == {}

    relationship = graph_models.Relationship(
        id="r1",
        user_id="u1",
        source_id="e1",
        target_id="e2",
        relation_type="owns",
        weight=0.7,
        properties={"confidence": "high"},
        metadata={"source": "test"},
    )
    relationship_round_trip = graph_models.Relationship.from_dict(relationship.to_dict())
    assert relationship_round_trip.id == "r1"
    assert relationship_round_trip.weight == 0.7
    assert relationship_round_trip.properties == {"confidence": "high"}
    assert relationship_round_trip.metadata == {"source": "test"}

    minimal_relationship = graph_models.Relationship.from_dict(
        {
            "id": "r2",
            "user_id": "u1",
            "source_id": "e2",
            "target_id": "e1",
            "created_at": relationship.created_at.isoformat(),
        }
    )
    assert minimal_relationship.relation_type == "related_to"
    assert minimal_relationship.weight == 1.0
    assert minimal_relationship.properties == {}
    assert minimal_relationship.metadata == {}

    subgraph = graph_models.Subgraph(
        entities=[entity, minimal_entity],
        relationships=[relationship, minimal_relationship],
        root_entity_ids=["e1"],
    )
    assert subgraph.entity_ids == {"e1", "e2"}
    assert subgraph.relationship_ids == {"r1", "r2"}
    assert subgraph.get_entity("e1") is entity
    assert subgraph.get_entity("missing") is None
    assert {neighbor.id for neighbor in subgraph.get_neighbors("e1")} == {"e2"}
    assert subgraph.to_dict()["root_entity_ids"] == ["e1"]

    rebuilt = graph_models.Subgraph.from_dict(subgraph.to_dict())
    assert rebuilt.root_entity_ids == ["e1"]
    assert rebuilt.entity_ids == {"e1", "e2"}
    assert rebuilt.relationship_ids == {"r1", "r2"}


def test_bridge_config_validates_and_normalizes_paths(monkeypatch) -> None:
    import headroom as headroom_package

    paths_module = types.ModuleType("headroom.paths")
    paths_module.bridge_state_path = lambda: Path("state") / "bridge.json"
    monkeypatch.setitem(sys.modules, "headroom.paths", paths_module)
    monkeypatch.setattr(headroom_package, "paths", paths_module, raising=False)

    bridge_config = _load_module(
        monkeypatch,
        "headroom.memory.bridge_config",
        ["headroom", "memory", "bridge_config.py"],
    )

    config = bridge_config.BridgeConfig(
        md_paths=["notes.md", Path("more.md")],
        sync_state_path="sync.json",
        export_path="export.md",
    )
    assert config.md_format is bridge_config.MarkdownFormat.AUTO
    assert config.export_format is bridge_config.MarkdownFormat.GENERIC
    assert config.md_paths == [Path("notes.md"), Path("more.md")]
    assert config.sync_state_path == Path("sync.json")
    assert config.export_path == Path("export.md")
    assert config.heading_importance_map[1] == 0.9
    assert config.source_tag == "memory_bridge"

    default_config = bridge_config.BridgeConfig()
    assert default_config.sync_state_path == Path("state") / "bridge.json"

    with pytest.raises(ValueError, match="default_importance"):
        bridge_config.BridgeConfig(default_importance=1.5)

    with pytest.raises(ValueError, match="dedup_similarity_threshold"):
        bridge_config.BridgeConfig(dedup_similarity_threshold=-0.1)


def test_memory_config_validates_and_uses_default_model(monkeypatch) -> None:
    config_source = types.ModuleType("headroom.models.config")
    config_source.ML_MODEL_DEFAULTS = types.SimpleNamespace(sentence_transformer="mini-model")
    monkeypatch.setitem(sys.modules, "headroom.models.config", config_source)

    memory_config = _load_module(
        monkeypatch,
        "headroom.memory.config",
        ["headroom", "memory", "config.py"],
    )

    config = memory_config.MemoryConfig(db_path="memory.db")
    assert config.store_backend is memory_config.StoreBackend.SQLITE
    assert config.vector_backend is memory_config.VectorBackend.AUTO
    assert config.text_backend is memory_config.TextBackend.FTS5
    assert config.embedder_backend is memory_config.EmbedderBackend.LOCAL
    assert config.embedder_model == "mini-model"
    assert config.db_path == Path("memory.db")
    assert memory_config.StoreBackend.EXTERNAL.value == "external"
    assert memory_config.VectorBackend.SQLITE_VEC.value == "sqlite_vec"
    assert memory_config.TextBackend.EXTERNAL.value == "external"
    assert memory_config.EmbedderBackend.ONNX.value == "onnx"

    with pytest.raises(ValueError, match="vector_dimension"):
        memory_config.MemoryConfig(vector_dimension=0)

    with pytest.raises(ValueError, match="hnsw_ef_construction"):
        memory_config.MemoryConfig(hnsw_ef_construction=0)

    with pytest.raises(ValueError, match="hnsw_m"):
        memory_config.MemoryConfig(hnsw_m=0)

    with pytest.raises(ValueError, match="hnsw_ef_search"):
        memory_config.MemoryConfig(hnsw_ef_search=0)

    with pytest.raises(ValueError, match="cache_max_size"):
        memory_config.MemoryConfig(cache_max_size=0)

    with pytest.raises(ValueError, match="openai_api_key"):
        memory_config.MemoryConfig(embedder_backend=memory_config.EmbedderBackend.OPENAI)
