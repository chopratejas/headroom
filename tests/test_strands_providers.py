from __future__ import annotations

import importlib
import sys
import types


def test_get_headroom_provider_and_model_name(monkeypatch) -> None:
    providers = importlib.import_module("headroom.integrations.strands.providers")

    class _AnthropicProvider:
        pass

    class _OpenAIProvider:
        pass

    class _GoogleProvider:
        pass

    monkeypatch.setattr(providers, "AnthropicProvider", _AnthropicProvider)
    monkeypatch.setattr(providers, "OpenAIProvider", _OpenAIProvider)
    monkeypatch.setattr(providers, "GoogleProvider", _GoogleProvider)
    monkeypatch.setattr(
        providers,
        "_STRANDS_MODEL_PROVIDERS",
        {
            "BedrockModel": _AnthropicProvider,
            "GeminiModel": _GoogleProvider,
            "OpenAIModel": _OpenAIProvider,
        },
    )

    bedrock_model = type("BedrockModel", (), {"__module__": "vendor.any"})()
    assert isinstance(providers.get_headroom_provider(bedrock_model), _AnthropicProvider)

    anthropic_mod = type("CustomModel", (), {"__module__": "sdk.anthropic.models"})()
    assert isinstance(providers.get_headroom_provider(anthropic_mod), _AnthropicProvider)

    google_mod = type("CustomModel", (), {"__module__": "sdk.google.gemini"})()
    assert isinstance(providers.get_headroom_provider(google_mod), _GoogleProvider)

    openai_mod = type("CustomModel", (), {"__module__": "sdk.openai.client"})()
    assert isinstance(providers.get_headroom_provider(openai_mod), _OpenAIProvider)

    claude_id = type("UnknownModel", (), {"__module__": "sdk.other", "model_id": "claude-3-5"})()
    assert isinstance(providers.get_headroom_provider(claude_id), _AnthropicProvider)

    gemini_id = type("UnknownModel", (), {"__module__": "sdk.other", "model": "gemini-2.5"})()
    assert isinstance(providers.get_headroom_provider(gemini_id), _GoogleProvider)

    gpt_id = type("UnknownModel", (), {"__module__": "sdk.other", "model_name": "gpt-4o"})()
    assert isinstance(providers.get_headroom_provider(gpt_id), _OpenAIProvider)

    fallback = type("UnknownModel", (), {"__module__": "sdk.other"})()
    assert isinstance(providers.get_headroom_provider(fallback), _OpenAIProvider)

    assert providers._extract_model_id(type("M", (), {"id": "id-model"})()) == "id-model"
    assert (
        providers._extract_model_id(type("M", (), {"config": {"model_id": "cfg-model"}})())
        == "cfg-model"
    )
    assert (
        providers._extract_model_id(
            type("M", (), {"config": type("Cfg", (), {"model_name": "cfg-name"})()})()
        )
        == "cfg-name"
    )
    assert (
        providers._extract_model_id(
            type("M", (), {"get_config": lambda self: {"model": "via-get-config"}})()
        )
        == "via-get-config"
    )
    assert (
        providers._extract_model_id(
            type(
                "M", (), {"get_config": lambda self: (_ for _ in ()).throw(RuntimeError("boom"))}
            )()
        )
        == ""
    )

    assert (
        providers.get_model_name_from_strands(type("M", (), {"model_id": "named-model"})())
        == "named-model"
    )
    assert providers.get_model_name_from_strands(type("M", (), {})()) == "gpt-4o"


def test_strands_package_lazy_exports(monkeypatch) -> None:
    module = importlib.import_module("headroom.integrations.strands")
    monkeypatch.setattr(
        module.importlib.util, "find_spec", lambda name: object() if name == "strands" else None
    )
    assert module.strands_available() is True
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)
    assert module.strands_available() is False

    fake_hooks = types.ModuleType("headroom.integrations.strands.hooks")
    fake_hooks.HeadroomHookProvider = "HOOK"
    fake_model = types.ModuleType("headroom.integrations.strands.model")
    fake_model.HeadroomStrandsModel = "MODEL"
    fake_model.OptimizationMetrics = "METRICS"
    fake_model.optimize_messages = "OPTIMIZE"
    fake_providers = types.ModuleType("headroom.integrations.strands.providers")
    fake_providers.get_headroom_provider = "GET_PROVIDER"
    fake_providers.get_model_name_from_strands = "GET_MODEL"

    monkeypatch.setitem(sys.modules, "headroom.integrations.strands.hooks", fake_hooks)
    monkeypatch.setitem(sys.modules, "headroom.integrations.strands.model", fake_model)
    monkeypatch.setitem(sys.modules, "headroom.integrations.strands.providers", fake_providers)

    assert module.__getattr__("HeadroomHookProvider") == "HOOK"
    assert module.__getattr__("HeadroomStrandsModel") == "MODEL"
    assert module.__getattr__("OptimizationMetrics") == "METRICS"
    assert module.__getattr__("optimize_messages") == "OPTIMIZE"
    assert module.__getattr__("get_headroom_provider") == "GET_PROVIDER"
    assert module.__getattr__("get_model_name_from_strands") == "GET_MODEL"

    try:
        module.__getattr__("missing")
    except AttributeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected AttributeError for missing export")
