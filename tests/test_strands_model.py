from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest


def _load_model_module(monkeypatch):
    strands_module = types.ModuleType("strands")
    strands_module.__spec__ = importlib.machinery.ModuleSpec("strands", loader=None)
    models_module = types.ModuleType("strands.models")
    models_module.__spec__ = importlib.machinery.ModuleSpec("strands.models", loader=None)
    content_module = types.ModuleType("strands.types.content")
    content_module.__spec__ = importlib.machinery.ModuleSpec("strands.types.content", loader=None)
    streaming_module = types.ModuleType("strands.types.streaming")
    streaming_module.__spec__ = importlib.machinery.ModuleSpec(
        "strands.types.streaming", loader=None
    )
    tools_module = types.ModuleType("strands.types.tools")
    tools_module.__spec__ = importlib.machinery.ModuleSpec("strands.types.tools", loader=None)

    class Model:
        pass

    models_module.Model = Model
    content_module.Message = dict
    content_module.Messages = list
    content_module.SystemContentBlock = dict
    streaming_module.StreamEvent = dict
    tools_module.ToolChoice = dict
    tools_module.ToolSpec = dict

    monkeypatch.setitem(sys.modules, "strands", strands_module)
    monkeypatch.setitem(sys.modules, "strands.models", models_module)
    monkeypatch.setitem(sys.modules, "strands.types.content", content_module)
    monkeypatch.setitem(sys.modules, "strands.types.streaming", streaming_module)
    monkeypatch.setitem(sys.modules, "strands.types.tools", tools_module)
    monkeypatch.delitem(sys.modules, "headroom.integrations.strands.model", raising=False)
    return importlib.import_module("headroom.integrations.strands.model")


class _FakeProvider:
    def __init__(self, limit: int = 32000) -> None:
        self.limit = limit
        self.requested_models: list[str] = []

    def get_context_limit(self, model: str) -> int:
        self.requested_models.append(model)
        return self.limit


class _FakePipeline:
    def __init__(self, *, config, provider) -> None:
        self.config = config
        self.provider = provider
        self.result = SimpleNamespace(
            messages=[{"role": "user", "content": "optimized"}],
            tokens_before=100,
            tokens_after=40,
            transforms_applied=["smart_crusher"],
        )
        self.calls: list[dict] = []
        self.error: Exception | None = None

    def apply(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class _WrappedModel:
    def __init__(self) -> None:
        self.config = {"model_id": "gpt-4o"}
        self.updated = None
        self.custom_attr = "wrapped"

    async def stream(self, messages, **kwargs):
        yield {"kind": "stream", "messages": messages, "kwargs": kwargs}

    async def structured_output(self, output_model, prompt, **kwargs):
        yield {
            "kind": "structured",
            "output_model": output_model,
            "prompt": prompt,
            "kwargs": kwargs,
        }

    def get_config(self):
        return {"wrapped": True}

    def update_config(self, **model_config):
        self.updated = model_config


def test_strands_model_conversion_and_optimization(monkeypatch) -> None:
    model_module = _load_model_module(monkeypatch)
    fake_provider = _FakeProvider()
    fake_pipelines: list[_FakePipeline] = []

    monkeypatch.setattr(model_module, "HeadroomConfig", lambda: SimpleNamespace(source="default"))
    monkeypatch.setattr(model_module, "OpenAIProvider", lambda: _FakeProvider(limit=64000))
    monkeypatch.setattr(model_module, "get_headroom_provider", lambda wrapped: fake_provider)
    monkeypatch.setattr(
        model_module, "get_model_name_from_strands", lambda wrapped: "detected-model"
    )
    monkeypatch.setattr(
        model_module,
        "TransformPipeline",
        lambda **kwargs: fake_pipelines.append(_FakePipeline(**kwargs)) or fake_pipelines[-1],
    )
    monkeypatch.setattr(model_module, "uuid4", lambda: "req-1")

    with pytest.raises(ValueError):
        model_module.HeadroomStrandsModel(None)

    wrapped = _WrappedModel()
    optimized_model = model_module.HeadroomStrandsModel(
        wrapped_model=wrapped,
        config=SimpleNamespace(name="cfg"),
    )
    assert optimized_model.config == wrapped.config
    assert optimized_model.total_tokens_saved == 0
    assert optimized_model.metrics_history == []

    class _MessageObj:
        role = "assistant"
        content = [{"type": "text", "text": "hello"}]
        tool_calls = [{"id": "tool"}]
        tool_call_id = "call-1"
        name = "assistant-name"

    converted = optimized_model._convert_messages_to_openai(
        [
            {
                "role": "user",
                "content": None,
                "tool_calls": [{"id": 1}],
                "tool_call_id": "x",
                "name": "nm",
            },
            _MessageObj(),
            42,
        ]
    )
    assert converted[0]["content"] == ""
    assert converted[1]["tool_call_id"] == "call-1"
    assert converted[2] == {"role": "user", "content": "42"}

    back = optimized_model._convert_messages_from_openai(
        [
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [1],
                "tool_call_id": "id",
                "name": "name",
            }
        ],
        [],
    )
    assert back == [
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [1],
            "tool_call_id": "id",
            "name": "name",
        }
    ]

    empty_messages, empty_metrics = optimized_model._optimize_messages([])
    assert empty_messages == []
    assert empty_metrics.tokens_before == 0
    assert empty_metrics.model == "detected-model"

    optimized_messages, metrics = optimized_model._optimize_messages(
        [{"role": "user", "content": "please optimize"}]
    )
    assert optimized_messages == [{"role": "user", "content": "optimized"}]
    assert metrics.tokens_saved == 60
    assert fake_provider.requested_models == ["detected-model"]
    assert fake_pipelines[0].calls[0]["model_limit"] == 32000

    fake_pipelines[0].error = RuntimeError("pipeline failed")
    fallback_messages, fallback_metrics = optimized_model._optimize_messages(
        [{"role": "user", "content": "12345678"}]
    )
    assert fallback_messages == [{"role": "user", "content": "12345678"}]
    assert fallback_metrics.transforms_applied == ["fallback:error"]
    assert fallback_metrics.tokens_before == 2

    for idx in range(110):
        optimized_model._metrics_history.append(
            model_module.OptimizationMetrics(
                request_id=str(idx),
                timestamp=fallback_metrics.timestamp,
                tokens_before=10,
                tokens_after=5,
                tokens_saved=5,
                savings_percent=50.0,
                transforms_applied=[],
                model="m",
            )
        )
    optimized_model._metrics_history = optimized_model._metrics_history[-100:]
    assert len(optimized_model.metrics_history) == 100
    assert optimized_model.get_savings_summary()["total_requests"] == 100

    optimized_model.reset()
    assert optimized_model.get_savings_summary() == {
        "total_requests": 0,
        "total_tokens_saved": 0,
        "average_savings_percent": 0,
        "total_tokens_before": 0,
        "total_tokens_after": 0,
    }

    assert optimized_model.custom_attr == "wrapped"
    with pytest.raises(AttributeError):
        optimized_model.__getattr__("wrapped_model")

    assert optimized_model.get_config() == {"wrapped": True}
    optimized_model.update_config(timeout=30)
    assert wrapped.updated == {"timeout": 30}

    manual_messages, manual_metrics = model_module.optimize_messages(
        [{"role": "user", "content": "hi"}, SimpleNamespace(role="assistant", content="there"), 7],
        config=SimpleNamespace(name="manual"),
        model="gpt-4o-mini",
    )
    assert manual_messages == [{"role": "user", "content": "optimized"}]
    assert manual_metrics["tokens_saved"] == 60


@pytest.mark.asyncio
async def test_strands_model_stream_and_structured_output(monkeypatch) -> None:
    model_module = _load_model_module(monkeypatch)
    fake_provider = _FakeProvider()
    fake_pipelines: list[_FakePipeline] = []

    monkeypatch.setattr(model_module, "HeadroomConfig", lambda: SimpleNamespace(source="default"))
    monkeypatch.setattr(model_module, "OpenAIProvider", lambda: _FakeProvider(limit=64000))
    monkeypatch.setattr(model_module, "get_headroom_provider", lambda wrapped: fake_provider)
    monkeypatch.setattr(
        model_module, "get_model_name_from_strands", lambda wrapped: "detected-model"
    )
    monkeypatch.setattr(
        model_module,
        "TransformPipeline",
        lambda **kwargs: fake_pipelines.append(_FakePipeline(**kwargs)) or fake_pipelines[-1],
    )

    wrapped = _WrappedModel()
    optimized_model = model_module.HeadroomStrandsModel(wrapped_model=wrapped)

    streamed = [
        event
        async for event in optimized_model.stream(
            [{"role": "user", "content": "stream me"}],
            tool_specs=[{"name": "tool"}],
            system_prompt="sys",
            tool_choice={"type": "auto"},
            system_prompt_content=[{"type": "text", "text": "sys"}],
            invocation_state={"id": 1},
            temperature=0.2,
        )
    ]
    assert streamed[0]["messages"] == [{"role": "user", "content": "optimized"}]
    assert streamed[0]["kwargs"]["temperature"] == 0.2

    structured = [
        event
        async for event in optimized_model.structured_output(
            dict,
            [{"role": "user", "content": "structure me"}],
            system_prompt="sys",
            max_tokens=10,
        )
    ]
    assert structured[0]["prompt"] == [{"role": "user", "content": "optimized"}]
    assert structured[0]["kwargs"]["max_tokens"] == 10
