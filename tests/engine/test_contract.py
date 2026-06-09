import dataclasses

import pytest

from headroom.engine.contract import (
    Flavor,
    Provider,
    RequestContext,
    RequestDecision,
    ResponseTelemetry,
    StreamContext,
)


def test_request_context_carries_fields():
    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view={"x-api-key": "redacted"},
        raw_body=b'{"model":"claude"}',
        session_key="abc123",
    )
    assert ctx.provider is Provider.ANTHROPIC
    assert ctx.flavor is Flavor.MESSAGES
    assert ctx.raw_body == b'{"model":"claude"}'
    assert ctx.session_key == "abc123"
    assert ctx.headers_view["x-api-key"] == "redacted"


def test_request_decision_defaults_to_passthrough_telemetry():
    body = b'{"x":1}'
    dec = RequestDecision(body=body, telemetry=ResponseTelemetry())
    assert dec.body is body
    assert dec.telemetry.compressed is False
    assert dec.telemetry.bytes_saved == 0
    assert dec.notes == {}


def test_stream_context_is_per_stream_handle():
    sc = StreamContext(session_key="k", provider=Provider.OPENAI, flavor=Flavor.CHAT)
    sc.state["seen_done"] = False
    assert sc.state["seen_done"] is False


def test_request_context_is_frozen():
    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view={"x-api-key": "k"},
        raw_body=b"{}",
        session_key="s",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.session_key = "mutated"  # type: ignore[misc]


def test_request_context_headers_isolated_from_source_mutation():
    src = {"x-api-key": "k"}
    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view=src,
        raw_body=b"{}",
        session_key="s",
    )
    src["injected"] = "yes"
    assert "injected" not in ctx.headers_view


def test_request_context_headers_view_is_read_only():
    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view={"a": "b"},
        raw_body=b"{}",
        session_key="s",
    )
    with pytest.raises(TypeError):
        ctx.headers_view["c"] = "d"  # type: ignore[index]
