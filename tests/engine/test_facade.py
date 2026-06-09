"""Tests for HeadroomEngine facade — Chunk 2.

TDD: these tests were written BEFORE the implementation. Each test targets a
specific contract assertion from the Chunk 2 spec.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from headroom.engine.contract import (
    Flavor,
    Provider,
    RequestContext,
    ResponseTelemetry,
    StreamContext,
)

# ── Fake pipeline (CompressionPipeline Protocol implementor) ──────────────────


@dataclass
class FakeTransformResult:
    messages: list[dict[str, Any]]
    tokens_before: int = 10
    tokens_after: int = 8
    transforms_applied: list[str] = field(default_factory=list)
    markers_inserted: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    diff_artifact: Any = None
    waste_signals: Any = None


class FakePipeline:
    """Fake CompressionPipeline for use in tests — records apply() calls."""

    def __init__(self, result_messages: list[dict[str, Any]] | None = None) -> None:
        self.called_with: list[dict[str, Any]] = []
        self._result_messages = result_messages or [{"role": "assistant", "content": "compressed"}]

    def apply(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> FakeTransformResult:
        self.called_with.append({"messages": messages, "model": model, **kwargs})
        return FakeTransformResult(messages=self._result_messages)


# ── Config-like stub that drives CompressionDecision.decide ──────────────────


class _Config:
    """Minimal config stub: optimize=True means compression is allowed."""

    def __init__(self, optimize: bool = True) -> None:
        self.optimize = optimize


# ── Helpers ───────────────────────────────────────────────────────────────────


_ANTHROPIC_KEY = (Provider.ANTHROPIC, Flavor.MESSAGES)

SAMPLE_MESSAGES = [{"role": "user", "content": "Hello"}]
SAMPLE_MODEL = "claude-3-5-sonnet-20241022"


def _build_body(
    messages: list[dict[str, Any]] | None = None,
    model: str = SAMPLE_MODEL,
) -> bytes:
    return json.dumps(
        {"messages": messages if messages is not None else SAMPLE_MESSAGES, "model": model}
    ).encode()


def _make_ctx(
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    provider: Provider = Provider.ANTHROPIC,
    flavor: Flavor = Flavor.MESSAGES,
) -> RequestContext:
    return RequestContext(
        provider=provider,
        flavor=flavor,
        headers_view=headers or {"x-api-key": "sk-ant-test"},
        raw_body=body if body is not None else _build_body(),
        session_key="test-session",
    )


def _make_engine(
    fake_pipeline: FakePipeline | None = None,
    optimize: bool = True,
) -> Any:
    from headroom.engine.facade import HeadroomEngine

    pipeline = fake_pipeline or FakePipeline()
    return HeadroomEngine(
        pipelines={_ANTHROPIC_KEY: pipeline},
        config=_Config(optimize=optimize),
        usage_reporter=None,
        salt=b"test-salt",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestPassthrough:
    """When should_compress is False, body must be BYTE-IDENTICAL to input."""

    def test_bypass_header_returns_raw_body(self) -> None:
        """x-headroom-bypass: true → passthrough, body is same object."""
        fake = FakePipeline()
        engine = _make_engine(fake)
        raw = _build_body()
        ctx = _make_ctx(
            headers={"x-api-key": "sk-ant-test", "x-headroom-bypass": "true"},
            body=raw,
        )
        decision = engine.on_request(ctx)

        assert decision.body is raw, "passthrough body must be ctx.raw_body itself"
        assert decision.telemetry.compressed is False
        assert fake.called_with == [], "pipeline must not be called on passthrough"

    def test_compression_disabled_returns_raw_body(self) -> None:
        """config.optimize=False → passthrough even with valid messages."""
        fake = FakePipeline()
        engine = _make_engine(fake, optimize=False)
        raw = _build_body()
        ctx = _make_ctx(body=raw)
        decision = engine.on_request(ctx)

        assert decision.body is raw
        assert decision.telemetry.compressed is False
        assert fake.called_with == []

    def test_empty_messages_returns_raw_body(self) -> None:
        """Empty messages list → passthrough (no_messages reason)."""
        fake = FakePipeline()
        engine = _make_engine(fake)
        raw = _build_body(messages=[])
        ctx = _make_ctx(body=raw)
        decision = engine.on_request(ctx)

        assert decision.body is raw
        assert decision.telemetry.compressed is False
        assert fake.called_with == []


class TestCompress:
    """When should_compress is True, facade calls pipeline and rewrites messages."""

    def test_compress_calls_pipeline(self) -> None:
        """Compression path calls pipeline.apply and body contains result.messages."""
        compressed_msg = [{"role": "assistant", "content": "compressed-output"}]
        fake = FakePipeline(result_messages=compressed_msg)
        engine = _make_engine(fake)
        ctx = _make_ctx()

        decision = engine.on_request(ctx)

        assert fake.called_with, "pipeline.apply must be called"
        result_body = json.loads(decision.body)
        assert result_body["messages"] == compressed_msg

    def test_compress_telemetry_compressed_true(self) -> None:
        """Compression path sets telemetry.compressed=True."""
        fake = FakePipeline()
        engine = _make_engine(fake)
        decision = engine.on_request(_make_ctx())

        assert decision.telemetry.compressed is True

    def test_compress_bytes_saved(self) -> None:
        """bytes_saved is non-negative; when output is smaller, equals the reduction."""
        # Use a very large input so the serialized output is definitely smaller.
        big_messages = [{"role": "user", "content": "x" * 500}]
        compressed_messages = [{"role": "assistant", "content": "c"}]
        fake = FakePipeline(result_messages=compressed_messages)
        engine = _make_engine(fake)
        raw = _build_body(messages=big_messages)
        ctx = _make_ctx(body=raw)
        decision = engine.on_request(ctx)

        assert decision.telemetry.bytes_saved >= 0
        # When the output is genuinely smaller, bytes_saved equals the reduction.
        expected = len(raw) - len(decision.body)
        assert decision.telemetry.bytes_saved == max(0, expected)

    def test_compress_passes_compression_policy(self) -> None:
        """pipeline.apply receives compression_policy kwarg."""
        fake = FakePipeline()
        engine = _make_engine(fake)
        engine.on_request(_make_ctx())

        assert fake.called_with, "pipeline.apply must be called"
        assert "compression_policy" in fake.called_with[0]

    def test_model_passthrough_to_pipeline(self) -> None:
        """The model from the request body is forwarded to pipeline.apply."""
        fake = FakePipeline()
        engine = _make_engine(fake)
        engine.on_request(_make_ctx())

        assert fake.called_with[0]["model"] == SAMPLE_MODEL


class TestAuthModeInfluencesPolicy:
    """Auth mode classification affects the CompressionPolicy passed to pipeline."""

    def test_subscription_ua_gets_subscription_policy(self) -> None:
        """A Subscription User-Agent receives a policy with live_zone_only=True."""
        fake = FakePipeline()
        engine = _make_engine(fake)
        ctx = _make_ctx(
            headers={
                "x-api-key": "sk-ant-test",
                "user-agent": "claude-cli/1.0",
            }
        )
        engine.on_request(ctx)

        assert fake.called_with, "pipeline.apply must be called"
        policy = fake.called_with[0]["compression_policy"]
        assert policy.live_zone_only is True, "subscription UA → live_zone_only policy"
        assert policy.toin_read_only is True

    def test_payg_key_gets_payg_policy(self) -> None:
        """A plain API key request receives PAYG policy (live_zone_only=False)."""
        fake = FakePipeline()
        engine = _make_engine(fake)
        ctx = _make_ctx(headers={"x-api-key": "sk-ant-test"})
        engine.on_request(ctx)

        assert fake.called_with
        policy = fake.called_with[0]["compression_policy"]
        assert policy.live_zone_only is False
        assert policy.toin_read_only is False


class TestUnregisteredProvider:
    """Unregistered (provider, flavor) combos must raise, not silently fall through."""

    def test_raises_on_unregistered_provider(self) -> None:
        """OpenAI CHAT is not registered → on_request must raise."""
        from headroom.engine.facade import HeadroomEngine

        engine = HeadroomEngine(
            pipelines={_ANTHROPIC_KEY: FakePipeline()},
            config=_Config(),
            usage_reporter=None,
            salt=b"test-salt",
        )
        openai_body = json.dumps({"messages": SAMPLE_MESSAGES, "model": "gpt-4o"}).encode()
        ctx = _make_ctx(
            provider=Provider.OPENAI,
            flavor=Flavor.CHAT,
            body=openai_body,
        )
        with pytest.raises(KeyError):
            engine.on_request(ctx)


class TestResponseHooks:
    """Response hooks return their input unchanged; on_response_end is safe."""

    def test_on_response_returns_unchanged(self) -> None:
        engine = _make_engine()
        ctx = _make_ctx()
        raw = b"raw-response-bytes"
        assert engine.on_response(ctx, raw) is raw

    def test_on_response_chunk_returns_unchanged(self) -> None:
        engine = _make_engine()
        sc = StreamContext(
            session_key="s",
            provider=Provider.ANTHROPIC,
            flavor=Flavor.MESSAGES,
        )
        chunk = b"chunk-data"
        assert engine.on_response_chunk(sc, chunk) is chunk

    def test_on_response_end_returns_telemetry(self) -> None:
        engine = _make_engine()
        sc = StreamContext(
            session_key="s",
            provider=Provider.ANTHROPIC,
            flavor=Flavor.MESSAGES,
        )
        telem = engine.on_response_end(sc, outcome=None)
        assert isinstance(telem, ResponseTelemetry)

    def test_on_response_end_safe_on_abort(self) -> None:
        """on_response_end must not raise when called with an abort/error outcome."""
        engine = _make_engine()
        sc = StreamContext(
            session_key="s",
            provider=Provider.ANTHROPIC,
            flavor=Flavor.MESSAGES,
        )
        telem = engine.on_response_end(sc, outcome=Exception("connection reset"))
        assert isinstance(telem, ResponseTelemetry)
