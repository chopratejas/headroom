from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


class Provider(str, enum.Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


class Flavor(str, enum.Enum):
    MESSAGES = "messages"  # Anthropic /v1/messages
    CHAT = "chat"  # OpenAI /v1/chat/completions
    RESPONSES = "responses"  # OpenAI /v1/responses
    GENERATE = "generate"  # Gemini generateContent
    INVOKE = "invoke"  # Bedrock /model/.../invoke
    RAW_PREDICT = "raw_predict"  # Vertex streamRawPredict


@dataclass(frozen=True)
class RequestContext:
    provider: Provider
    flavor: Flavor
    headers_view: Mapping[str, str]
    raw_body: bytes
    session_key: str
    request_id: str = ""
    # Optional pre-fetched memory context (option (a) async-bridge pattern).
    # The async handler awaits memory_handler.search_and_format_context, then
    # passes the result here before calling engine.on_request.  The engine
    # uses this value directly — no sync/async gymnastics inside the engine.
    # None → memory injection is a no-op for this request.
    prefetched_memory_context: str | None = None

    def __post_init__(self) -> None:
        # Snapshot headers at the engine boundary: copy to isolate from later caller
        # mutation, then wrap read-only. Frozen dataclass → set via object.__setattr__.
        object.__setattr__(self, "headers_view", MappingProxyType(dict(self.headers_view)))


@dataclass
class ResponseTelemetry:
    """Opaque to the host; routed to its own metrics sink. The serving front emits."""

    tokens_in: int = 0
    tokens_out: int = 0
    bytes_saved: int = 0
    compressed: bool = False
    ccr_fired: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestDecision:
    body: bytes
    telemetry: ResponseTelemetry
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamContext:
    session_key: str
    provider: Provider
    flavor: Flavor
    state: dict[str, Any] = field(default_factory=dict)
