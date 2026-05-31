"""Broad differential parity test — legacy Anthropic handler vs HeadroomEngine.

For each generated request variation this test:
  1. Drives the REAL legacy proxy handler (via _make_app_and_transport) and
     captures exact outbound bytes (legacy_bytes).
  2. Builds a REAL HeadroomEngine (same pipeline/provider, same seeded state)
     and calls engine.on_request — capturing engine_bytes.
  3. Asserts byte-equality.

If any pair diverges the test collects ALL divergences, reports each with the
input spec, byte-level diff, and a cause hypothesis. No silent xfail unless a
case is genuinely nondeterministic AND that is demonstrated here.

Knobs varied across the matrix (50+ pruned combinations):
  - mode: token / cache / non-cache (optimize=False)
  - auth: PAYG / OAuth / Subscription
  - bypass: none / x-headroom-bypass / x-headroom-mode=passthrough
  - tools: absent / sorted / unsorted / nested JSON-schema / empty-list
  - content: tiny / large-tool_result (ContentRouter→LogCompressor) / large user text
  - system: absent / string / block-list
  - frozen_message_count: 0 / >0; single-turn / multi-turn (seeded prev msgs)
  - unicode content; numeric-precision floats

Determinism constraints (nondeterministic features kept OFF):
  - ccr_proactive_expansion=False (ML-nondeterministic)
  - image_optimize=False (codec nondeterminism)
  - memory disabled
  - LLMLingua / IntelligentContext paths excluded
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("fastapi")

from tests.parity.engine_request_recorder import (  # noqa: E402
    GoldenCaseSpec,
    _FixedTracker,  # noqa: PLC2701
    _FreshCompressionCache,  # noqa: PLC2701
    _make_app_and_transport,  # noqa: PLC2701
)

# ---------------------------------------------------------------------------
# Engine factory (mirrors test_engine_request_parity._build_engine_for_fixture)
# ---------------------------------------------------------------------------


@dataclass
class _ControlledStore:
    tracker: _FixedTracker
    session_id: str = "diff-parity"
    fresh_caches: dict[str, _FreshCompressionCache] = field(default_factory=dict)

    def compute_session_id(self, request: Any, model: str, messages: Any) -> str:
        return self.session_id

    def get_or_create(self, session_id: str, provider: str) -> _FixedTracker:
        return self.tracker

    def get_fresh_cache(self, session_id: str) -> _FreshCompressionCache:
        if session_id not in self.fresh_caches:
            self.fresh_caches[session_id] = _FreshCompressionCache()
        return self.fresh_caches[session_id]


def _build_engine_for_spec(spec: GoldenCaseSpec) -> Any:
    from headroom.engine.contract import Flavor, Provider
    from headroom.engine.facade import AnthropicComponents, HeadroomEngine
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config_kwargs: dict[str, Any] = {
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "image_optimize": False,
    }
    config_kwargs.update(spec.proxy_config)
    config = ProxyConfig(**config_kwargs)

    proxy = HeadroomProxy(config)

    tracker = _FixedTracker(frozen_count=spec.frozen_count)
    if spec.prev_original_messages:
        tracker._last_original_messages = list(spec.prev_original_messages)
    if spec.prev_forwarded_messages:
        tracker._last_forwarded_messages = list(spec.prev_forwarded_messages)

    controlled_store = _ControlledStore(tracker=tracker, session_id=f"diff-{spec.name}")

    ac = AnthropicComponents(
        pipeline=proxy.anthropic_pipeline,
        provider=proxy.anthropic_provider,
        session_tracker_store=controlled_store,
        get_compression_cache=controlled_store.get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )

    engine = HeadroomEngine(
        pipelines={(Provider.ANTHROPIC, Flavor.MESSAGES): proxy.anthropic_pipeline},
        config=proxy.config,
        usage_reporter=None,
        salt=b"diff-parity-salt",
        anthropic_components=ac,
    )
    return engine


def _run_spec(spec: GoldenCaseSpec) -> tuple[bytes, bytes]:
    """Return (legacy_bytes, engine_bytes) for spec."""
    from headroom.engine.contract import Flavor, Provider, RequestContext

    body_bytes = json.dumps(spec.body, separators=(",", ":"), ensure_ascii=False).encode()

    # --- legacy path ---
    client, transport = _make_app_and_transport(spec)
    if spec.streaming:
        with client.stream(
            "POST",
            "/v1/messages",
            headers=spec.inbound_headers,
            content=body_bytes,
        ) as resp:
            for _ in resp.iter_bytes():
                pass
        legacy_bytes = transport.captured_body  # type: ignore[union-attr]
    else:
        client.post(
            "/v1/messages",
            headers=spec.inbound_headers,
            content=body_bytes,
        )
        legacy_bytes = transport.captured_body  # type: ignore[union-attr]

    if legacy_bytes is None:
        raise RuntimeError(f"Differential '{spec.name}': legacy transport captured no body.")

    # --- engine path ---
    engine = _build_engine_for_spec(spec)
    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view=spec.inbound_headers,
        raw_body=body_bytes,
        session_key=f"diff-{spec.name}",
        request_id="",
    )
    decision = engine.on_request(ctx)
    engine_bytes = decision.body

    return legacy_bytes, engine_bytes


# ---------------------------------------------------------------------------
# Shared content constants
# ---------------------------------------------------------------------------

_PAYG_HEADERS = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

_OAUTH_HEADERS = {
    "authorization": "Bearer sk-ant-oat-abc123def456ghi789jkl0",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

_SUBSCRIPTION_HEADERS = {
    "authorization": "Bearer sk-ant-oat-sub-xyz789abc",
    "x-headroom-client": "claude-code",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

_BYPASS_HEADERS = {
    **_PAYG_HEADERS,
    "x-headroom-bypass": "true",
}

_MODE_PASSTHROUGH_HEADERS = {
    **_PAYG_HEADERS,
    "x-headroom-mode": "passthrough",
}

# ~1 KB deterministic log text (same as engine_request_recorder.py)
_LARGE_LOG = (
    "INFO [2026-05-29T10:00:00Z] Starting batch job batch-9821\n"
    "INFO [2026-05-29T10:00:01Z] Loading model weights from /data/models/v7\n"
    "INFO [2026-05-29T10:00:05Z] Model loaded (3.2 GB) in 4.1s\n"
    "WARN [2026-05-29T10:00:05Z] GPU memory fragmented; defragmenting\n"
    "INFO [2026-05-29T10:00:07Z] Processing shard 1/8 (125k records)\n"
    "INFO [2026-05-29T10:00:15Z] Shard 1 done: 124,982 processed, 18 skipped\n"
    "INFO [2026-05-29T10:00:15Z] Processing shard 2/8 (125k records)\n"
    "INFO [2026-05-29T10:00:23Z] Shard 2 done: 124,999 processed, 1 skipped\n"
    "ERROR [2026-05-29T10:00:24Z] Shard 3: connection reset by peer during write\n"
    "WARN [2026-05-29T10:00:24Z] Retrying shard 3 (attempt 1/3)\n"
    "INFO [2026-05-29T10:00:28Z] Shard 3 retry succeeded\n"
    "INFO [2026-05-29T10:00:28Z] Processing shard 4/8 (125k records)\n"
    "INFO [2026-05-29T10:00:37Z] Shard 4 done: 125,000 processed, 0 skipped\n"
    "INFO [2026-05-29T10:00:37Z] Processing shard 5/8 (125k records)\n"
    "INFO [2026-05-29T10:00:46Z] Shard 5 done: 125,000 processed, 0 skipped\n"
    "INFO [2026-05-29T10:00:46Z] Processing shard 6/8 (125k records)\n"
    "INFO [2026-05-29T10:00:55Z] Shard 6 done: 124,950 processed, 50 skipped\n"
    "INFO [2026-05-29T10:00:55Z] Processing shard 7/8 (125k records)\n"
    "INFO [2026-05-29T10:01:05Z] Shard 7 done: 125,000 processed, 0 skipped\n"
    "INFO [2026-05-29T10:01:05Z] Processing shard 8/8 (125k records)\n"
    "INFO [2026-05-29T10:01:14Z] Shard 8 done: 125,000 processed, 0 skipped\n"
    "INFO [2026-05-29T10:01:14Z] All shards complete. Total: 999,931 processed\n"
    "INFO [2026-05-29T10:01:14Z] Writing results to s3://data-prod/batch-9821/\n"
    "INFO [2026-05-29T10:01:18Z] Upload complete (42 MB in 4s)\n"
    "INFO [2026-05-29T10:01:18Z] Job batch-9821 finished OK\n"
)

# ~600 chars of large user text
_LARGE_USER_TEXT = (
    "Please analyze the following system architecture document carefully. "
    "The microservices architecture consists of an API gateway, authentication service, "
    "user management service, product catalog service, order processing service, "
    "payment gateway integration, inventory management service, notification service, "
    "analytics aggregation service, and a reporting dashboard. Each service communicates "
    "via REST APIs with JWT authentication. The data stores include PostgreSQL for "
    "transactional data, Redis for caching session state and rate-limit counters, "
    "Elasticsearch for full-text search across product catalog and order history, "
    "S3 for binary asset storage, and a Kafka cluster for async event streaming "
    "between services. Identify any single points of failure and suggest improvements."
)

_TOOLS_SORTED = [
    {
        "name": "alpha_tool",
        "description": "a",
        "input_schema": {"type": "object", "properties": {}},
    },
    {"name": "beta_tool", "description": "b", "input_schema": {"type": "object", "properties": {}}},
    {
        "name": "gamma_tool",
        "description": "g",
        "input_schema": {"type": "object", "properties": {}},
    },
]

_TOOLS_UNSORTED = [
    {"name": "zeta_tool", "description": "z", "input_schema": {"type": "object", "properties": {}}},
    {
        "name": "alpha_tool",
        "description": "a",
        "input_schema": {"type": "object", "properties": {}},
    },
    {"name": "mu_tool", "description": "m", "input_schema": {"type": "object", "properties": {}}},
]

_TOOLS_NESTED_SCHEMA = [
    {
        "name": "search_files",
        "description": "Search files by pattern",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "options": {
                    "type": "object",
                    "properties": {
                        "case_sensitive": {"type": "boolean"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
                    },
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "encoding": {"type": "string"}},
            "required": ["path"],
        },
    },
]

_SYSTEM_STRING = "You are a helpful assistant. Be concise."

_SYSTEM_BLOCK_LIST = [
    {"type": "text", "text": "You are a helpful assistant."},
    {"type": "text", "text": "Always respond in English. Be concise and accurate."},
]

_DEFAULT_PC: dict[str, Any] = {
    "cache_enabled": False,
    "rate_limit_enabled": False,
    "cost_tracking_enabled": False,
    "log_requests": False,
    "ccr_inject_tool": False,
    "ccr_handle_responses": False,
    "ccr_context_tracking": False,
    "image_optimize": False,
}


def _pc(**extra: Any) -> dict[str, Any]:
    return {**_DEFAULT_PC, **extra}


# ---------------------------------------------------------------------------
# Matrix — 54 cases spanning the deterministic knob space
# ---------------------------------------------------------------------------

_MATRIX: list[GoldenCaseSpec] = [
    # =========================================================================
    # GROUP A — passthrough (optimize=False): all auth modes, no/bypass headers
    # =========================================================================
    # A-01: PAYG, optimize=False, tiny body
    GoldenCaseSpec(
        name="diff_A01_payg_no_opt_tiny",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        },
        proxy_config=_pc(optimize=False),
        notes="PAYG passthrough; tiny body",
    ),
    # A-02: OAuth, optimize=False
    GoldenCaseSpec(
        name="diff_A02_oauth_no_opt",
        inbound_headers=_OAUTH_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Summarize the document."}],
        },
        proxy_config=_pc(optimize=False),
        notes="OAuth passthrough",
    ),
    # A-03: Subscription, optimize=False
    GoldenCaseSpec(
        name="diff_A03_subscription_no_opt",
        inbound_headers=_SUBSCRIPTION_HEADERS,
        body={
            "model": "claude-opus-4-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Write a poem about rivers."}],
        },
        proxy_config=_pc(optimize=False),
        notes="Subscription passthrough",
    ),
    # A-04: PAYG, x-headroom-bypass header
    GoldenCaseSpec(
        name="diff_A04_payg_bypass_header",
        inbound_headers=_BYPASS_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "bypass me"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="x-headroom-bypass=true overrides optimize",
    ),
    # A-05: PAYG, x-headroom-mode=passthrough
    GoldenCaseSpec(
        name="diff_A05_payg_passthrough_mode_header",
        inbound_headers=_MODE_PASSTHROUGH_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "passthrough via mode header"}],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="x-headroom-mode=passthrough overrides optimize",
    ),
    # A-06: OAuth, bypass header
    GoldenCaseSpec(
        name="diff_A06_oauth_bypass_header",
        inbound_headers={**_OAUTH_HEADERS, "x-headroom-bypass": "true"},
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "oauth bypass"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="OAuth + x-headroom-bypass",
    ),
    # =========================================================================
    # GROUP B — numeric precision, unicode, float fields
    # =========================================================================
    # B-01: Float fields (temperature + top_p)
    GoldenCaseSpec(
        name="diff_B01_float_fields_passthrough",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "temperature": 1.0,
            "top_p": 0.95,
            "messages": [{"role": "user", "content": "precision test"}],
        },
        proxy_config=_pc(optimize=False),
        notes="Floats in body; passthrough path",
    ),
    # B-02: Float fields with token-mode compression
    GoldenCaseSpec(
        name="diff_B02_float_fields_token_mode",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "messages": [{"role": "user", "content": "token mode with floats"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Floats + token mode",
    ),
    # B-03: Unicode content, passthrough
    GoldenCaseSpec(
        name="diff_B03_unicode_passthrough",
        inbound_headers=_BYPASS_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "Hello \U0001f525 — 世界 — emoji 🚀 — Ünïcödé"}
            ],
        },
        proxy_config=_pc(optimize=False),
        notes="Unicode bypass; no \\uXXXX escaping",
    ),
    # B-04: Unicode content, cache mode
    GoldenCaseSpec(
        name="diff_B04_unicode_cache_mode",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "CJK: 日本語テスト — αβγδ — ñ ü ö"}],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="Unicode in cache mode; CacheAligner must not mangle multibyte",
    ),
    # B-05: Large integer + negative int
    GoldenCaseSpec(
        name="diff_B05_int_precision",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "int precision"}],
        },
        proxy_config=_pc(optimize=False),
        notes="Large max_tokens integer",
    ),
    # =========================================================================
    # GROUP C — tool variations (sorted / unsorted / nested / empty / absent)
    # =========================================================================
    # C-01: Tools absent, token mode
    GoldenCaseSpec(
        name="diff_C01_no_tools_token_mode",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "no tools"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="No tools field, token mode",
    ),
    # C-02: Tools pre-sorted, cache mode
    GoldenCaseSpec(
        name="diff_C02_tools_sorted_cache_mode",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": _TOOLS_SORTED,
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="Pre-sorted tools + cache mode",
    ),
    # C-03: Tools unsorted, cache mode (handler must sort)
    GoldenCaseSpec(
        name="diff_C03_tools_unsorted_cache_mode",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": _TOOLS_UNSORTED,
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="Unsorted tools, cache mode; handler/engine must both sort",
    ),
    # C-04: Tools unsorted, token mode
    GoldenCaseSpec(
        name="diff_C04_tools_unsorted_token_mode",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": _TOOLS_UNSORTED,
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Unsorted tools, token mode",
    ),
    # C-05: Tools unsorted, optimize=False (no-compress path still sorts)
    GoldenCaseSpec(
        name="diff_C05_tools_unsorted_no_opt",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "tool no opt"}],
            "tools": _TOOLS_UNSORTED,
        },
        proxy_config=_pc(optimize=False),
        notes="Unsorted tools, optimize=False — pre-send sort still applies",
    ),
    # C-06: Nested-schema tools, cache mode
    GoldenCaseSpec(
        name="diff_C06_nested_schema_tools_cache",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "search files"}],
            "tools": _TOOLS_NESTED_SCHEMA,
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="Nested JSON-schema tools, cache mode",
    ),
    # C-07: Nested-schema tools, token mode
    GoldenCaseSpec(
        name="diff_C07_nested_schema_tools_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "search files"}],
            "tools": _TOOLS_NESTED_SCHEMA,
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Nested JSON-schema tools, token mode",
    ),
    # C-08: Empty tools list, cache mode
    GoldenCaseSpec(
        name="diff_C08_empty_tools_cache",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "empty tools list"}],
            "tools": [],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="tools=[] empty list",
    ),
    # C-09: Many tools (trigger sort correctness), token mode
    GoldenCaseSpec(
        name="diff_C09_many_tools_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "many tools"}],
            "tools": [
                {"name": n, "description": n, "input_schema": {"type": "object"}}
                for n in [
                    "zeta",
                    "omega",
                    "delta",
                    "alpha",
                    "beta",
                    "gamma",
                    "eta",
                    "theta",
                    "iota",
                    "kappa",
                ]
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="10 tools in reverse alpha order; both paths must sort",
    ),
    # =========================================================================
    # GROUP D — system prompt variations
    # =========================================================================
    # D-01: System string, token mode
    GoldenCaseSpec(
        name="diff_D01_system_string_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "system": _SYSTEM_STRING,
            "messages": [{"role": "user", "content": "with system"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="system as string, token mode",
    ),
    # D-02: System block-list, cache mode
    GoldenCaseSpec(
        name="diff_D02_system_block_list_cache",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "system": _SYSTEM_BLOCK_LIST,
            "messages": [{"role": "user", "content": "with system block-list"}],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="system as block-list, cache mode",
    ),
    # D-03: System absent, cache mode
    GoldenCaseSpec(
        name="diff_D03_no_system_cache",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "no system field"}],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="No system field, cache mode",
    ),
    # D-04: System string + tools, cache mode
    GoldenCaseSpec(
        name="diff_D04_system_plus_tools_cache",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "system": _SYSTEM_STRING,
            "messages": [{"role": "user", "content": "system + tools"}],
            "tools": _TOOLS_SORTED,
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="system string + sorted tools, cache mode",
    ),
    # D-05: System block-list + tools, token mode
    GoldenCaseSpec(
        name="diff_D05_system_block_list_tools_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "system": _SYSTEM_BLOCK_LIST,
            "messages": [{"role": "user", "content": "system block + tools"}],
            "tools": _TOOLS_UNSORTED,
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="system block-list + unsorted tools, token mode",
    ),
    # =========================================================================
    # GROUP E — frozen_message_count / multi-turn
    # =========================================================================
    # E-01: frozen=0, token mode, single turn
    GoldenCaseSpec(
        name="diff_E01_frozen0_token_single",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "single turn frozen=0"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        frozen_count=0,
        notes="frozen=0, token, single-turn",
    ),
    # E-02: frozen=2, token mode
    GoldenCaseSpec(
        name="diff_E02_frozen2_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "frozen turn 1"},
                {"role": "assistant", "content": "answer 1"},
                {"role": "user", "content": "current question"},
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        frozen_count=2,
        notes="frozen=2, token mode; prefix excluded from compression",
    ),
    # E-03: frozen=4, cache mode
    GoldenCaseSpec(
        name="diff_E03_frozen4_cache",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "frozen t1"},
                {"role": "assistant", "content": "frozen a1"},
                {"role": "user", "content": "frozen t2"},
                {"role": "assistant", "content": "frozen a2"},
                {"role": "user", "content": "live question"},
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        frozen_count=4,
        notes="frozen=4, cache mode; only last message live",
    ),
    # E-04: frozen=0, cache mode, multi-turn
    GoldenCaseSpec(
        name="diff_E04_frozen0_cache_multi",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": "answer 1"},
                {"role": "user", "content": "turn 2"},
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        frozen_count=0,
        notes="frozen=0, cache mode, 3-turn; CacheAligner processes all",
    ),
    # E-05: Seeded prev messages (delta path), cache mode
    GoldenCaseSpec(
        name="diff_E05_delta_prev_msgs_cache",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "shared prefix turn"},
                {"role": "assistant", "content": "shared answer"},
                {"role": "user", "content": "new turn 2"},
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        frozen_count=0,
        prev_original_messages=[
            {"role": "user", "content": "shared prefix turn"},
            {"role": "assistant", "content": "shared answer"},
        ],
        prev_forwarded_messages=[
            {"role": "user", "content": "shared prefix turn"},
            {"role": "assistant", "content": "shared answer"},
        ],
        notes="Delta path: prev_original_messages seeded; handler reuses forwarded prefix",
    ),
    # E-06: Seeded prev, token mode
    GoldenCaseSpec(
        name="diff_E06_delta_prev_msgs_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "prev turn"},
                {"role": "assistant", "content": "prev answer"},
                {"role": "user", "content": "current token turn"},
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        frozen_count=0,
        prev_original_messages=[
            {"role": "user", "content": "prev turn"},
            {"role": "assistant", "content": "prev answer"},
        ],
        prev_forwarded_messages=[
            {"role": "user", "content": "prev turn"},
            {"role": "assistant", "content": "prev answer"},
        ],
        notes="Delta path token mode with seeded prev messages",
    ),
    # E-07: frozen=2, cache mode, seeded prev
    GoldenCaseSpec(
        name="diff_E07_frozen2_cache_seeded_prev",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "frozen prefix A"},
                {"role": "assistant", "content": "frozen reply A"},
                {"role": "user", "content": "new live question"},
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        frozen_count=2,
        prev_original_messages=[
            {"role": "user", "content": "frozen prefix A"},
            {"role": "assistant", "content": "frozen reply A"},
        ],
        prev_forwarded_messages=[
            {"role": "user", "content": "frozen prefix A"},
            {"role": "assistant", "content": "frozen reply A"},
        ],
        notes="frozen=2 + seeded prev in cache mode",
    ),
    # =========================================================================
    # GROUP F — content size / ContentRouter triggers
    # =========================================================================
    # F-01: Large tool_result, token mode (ContentRouter→LogCompressor)
    GoldenCaseSpec(
        name="diff_F01_large_tool_result_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "run_job",
                            "input": {"job": "batch-001"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": _LARGE_LOG}
                    ],
                },
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Large tool_result log text; ContentRouter→LogCompressor path; "
        "deterministic because BM25 short-circuits on empty query",
    ),
    # F-02: Large tool_result, cache mode
    GoldenCaseSpec(
        name="diff_F02_large_tool_result_cache",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "run_job",
                            "input": {"job": "batch-002"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_2", "content": _LARGE_LOG}
                    ],
                },
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="Large tool_result in cache mode",
    ),
    # F-03: Large user text, token mode
    GoldenCaseSpec(
        name="diff_F03_large_user_text_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": _LARGE_USER_TEXT}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Large user text in token mode",
    ),
    # F-04: Large user text, compress_user_messages=True, token mode
    GoldenCaseSpec(
        name="diff_F04_large_user_compress_user_msgs",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": _LARGE_USER_TEXT}],
        },
        proxy_config=_pc(optimize=True, mode="token", compress_user_messages=True),
        notes="Large user text with compress_user_messages=True",
    ),
    # F-05: Large tool_result, frozen=2, token mode
    GoldenCaseSpec(
        name="diff_F05_large_tool_result_frozen2",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "frozen prefix question"},
                {"role": "assistant", "content": "frozen prefix answer"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_3",
                            "name": "run_batch",
                            "input": {"batch": "x"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_3", "content": _LARGE_LOG}
                    ],
                },
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        frozen_count=2,
        notes="Large tool_result with frozen=2; tool_result is in live zone",
    ),
    # F-06: Empty messages, token mode
    GoldenCaseSpec(
        name="diff_F06_empty_messages_token",
        inbound_headers=_PAYG_HEADERS,
        body={"model": "claude-sonnet-4-6", "max_tokens": 64, "messages": []},
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Empty messages array; passthrough without compression",
    ),
    # F-07: Large tool_result with tools in body, token mode
    GoldenCaseSpec(
        name="diff_F07_large_tool_result_with_tools",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "tools": _TOOLS_UNSORTED,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_4",
                            "name": "run_job",
                            "input": {"job": "batch-003"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_4", "content": _LARGE_LOG}
                    ],
                },
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Large tool_result + unsorted tools; both ContentRouter + sort must fire",
    ),
    # =========================================================================
    # GROUP G — streaming
    # =========================================================================
    # G-01: Streaming, passthrough
    GoldenCaseSpec(
        name="diff_G01_streaming_passthrough",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "stream passthrough"}],
        },
        proxy_config=_pc(optimize=False),
        streaming=True,
        notes="Streaming, optimize=False",
    ),
    # G-02: Streaming, cache mode
    GoldenCaseSpec(
        name="diff_G02_streaming_cache",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "stream": True,
            "messages": [
                {"role": "user", "content": "prev"},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "now stream"},
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        streaming=True,
        notes="Streaming + cache mode",
    ),
    # G-03: Streaming, token mode
    GoldenCaseSpec(
        name="diff_G03_streaming_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "stream token mode"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        streaming=True,
        notes="Streaming + token mode",
    ),
    # G-04: Streaming, bypass
    GoldenCaseSpec(
        name="diff_G04_streaming_bypass",
        inbound_headers=_BYPASS_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "stream bypass"}],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        streaming=True,
        notes="Streaming + bypass header",
    ),
    # G-05: Streaming, cache mode with tools
    GoldenCaseSpec(
        name="diff_G05_streaming_cache_tools",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "stream": True,
            "tools": _TOOLS_UNSORTED,
            "messages": [{"role": "user", "content": "stream tool call"}],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        streaming=True,
        notes="Streaming + cache mode + unsorted tools",
    ),
    # =========================================================================
    # GROUP H — auth × mode cross-product (non-trivial combinations)
    # =========================================================================
    # H-01: OAuth, token mode (OAuth → usually passthrough-prefer, but optimize=True)
    GoldenCaseSpec(
        name="diff_H01_oauth_token_mode",
        inbound_headers=_OAUTH_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "oauth token mode"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="OAuth + optimize=True + token mode",
    ),
    # H-02: OAuth, cache mode
    GoldenCaseSpec(
        name="diff_H02_oauth_cache_mode",
        inbound_headers=_OAUTH_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "oauth t1"},
                {"role": "assistant", "content": "oauth a1"},
                {"role": "user", "content": "oauth t2 cache"},
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="OAuth + optimize=True + cache mode multi-turn",
    ),
    # H-03: Subscription, token mode
    GoldenCaseSpec(
        name="diff_H03_subscription_token_mode",
        inbound_headers=_SUBSCRIPTION_HEADERS,
        body={
            "model": "claude-opus-4-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "subscription token mode"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Subscription auth + optimize=True + token mode",
    ),
    # H-04: Subscription, cache mode, frozen=2
    GoldenCaseSpec(
        name="diff_H04_subscription_cache_frozen2",
        inbound_headers=_SUBSCRIPTION_HEADERS,
        body={
            "model": "claude-opus-4-5",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "frozen sub 1"},
                {"role": "assistant", "content": "frozen sub answer 1"},
                {"role": "user", "content": "live sub question"},
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        frozen_count=2,
        notes="Subscription + cache mode + frozen=2",
    ),
    # H-05: PAYG, cache mode, with system + tools + large tool_result
    GoldenCaseSpec(
        name="diff_H05_payg_cache_system_tools_large_result",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 256,
            "system": _SYSTEM_STRING,
            "tools": _TOOLS_UNSORTED,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_5",
                            "name": "run_batch",
                            "input": {"job": "x"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_5", "content": _LARGE_LOG}
                    ],
                },
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        notes="Full combo: PAYG + system + unsorted tools + large tool_result in cache mode",
    ),
    # H-06: PAYG, token mode, with system + tools + large tool_result + frozen=2
    GoldenCaseSpec(
        name="diff_H06_payg_token_full_combo_frozen2",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 256,
            "system": _SYSTEM_STRING,
            "tools": _TOOLS_UNSORTED,
            "messages": [
                {"role": "user", "content": "frozen q"},
                {"role": "assistant", "content": "frozen a"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_6",
                            "name": "run_batch",
                            "input": {"job": "y"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_6", "content": _LARGE_LOG}
                    ],
                },
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        frozen_count=2,
        notes="Full combo token mode: system + unsorted tools + large tool_result + frozen=2",
    ),
    # =========================================================================
    # GROUP I — edge cases
    # =========================================================================
    # I-01: Single assistant turn in messages (unusual but valid)
    GoldenCaseSpec(
        name="diff_I01_single_assistant_msg",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "assistant", "content": "I am the assistant"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Single assistant message (odd but valid shape)",
    ),
    # I-02: tool_use + tool_result pair, no large content (tiny), token mode
    GoldenCaseSpec(
        name="diff_I02_tiny_tool_result_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_7", "name": "get_time", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_7",
                            "content": "2026-05-30T12:00:00Z",
                        }
                    ],
                },
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Tiny tool_result (no compression trigger); token mode",
    ),
    # I-03: Multiple tool_use calls in one assistant turn
    GoldenCaseSpec(
        name="diff_I03_multi_tool_use_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "tools": _TOOLS_NESTED_SCHEMA,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_8a",
                            "name": "search_files",
                            "input": {"pattern": "*.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_8b",
                            "name": "read_file",
                            "input": {"path": "/foo.py"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_8a",
                            "content": "file1.py\nfile2.py\nfile3.py",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_8b",
                            "content": "def hello(): pass",
                        },
                    ],
                },
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="Multiple parallel tool_use calls; both tool_results in one user turn",
    ),
    # I-04: null content in a message (test graceful handling)
    GoldenCaseSpec(
        name="diff_I04_null_content_passthrough",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": None}],
        },
        proxy_config=_pc(optimize=False),
        notes="null content field; passthrough path",
    ),
    # I-05: Very deep conversation (10 turns), cache mode, frozen=6
    GoldenCaseSpec(
        name="diff_I05_deep_convo_cache_frozen6",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": f"turn {i}"}
                if i % 2 == 0
                else {"role": "assistant", "content": f"answer {i}"}
                for i in range(10)
            ],
        },
        proxy_config=_pc(optimize=True, mode="cache"),
        frozen_count=6,
        notes="10-turn conversation, frozen=6, cache mode",
    ),
    # I-06: tool_result with list content (array format), token mode
    GoldenCaseSpec(
        name="diff_I06_tool_result_list_content",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_9",
                            "name": "search",
                            "input": {"q": "python"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_9",
                            "content": [
                                {"type": "text", "text": "result 1"},
                                {"type": "text", "text": "result 2"},
                            ],
                        }
                    ],
                },
            ],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="tool_result with content as list of blocks (array format)",
    ),
    # I-07: metadata field in body, passthrough
    GoldenCaseSpec(
        name="diff_I07_metadata_field_passthrough",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "metadata": {"user_id": "user-42"},
            "messages": [{"role": "user", "content": "metadata test"}],
        },
        proxy_config=_pc(optimize=False),
        notes="metadata field in body; passthrough preserves it",
    ),
    # I-08: metadata field in body, token mode
    GoldenCaseSpec(
        name="diff_I08_metadata_field_token",
        inbound_headers=_PAYG_HEADERS,
        body={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "metadata": {"user_id": "user-99"},
            "messages": [{"role": "user", "content": "metadata token mode"}],
        },
        proxy_config=_pc(optimize=True, mode="token"),
        notes="metadata field in body; token mode must preserve it",
    ),
]

assert len(_MATRIX) >= 50, f"Matrix has only {len(_MATRIX)} cases; need 50+"


# ---------------------------------------------------------------------------
# Parametrize
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "diff_spec" in metafunc.fixturenames:
        metafunc.parametrize(
            "diff_spec",
            _MATRIX,
            ids=[s.name for s in _MATRIX],
        )


# ---------------------------------------------------------------------------
# Divergence collector
# ---------------------------------------------------------------------------


def _byte_diff_summary(legacy: bytes, engine: bytes) -> str:
    """Produce a concise diff summary for diagnostic output."""
    if legacy == engine:
        return "(identical)"
    lines: list[str] = [f"  legacy_len={len(legacy)}  engine_len={len(engine)}"]
    try:
        leg_j = json.loads(legacy)
        eng_j = json.loads(engine)
        leg_s = json.dumps(leg_j, indent=2, sort_keys=True)
        eng_s = json.dumps(eng_j, indent=2, sort_keys=True)
        # Collect changed lines
        leg_lines = leg_s.splitlines()
        eng_lines = eng_s.splitlines()
        import difflib

        diff = list(
            difflib.unified_diff(
                leg_lines, eng_lines, lineterm="", fromfile="legacy", tofile="engine", n=3
            )
        )
        lines.extend(diff[:60])  # cap at 60 diff lines
    except Exception:
        lines.append(f"  legacy[:200]={legacy[:200]!r}")
        lines.append(f"  engine[:200]={engine[:200]!r}")
    return "\n".join(lines)


def _cause_hypothesis(spec: GoldenCaseSpec, legacy: bytes, engine: bytes) -> str:
    """Guess the most likely cause of a divergence."""
    try:
        leg_j = json.loads(legacy)
        eng_j = json.loads(engine)
    except Exception:
        return "non-JSON divergence — raw byte mismatch"

    leg_tools = leg_j.get("tools")
    eng_tools = eng_j.get("tools")
    if leg_tools != eng_tools:
        return "tool-list mismatch — engine tool-sort or injection differs from handler"

    leg_msgs = leg_j.get("messages", [])
    eng_msgs = eng_j.get("messages", [])
    if len(leg_msgs) != len(eng_msgs):
        return (
            f"message-count mismatch ({len(leg_msgs)} legacy vs {len(eng_msgs)} engine) — "
            "engine drop/inject logic differs"
        )
    for i, (lm, em) in enumerate(zip(leg_msgs, eng_msgs)):
        if lm != em:
            lm_content = lm.get("content")
            em_content = em.get("content")
            if lm_content != em_content:
                return (
                    f"message[{i}].content differs — compression or cache_control "
                    "injection mismatch (role={lm.get('role')})"
                )
            return f"message[{i}] differs (non-content field)"

    leg_sys = leg_j.get("system")
    eng_sys = eng_j.get("system")
    if leg_sys != eng_sys:
        return "system field mismatch — system-prompt rewriting differs"

    return "unknown divergence — bodies differ but specific field not isolated"


# ---------------------------------------------------------------------------
# Main differential test
# ---------------------------------------------------------------------------


def test_engine_differential(diff_spec: GoldenCaseSpec) -> None:
    """Assert legacy handler and HeadroomEngine produce byte-identical outbound bodies.

    On divergence: fails immediately with a detailed diff + cause hypothesis.
    No silent xfail unless a case is marked nondeterministic_flag=True.
    """
    legacy_bytes, engine_bytes = _run_spec(diff_spec)

    if diff_spec.nondeterministic_flag:
        # Existence check only — caller must document WHY it's nondeterministic.
        assert engine_bytes, (
            f"Case '{diff_spec.name}' (nondeterministic_flag=True): engine produced no output"
        )
        return

    if legacy_bytes == engine_bytes:
        return  # fast-path pass

    diff_summary = _byte_diff_summary(legacy_bytes, engine_bytes)
    hypothesis = _cause_hypothesis(diff_spec, legacy_bytes, engine_bytes)

    pytest.fail(
        f"\nDIVERGENCE in '{diff_spec.name}'\n"
        f"  proxy_config : {diff_spec.proxy_config}\n"
        f"  frozen_count : {diff_spec.frozen_count}\n"
        f"  streaming    : {diff_spec.streaming}\n"
        f"  auth headers : {diff_spec.inbound_headers}\n"
        f"  notes        : {diff_spec.notes}\n"
        f"\nCAUSE HYPOTHESIS: {hypothesis}\n"
        f"\nBYTE DIFF:\n{diff_summary}\n"
    )


# ---------------------------------------------------------------------------
# Bulk divergence summary (always passes; shows matrix health in -v output)
# ---------------------------------------------------------------------------


def test_differential_coverage_summary() -> None:
    """Print a coverage breakdown of the differential matrix.

    Always passes; intended for human review via pytest -v output.
    Shows how many cases cover each axis of the matrix.
    """
    auth_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    tool_count = 0
    large_content_count = 0
    frozen_gt0_count = 0
    seeded_prev_count = 0
    streaming_count = 0

    for spec in _MATRIX:
        # auth
        if "x-api-key" in spec.inbound_headers and "x-headroom-client" not in spec.inbound_headers:
            key = "payg"
        elif "x-headroom-client" in spec.inbound_headers:
            key = "subscription"
        else:
            key = "oauth"
        auth_counts[key] = auth_counts.get(key, 0) + 1

        # mode
        pc = spec.proxy_config
        if not pc.get("optimize", True):
            mk = "passthrough"
        else:
            mk = pc.get("mode", "token")
        mode_counts[mk] = mode_counts.get(mk, 0) + 1

        if spec.body.get("tools"):
            tool_count += 1
        if spec.prev_original_messages:
            seeded_prev_count += 1
        if spec.frozen_count > 0:
            frozen_gt0_count += 1
        if spec.streaming:
            streaming_count += 1
        # large content: body text > 300 chars
        body_str = json.dumps(spec.body)
        if len(body_str) > 300:
            large_content_count += 1

    total = len(_MATRIX)
    assert total >= 50, f"Matrix too small: {total} < 50 cases"
    assert auth_counts.get("payg", 0) >= 5
    assert auth_counts.get("oauth", 0) >= 3
    assert auth_counts.get("subscription", 0) >= 2
    assert mode_counts.get("token", 0) >= 10
    assert mode_counts.get("cache", 0) >= 10
    assert mode_counts.get("passthrough", 0) >= 5
    assert tool_count >= 8
    assert frozen_gt0_count >= 5
    assert seeded_prev_count >= 3
    assert streaming_count >= 3
    assert large_content_count >= 5
