"""Runtime inventory for Headroom telemetry and observability surfaces."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from headroom import paths

_OFF_VALUES = frozenset(("off", "false", "0", "no", "disable", "disabled"))
_ON_VALUES = frozenset(("on", "true", "1", "yes", "enable", "enabled"))


@dataclass(frozen=True, slots=True)
class TelemetrySurface:
    """A single place a Headroom operator can observe or export runtime data."""

    id: str
    name: str
    status: str
    default: str
    emits: str
    observe: str
    export: str
    retention: str
    prompt_content: str
    leaves_host: str
    target: str
    controls: tuple[str, ...]
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""

        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "default": self.default,
            "emits": self.emits,
            "observe": self.observe,
            "export": self.export,
            "retention": self.retention,
            "prompt_content": self.prompt_content,
            "leaves_host": self.leaves_host,
            "target": self.target,
            "controls": list(self.controls),
            "notes": self.notes,
        }


def _env(env: Mapping[str, str], name: str, default: str = "") -> str:
    return env.get(name, default).strip()


def _env_bool(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    raw = _env(env, name)
    if not raw:
        return default
    value = raw.lower()
    if value in _ON_VALUES:
        return True
    if value in _OFF_VALUES:
        return False
    return default


def _telemetry_beacon_enabled(env: Mapping[str, str]) -> bool:
    value = _env(env, "HEADROOM_TELEMETRY", "on").lower()
    return value not in _OFF_VALUES


def _proxy_base_url(env: Mapping[str, str]) -> str:
    host = _env(env, "HEADROOM_HOST", "127.0.0.1") or "127.0.0.1"
    port = _env(env, "HEADROOM_PORT", "8787") or "8787"
    return f"http://{host}:{port}"


def _langfuse_endpoint(env: Mapping[str, str]) -> str:
    base_url = (
        _env(env, "LANGFUSE_BASE_URL")
        or _env(env, "LANGFUSE_OTEL_HOST")
        or "https://cloud.langfuse.com"
    )
    return f"{base_url.rstrip('/')}/api/public/otel/v1/traces"


def _otel_exporter(env: Mapping[str, str]) -> str:
    exporter = _env(env, "HEADROOM_OTEL_METRICS_EXPORTER", "otlp_http")
    normalized = exporter.lower().replace("-", "_")
    return normalized if normalized in {"console", "otlp_http"} else "otlp_http"


def _toin_status_and_target(env: Mapping[str, str]) -> tuple[str, str, str]:
    backend = _env(env, "HEADROOM_TOIN_BACKEND").lower()
    stateless = _env_bool(env, "HEADROOM_STATELESS", default=False)
    path = str(paths.toin_path())

    if backend == "none" or stateless:
        return (
            "off-requested",
            "memory only",
            "HEADROOM_TOIN_BACKEND=none is the intended in-memory/stateless control.",
        )
    if backend and backend != "filesystem":
        target = _env(env, "HEADROOM_TOIN_URL") or f"entry point backend: {backend}"
        return (
            "on",
            target,
            "Storage is provided by the registered headroom.toin_backend entry point.",
        )
    return "on", path, "Uses the filesystem backend unless a custom backend is configured."


def collect_telemetry_surfaces(
    env: Mapping[str, str] | None = None,
) -> list[TelemetrySurface]:
    """Return Headroom telemetry surfaces using only local environment state.

    This function intentionally does not contact the proxy, start background
    tasks, import the FastAPI server, or create any filesystem paths.
    """

    resolved_env = env or os.environ
    base_url = _proxy_base_url(resolved_env)

    telemetry_on = _telemetry_beacon_enabled(resolved_env)
    otel_enabled = _env_bool(resolved_env, "HEADROOM_OTEL_METRICS_ENABLED", default=False)
    otel_exporter = _otel_exporter(resolved_env)
    otel_endpoint = _env(resolved_env, "HEADROOM_OTEL_METRICS_ENDPOINT")
    langfuse_enabled = _env_bool(resolved_env, "HEADROOM_LANGFUSE_ENABLED", default=False)
    langfuse_complete = bool(
        _env(resolved_env, "LANGFUSE_PUBLIC_KEY") and _env(resolved_env, "LANGFUSE_SECRET_KEY")
    )
    log_messages = _env_bool(resolved_env, "HEADROOM_LOG_MESSAGES", default=False)
    log_file = _env(resolved_env, "HEADROOM_LOG_FILE")
    codex_wire_debug = _env_bool(resolved_env, "HEADROOM_CODEX_WIRE_DEBUG", default=False)
    toin_status, toin_target, toin_notes = _toin_status_and_target(resolved_env)

    if otel_exporter == "console":
        otel_target = "stdout"
        otel_leaves_host = "no"
    else:
        otel_target = otel_endpoint or "OTLP/HTTP exporter default endpoint"
        otel_leaves_host = "yes" if otel_enabled else "no"

    if langfuse_enabled and langfuse_complete:
        langfuse_status = "on"
        langfuse_target = _langfuse_endpoint(resolved_env)
        langfuse_leaves_host = "yes"
    elif langfuse_enabled:
        langfuse_status = "incomplete"
        langfuse_target = _langfuse_endpoint(resolved_env)
        langfuse_leaves_host = "no"
    else:
        langfuse_status = "off"
        langfuse_target = "-"
        langfuse_leaves_host = "no"

    request_log_target = "memory"
    if log_file:
        request_log_target = log_file

    return [
        TelemetrySurface(
            id="anonymous_beacon",
            name="HEADROOM_TELEMETRY beacon",
            status="on" if telemetry_on else "off",
            default="on",
            emits=(
                "Anonymous aggregate proxy stats: version, Python/OS, install mode, "
                "request counts, token savings, cache/latency and strategy aggregates."
            ),
            observe="Startup banner and /stats anon_telemetry_shipping.",
            export="Headroom Supabase REST endpoint when enabled.",
            retention="Remote service controlled; not persisted locally by the beacon.",
            prompt_content="no",
            leaves_host="yes" if telemetry_on else "no",
            target="Headroom anonymous telemetry endpoint" if telemetry_on else "-",
            controls=("HEADROOM_TELEMETRY=off", "headroom proxy --no-telemetry"),
            notes="The beacon reads local /stats and silently skips on network failures.",
        ),
        TelemetrySurface(
            id="prometheus_metrics",
            name="Prometheus /metrics",
            status="on",
            default="on when proxy runs",
            emits="Request, token, cache, latency, compression, subscription and health metrics.",
            observe=f"GET {base_url}/metrics",
            export="Prometheus scrape or any compatible collector.",
            retention="In-memory counters for the running proxy process.",
            prompt_content="no",
            leaves_host="no active export; reachable if the proxy is exposed",
            target="/metrics",
            controls=("Proxy host/port binding", "Network/firewall policy"),
        ),
        TelemetrySurface(
            id="otel_metrics",
            name="OpenTelemetry metrics",
            status="on" if otel_enabled else "off",
            default="off",
            emits="The same operational counters/histograms as the proxy metrics facade.",
            observe="/stats otel block, collector output, or console exporter.",
            export="OTLP/HTTP or console exporter.",
            retention="Exporter controlled.",
            prompt_content="no",
            leaves_host=otel_leaves_host,
            target=otel_target if otel_enabled else "-",
            controls=(
                "HEADROOM_OTEL_METRICS_ENABLED",
                "HEADROOM_OTEL_METRICS_EXPORTER",
                "HEADROOM_OTEL_METRICS_ENDPOINT",
                "HEADROOM_OTEL_METRICS_HEADERS",
            ),
        ),
        TelemetrySurface(
            id="langfuse_tracing",
            name="Langfuse tracing",
            status=langfuse_status,
            default="off",
            emits="Compression pipeline spans with model/provider, token counts and transform names.",
            observe="/stats langfuse block or the Langfuse dashboard.",
            export="Langfuse OTLP trace endpoint.",
            retention="Langfuse controlled.",
            prompt_content="no by default",
            leaves_host=langfuse_leaves_host,
            target=langfuse_target,
            controls=(
                "HEADROOM_LANGFUSE_ENABLED",
                "LANGFUSE_PUBLIC_KEY",
                "LANGFUSE_SECRET_KEY",
                "LANGFUSE_BASE_URL",
                "LANGFUSE_OTEL_HOST",
            ),
            notes="Status is incomplete until both Langfuse keys are present.",
        ),
        TelemetrySurface(
            id="savings_tracker",
            name="Savings tracker",
            status="on",
            default="on",
            emits="Durable token/cost savings totals and bounded history rollups.",
            observe=f"GET {base_url}/stats-history or read the JSON file.",
            export="/stats-history JSON/CSV or the configured file path.",
            retention="Bounded history plus lifetime totals until the file is removed.",
            prompt_content="no",
            leaves_host="no",
            target=str(paths.savings_path()),
            controls=("HEADROOM_SAVINGS_PATH", "HEADROOM_WORKSPACE_DIR"),
            notes="HEADROOM_SAVINGS_PATH relocates the ledger; there is no dedicated off switch.",
        ),
        TelemetrySurface(
            id="dashboard",
            name="Dashboard",
            status="on",
            default="on when proxy runs",
            emits="Live proxy, savings, request, cache, quota, TOIN and subscription views.",
            observe=f"GET {base_url}/dashboard",
            export="Browser view; reads /stats and /stats-history.",
            retention="Browser/session only; backed by runtime stats and savings history.",
            prompt_content="no by default",
            leaves_host="no active export; reachable if the proxy is exposed",
            target="/dashboard",
            controls=("Proxy host/port binding", "Network/firewall policy"),
        ),
        TelemetrySurface(
            id="stats_endpoints",
            name="Stats endpoints",
            status="on",
            default="on when proxy runs",
            emits="Live stats, durable history, quotas, request metadata and subsystem status.",
            observe=f"GET {base_url}/stats, /stats-history, /quota",
            export="/stats JSON and /stats-history JSON/CSV.",
            retention="In-memory stats plus savings ledger for history.",
            prompt_content="no by default",
            leaves_host="no active export; reachable if the proxy is exposed",
            target="/stats, /stats-history, /quota",
            controls=("Proxy host/port binding", "Network/firewall policy"),
        ),
        TelemetrySurface(
            id="toin_aggregation",
            name="TOIN aggregation",
            status=toin_status,
            default="on",
            emits="Aggregated tool-output compression/retrieval patterns and strategy outcomes.",
            observe=f"GET {base_url}/v1/toin/stats or inspect the configured backend.",
            export="Filesystem JSON by default; custom headroom.toin_backend entry points when enabled.",
            retention="Backend controlled; filesystem data remains until removed.",
            prompt_content="no raw values; structure and outcome aggregates only",
            leaves_host="no unless a custom backend points outside the host",
            target=toin_target,
            controls=(
                "HEADROOM_TOIN_BACKEND",
                "HEADROOM_TOIN_PATH",
                "HEADROOM_TOIN_URL",
                "HEADROOM_TOIN_TENANT_PREFIX",
            ),
            notes=toin_notes,
        ),
        TelemetrySurface(
            id="proxy_request_log",
            name="Proxy request log",
            status="on",
            default="metadata in memory; file off unless configured",
            emits="Recent request metadata, outcomes, tags, token counts, model/provider and timings.",
            observe="/stats recent_requests, /transformations/feed, or the JSONL file.",
            export="In-memory feed and optional HEADROOM_LOG_FILE/--log-file JSONL.",
            retention="Up to 10,000 in-memory entries; file retention is operator controlled.",
            prompt_content="opt-in" if log_messages else "no",
            leaves_host="no",
            target=request_log_target,
            controls=("HEADROOM_LOG_FILE", "--log-file", "HEADROOM_LOG_MESSAGES", "--log-messages"),
            notes="HEADROOM_LOG_MESSAGES stores request/response content for the live feed.",
        ),
        TelemetrySurface(
            id="debug_400_dumps",
            name="HTTP 400 debug dumps",
            status="on for upstream 4xx",
            default="on for upstream error diagnostics",
            emits="Full local diagnostic JSON for upstream errors, including sent/original messages.",
            observe=f"Read files under {paths.debug_400_dir()}.",
            export="Local filesystem only.",
            retention="Until files are removed by the operator.",
            prompt_content="yes, on upstream 4xx error dumps",
            leaves_host="no",
            target=str(paths.debug_400_dir()),
            controls=("HEADROOM_WORKSPACE_DIR",),
            notes="No dedicated off switch is currently exposed for these diagnostic dumps.",
        ),
        TelemetrySurface(
            id="codex_wire_debug",
            name="Codex wire debug captures",
            status="on" if codex_wire_debug else "off",
            default="off",
            emits="Opt-in Codex wire snapshots and matching proxy log frame traces.",
            observe=f"Read files under {paths.codex_wire_debug_dir()} or HEADROOM_CODEX_WIRE_DEBUG_DIR.",
            export="Local filesystem only.",
            retention="Until files are removed by the operator.",
            prompt_content="yes, when explicitly enabled; secrets are redacted",
            leaves_host="no",
            target=_env(resolved_env, "HEADROOM_CODEX_WIRE_DEBUG_DIR")
            or str(paths.codex_wire_debug_dir()),
            controls=(
                "HEADROOM_CODEX_WIRE_DEBUG",
                "HEADROOM_CODEX_WIRE_DEBUG_DIR",
                "--codex-wire-debug",
                "--codex-wire-debug-dir",
            ),
        ),
    ]


def collect_telemetry_surface_dicts(
    env: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """Return telemetry surfaces as dictionaries for CLI JSON output."""

    return [surface.as_dict() for surface in collect_telemetry_surfaces(env)]
