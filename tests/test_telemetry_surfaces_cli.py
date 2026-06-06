"""Regression tests for the telemetry surface inventory CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from headroom.cli.main import main
from headroom.telemetry.surfaces import collect_telemetry_surface_dicts

_ENV_VARS = (
    "HEADROOM_CODEX_WIRE_DEBUG",
    "HEADROOM_CODEX_WIRE_DEBUG_DIR",
    "HEADROOM_HOST",
    "HEADROOM_LANGFUSE_ENABLED",
    "HEADROOM_LOG_FILE",
    "HEADROOM_LOG_MESSAGES",
    "HEADROOM_OTEL_METRICS_ENABLED",
    "HEADROOM_OTEL_METRICS_ENDPOINT",
    "HEADROOM_OTEL_METRICS_EXPORTER",
    "HEADROOM_PORT",
    "HEADROOM_SAVINGS_PATH",
    "HEADROOM_STATELESS",
    "HEADROOM_TELEMETRY",
    "HEADROOM_TOIN_BACKEND",
    "HEADROOM_TOIN_PATH",
    "HEADROOM_TOIN_TENANT_PREFIX",
    "HEADROOM_TOIN_URL",
    "HEADROOM_WORKSPACE_DIR",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_OTEL_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


def _clear_env(monkeypatch) -> None:  # noqa: ANN001
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _by_id(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["id"]): row for row in rows}


def test_collect_telemetry_surfaces_defaults(monkeypatch, tmp_path: Path) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))

    rows = _by_id(collect_telemetry_surface_dicts())

    assert rows["anonymous_beacon"]["status"] == "on"
    assert rows["anonymous_beacon"]["leaves_host"] == "yes"
    assert rows["otel_metrics"]["status"] == "off"
    assert rows["langfuse_tracing"]["status"] == "off"
    assert rows["savings_tracker"]["target"] == str(tmp_path / "proxy_savings.json")
    assert rows["debug_400_dumps"]["prompt_content"] == "yes, on upstream 4xx error dumps"
    assert "HEADROOM_LOG_MESSAGES" in rows["proxy_request_log"]["controls"]


def test_collect_telemetry_surfaces_env_overrides(monkeypatch, tmp_path: Path) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
    monkeypatch.setenv("HEADROOM_OTEL_METRICS_ENABLED", "1")
    monkeypatch.setenv("HEADROOM_OTEL_METRICS_ENDPOINT", "http://collector:4318/v1/metrics")
    monkeypatch.setenv("HEADROOM_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example")
    monkeypatch.setenv("HEADROOM_LOG_MESSAGES", "yes")
    monkeypatch.setenv("HEADROOM_LOG_FILE", str(tmp_path / "proxy.jsonl"))
    monkeypatch.setenv("HEADROOM_CODEX_WIRE_DEBUG", "on")
    monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "redis")
    monkeypatch.setenv("HEADROOM_TOIN_URL", "redis://localhost:6379/0")

    rows = _by_id(collect_telemetry_surface_dicts())

    assert rows["anonymous_beacon"]["status"] == "off"
    assert rows["anonymous_beacon"]["leaves_host"] == "no"
    assert rows["otel_metrics"]["status"] == "on"
    assert rows["otel_metrics"]["target"] == "http://collector:4318/v1/metrics"
    assert rows["langfuse_tracing"]["status"] == "on"
    assert rows["langfuse_tracing"]["target"] == (
        "https://langfuse.example/api/public/otel/v1/traces"
    )
    assert rows["proxy_request_log"]["prompt_content"] == "opt-in"
    assert rows["proxy_request_log"]["target"] == str(tmp_path / "proxy.jsonl")
    assert rows["codex_wire_debug"]["status"] == "on"
    assert rows["toin_aggregation"]["target"] == "redis://localhost:6379/0"


def test_telemetry_list_json_cli(monkeypatch, tmp_path: Path) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("HEADROOM_TELEMETRY", "off")

    result = CliRunner().invoke(main, ["telemetry", "list", "--json"])

    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    by_id = _by_id(rows)
    assert by_id["anonymous_beacon"]["status"] == "off"
    assert by_id["prometheus_metrics"]["target"] == "/metrics"
    assert by_id["debug_400_dumps"]["target"] == str(tmp_path / "logs" / "debug_400")


def test_telemetry_list_table_cli(monkeypatch, tmp_path: Path) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))

    result = CliRunner().invoke(main, ["telemetry", "list"])

    assert result.exit_code == 0, result.output
    assert "HEADROOM_TELEMETRY beacon" in result.output
    assert "Prometheus /metrics" in result.output
    assert "HTTP 400 debug dumps" in result.output


def test_root_help_includes_telemetry_command() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "telemetry" in result.output
