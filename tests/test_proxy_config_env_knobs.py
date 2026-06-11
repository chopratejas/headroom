"""ProxyConfig resolves compression knobs from env vars (issue #825).

HEADROOM_MIN_TOKENS / HEADROOM_MAX_ITEMS / HEADROOM_EXCLUDE_TOOLS are
documented on the ProxyConfig fields but were only parsed by the legacy
argparse entrypoint in server.py — the Click entrypoint (`headroom proxy`,
used by `headroom install agent run` deployments) builds ProxyConfig without
reading them, so they were silently inert. The fields now resolve the env
vars via field(default_factory=...), same pattern as memory_qdrant_*.
"""

from __future__ import annotations

import pytest

from headroom.proxy.models import ProxyConfig

_ENV_VARS = ("HEADROOM_MIN_TOKENS", "HEADROOM_MAX_ITEMS", "HEADROOM_EXCLUDE_TOOLS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure ambient env vars never leak into these tests."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_when_env_unset() -> None:
    config = ProxyConfig()

    assert config.min_tokens_to_crush == 500
    assert config.max_items_after_crush == 50
    assert config.exclude_tools is None


def test_env_vars_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_MIN_TOKENS", "8000")
    monkeypatch.setenv("HEADROOM_MAX_ITEMS", "200")
    monkeypatch.setenv("HEADROOM_EXCLUDE_TOOLS", "read_file,WebSearch")

    config = ProxyConfig()

    assert config.min_tokens_to_crush == 8000
    assert config.max_items_after_crush == 200
    # Names are kept in original and lowercase form for case-insensitive
    # matching, mirroring DEFAULT_EXCLUDE_TOOLS.
    assert config.exclude_tools == {"read_file", "WebSearch", "websearch"}


def test_explicit_arguments_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_MIN_TOKENS", "8000")
    monkeypatch.setenv("HEADROOM_MAX_ITEMS", "200")
    monkeypatch.setenv("HEADROOM_EXCLUDE_TOOLS", "read_file")

    config = ProxyConfig(
        min_tokens_to_crush=1234,
        max_items_after_crush=42,
        exclude_tools={"MyTool"},
    )

    assert config.min_tokens_to_crush == 1234
    assert config.max_items_after_crush == 42
    assert config.exclude_tools == {"MyTool"}


@pytest.mark.parametrize("bad", ["", "   ", "abc", "8e3"])
def test_invalid_int_values_fall_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv("HEADROOM_MIN_TOKENS", bad)
    monkeypatch.setenv("HEADROOM_MAX_ITEMS", bad)

    config = ProxyConfig()

    assert config.min_tokens_to_crush == 500
    assert config.max_items_after_crush == 50


@pytest.mark.parametrize("raw", ["", " ,, ", ","])
def test_empty_exclude_tools_means_none(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("HEADROOM_EXCLUDE_TOOLS", raw)

    config = ProxyConfig()

    assert config.exclude_tools is None


def test_exclude_tools_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_EXCLUDE_TOOLS", " read_file , terminal ")

    config = ProxyConfig()

    assert config.exclude_tools == {"read_file", "terminal"}
