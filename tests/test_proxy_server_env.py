from __future__ import annotations

from headroom.proxy.server import (
    _get_env_bool,
    _get_env_float,
    _get_env_int,
    _get_env_str,
    _parse_tool_profiles,
)


def test_env_helpers_parse_values_and_fallbacks(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_BOOL", raising=False)
    monkeypatch.setenv("HEADROOM_INT", "42")
    monkeypatch.setenv("HEADROOM_FLOAT", "0.75")
    monkeypatch.setenv("HEADROOM_STR", "value")

    assert _get_env_bool("HEADROOM_BOOL", True) is True
    monkeypatch.setenv("HEADROOM_BOOL", "yes")
    assert _get_env_bool("HEADROOM_BOOL", False) is True
    monkeypatch.setenv("HEADROOM_BOOL", "off")
    assert _get_env_bool("HEADROOM_BOOL", True) is False

    assert _get_env_int("HEADROOM_INT", 1) == 42
    monkeypatch.setenv("HEADROOM_INT", "nope")
    assert _get_env_int("HEADROOM_INT", 7) == 7

    assert _get_env_float("HEADROOM_FLOAT", 1.0) == 0.75
    monkeypatch.setenv("HEADROOM_FLOAT", "bad")
    assert _get_env_float("HEADROOM_FLOAT", 2.5) == 2.5

    assert _get_env_str("HEADROOM_STR", "default") == "value"
    monkeypatch.delenv("HEADROOM_STR", raising=False)
    assert _get_env_str("HEADROOM_STR", "default") == "default"


def test_parse_tool_profiles_merges_cli_and_env_and_warns(caplog, monkeypatch) -> None:
    monkeypatch.setenv(
        "HEADROOM_TOOL_PROFILES",
        "Bash:moderate, invalid-entry, WebFetch:aggressive, Read:unknown",
    )

    profiles = _parse_tool_profiles(["Grep:conservative"])

    assert profiles["Grep"].bias == 1.5
    assert profiles["Bash"].bias == 1.0
    assert profiles["WebFetch"].bias == 0.7
    assert "Read" not in profiles
