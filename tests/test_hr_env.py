"""Tests for headroom.env.get_hr_env — HR_* prefix with HEADROOM_* fallback."""

from __future__ import annotations

import warnings

import pytest

from headroom.env import _deprecation_warned, get_hr_env


@pytest.fixture(autouse=True)
def clear_deprecation_cache():
    """Reset the per-process deprecation-warned set before each test."""
    _deprecation_warned.clear()
    yield
    _deprecation_warned.clear()


class TestGetHrEnv:
    def test_returns_none_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("HR_FOO_TEST", raising=False)
        monkeypatch.delenv("HEADROOM_FOO_TEST", raising=False)
        assert get_hr_env("FOO_TEST") is None

    def test_returns_default_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("HR_FOO_TEST", raising=False)
        monkeypatch.delenv("HEADROOM_FOO_TEST", raising=False)
        assert get_hr_env("FOO_TEST", "fallback") == "fallback"

    def test_hr_prefix_takes_priority(self, monkeypatch):
        monkeypatch.setenv("HR_FOO_TEST", "new_value")
        monkeypatch.setenv("HEADROOM_FOO_TEST", "old_value")
        assert get_hr_env("FOO_TEST") == "new_value"

    def test_hr_prefix_alone(self, monkeypatch):
        monkeypatch.setenv("HR_FOO_TEST", "new_value")
        monkeypatch.delenv("HEADROOM_FOO_TEST", raising=False)
        assert get_hr_env("FOO_TEST") == "new_value"

    def test_headroom_prefix_fallback_returns_value(self, monkeypatch):
        monkeypatch.delenv("HR_FOO_TEST", raising=False)
        monkeypatch.setenv("HEADROOM_FOO_TEST", "old_value")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = get_hr_env("FOO_TEST")
        assert result == "old_value"

    def test_headroom_prefix_emits_deprecation_warning(self, monkeypatch):
        monkeypatch.delenv("HR_FOO_TEST", raising=False)
        monkeypatch.setenv("HEADROOM_FOO_TEST", "old_value")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_hr_env("FOO_TEST")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "HEADROOM_FOO_TEST" in str(w[0].message)
        assert "HR_FOO_TEST" in str(w[0].message)

    def test_headroom_prefix_warns_only_once_per_key(self, monkeypatch):
        monkeypatch.delenv("HR_FOO_TEST", raising=False)
        monkeypatch.setenv("HEADROOM_FOO_TEST", "old_value")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_hr_env("FOO_TEST")
            get_hr_env("FOO_TEST")
            get_hr_env("FOO_TEST")
        # Only one warning per key per process (deduplication via _deprecation_warned)
        assert sum(1 for warning in w if "HEADROOM_FOO_TEST" in str(warning.message)) == 1

    def test_hr_prefix_never_warns(self, monkeypatch):
        monkeypatch.setenv("HR_FOO_TEST", "new_value")
        monkeypatch.delenv("HEADROOM_FOO_TEST", raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_hr_env("FOO_TEST")
        assert result == "new_value"
        assert len(w) == 0
