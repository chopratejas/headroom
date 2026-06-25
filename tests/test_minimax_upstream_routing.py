"""Tests for the MiniMax upstream URL routing logic.

The previous review flagged that ``ProxyConfig.minimax_api_url`` and
``MINIMAX_TARGET_API_URL`` were plumbed through to the handler but the
handler always fell back to ``self.ANTHROPIC_API_URL`` — meaning the
new config surface was decorative. These tests pin the routing
behaviour so future refactors don't reintroduce the drift.

The tests cover:
    1. ``_resolve_minimax_upstream_url`` priority chain:
       override > ProxyConfig field > env var > Anthropic fallback.
    2. ``_strip_minimax_prefix`` removes the ``minimax/`` prefix.
    3. ``_is_minimax_model`` accepts both bare and prefixed forms.

These are pure-logic tests — no live upstream, no FastAPI app, no
network. They run in <0.1s and gate the routing contract.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from headroom.proxy.handlers.minimax import MiniMaxHandlerMixin


class TestStripMiniMaxPrefix:
    """The model name `minimax/MiniMax-M3` must become `MiniMax-M3`."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("MiniMax-M3", "MiniMax-M3"),  # bare — already clean
            ("minimax/MiniMax-M3", "MiniMax-M3"),  # the case we actually receive
            ("minimax/MiniMax-M2.7-highspeed", "MiniMax-M2.7-highspeed"),
            ("claude-sonnet-4-5", "claude-sonnet-4-5"),  # non-MiniMax — no-op
            ("", ""),  # empty — no-op
        ],
    )
    def test_strip(self, raw: str, expected: str) -> None:
        assert MiniMaxHandlerMixin._strip_minimax_prefix(raw) == expected


class TestIsMiniMaxModel:
    """Model classification: only MiniMax-M* family."""

    @pytest.mark.parametrize(
        "model",
        [
            "MiniMax-M3",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.7",
            "MiniMax-M2.5-highspeed",
            "MiniMax-M2",
            "minimax/MiniMax-M3",  # prefixed form
            "MINIMAX-M3",  # case-insensitive
        ],
    )
    def test_minimax_models_recognised(self, model: str) -> None:
        assert MiniMaxHandlerMixin._is_minimax_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-5",
            "claude-3-5-sonnet-20241022",
            "gpt-4o",
            "o1",
            "gemini-1.5-pro",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "",  # empty
            None,  # None
        ],
    )
    def test_non_minimax_models_rejected(self, model) -> None:
        assert MiniMaxHandlerMixin._is_minimax_model(model) is False


class TestResolveMiniMaxUpstreamUrl:
    """The priority chain is the central contract of this PR.

    Without this, ``ProxyConfig.minimax_api_url`` would be a
    decorative config field. These tests prove the handler actually
    routes traffic to the configured MiniMax endpoint.
    """

    def _make_handler(
        self,
        *,
        minimax_api_url: str | None = None,
    ) -> SimpleNamespace:
        """Build a stub with just enough surface for _resolve_minimax_upstream_url."""
        return SimpleNamespace(
            config=SimpleNamespace(minimax_api_url=minimax_api_url),
        )

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Make sure MINIMAX_TARGET_API_URL isn't accidentally set."""
        monkeypatch.delenv("MINIMAX_TARGET_API_URL", raising=False)

    @patch("headroom.proxy.server.HeadroomProxy")
    def test_override_wins_over_everything(self, mock_proxy_cls) -> None:
        """Caller-supplied URL beats config and env and fallback."""
        mock_proxy_cls.ANTHROPIC_API_URL = "https://api.anthropic.com"
        handler = self._make_handler(
            minimax_api_url="https://cfg.example/minimax"
        )
        with patch.dict(os.environ, {"MINIMAX_TARGET_API_URL": "https://env.example/minimax"}):
            result = MiniMaxHandlerMixin._resolve_minimax_upstream_url(
                handler, override="https://override.example/minimax"
            )
        assert result == "https://override.example/minimax"

    @patch("headroom.proxy.server.HeadroomProxy")
    def test_config_field_beats_env_and_fallback(self, mock_proxy_cls) -> None:
        """ProxyConfig.minimax_api_url beats env var and fallback."""
        mock_proxy_cls.ANTHROPIC_API_URL = "https://api.anthropic.com"
        handler = self._make_handler(
            minimax_api_url="https://cfg.example/minimax"
        )
        with patch.dict(os.environ, {"MINIMAX_TARGET_API_URL": "https://env.example/minimax"}):
            result = MiniMaxHandlerMixin._resolve_minimax_upstream_url(handler)
        assert result == "https://cfg.example/minimax"

    @patch("headroom.proxy.server.HeadroomProxy")
    def test_env_var_beats_fallback(self, mock_proxy_cls) -> None:
        """MINIMAX_TARGET_API_URL beats the Anthropic fallback."""
        mock_proxy_cls.ANTHROPIC_API_URL = "https://api.anthropic.com"
        handler = self._make_handler()  # no config field
        with patch.dict(os.environ, {"MINIMAX_TARGET_API_URL": "https://env.example/minimax"}):
            result = MiniMaxHandlerMixin._resolve_minimax_upstream_url(handler)
        assert result == "https://env.example/minimax"

    @patch("headroom.proxy.server.HeadroomProxy")
    def test_anthropic_fallback_when_nothing_set(self, mock_proxy_cls) -> None:
        """No config, no env, no override → use Anthropic upstream.

        This is the legacy behaviour that operators rely on when they
        pin ANTHROPIC_TARGET_API_URL to a Mavis Code gateway and want
        MiniMax to share the same upstream. Without this fallback,
        removing the field would break the original use case.
        """
        mock_proxy_cls.ANTHROPIC_API_URL = "https://agent.minimax.io/mavis/api/v1/llm"
        handler = self._make_handler()
        result = MiniMaxHandlerMixin._resolve_minimax_upstream_url(handler)
        assert result == "https://agent.minimax.io/mavis/api/v1/llm"

    @patch("headroom.proxy.server.HeadroomProxy")
    def test_explicit_minimax_upstream_does_not_route_to_anthropic(
        self, mock_proxy_cls
    ) -> None:
        """When ProxyConfig.minimax_api_url is the direct MiniMax API,
        MiniMax traffic MUST NOT silently route to Anthropic.

        This is the regression the reviewer caught: previously,
        ``minimax_api_url`` was plumbed but unused, so even operators
        who configured ``minimax_api_url='https://api.minimaxi.com/anthropic'``
        saw their MiniMax traffic silently going to Anthropic.
        """
        mock_proxy_cls.ANTHROPIC_API_URL = "https://api.anthropic.com"
        handler = self._make_handler(
            minimax_api_url="https://api.minimaxi.com/anthropic"
        )
        result = MiniMaxHandlerMixin._resolve_minimax_upstream_url(handler)
        assert result == "https://api.minimaxi.com/anthropic"
        assert "anthropic.com" not in result, (
            "Regression: minimax_api_url is being ignored, "
            "MiniMax traffic is falling back to Anthropic"
        )
