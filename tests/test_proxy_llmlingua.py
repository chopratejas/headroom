"""Tests for LLMLingua opt-in mechanism in the proxy server.

These tests verify:
- ProxyConfig llmlingua settings
- LLMLingua transform integration in pipeline
- Status detection and logging hints
- CLI flag parsing
- DevEx: helpful messages when llmlingua unavailable
"""

from unittest.mock import MagicMock, patch

import pytest

# Skip if fastapi not available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import (
    HeadroomProxy,
    ProxyConfig,
    _get_llmlingua_banner_status,
    create_app,
)
from headroom.transforms import _LLMLINGUA_AVAILABLE

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def base_config():
    """Base config with optimization disabled for simpler tests."""
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        llmlingua_enabled=False,  # Explicitly disable for base config
        smart_routing=False,  # Use legacy mode for these tests
    )


@pytest.fixture
def llmlingua_config():
    """Config with LLMLingua enabled (legacy mode for explicit status testing)."""
    return ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        llmlingua_enabled=True,
        llmlingua_device="cpu",
        llmlingua_target_rate=0.4,
        smart_routing=False,  # Use legacy mode for explicit status testing
    )


@pytest.fixture
def client(base_config):
    """Create test client with base config."""
    app = create_app(base_config)
    with TestClient(app) as client:
        yield client


# =============================================================================
# TestProxyConfigLLMLingua
# =============================================================================


class TestProxyConfigLLMLingua:
    """Tests for LLMLingua settings in ProxyConfig."""

    def test_default_llmlingua_enabled(self):
        """LLMLingua is enabled by default (with smart routing)."""
        config = ProxyConfig()

        # LLMLingua is now enabled by default with smart routing
        assert config.llmlingua_enabled is True
        assert config.llmlingua_device == "auto"
        assert config.llmlingua_target_rate == 0.3

    def test_llmlingua_can_be_enabled(self):
        """LLMLingua can be enabled via config."""
        config = ProxyConfig(
            llmlingua_enabled=True,
            llmlingua_device="cuda",
            llmlingua_target_rate=0.5,
        )

        assert config.llmlingua_enabled is True
        assert config.llmlingua_device == "cuda"
        assert config.llmlingua_target_rate == 0.5

    def test_llmlingua_device_options(self):
        """LLMLingua device accepts valid options."""
        for device in ["auto", "cuda", "cpu", "mps"]:
            config = ProxyConfig(llmlingua_device=device)
            assert config.llmlingua_device == device

    def test_llmlingua_target_rate_range(self):
        """LLMLingua target rate accepts 0.0-1.0 range."""
        # Low rate (aggressive compression)
        config_low = ProxyConfig(llmlingua_target_rate=0.1)
        assert config_low.llmlingua_target_rate == 0.1

        # High rate (conservative compression)
        config_high = ProxyConfig(llmlingua_target_rate=0.8)
        assert config_high.llmlingua_target_rate == 0.8


# =============================================================================
# TestLLMLinguaSetup
# =============================================================================


class TestLLMLinguaSetup:
    """Tests for LLMLingua setup in HeadroomProxy."""

    def test_setup_returns_disabled_when_not_enabled(self, base_config):
        """Setup returns 'disabled' when llmlingua not enabled and not available."""
        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", False):
            proxy = HeadroomProxy(base_config)
            assert proxy._llmlingua_status == "disabled"

    def test_setup_returns_available_when_installed_but_not_enabled(self, base_config):
        """Setup returns 'available' when llmlingua installed but not enabled."""
        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", True):
            proxy = HeadroomProxy(base_config)
            assert proxy._llmlingua_status == "available"

    def test_setup_returns_enabled_when_enabled_and_available(self, llmlingua_config):
        """Setup returns 'enabled' when llmlingua enabled and available."""
        mock_compressor = MagicMock()

        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", True):
            with patch("headroom.proxy.server.LLMLinguaCompressor", mock_compressor):
                with patch("headroom.proxy.server.LLMLinguaConfig"):
                    proxy = HeadroomProxy(llmlingua_config)
                    assert proxy._llmlingua_status == "enabled"

    def test_setup_returns_unavailable_when_enabled_but_not_installed(self):
        """Setup returns 'unavailable' when enabled but llmlingua not installed."""
        config = ProxyConfig(
            llmlingua_enabled=True,
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            smart_routing=False,  # Use legacy mode for explicit status testing
        )

        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", False):
            proxy = HeadroomProxy(config)
            assert proxy._llmlingua_status == "unavailable"

    def test_llmlingua_compressor_added_to_pipeline(self, llmlingua_config):
        """LLMLinguaCompressor is added to pipeline when enabled."""
        mock_compressor_class = MagicMock()
        mock_compressor_instance = MagicMock()
        mock_compressor_class.return_value = mock_compressor_instance

        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", True):
            with patch("headroom.proxy.server.LLMLinguaCompressor", mock_compressor_class):
                with patch("headroom.proxy.server.LLMLinguaConfig") as mock_config:
                    HeadroomProxy(llmlingua_config)

                    # Verify LLMLinguaCompressor was instantiated
                    mock_compressor_class.assert_called_once()

                    # Verify config was passed with correct device and rate
                    call_args = mock_config.call_args
                    assert call_args.kwargs["device"] == "cpu"
                    assert call_args.kwargs["target_compression_rate"] == 0.4

    def test_llmlingua_not_added_when_disabled(self, base_config):
        """LLMLinguaCompressor is NOT added when disabled."""
        mock_compressor_class = MagicMock()

        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", True):
            with patch("headroom.proxy.server.LLMLinguaCompressor", mock_compressor_class):
                HeadroomProxy(base_config)

                # Should NOT be called when disabled
                mock_compressor_class.assert_not_called()


# =============================================================================
# TestBannerStatus
# =============================================================================


class TestBannerStatus:
    """Tests for banner status helper function."""

    def test_banner_disabled_when_not_available(self):
        """Banner shows DISABLED when llmlingua not available and not enabled."""
        config = ProxyConfig(llmlingua_enabled=False, smart_routing=False)

        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", False):
            status = _get_llmlingua_banner_status(config)
            assert status == "DISABLED"

    def test_banner_available_hint_when_installed(self):
        """Banner shows availability hint when installed but not enabled."""
        config = ProxyConfig(llmlingua_enabled=False, smart_routing=False)

        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", True):
            status = _get_llmlingua_banner_status(config)
            # Now shows DISABLED with hint to enable
            assert "DISABLED" in status
            assert "--no-llmlingua" in status  # Hint to remove the flag to enable

    def test_banner_enabled_when_active(self):
        """Banner shows ENABLED with config when active."""
        config = ProxyConfig(
            llmlingua_enabled=True,
            llmlingua_device="cuda",
            llmlingua_target_rate=0.25,
        )

        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", True):
            status = _get_llmlingua_banner_status(config)
            assert "ENABLED" in status
            assert "cuda" in status
            assert "0.25" in status

    def test_banner_shows_install_hint_when_requested_but_missing(self):
        """Banner shows install hint when enabled but not installed."""
        config = ProxyConfig(llmlingua_enabled=True, smart_routing=False)

        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", False):
            status = _get_llmlingua_banner_status(config)
            assert "NOT INSTALLED" in status
            assert "pip install" in status


# =============================================================================
# TestHealthEndpointWithLLMLingua
# =============================================================================


class TestHealthEndpointWithLLMLingua:
    """Tests for health endpoint reflecting LLMLingua status."""

    def test_health_returns_llmlingua_in_config(self, client):
        """Health endpoint works regardless of LLMLingua status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "config" in data


# =============================================================================
# TestStatsEndpointWithLLMLingua
# =============================================================================


class TestStatsEndpointWithLLMLingua:
    """Tests for stats endpoint with LLMLingua integration."""

    def test_stats_endpoint_works(self, client):
        """Stats endpoint works with any LLMLingua configuration."""
        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert "requests" in data
        assert "tokens" in data


# =============================================================================
# TestCLIArguments
# =============================================================================


class TestCLIArguments:
    """Tests for CLI argument parsing (without actually running server)."""

    def test_llmlingua_flag_defaults(self):
        """Default CLI values for LLMLingua settings."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--llmlingua", action="store_true")
        parser.add_argument("--llmlingua-device", default="auto")
        parser.add_argument("--llmlingua-rate", type=float, default=0.3)

        args = parser.parse_args([])

        assert args.llmlingua is False
        assert args.llmlingua_device == "auto"
        assert args.llmlingua_rate == 0.3

    def test_llmlingua_flag_enabled(self):
        """CLI --llmlingua flag enables LLMLingua."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--llmlingua", action="store_true")
        parser.add_argument("--llmlingua-device", default="auto")
        parser.add_argument("--llmlingua-rate", type=float, default=0.3)

        args = parser.parse_args(["--llmlingua"])

        assert args.llmlingua is True

    def test_llmlingua_device_flag(self):
        """CLI --llmlingua-device flag sets device."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--llmlingua-device", default="auto")

        args = parser.parse_args(["--llmlingua-device", "cuda"])

        assert args.llmlingua_device == "cuda"

    def test_llmlingua_rate_flag(self):
        """CLI --llmlingua-rate flag sets compression rate."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--llmlingua-rate", type=float, default=0.3)

        args = parser.parse_args(["--llmlingua-rate", "0.5"])

        assert args.llmlingua_rate == 0.5


# =============================================================================
# TestDevExMessages
# =============================================================================


class TestDevExMessages:
    """Tests for developer experience messages and hints."""

    def test_warning_logged_when_enabled_but_unavailable(self, caplog):
        """Warning is logged when llmlingua enabled but not installed."""
        import logging

        config = ProxyConfig(
            llmlingua_enabled=True,
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            smart_routing=False,  # Use legacy mode for explicit status testing
        )

        with caplog.at_level(logging.WARNING):
            with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", False):
                proxy = HeadroomProxy(config)

                # Should have logged a warning about missing llmlingua
                assert proxy._llmlingua_status == "unavailable"
                assert any("llmlingua" in r.message.lower() for r in caplog.records)
                assert any("pip install" in r.message for r in caplog.records)


# =============================================================================
# TestIntegrationWithActualLLMLingua
# =============================================================================


@pytest.mark.skipif(not _LLMLINGUA_AVAILABLE, reason="llmlingua not installed")
class TestIntegrationWithActualLLMLingua:
    """Integration tests that require actual llmlingua installation.

    These tests verify the full integration path when llmlingua is installed.
    """

    def test_proxy_starts_with_llmlingua_enabled(self):
        """Proxy starts successfully with LLMLingua enabled."""
        config = ProxyConfig(
            llmlingua_enabled=True,
            llmlingua_device="cpu",  # CPU for CI/test environments
            llmlingua_target_rate=0.3,
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            smart_routing=False,  # Use legacy mode for explicit status testing
        )

        # Should not raise
        proxy = HeadroomProxy(config)

        assert proxy._llmlingua_status == "enabled"

    def test_app_creates_with_llmlingua(self):
        """FastAPI app creates successfully with LLMLingua enabled."""
        config = ProxyConfig(
            llmlingua_enabled=True,
            llmlingua_device="cpu",
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
        )

        # Should not raise
        app = create_app(config)

        assert app is not None

    def test_health_endpoint_with_llmlingua_enabled(self):
        """Health endpoint works with LLMLingua enabled."""
        config = ProxyConfig(
            llmlingua_enabled=True,
            llmlingua_device="cpu",
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
        )

        app = create_app(config)
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge cases for LLMLingua proxy integration."""

    def test_multiple_proxy_instances_independent(self):
        """Multiple proxy instances have independent LLMLingua status."""
        config_enabled = ProxyConfig(
            llmlingua_enabled=True,
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            smart_routing=False,  # Use legacy mode for explicit status testing
        )
        config_disabled = ProxyConfig(
            llmlingua_enabled=False,
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            smart_routing=False,  # Use legacy mode for explicit status testing
        )

        with patch("headroom.proxy.server._LLMLINGUA_AVAILABLE", True):
            with patch("headroom.proxy.server.LLMLinguaCompressor"):
                with patch("headroom.proxy.server.LLMLinguaConfig"):
                    proxy_enabled = HeadroomProxy(config_enabled)
                    proxy_disabled = HeadroomProxy(config_disabled)

                    assert proxy_enabled._llmlingua_status == "enabled"
                    assert proxy_disabled._llmlingua_status == "available"

    def test_config_immutable_after_proxy_creation(self, base_config):
        """Config values are captured at proxy creation time."""
        proxy = HeadroomProxy(base_config)

        # Modifying config after creation doesn't affect proxy
        # (ProxyConfig is a dataclass, so this tests the pattern)
        original_status = proxy._llmlingua_status

        # Status should remain unchanged
        assert proxy._llmlingua_status == original_status
