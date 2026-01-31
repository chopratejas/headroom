"""Tests for Strands hooks integration.

Tests cover:
1. HeadroomHookProvider - HookProvider for Strands SDK lifecycle events
2. Tool output compression - Core feature
3. Token alerts and metrics tracking
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

# Check if Strands is available
try:
    import strands  # noqa: F401

    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False

from headroom import HeadroomConfig, HeadroomMode

# Skip all tests if Strands not installed
pytestmark = pytest.mark.skipif(not STRANDS_AVAILABLE, reason="Strands not installed")


class TestHeadroomHookProvider:
    """Tests for HeadroomHookProvider."""

    def test_init_and_defaults(self):
        """Initialize with default and custom settings."""
        from headroom.integrations.strands import HeadroomHookProvider

        # Test defaults
        hook = HeadroomHookProvider()
        assert hook.mode == HeadroomMode.OPTIMIZE
        assert hook.compress_tool_outputs is True
        assert hook.total_tokens_saved == 0
        assert hook.metrics_history == []

        # Test custom config
        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)
        hook2 = HeadroomHookProvider(
            config=config,
            mode=HeadroomMode.SIMULATE,
            compress_tool_outputs=False,
            min_tokens_to_compress=100,
            token_alert_threshold=5000,
        )
        assert hook2.config is config
        assert hook2.mode == HeadroomMode.SIMULATE
        assert hook2.compress_tool_outputs is False
        assert hook2.token_alert_threshold == 5000

    def test_register_hooks_with_registry(self):
        """register_hooks registers all 6 callbacks with registry."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        mock_registry = MagicMock()

        hook.register_hooks(mock_registry)

        assert mock_registry.add_callback.call_count == 6

    def test_compress_large_tool_output(self):
        """Compresses large tool output and tracks metrics."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(
            compress_tool_outputs=True,
            min_tokens_to_compress=10,
            max_items_after_crush=5,
        )

        # Create large JSON output (50 items)
        large_data = [{"id": i, "value": f"item-{i}", "data": "x" * 50} for i in range(50)]
        large_json = json.dumps(large_data)

        mock_event = MagicMock()
        mock_event.tool_use = {"name": "get_items"}
        mock_event.result = {"content": [{"text": large_json}]}

        hook._on_after_tool_call(mock_event)

        # Verify compression happened
        assert hook.tool_compression_count == 1
        assert hook.total_tokens_saved > 0
        assert len(hook.metrics_history) == 1
        assert hook.metrics_history[0].tool_name == "get_items"
        assert hook.metrics_history[0].savings_percent > 0

    def test_skip_compression_when_disabled_or_below_threshold(self):
        """Does not compress when disabled or output below threshold."""
        from headroom.integrations.strands import HeadroomHookProvider

        # Test disabled
        hook1 = HeadroomHookProvider(compress_tool_outputs=False)
        mock_event = MagicMock()
        mock_event.tool_use = {"name": "test"}
        mock_event.result = {"content": [{"text": '{"data": [1,2,3]}'}]}
        hook1._on_after_tool_call(mock_event)
        assert hook1.tool_compression_count == 0

        # Test below threshold
        hook2 = HeadroomHookProvider(compress_tool_outputs=True, min_tokens_to_compress=10000)
        hook2._on_after_tool_call(mock_event)
        assert hook2.tool_compression_count == 0

    def test_token_alert_threshold(self):
        """Triggers alert when tokens exceed threshold."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(token_alert_threshold=10)

        mock_event = MagicMock()
        mock_event.messages = [{"role": "user", "content": "x" * 10000}]

        hook._on_before_model_call(mock_event)

        assert len(hook.alerts) == 1
        assert "Token alert" in hook.alerts[0]

    def test_get_savings_summary(self):
        """get_savings_summary returns correct metrics."""
        from headroom.integrations.strands import HeadroomHookProvider
        from headroom.integrations.strands.hooks import CallbackMetrics

        hook = HeadroomHookProvider()

        # Empty summary
        summary = hook.get_savings_summary()
        assert summary["total_events"] == 0

        # With data
        hook._metrics_history = [
            CallbackMetrics(
                request_id="1",
                timestamp=datetime.now(),
                event_type="tool_compress",
                tokens_before=100,
                tokens_after=60,
                tokens_saved=40,
                savings_percent=40.0,
            ),
            CallbackMetrics(
                request_id="2",
                timestamp=datetime.now(),
                event_type="tool_compress",
                tokens_before=200,
                tokens_after=100,
                tokens_saved=100,
                savings_percent=50.0,
            ),
        ]
        hook._total_tokens_saved = 140
        hook._tool_compression_count = 2

        summary = hook.get_savings_summary()
        assert summary["total_events"] == 2
        assert summary["total_tokens_saved"] == 140
        assert summary["average_savings_percent"] == 45.0

    def test_reset_clears_state(self):
        """reset() clears all tracked state."""
        from headroom.integrations.strands import HeadroomHookProvider
        from headroom.integrations.strands.hooks import CallbackMetrics

        hook = HeadroomHookProvider()
        hook._metrics_history = [
            CallbackMetrics(request_id="1", timestamp=datetime.now(), event_type="test")
        ]
        hook._total_tokens_saved = 100
        hook._tool_compression_count = 2
        hook._alerts = ["alert"]

        hook.reset()

        assert hook._metrics_history == []
        assert hook._total_tokens_saved == 0
        assert hook._tool_compression_count == 0
        assert hook._alerts == []


class TestCreateHeadroomCallback:
    """Tests for create_headroom_callback convenience function."""

    def test_creates_configured_hook_provider(self):
        """Creates HeadroomHookProvider with all parameters."""
        from headroom.integrations.strands import HeadroomHookProvider, create_headroom_callback

        config = HeadroomConfig()
        hook = create_headroom_callback(
            config=config,
            mode=HeadroomMode.SIMULATE,
            model="gpt-4o",
            compress_tool_outputs=False,
            log_level="DEBUG",
            token_alert_threshold=5000,
        )

        assert isinstance(hook, HeadroomHookProvider)
        assert hook.config is config
        assert hook.mode == HeadroomMode.SIMULATE
        assert hook.model == "gpt-4o"
        assert hook.compress_tool_outputs is False
        assert hook.log_level == "DEBUG"
        assert hook.token_alert_threshold == 5000
