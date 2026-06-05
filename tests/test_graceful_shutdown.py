"""Tests for graceful shutdown and Ctrl+C signal handling.

Covers:
- _SuppressCancelledErrorFilter suppresses "Exception in ASGI application"
  log records whose exc_info is CancelledError
- _SuppressCancelledErrorFilter passes through unrelated error records
- timeout_graceful_shutdown is present in the uvicorn.run() call path
- The lifespan shutdown branch logs the proxy_shutdown event
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from headroom.proxy.server import (
    ProxyConfig,
    _SuppressCancelledErrorFilter,
    create_app,
)

# ---------------------------------------------------------------------------
# Unit tests for the logging filter
# ---------------------------------------------------------------------------


class TestSuppressCancelledErrorFilter:
    """_SuppressCancelledErrorFilter silences CancelledError noise from uvicorn."""

    def _make_record(
        self,
        level: int = logging.ERROR,
        exc_type: type | None = None,
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name="uvicorn.error",
            level=level,
            pathname="",
            lineno=0,
            msg="Exception in ASGI application",
            args=(),
            exc_info=(exc_type, exc_type() if exc_type else None, None) if exc_type else None,
        )
        return record

    def test_suppresses_cancelled_error_at_error_level(self) -> None:
        f = _SuppressCancelledErrorFilter()
        record = self._make_record(logging.ERROR, asyncio.CancelledError)
        assert f.filter(record) is False

    def test_passes_through_cancelled_error_at_warning_level(self) -> None:
        # Only suppress ERROR, not lower-severity records
        f = _SuppressCancelledErrorFilter()
        record = self._make_record(logging.WARNING, asyncio.CancelledError)
        assert f.filter(record) is True

    def test_passes_through_other_exception_at_error_level(self) -> None:
        f = _SuppressCancelledErrorFilter()
        record = self._make_record(logging.ERROR, ValueError)
        assert f.filter(record) is True

    def test_passes_through_record_without_exc_info(self) -> None:
        f = _SuppressCancelledErrorFilter()
        record = self._make_record(logging.ERROR, None)
        # exc_info is set to None tuple when exc_type is None
        record.exc_info = None
        assert f.filter(record) is True

    def test_passes_through_record_with_none_exc_type(self) -> None:
        f = _SuppressCancelledErrorFilter()
        record = self._make_record(logging.ERROR, None)
        record.exc_info = (None, None, None)
        assert f.filter(record) is True

    def test_suppresses_subclass_of_cancelled_error(self) -> None:
        """BaseException subclasses of CancelledError are also suppressed."""

        class MyCancelled(asyncio.CancelledError):
            pass

        f = _SuppressCancelledErrorFilter()
        record = self._make_record(logging.ERROR, MyCancelled)
        assert f.filter(record) is False


# ---------------------------------------------------------------------------
# Integration: filter is installed on uvicorn.error in run_server()
# ---------------------------------------------------------------------------


def test_run_server_installs_cancelled_error_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_server() attaches _SuppressCancelledErrorFilter to uvicorn.error logger."""
    installed_filters: list = []

    original_add_filter = logging.Logger.addFilter

    def capturing_add_filter(self: logging.Logger, f: logging.Filter) -> None:
        if self.name == "uvicorn.error" and isinstance(f, _SuppressCancelledErrorFilter):
            installed_filters.append(f)
        original_add_filter(self, f)

    monkeypatch.setattr(logging.Logger, "addFilter", capturing_add_filter)

    # Intercept uvicorn.run so we don't actually start a server
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)

    from headroom.proxy.server import run_server

    run_server(ProxyConfig(), print_banner=False)

    assert len(installed_filters) == 1, "Expected exactly one _SuppressCancelledErrorFilter"


# ---------------------------------------------------------------------------
# Integration: timeout_graceful_shutdown is forwarded to uvicorn.run()
# ---------------------------------------------------------------------------


def test_run_server_passes_timeout_graceful_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_server() passes timeout_graceful_shutdown=10 to uvicorn.run()."""
    captured: dict = {}

    def fake_uvicorn_run(*args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)

    from headroom.proxy.server import run_server

    run_server(ProxyConfig(), print_banner=False)

    assert "timeout_graceful_shutdown" in captured, (
        "uvicorn.run() must receive timeout_graceful_shutdown kwarg"
    )
    assert captured["timeout_graceful_shutdown"] == 10


# ---------------------------------------------------------------------------
# Integration: lifespan logs proxy_shutdown event on teardown
# ---------------------------------------------------------------------------


def test_lifespan_logs_shutdown_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lifespan finally-block logs event=proxy_shutdown when the app tears down.

    caplog cannot capture records from loggers that emit before propagation is
    configured, so this test installs a custom handler directly on
    ``headroom.proxy`` and checks that handler's records.
    """
    # Collect log records manually because caplog propagation is unreliable
    # when the root logger has pre-existing basicConfig handlers.
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    proxy_logger = logging.getLogger("headroom.proxy")
    capture_handler = _Capture()
    proxy_logger.addHandler(capture_handler)

    try:
        # Prevent sys.exit(78) from _check_rust_core when Rust extension absent
        monkeypatch.setattr(
            "headroom.proxy.server._check_rust_core", lambda: ("disabled", "test-mock")
        )

        config = ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
        )
        app = create_app(config)

        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=False):
            pass  # lifespan shutdown runs when the context manager exits

    finally:
        proxy_logger.removeHandler(capture_handler)

    shutdown_records = [r for r in captured if "event=proxy_shutdown" in r.getMessage()]
    assert shutdown_records, "Expected at least one log record containing 'event=proxy_shutdown'"
