"""Tests for startup log noise suppression.

Covers the fixes in:
- headroom/memory/adapters/embedders.py (HF env vars, httpx logger)
- headroom/providers/anthropic.py (warn=False suppresses tiktoken warning)
- headroom/providers/litellm.py (suppress_debug_info, set_verbose)
- headroom/transforms/html_extractor.py (trafilatura logger CRITICAL)
"""

from __future__ import annotations

import importlib
import logging
import os
import warnings
from types import SimpleNamespace

_EMBEDDER_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class TestAnthropicWarnParameter:
    """AnthropicProvider.warn=False suppresses the no-client tiktoken warning."""

    def test_warn_true_emits_warning_without_client(self):
        """Default warn=True should emit UserWarning when no client is given."""
        import headroom.providers.anthropic as _mod
        from headroom.providers.anthropic import AnthropicProvider

        # Only runs if the module-level dedup flag hasn't fired yet in this process;
        # we reset it to guarantee the warning fires.

        original = _mod._FALLBACK_WARNING_SHOWN
        _mod._FALLBACK_WARNING_SHOWN = False
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                provider = AnthropicProvider(warn=True)
                # Trigger token-counter creation which is where warning fires
                try:
                    provider.get_token_counter("claude-3-5-sonnet-20241022")
                except Exception:
                    pass
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert any("tiktoken" in str(warning.message) for warning in user_warnings)
        finally:
            _mod._FALLBACK_WARNING_SHOWN = original

    def test_warn_false_suppresses_warning(self):
        """warn=False must produce zero UserWarnings about tiktoken fallback."""
        import headroom.providers.anthropic as _mod
        from headroom.providers.anthropic import AnthropicProvider

        original = _mod._FALLBACK_WARNING_SHOWN
        _mod._FALLBACK_WARNING_SHOWN = False
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                provider = AnthropicProvider(warn=False)
                try:
                    provider.get_token_counter("claude-3-5-sonnet-20241022")
                except Exception:
                    pass
            tiktoken_warnings = [
                x for x in w if issubclass(x.category, UserWarning) and "tiktoken" in str(x.message)
            ]
            assert tiktoken_warnings == [], (
                f"Expected no tiktoken warnings with warn=False, got: {tiktoken_warnings}"
            )
        finally:
            _mod._FALLBACK_WARNING_SHOWN = original

    def test_registry_uses_warn_false(self):
        """The internal proxy provider registry must pass warn=False to AnthropicProvider."""
        import inspect

        from headroom.providers import registry as _registry_mod

        source = inspect.getsource(_registry_mod)
        assert "AnthropicProvider(warn=False)" in source, (
            "registry.py must instantiate AnthropicProvider with warn=False"
        )


class TestEmbedderLogLevels:
    """headroom.memory.adapters.embedders must set specific logger levels at import time."""

    def test_huggingface_hub_logger_is_error_or_higher(self):
        """huggingface_hub logger must be silenced to ERROR or above."""
        import headroom.memory.adapters.embedders  # noqa: F401

        level = logging.getLogger("huggingface_hub").level
        assert level >= logging.ERROR, (
            f"Expected huggingface_hub logger level >= ERROR ({logging.ERROR}), got {level}"
        )

    def test_httpx_logger_is_warning_or_higher(self):
        """httpx logger must be set to WARNING or above to suppress HEAD request INFO lines."""
        import headroom.memory.adapters.embedders  # noqa: F401

        level = logging.getLogger("httpx").level
        assert level >= logging.WARNING, (
            f"Expected httpx logger level >= WARNING ({logging.WARNING}), got {level}"
        )

    def test_hf_hub_env_vars_are_set(self):
        """HF Hub env vars to disable progress bars and implicit tokens must be set."""
        import os

        import headroom.memory.adapters.embedders  # noqa: F401

        assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
        assert os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1"


class TestEmbedderThreadCaps:
    """Local sentence-transformers embedding should not oversubscribe BLAS threads."""

    def test_blas_thread_env_defaults_to_one(self, monkeypatch):
        """Import-time defaults cap BLAS/OpenMP pools unless the user configured them."""
        for name in (*_EMBEDDER_THREAD_ENV_VARS, "HEADROOM_MEMORY_EMBED_THREADS"):
            monkeypatch.delenv(name, raising=False)

        import headroom.memory.adapters.embedders as embedders

        importlib.reload(embedders)

        for name in _EMBEDDER_THREAD_ENV_VARS:
            assert os.environ.get(name) == "1"

    def test_blas_thread_env_respects_existing_values(self, monkeypatch):
        """Existing BLAS/OpenMP env vars must win over Headroom defaults."""
        for name in _EMBEDDER_THREAD_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("HEADROOM_MEMORY_EMBED_THREADS", "2")
        monkeypatch.setenv("OMP_NUM_THREADS", "4")

        import headroom.memory.adapters.embedders as embedders

        importlib.reload(embedders)

        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["MKL_NUM_THREADS"] == "2"

    def test_configure_torch_thread_cap_calls_torch_setters(self, monkeypatch):
        """If torch is already imported, apply the same cap through torch APIs."""
        import headroom.memory.adapters.embedders as embedders

        calls: list[tuple[str, int]] = []
        fake_torch = SimpleNamespace(
            set_num_threads=lambda value: calls.append(("threads", value)),
            set_num_interop_threads=lambda value: calls.append(("interop", value)),
        )

        monkeypatch.setattr(embedders, "_MEMORY_EMBED_THREAD_CAP", 2)

        embedders._configure_torch_thread_cap(fake_torch)

        assert calls == [("threads", 2), ("interop", 2)]


class TestLiteLLMLogSuppression:
    """litellm startup banner suppression must be applied at import time."""

    def test_litellm_suppress_debug_info_is_set(self):
        """litellm.suppress_debug_info must be True after importing the litellm provider."""
        litellm = pytest_importorskip_litellm()
        if litellm is None:
            return  # litellm not installed — skip gracefully

        import headroom.providers.litellm  # noqa: F401

        assert litellm.suppress_debug_info is True, (
            "litellm.suppress_debug_info must be True to silence startup banner"
        )

    def test_litellm_set_verbose_is_false(self):
        """litellm.set_verbose must be False after importing the litellm provider."""
        litellm = pytest_importorskip_litellm()
        if litellm is None:
            return

        import headroom.providers.litellm  # noqa: F401

        assert litellm.set_verbose is False, (
            "litellm.set_verbose must be False to suppress verbose debug output"
        )


def pytest_importorskip_litellm():
    """Return litellm if installed, else None (for graceful skip in optional-dep tests)."""
    try:
        import litellm

        return litellm
    except ImportError:
        return None


class TestTrafilaturaLogLevel:
    """trafilatura logger must be raised to CRITICAL to suppress parse-error noise."""

    def test_trafilatura_logger_is_critical(self):
        """trafilatura logger must be CRITICAL or above after importing html_extractor."""
        pytest_importorskip_trafilatura()

        import headroom.transforms.html_extractor  # noqa: F401

        level = logging.getLogger("trafilatura").level
        assert level >= logging.CRITICAL, (
            f"Expected trafilatura logger level >= CRITICAL ({logging.CRITICAL}), got {level}"
        )


def pytest_importorskip_trafilatura():
    """Skip test if trafilatura is not installed."""
    try:
        import trafilatura  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("trafilatura not installed")
