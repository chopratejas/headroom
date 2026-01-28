"""Test fixtures for Headroom Memory."""

from __future__ import annotations

# CRITICAL: Must be set before ANY imports that could trigger sentence_transformers
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytest

# Import httpx for type checking (will be available since it's a dependency)
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Wrap test execution to catch httpx.ReadTimeout and skip instead of fail.

    This handles flaky network timeouts that occur when:
    - HuggingFace Hub is slow during model downloads (sentence-transformers)
    - External embedding APIs timeout
    - Network connectivity issues in CI
    """
    outcome = yield

    if HTTPX_AVAILABLE and outcome.excinfo is not None:
        exc_type, exc_value, exc_tb = outcome.excinfo
        if isinstance(exc_value, httpx.ReadTimeout):
            pytest.skip("Skipped due to network timeout (flaky CI)")
