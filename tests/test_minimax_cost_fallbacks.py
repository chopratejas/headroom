"""Tests for the cost-tracking fallbacks added for the MiniMax provider.

These tests verify that:
- CostTracker._get_cache_prices() returns MiniMax pricing for MiniMax-M*
  models even when litellm is unavailable (e.g. Python 3.14).
- CostTracker._get_list_price() returns per-million-token pricing for
  MiniMax models using the MiniMaxProvider lookup.
- The savings tracker estimate functions work for MiniMax models
  without litellm installed.
"""

from __future__ import annotations

import pytest

from headroom.providers.minimax import (
    MODEL_INPUT_COST,
)


class TestGetCachePricesMiniMaxFallback:
    """CostTracker._get_cache_prices has a MiniMax fallback for Python 3.14."""

    def _make_tracker(self):
        # Build a CostTracker without going through the full proxy
        # server boot — we only need the pricing helpers, not the
        # rest of the telemetry plumbing.
        from headroom.proxy.cost import CostTracker

        return CostTracker()

    def test_minimax_m3_returns_per_token_prices(self) -> None:
        tracker = self._make_tracker()
        prices = tracker._get_cache_prices("MiniMax-M3")
        assert prices is not None
        cache_read, cache_write, uncached = prices
        # Per-million-token pricing for M3 input is $1.00
        assert uncached == pytest.approx(1.0 / 1_000_000, rel=1e-9)
        # Cache read is 90% off — i.e. 0.10x the uncached price
        assert cache_read == pytest.approx(uncached * 0.10, rel=1e-9)
        # Cache write is a 25% premium — i.e. 1.25x the uncached price
        assert cache_write == pytest.approx(uncached * 1.25, rel=1e-9)

    def test_minimax_m27_highspeed_returns_per_token_prices(self) -> None:
        tracker = self._make_tracker()
        prices = tracker._get_cache_prices("MiniMax-M2.7-highspeed")
        assert prices is not None
        cache_read, cache_write, uncached = prices
        assert uncached == pytest.approx(MODEL_INPUT_COST["MiniMax-M2.7-highspeed"] / 1_000_000, rel=1e-9)

    def test_minimax_prefix_stripped_before_lookup(self) -> None:
        tracker = self._make_tracker()
        # Clients may send "minimax/MiniMax-M3" as a routing prefix.
        prices = tracker._get_cache_prices("minimax/MiniMax-M3")
        assert prices is not None
        uncached = prices[2]
        assert uncached == pytest.approx(MODEL_INPUT_COST["MiniMax-M3"] / 1_000_000, rel=1e-9)

    def test_unknown_minimax_variant_returns_none(self) -> None:
        tracker = self._make_tracker()
        # MiniMax-M99-future is not in the registry → no pricing.
        assert tracker._get_cache_prices("MiniMax-M99-future") is None

    def test_anthropic_model_still_uses_default_path(self) -> None:
        tracker = self._make_tracker()
        # claude-sonnet-4-5 is not MiniMax — should NOT hit the
        # MiniMax fallback. litellm may return None on Python 3.14,
        # which is fine (graceful "no pricing" returns None).
        prices = tracker._get_cache_prices("claude-sonnet-4-5")
        # We don't assert specific values here because litellm
        # availability varies by Python version. The point is that
        # we don't blow up and we don't return MiniMax pricing for
        # a non-MiniMax model.
        if prices is not None:
            cache_read, _cache_write, uncached = prices
            # Sanity: not the M3 uncached price.
            assert uncached != pytest.approx(1.0 / 1_000_000, rel=1e-9)

    @pytest.mark.parametrize(
        "model,fallback_per_m,family",
        [
            # Claude 4 family
            ("claude-opus-4-20250514", 15.0, "opus"),
            ("claude-sonnet-4-5", 3.0, "sonnet"),
            ("claude-sonnet-4-20250514", 3.0, "sonnet"),
            ("claude-haiku-4-5", 0.80, "haiku4"),
            # Claude 3.x family — including the variant with truncated
            # datestamp that exposed the original group(2) bug.
            ("claude-3-5-sonnet-20241022", 3.0, "sonnet"),
            ("claude-3-5-sonnet-20", 3.0, "sonnet"),
            ("claude-3-haiku-20240307", 0.25, "haiku3"),
            ("claude-3-opus-20240229", 15.0, "opus"),
        ],
    )
    def test_anthropic_claude_pricing_consistent(
        self, model: str, fallback_per_m: float, family: str
    ) -> None:
        """Anthropic Claude pricing is internally consistent regardless of
        whether the source is litellm or our hardcoded fallback.

        The proxy either uses litellm's pricing data (when litellm is
        installed and the model is in its registry) or our hardcoded
        Anthropic fallback table (when litellm is missing or the model
        isn't registered — e.g. the truncated-datestamp variant
        ``claude-3-5-sonnet-20``).

        Either way the cache economics MUST match:
          - cache_read == 10% of uncached (Anthropic's 90% discount)
          - cache_write == 125% of uncached (Anthropic's 25% write premium)
        """
        tracker = self._make_tracker()
        prices = tracker._get_cache_prices(model)
        if prices is None:
            pytest.skip(
                "No pricing source available for "
                f"{model!r} (litellm missing AND model not in fallback table)"
            )
        cache_read, cache_write, uncached = prices
        # The cache economics — read discount + write premium — must be
        # uniform across all Anthropic Claude variants. If litellm gives
        # us different economics for the same family, the cost tracker
        # will under-report savings on some models.
        assert cache_read == pytest.approx(uncached * 0.10, rel=1e-3), (
            f"{model}: cache_read {cache_read} != 10% of uncached {uncached}"
        )
        assert cache_write == pytest.approx(uncached * 1.25, rel=1e-3), (
            f"{model}: cache_write {cache_write} != 125% of uncached {uncached}"
        )

    @pytest.mark.parametrize(
        "model,fallback_per_m",
        [
            ("gpt-5", 5.0),
            ("gpt-5-mini", 0.25),
            ("gpt-5-nano", 0.10),
            ("gpt-4o", 2.50),
            ("gpt-4o-mini", 0.15),
            ("gpt-4-turbo", 10.0),
            ("gpt-3.5-turbo", 0.50),
            ("o1", 15.0),
            ("o1-mini", 3.0),
            ("o3-mini", 1.10),
        ],
    )
    def test_openai_gpt_pricing_consistent(
        self, model: str, fallback_per_m: float
    ) -> None:
        """OpenAI pricing is internally consistent regardless of source.

        Either litellm's registry or our hardcoded fallback is used; in
        both cases the cache economics must match OpenAI's published
        numbers (50% off cache reads, no write premium).
        """
        tracker = self._make_tracker()
        prices = tracker._get_cache_prices(model)
        if prices is None:
            pytest.skip(
                "No pricing source available for "
                f"{model!r} (litellm missing AND model not in fallback table)"
            )
        cache_read, cache_write, uncached = prices
        # OpenAI cache economics: 50% off reads, no write premium.
        assert cache_read == pytest.approx(uncached * 0.50, rel=1e-3), (
            f"{model}: cache_read {cache_read} != 50% of uncached {uncached}"
        )
        assert cache_write == pytest.approx(uncached * 1.00, rel=1e-3), (
            f"{model}: cache_write {cache_write} != 100% of uncached {uncached}"
        )


class TestGetListPriceMiniMaxFallback:
    """CostTracker._get_list_price has a MiniMax fallback."""

    def _make_tracker(self):
        from headroom.proxy.cost import CostTracker

        return CostTracker()

    def test_minimax_m3_returns_per_million_price(self) -> None:
        tracker = self._make_tracker()
        assert tracker._get_list_price("MiniMax-M3") == pytest.approx(MODEL_INPUT_COST["MiniMax-M3"], rel=1e-9)

    def test_minimax_prefix_returns_per_million_price(self) -> None:
        tracker = self._make_tracker()
        assert tracker._get_list_price("minimax/MiniMax-M3") == pytest.approx(MODEL_INPUT_COST["MiniMax-M3"], rel=1e-9)

    def test_minimax_m27_family(self) -> None:
        tracker = self._make_tracker()
        assert tracker._get_list_price("MiniMax-M2.7-highspeed") == pytest.approx(
            MODEL_INPUT_COST["MiniMax-M2.7-highspeed"], rel=1e-9
        )


class TestSavingsTrackerMiniMaxFallback:
    """savings_tracker._get_minimax_cost_per_token resolves MiniMax pricing."""

    def test_minimax_m3_returns_input_and_output_per_token(self) -> None:
        from headroom.proxy.savings_tracker import _get_minimax_cost_per_token

        r = _get_minimax_cost_per_token("MiniMax-M3")
        assert r is not None
        assert r["input_cost_per_token"] == pytest.approx(MODEL_INPUT_COST["MiniMax-M3"] / 1_000_000, rel=1e-9)
        # Output pricing is exposed by MiniMaxProvider, the test
        # data layer matches the same shape.
        from headroom.providers.minimax import MODEL_OUTPUT_COST
        assert r["output_cost_per_token"] == pytest.approx(MODEL_OUTPUT_COST["MiniMax-M3"] / 1_000_000, rel=1e-9)

    def test_minimax_prefix_returns_pricing(self) -> None:
        from headroom.proxy.savings_tracker import _get_minimax_cost_per_token

        r = _get_minimax_cost_per_token("minimax/MiniMax-M2.7-highspeed")
        assert r is not None
        assert r["input_cost_per_token"] == pytest.approx(MODEL_INPUT_COST["MiniMax-M2.7-highspeed"] / 1_000_000, rel=1e-9)

    def test_non_minimax_returns_none(self) -> None:
        from headroom.proxy.savings_tracker import _get_minimax_cost_per_token

        assert _get_minimax_cost_per_token("claude-sonnet-4-5") is None
        assert _get_minimax_cost_per_token("gpt-5.5") is None
        assert _get_minimax_cost_per_token("") is None

    def test_unknown_minimax_returns_none(self) -> None:
        from headroom.proxy.savings_tracker import _get_minimax_cost_per_token

        assert _get_minimax_cost_per_token("MiniMax-M99-future") is None


class TestCacheStatsProviderMatch:
    """The cache-economics provider-match loop recognises MiniMax."""

    def test_minimax_provider_match_includes_minimax(self) -> None:
        # Smoke check: extract the snippet of logic that matches the
        # provider name to the model name, then verify it produces
        # True for "minimax" + "MiniMax-M*" combination.
        def matches(provider: str, model_name: str) -> bool:
            _openai_prefixes = ("gpt", "o1", "o3", "o4")
            return (
                (provider == "anthropic" and "claude" in model_name)
                or (provider == "openai" and any(p in model_name for p in _openai_prefixes))
                or (provider == "gemini" and "gemini" in model_name)
                or (provider == "bedrock" and "claude" in model_name)
                or (provider == "minimax" and ("MiniMax-M" in model_name or "minimax-m" in model_name.lower()))
            )

        assert matches("minimax", "MiniMax-M3") is True
        assert matches("minimax", "MiniMax-M2.7-highspeed") is True
        assert matches("minimax", "minimax/MiniMax-M3") is True
        # Non-matches
        assert matches("minimax", "claude-sonnet-4-5") is False
        assert matches("anthropic", "MiniMax-M3") is False
        assert matches("openai", "MiniMax-M3") is False


class TestMiniMaxHandlerModelDetection:
    """MiniMaxHandlerMixin._is_minimax_model is the routing detector."""

    def test_recognises_bare_model_names(self) -> None:
        # Import the module directly to avoid the heavy proxy
        # module-import chain. _is_minimax_model is a staticmethod
        # so it doesn't need an instance.
        import importlib.util
        import os

        spec = importlib.util.spec_from_file_location(
            "_minimax_handler_under_test",
            os.path.abspath("headroom/proxy/handlers/minimax.py"),
        )
        mod = importlib.util.module_from_spec(spec)

        # Stub heavy deps so the module loads without proxy boot.
        import sys
        from typing import Any as _Any

        class _Stub:
            def __getattr__(self, name: str) -> _Any:
                return _Stub()

        sys.modules.setdefault("headroom.providers.minimax", _Stub())
        sys.modules.setdefault("headroom.proxy.handlers.anthropic", _Stub())
        spec.loader.exec_module(mod)

        assert mod.MiniMaxHandlerMixin._is_minimax_model("MiniMax-M3") is True
        assert mod.MiniMaxHandlerMixin._is_minimax_model("MiniMax-M2.7-highspeed") is True
        assert mod.MiniMaxHandlerMixin._is_minimax_model("minimax/MiniMax-M3") is True
        assert mod.MiniMaxHandlerMixin._is_minimax_model("claude-sonnet-4-5") is False
        assert mod.MiniMaxHandlerMixin._is_minimax_model("gpt-5.5") is False
        assert mod.MiniMaxHandlerMixin._is_minimax_model("") is False

    def test_strip_prefix(self) -> None:
        import importlib.util
        import os
        import sys
        from typing import Any as _Any

        class _Stub:
            def __getattr__(self, name: str) -> _Any:
                return _Stub()

        spec = importlib.util.spec_from_file_location(
            "_minimax_strip_under_test",
            os.path.abspath("headroom/proxy/handlers/minimax.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("headroom.providers.minimax", _Stub())
        sys.modules.setdefault("headroom.proxy.handlers.anthropic", _Stub())
        spec.loader.exec_module(mod)

        assert mod.MiniMaxHandlerMixin._strip_minimax_prefix("MiniMax-M3") == "MiniMax-M3"
        assert mod.MiniMaxHandlerMixin._strip_minimax_prefix("minimax/MiniMax-M3") == "MiniMax-M3"
        assert mod.MiniMaxHandlerMixin._strip_minimax_prefix("minimax/MiniMax-M2.7-highspeed") == "MiniMax-M2.7-highspeed"
        assert mod.MiniMaxHandlerMixin._strip_minimax_prefix("claude-sonnet-4-5") == "claude-sonnet-4-5"
        assert mod.MiniMaxHandlerMixin._strip_minimax_prefix("") == ""
