"""Tests for carbon-savings estimation (headroom.proxy.carbon)."""

from __future__ import annotations

import pytest

from headroom.proxy import carbon


def test_calculate_emission_matches_source_methodology():
    # Claude Opus 4.1, 3000 tokens. Tier rates from model_energy.json:
    #   tier1 = 6.97 / 2000 Wh/token over the first 2000 tokens
    #   tier2 = 13.22 / 11500 Wh/token over the remaining 1000 tokens
    # energy_wh = 2000*(6.97/2000) + 1000*(13.22/11500)
    energy_wh = 2000 * (6.97 / 2000) + 1000 * (13.22 / 11500)
    expected_g = energy_wh / 1000 * carbon.CARBON_INTENSITY_G_PER_KWH
    assert carbon.calculate_emission_g("claude-opus-4.1", 3000) == pytest.approx(expected_g)


def test_versioned_provider_ids_match_specific_tier():
    # Real provider IDs use dashes and date suffixes; they must resolve to the
    # specific model, not fall back to the generic Claude rate.
    assert carbon.get_model("claude-sonnet-4-5-20250929").name == "Claude Sonnet 4.5"
    assert carbon.get_model("claude-opus-4-1-20250805").name == "Claude Opus 4.1"
    assert carbon.get_model("gpt-4o-2024-11-20").name == "GPT4o November"


def test_unknown_model_returns_zero_emission():
    assert carbon.get_model("totally-made-up-model") is None
    assert carbon.calculate_emission_g("totally-made-up-model", 5000) == 0.0


def test_build_carbon_stats_weights_by_request_mix():
    stats = carbon.build_carbon_stats(
        session_tokens_saved=100_000,
        lifetime_tokens_saved=1_000_000,
        requests_by_model={"claude-sonnet-4-5": 90, "gpt-4o": 10},
    )
    assert stats["available"] is True
    assert stats["session_grams"] > 0
    assert stats["lifetime_grams"] > stats["session_grams"]
    # Carbon is linear in tokens given a fixed mix.
    factor = stats["effective_g_per_1k_tokens"] / 1000
    assert stats["lifetime_grams"] == pytest.approx(1_000_000 * factor, rel=1e-3)
    # Equivalents are derived from the larger (lifetime) figure.
    assert stats["equivalents"]["phone_charges"] > 0
    assert {"model", "matched", "requests", "g_per_1k_tokens"} <= set(stats["model_mix"][0])


def test_build_carbon_stats_falls_back_when_mix_unknown():
    # Empty / unrecognised mix still yields a non-zero factor (default model)
    # so carbon shows rather than silently reading zero.
    stats = carbon.build_carbon_stats(
        session_tokens_saved=50_000,
        lifetime_tokens_saved=50_000,
        requests_by_model={},
    )
    assert stats["effective_g_per_1k_tokens"] > 0
    assert stats["session_grams"] > 0


def test_no_savings_marks_unavailable():
    stats = carbon.build_carbon_stats(
        session_tokens_saved=0,
        lifetime_tokens_saved=0,
        requests_by_model={"claude-sonnet-4-5": 5},
    )
    assert stats["available"] is False
    assert stats["session_grams"] == 0.0
    assert stats["lifetime_grams"] == 0.0
