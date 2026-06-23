"""Carbon-savings estimation for the Headroom dashboard.

Headroom avoids sending tokens to LLMs (compression + CLI output filtering).
Every token we *don't* send is energy not spent on inference and therefore
carbon not emitted. This module turns "tokens saved" into an estimate of
grams of CO2e avoided.

Methodology is ported from the AI Carbon Tracker (PRISM) project. Per-token
energy figures come from Jegham et al. (2025), "How Hungry is AI?
Benchmarking Energy, Water, and Carbon Footprint of LLM Inference". Carbon
intensity is the global grid average from Ember's Global Electricity Review.

    carbon (gCO2e) = energy (kWh) x grid carbon intensity (gCO2e/kWh)

Energy uses a two-tier per-token model (a cheaper rate for the first 2000
tokens, a marginal rate beyond that), matching the source tool. Because
"tokens saved" is an aggregate across many requests rather than a single
call, the saved tokens sit at the margin, so the aggregate estimate is
dominated by the tier-2 (marginal) rate — the faithful and slightly
conservative choice.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("headroom.proxy")

# Global average grid carbon intensity (gCO2e/kWh).
# Ref: https://ember-energy.org/data/electricity-data-explorer/
CARBON_INTENSITY_G_PER_KWH = 471.0

# The per-token energy figures in the registry are benchmarked over total token
# counts (2000 / 11500) and are dominated by *generation* (decode) energy. But
# the tokens Headroom keeps out of context are overwhelmingly *input* / context
# tokens (prompt compression + CLI-output filtering). Prefill of input tokens is
# heavily parallelised and far cheaper per token than autoregressive decode, so
# pricing a saved input token at the full generation rate overstates the saving.
# We therefore scale the saved-token rate by this share. It is deliberately a
# single, conservative, clearly-labelled factor rather than a precise model:
# saving an input token still has a real cost (prefill compute + a marginally
# longer KV cache on every later decode step), just well below a decode token.
# Tunable; 0.3 is a conservative midpoint of published prefill/decode ratios.
SAVED_TOKEN_ENERGY_SHARE = 0.3

# Model family keywords, used as a version-agnostic fallback so that any current
# or future "claude-<...>-<family>-<...>" id still resolves to a sensible rate
# instead of being silently dropped. Order is not significant here.
_FAMILIES: tuple[str, ...] = ("opus", "sonnet", "haiku")

# Tier boundary and the two measurement points (in tokens) at which the
# source paper sampled energy. The raw JSON value for tier 1 is total Wh at
# 2000 tokens; for tier 2 it is total Wh at 11500 tokens. Dividing by these
# gives a per-token Wh rate, exactly as the source tool does.
_TIER1_LIMIT = 2000
_TIER1_DIVISOR = 2000
_TIER2_DIVISOR = 11500

_REGISTRY_PATH = Path(__file__).with_name("model_energy.json")

# Ordered most-specific-first so substring matching resolves the right model
# (e.g. "claude-opus-4.1" before the generic "claude-opus"). Mirrors the
# registry ordering in the source project's convert.ts.
_MODEL_ORDER: tuple[str, ...] = (
    # OpenAI o-series
    "o4-mini-high", "o3-pro", "o3-mini-high", "o3-mini", "o3-medium",
    "o1-medium", "o3", "o1", "o4",
    # GPT-5
    "gpt-5-mini-high", "gpt-5-mini-medium", "gpt-5-nano-high",
    "gpt-5-nano-medium", "gpt-5-nano-minimal", "gpt-5-minimal", "gpt-5-high",
    "gpt-5-medium", "gpt-5-low", "gpt-5-mini", "gpt-5",
    # GPT-4
    "gpt-4-turbo", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
    "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13",
    "gpt-4o-mini", "gpt-4o",
    # Anthropic Claude (most-specific-first; version infixes like "3-5" must
    # precede their plain "3" forms, and all specific keys precede generics).
    "claude-haiku-4.5", "claude-opus-4.1", "claude-sonnet-4.5",
    "claude-sonnet-4", "claude-opus-4", "claude-fable-5",
    "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus",
    "claude-3-sonnet", "claude-3-haiku", "claude-haiku-3",
    "claude-sonnet", "claude-haiku", "claude-opus",
    # Google Gemini
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro", "gemini-3-flash",
)

# Default model used when the request mix is unknown. Headroom traffic is
# Claude-heavy (Claude Code), so a mid-tier Claude rate is a reasonable prior.
_DEFAULT_MODEL = "claude-sonnet-4.5"


class _TieredModel:
    """Two-tier per-token energy model. Rates are in Wh per token."""

    __slots__ = ("name", "rate1_wh", "rate2_wh")

    def __init__(self, name: str, rate1_wh: float, rate2_wh: float) -> None:
        self.name = name
        self.rate1_wh = rate1_wh
        self.rate2_wh = rate2_wh

    def energy_wh(self, tokens: int) -> float:
        """Energy in watt-hours for ``tokens`` (tiered, per source tool)."""
        if tokens <= 0:
            return 0.0
        tier1 = min(tokens, _TIER1_LIMIT)
        surplus = max(0, tokens - _TIER1_LIMIT)
        return (tier1 * self.rate1_wh) + (surplus * self.rate2_wh)

    @property
    def marginal_wh_per_token(self) -> float:
        """Per-token Wh beyond the tier-1 window — the marginal rate."""
        return self.rate2_wh


@lru_cache(maxsize=1)
def _registry() -> dict[str, _TieredModel]:
    """Load and cache the model -> energy registry from JSON."""
    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        logger.warning("carbon: failed to load model energy registry: %s", exc)
        return {}

    registry: dict[str, _TieredModel] = {}
    keys = [k for k in _MODEL_ORDER if k in raw]
    # Include any models present in JSON but missing from the order list,
    # appended last so they don't shadow more-specific keys.
    keys += [k for k in raw if k not in registry and k not in _MODEL_ORDER]
    for key in keys:
        entry = raw.get(key)
        if not isinstance(entry, dict):
            continue
        tiers = entry.get("tiers") or []
        if len(tiers) < 2:
            continue
        try:
            rate1 = float(tiers[0]["energyPerToken"]) / _TIER1_DIVISOR
            rate2 = float(tiers[1]["energyPerToken"]) / _TIER2_DIVISOR
        except (KeyError, TypeError, ValueError):
            continue
        registry[key] = _TieredModel(str(entry.get("name", key)), rate1, rate2)
    return registry


def _family_of(dashed_name: str) -> str | None:
    """Return the model family (opus/sonnet/haiku) found in a name, if any."""
    for family in _FAMILIES:
        if family in dashed_name:
            return family
    return None


def get_model(name: str | None) -> _TieredModel | None:
    """Resolve a model name to an energy model.

    Resolution order, most precise first:
      1. exact match (with dot/dash normalisation),
      2. substring match against registry keys (most-specific-first),
      3. version-agnostic family fallback (opus/sonnet/haiku -> generic key).

    Step 3 is what keeps version *infix* ids such as ``claude-3-5-sonnet`` or a
    brand-new ``claude-sonnet-9`` from falling through: substring matching fails
    on them (the ``-3-5-`` / ``-9`` breaks the ``claude-sonnet`` run), so we fall
    back to the family's generic rate rather than dropping the request.
    """
    if not name:
        return None
    normalised = name.strip().lower()
    if not normalised:
        return None
    # Real provider IDs use dashes for versions (claude-sonnet-4-5-...) while
    # the registry keys use dots (claude-sonnet-4.5). Compare with dots
    # collapsed to dashes so versioned IDs match their specific tier instead
    # of falling back to the generic model rate.
    dashed = normalised.replace(".", "-")
    registry = _registry()
    for key, model in registry.items():
        kl = key.lower()
        if kl == normalised or kl.replace(".", "-") == dashed:
            return model
    # Ordered most-specific-first, so the first substring hit is best.
    for key, model in registry.items():
        kl = key.lower()
        if kl in normalised or kl.replace(".", "-") in dashed:
            return model
    # Family fallback: any opus/sonnet/haiku id resolves to its generic rate.
    family = _family_of(dashed)
    if family is not None:
        generic = registry.get(f"claude-{family}")
        if generic is not None:
            return generic
        for key, model in registry.items():
            if family in key.lower():
                return model
    return None


def energy_kwh(model_name: str, tokens: int) -> float:
    """Energy in kWh for ``tokens`` on ``model_name`` (0 if unknown)."""
    model = get_model(model_name)
    if model is None:
        return 0.0
    return model.energy_wh(tokens) / 1000.0


def calculate_emission_g(model_name: str, tokens: int) -> float:
    """Carbon emissions in grams CO2e for ``tokens`` on ``model_name``."""
    return energy_kwh(model_name, tokens) * CARBON_INTENSITY_G_PER_KWH


def _marginal_g_per_token(model: _TieredModel) -> float:
    """Marginal gCO2e per token for a model (tier-2 rate x intensity)."""
    return model.marginal_wh_per_token / 1000.0 * CARBON_INTENSITY_G_PER_KWH


def _saved_g_per_token(model: _TieredModel) -> float:
    """Effective gCO2e per *saved* token (marginal rate x input-token share)."""
    return _marginal_g_per_token(model) * SAVED_TOKEN_ENERGY_SHARE


def _weighted_g_per_token(
    requests_by_model: dict[str, int],
) -> tuple[float, list[dict[str, Any]], int]:
    """Request-weighted saved-token gCO2e plus a per-model breakdown.

    Returns ``(factor, model_mix, unpriced_requests)``. Models that cannot be
    resolved are *not* dropped: they are priced at the default-model rate and
    flagged ``estimated: True`` in the mix so the figure stays honest and the
    unrecognised traffic is visible rather than silently excluded.
    """
    default = get_model(_DEFAULT_MODEL)
    default_g = _saved_g_per_token(default) if default else 0.0

    mix: list[dict[str, Any]] = []
    total_requests = 0
    weighted_sum = 0.0
    unpriced_requests = 0
    for raw_name, count in requests_by_model.items():
        if count <= 0:
            continue
        model = get_model(raw_name)
        if model is None:
            g_per_token = default_g
            matched_name = f"{_DEFAULT_MODEL} (est.)"
            estimated = True
            unpriced_requests += count
        else:
            g_per_token = _saved_g_per_token(model)
            matched_name = model.name
            estimated = False
        weighted_sum += g_per_token * count
        total_requests += count
        mix.append(
            {
                "model": raw_name,
                "matched": matched_name,
                "requests": count,
                "g_per_1k_tokens": round(g_per_token * 1000, 4),
                "estimated": estimated,
            }
        )

    factor = weighted_sum / total_requests if total_requests > 0 else default_g

    mix.sort(key=lambda item: item["requests"], reverse=True)
    return factor, mix, unpriced_requests


# Real-world equivalency factors, for making the number tangible.
_G_PER_PHONE_CHARGE = 8.22  # EPA: smartphone charged, gCO2e
_G_PER_KM_DRIVEN = 251.0  # EPA avg passenger vehicle, ~404 g/mile -> g/km
_G_SEQUESTERED_PER_TREE_YEAR = 21770.0  # EPA: one urban tree, gCO2e/yr


def _equivalents(grams: float) -> dict[str, float]:
    return {
        "phone_charges": round(grams / _G_PER_PHONE_CHARGE, 1) if grams else 0.0,
        "km_driven": round(grams / _G_PER_KM_DRIVEN, 2) if grams else 0.0,
        "tree_years": round(grams / _G_SEQUESTERED_PER_TREE_YEAR, 3) if grams else 0.0,
    }


def build_carbon_stats(
    *,
    session_tokens_saved: int,
    lifetime_tokens_saved: int,
    requests_by_model: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the ``carbon`` block for the ``/stats`` payload.

    Estimates grams of CO2e avoided from tokens Headroom kept out of model
    context, using a request-weighted marginal emission factor across the
    models actually seen this session.
    """
    session_tokens_saved = max(0, int(session_tokens_saved or 0))
    lifetime_tokens_saved = max(0, int(lifetime_tokens_saved or 0))
    factor, model_mix, unpriced_requests = _weighted_g_per_token(requests_by_model or {})

    session_grams = round(session_tokens_saved * factor, 2)
    lifetime_grams = round(lifetime_tokens_saved * factor, 2)

    return {
        "available": (session_grams > 0 or lifetime_grams > 0),
        "session_grams": session_grams,
        "lifetime_grams": lifetime_grams,
        "session_kg": round(session_grams / 1000.0, 4),
        "lifetime_kg": round(lifetime_grams / 1000.0, 4),
        "session_tokens_saved": session_tokens_saved,
        "lifetime_tokens_saved": lifetime_tokens_saved,
        "carbon_intensity_g_per_kwh": CARBON_INTENSITY_G_PER_KWH,
        "effective_g_per_1k_tokens": round(factor * 1000, 4),
        "input_token_energy_share": SAVED_TOKEN_ENERGY_SHARE,
        "unpriced_requests": unpriced_requests,
        "model_mix": model_mix[:8],
        # Equivalents for each scope so the live dashboard can show the session
        # and the historical view can show lifetime. "equivalents" stays as the
        # lifetime figure for backward compatibility.
        "session_equivalents": _equivalents(session_grams),
        "lifetime_equivalents": _equivalents(lifetime_grams),
        "equivalents": _equivalents(lifetime_grams or session_grams),
        "methodology": (
            "Energy per token from Jegham et al. (2025), 'How Hungry is AI?'; "
            "grid carbon intensity 471 gCO2e/kWh (Ember global average). Saved "
            f"tokens are scaled to {int(SAVED_TOKEN_ENERGY_SHARE * 100)}% of the "
            "generation-token rate because they are mostly input/context tokens "
            "(cheaper prefill). Estimate of emissions avoided by tokens Headroom "
            "kept out of model context."
        ),
    }
