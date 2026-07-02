"""Reference pricing for OpenCode Zen/Go dashboard estimates.

These prices are display-only fallbacks for OpenCode's OpenAI-compatible
surfaces. They are intentionally separate from LiteLLM pricing so a missing
LiteLLM entry can still render useful dashboard cost estimates without turning
reference prices into budget-enforcement data.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from html import unescape

SOURCE_URL = "https://opencode.ai/docs/zen/"
GO_SOURCE_URL = "https://opencode.ai/docs/es/go/"
LAST_VERIFIED = "2026-07-02"

OPENCODE_ZEN_SURFACE = "opencode-zen"
OPENCODE_GO_SURFACE = "opencode-go"
OPENCODE_SURFACES = {OPENCODE_ZEN_SURFACE, OPENCODE_GO_SURFACE}


@dataclass(frozen=True)
class OpenCodeReferencePricing:
    model: str
    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: float | None = None
    cache_write_input_token_cost: float | None = None
    source: str = "fallback"
    source_url: str | None = None
    note: str = ""

    @property
    def input_per_1m(self) -> float:
        return self.input_cost_per_token * 1_000_000

    @property
    def output_per_1m(self) -> float:
        return self.output_cost_per_token * 1_000_000

    @property
    def cache_read_per_1m(self) -> float | None:
        if self.cache_read_input_token_cost is None:
            return None
        return self.cache_read_input_token_cost * 1_000_000

    @property
    def cache_write_per_1m(self) -> float | None:
        if self.cache_write_input_token_cost is None:
            return None
        return self.cache_write_input_token_cost * 1_000_000


def _per_token(price_per_1m: float) -> float:
    return price_per_1m / 1_000_000


_SCRAPE_TTL_SECONDS = 6 * 60 * 60
_SCRAPE_TIMEOUT_SECONDS = 1.5
_SCRAPE_CACHE: dict[str, tuple[float, dict[str, OpenCodeReferencePricing]]] = {}
_SCRAPE_LOCK = threading.Lock()


def _pricing_from_1m(
    model: str,
    input_per_1m: float,
    output_per_1m: float,
    cache_read_per_1m: float | None = None,
    cache_write_per_1m: float | None = None,
    *,
    source: str = "fallback",
    source_url: str | None = None,
    note: str = "",
) -> OpenCodeReferencePricing:
    return OpenCodeReferencePricing(
        model=model,
        input_cost_per_token=_per_token(input_per_1m),
        output_cost_per_token=_per_token(output_per_1m),
        cache_read_input_token_cost=_per_token(cache_read_per_1m)
        if cache_read_per_1m is not None
        else None,
        cache_write_input_token_cost=_per_token(cache_write_per_1m)
        if cache_write_per_1m is not None
        else None,
        source=source,
        source_url=source_url,
        note=note,
    )


def _model_id_from_label(label: str) -> str:
    normalized = unescape(label)
    normalized = re.sub(r"\([^)]*\)", "", normalized)
    normalized = normalized.replace("≤", "").replace(">", "")
    normalized = normalized.strip().lower()
    normalized = re.sub(r"[^a-z0-9.]+", "-", normalized)
    return normalized.strip("-")


def _parse_money_cell(value: str) -> float | None:
    text = unescape(value).strip()
    if text in {"", "-", "Free"}:
        return 0.0 if text == "Free" else None
    match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def _parse_pricing_table(html: str, *, source_url: str) -> dict[str, OpenCodeReferencePricing]:
    prices: dict[str, OpenCodeReferencePricing] = {}
    for row in re.findall(r"<tr>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.IGNORECASE | re.DOTALL)
        if len(cells) < 4:
            continue
        cleaned = [
            re.sub(r"<[^>]+>", "", cell, flags=re.IGNORECASE | re.DOTALL).strip()
            for cell in cells[:5]
        ]
        model_label = unescape(cleaned[0]).strip()
        if not model_label:
            continue
        input_price = _parse_money_cell(cleaned[1])
        output_price = _parse_money_cell(cleaned[2])
        cache_read = _parse_money_cell(cleaned[3])
        cache_write = _parse_money_cell(cleaned[4]) if len(cleaned) >= 5 else None
        if input_price is None or output_price is None:
            continue
        model_id = _model_id_from_label(model_label)
        if not model_id or model_id == "model":
            continue
        # Keep the first row for tiered models, which is the lower context tier
        # and matches the local fallback convention.
        prices.setdefault(
            model_id,
            _pricing_from_1m(
                model_id,
                input_price,
                output_price,
                cache_read,
                cache_write,
                source="scraped_docs",
                source_url=source_url,
                note="scraped from OpenCode docs",
            ),
        )
    return prices


def _fetch_docs_pricing(pricing_surface: str) -> dict[str, OpenCodeReferencePricing]:
    source_url = GO_SOURCE_URL if pricing_surface == OPENCODE_GO_SURFACE else SOURCE_URL
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "headroom-proxy/opencode-pricing"},
    )
    with urllib.request.urlopen(request, timeout=_SCRAPE_TIMEOUT_SECONDS) as response:
        html = response.read().decode("utf-8", errors="replace")
    return _parse_pricing_table(html, source_url=source_url)


def _scraped_prices(pricing_surface: str) -> dict[str, OpenCodeReferencePricing]:
    if pricing_surface not in OPENCODE_SURFACES:
        return {}
    now = time.monotonic()
    with _SCRAPE_LOCK:
        cached = _SCRAPE_CACHE.get(pricing_surface)
        if cached and now - cached[0] < _SCRAPE_TTL_SECONDS:
            return cached[1]
        try:
            prices = _fetch_docs_pricing(pricing_surface)
        except Exception:
            prices = {}
        _SCRAPE_CACHE[pricing_surface] = (now, prices)
        return prices


# Prices from OpenCode Zen docs, USD per 1M tokens. Tiered context-window rows
# use the lower tier for a stable dashboard reference.
_ZEN_REFERENCE_PRICES: dict[str, OpenCodeReferencePricing] = {
    "big-pickle": OpenCodeReferencePricing("big-pickle", _per_token(0.00), _per_token(0.00)),
    "deepseek-v4-flash-free": OpenCodeReferencePricing(
        "deepseek-v4-flash-free", _per_token(0.00), _per_token(0.00)
    ),
    "mimo-v2.5-free": OpenCodeReferencePricing(
        "mimo-v2.5-free", _per_token(0.00), _per_token(0.00)
    ),
    "north-mini-code-free": OpenCodeReferencePricing(
        "north-mini-code-free", _per_token(0.00), _per_token(0.00)
    ),
    "nemotron-3-ultra-free": OpenCodeReferencePricing(
        "nemotron-3-ultra-free", _per_token(0.00), _per_token(0.00)
    ),
    "minimax-m3": OpenCodeReferencePricing(
        "minimax-m3", _per_token(0.30), _per_token(1.20), _per_token(0.06)
    ),
    "minimax-m2.7": OpenCodeReferencePricing(
        "minimax-m2.7", _per_token(0.30), _per_token(1.20), _per_token(0.06)
    ),
    "minimax-m2.5": OpenCodeReferencePricing(
        "minimax-m2.5", _per_token(0.30), _per_token(1.20), _per_token(0.06)
    ),
    "glm-5.2": OpenCodeReferencePricing(
        "glm-5.2", _per_token(1.40), _per_token(4.40), _per_token(0.26)
    ),
    "glm-5.1": OpenCodeReferencePricing(
        "glm-5.1", _per_token(1.40), _per_token(4.40), _per_token(0.26)
    ),
    "glm-5": OpenCodeReferencePricing(
        "glm-5", _per_token(1.00), _per_token(3.20), _per_token(0.20)
    ),
    "kimi-k2.7-code": OpenCodeReferencePricing(
        "kimi-k2.7-code", _per_token(0.95), _per_token(4.00), _per_token(0.19)
    ),
    "kimi-k2.6": OpenCodeReferencePricing(
        "kimi-k2.6", _per_token(0.95), _per_token(4.00), _per_token(0.16)
    ),
    "kimi-k2.5": OpenCodeReferencePricing(
        "kimi-k2.5", _per_token(0.60), _per_token(3.00), _per_token(0.10)
    ),
    "qwen3.7-max": OpenCodeReferencePricing(
        "qwen3.7-max",
        _per_token(2.50),
        _per_token(7.50),
        _per_token(0.50),
        _per_token(3.125),
    ),
    "qwen3.7-plus": OpenCodeReferencePricing(
        "qwen3.7-plus",
        _per_token(0.40),
        _per_token(1.60),
        _per_token(0.04),
        _per_token(0.50),
        note="lower tier (<=256k context)",
    ),
    "qwen3.6-plus": OpenCodeReferencePricing(
        "qwen3.6-plus",
        _per_token(0.50),
        _per_token(3.00),
        _per_token(0.05),
        _per_token(0.625),
        note="lower tier (<=256k context)",
    ),
    "qwen3.5-plus": OpenCodeReferencePricing(
        "qwen3.5-plus",
        _per_token(0.20),
        _per_token(1.20),
        _per_token(0.02),
        _per_token(0.25),
    ),
    "deepseek-v4-pro": OpenCodeReferencePricing(
        "deepseek-v4-pro", _per_token(1.74), _per_token(3.48), _per_token(0.145)
    ),
    "deepseek-v4-flash": OpenCodeReferencePricing(
        "deepseek-v4-flash", _per_token(0.14), _per_token(0.28), _per_token(0.028)
    ),
    "grok-build-0.1": OpenCodeReferencePricing(
        "grok-build-0.1", _per_token(1.00), _per_token(2.00), _per_token(0.20)
    ),
    "gpt-5.5": OpenCodeReferencePricing(
        "gpt-5.5",
        _per_token(5.00),
        _per_token(30.00),
        _per_token(0.50),
        note="lower tier (<=272k context)",
    ),
    "gpt-5.5-pro": OpenCodeReferencePricing(
        "gpt-5.5-pro", _per_token(30.00), _per_token(180.00), _per_token(30.00)
    ),
    "gpt-5.4": OpenCodeReferencePricing(
        "gpt-5.4",
        _per_token(2.50),
        _per_token(15.00),
        _per_token(0.25),
        note="lower tier (<=272k context)",
    ),
    "gpt-5.4-pro": OpenCodeReferencePricing(
        "gpt-5.4-pro", _per_token(30.00), _per_token(180.00), _per_token(30.00)
    ),
    "gpt-5.4-mini": OpenCodeReferencePricing(
        "gpt-5.4-mini", _per_token(0.75), _per_token(4.50), _per_token(0.075)
    ),
    "gpt-5.4-nano": OpenCodeReferencePricing(
        "gpt-5.4-nano", _per_token(0.20), _per_token(1.25), _per_token(0.02)
    ),
    "gpt-5.3-codex-spark": OpenCodeReferencePricing(
        "gpt-5.3-codex-spark", _per_token(1.75), _per_token(14.00), _per_token(0.175)
    ),
    "gpt-5.3-codex": OpenCodeReferencePricing(
        "gpt-5.3-codex", _per_token(1.75), _per_token(14.00), _per_token(0.175)
    ),
    "gpt-5.2": OpenCodeReferencePricing(
        "gpt-5.2", _per_token(1.75), _per_token(14.00), _per_token(0.175)
    ),
    "gpt-5.2-codex": OpenCodeReferencePricing(
        "gpt-5.2-codex", _per_token(1.75), _per_token(14.00), _per_token(0.175)
    ),
    "gpt-5.1": OpenCodeReferencePricing(
        "gpt-5.1", _per_token(1.07), _per_token(8.50), _per_token(0.107)
    ),
    "gpt-5.1-codex": OpenCodeReferencePricing(
        "gpt-5.1-codex", _per_token(1.07), _per_token(8.50), _per_token(0.107)
    ),
    "gpt-5.1-codex-max": OpenCodeReferencePricing(
        "gpt-5.1-codex-max", _per_token(1.25), _per_token(10.00), _per_token(0.125)
    ),
    "gpt-5.1-codex-mini": OpenCodeReferencePricing(
        "gpt-5.1-codex-mini", _per_token(0.25), _per_token(2.00), _per_token(0.025)
    ),
    "gpt-5": OpenCodeReferencePricing(
        "gpt-5", _per_token(1.07), _per_token(8.50), _per_token(0.107)
    ),
    "gpt-5-codex": OpenCodeReferencePricing(
        "gpt-5-codex", _per_token(1.07), _per_token(8.50), _per_token(0.107)
    ),
    "gpt-5-nano": OpenCodeReferencePricing(
        "gpt-5-nano", _per_token(0.05), _per_token(0.40), _per_token(0.005)
    ),
}

# OpenCode Go is subscription-backed, but its docs publish the dollar-equivalent
# prices used for limits. These override Zen where Go differs.
_GO_REFERENCE_OVERRIDES: dict[str, OpenCodeReferencePricing] = {
    "mimo-v2.5": OpenCodeReferencePricing(
        "mimo-v2.5", _per_token(0.14), _per_token(0.28), _per_token(0.0028)
    ),
    "mimo-v2.5-pro": OpenCodeReferencePricing(
        "mimo-v2.5-pro", _per_token(1.74), _per_token(3.48), _per_token(0.0145)
    ),
    "deepseek-v4-pro": OpenCodeReferencePricing(
        "deepseek-v4-pro", _per_token(1.74), _per_token(3.48), _per_token(0.0145)
    ),
    "deepseek-v4-flash": OpenCodeReferencePricing(
        "deepseek-v4-flash", _per_token(0.14), _per_token(0.28), _per_token(0.0028)
    ),
    "minimax-m2.7": OpenCodeReferencePricing(
        "minimax-m2.7",
        _per_token(0.30),
        _per_token(1.20),
        _per_token(0.06),
        _per_token(0.375),
    ),
    "minimax-m2.5": OpenCodeReferencePricing(
        "minimax-m2.5",
        _per_token(0.30),
        _per_token(1.20),
        _per_token(0.06),
        _per_token(0.375),
    ),
}


def pricing_surface_from_tags(tags: dict[str, str] | None) -> str | None:
    """Detect OpenCode Zen/Go routing from request tags."""
    if not tags:
        return None
    base_url = (tags.get("base-url") or "").lower()
    original_path = (tags.get("original-path") or "").lower()
    if "opencode.ai" not in base_url:
        return None
    if "/zen/go/" in original_path or original_path.startswith("/zen/go"):
        return OPENCODE_GO_SURFACE
    if "/zen/" in original_path or original_path.startswith("/zen"):
        return OPENCODE_ZEN_SURFACE
    return None


def get_opencode_reference_pricing(
    model: str,
    pricing_surface: str | None,
) -> OpenCodeReferencePricing | None:
    """Return a reference price for OpenCode Zen/Go surfaces only."""
    if pricing_surface not in OPENCODE_SURFACES:
        return None
    model_key = model.lower()
    scraped = _scraped_prices(pricing_surface)
    if model_key in scraped:
        return scraped[model_key]
    if pricing_surface == OPENCODE_GO_SURFACE:
        return _GO_REFERENCE_OVERRIDES.get(model_key) or _ZEN_REFERENCE_PRICES.get(model_key)
    return _ZEN_REFERENCE_PRICES.get(model_key)


def reference_pricing_metadata(
    model: str,
    pricing_surface: str,
    pricing: OpenCodeReferencePricing,
) -> dict[str, object]:
    """Metadata payload surfaced in /stats so dashboards can label estimates."""
    note = (
        "Reference-only OpenCode pricing for dashboard display. "
        "OpenCode Go is subscription-backed and may not represent incremental spend; "
        "do not treat this as actual billed cost."
    )
    if pricing.note:
        note = f"{note} {pricing.note}."
    return {
        "source": "opencode_reference",
        "surface": pricing_surface,
        "reference_only": True,
        "pricing_source": pricing.source,
        "model": model,
        "input_per_1m": round(pricing.input_per_1m, 6),
        "output_per_1m": round(pricing.output_per_1m, 6),
        "cache_read_per_1m": (
            round(pricing.cache_read_per_1m, 6)
            if pricing.cache_read_per_1m is not None
            else None
        ),
        "cache_write_per_1m": (
            round(pricing.cache_write_per_1m, 6)
            if pricing.cache_write_per_1m is not None
            else None
        ),
        "source_url": pricing.source_url
        or (GO_SOURCE_URL if pricing_surface == OPENCODE_GO_SURFACE else SOURCE_URL),
        "last_verified": LAST_VERIFIED,
        "note": note,
    }
