from headroom.pricing.opencode_prices import (
    OPENCODE_GO_SURFACE,
    OPENCODE_ZEN_SURFACE,
    OpenCodeReferencePricing,
    get_opencode_reference_pricing,
    pricing_surface_from_tags,
)


def test_pricing_surface_from_opencode_tags_detects_zen_and_go():
    assert (
        pricing_surface_from_tags(
            {
                "base-url": "https://opencode.ai",
                "original-path": "/zen/v1/chat/completions",
            }
        )
        == OPENCODE_ZEN_SURFACE
    )
    assert (
        pricing_surface_from_tags(
            {
                "base-url": "https://opencode.ai",
                "original-path": "/zen/go/v1/chat/completions",
            }
        )
        == OPENCODE_GO_SURFACE
    )
    assert (
        pricing_surface_from_tags(
            {
                "base-url": "https://api.openai.com",
                "original-path": "/v1/chat/completions",
            }
        )
        is None
    )


def test_opencode_reference_pricing_uses_scraped_prices_first(monkeypatch):
    import headroom.pricing.opencode_prices as opencode_prices

    opencode_prices._SCRAPE_CACHE.clear()
    monkeypatch.setattr(
        opencode_prices,
        "_fetch_docs_pricing",
        lambda surface: {
            "glm-5.2": OpenCodeReferencePricing(
                "glm-5.2",
                9.99 / 1_000_000,
                8.88 / 1_000_000,
                source="scraped_docs",
                source_url="https://opencode.ai/docs/test/",
            )
        },
    )

    pricing = get_opencode_reference_pricing("glm-5.2", OPENCODE_ZEN_SURFACE)
    assert pricing.input_per_1m == 9.99
    assert pricing.output_per_1m == 8.88
    assert pricing.source == "scraped_docs"


def test_opencode_reference_pricing_falls_back_when_scrape_has_no_model(monkeypatch):
    import headroom.pricing.opencode_prices as opencode_prices

    opencode_prices._SCRAPE_CACHE.clear()
    monkeypatch.setattr(opencode_prices, "_fetch_docs_pricing", lambda surface: {})

    assert get_opencode_reference_pricing("glm-5.2", OPENCODE_ZEN_SURFACE).input_per_1m == 1.4
    assert get_opencode_reference_pricing("glm-5.2", OPENCODE_GO_SURFACE).output_per_1m == 4.4
    assert get_opencode_reference_pricing("glm-5.2", None) is None


def test_opencode_pricing_parser_reads_docs_table():
    import headroom.pricing.opencode_prices as opencode_prices

    prices = opencode_prices._parse_pricing_table(
        """
        <table><tbody>
        <tr><td>GLM-5.2</td><td>$1.40</td><td>$4.40</td><td>$0.26</td><td>-</td></tr>
        <tr><td>Qwen3.7 Max</td><td>$2.50</td><td>$7.50</td><td>$0.50</td><td>$3.125</td></tr>
        </tbody></table>
        """,
        source_url="https://opencode.ai/docs/go/",
    )

    assert prices["glm-5.2"].input_per_1m == 1.4
    assert prices["qwen3.7-max"].cache_write_per_1m == 3.125
    assert prices["glm-5.2"].source == "scraped_docs"
