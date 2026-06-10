from __future__ import annotations

from headroom.providers.codex.install import build_provider_section


def test_codex_provider_section_no_requires_openai_auth() -> None:
    """Bug 3 (#406): build_provider_section must NOT include requires_openai_auth.

    Setting requires_openai_auth on a custom [model_providers.headroom] block
    forces codex to demand OpenAI OAuth login for every headroom-routed request.
    Headroom is a local proxy — it must never carry this flag.
    """
    section = build_provider_section(port=8787, name="OpenAI via Headroom proxy")

    assert 'name = "OpenAI via Headroom proxy"' in section
    assert 'base_url = "http://127.0.0.1:8787/v1"' in section
    assert "requires_openai_auth" not in section, (
        f"requires_openai_auth must be absent from the headroom provider section; got:\n{section}"
    )
    assert "supports_websockets = true" in section
    assert 'env_key = "OPENAI_API_KEY"' not in section


def test_codex_provider_section_supports_custom_markers() -> None:
    section = build_provider_section(
        port=9100,
        name="Headroom init proxy",
        marker_start="# --- start ---",
        marker_end="# --- end ---",
    )

    assert section.startswith("# --- start ---\n")
    assert section.endswith("# --- end ---\n")
    assert 'base_url = "http://127.0.0.1:9100/v1"' in section
    assert 'env_key = "OPENAI_API_KEY"' not in section
