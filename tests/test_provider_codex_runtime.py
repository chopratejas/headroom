from __future__ import annotations

from headroom.providers.codex import build_launch_env, proxy_base_url


def test_codex_proxy_base_url_and_launch_env() -> None:
    env, lines = build_launch_env(9999, {"OPENAI_API_KEY": "sk-test"})

    assert proxy_base_url(9999) == "http://127.0.0.1:9999/v1"
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert lines == ["OPENAI_BASE_URL=http://127.0.0.1:9999/v1"]
