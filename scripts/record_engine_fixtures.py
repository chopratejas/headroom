"""Seed engine-contract parity fixtures.

Usage:
    .venv/bin/python scripts/record_engine_fixtures.py

Writes passthrough fixtures to tests/parity/fixtures/engine_contract/.
Compression fixtures are deferred to Chunk 4 (see the README in that dir).

The script is idempotent — existing fixtures are not overwritten unless
--overwrite is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on the path when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from headroom.engine.contract import Flavor, Provider  # noqa: E402
from headroom.engine.session import derive_session_key  # noqa: E402
from tests.parity.engine_recorder import PassthroughSpec, record_engine_fixture  # noqa: E402


def _body(messages: list[dict] | None = None, model: str = "claude-3-5-sonnet-20241022") -> bytes:
    msgs = messages if messages is not None else [{"role": "user", "content": "Hello"}]
    return json.dumps({"messages": msgs, "model": model}).encode()


def _session(credential: str = "test-cred", scope: str = "conv-1") -> str:
    return derive_session_key(
        credential=credential,
        conversation_scope=scope,
        salt=b"parity-salt",
    )


# ---------------------------------------------------------------------------
# Passthrough fixture specs
# ---------------------------------------------------------------------------
# Coverage matrix:
#   provider  × flavor   × auth_mode   × passthrough_reason
#   -------- ----------- ------------- ---------------------
#   anthropic  messages    payg          bypass_header (x-headroom-bypass: true)
#   anthropic  messages    oauth         bypass_header (Bearer sk-ant-oat-*)
#   anthropic  messages    subscription  bypass_header (claude-cli UA)
#   openai     chat        payg          bypass_header (sk- Bearer)
#   gemini     generate    payg          bypass_header (x-goog-api-key)
#   anthropic  messages    payg          compression_disabled (optimize=False)
#   openai     chat        payg          compression_disabled (optimize=False)
#   anthropic  messages    payg          no_messages (empty messages list)
#   openai     chat        oauth         no_messages (x-headroom-mode: passthrough)

_PASSTHROUGH_SPECS: list[PassthroughSpec] = [
    # --- bypass_header via x-headroom-bypass: true -------------------------
    PassthroughSpec(
        name="anthropic_messages_payg_bypass_header",
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers={
            "x-api-key": "sk-ant-api-test-key",
            "x-headroom-bypass": "true",
            "content-type": "application/json",
        },
        raw_body=_body(),
        session_key=_session("sk-ant-api-test-key", "conv-payg-1"),
        optimize=True,
        expected_reason="bypass_header",
    ),
    # --- bypass_header via x-headroom-mode: passthrough --------------------
    PassthroughSpec(
        name="anthropic_messages_oauth_bypass_mode_passthrough",
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers={
            "authorization": "Bearer sk-ant-oat-01-abcdef1234567890",
            "x-headroom-mode": "passthrough",
            "content-type": "application/json",
        },
        raw_body=_body(),
        session_key=_session("sk-ant-oat-01-abcdef1234567890", "conv-oauth-1"),
        optimize=True,
        expected_reason="bypass_header",
    ),
    # --- bypass_header with subscription UA --------------------------------
    PassthroughSpec(
        name="anthropic_messages_subscription_bypass_header",
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers={
            "authorization": "Bearer sk-ant-api-sub-key",
            "user-agent": "claude-cli/1.2.3",
            "x-headroom-bypass": "true",
            "content-type": "application/json",
        },
        raw_body=_body(),
        session_key=_session("sk-ant-api-sub-key", "conv-sub-1"),
        optimize=True,
        expected_reason="bypass_header",
    ),
    # --- OpenAI CHAT passthrough -------------------------------------------
    PassthroughSpec(
        name="openai_chat_payg_bypass_header",
        provider=Provider.OPENAI,
        flavor=Flavor.CHAT,
        headers={
            "authorization": "Bearer sk-openai-test-key",
            "x-headroom-bypass": "true",
            "content-type": "application/json",
        },
        raw_body=_body(model="gpt-4o"),
        session_key=_session("sk-openai-test-key", "conv-oai-1"),
        optimize=True,
        expected_reason="bypass_header",
    ),
    # --- Gemini passthrough (x-goog-api-key PAYG) --------------------------
    PassthroughSpec(
        name="gemini_generate_payg_bypass_header",
        provider=Provider.GEMINI,
        flavor=Flavor.GENERATE,
        headers={
            "x-goog-api-key": "AIza-test-gemini-key",
            "x-headroom-bypass": "true",
            "content-type": "application/json",
        },
        raw_body=_body(model="gemini-2.0-flash"),
        session_key=_session("AIza-test-gemini-key", "conv-gemini-1"),
        optimize=True,
        expected_reason="bypass_header",
    ),
    # --- compression_disabled (config.optimize=False) ----------------------
    PassthroughSpec(
        name="anthropic_messages_payg_compression_disabled",
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers={
            "x-api-key": "sk-ant-api-test-key",
            "content-type": "application/json",
        },
        raw_body=_body(),
        session_key=_session("sk-ant-api-test-key", "conv-payg-2"),
        optimize=False,  # engine built with optimize=False
        expected_reason="compression_disabled",
    ),
    PassthroughSpec(
        name="openai_chat_oauth_compression_disabled",
        provider=Provider.OPENAI,
        flavor=Flavor.CHAT,
        headers={
            # JWT-shaped bearer → OAUTH
            "authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.sig",
            "content-type": "application/json",
        },
        raw_body=_body(model="gpt-4o"),
        session_key=_session("jwt-oauth-user1", "conv-oauth-oai-1"),
        optimize=False,
        expected_reason="compression_disabled",
    ),
    # --- no_messages -------------------------------------------------------
    PassthroughSpec(
        name="anthropic_messages_payg_no_messages",
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers={
            "x-api-key": "sk-ant-api-test-key",
            "content-type": "application/json",
        },
        raw_body=_body(messages=[]),  # empty messages list
        session_key=_session("sk-ant-api-test-key", "conv-payg-3"),
        optimize=True,
        expected_reason="no_messages",
    ),
    PassthroughSpec(
        name="openai_chat_subscription_no_messages",
        provider=Provider.OPENAI,
        flavor=Flavor.CHAT,
        headers={
            "authorization": "Bearer sk-openai-test-key",
            "user-agent": "codex-cli/0.5.0",
            "content-type": "application/json",
        },
        raw_body=_body(messages=[], model="gpt-4o"),
        session_key=_session("sk-openai-test-key", "conv-sub-oai-1"),
        optimize=True,
        expected_reason="no_messages",
    ),
]


def main(*, overwrite: bool = False) -> None:
    print(f"Recording {len(_PASSTHROUGH_SPECS)} passthrough engine-contract fixtures...")
    for spec in _PASSTHROUGH_SPECS:
        path = record_engine_fixture(spec, overwrite=overwrite)
        status = "wrote" if overwrite or not path.stat().st_size == 0 else "skipped"
        print(f"  [{spec.name}] → {path.name} ({status})")

    print("Done. Compression fixtures are deferred to Chunk 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed engine-contract parity fixtures.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fixtures.")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
