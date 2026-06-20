"""Encoding tolerance tests for headroom learn transcript inputs."""

from __future__ import annotations

import json
from pathlib import Path

from headroom.learn.plugins.claude import ClaudeCodePlugin
from headroom.learn.plugins.codex import CodexPlugin
from headroom.learn.plugins.gemini import GeminiPlugin


def test_codex_jsonl_scanner_tolerates_invalid_transcript_bytes(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    function_call = {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "call_id": "call-1",
            "name": "exec_command",
            "arguments": json.dumps({"cmd": "pytest"}),
        },
    }
    output_line = (
        b'{"type":"response_item","payload":{"type":"function_call_output",'
        b'"call_id":"call-1","output":"Error before invalid byte: \x9d"}}'
    )
    path.write_bytes(json.dumps(function_call).encode("utf-8") + b"\n" + output_line + b"\n")

    session = CodexPlugin(codex_dir=tmp_path)._scan_jsonl_session(path)

    assert session is not None
    assert len(session.tool_calls) == 1
    assert "invalid byte: \ufffd" in session.tool_calls[0].output


def test_claude_jsonl_scanner_tolerates_invalid_transcript_bytes(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    assistant = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"command": "pytest"},
                }
            ]
        },
    }
    user_result = (
        b'{"type":"user","message":{"content":[{"type":"tool_result",'
        b'"tool_use_id":"toolu_1","content":"Output with invalid byte: \x9d"}]}}'
    )
    path.write_bytes(json.dumps(assistant).encode("utf-8") + b"\n" + user_result + b"\n")

    session = ClaudeCodePlugin(claude_dir=tmp_path)._scan_session(path)

    assert session is not None
    assert len(session.tool_calls) == 1
    assert "invalid byte: \ufffd" in session.tool_calls[0].output


def test_gemini_json_scanner_tolerates_invalid_transcript_bytes(tmp_path: Path) -> None:
    path = tmp_path / "session-test.json"
    path.write_bytes(
        b'{"id":"session-test","messages":['
        b'{"role":"model","parts":[{"functionCall":{"name":"read_file",'
        b'"args":{"path":"README.md"}}}]},'
        b'{"role":"user","parts":[{"functionResponse":{"name":"read_file",'
        b'"response":{"output":"Output with invalid byte: \x9d"}}}]}'
        b"]}"
    )

    session = GeminiPlugin(gemini_dir=tmp_path)._scan_json_session(path)

    assert session is not None
    assert len(session.tool_calls) == 1
    assert "invalid byte: \ufffd" in session.tool_calls[0].output
