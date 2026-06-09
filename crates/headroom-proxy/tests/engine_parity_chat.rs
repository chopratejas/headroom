//! D2 — cross-language engine parity for OpenAI Chat Completions
//! (`/v1/chat/completions`).
//!
//! Same bar as the responses harness (`engine_parity_responses.rs`): the Rust
//! engine reassembles via byte-range surgery (cache-safe — untouched bytes kept
//! verbatim) while the Python engine re-serializes with `json.dumps`, so equal-
//! quality outputs are never byte-equal. We assert what matters — cache-safety
//! + quality + correctness — not byte-identity.
//!
//! Mirrors Python's `handle_openai_chat` compression gate, INCLUDING the
//! `should_skip_compression` rules (n>1 etc.), minus the token-mode
//! compression-cache / CCR / memory: the Rust live-zone is cache-safe by
//! construction, so the Python "cache state machine" is largely superseded.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use headroom_core::auth_mode::classify as classify_auth_mode;
use headroom_proxy::compression::{compress_openai_chat_request, should_skip_compression, Outcome};
use headroom_proxy::config::CompressionMode;
use http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("fixtures")
        .join("engine_request_golden_openai")
}

fn build_headers(map: &serde_json::Map<String, Value>) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (key, value) in map {
        let Some(s) = value.as_str() else { continue };
        let (Ok(name), Ok(val)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(s),
        ) else {
            continue;
        };
        headers.insert(name, val);
    }
    headers
}

fn is_bypass(headers: &HeaderMap) -> bool {
    let truthy = |name: &str, want: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().eq_ignore_ascii_case(want))
            .unwrap_or(false)
    };
    truthy("x-headroom-bypass", "true") || truthy("x-headroom-mode", "passthrough")
}

/// Minimal Rust engine entry for `/v1/chat/completions`. Mirrors `forward_http`'s
/// chat gate: bypass → original; else mode from `optimize`; skip-rules
/// (`should_skip_compression`) → original; else compress; `Compressed` → new
/// body, otherwise original.
fn engine_on_request_chat(original: Bytes, headers: &HeaderMap, optimize: bool) -> Bytes {
    if is_bypass(headers) {
        return original;
    }
    let mode = if optimize {
        CompressionMode::LiveZone
    } else {
        CompressionMode::Off
    };
    if should_skip_compression(&original).is_skip() {
        return original;
    }
    let auth_mode = classify_auth_mode(headers);
    match compress_openai_chat_request(&original, mode, auth_mode, "d2-parity-chat") {
        Outcome::Compressed { body, .. } => body,
        _ => original,
    }
}

#[test]
fn rust_engine_chat_cache_safe_and_compresses() {
    let dir = fixtures_dir();
    let entries =
        fs::read_dir(&dir).unwrap_or_else(|e| panic!("read fixtures dir {}: {e}", dir.display()));

    let empty = serde_json::Map::new();
    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for entry in entries {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let fix: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

        if fix.get("endpoint").and_then(Value::as_str) != Some("/v1/chat/completions") {
            continue;
        }
        if fix.get("nondeterministic_flag").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        // Streaming chat injects `stream_options: {include_usage}` — a
        // streaming-setup concern of the engine entry (D2.8), not compression.
        if fix.get("streaming").and_then(Value::as_bool) == Some(true) {
            continue;
        }

        let name = fix
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let inbound = STANDARD
            .decode(fix.get("inbound_b64").and_then(Value::as_str).unwrap())
            .unwrap();
        let golden = STANDARD
            .decode(fix.get("outbound_b64").and_then(Value::as_str).unwrap())
            .unwrap();
        let headers = build_headers(
            fix.get("headers")
                .and_then(Value::as_object)
                .unwrap_or(&empty),
        );
        let optimize = fix
            .get("proxy_config")
            .and_then(|c| c.get("optimize"))
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let got = engine_on_request_chat(Bytes::from(inbound.clone()), &headers, optimize);
        checked += 1;

        if got.as_ref() == inbound.as_slice() {
            // Engine passed through (bypass / optimize off / skip rule / no-op).
            // Python passes through too, so the golden must match byte-for-byte.
            if got.as_ref() != golden.as_slice() {
                failures.push(format!(
                    "{name}: passthrough not byte-identical (engine={} golden={})",
                    got.len(),
                    golden.len()
                ));
            }
            continue;
        }

        // Engine compressed: assert correctness + cache-safety + quality
        // (NOT byte-identity — byte-surgery != json.dumps reassembly).
        let in_json: Value = serde_json::from_slice(&inbound).unwrap();
        let got_json: Value = match serde_json::from_slice(&got) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{name}: engine output is not valid JSON: {e}"));
                continue;
            }
        };
        // Cache-safety (structural): top-level keys + `model` untouched — byte
        // surgery only rewrites the live-zone message content.
        let in_keys: BTreeSet<&String> = in_json
            .as_object()
            .map(|m| m.keys().collect())
            .unwrap_or_default();
        let got_keys: BTreeSet<&String> = got_json
            .as_object()
            .map(|m| m.keys().collect())
            .unwrap_or_default();
        if in_keys != got_keys {
            failures.push(format!(
                "{name}: top-level keys changed (cache hot zone perturbed)"
            ));
        }
        if got_json.get("model") != in_json.get("model") {
            failures.push(format!("{name}: `model` field perturbed (cache hot zone)"));
        }
        // Quality: "less is ok as long as it's logical" — Rust compresses at
        // least as well as the Python golden, and strictly below the input.
        // (got != inbound here, so the engine DID change the body — compression
        // or cache-stability normalization like tool-sort, which reorders bytes
        // without shrinking. Size-reduction is therefore not required; the
        // quality bar is simply "no worse than Python".)
        if got.len() > golden.len() {
            failures.push(format!(
                "{name}: Rust ({}) larger than Python golden ({}) — quality regression",
                got.len(),
                golden.len()
            ));
        }
    }

    assert!(
        checked >= 10,
        "expected >=10 /v1/chat/completions fixtures, checked {checked}"
    );
    eprintln!(
        "D2 chat cache-safety + quality: {} ok / {} checked",
        checked - failures.len(),
        checked
    );
    assert!(
        failures.is_empty(),
        "{} chat fixtures failed cache-safety/quality:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
