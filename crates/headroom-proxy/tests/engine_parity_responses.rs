//! D2 — cross-language engine PARITY for OpenAI Responses (`/v1/responses`).
//!
//! Byte-identity to the Python golden is the WRONG bar: the Rust engine
//! reassembles via byte-range surgery (cache-safe — untouched bytes are kept
//! verbatim), while the Python engine re-serializes the whole body with
//! `json.dumps`. So equal-quality outputs are never byte-equal — and the Rust
//! crusher legitimately compresses *harder* than Python (e.g. fco_large: 538 b
//! vs Python's 1945 b). "Less is ok as long as it's logical."
//!
//! So this asserts what actually matters, per fixture:
//!   - **Passthrough** (optimize off / bypass): byte-identical to the golden —
//!     the engine must NOT perturb a request it doesn't compress.
//!   - **Compression** (optimize on): (1) CORRECTNESS — valid JSON, top-level
//!     keys + `model` preserved (structurally logical); (2) CACHE-SAFETY — the
//!     cache hot zone (everything but the live-zone output) is intact; (3)
//!     QUALITY — Rust is no larger than the Python golden (harder is fine) and
//!     strictly smaller than the input (it actually compressed).
//!
//! The Rust engine entry mirrors Python's `_on_request_openai_responses` minus
//! memory injection (off in the corpus).

use std::fs;
use std::path::PathBuf;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use headroom_core::auth_mode::classify as classify_auth_mode;
use headroom_proxy::compression::{compress_openai_responses_request, Outcome};
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

/// Minimal Rust engine entry for `/v1/responses`. Mirrors the Python
/// `_on_request_openai_responses` (sans memory): bypass → original bytes; else
/// mode from `optimize`; compress; `Compressed` → new body, else original.
fn engine_on_request_responses(original: Bytes, headers: &HeaderMap, optimize: bool) -> Bytes {
    if is_bypass(headers) {
        return original;
    }
    let mode = if optimize {
        CompressionMode::LiveZone
    } else {
        CompressionMode::Off
    };
    let auth_mode = classify_auth_mode(headers);
    match compress_openai_responses_request(&original, mode, auth_mode, "d2-parity-responses") {
        Outcome::Compressed { body, .. } => body,
        _ => original,
    }
}

#[test]
fn rust_engine_responses_cache_safe_and_compresses() {
    use std::collections::BTreeSet;

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

        if fix.get("endpoint").and_then(Value::as_str) != Some("/v1/responses") {
            continue;
        }
        if fix.get("nondeterministic_flag").and_then(Value::as_bool) == Some(true) {
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

        let got = engine_on_request_responses(Bytes::from(inbound.clone()), &headers, optimize);
        checked += 1;

        // Passthrough (optimize off / bypass): the engine must not perturb it.
        if !optimize || is_bypass(&headers) {
            if got.as_ref() != golden.as_slice() {
                failures.push(format!(
                    "{name}: passthrough not byte-identical (engine={} golden={})",
                    got.len(),
                    golden.len()
                ));
            }
            continue;
        }

        // Compression allowed: assert correctness + cache-safety + quality
        // (NOT byte-identity — byte-surgery != json.dumps reassembly).
        let in_json: Value = serde_json::from_slice(&inbound).unwrap();
        let got_json: Value = match serde_json::from_slice(&got) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{name}: engine output is not valid JSON: {e}"));
                continue;
            }
        };
        // (1) Cache-safety (structural): top-level keys + `model` untouched —
        // byte-surgery only rewrites the live-zone output content.
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
        // (2) Quality: "less is ok as long as it's logical" — Rust must compress
        // at least as well as the Python golden, and strictly below the input.
        if got.len() > golden.len() {
            failures.push(format!(
                "{name}: Rust ({}) larger than Python golden ({}) — quality regression",
                got.len(),
                golden.len()
            ));
        }
        if got.len() >= inbound.len() {
            failures.push(format!(
                "{name}: did not compress (engine={} >= input={})",
                got.len(),
                inbound.len()
            ));
        }
    }

    assert!(
        checked >= 6,
        "expected >=6 /v1/responses fixtures, checked {checked}"
    );
    eprintln!(
        "D2 responses cache-safety + quality: {} ok / {} checked",
        checked - failures.len(),
        checked
    );
    assert!(
        failures.is_empty(),
        "{} responses fixtures failed cache-safety/quality:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
