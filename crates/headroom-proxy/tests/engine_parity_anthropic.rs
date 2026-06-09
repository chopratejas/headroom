//! D2 — cross-language engine parity for Anthropic Messages (`/v1/messages`).
//!
//! Same bar as the OpenAI harnesses (`engine_parity_responses.rs`,
//! `engine_parity_chat.rs`): the Rust engine reassembles via byte-range surgery
//! (cache-safe — untouched bytes kept verbatim) while the Python engine
//! re-serializes with `json.dumps`, so equal-quality outputs are never
//! byte-equal. We assert what matters — cache-safety + quality + correctness —
//! not byte-identity.
//!
//! THE HYPOTHESIS THIS HARNESS TESTS — "the Python cache state machine is
//! superseded by byte-surgery". Python's `cache`/`token` modes restore a frozen
//! prefix from `frozen_count` + `prev_forwarded_messages` (the cache state
//! machine) so prior turns keep their cache hits. The Rust live-zone path takes
//! NONE of that state: it only rewrites the live zone and splices it back into
//! the original bytes, so the frozen prefix is preserved BY CONSTRUCTION. We
//! prove it by reading the golden's recorded `frozen_count` and asserting the
//! Rust output preserves messages `[0..frozen_count]` verbatim — i.e. Rust's
//! own `cache_control`-derived frozen boundary is at least as conservative as
//! the boundary Python computed with the full state machine.
//!
//! TOOL NORMALIZATION (PAYG, optimize on): the Rust path sorts `tools[]` (E1) +
//! schema keys (E2) and auto-places a `cache_control` cache breakpoint (E3) —
//! deterministic cache-stability ops the Python golden predates (E3 grows the
//! tools block for a cache win). So we assert live-zone (`messages`) quality vs
//! golden + the tool-SET (no drops), NOT whole-body size. Passthrough (optimize
//! off) asserts purity vs the INPUT, not the golden: Python sorted tools even in
//! passthrough; the Rust engine intentionally does not mutate un-optimized
//! requests (production early-returns `ModeOff` before the Phase-E pass).
//!
//! Streaming fixtures are deferred (consistent with the chat harness) — the
//! engine-entry streaming setup is a D2.8 concern, not request compression.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use headroom_core::auth_mode::classify as classify_auth_mode;
use headroom_proxy::compression::{compress_anthropic_request, Outcome};
use headroom_proxy::config::{CacheControlAutoFrozen, CacheControlAutoPlace, CompressionMode};
use http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("fixtures")
        .join("engine_request_golden")
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

/// Minimal Rust engine entry for `/v1/messages`. Mirrors `forward_http`'s
/// anthropic gate: bypass → original; else mode from `optimize`; compress with
/// the production `cache_control_auto_frozen=Enabled` policy; `Compressed` →
/// new body, otherwise original. NOTE the absence of any frozen-prefix / prev-
/// message inputs — that is the whole point (the state machine is superseded).
fn engine_on_request_anthropic(original: Bytes, headers: &HeaderMap, optimize: bool) -> Bytes {
    if is_bypass(headers) {
        return original;
    }
    let mode = if optimize {
        CompressionMode::LiveZone
    } else {
        CompressionMode::Off
    };
    let auth_mode = classify_auth_mode(headers);
    match compress_anthropic_request(
        &original,
        mode,
        CacheControlAutoFrozen::Enabled,
        CacheControlAutoPlace::Enabled,
        auth_mode,
        "d2-parity-anthropic",
    ) {
        Outcome::Compressed { body, .. } => body,
        _ => original,
    }
}

#[test]
fn rust_engine_anthropic_cache_safe_and_compresses() {
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

        // The whole dir is `/v1/messages` (no `endpoint` field), so no endpoint
        // filter — just drop the nondeterministic + streaming fixtures.
        if fix.get("nondeterministic_flag").and_then(Value::as_bool) == Some(true) {
            continue;
        }
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
        let frozen_count = fix.get("frozen_count").and_then(Value::as_u64).unwrap_or(0) as usize;

        let got = engine_on_request_anthropic(Bytes::from(inbound.clone()), &headers, optimize);
        checked += 1;

        // ── Passthrough purity (optimize off / bypass) ───────────────
        // The Rust engine is a PURE passthrough here: it returns the input
        // bytes verbatim. NOTE this can differ from the Python golden — Python
        // sorted `tools[]` even when optimization was off, whereas the Rust
        // engine intentionally does NOT mutate bytes unless optimization is on
        // (the production path early-returns `ModeOff` before the Phase-E pass;
        // mutating an un-optimized request risks looking like cache-evasion to
        // the upstream). We assert the Rust contract — purity — not Python's
        // now-superseded passthrough-mutation behavior.
        if !optimize || is_bypass(&headers) {
            if got.as_ref() != inbound.as_slice() {
                failures.push(format!(
                    "{name}: passthrough must equal INPUT verbatim (engine={} input={})",
                    got.len(),
                    inbound.len()
                ));
            }
            continue;
        }

        // Compression path. Correctness + cache-safety + quality — NOT byte- or
        // whole-body-size identity to the Python golden: the Rust PAYG path
        // legitimately (a) sorts tools (E1) + schema keys (E2) and (b)
        // auto-places a `cache_control` cache breakpoint (E3) the Python golden
        // lacks. E3 GROWS the tools block for a deterministic cache win, so
        // whole-body size vs golden is meaningless. We isolate the compressible
        // live zone (`messages`) for the quality bar and assert structural
        // invariants on the rest.
        let in_json: Value = serde_json::from_slice(&inbound).unwrap();
        let golden_json: Value = serde_json::from_slice(&golden).unwrap();
        let got_json: Value = match serde_json::from_slice(&got) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{name}: engine output is not valid JSON: {e}"));
                continue;
            }
        };

        // (1) Cache hot zone: top-level keys + `model` + `system` untouched.
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
        if got_json.get("system") != in_json.get("system") {
            failures.push(format!(
                "{name}: `system` field perturbed (frozen prefix — cache hot zone)"
            ));
        }

        // (2) THE HYPOTHESIS: byte-surgery preserves the same frozen prefix the
        // Python state machine computed. Messages `[0..frozen_count]` must be
        // verbatim-identical and the message COUNT unchanged (the live zone
        // rewrites content, never drops a turn). If Rust compressed a frozen
        // turn, the state machine is NOT superseded → cache-safety regression.
        let in_msgs = in_json.get("messages").and_then(Value::as_array);
        let got_msgs = got_json.get("messages").and_then(Value::as_array);
        match (in_msgs, got_msgs) {
            (Some(im), Some(gm)) => {
                if im.len() != gm.len() {
                    failures.push(format!(
                        "{name}: message count changed ({} -> {}) — live-zone must rewrite content, not drop messages",
                        im.len(),
                        gm.len()
                    ));
                } else {
                    for i in 0..frozen_count.min(im.len()) {
                        if im[i] != gm[i] {
                            failures.push(format!(
                                "{name}: frozen message [{i}] of {frozen_count} perturbed — cache state machine NOT superseded (Rust compressed a frozen turn)"
                            ));
                            break;
                        }
                    }
                }
            }
            _ => {
                failures.push(format!("{name}: `messages` missing or not an array"));
            }
        }

        // (3) Tool set preserved — E1/E2/E3 may reorder tools, sort schema keys,
        // and inject a `cache_control` marker, but must never drop or invent a
        // tool. (The E1/E2/E3 mechanics themselves are covered by the Rust unit
        // tests in `live_zone_anthropic.rs`.)
        let tool_names = |v: &Value| -> BTreeSet<String> {
            v.get("tools")
                .and_then(Value::as_array)
                .map(|ts| {
                    ts.iter()
                        .filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default()
        };
        if tool_names(&in_json) != tool_names(&got_json) {
            failures.push(format!(
                "{name}: tool name set changed (a tool was dropped or invented)"
            ));
        }

        // (4) Quality on the compressible live zone (`messages`), isolated from
        // the tool `cache_control` growth: Rust must never inflate messages, and
        // must compress at least as well as the Python golden on that zone.
        let msgs_len = |v: &Value| -> usize {
            v.get("messages")
                .map(|m| serde_json::to_vec(m).map(|b| b.len()).unwrap_or(0))
                .unwrap_or(0)
        };
        let (in_m, got_m, gold_m) = (
            msgs_len(&in_json),
            msgs_len(&got_json),
            msgs_len(&golden_json),
        );
        if got_m > in_m {
            failures.push(format!(
                "{name}: messages grew ({in_m} -> {got_m}) — live zone inflated"
            ));
        }
        if got_m > gold_m {
            failures.push(format!(
                "{name}: messages Rust ({got_m}) larger than Python golden ({gold_m}) — quality regression"
            ));
        }
    }

    assert!(
        checked >= 10,
        "expected >=10 /v1/messages fixtures, checked {checked}"
    );
    eprintln!(
        "D2 anthropic cache-safety + quality: {} ok / {} checked",
        checked - failures.len(),
        checked
    );
    assert!(
        failures.is_empty(),
        "{} anthropic fixtures failed cache-safety/quality:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
