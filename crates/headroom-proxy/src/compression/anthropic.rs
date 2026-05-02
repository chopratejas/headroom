//! Anthropic `/v1/messages` request compression — Phase A passthrough stub.
//!
//! # Phase A lockdown (PR-A1)
//!
//! Per `REALIGNMENT/03-phase-A-lockdown.md`, this function is now a
//! byte-faithful passthrough. The previous implementation invoked
//! `IntelligentContextManager` with a hardcoded `frozen_message_count: 0`,
//! which destroyed Anthropic prompt-cache hit rate by dropping messages
//! out of the cache hot zone. That bug cluster (P0-3, P0-4, P0-5,
//! P1-13) is eliminated by *not running the compressor at all* until
//! Phase B builds the live-zone-only replacement.
//!
//! The function signature is preserved so the call site in `proxy.rs`
//! still compiles unchanged. Phase B PR-B2 fills this back in with the
//! live-zone block dispatcher (compress only the latest user message,
//! latest tool/function/shell/patch outputs — never historical turns).
//!
//! # What this returns
//!
//! Always `Outcome::NoCompression`. The caller (`proxy.rs`) reacts to
//! that by forwarding the original buffered bytes verbatim.
//!
//! # What it does NOT do
//!
//! - Does NOT parse the JSON body. The whole point of Phase A is byte
//!   faithfulness; parsing + re-serialization could perturb whitespace,
//!   numeric precision, key ordering, and Unicode escaping. Even
//!   though we wouldn't re-emit the parsed value here, parsing is
//!   wasted work and would invite future "while we're here" mutations.
//! - Does NOT touch headers, body, or any other request state.
//! - Does NOT depend on `IntelligentContextManager` (the type is gone
//!   from this module's call graph; `mod.rs` no longer imports `icm`).
//!
//! # Logging
//!
//! Emits exactly one structured `tracing::info!` per call, with the
//! decision (`"passthrough"`), the reason (`"phase_a_lockdown"`), the
//! configured `compression_mode`, and the body byte count. The
//! `request_id` and HTTP method/path come from the caller's
//! existing log context (added in `proxy.rs`).

use bytes::Bytes;
use serde_json::Value;

use crate::config::{CacheControlAutoFrozen, CompressionMode};

/// What happened. The caller uses the variant to decide whether to
/// forward the original bytes (everything) or a modified body
/// (currently never).
///
/// PR-A1 lockdown: `compress_anthropic_request` always returns
/// `Outcome::NoCompression`. The other variants remain in the enum
/// because Phase B PR-B2 reintroduces them with live-zone semantics
/// — keeping the surface stable lets us land Phase B as a pure
/// implementation swap rather than a disruptive enum redesign.
#[derive(Debug)]
pub enum Outcome {
    /// Body was not compressed. Caller forwards the original buffered
    /// bytes byte-equal. This is the only variant Phase A produces.
    NoCompression,
    /// Reserved for Phase B: live-zone compression actually ran and
    /// produced a (smaller) body. Unused in PR-A1; kept so adding it
    /// later is a non-breaking change.
    #[allow(dead_code)]
    Compressed {
        body: Bytes,
        tokens_before: usize,
        tokens_after: usize,
        strategies_applied: Vec<&'static str>,
        markers_inserted: Vec<String>,
    },
    /// Reserved for Phase B: parse/serialize edge cases the live-zone
    /// dispatcher will distinguish from a normal pass. Unused in
    /// PR-A1.
    #[allow(dead_code)]
    Passthrough { reason: PassthroughReason },
}

/// Why the live-zone dispatcher (Phase B) opted out. Unused in PR-A1
/// but kept for surface compatibility.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum PassthroughReason {
    /// JSON parse failed.
    NotJson,
    /// `messages` was missing or not a JSON array.
    NoMessages,
    /// Re-serialization of the modified body failed.
    SerializeFailed,
}

/// Phase A passthrough stub for Anthropic `/v1/messages`.
///
/// Always returns `Outcome::NoCompression`. The function signature
/// matches what Phase B PR-B2 will fill in (live-zone block
/// dispatcher); keeping the signature stable means the proxy's
/// catch-all handler doesn't need to change again then.
///
/// # Arguments
///
/// - `body`: the full buffered request body. NOT inspected, NOT
///   parsed. We log only its byte length.
/// - `mode`: configured compression mode. PR-A1 logs the mode but
///   both `Off` and `LiveZone` result in passthrough. The caller
///   emits a `tracing::warn!` for `LiveZone` (since the live-zone
///   dispatcher isn't built yet) — see `proxy.rs`.
/// - `request_id`: the per-request id used for log correlation. The
///   caller already produced it (`ensure_request_id`); we accept it
///   as a borrowed `&str` so this function doesn't need its own
///   uuid dep.
///
/// # Returns
///
/// Always `Outcome::NoCompression`. Compression returns in Phase B.
pub fn compress_anthropic_request(
    body: &Bytes,
    mode: CompressionMode,
    request_id: &str,
) -> Outcome {
    tracing::info!(
        request_id = %request_id,
        path = "/v1/messages",
        method = "POST",
        compression_mode = mode.as_str(),
        decision = "passthrough",
        reason = "phase_a_lockdown",
        body_bytes = body.len(),
        "anthropic compression decision"
    );
    Outcome::NoCompression
}

/// Resolve the `frozen_message_count` floor for a parsed Anthropic
/// `/v1/messages` request body, honouring the
/// `cache_control_auto_frozen` config gate (PR-A4).
///
/// This is a thin wrapper around [`headroom_core::compute_frozen_count`]
/// that returns `0` when the operator has disabled automatic
/// derivation, regardless of the markers in `parsed`. Intended to be
/// called by Phase B's live-zone dispatcher; PR-A4 ships it ready
/// for that consumer alongside the underlying core helper.
///
/// # Arguments
///
/// - `parsed`: parsed JSON body. The walker reads `messages`,
///   `system`, and `tools`; other fields are ignored. The function
///   itself does NOT mutate the value.
/// - `policy`: the resolved [`CacheControlAutoFrozen`] from `Config`.
///   `Disabled` short-circuits to `0` without inspecting the body.
/// - `request_id`: the per-request id used for log correlation,
///   matched against the proxy's existing `tracing` span fields.
///
/// # Returns
///
/// The frozen-count floor (smallest `N` such that `messages[i]` for
/// `i < N` is in the cache hot zone), or `0` when auto-derivation
/// is disabled. Phase B PR-B2 forbids the live-zone dispatcher from
/// touching any index below this value.
pub fn resolve_frozen_count(
    parsed: &Value,
    policy: CacheControlAutoFrozen,
    request_id: &str,
) -> usize {
    if !policy.is_enabled() {
        tracing::debug!(
            request_id = %request_id,
            cache_control_auto_frozen = policy.as_str(),
            "cache_control auto-derivation disabled; floor=0"
        );
        return 0;
    }
    let count = headroom_core::compute_frozen_count(parsed);
    tracing::debug!(
        request_id = %request_id,
        cache_control_auto_frozen = policy.as_str(),
        frozen_count = count,
        "cache_control auto-derivation result"
    );
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_mode_off() {
        let body = Bytes::from_static(b"{\"model\":\"claude\",\"messages\":[]}");
        match compress_anthropic_request(&body, CompressionMode::Off, "test-req-id-1") {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_when_mode_live_zone_in_phase_a() {
        // PR-A1: live_zone is reserved for Phase B and currently
        // falls through to passthrough. The proxy's call site emits
        // the warning; this function uniformly logs and returns
        // NoCompression.
        let body = Bytes::from_static(b"{\"model\":\"claude\",\"messages\":[]}");
        match compress_anthropic_request(&body, CompressionMode::LiveZone, "test-req-id-2") {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_does_not_parse_invalid_json() {
        // Body deliberately not JSON. We must not error or parse —
        // passthrough is byte-faithful regardless of payload shape.
        let body = Bytes::from_static(b"not json at all \xFF\xFE");
        match compress_anthropic_request(&body, CompressionMode::Off, "test-req-id-3") {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_handles_empty_body() {
        let body = Bytes::new();
        match compress_anthropic_request(&body, CompressionMode::Off, "test-req-id-4") {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_handles_large_body() {
        // 4MB of payload — confirm we don't accidentally allocate or
        // iterate the body.
        let body = Bytes::from(vec![b'a'; 4 * 1024 * 1024]);
        match compress_anthropic_request(&body, CompressionMode::Off, "test-req-id-5") {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }
}
