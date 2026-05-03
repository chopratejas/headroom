//! Anthropic `/v1/messages` request compression — live-zone
//! dispatcher entry point.
//!
//! # Provider scope
//!
//! This module is **Anthropic-only**. The proxy gates compression
//! on `path == "/v1/messages"` (see `compression::is_compressible_path`).
//! OpenAI Chat Completions, OpenAI Responses, and Google Gemini
//! each get their own sibling module in Phase C — they share
//! [`headroom_core::transforms::LiveZoneOutcome`] and the
//! per-content-type compressor backend, but the walkers are
//! provider-specific because the request shapes diverge.
//!
//! # Pipeline
//!
//! 1. Resolve `frozen_message_count` from the request body via
//!    [`crate::compression::resolve_frozen_count`] (PR-A4 helper).
//!    The proxy's `cache_control_auto_frozen` config gates whether
//!    the body is parsed at all — when disabled, floor=0 without
//!    inspection.
//! 2. Hand the buffered body bytes to
//!    [`headroom_core::transforms::compress_anthropic_live_zone`].
//!    The dispatcher inspects the live zone (latest user message),
//!    detects per-block content type, dispatches each block to the
//!    matching compressor (SmartCrusher / LogCompressor /
//!    SearchCompressor / DiffCompressor), and rewrites the body
//!    via byte-range surgery so unmodified bytes round-trip
//!    byte-equal.
//! 3. Translate [`LiveZoneOutcome::Modified`] →
//!    [`Outcome::Compressed`] (caller forwards the new body) or
//!    [`LiveZoneOutcome::NoChange`] → [`Outcome::NoCompression`]
//!    (caller forwards the original body verbatim).
//!
//! # Cache-safety invariant
//!
//! Bytes outside the rewritten block round-trip byte-equal. The
//! `byte_fidelity_outside_compressed_block` integration test in
//! `crates/headroom-core/tests/live_zone_dispatch.rs` pins the
//! SHA-256 prefix-and-suffix invariant in CI.

use bytes::Bytes;
use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_anthropic_live_zone, AuthMode, BlockAction, ExclusionReason, LiveZoneError,
    LiveZoneOutcome,
};

use crate::compression::resolve_frozen_count;
use crate::config::{CacheControlAutoFrozen, CompressionMode};

/// What happened. The caller uses the variant to decide whether to
/// forward the original bytes (everything PR-B2 lands on) or a
/// modified body (PR-B3+).
#[derive(Debug)]
pub enum Outcome {
    /// Body was not compressed. Caller forwards the original
    /// buffered bytes byte-equal. Always returned in PR-B2.
    NoCompression,
    /// Reserved for PR-B3+: live-zone compression actually ran and
    /// produced a (smaller) body.
    #[allow(dead_code)]
    Compressed {
        body: Bytes,
        tokens_before: usize,
        tokens_after: usize,
        strategies_applied: Vec<&'static str>,
        markers_inserted: Vec<String>,
    },
    /// Dispatcher opted out for a reason we can name.
    Passthrough { reason: PassthroughReason },
}

/// Reason the live-zone dispatcher fell through. Each variant is
/// logged at warn level by the proxy.
#[derive(Debug, Clone, Copy)]
pub enum PassthroughReason {
    /// Body was not valid JSON — never our job to fix that, but we
    /// log so operators know which requests opted out.
    NotJson,
    /// `messages` was missing or not a JSON array — the upstream
    /// API will reject with a 400 anyway; we're just bystanders.
    NoMessages,
    /// The compression-mode config is `Off`. The dispatcher is not
    /// invoked.
    ModeOff,
}

/// Live-zone compression entry point for Anthropic `/v1/messages`.
///
/// Returns one of:
///
/// - [`Outcome::NoCompression`] — proxy forwards the original
///   buffered body verbatim. PR-B2 always lands here.
/// - [`Outcome::Compressed`] — PR-B3+ produces this when at least
///   one block was rewritten.
/// - [`Outcome::Passthrough`] — invalid body shape; proxy forwards
///   the original bytes anyway.
///
/// # Arguments
///
/// - `body`: the buffered request body. Owned by the caller for the
///   lifetime of the upstream request — we only borrow.
/// - `mode`: configured compression mode. `Off` short-circuits to
///   [`Outcome::Passthrough { reason: ModeOff }`]; `LiveZone` runs
///   the dispatcher.
/// - `cache_control_policy`: gates auto-derivation of
///   `frozen_message_count` from explicit `cache_control` markers
///   in the body. Disabled → floor=0 (everything is in the live
///   zone).
/// - `request_id`: per-request id used for log correlation.
pub fn compress_anthropic_request(
    body: &Bytes,
    mode: CompressionMode,
    cache_control_policy: CacheControlAutoFrozen,
    request_id: &str,
) -> Outcome {
    if matches!(mode, CompressionMode::Off) {
        tracing::info!(
            request_id = %request_id,
            path = "/v1/messages",
            method = "POST",
            compression_mode = mode.as_str(),
            decision = "passthrough",
            reason = "mode_off",
            body_bytes = body.len(),
            "anthropic compression decision"
        );
        return Outcome::Passthrough {
            reason: PassthroughReason::ModeOff,
        };
    }

    // Mode is LiveZone. Resolve the cache-hot floor first; this is
    // the only place the body is parsed at all when the policy is
    // Disabled (resolve_frozen_count short-circuits).
    let parsed: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "not_json",
                body_bytes = body.len(),
                "anthropic compression decision"
            );
            return Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            };
        }
    };

    let frozen_count = resolve_frozen_count(&parsed, cache_control_policy, request_id);

    // PR-B4: extract `body["model"]` so the live-zone dispatcher can
    // route the tokenizer registry to the right backend for the
    // per-block token-count rejection gate. Anthropic
    // `/v1/messages` always carries a `model` string per the API
    // schema, but the proxy never breaks on a missing field — we
    // fall back to `DEFAULT_MODEL` (a Claude name, so the
    // chars-per-token estimator picks the calibrated 3.5 cpt
    // density) and continue.
    let model = parsed
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(DEFAULT_MODEL);

    // Run the live-zone dispatcher. PR-B3 wires per-type compressors:
    // SmartCrusher / LogCompressor / SearchCompressor / DiffCompressor.
    // PR-B4 added the per-content-type byte-threshold gate and the
    // tokenizer-validated rejection check. The dispatcher returns
    // `Modified` whenever at least one block was rewritten and
    // `NoChange` otherwise (live zone empty, every compressor
    // declined, or every compressor produced output whose token
    // count was not strictly less than the input's).
    match compress_anthropic_live_zone(body, frozen_count, AuthMode::Payg, model) {
        Ok(LiveZoneOutcome::NoChange { manifest }) => {
            let block_count = manifest.block_outcomes.len();
            let blocks_excluded = manifest
                .block_outcomes
                .iter()
                .filter(|b| {
                    matches!(
                        b.action,
                        BlockAction::Excluded {
                            reason: ExclusionReason::HotZoneBlockType
                        }
                    )
                })
                .count();
            tracing::info!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "no_change",
                reason = "no_block_compressed",
                body_bytes = body.len(),
                frozen_message_count = frozen_count,
                messages_total = manifest.messages_total,
                latest_user_message_index = ?manifest.latest_user_message_index,
                live_zone_blocks = block_count,
                live_zone_blocks_excluded = blocks_excluded,
                "anthropic live-zone dispatch"
            );
            Outcome::NoCompression
        }
        Ok(LiveZoneOutcome::Modified { new_body, manifest }) => {
            // Aggregate manifest into the proxy's `Compressed` payload.
            // PR-B4 reports token counts via the same tokenizer the
            // dispatcher used to gate per-block acceptance — so the
            // saving the proxy logs is the saving the cache will
            // actually see.
            let mut original_bytes_total: usize = 0;
            let mut compressed_bytes_total: usize = 0;
            let mut original_tokens_total: usize = 0;
            let mut compressed_tokens_total: usize = 0;
            let mut strategies: Vec<&'static str> = Vec::new();
            for entry in &manifest.block_outcomes {
                if let BlockAction::Compressed {
                    strategy,
                    original_bytes,
                    compressed_bytes,
                    original_tokens,
                    compressed_tokens,
                } = entry.action
                {
                    original_bytes_total += original_bytes;
                    compressed_bytes_total += compressed_bytes;
                    original_tokens_total += original_tokens;
                    compressed_tokens_total += compressed_tokens;
                    if !strategies.contains(&strategy) {
                        strategies.push(strategy);
                    }
                }
            }
            let body_bytes_in = body.len();
            let new_body_bytes = Bytes::copy_from_slice(new_body.get().as_bytes());
            let body_bytes_out = new_body_bytes.len();
            let block_count = manifest.block_outcomes.len();
            tracing::info!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "compressed",
                reason = "live_zone_blocks_rewritten",
                body_bytes_in = body_bytes_in,
                body_bytes_out = body_bytes_out,
                bytes_freed = body_bytes_in.saturating_sub(body_bytes_out),
                frozen_message_count = frozen_count,
                messages_total = manifest.messages_total,
                latest_user_message_index = ?manifest.latest_user_message_index,
                live_zone_blocks = block_count,
                live_zone_strategies = ?strategies,
                live_zone_block_original_bytes = original_bytes_total,
                live_zone_block_compressed_bytes = compressed_bytes_total,
                live_zone_block_original_tokens = original_tokens_total,
                live_zone_block_compressed_tokens = compressed_tokens_total,
                model = model,
                "anthropic live-zone dispatch"
            );
            Outcome::Compressed {
                body: new_body_bytes,
                tokens_before: original_tokens_total,
                tokens_after: compressed_tokens_total,
                strategies_applied: strategies,
                // PR-B7 wires CCR retrieval-marker injection.
                markers_inserted: Vec::new(),
            }
        }
        Err(LiveZoneError::BodyNotJson(_)) => {
            // We already parsed successfully above; the dispatcher's
            // independent parse can only fail on a state we missed.
            // Pass through with the same byte-faithful guarantee.
            tracing::warn!(
                request_id = %request_id,
                path = "/v1/messages",
                "live-zone dispatcher rejected JSON body that this layer parsed; \
                 falling back to passthrough"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            }
        }
        Err(LiveZoneError::NoMessagesArray) => {
            tracing::info!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "no_messages",
                body_bytes = body.len(),
                "anthropic compression decision"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_of(value: serde_json::Value) -> Bytes {
        Bytes::from(serde_json::to_vec(&value).unwrap())
    }

    #[test]
    fn mode_off_short_circuits_without_parsing() {
        // Invalid JSON — would fail parse — but mode=Off must not
        // attempt to parse, and instead Passthrough{ModeOff}.
        let body = Bytes::from_static(b"not valid json");
        let out = compress_anthropic_request(
            &body,
            CompressionMode::Off,
            CacheControlAutoFrozen::Disabled,
            "req-1",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::ModeOff,
            } => {}
            other => panic!("expected Passthrough{{ModeOff}}, got {other:?}"),
        }
    }

    #[test]
    fn live_zone_mode_with_no_messages_field_passthrough() {
        let body = body_of(serde_json::json!({"model": "claude"}));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Enabled,
            "req-2",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            } => {}
            other => panic!("expected Passthrough{{NoMessages}}, got {other:?}"),
        }
    }

    #[test]
    fn live_zone_mode_with_invalid_json_passthrough() {
        let body = Bytes::from_static(b"\x01\x02 not json");
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Enabled,
            "req-3",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            } => {}
            other => panic!("expected Passthrough{{NotJson}}, got {other:?}"),
        }
    }

    #[test]
    fn live_zone_mode_with_valid_body_returns_no_compression_pr_b2() {
        // PR-B2 invariant: every well-formed body returns NoCompression.
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "hello"}
                ]}
            ]
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            "req-4",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }

    #[test]
    fn empty_body_with_live_zone_mode_passthrough_not_json() {
        let body = Bytes::new();
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Enabled,
            "req-5",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            } => {}
            other => panic!("expected Passthrough{{NotJson}}, got {other:?}"),
        }
    }

    #[test]
    fn cache_control_disabled_yields_floor_zero() {
        // With auto-derivation Disabled, frozen floor is 0 even
        // though the body marks every message as cached. The
        // dispatcher will treat the entire array as live zone.
        // (PR-B2: still returns NoCompression — this test pins the
        // policy plumbing rather than compression behaviour.)
        let body = body_of(serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
                    ]
                }
            ]
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            "req-6",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }
}
