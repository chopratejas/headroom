//! Anthropic `/v1/messages` request compression.
//!
//! # Request shape (relevant subset)
//!
//! ```json
//! {
//!   "model": "claude-3-5-sonnet-20241022",
//!   "system": "...",                        // string OR list of blocks (optional)
//!   "messages": [
//!     {"role": "user",      "content": "..."},
//!     {"role": "assistant", "content": [...]}
//!   ],
//!   "tools": [...],                         // optional
//!   "max_tokens": 1024,                     // required
//!   ...
//! }
//! ```
//!
//! # What we do
//!
//! 1. Parse the body as JSON. On failure → passthrough.
//! 2. Pull `messages` (the only field we touch). On absence → passthrough.
//! 3. Pull `model` and `max_tokens` to compute the available budget.
//! 4. Run the ICM's `should_apply` gate. Under-budget → passthrough.
//! 5. Run `apply()`. Re-insert the (possibly trimmed) `messages` into
//!    the parsed JSON. Re-serialize.
//! 6. On *any* error along the way: passthrough with a warn log. The
//!    proxy must never break a request because compression failed.
//!
//! # What we DON'T do
//!
//! - Touch `system`. Anthropic separates system from messages; our
//!   ICM operates on the messages list. The system tokens are
//!   "invisible" to ICM's budget calculation, which means we
//!   under-count slightly — that's fine (we'll compress less than
//!   strictly necessary, never more).
//! - Touch `tools`, `temperature`, `top_p`, etc. These pass through
//!   verbatim because they're tiny and load-bearing for behaviour.
//! - Compress individual content blocks. That's content-router /
//!   pipeline work, scoped to a follow-up PR. ICM operates at the
//!   message-list level only.

use bytes::Bytes;
use serde_json::Value;

use headroom_core::context::{ApplyCtx, IntelligentContextManager};

use super::model_limits::context_window_for;

/// What happened. Used for the request-level tracing log.
#[derive(Debug)]
pub enum Outcome {
    /// Body was unchanged. Reasons listed in `reason`.
    Passthrough { reason: PassthroughReason },
    /// ICM ran but didn't drop anything (already under budget).
    NoCompression { tokens_before: usize },
    /// ICM ran and trimmed the message list.
    Compressed {
        body: Bytes,
        tokens_before: usize,
        tokens_after: usize,
        strategies_applied: Vec<&'static str>,
        markers_inserted: Vec<String>,
    },
}

/// Why we passed the body through unchanged.
#[derive(Debug, Clone, Copy)]
pub enum PassthroughReason {
    /// JSON parse failed.
    NotJson,
    /// `messages` was missing or not a JSON array.
    NoMessages,
    /// Re-serialization of the modified body failed (shouldn't
    /// happen — we just deserialized this shape).
    SerializeFailed,
}

/// Run ICM over an Anthropic-shape body. Returns one of:
///
/// - `Outcome::Compressed` — caller should forward `outcome.body`
///   instead of the original bytes.
/// - `Outcome::NoCompression` — caller forwards the original
///   bytes; ICM's `should_apply` returned false.
/// - `Outcome::Passthrough` — same as `NoCompression` from the
///   caller's perspective, but the reason is parse/serialize-related.
///
/// Never returns an error. Compression failures degrade to
/// passthrough; this is the proxy's safety contract.
pub fn maybe_compress(body: &Bytes, icm: &IntelligentContextManager) -> Outcome {
    let mut parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            return Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            };
        }
    };

    // Move the messages array out of the object so we can hand
    // ownership to ICM. We re-insert at the end. If `messages` is
    // missing or not an array, passthrough.
    let messages = match parsed.get_mut("messages") {
        Some(Value::Array(_)) => match parsed["messages"].take() {
            Value::Array(a) => a,
            _ => unreachable!("just matched as_array"),
        },
        _ => {
            return Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            };
        }
    };

    let model = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    // `context_window_for` returns `u32` (LiteLLM-sourced). ICM's
    // `ApplyCtx::model_limit` wants `usize`. The cast is lossless on
    // every platform we run on — context windows are far below 4GB.
    let model_limit = context_window_for(model) as usize;

    // Anthropic requires `max_tokens`; if absent (malformed), assume
    // a small reservation rather than zero so we don't pretend the
    // whole window is available for input.
    let output_buffer = parsed
        .get("max_tokens")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(4_096);

    // Cheap pre-check before the real apply call. Saves the cost of
    // a full message-list traversal when the request is small.
    if !icm.should_apply(&messages, model_limit, output_buffer) {
        // Restore messages and return without compression.
        parsed["messages"] = Value::Array(messages);
        // Compute tokens_before from the re-inserted form for log
        // accuracy. Cheap: it's the same walk should_apply already
        // did, but we don't have the count back from that call. We
        // skip the recount and return 0; the caller's log just
        // shows "no_compression" without a number, which is fine.
        return Outcome::NoCompression { tokens_before: 0 };
    }

    let result = icm.apply(
        messages,
        ApplyCtx {
            model_limit,
            output_buffer: Some(output_buffer),
            // TODO: detect provider prefix-cached messages from the
            // request. Anthropic exposes prompt caching via
            // `cache_control` on content blocks. Until we wire that
            // detection, we treat the whole list as droppable.
            frozen_message_count: 0,
        },
    );

    // ICM may return tokens_after >= tokens_before when no drops
    // happened (e.g. everything is protected). Treat that as
    // no-compression rather than ship a needless re-serialize.
    if result.tokens_after >= result.tokens_before {
        // Reinsert the (unchanged) messages and report.
        parsed["messages"] = Value::Array(result.messages);
        return Outcome::NoCompression {
            tokens_before: result.tokens_before,
        };
    }

    parsed["messages"] = Value::Array(result.messages);

    let new_body = match serde_json::to_vec(&parsed) {
        Ok(v) => Bytes::from(v),
        Err(_) => {
            return Outcome::Passthrough {
                reason: PassthroughReason::SerializeFailed,
            };
        }
    };

    Outcome::Compressed {
        body: new_body,
        tokens_before: result.tokens_before,
        tokens_after: result.tokens_after,
        strategies_applied: result.strategies_applied,
        markers_inserted: result.markers_inserted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::icm::build_icm;
    use serde_json::json;

    fn icm() -> std::sync::Arc<IntelligentContextManager> {
        build_icm().expect("ICM builds")
    }

    #[test]
    fn passthrough_on_invalid_json() {
        let icm = icm();
        let body = Bytes::from_static(b"not json");
        match maybe_compress(&body, &icm) {
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            } => {}
            other => panic!("expected NotJson passthrough, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_when_messages_field_missing() {
        let icm = icm();
        let body = Bytes::from(json!({"model": "claude-3-5-sonnet-20241022"}).to_string());
        match maybe_compress(&body, &icm) {
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            } => {}
            other => panic!("expected NoMessages passthrough, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_when_messages_not_array() {
        let icm = icm();
        let body = Bytes::from(
            json!({"model": "claude-3-5-sonnet", "messages": "not-an-array"}).to_string(),
        );
        match maybe_compress(&body, &icm) {
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            } => {}
            other => panic!("expected NoMessages passthrough, got {other:?}"),
        }
    }

    #[test]
    fn no_compression_when_under_budget() {
        let icm = icm();
        let body = Bytes::from(
            json!({
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "hello"}]
            })
            .to_string(),
        );
        match maybe_compress(&body, &icm) {
            Outcome::NoCompression { .. } => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }

    #[test]
    fn compresses_when_over_budget() {
        let icm = icm();
        // Squeeze the available budget by setting a huge max_tokens
        // so output_buffer eats almost the whole window. With a
        // 200K window for Claude and max_tokens=199_500, only ~500
        // tokens are available — anything bigger forces compression.
        let big_messages: Vec<Value> = (0..30)
            .map(|i| {
                json!({
                    "role": if i % 2 == 0 { "user" } else { "assistant" },
                    "content": format!("padding token {i} ").repeat(20),
                })
            })
            .collect();
        let body = Bytes::from(
            json!({
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 199_500,
                "messages": big_messages,
            })
            .to_string(),
        );
        match maybe_compress(&body, &icm) {
            Outcome::Compressed {
                body: new_body,
                tokens_before,
                tokens_after,
                ..
            } => {
                assert!(tokens_after < tokens_before);
                // The new body is valid JSON with a shorter messages
                // array (or includes the CCR marker that the shim's
                // injection logic adds — but the proxy doesn't do
                // marker injection; that lives in the Python shim).
                let parsed: Value = serde_json::from_slice(&new_body).unwrap();
                assert!(parsed["messages"].as_array().is_some());
            }
            other => panic!("expected Compressed, got {other:?}"),
        }
    }

    #[test]
    fn unknown_model_does_not_panic() {
        // Should fall back to the default 128K window and behave
        // like any other request.
        let icm = icm();
        let body = Bytes::from(
            json!({
                "model": "future-model-2099",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}]
            })
            .to_string(),
        );
        let _ = maybe_compress(&body, &icm); // shouldn't panic
    }

    #[test]
    fn missing_max_tokens_does_not_panic() {
        let icm = icm();
        let body = Bytes::from(
            json!({
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}]
            })
            .to_string(),
        );
        let _ = maybe_compress(&body, &icm); // shouldn't panic
    }
}
