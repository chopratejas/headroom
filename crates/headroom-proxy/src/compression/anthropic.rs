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
//! Delegate to [`super::messages::compress_messages`] with an
//! Anthropic-specific output-buffer extractor: pull `max_tokens`,
//! fall back to the shared default when absent. Everything else —
//! parse, gate, apply, re-serialize, passthrough on failure — lives
//! in the provider-agnostic core.
//!
//! # What we DON'T do
//!
//! - Touch `system`. Anthropic separates system from messages; ICM
//!   operates on the messages list. The system tokens are
//!   "invisible" to ICM's budget calculation, which means we
//!   under-count slightly — that's fine (we'll compress less than
//!   strictly necessary, never more).
//! - Touch `tools`, `temperature`, `top_p`, etc. These pass through
//!   verbatim because they're tiny and load-bearing for behaviour.

use bytes::Bytes;
use serde_json::Value;

use headroom_core::context::IntelligentContextManager;

use super::messages::{compress_messages, Outcome};

/// Run ICM over an Anthropic-shape body. See
/// [`super::messages::Outcome`] for the result variants. Never
/// returns an error; failures degrade to passthrough.
pub fn maybe_compress(body: &Bytes, icm: &IntelligentContextManager) -> Outcome {
    compress_messages(body, icm, |parsed: &Value| {
        // Anthropic requires `max_tokens`; absence implies a
        // malformed request, but we degrade gracefully via the
        // shared default rather than refusing to compress.
        parsed
            .get("max_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::icm::build_icm;
    use crate::compression::messages::PassthroughReason;
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
        // Squeeze the available budget: max_tokens=199_500 leaves
        // ~500 tokens for input on a 200K-window model.
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
                let parsed: Value = serde_json::from_slice(&new_body).unwrap();
                assert!(parsed["messages"].as_array().is_some());
            }
            other => panic!("expected Compressed, got {other:?}"),
        }
    }

    #[test]
    fn unknown_model_does_not_panic() {
        let icm = icm();
        let body = Bytes::from(
            json!({
                "model": "future-model-2099",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}]
            })
            .to_string(),
        );
        let _ = maybe_compress(&body, &icm);
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
        let _ = maybe_compress(&body, &icm);
    }
}
