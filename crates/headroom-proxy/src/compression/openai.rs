//! OpenAI `/v1/chat/completions` request compression.
//!
//! # Request shape (relevant subset)
//!
//! ```json
//! {
//!   "model": "gpt-4o-mini",
//!   "messages": [
//!     {"role": "system", "content": "..."},
//!     {"role": "user", "content": "..."},
//!     {"role": "assistant", "content": "...", "tool_calls": [...]},
//!     {"role": "tool",    "content": "...", "tool_call_id": "..."}
//!   ],
//!   "tools": [...],                          // optional
//!   "max_tokens": 1024,                      // optional, classic
//!   "max_completion_tokens": 1024,           // optional, o-series
//!   "stream": false,
//!   ...
//! }
//! ```
//!
//! # Differences from Anthropic
//!
//! 1. **System prompt** lives inside `messages` as `role:"system"`.
//!    ICM's SafetyRails already protects `role:"system"`, so we don't
//!    need to do anything special — that's exactly the OSS-defining
//!    "messages-list-shaped" abstraction working as designed.
//!
//! 2. **Output buffer field** is either `max_completion_tokens` (the
//!    new, mandatory field for o1/o3-style reasoning models) OR
//!    `max_tokens` (the classic field). When both are absent, OpenAI
//!    defaults the response to "as much as the model can produce";
//!    we fall back to the shared 4K default rather than assume the
//!    whole window is for input.
//!
//! 3. **Tool atomicity**: OpenAI uses `assistant.tool_calls[]` paired
//!    with `tool` messages whose `tool_call_id` references the
//!    assistant message. ICM's `find_tool_units` already handles this
//!    shape (along with Anthropic's content-block form).
//!
//! # What we DON'T do
//!
//! - Touch `tools`, `tool_choice`, `response_format`, `seed`,
//!   `temperature`, `top_p`. These are small and behaviour-defining.
//! - Touch `function_call` / `functions` (the deprecated pre-tools
//!   API). ICM doesn't drop these either; they pass through verbatim.
//! - Compress streaming response bodies. The proxy only buffers the
//!   *request* body for compression; the SSE response stream is
//!   forwarded chunk-by-chunk untouched.
//!
//! # Azure / OpenRouter / other compatible endpoints
//!
//! Path-matched on `/v1/chat/completions` exactly. Azure OpenAI uses
//! `/openai/deployments/{deployment}/chat/completions` and won't
//! match — that's intentional. When/if we want Azure support, add a
//! distinct path arm.

use bytes::Bytes;
use serde_json::Value;

use headroom_core::context::IntelligentContextManager;

use super::messages::{compress_messages, Outcome};

/// Run ICM over an OpenAI-shape body.
pub fn maybe_compress(body: &Bytes, icm: &IntelligentContextManager) -> Outcome {
    compress_messages(body, icm, |parsed: &Value| {
        // Prefer max_completion_tokens (o-series, newer). Fall back
        // to max_tokens (classic). OpenAI accepts only one or the
        // other on a given model; we don't need to add them.
        parsed
            .get("max_completion_tokens")
            .or_else(|| parsed.get("max_tokens"))
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
        let body = Bytes::from_static(b"not json at all");
        match maybe_compress(&body, &icm) {
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            } => {}
            other => panic!("expected NotJson, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_when_messages_missing() {
        let icm = icm();
        let body = Bytes::from(json!({"model": "gpt-4o-mini"}).to_string());
        match maybe_compress(&body, &icm) {
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            } => {}
            other => panic!("expected NoMessages, got {other:?}"),
        }
    }

    #[test]
    fn no_compression_when_under_budget() {
        let icm = icm();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "max_tokens": 1024,
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user",   "content": "hi"}
                ]
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
        // gpt-4o-mini context window is 128K. Set max_tokens close to
        // it so output_buffer eats nearly the whole window, forcing
        // even modest histories over the input budget.
        let big_messages: Vec<Value> = std::iter::once(json!({
            "role": "system",
            "content": "Always be concise."
        }))
        .chain((0..40).map(|i| {
            json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("history turn {i} ").repeat(40),
            })
        }))
        .collect();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "max_tokens": 127_000,
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
                let msgs = parsed["messages"].as_array().unwrap();
                // System message must survive — that's the safety
                // contract on OpenAI shape (system lives in messages).
                assert!(
                    msgs.iter()
                        .any(|m| m.get("role").and_then(Value::as_str) == Some("system")),
                    "system message must be preserved after compression"
                );
            }
            other => panic!("expected Compressed, got {other:?}"),
        }
    }

    #[test]
    fn max_completion_tokens_takes_precedence_over_max_tokens() {
        // When both are present (transition shape), the newer field
        // wins because that's the field OpenAI actually applies on
        // o-series models. We don't simulate apply here — we just
        // verify the call path doesn't panic and the body parses.
        let icm = icm();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "max_tokens": 100,
                "max_completion_tokens": 200,
                "messages": [{"role": "user", "content": "hi"}]
            })
            .to_string(),
        );
        let _ = maybe_compress(&body, &icm); // shouldn't panic
    }

    #[test]
    fn missing_both_token_fields_does_not_panic() {
        let icm = icm();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}]
            })
            .to_string(),
        );
        let _ = maybe_compress(&body, &icm);
    }
}
