//! Provider-agnostic core for "JSON body with a `messages` array".
//!
//! Both Anthropic `/v1/messages` and OpenAI `/v1/chat/completions`
//! follow the same overall pattern:
//!
//! 1. Body is JSON.
//! 2. Top-level field `messages` is an array we can hand to ICM.
//! 3. Top-level `model` field selects the context window.
//! 4. Some "reserved for output" field caps how much input we can fit.
//!
//! What differs between providers is small:
//!
//! - **Anthropic** uses `max_tokens` (required) and keeps the system
//!   prompt in a separate top-level `system` field.
//! - **OpenAI** uses `max_completion_tokens` (newer, o-series) or
//!   `max_tokens` (classic) — both optional — and includes the system
//!   message inside `messages` as `role:"system"`.
//!
//! Neither difference matters at this layer: ICM operates on the
//! messages list and respects `role:"system"` via SafetyRails. The
//! only provider-specific bit is *where to find the output buffer
//! token count*. We capture that as a closure passed by the caller.
//!
//! # Failure-mode contract
//!
//! Compression must never break a request. Every error path
//! (parse, missing field, serialize) degrades to a passthrough
//! variant. The proxy then forwards the original buffered bytes.

use bytes::Bytes;
use serde_json::Value;

use headroom_core::context::{ApplyCtx, IntelligentContextManager};

use super::model_limits::context_window_for;

/// What happened. Used for the request-level tracing log.
#[derive(Debug)]
pub enum Outcome {
    /// Body was unchanged. Reason explains why.
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

/// Why the body was passed through unchanged.
#[derive(Debug, Clone, Copy)]
pub enum PassthroughReason {
    /// JSON parse failed.
    NotJson,
    /// `messages` was missing or not a JSON array.
    NoMessages,
    /// Re-serialization of the modified body failed.
    SerializeFailed,
}

/// Default output reservation when the caller's extractor returns
/// `None`. 4096 is a defensible middle: small enough that it doesn't
/// pretend the whole context window is for input, large enough that
/// short replies fit comfortably.
pub const DEFAULT_OUTPUT_BUFFER: usize = 4_096;

/// Run ICM over a body whose top-level shape has a `messages` array
/// and a `model` field. The caller supplies an `extract_output_buffer`
/// closure that pulls the provider-specific output reservation field
/// (e.g. Anthropic `max_tokens`, OpenAI `max_completion_tokens`).
///
/// Never returns an error; failures degrade to `Outcome::Passthrough`
/// with a reason variant for the trace log. The caller forwards the
/// original `body` bytes in that case.
pub fn compress_messages<F>(
    body: &Bytes,
    icm: &IntelligentContextManager,
    extract_output_buffer: F,
) -> Outcome
where
    F: FnOnce(&Value) -> Option<usize>,
{
    let mut parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            return Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            };
        }
    };

    // Take ownership of the messages array so we can pass `Vec<Value>`
    // into ICM. Re-insert below.
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
    let model_limit = context_window_for(model) as usize;

    let output_buffer = extract_output_buffer(&parsed).unwrap_or(DEFAULT_OUTPUT_BUFFER);

    if !icm.should_apply(&messages, model_limit, output_buffer) {
        parsed["messages"] = Value::Array(messages);
        return Outcome::NoCompression { tokens_before: 0 };
    }

    let result = icm.apply(
        messages,
        ApplyCtx {
            model_limit,
            output_buffer: Some(output_buffer),
            // TODO: detect provider prefix-cached messages (Anthropic
            // cache_control on content blocks; OpenAI doesn't expose
            // an equivalent yet). Until wired, treat the whole list
            // as droppable.
            frozen_message_count: 0,
        },
    );

    if result.tokens_after >= result.tokens_before {
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
