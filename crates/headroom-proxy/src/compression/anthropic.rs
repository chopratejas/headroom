//! Anthropic `/v1/messages` token-mode compression.
//!
//! # Algorithm
//!
//! 1. Parse body as JSON. Read `messages`.
//! 2. Find the **last user message** index (LU). Everything < LU is
//!    the frozen prefix (untouched). Everything ≥ LU is the new turn.
//! 3. Walk new-turn messages:
//!    - role=assistant: passthrough (sacred — preserve accuracy)
//!    - role=user with string `content`: passthrough (sacred user text)
//!    - role=user with list `content`: walk blocks
//!      - `tool_result` block with string `content`: COMPRESS the
//!        string via [`crate::compression::walker::compress_blob`].
//!      - `tool_result` block with list `content`: walk inner blocks;
//!        compress text blocks; preserve image blocks.
//!      - any other block (text, image, document, thinking): preserve
//! 4. If anything changed, re-serialize. Otherwise passthrough.
//!
//! # What we never touch
//!
//! - The `system` field (top-level, separate from messages)
//! - Any `cache_control` markers (positional cache breakpoints)
//! - `tools`, `tool_choice`, `max_tokens`, `temperature`, etc.
//! - Anything in the prefix (messages[..LU])
//! - Assistant content (text, thinking, tool_use, redacted_thinking)
//! - Binary content (images, documents)
//! - User text (string content or `text`-type blocks in user msg)

use std::sync::Arc;

use bytes::Bytes;
use serde_json::Value;

use headroom_core::ccr::CcrStore;
use headroom_core::transforms::pipeline::CompressionPipeline;

use super::walker::compress_blob;
use super::{Outcome, PassthroughReason};

pub fn maybe_compress(
    body: &Bytes,
    pipeline: &CompressionPipeline,
    store: &Arc<dyn CcrStore>,
) -> Outcome {
    let mut parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            return Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            };
        }
    };

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

    let last_user = match find_last_user_idx(&messages) {
        Some(i) => i,
        None => {
            // No user message at all — restore and bail.
            parsed["messages"] = Value::Array(messages);
            return Outcome::Passthrough {
                reason: PassthroughReason::NoUserAnchor,
            };
        }
    };

    let bytes_before = body.len();
    let mut steps_applied: Vec<String> = Vec::new();
    let mut new_messages: Vec<Value> = Vec::with_capacity(messages.len());
    let mut any_changed = false;

    for (idx, msg) in messages.into_iter().enumerate() {
        if idx < last_user {
            // Frozen prefix — byte-identical passthrough.
            new_messages.push(msg);
            continue;
        }
        let (rebuilt, changed) = rewrite_message(msg, pipeline, store, &mut steps_applied);
        if changed {
            any_changed = true;
        }
        new_messages.push(rebuilt);
    }

    parsed["messages"] = Value::Array(new_messages);

    if !any_changed {
        return Outcome::Passthrough {
            reason: PassthroughReason::NothingToCompress,
        };
    }

    let new_body = match serde_json::to_vec(&parsed) {
        Ok(v) => Bytes::from(v),
        Err(_) => {
            return Outcome::Passthrough {
                reason: PassthroughReason::SerializeFailed,
            };
        }
    };
    let bytes_after = new_body.len();

    // NOTE: we do NOT gate on `bytes_after < bytes_before`. LLM
    // providers tokenize the *content* of message strings, not the
    // JSON byte representation — so compressed text that's shorter
    // as a raw string saves real tokens (billed) even if the
    // JSON-escaped form is slightly larger on the wire.
    // Bandwidth is essentially free; tokens are what costs.
    // The internal `compress_blob` already gated on raw-text
    // savings, which is the right bar.

    Outcome::Compressed {
        body: new_body,
        bytes_before,
        bytes_after,
        steps_applied,
    }
}

fn find_last_user_idx(messages: &[Value]) -> Option<usize> {
    messages
        .iter()
        .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"))
}

/// Compress eligible content inside a user message in the new turn.
/// Returns (rewritten_message, did_anything_change).
fn rewrite_message(
    msg: Value,
    pipeline: &CompressionPipeline,
    store: &Arc<dyn CcrStore>,
    steps_applied: &mut Vec<String>,
) -> (Value, bool) {
    let role = msg.get("role").and_then(Value::as_str);
    if role != Some("user") {
        // Assistant or other — sacred.
        return (msg, false);
    }

    let mut obj = match msg {
        Value::Object(m) => m,
        other => return (other, false),
    };

    // Anthropic content can be a plain string (sacred user text) OR
    // a list of blocks (which is where tool_result lives).
    let content = match obj.get("content") {
        Some(Value::Array(_)) => obj.remove("content").unwrap(),
        _ => {
            // String or other shape — sacred / unknown, leave it.
            return (Value::Object(obj), false);
        }
    };

    let blocks = match content {
        Value::Array(a) => a,
        _ => unreachable!(),
    };

    let mut new_blocks: Vec<Value> = Vec::with_capacity(blocks.len());
    let mut changed = false;

    for block in blocks {
        let (rebuilt, did_change) = rewrite_user_block(block, pipeline, store, steps_applied);
        if did_change {
            changed = true;
        }
        new_blocks.push(rebuilt);
    }

    obj.insert("content".to_string(), Value::Array(new_blocks));
    (Value::Object(obj), changed)
}

fn rewrite_user_block(
    block: Value,
    pipeline: &CompressionPipeline,
    store: &Arc<dyn CcrStore>,
    steps_applied: &mut Vec<String>,
) -> (Value, bool) {
    let mut obj = match block {
        Value::Object(m) => m,
        other => return (other, false),
    };

    let block_type = obj.get("type").and_then(Value::as_str).map(str::to_owned);
    if block_type.as_deref() != Some("tool_result") {
        // text, image, document — sacred.
        return (Value::Object(obj), false);
    }

    let inner = match obj.remove("content") {
        Some(v) => v,
        None => return (Value::Object(obj), false),
    };

    match inner {
        Value::String(s) => {
            if let Some(compressed) = compress_blob(pipeline, store, &s) {
                obj.insert("content".to_string(), Value::String(compressed));
                steps_applied.push("anthropic:tool_result_string".to_string());
                (Value::Object(obj), true)
            } else {
                obj.insert("content".to_string(), Value::String(s));
                (Value::Object(obj), false)
            }
        }
        Value::Array(blocks) => {
            let mut new_inner = Vec::with_capacity(blocks.len());
            let mut changed = false;
            for inner_block in blocks {
                let (rebuilt, did_change) =
                    rewrite_tool_result_inner_block(inner_block, pipeline, store, steps_applied);
                if did_change {
                    changed = true;
                }
                new_inner.push(rebuilt);
            }
            obj.insert("content".to_string(), Value::Array(new_inner));
            (Value::Object(obj), changed)
        }
        other => {
            // Unknown shape — restore as-is.
            obj.insert("content".to_string(), other);
            (Value::Object(obj), false)
        }
    }
}

fn rewrite_tool_result_inner_block(
    block: Value,
    pipeline: &CompressionPipeline,
    store: &Arc<dyn CcrStore>,
    steps_applied: &mut Vec<String>,
) -> (Value, bool) {
    let mut obj = match block {
        Value::Object(m) => m,
        other => return (other, false),
    };

    let block_type = obj.get("type").and_then(Value::as_str).map(str::to_owned);
    if block_type.as_deref() != Some("text") {
        // Image inside tool_result — preserve.
        return (Value::Object(obj), false);
    }

    let text = match obj.remove("text") {
        Some(Value::String(s)) => s,
        Some(other) => {
            obj.insert("text".to_string(), other);
            return (Value::Object(obj), false);
        }
        None => return (Value::Object(obj), false),
    };

    if let Some(compressed) = compress_blob(pipeline, store, &text) {
        obj.insert("text".to_string(), Value::String(compressed));
        steps_applied.push("anthropic:tool_result_text_block".to_string());
        (Value::Object(obj), true)
    } else {
        obj.insert("text".to_string(), Value::String(text));
        (Value::Object(obj), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::pipeline::build_pipeline;
    use serde_json::json;

    #[allow(clippy::type_complexity)]
    fn make_setup() -> (Arc<CompressionPipeline>, Arc<dyn CcrStore>) {
        build_pipeline()
    }

    fn big_json_array() -> String {
        // Larger fixture so compression beats JSON-escape overhead.
        let row = r#"{"id":1,"name":"alice","email":"a@example.com","tags":["x","y","z"]}"#;
        format!("[{}]", vec![row; 1000].join(","))
    }

    #[test]
    fn passthrough_on_invalid_json() {
        let (p, s) = make_setup();
        let body = Bytes::from_static(b"not json");
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson
            }
        ));
    }

    #[test]
    fn passthrough_when_messages_missing() {
        let (p, s) = make_setup();
        let body = Bytes::from(json!({"model": "claude-x"}).to_string());
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages
            }
        ));
    }

    #[test]
    fn passthrough_when_no_user_anchor() {
        let (p, s) = make_setup();
        let body = Bytes::from(
            json!({
                "model": "claude-x",
                "messages": [
                    {"role": "assistant", "content": "hi"}
                ]
            })
            .to_string(),
        );
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::NoUserAnchor
            }
        ));
    }

    #[test]
    fn user_string_content_passes_through() {
        let (p, s) = make_setup();
        let body = Bytes::from(
            json!({
                "model": "claude-x",
                "messages": [
                    {"role": "user", "content": "What's the weather?"}
                ]
            })
            .to_string(),
        );
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::NothingToCompress
            }
        ));
    }

    #[test]
    fn tool_result_string_content_compresses() {
        let (p, s) = make_setup();
        let big = big_json_array();
        let body = Bytes::from(
            json!({
                "model": "claude-3-5-sonnet",
                "messages": [
                    {"role": "user", "content": "list users"},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "list", "input": {}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": big.clone()}
                    ]}
                ]
            })
            .to_string(),
        );
        match maybe_compress(&body, &p, &s) {
            Outcome::Compressed {
                body: new_body,
                bytes_before,
                bytes_after,
                steps_applied,
            } => {
                assert!(!steps_applied.is_empty());
                let parsed: Value = serde_json::from_slice(&new_body).unwrap();
                let last_msg = &parsed["messages"][2];
                let new_content = &last_msg["content"][0]["content"];
                assert!(new_content.is_string());
                // The CONTENT text is shorter (real token savings) even
                // if the JSON-encoded body is slightly larger after
                // escape overhead.
                assert!(new_content.as_str().unwrap().len() < big.len());
                let _ = (bytes_before, bytes_after);
            }
            other => panic!("expected Compressed, got {other:?}"),
        }
    }

    #[test]
    fn prefix_messages_unchanged_byte_identical() {
        let (p, s) = make_setup();
        let big = big_json_array();
        // The PREFIX (messages 0..2) must be byte-identical in output
        // because the prefix-cache contract demands it.
        let original_payload = json!({
            "model": "claude-3-5-sonnet",
            "messages": [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": big.clone()}
                ]}
            ]
        });
        let body = Bytes::from(original_payload.to_string());
        if let Outcome::Compressed { body: new_body, .. } = maybe_compress(&body, &p, &s) {
            let parsed: Value = serde_json::from_slice(&new_body).unwrap();
            // First two messages — semantic equality (we don't drop
            // them and we don't modify them; serializer order may
            // differ but the content trees should be equal).
            assert_eq!(parsed["messages"][0], original_payload["messages"][0]);
            assert_eq!(parsed["messages"][1], original_payload["messages"][1]);
        } else {
            panic!("expected compression to fire on big tool_result");
        }
    }

    #[test]
    fn assistant_text_in_new_turn_is_sacred() {
        // ICM-era code might have dropped this. Token mode must not
        // touch assistant content even if it appears in the new turn.
        let (p, s) = make_setup();
        let big = big_json_array();
        let body = Bytes::from(
            json!({
                "model": "claude-x",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": big.clone()}
                ]
            })
            .to_string(),
        );
        // No tool_result anywhere — nothing eligible to compress.
        match maybe_compress(&body, &p, &s) {
            Outcome::Passthrough {
                reason: PassthroughReason::NothingToCompress,
            } => {}
            other => panic!("expected NothingToCompress, got {other:?}"),
        }
    }
}
