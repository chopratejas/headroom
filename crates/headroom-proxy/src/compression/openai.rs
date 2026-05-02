//! OpenAI `/v1/chat/completions` token-mode compression.
//!
//! # Algorithm
//!
//! 1. Parse body. Read `messages`.
//! 2. Find the last user message index (LU). Freeze `messages[..LU]`.
//! 3. Walk new turn (messages[LU..]):
//!    - role=tool: compress `content` (string) via the pipeline
//!    - role=user/system/developer/assistant: passthrough
//!
//! # What we never touch
//!
//! - `role: system|developer` (sacred framing)
//! - `role: user` (sacred input)
//! - `role: assistant` text or tool_calls
//! - Any non-`messages` field (`tools`, `tool_choice`, `max_tokens`,
//!   `max_completion_tokens`, `temperature`, `seed`, `response_format`,
//!   `stream`, ...)
//! - Anything in the prefix
//! - List-shaped content with image_url / input_audio parts (we don't
//!   walk into multimodal content; we simply leave non-string content
//!   alone — the relevant compression target is tool messages whose
//!   content is a string)

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

    // No bytes_after gate — see anthropic::maybe_compress note;
    // tokens billed track raw text length, not JSON-escaped wire bytes.

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

fn rewrite_message(
    msg: Value,
    pipeline: &CompressionPipeline,
    store: &Arc<dyn CcrStore>,
    steps_applied: &mut Vec<String>,
) -> (Value, bool) {
    let role = msg.get("role").and_then(Value::as_str);
    if role != Some("tool") {
        // user / system / developer / assistant — sacred.
        return (msg, false);
    }

    let mut obj = match msg {
        Value::Object(m) => m,
        other => return (other, false),
    };

    let content = match obj.remove("content") {
        Some(Value::String(s)) => s,
        Some(other) => {
            // Non-string content (rare on tool messages; possibly a
            // structured payload). Leave alone — we don't speculate.
            obj.insert("content".to_string(), other);
            return (Value::Object(obj), false);
        }
        None => return (Value::Object(obj), false),
    };

    if let Some(compressed) = compress_blob(pipeline, store, &content) {
        obj.insert("content".to_string(), Value::String(compressed));
        steps_applied.push("openai:tool_message".to_string());
        (Value::Object(obj), true)
    } else {
        obj.insert("content".to_string(), Value::String(content));
        (Value::Object(obj), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::pipeline::build_pipeline;
    use serde_json::json;

    fn setup() -> (Arc<CompressionPipeline>, Arc<dyn CcrStore>) {
        build_pipeline()
    }

    fn big_json() -> String {
        let row = r#"{"id":1,"name":"alice","email":"a@example.com","tags":["x","y","z"]}"#;
        format!("[{}]", vec![row; 1000].join(","))
    }

    #[test]
    fn passthrough_on_invalid_json() {
        let (p, s) = setup();
        let body = Bytes::from_static(b"nope");
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson
            }
        ));
    }

    #[test]
    fn user_only_passes_through() {
        let (p, s) = setup();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}]
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
    fn tool_message_in_new_turn_compresses() {
        let (p, s) = setup();
        let big = big_json();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "list users"},
                    {"role": "assistant", "content": null,
                     "tool_calls": [{"id": "c1", "type": "function",
                                     "function": {"name": "list", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": big.clone()}
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
                let _ = (bytes_before, bytes_after);
                assert_eq!(steps_applied, vec!["openai:tool_message".to_string()]);
                let parsed: Value = serde_json::from_slice(&new_body).unwrap();
                let new_tool_content = parsed["messages"][2]["content"]
                    .as_str()
                    .unwrap()
                    .to_string();
                assert!(new_tool_content.len() < big.len());
            }
            other => panic!("expected Compressed, got {other:?}"),
        }
    }

    #[test]
    fn assistant_text_in_new_turn_is_sacred() {
        let (p, s) = setup();
        let big = big_json();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "ask"},
                    {"role": "assistant", "content": big}
                ]
            })
            .to_string(),
        );
        // No tool message → nothing eligible.
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::NothingToCompress
            }
        ));
    }

    #[test]
    fn prefix_tool_message_not_compressed() {
        // Tool message in the PREFIX (before last user turn) must
        // be preserved — it's already in the provider's cache.
        let (p, s) = setup();
        let big = big_json();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "earlier"},
                    {"role": "assistant", "content": null,
                     "tool_calls": [{"id": "c1", "type": "function",
                                     "function": {"name": "list", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": big.clone()},
                    {"role": "assistant", "content": "summary"},
                    {"role": "user", "content": "follow-up"}
                ]
            })
            .to_string(),
        );
        // last_user_idx = 4. Tool message at idx=2 is in prefix
        // → never touched → no compression even though it's huge.
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::NothingToCompress
            }
        ));
    }
}
