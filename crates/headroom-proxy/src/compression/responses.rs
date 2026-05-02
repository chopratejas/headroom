//! OpenAI `/v1/responses` token-mode compression.
//!
//! # Algorithm
//!
//! 1. Parse body. If `previous_response_id` is set: passthrough (the
//!    server holds the conversation; this body is one new turn that's
//!    not worth compressing).
//! 2. Read `input`. If it's a string: passthrough (one user message,
//!    nothing to compress). If it's an array of items: walk it.
//! 3. Find the **last user message item** (item with `role:user`,
//!    optionally `type:message`). Freeze items before it.
//! 4. Walk new-turn items:
//!    - `function_call_output`: COMPRESS the `output` string
//!    - `message` items (any role): passthrough
//!    - `function_call`: passthrough (small structured call)
//!    - `reasoning`: passthrough (encrypted_content, signed)
//!    - `*_call` (computer_call, file_search_call, etc.): passthrough
//!      for now — these can carry big payloads but we don't yet have
//!      the per-shape compression strategy and the safest move is to
//!      preserve them
//!
//! # What we never touch
//!
//! - `instructions` (top-level system prompt analog)
//! - `prompt_cache_key` (the *binding* for the cache; mustn't change)
//! - `tools`, `tool_choice`, `reasoning.effort`, `max_output_tokens`,
//!   `store`, `metadata`, etc.
//! - User text, system/developer text, assistant text in messages
//! - Reasoning items (`encrypted_content` is opaque + signed)
//! - Image/file/audio items / content parts
//! - Anything in the prefix

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

    // Stateful mode: server holds the conversation. Skip.
    if parsed
        .get("previous_response_id")
        .and_then(Value::as_str)
        .is_some()
    {
        return Outcome::Passthrough {
            reason: PassthroughReason::StatefulMode,
        };
    }

    let items = match parsed.get_mut("input") {
        // Array shape — walk it.
        Some(Value::Array(_)) => match parsed["input"].take() {
            Value::Array(a) => a,
            _ => unreachable!(),
        },
        // String shape — single user message, nothing to compress.
        Some(Value::String(_)) => {
            return Outcome::Passthrough {
                reason: PassthroughReason::NothingToCompress,
            };
        }
        _ => {
            return Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            };
        }
    };

    let last_user = match find_last_user_item_idx(&items) {
        Some(i) => i,
        None => {
            parsed["input"] = Value::Array(items);
            return Outcome::Passthrough {
                reason: PassthroughReason::NoUserAnchor,
            };
        }
    };

    let bytes_before = body.len();
    let mut steps_applied: Vec<String> = Vec::new();
    let mut new_items: Vec<Value> = Vec::with_capacity(items.len());
    let mut any_changed = false;

    for (idx, item) in items.into_iter().enumerate() {
        if idx < last_user {
            new_items.push(item);
            continue;
        }
        let (rebuilt, changed) = rewrite_item(item, pipeline, store, &mut steps_applied);
        if changed {
            any_changed = true;
        }
        new_items.push(rebuilt);
    }

    parsed["input"] = Value::Array(new_items);

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

/// A user-message item in /v1/responses input is either:
/// - `{"type": "message", "role": "user", ...}`, or
/// - `{"role": "user", ...}` (legacy / shorthand — `type` may be absent)
fn find_last_user_item_idx(items: &[Value]) -> Option<usize> {
    items.iter().rposition(|it| {
        let role = it.get("role").and_then(Value::as_str);
        if role != Some("user") {
            return false;
        }
        match it.get("type").and_then(Value::as_str) {
            None | Some("message") => true,
            // Items like `function_call`/`reasoning` shouldn't carry
            // role:user, but reject defensively if they do.
            _ => false,
        }
    })
}

fn rewrite_item(
    item: Value,
    pipeline: &CompressionPipeline,
    store: &Arc<dyn CcrStore>,
    steps_applied: &mut Vec<String>,
) -> (Value, bool) {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");

    if item_type != "function_call_output" {
        // message, function_call, reasoning, *_call, image_generation_call
        // — all sacred for the current scope.
        return (item, false);
    }

    let mut obj = match item {
        Value::Object(m) => m,
        other => return (other, false),
    };

    let output = match obj.remove("output") {
        Some(Value::String(s)) => s,
        Some(other) => {
            obj.insert("output".to_string(), other);
            return (Value::Object(obj), false);
        }
        None => return (Value::Object(obj), false),
    };

    if let Some(compressed) = compress_blob(pipeline, store, &output) {
        obj.insert("output".to_string(), Value::String(compressed));
        steps_applied.push("responses:function_call_output".to_string());
        (Value::Object(obj), true)
    } else {
        obj.insert("output".to_string(), Value::String(output));
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
    fn previous_response_id_skips() {
        let (p, s) = setup();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "previous_response_id": "resp_xyz",
                "input": "hello"
            })
            .to_string(),
        );
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::StatefulMode
            }
        ));
    }

    #[test]
    fn string_input_passes_through() {
        let (p, s) = setup();
        let body = Bytes::from(json!({"model": "gpt-4o-mini", "input": "say hi"}).to_string());
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::NothingToCompress
            }
        ));
    }

    #[test]
    fn no_user_item_skips() {
        let (p, s) = setup();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "input": [
                    {"type": "message", "role": "developer", "content": "be terse"}
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
    fn function_call_output_in_new_turn_compresses() {
        let (p, s) = setup();
        let big = big_json();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "input": [
                    {"type": "message", "role": "user", "content": "list users"},
                    {"type": "function_call", "call_id": "c1",
                     "name": "list", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "c1", "output": big.clone()}
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
                assert_eq!(
                    steps_applied,
                    vec!["responses:function_call_output".to_string()]
                );
                let parsed: Value = serde_json::from_slice(&new_body).unwrap();
                let new_output = parsed["input"][2]["output"].as_str().unwrap().to_string();
                assert!(new_output.len() < big.len());
            }
            other => panic!("expected Compressed, got {other:?}"),
        }
    }

    #[test]
    fn reasoning_item_preserved_verbatim() {
        let (p, s) = setup();
        // Reasoning items in the new turn must NOT be touched even
        // though they could be huge — `encrypted_content` is opaque
        // and signed. Future work may compress old reasoning items
        // wholesale, but never in-place.
        let body = Bytes::from(
            json!({
                "model": "o3-mini",
                "input": [
                    {"type": "message", "role": "user", "content": "think"},
                    {"type": "reasoning", "id": "rs_1",
                     "summary": "I should think step by step.",
                     "encrypted_content": "OPAQUE-BLOB"}
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
    fn prefix_function_call_output_not_compressed() {
        let (p, s) = setup();
        let big = big_json();
        let body = Bytes::from(
            json!({
                "model": "gpt-4o-mini",
                "input": [
                    {"type": "message", "role": "user", "content": "earlier turn"},
                    {"type": "function_call", "call_id": "c1",
                     "name": "list", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "c1", "output": big.clone()},
                    {"type": "message", "role": "assistant", "content": "summary"},
                    {"type": "message", "role": "user", "content": "follow-up"}
                ]
            })
            .to_string(),
        );
        // last_user_idx = 4. Output at idx=2 is in prefix → preserved.
        assert!(matches!(
            maybe_compress(&body, &p, &s),
            Outcome::Passthrough {
                reason: PassthroughReason::NothingToCompress
            }
        ));
    }
}
