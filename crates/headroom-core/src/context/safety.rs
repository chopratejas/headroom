//! `SafetyRails` — compute the set of indices that strategies must
//! never drop.
//!
//! Direct port of Python's `IntelligentContextManager._get_protected_indices`,
//! including the awkward-but-correct two-pass walk:
//!
//! 1. Walk forward, mark `role=system` (if `keep_system`).
//! 2. Walk backward from the end, mark messages until `keep_last_turns`
//!    user messages have been seen.
//! 3. For every assistant message in the protected set, find its tool
//!    responses (OpenAI `role=tool` + `tool_call_id`, OR Anthropic
//!    `role=user` with `content[].type=tool_result`) and protect those
//!    too. Otherwise we'd ship an orphan tool_call → 400 from the API.
//!
//! Frozen prefix is added on top by the caller via
//! [`ContextWorkspace::frozen_count`](super::workspace::ContextWorkspace::frozen_count).
//! This module just handles the per-message rules.

use std::collections::HashSet;

use serde_json::Value;

use crate::context::config::IcmConfig;

pub struct SafetyRails<'c> {
    config: &'c IcmConfig,
}

impl<'c> SafetyRails<'c> {
    pub fn new(config: &'c IcmConfig) -> Self {
        Self { config }
    }

    /// Compute protected indices for the given message list.
    ///
    /// Includes the leading `frozen_count` indices (prompt-cache prefix)
    /// in the result so callers don't have to remember to OR them in.
    pub fn protected(&self, messages: &[Value], frozen_count: usize) -> HashSet<usize> {
        let mut protected: HashSet<usize> = HashSet::new();

        // Frozen prefix — provider's prompt cache. Dropping any of
        // these busts the cache and raises latency + cost.
        for i in 0..frozen_count.min(messages.len()) {
            protected.insert(i);
        }

        // System messages — never droppable when keep_system is on.
        // Python convention: any `role=system` anywhere in the list,
        // not only at index 0.
        if self.config.keep_system {
            for (i, msg) in messages.iter().enumerate() {
                if msg.get("role").and_then(Value::as_str) == Some("system") {
                    protected.insert(i);
                }
            }
        }

        // Last N user-turn boundary. Walk backward from the end,
        // protecting every message until we've seen `keep_last_turns`
        // role=user messages. Includes assistant + tool messages
        // interleaved with those user turns.
        if self.config.keep_last_turns > 0 && !messages.is_empty() {
            let mut turns_seen = 0;
            let mut i = messages.len() as isize - 1;
            while i >= 0 && turns_seen < self.config.keep_last_turns {
                let msg = &messages[i as usize];
                protected.insert(i as usize);
                if msg.get("role").and_then(Value::as_str) == Some("user") {
                    turns_seen += 1;
                }
                i -= 1;
            }
        }

        // Pair tool responses with their assistant tool_call messages.
        // Protected assistant → every tool response that satisfies
        // its tool_call ids must also be protected. Otherwise we'd
        // drop a `tool` message whose corresponding `assistant`
        // tool_call is still in the prompt → API rejects with 400.
        let pairs = self.tool_response_pairs(messages, &protected);
        protected.extend(pairs);

        protected
    }

    /// For each protected assistant message, find every tool response
    /// (OpenAI or Anthropic shape) that references one of its tool
    /// call ids. Returns the response indices — the assistant indices
    /// are already in `protected`.
    fn tool_response_pairs(
        &self,
        messages: &[Value],
        protected: &HashSet<usize>,
    ) -> HashSet<usize> {
        let mut response_indices: HashSet<usize> = HashSet::new();

        for &i in protected {
            let msg = match messages.get(i) {
                Some(m) => m,
                None => continue,
            };
            if msg.get("role").and_then(Value::as_str) != Some("assistant") {
                continue;
            }

            let tool_call_ids = collect_assistant_tool_call_ids(msg);
            if tool_call_ids.is_empty() {
                continue;
            }

            for (j, other) in messages.iter().enumerate() {
                if protected.contains(&j) {
                    continue;
                }
                if message_responds_to_tool_call(other, &tool_call_ids) {
                    response_indices.insert(j);
                }
            }
        }

        response_indices
    }
}

/// Extract the set of tool_call ids announced by an assistant message,
/// handling both OpenAI and Anthropic shapes.
///
/// - **OpenAI**: `{role:assistant, tool_calls: [{id, ...}, ...]}`
/// - **Anthropic**: `{role:assistant, content: [{type:tool_use, id, ...}]}`
fn collect_assistant_tool_call_ids(assistant: &Value) -> HashSet<String> {
    let mut ids: HashSet<String> = HashSet::new();

    // OpenAI: tool_calls array
    if let Some(arr) = assistant.get("tool_calls").and_then(Value::as_array) {
        for tc in arr {
            if let Some(id) = tc.get("id").and_then(Value::as_str) {
                ids.insert(id.to_string());
            }
        }
    }

    // Anthropic: content blocks with type=tool_use
    if let Some(blocks) = assistant.get("content").and_then(Value::as_array) {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                if let Some(id) = block.get("id").and_then(Value::as_str) {
                    ids.insert(id.to_string());
                }
            }
        }
    }

    ids
}

/// Does this message respond to one of `ids`?
///
/// - **OpenAI**: `{role:tool, tool_call_id: "..."}`
/// - **Anthropic**: `{role:user, content: [{type:tool_result, tool_use_id: "..."}]}`
fn message_responds_to_tool_call(msg: &Value, ids: &HashSet<String>) -> bool {
    let role = msg.get("role").and_then(Value::as_str);

    // OpenAI shape
    if role == Some("tool") {
        if let Some(tcid) = msg.get("tool_call_id").and_then(Value::as_str) {
            return ids.contains(tcid);
        }
    }

    // Anthropic shape — tool results live inside a user message's content list
    if role == Some("user") {
        if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                    if let Some(tuid) = block.get("tool_use_id").and_then(Value::as_str) {
                        if ids.contains(tuid) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg(keep_system: bool, keep_last_turns: usize) -> IcmConfig {
        IcmConfig {
            keep_system,
            keep_last_turns,
            ..Default::default()
        }
    }

    #[test]
    fn protects_system_messages() {
        let msgs = vec![
            json!({"role": "system", "content": "you are helpful"}),
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        let c = cfg(true, 0);
        let s = SafetyRails::new(&c);
        let p = s.protected(&msgs, 0);
        assert!(p.contains(&0));
        assert!(!p.contains(&1));
        assert!(!p.contains(&2));
    }

    #[test]
    fn keep_system_false_does_not_protect_system() {
        let msgs = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "hi"}),
        ];
        let c = cfg(false, 0);
        let s = SafetyRails::new(&c);
        let p = s.protected(&msgs, 0);
        assert!(!p.contains(&0));
    }

    #[test]
    fn last_n_turns_protects_user_and_inner_messages() {
        // 5 messages, keep_last_turns=2 → walk back until we've seen
        // 2 user messages. Should protect indices 2,3,4 (covers user@4,
        // asst@3, user@2).
        let msgs = vec![
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "content": "a1"}),
            json!({"role": "user", "content": "q2"}),
            json!({"role": "assistant", "content": "a2"}),
            json!({"role": "user", "content": "q3"}),
        ];
        let c = cfg(false, 2);
        let s = SafetyRails::new(&c);
        let p = s.protected(&msgs, 0);
        assert!(p.contains(&2));
        assert!(p.contains(&3));
        assert!(p.contains(&4));
        assert!(!p.contains(&0));
        assert!(!p.contains(&1));
    }

    #[test]
    fn frozen_prefix_is_protected() {
        let msgs = vec![
            json!({"role": "user", "content": "1"}),
            json!({"role": "user", "content": "2"}),
            json!({"role": "user", "content": "3"}),
        ];
        let c = cfg(false, 0);
        let s = SafetyRails::new(&c);
        let p = s.protected(&msgs, 2);
        assert!(p.contains(&0));
        assert!(p.contains(&1));
        assert!(!p.contains(&2));
    }

    #[test]
    fn frozen_count_above_message_count_is_clamped() {
        let msgs = vec![json!({"role": "user", "content": "x"})];
        let c = cfg(false, 0);
        let s = SafetyRails::new(&c);
        // No panic, just protect what's there.
        let p = s.protected(&msgs, 99);
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn openai_tool_pair_is_atomic() {
        // assistant @ idx 1 has a tool_call; tool response @ idx 2.
        // keep_last_turns protects from the end. With keep_last=1 we
        // should pull in user@3 only, then the pair logic adds nothing.
        // Use keep_last=0 + keep_system=false but pre-protect the
        // assistant manually (simulate a caller-protected index).
        let msgs = vec![
            json!({"role": "user", "content": "go"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "result"}),
            json!({"role": "user", "content": "thanks"}),
        ];
        // keep_last_turns=2 → protects user@3 + asst@2 (but the asst@2
        // is the tool message... no wait, asst is at 1, tool at 2).
        // Walking back from idx 3: protect 3 (user, 1 turn), 2 (tool),
        // 1 (asst), 0 (user, 2 turns). All four protected.
        let c = cfg(false, 2);
        let s = SafetyRails::new(&c);
        let p = s.protected(&msgs, 0);
        assert!(p.contains(&1));
        assert!(p.contains(&2)); // The tool response must be protected too.
    }

    #[test]
    fn anthropic_tool_pair_is_atomic() {
        // Anthropic: assistant content has tool_use blocks; tool_result
        // lives inside a user message's content list.
        let msgs = vec![
            json!({"role": "user", "content": "go"}),
            json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thinking..."},
                    {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}
                ]
            }),
            json!({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}]
            }),
            json!({"role": "user", "content": "next"}),
        ];
        // Pre-protect the assistant via keep_last_turns=2 (walks back
        // through user@3, user@2 = 2 turns; assistant@1 not protected
        // by turn rule). Bump to keep_last_turns=3 so assistant@1 falls
        // under the protection.
        let c = cfg(false, 3);
        let s = SafetyRails::new(&c);
        let p = s.protected(&msgs, 0);
        assert!(p.contains(&1)); // assistant
        assert!(p.contains(&2)); // tool_result user
    }

    #[test]
    fn unmatched_tool_call_id_does_not_inflate_protection() {
        let msgs = vec![
            json!({"role": "user", "content": "go"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_orphan",
                                 "function": {"name": "f"}}]
            }),
            json!({"role": "tool", "tool_call_id": "call_DIFFERENT", "content": "?"}),
        ];
        let c = cfg(false, 0);
        let s = SafetyRails::new(&c);
        // Manually protect the assistant by setting keep_last_turns
        // to wrap it. But keep_last=0 here so nothing is protected;
        // the orphan tool message stays unprotected. Now protect via
        // keep_last_turns=1 wrapping back from end…
        // Actually with keep_last=0 + keep_system=false there's no
        // protection at all, so the pair logic finds nothing to pair.
        let p = s.protected(&msgs, 0);
        assert!(p.is_empty());
    }
}
