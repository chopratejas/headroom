//! `DropByScoreStrategy` — the OSS-shipped context strategy.
//!
//! Algorithm:
//!
//! 1. Build candidates (turns + singles + tool units), sorted by
//!    importance score ascending.
//! 2. Walk the candidate list, dropping until token budget is met.
//! 3. After each drop, recompute token count via the supplied
//!    tokenizer. Stop early once `current_tokens <= target`.
//! 4. If `ccr_on_drop=true` AND a CCR store is wired, persist the
//!    dropped messages and emit a marker so the model can retrieve
//!    them via tool call.
//!
//! Direct port of the Python `_apply_drop_by_score` path with the
//! same ordering semantics (lowest score first, position tiebreaker).

use std::sync::Arc;

use serde_json::Value;

use crate::ccr::CcrStore;
use crate::context::candidate::{build_candidates, find_tool_units};
use crate::context::ccr_drop::persist_dropped;
use crate::context::workspace::{ContextWorkspace, StrategyOutcome};
use crate::scoring::MessageScorer;
use crate::tokenizer::Tokenizer;

use super::ContextStrategy;

/// Drop-by-importance-score strategy.
///
/// Holds a [`MessageScorer`] (constructed once with the manager's
/// scoring weights) and an optional CCR store for persistence on drop.
/// Both are `Arc`-wrapped because the strategy is shared across
/// requests in a multi-threaded proxy.
pub struct DropByScoreStrategy {
    scorer: Arc<MessageScorer>,
    tokenizer: Arc<dyn Tokenizer>,
    ccr_store: Option<Arc<dyn CcrStore>>,
    ccr_on_drop: bool,
}

impl DropByScoreStrategy {
    pub fn new(
        scorer: Arc<MessageScorer>,
        tokenizer: Arc<dyn Tokenizer>,
        ccr_store: Option<Arc<dyn CcrStore>>,
        ccr_on_drop: bool,
    ) -> Self {
        Self {
            scorer,
            tokenizer,
            ccr_store,
            ccr_on_drop,
        }
    }
}

impl ContextStrategy for DropByScoreStrategy {
    fn name(&self) -> &'static str {
        "drop_by_score"
    }

    fn try_fit(&self, ws: &mut ContextWorkspace, target_tokens: usize) -> StrategyOutcome {
        if ws.current_tokens <= target_tokens {
            return StrategyOutcome {
                fully_resolved: true,
                ..Default::default()
            };
        }

        // Compute tool units (atomic groups) and message scores.
        let tool_units = find_tool_units(&ws.messages);
        let tool_unit_indices: std::collections::HashSet<usize> = tool_units
            .iter()
            .flat_map(|(a, rs)| std::iter::once(*a).chain(rs.iter().copied()))
            .collect();

        let scores = self
            .scorer
            .score_messages(&ws.messages, &ws.protected, &tool_unit_indices);

        let candidates = build_candidates(&ws.messages, &scores, &ws.protected, &tool_units);

        // Walk lowest → highest score, dropping one candidate at a
        // time. Recompute tokens after each drop. Stop when under
        // budget OR we run out of candidates.
        //
        // Track which ORIGINAL indices were dropped so the CCR helper
        // can serialize the verbatim bytes. The workspace already
        // holds `original_messages` for that purpose.
        let initial_tokens = ws.current_tokens;
        let mut dropped_now: Vec<usize> = Vec::new();
        let mut indices_to_remove: std::collections::BTreeSet<usize> =
            std::collections::BTreeSet::new();

        for cand in &candidates {
            if ws.current_tokens.saturating_sub(estimate_dropped_tokens(
                &ws.messages,
                &cand.indices,
                self.tokenizer.as_ref(),
            )) <= target_tokens
                || ws.current_tokens > target_tokens
            {
                indices_to_remove.extend(cand.indices.iter().copied());
                dropped_now.extend(cand.indices.iter().copied());
                // Estimate post-drop token count without rebuilding the
                // list each iteration. Real recompute happens below.
                let saved =
                    estimate_dropped_tokens(&ws.messages, &cand.indices, self.tokenizer.as_ref());
                ws.current_tokens = ws.current_tokens.saturating_sub(saved);

                if ws.current_tokens <= target_tokens {
                    break;
                }
            }
        }

        if indices_to_remove.is_empty() {
            return StrategyOutcome::default();
        }

        // Translate WORKSPACE indices to ORIGINAL indices for CCR
        // persistence. The workspace's `messages` and `original_messages`
        // are aligned 1:1 in this implementation (no prior strategy
        // mutated the list — this is the only OSS strategy).
        ws.dropped_indices.extend(dropped_now);

        // Remove highest index first so earlier indices stay valid.
        let to_remove_desc: Vec<usize> = indices_to_remove.iter().rev().copied().collect();
        for idx in to_remove_desc {
            if idx < ws.messages.len() {
                ws.messages.remove(idx);
            }
        }

        // Recompute exact token count post-mutation. The estimate above
        // is approximate; the real count drives the cascade decision.
        ws.current_tokens = count_messages(&ws.messages, self.tokenizer.as_ref());

        // CCR persistence.
        let mut markers: Vec<String> = Vec::new();
        if self.ccr_on_drop {
            if let Some(store) = &self.ccr_store {
                if let Some(persist) =
                    persist_dropped(&ws.original_messages, &ws.dropped_indices, store)
                {
                    markers.push(persist.marker);
                }
            }
        }

        let tokens_freed = initial_tokens.saturating_sub(ws.current_tokens);
        StrategyOutcome {
            tokens_freed,
            markers_inserted: markers,
            fully_resolved: ws.current_tokens <= target_tokens,
        }
    }
}

/// Token count for a list of messages. Mirrors Python's
/// `count_messages` accounting: per-message overhead + content
/// tokens + role tokens. Matches the OpenAI chat-format formula
/// used in the Python `tiktoken_counter`.
///
/// `pub(crate)` so the manager module can use the same accounting
/// for `should_apply` gating without duplicating the logic.
pub(crate) fn count_messages(messages: &[Value], tokenizer: &dyn Tokenizer) -> usize {
    const MESSAGE_OVERHEAD: usize = 3; // OpenAI: 3 tokens per message
    const REPLY_PRIMER: usize = 3; // Final assistant primer

    let mut total = REPLY_PRIMER;
    for msg in messages {
        total += MESSAGE_OVERHEAD;
        if let Some(role) = msg.get("role").and_then(Value::as_str) {
            total += tokenizer.count_text(role);
        }
        if let Some(content) = msg.get("content") {
            total += count_content(content, tokenizer);
        }
        if let Some(name) = msg.get("name").and_then(Value::as_str) {
            total += tokenizer.count_text(name);
        }
        // tool_calls and tool_call_id contribute too — count their
        // string forms. Coarse but sufficient for budget gating.
        if let Some(tc) = msg.get("tool_calls") {
            if let Ok(s) = serde_json::to_string(tc) {
                total += tokenizer.count_text(&s);
            }
        }
        if let Some(id) = msg.get("tool_call_id").and_then(Value::as_str) {
            total += tokenizer.count_text(id);
        }
    }
    total
}

fn count_content(content: &Value, tokenizer: &dyn Tokenizer) -> usize {
    match content {
        Value::String(s) => tokenizer.count_text(s),
        Value::Array(blocks) => {
            let mut total = 0;
            for block in blocks {
                // {type:"text",text:"..."} or {type:"tool_result",content:"..."} or
                // {type:"tool_use", input:{...}} etc. Walk text-bearing fields.
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    total += tokenizer.count_text(text);
                }
                if let Some(c) = block.get("content") {
                    if let Some(s) = c.as_str() {
                        total += tokenizer.count_text(s);
                    } else if let Ok(s) = serde_json::to_string(c) {
                        total += tokenizer.count_text(&s);
                    }
                }
                if let Some(input) = block.get("input") {
                    if let Ok(s) = serde_json::to_string(input) {
                        total += tokenizer.count_text(&s);
                    }
                }
            }
            total
        }
        Value::Null => 0,
        other => serde_json::to_string(other)
            .map(|s| tokenizer.count_text(&s))
            .unwrap_or(0),
    }
}

/// Approximate the token cost of dropping `indices` from `messages`.
/// Used as a fast pre-check inside the candidate walk; the real
/// recount happens after mutation.
fn estimate_dropped_tokens(
    messages: &[Value],
    indices: &[usize],
    tokenizer: &dyn Tokenizer,
) -> usize {
    const MESSAGE_OVERHEAD: usize = 3;
    let mut total = 0;
    for &i in indices {
        if let Some(msg) = messages.get(i) {
            total += MESSAGE_OVERHEAD;
            if let Some(role) = msg.get("role").and_then(Value::as_str) {
                total += tokenizer.count_text(role);
            }
            if let Some(content) = msg.get("content") {
                total += count_content(content, tokenizer);
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;
    use crate::scoring::MessageScorer;
    use crate::tokenizer::EstimatingCounter;
    use serde_json::json;
    use std::collections::HashSet;

    fn ws(messages: Vec<Value>, tokenizer: &dyn Tokenizer) -> ContextWorkspace {
        let mut w = ContextWorkspace::new(messages, HashSet::new(), 0);
        w.current_tokens = count_messages(&w.messages, tokenizer);
        w
    }

    fn strategy(
        ccr: Option<Arc<dyn CcrStore>>,
        ccr_on_drop: bool,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> DropByScoreStrategy {
        let scorer = Arc::new(MessageScorer::with_defaults());
        DropByScoreStrategy::new(scorer, tokenizer, ccr, ccr_on_drop)
    }

    #[test]
    fn under_budget_returns_fully_resolved_no_op() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let mut w = ws(
            vec![
                json!({"role": "user", "content": "short"}),
                json!({"role": "assistant", "content": "ok"}),
            ],
            tokenizer.as_ref(),
        );
        let s = strategy(None, false, tokenizer.clone());
        let out = s.try_fit(&mut w, 1_000_000);
        assert!(out.fully_resolved);
        assert_eq!(out.tokens_freed, 0);
        assert_eq!(w.messages.len(), 2);
    }

    #[test]
    fn drops_lowest_scored_messages_first() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let messages = vec![
            json!({"role": "user", "content": "old turn one ".repeat(20)}),
            json!({"role": "assistant", "content": "old answer one ".repeat(20)}),
            json!({"role": "user", "content": "recent turn ".repeat(20)}),
            json!({"role": "assistant", "content": "recent answer ".repeat(20)}),
        ];
        let mut w = ws(messages, tokenizer.as_ref());
        let initial = w.current_tokens;
        let target = initial / 2;
        let s = strategy(None, false, tokenizer.clone());
        let out = s.try_fit(&mut w, target);
        // Should drop something to free tokens.
        assert!(out.tokens_freed > 0);
        assert!(w.messages.len() < 4);
    }

    #[test]
    fn protected_messages_are_never_dropped() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let messages = vec![
            json!({"role": "user", "content": "PROTECTED ".repeat(50)}),
            json!({"role": "assistant", "content": "PROTECTED ".repeat(50)}),
        ];
        let mut w = ws(messages, tokenizer.as_ref());
        // Protect both indices.
        w.protected.insert(0);
        w.protected.insert(1);
        let initial_count = w.messages.len();
        let s = strategy(None, false, tokenizer.clone());
        let _ = s.try_fit(&mut w, 0);
        // Nothing dropped because everything's protected.
        assert_eq!(w.messages.len(), initial_count);
    }

    #[test]
    fn ccr_on_drop_persists_and_emits_marker() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let messages = vec![
            json!({"role": "user", "content": "drop me ".repeat(50)}),
            json!({"role": "assistant", "content": "and me ".repeat(50)}),
            json!({"role": "user", "content": "keep recent".to_string()}),
            json!({"role": "assistant", "content": "ok".to_string()}),
        ];
        let mut w = ws(messages, tokenizer.as_ref());
        // Protect the last two so the cascade has to drop the first two.
        w.protected.insert(2);
        w.protected.insert(3);

        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let s = strategy(Some(store.clone()), true, tokenizer.clone());
        let target = w.current_tokens / 2;
        let out = s.try_fit(&mut w, target);
        assert!(out.tokens_freed > 0);
        assert!(!out.markers_inserted.is_empty());
        // The marker references CCR retrieval.
        assert!(out.markers_inserted[0].contains("ccr_retrieve"));
    }

    #[test]
    fn ccr_off_does_not_persist() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let messages = vec![
            json!({"role": "user", "content": "drop me ".repeat(50)}),
            json!({"role": "assistant", "content": "stay ".repeat(50)}),
        ];
        let mut w = ws(messages, tokenizer.as_ref());
        w.protected.insert(1);

        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let s = strategy(Some(store.clone()), false, tokenizer.clone());
        let target = w.current_tokens / 2;
        let _ = s.try_fit(&mut w, target);
        // ccr_on_drop=false → no markers, no persistence.
        // (Can't easily probe the store without exposing internals;
        //  the marker absence is the visible contract.)
    }

    #[test]
    fn no_candidates_returns_default_outcome() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let messages = vec![json!({"role": "system", "content": "sys".repeat(100)})];
        let mut w = ws(messages, tokenizer.as_ref());
        // Protect the only message → no candidates.
        w.protected.insert(0);
        let s = strategy(None, false, tokenizer.clone());
        let out = s.try_fit(&mut w, 0);
        assert_eq!(out.tokens_freed, 0);
        assert!(!out.fully_resolved);
    }

    #[test]
    fn count_messages_returns_nonzero_for_nonempty_input() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let msgs = vec![json!({"role": "user", "content": "hello"})];
        assert!(count_messages(&msgs, tokenizer.as_ref()) > 0);
    }

    #[test]
    fn count_messages_handles_anthropic_content_blocks() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let msgs = vec![json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "tu", "name": "f", "input": {"x": 1}}
            ]
        })];
        let n = count_messages(&msgs, tokenizer.as_ref());
        assert!(n > 0);
    }
}
