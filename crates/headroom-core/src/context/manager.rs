//! `IntelligentContextManager` — the cascade orchestrator.
//!
//! Holds the config, the safety-rails computer, and the registered
//! list of strategies. On `apply()`, it:
//!
//! 1. Tokenizes, exits early if under budget.
//! 2. Computes safety-rail protections (system, last-N-turns, frozen
//!    prefix, paired tool responses).
//! 3. Builds a [`ContextWorkspace`] and walks each registered strategy
//!    in registration order until a strategy reports `fully_resolved`
//!    or the list is exhausted.
//! 4. Returns the (possibly mutated) message list plus an
//!    [`ApplyResult`] describing what happened.
//!
//! OSS pre-registers exactly one strategy: [`DropByScoreStrategy`].
//! Enterprise calls `with_strategy` more times.

use std::sync::Arc;

use serde_json::Value;

use crate::ccr::CcrStore;
use crate::context::config::IcmConfig;
use crate::context::safety::SafetyRails;
use crate::context::strategy::{ContextStrategy, DropByScoreStrategy};
use crate::context::workspace::ContextWorkspace;
use crate::scoring::MessageScorer;
use crate::tokenizer::Tokenizer;

/// Per-call inputs that vary by request.
pub struct ApplyCtx {
    /// Provider's context-window size in tokens.
    pub model_limit: usize,
    /// Override the manager's `output_buffer_tokens`. `None` uses the
    /// configured default.
    pub output_buffer: Option<usize>,
    /// Number of leading messages that are part of the provider's
    /// prompt cache. Always protected. Provider-aware caller computes
    /// this; ICM just receives it.
    pub frozen_message_count: usize,
}

impl Default for ApplyCtx {
    fn default() -> Self {
        Self {
            model_limit: 128_000,
            output_buffer: None,
            frozen_message_count: 0,
        }
    }
}

/// Outcome of a single `apply()` call.
#[derive(Debug, Clone)]
pub struct ApplyResult {
    /// The (possibly modified) message list.
    pub messages: Vec<Value>,
    pub tokens_before: usize,
    pub tokens_after: usize,
    /// Names of strategies that ran (in order). Strategies that were
    /// short-circuited by an earlier `fully_resolved=true` don't appear.
    pub strategies_applied: Vec<&'static str>,
    /// Marker strings emitted by the cascade (e.g. CCR retrieval hints).
    pub markers_inserted: Vec<String>,
}

/// The orchestrator. Constructed once per process and reused across
/// requests; `Send + Sync`.
pub struct IntelligentContextManager {
    config: IcmConfig,
    tokenizer: Arc<dyn Tokenizer>,
    strategies: Vec<Box<dyn ContextStrategy>>,
}

impl IntelligentContextManager {
    /// Construct with the OSS default strategy stack: just
    /// [`DropByScoreStrategy`]. The scorer is built from the config's
    /// `scoring_weights`. Enterprise can call `with_strategy` to
    /// register additional strategies before serving requests.
    ///
    /// `ccr` is the CCR store used for drop persistence (and is also
    /// stored on the strategy). Pass `None` to disable CCR-on-drop
    /// even if `config.ccr_on_drop = true`.
    pub fn new(
        config: IcmConfig,
        tokenizer: Arc<dyn Tokenizer>,
        ccr: Option<Arc<dyn CcrStore>>,
    ) -> Self {
        let scorer = Arc::new(MessageScorer::new(
            Some(config.scoring_weights),
            None,
            None,
            0.1,
        ));
        let drop_strategy: Box<dyn ContextStrategy> = Box::new(DropByScoreStrategy::new(
            scorer,
            tokenizer.clone(),
            ccr,
            config.ccr_on_drop,
        ));
        Self {
            config,
            tokenizer,
            strategies: vec![drop_strategy],
        }
    }

    /// Append an Enterprise strategy. Strategies run in registration
    /// order; the OSS `DropByScoreStrategy` is always first unless the
    /// caller explicitly bypasses it via [`Self::without_default_strategy`].
    pub fn with_strategy(mut self, strategy: Box<dyn ContextStrategy>) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Drop the OSS default strategy from the stack. Use only when
    /// the caller wants a fully custom strategy chain — e.g. an
    /// Enterprise build that handles dropping itself.
    pub fn without_default_strategy(mut self) -> Self {
        // Removes the default DropByScoreStrategy registered in `new`.
        self.strategies.clear();
        self
    }

    pub fn config(&self) -> &IcmConfig {
        &self.config
    }

    /// Cheap pre-check. `true` means `apply()` would do work; `false`
    /// means the request is under budget and a no-op pass-through is
    /// safe (and faster — skips tokenization on the hot path).
    pub fn should_apply(
        &self,
        messages: &[Value],
        model_limit: usize,
        output_buffer: usize,
    ) -> bool {
        if !self.config.enabled {
            return false;
        }
        let current = count_messages(messages, self.tokenizer.as_ref());
        let available = model_limit.saturating_sub(output_buffer);
        current > available
    }

    /// Run the cascade.
    pub fn apply(&self, messages: Vec<Value>, ctx: ApplyCtx) -> ApplyResult {
        let output_buffer = ctx
            .output_buffer
            .unwrap_or(self.config.output_buffer_tokens);
        let target = ctx.model_limit.saturating_sub(output_buffer);

        let tokens_before = count_messages(&messages, self.tokenizer.as_ref());

        // Early exit: under budget OR disabled.
        if !self.config.enabled || tokens_before <= target {
            return ApplyResult {
                messages,
                tokens_before,
                tokens_after: tokens_before,
                strategies_applied: Vec::new(),
                markers_inserted: Vec::new(),
            };
        }

        // Compute safety rails + build workspace.
        let safety = SafetyRails::new(&self.config);
        let protected = safety.protected(&messages, ctx.frozen_message_count);
        let mut ws = ContextWorkspace::new(messages, protected, ctx.frozen_message_count);
        ws.current_tokens = tokens_before;

        let mut strategies_applied: Vec<&'static str> = Vec::new();
        let mut markers: Vec<String> = Vec::new();

        for strategy in &self.strategies {
            let outcome = strategy.try_fit(&mut ws, target);
            strategies_applied.push(strategy.name());
            markers.extend(outcome.markers_inserted);
            // Recompute protections — indices may have shifted post-drop.
            // Subsequent strategies need a fresh view.
            ws.protected = safety.protected(&ws.messages, ws.frozen_count);
            if outcome.fully_resolved {
                break;
            }
        }

        ApplyResult {
            tokens_before,
            tokens_after: ws.current_tokens,
            messages: ws.messages,
            strategies_applied,
            markers_inserted: markers,
        }
    }
}

// Re-export the strategy's count_messages for tests + manager-level
// gating. Keeping a single source of truth avoids drift between
// `should_apply` and the strategy's own accounting.
pub(crate) use crate::context::strategy::drop_by_score::count_messages;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;
    use crate::tokenizer::EstimatingCounter;
    use serde_json::json;

    fn manager() -> IntelligentContextManager {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        IntelligentContextManager::new(IcmConfig::default(), tokenizer, Some(store))
    }

    #[test]
    fn should_apply_returns_false_when_disabled() {
        let cfg = IcmConfig {
            enabled: false,
            ..Default::default()
        };
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingCounter::default());
        let m = IntelligentContextManager::new(cfg, tokenizer, None);
        let huge = vec![json!({"role": "user", "content": "x".repeat(1_000_000)})];
        assert!(!m.should_apply(&huge, 1000, 100));
    }

    #[test]
    fn should_apply_false_under_budget() {
        let m = manager();
        let small = vec![json!({"role": "user", "content": "hi"})];
        assert!(!m.should_apply(&small, 128_000, 4_000));
    }

    #[test]
    fn should_apply_true_over_budget() {
        let m = manager();
        let huge = vec![json!({"role": "user", "content": "x".repeat(1_000_000)})];
        assert!(m.should_apply(&huge, 1000, 100));
    }

    #[test]
    fn apply_under_budget_is_passthrough() {
        let m = manager();
        let msgs = vec![json!({"role": "user", "content": "hello"})];
        let n = msgs.len();
        let r = m.apply(
            msgs,
            ApplyCtx {
                model_limit: 128_000,
                output_buffer: Some(4_000),
                frozen_message_count: 0,
            },
        );
        assert_eq!(r.messages.len(), n);
        assert!(r.strategies_applied.is_empty());
        assert_eq!(r.tokens_before, r.tokens_after);
    }

    #[test]
    fn apply_over_budget_runs_drop_by_score() {
        let m = manager();
        // Many large-ish messages; tight budget forces drops. Last
        // turn is protected by default config (keep_last_turns=2).
        let mut msgs: Vec<Value> = (0..10)
            .map(|i| json!({"role": "user", "content": format!("message {i} ").repeat(50)}))
            .collect();
        msgs.push(json!({"role": "assistant", "content": "ack"}));
        msgs.push(json!({"role": "user", "content": "final"}));
        let initial = count_messages(&msgs, &EstimatingCounter::default());
        let r = m.apply(
            msgs.clone(),
            ApplyCtx {
                model_limit: initial / 2,
                output_buffer: Some(0),
                frozen_message_count: 0,
            },
        );
        assert_eq!(r.strategies_applied, vec!["drop_by_score"]);
        assert!(r.tokens_after < r.tokens_before);
        assert!(r.messages.len() < msgs.len());
        // Final user message ("final") is protected; it must survive.
        let last = r.messages.last().unwrap();
        assert_eq!(last["content"], "final");
    }

    #[test]
    fn frozen_prefix_is_never_dropped() {
        let m = manager();
        let mut msgs: Vec<Value> = Vec::new();
        msgs.push(json!({"role": "user", "content": "FROZEN PREFIX MARKER".repeat(50)}));
        for i in 0..15 {
            msgs.push(json!({"role": "user", "content": format!("filler {i} ").repeat(50)}));
        }
        let initial = count_messages(&msgs, &EstimatingCounter::default());
        let r = m.apply(
            msgs,
            ApplyCtx {
                model_limit: initial / 4,
                output_buffer: Some(0),
                frozen_message_count: 1,
            },
        );
        // The first message (frozen) must still be present.
        assert!(r.messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("FROZEN PREFIX MARKER"));
    }

    #[test]
    fn system_message_protected_by_default() {
        let m = manager();
        let mut msgs: Vec<Value> = Vec::new();
        msgs.push(json!({"role": "system", "content": "you are helpful"}));
        for i in 0..15 {
            msgs.push(json!({"role": "user", "content": format!("filler {i} ").repeat(50)}));
        }
        let initial = count_messages(&msgs, &EstimatingCounter::default());
        let r = m.apply(
            msgs,
            ApplyCtx {
                model_limit: initial / 3,
                output_buffer: Some(0),
                frozen_message_count: 0,
            },
        );
        // System message present somewhere.
        assert!(r
            .messages
            .iter()
            .any(|m| m.get("role").and_then(Value::as_str) == Some("system")));
    }

    #[test]
    fn with_strategy_appends_to_chain() {
        struct Sentinel;
        impl ContextStrategy for Sentinel {
            fn name(&self) -> &'static str {
                "sentinel"
            }
            fn try_fit(
                &self,
                _ws: &mut ContextWorkspace,
                _target: usize,
            ) -> crate::context::workspace::StrategyOutcome {
                crate::context::workspace::StrategyOutcome::default()
            }
        }
        let m = manager().with_strategy(Box::new(Sentinel));
        // Force a drop scenario.
        let msgs: Vec<Value> = (0..20)
            .map(|i| json!({"role": "user", "content": format!("m{i} ").repeat(50)}))
            .collect();
        let initial = count_messages(&msgs, &EstimatingCounter::default());
        let r = m.apply(
            msgs,
            ApplyCtx {
                model_limit: initial / 4,
                output_buffer: Some(0),
                frozen_message_count: 0,
            },
        );
        // drop_by_score may or may not fully resolve. If it does,
        // sentinel is short-circuited (no entry). If not, sentinel
        // appears second.
        assert_eq!(r.strategies_applied[0], "drop_by_score");
    }

    #[test]
    fn ccr_marker_emitted_on_drop() {
        let m = manager();
        let msgs: Vec<Value> = (0..20)
            .map(|i| json!({"role": "user", "content": format!("m{i} ").repeat(50)}))
            .collect();
        let initial = count_messages(&msgs, &EstimatingCounter::default());
        let r = m.apply(
            msgs,
            ApplyCtx {
                model_limit: initial / 4,
                output_buffer: Some(0),
                frozen_message_count: 0,
            },
        );
        assert!(!r.markers_inserted.is_empty());
        assert!(r.markers_inserted[0].contains("ccr_retrieve"));
    }
}
