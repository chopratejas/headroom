//! Mutable working set passed between strategies.
//!
//! Each [`ContextStrategy`](super::strategy::ContextStrategy) implementation
//! takes a `&mut ContextWorkspace`, mutates it (drops messages, inserts
//! markers, records bookkeeping), and returns a [`StrategyOutcome`].
//!
//! The workspace owns the messages — strategies never see the original
//! input list. The manager copies the input into the workspace before the
//! cascade and pulls the (possibly modified) list out at the end.

use std::collections::HashSet;

use serde_json::Value;

/// Mutable state threaded through the strategy cascade.
///
/// One workspace per `apply()` call. Not `Send` across the FFI boundary —
/// the manager constructs it inline and consumes it before returning.
pub struct ContextWorkspace {
    /// The current message list. Strategies mutate this in place.
    pub messages: Vec<Value>,

    /// Indices that must NOT be dropped — system messages, last-N-turns,
    /// frozen prefix, paired tool responses for protected assistants.
    /// Computed by [`SafetyRails`](super::safety::SafetyRails) before
    /// the cascade starts and re-computed after a strategy mutates the
    /// list (since indices shift on drop).
    pub protected: HashSet<usize>,

    /// Number of leading messages that are part of a provider's
    /// prompt-cache prefix. Dropping any of these busts the cache and
    /// raises latency + cost. Always added to `protected`.
    ///
    /// Provider-specific signal supplied by the caller via `ApplyCtx`.
    /// Anthropic and OpenAI both expose prompt caching; OSS Headroom
    /// receives this from the proxy layer (which knows the provider).
    pub frozen_count: usize,

    /// Token count of the current messages. Strategies update this
    /// after each mutation so subsequent strategies see fresh state.
    pub current_tokens: usize,

    /// Indices the cascade has dropped, in drop order. Used by the
    /// CCR-on-drop helper to serialize the originals into the cache.
    /// Strategies append to this; they don't read it.
    pub dropped_indices: Vec<usize>,

    /// Original message list at workspace construction time. Preserved
    /// so [`ccr_drop`](super::ccr_drop) can recover the *exact* dropped
    /// content even after several drop rounds have shifted indices.
    /// Strategies don't read this.
    pub original_messages: Vec<Value>,
}

impl ContextWorkspace {
    pub fn new(messages: Vec<Value>, protected: HashSet<usize>, frozen_count: usize) -> Self {
        let original_messages = messages.clone();
        Self {
            messages,
            protected,
            frozen_count,
            current_tokens: 0,
            dropped_indices: Vec::new(),
            original_messages,
        }
    }
}

/// Result of one strategy invocation.
#[derive(Debug, Clone, Default)]
pub struct StrategyOutcome {
    /// Tokens freed by this strategy. The manager subtracts this from
    /// `current_tokens` and decides whether to invoke the next strategy.
    pub tokens_freed: usize,

    /// Marker strings inserted by this strategy (e.g. CCR retrieval
    /// hints). Aggregated into `ApplyResult.markers_inserted`.
    pub markers_inserted: Vec<String>,

    /// Set when the strategy fully resolved the budget. The manager
    /// stops the cascade — later strategies don't run. Distinct from
    /// "I freed some tokens" — a strategy can free tokens *and* still
    /// not have brought the request under budget.
    pub fully_resolved: bool,
}
