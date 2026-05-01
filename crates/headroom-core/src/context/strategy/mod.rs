//! `ContextStrategy` trait — the extension point for the Enterprise edition.
//!
//! OSS ships exactly one implementation: [`DropByScoreStrategy`]. Enterprise
//! plugs in additional strategies (compress-first, summarize, memory tiers)
//! and registers them via [`IntelligentContextManager::with_strategy`].
//!
//! # Why minimal
//!
//! The trait is intentionally one method. Earlier drafts added observer
//! hooks, telemetry callbacks, and tuning-hint inputs — speculative for
//! Enterprise needs we don't yet understand. Smaller surface = lower risk
//! of getting the API wrong on first try; we can extend with more methods
//! later (default-implemented for backwards compatibility).

pub mod drop_by_score;

use super::workspace::{ContextWorkspace, StrategyOutcome};

/// One stage in the context-fitting cascade.
///
/// Strategies run in registration order; each can either fully resolve
/// the budget (`fully_resolved=true` short-circuits the rest) or free
/// some tokens and let the next strategy continue.
///
/// Implementations must be `Send + Sync` — the manager is shared across
/// requests in a multi-threaded proxy. State mutations belong on the
/// `ContextWorkspace`, not `&self`.
pub trait ContextStrategy: Send + Sync {
    /// Stable identifier shown in logs / `ApplyResult.strategies_applied`.
    /// Use snake_case (`"drop_by_score"`, `"compress_first"`, etc.).
    fn name(&self) -> &'static str;

    /// Mutate the workspace toward fitting `target_tokens`.
    ///
    /// Implementations should:
    /// - Respect `ws.protected` and the leading `ws.frozen_count` indices
    ///   absolutely. Dropping a protected index is a contract violation.
    /// - Update `ws.current_tokens` after mutating `ws.messages`.
    /// - Append to `ws.dropped_indices` when removing messages.
    /// - Report what they freed via `tokens_freed` so the manager can
    ///   decide whether to invoke the next strategy.
    fn try_fit(&self, ws: &mut ContextWorkspace, target_tokens: usize) -> StrategyOutcome;
}

pub use drop_by_score::DropByScoreStrategy;
