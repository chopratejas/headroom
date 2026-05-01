//! `IcmConfig` — six-field config for the OSS context manager.
//!
//! Cut from the 12+ Python config: `compress_threshold`,
//! `summarize_threshold`, `summarization_*`, `memory_tiers_*`,
//! `warm_tier_*`, `cold_tier_*`. Those belong to strategies that don't
//! ship in OSS. `recency_decay_rate` and `toin_*` moved to
//! [`MessageScorer`](crate::scoring::MessageScorer) where they belong
//! semantically.

use serde::{Deserialize, Serialize};

use crate::scoring::ScoringWeights;

/// Configuration for [`IntelligentContextManager`](super::IntelligentContextManager).
///
/// Defaults are tuned for the OSS sweet spot: keep system messages and
/// the last 2 turns sacred, leave 4K tokens of headroom for the model's
/// reply, score by importance, persist drops to CCR so they're
/// retrievable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IcmConfig {
    /// Master switch. When `false`, `should_apply` always returns
    /// `false` regardless of token count.
    pub enabled: bool,

    /// If `true`, `role=system` messages are never droppable. Default
    /// `true` — dropping system prompts breaks behavior in subtle
    /// ways. Disable only if the caller manages system prompts
    /// themselves and explicitly wants them in the candidate pool.
    pub keep_system: bool,

    /// Number of recent user turns to protect from dropping. A "turn"
    /// here means a `role=user` message; everything from the last
    /// `keep_last_turns`th user message to the end is protected,
    /// including all assistant replies and tool exchanges in between.
    pub keep_last_turns: usize,

    /// Reserved tokens for the model's response. Effective budget is
    /// `model_limit - output_buffer_tokens`. Default 4000 covers most
    /// reasonable replies; bump for long-form generation.
    pub output_buffer_tokens: usize,

    /// Weights for the six-factor message scorer. See
    /// [`ScoringWeights`] for the factor breakdown. Defaults match
    /// Python's `ScoringWeights()` so existing tuning carries over.
    pub scoring_weights: ScoringWeights,

    /// When `true` (default), dropped messages are serialized into
    /// the CCR store before removal — the model can retrieve them via
    /// a tool call. This is the OSS-defining behaviour: with
    /// `ccr_on_drop=true`, drop ≈ "moved to retrievable cache"; with
    /// `false`, drop ≈ rolling window.
    pub ccr_on_drop: bool,
}

impl Default for IcmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            keep_system: true,
            keep_last_turns: 2,
            output_buffer_tokens: 4000,
            scoring_weights: ScoringWeights::default(),
            ccr_on_drop: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_oss_intent() {
        let c = IcmConfig::default();
        assert!(c.enabled);
        assert!(c.keep_system);
        assert_eq!(c.keep_last_turns, 2);
        assert_eq!(c.output_buffer_tokens, 4000);
        assert!(c.ccr_on_drop);
    }

    #[test]
    fn round_trips_through_serde() {
        let c = IcmConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let back: IcmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
