//! SmartCrusher configuration.
//!
//! Direct port of `SmartCrusherConfig` at `smart_crusher.py:927-957`. The
//! defaults must match Python exactly — they're consulted everywhere
//! during compression and any drift breaks parity fixtures.

/// Configuration for SmartCrusher.
///
/// SCHEMA-PRESERVING: Output contains only items from the original array.
/// No wrappers, no generated text, no metadata keys. (Python comment at
/// line 930-931.)
#[derive(Debug, Clone)]
pub struct SmartCrusherConfig {
    pub enabled: bool,
    /// Don't analyze arrays smaller than this. Default 5.
    pub min_items_to_analyze: usize,
    /// Only crush content with more than this many tokens. Default 200.
    pub min_tokens_to_crush: usize,
    /// Standard deviations from the mean to count as a change point.
    /// Default 2.0.
    pub variance_threshold: f64,
    /// Below this unique-ratio, a field is treated as nearly constant.
    /// Default 0.1.
    pub uniqueness_threshold: f64,
    /// Similarity score above which strings cluster together. Default 0.8.
    pub similarity_threshold: f64,
    /// Target maximum items in the output. Default 15.
    pub max_items_after_crush: usize,
    /// Whether to preserve detected change points. Default true.
    pub preserve_change_points: bool,
    /// Factor out fields with constant values across all items. Default
    /// false (disabled — preserves original schema).
    pub factor_out_constants: bool,
    /// Include generated text summaries in output. Default false (disabled
    /// — no generated text).
    pub include_summaries: bool,
    /// Use feedback hints to adjust compression aggressiveness. Default true.
    pub use_feedback_hints: bool,
    /// Minimum confidence required to apply TOIN recommendations.
    /// Default 0.5. (Python LOW FIX #21.)
    pub toin_confidence_threshold: f64,
    /// Drop content-identical items before sampling. Default true.
    pub dedup_identical_items: bool,
    /// Fraction of K to allocate to the start of the array. Default 0.3.
    pub first_fraction: f64,
    /// Fraction of K to allocate to the end of the array. Default 0.15.
    pub last_fraction: f64,
    /// Items with `RelevanceScore.score >= this` are pinned by the
    /// planning methods. Mirrors Python's `RelevanceConfig.relevance_threshold`.
    /// Default 0.3 — matches the Python default.
    pub relevance_threshold: f64,
    /// Minimum byte-savings ratio (0.0..1.0) for the lossless compaction
    /// path to be chosen over lossy. Computed as
    /// `1 - len(rendered) / len(input)`. If lossless saves less than
    /// this fraction, `crush_array` falls through to the lossy path
    /// (with CCR-Dropped retrieval markers). Default `0.30`.
    ///
    /// **Override semantics.** OSS users can tune this via the config
    /// directly. Enterprise plug-ins replace the entire decision via
    /// a custom builder; the threshold is the OSS-default policy
    /// expressed as a single knob. Set to `0.0` to always prefer
    /// lossless when available; set to `1.0` to effectively disable
    /// the lossless path (lossy + CCR always).
    pub lossless_min_savings_ratio: f64,
    /// Whether `crush_array` emits the `<<ccr:HASH N_rows_offloaded>>`
    /// marker text (and the `_ccr_dropped` sentinel that wraps it)
    /// when the lossy path drops rows. Default `true`.
    ///
    /// This is the Rust-side gate for the Python `CCRConfig.
    /// inject_retrieval_marker` flag. When `false`:
    ///
    /// - Rows are still dropped (compression still happens).
    /// - The originals are NOT placed in the CCR store and the
    ///   `ccr_hash` field on the result is `None` (no retrieval is
    ///   possible — and we don't want to waste store space when no
    ///   marker will reference it).
    /// - `dropped_summary` stays empty so the lossy-path callsite
    ///   (which appends a sentinel iff `dropped_summary.is_empty() ==
    ///   false`) skips the sentinel append.
    ///
    /// Scope: gates ONLY the row-drop sentinel path. The Stage-3c.2
    /// opaque-string CCR substitutions (`<<ccr:HASH,KIND,SIZE>>`
    /// emitted by the document walker) are deliberately not gated —
    /// they have no Python equivalent, no production caller has
    /// asked for opaque-string suppression, and gating them would
    /// silently disable compression of large strings (more confusing
    /// than no marker at all).
    pub enable_ccr_marker: bool,
}

impl Default for SmartCrusherConfig {
    fn default() -> Self {
        // These defaults must match smart_crusher.py:934-957 byte-for-byte.
        // The PR4 additions (`lossless_min_savings_ratio`) have no
        // Python counterpart — they govern Rust-side dispatch only.
        SmartCrusherConfig {
            enabled: true,
            min_items_to_analyze: 5,
            min_tokens_to_crush: 200,
            variance_threshold: 2.0,
            uniqueness_threshold: 0.1,
            similarity_threshold: 0.8,
            max_items_after_crush: 15,
            preserve_change_points: true,
            factor_out_constants: false,
            include_summaries: false,
            use_feedback_hints: true,
            toin_confidence_threshold: 0.5,
            dedup_identical_items: true,
            first_fraction: 0.3,
            last_fraction: 0.15,
            relevance_threshold: 0.3,
            lossless_min_savings_ratio: 0.30,
            enable_ccr_marker: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        // Pin every default. Each field is consulted by some compression
        // path and a drift would break parity. If Python ever changes a
        // default, this test must be updated in lockstep.
        let c = SmartCrusherConfig::default();
        assert!(c.enabled);
        assert_eq!(c.min_items_to_analyze, 5);
        assert_eq!(c.min_tokens_to_crush, 200);
        assert_eq!(c.variance_threshold, 2.0);
        assert_eq!(c.uniqueness_threshold, 0.1);
        assert_eq!(c.similarity_threshold, 0.8);
        assert_eq!(c.max_items_after_crush, 15);
        assert!(c.preserve_change_points);
        assert!(!c.factor_out_constants);
        assert!(!c.include_summaries);
        assert!(c.use_feedback_hints);
        assert_eq!(c.toin_confidence_threshold, 0.5);
        assert!(c.dedup_identical_items);
        assert_eq!(c.first_fraction, 0.3);
        assert_eq!(c.last_fraction, 0.15);
        assert_eq!(c.relevance_threshold, 0.3);
        assert_eq!(c.lossless_min_savings_ratio, 0.30);
        // `enable_ccr_marker` defaults to true (preserves prior
        // behavior — markers always emitted). The Python shim flips
        // this when the caller passes `inject_retrieval_marker=False`.
        assert!(c.enable_ccr_marker);
    }
}
