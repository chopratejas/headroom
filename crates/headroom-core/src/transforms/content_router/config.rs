//! `ContentRouter` configuration.
//!
//! Mirrors the simple fields of
//! `headroom.transforms.content_router.ContentRouterConfig`. The
//! complex Python-only fields — `exclude_tools`, `read_lifecycle`,
//! `tool_profiles` — feed the `apply()` proxy entry point and are
//! ported in PR6 along with that pipeline. They're deliberately
//! omitted here so PR2 stays focused on the dispatch-decision data.

use super::types::CompressionStrategy;

/// Configuration knobs the `ContentRouter` reads at compress time.
/// All defaults match Python.
#[derive(Debug, Clone)]
pub struct ContentRouterConfig {
    // ── Enable/disable specific compressors ──────────────────────────
    /// Disabled by default — code passes through unmangled (Python:
    /// `enable_code_aware = False`).
    pub enable_code_aware: bool,
    pub enable_kompress: bool,
    pub enable_smart_crusher: bool,
    pub enable_search_compressor: bool,
    pub enable_log_compressor: bool,
    pub enable_html_extractor: bool,
    pub enable_image_optimizer: bool,

    // ── Routing preferences ──────────────────────────────────────────
    pub prefer_code_aware_for_code: bool,
    /// Minimum number of distinct content types to consider a payload
    /// "mixed". `2` matches Python.
    pub mixed_content_threshold: usize,
    /// Sections smaller than this don't justify the per-section
    /// compress overhead.
    pub min_section_tokens: usize,

    /// Strategy applied when no compressor matches the detected type.
    pub fallback_strategy: CompressionStrategy,

    // ── Protection: "don't compress what's likely the analysis subject"
    pub skip_user_messages: bool,
    /// Don't compress code in the last N messages. `0` disables.
    pub protect_recent_code: usize,
    pub protect_analysis_context: bool,

    /// Fraction of total messages to protect from compression for
    /// adaptive Read protection. `0.0` = always exclude all (safest
    /// for coding agents).
    pub protect_recent_reads_fraction: f64,

    // ── Adaptive compression ratio ───────────────────────────────────
    /// Threshold when context is mostly empty.
    pub min_ratio_relaxed: f64,
    /// Threshold when context is nearly full.
    pub min_ratio_aggressive: f64,

    // ── CCR settings (forwarded to SmartCrusher) ─────────────────────
    pub ccr_enabled: bool,
    pub ccr_inject_marker: bool,

    /// Tag protection: when `false`, entire `<custom-tag>...</custom-tag>`
    /// blocks are protected verbatim. When `true`, only the markers are
    /// protected and inner content is compressed.
    pub compress_tagged_content: bool,
    //
    // Deferred to PR6 (proxy `apply()` port):
    //   - exclude_tools: Option<HashSet<String>>
    //   - read_lifecycle: ReadLifecycleConfig
    //   - tool_profiles: Option<HashMap<String, CompressionProfile>>
}

impl Default for ContentRouterConfig {
    fn default() -> Self {
        Self {
            enable_code_aware: false,
            enable_kompress: true,
            enable_smart_crusher: true,
            enable_search_compressor: true,
            enable_log_compressor: true,
            enable_html_extractor: true,
            enable_image_optimizer: true,
            prefer_code_aware_for_code: false,
            mixed_content_threshold: 2,
            min_section_tokens: 20,
            fallback_strategy: CompressionStrategy::Kompress,
            skip_user_messages: true,
            protect_recent_code: 4,
            protect_analysis_context: true,
            protect_recent_reads_fraction: 0.0,
            min_ratio_relaxed: 0.85,
            min_ratio_aggressive: 0.65,
            ccr_enabled: true,
            ccr_inject_marker: true,
            compress_tagged_content: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let c = ContentRouterConfig::default();
        // Python defaults — copied verbatim from
        // headroom/transforms/content_router.py:391-435.
        assert!(!c.enable_code_aware);
        assert!(c.enable_kompress);
        assert!(c.enable_smart_crusher);
        assert!(c.enable_search_compressor);
        assert!(c.enable_log_compressor);
        assert!(c.enable_html_extractor);
        assert!(c.enable_image_optimizer);
        assert!(!c.prefer_code_aware_for_code);
        assert_eq!(c.mixed_content_threshold, 2);
        assert_eq!(c.min_section_tokens, 20);
        assert_eq!(c.fallback_strategy, CompressionStrategy::Kompress);
        assert!(c.skip_user_messages);
        assert_eq!(c.protect_recent_code, 4);
        assert!(c.protect_analysis_context);
        assert_eq!(c.protect_recent_reads_fraction, 0.0);
        assert_eq!(c.min_ratio_relaxed, 0.85);
        assert_eq!(c.min_ratio_aggressive, 0.65);
        assert!(c.ccr_enabled);
        assert!(c.ccr_inject_marker);
        assert!(!c.compress_tagged_content);
    }

    #[test]
    fn config_is_clone() {
        let c = ContentRouterConfig::default();
        let c2 = c.clone();
        assert_eq!(c.mixed_content_threshold, c2.mixed_content_threshold);
    }
}
