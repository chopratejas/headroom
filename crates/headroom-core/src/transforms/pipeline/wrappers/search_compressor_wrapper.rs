//! `SearchCompressorWrapper` — adapt the existing [`SearchCompressor`]
//! to the Phase 3g pipeline's [`LossyTransform`] surface.
//!
//! The underlying compressor handles parsing (grep/ripgrep with Windows
//! drive-letter prefixes and dashes-in-filename support fixed in Phase
//! 3e.2), per-file match scoring, head/tail anchor preservation, and
//! CCR offload. This wrapper threads the user's question through to
//! the compressor's relevance scoring so query-relevant matches
//! survive when the budget tightens.
//!
//! # Why `structure_preserved = false`
//!
//! `SearchCompressor` deliberately drops matches: it caps total
//! matches across files, caps per-file matches, and selects by
//! relevance + first/last anchors. The dropped lines are gone unless
//! CCR retrieval is invoked.
//!
//! # Why `confidence = 0.8`
//!
//! Calibrated between `DiffCompressorWrapper` (0.85) and
//! `LogCompressorWrapper` (0.75). Search compression has the user's
//! query as a relevance signal — that's a real semantic input the log
//! compressor lacks. Not as high as diff because relevance scoring is
//! still heuristic (keyword + always-keep-first/last anchors), not
//! the byte-deterministic preserve-additions/deletions guarantee
//! diffs offer.
//!
//! # CCR propagation
//!
//! `SearchCompressor::compress` returns a `cache_key: Option<String>`.
//! When set (CCR fired because the result cleared the thresholds), we
//! propagate it to [`TransformResult::reversible_via`] so the
//! orchestrator can later emit a retrieval marker. When unset, the
//! offload was unprofitable and the dropped matches stay dropped.
//!
//! # No regex
//!
//! All regex sits inside `SearchCompressor` (parse + scoring use
//! `regex::Regex` for path/line-number extraction). The wrapper itself
//! is pure adaptation.

use crate::ccr::InMemoryCcrStore;
use crate::transforms::pipeline::traits::{
    CompressionContext, LossyTransform, TransformError, TransformResult,
};
use crate::transforms::search_compressor::{SearchCompressor, SearchCompressorConfig};
use crate::transforms::ContentType;

/// Pipeline wrapper for [`SearchCompressor`].
///
/// # CCR consistency note
///
/// `SearchCompressor::compress` short-circuits CCR emission when no
/// `CcrStore` is supplied (it only sets `cache_key` after a successful
/// `store.put`). That makes its CCR behavior *inconsistent* with
/// `DiffCompressor`, which emits the cache_key whenever the savings
/// threshold is met regardless of storage. To keep wrappers behaving
/// uniformly through the pipeline trait, this wrapper holds an
/// internal [`InMemoryCcrStore`] and routes through
/// `compress_with_store`. The store keeps the original payload only
/// for the lifetime of the wrapper — production CCR retrieval still
/// goes through the Python `CompressionStore` via the existing PyO3
/// path; this internal store is the minimum needed to make the
/// underlying compressor emit a `cache_key` so the trait surface
/// reports it.
///
/// Filed as a follow-up bug audit finding (Phase 3g PR2): unify the
/// CCR emission path across Diff/Log/Search compressors so callers
/// don't need to know which one needs a store and which one doesn't.
pub struct SearchCompressorWrapper {
    inner: SearchCompressor,
    store: InMemoryCcrStore,
}

impl SearchCompressorWrapper {
    pub const NAME: &'static str = "search_compressor";

    pub fn new(config: SearchCompressorConfig) -> Self {
        Self {
            inner: SearchCompressor::new(config),
            store: InMemoryCcrStore::new(),
        }
    }
}

impl Default for SearchCompressorWrapper {
    fn default() -> Self {
        Self::new(SearchCompressorConfig::default())
    }
}

impl LossyTransform for SearchCompressorWrapper {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        &[ContentType::SearchResults]
    }

    fn apply(
        &self,
        content: &str,
        ctx: &CompressionContext,
    ) -> Result<TransformResult, TransformError> {
        if content.is_empty() {
            return Err(TransformError::skipped(Self::NAME, "empty input"));
        }
        // Pass the user's query through to the compressor's relevance
        // scoring. Empty `ctx.query` is the "no relevance bias" path
        // SearchCompressor already supports.
        let (result, _stats) =
            self.inner
                .compress_with_store(content, &ctx.query, 1.0, Some(&self.store));
        let bytes_saved = content.len().saturating_sub(result.compressed.len());
        Ok(TransformResult {
            output: result.compressed,
            bytes_saved,
            structure_preserved: false,
            reversible_via: result.cache_key,
        })
    }

    fn confidence(&self) -> f32 {
        0.8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> CompressionContext {
        CompressionContext::default()
    }

    /// Build a 30-match grep-style fixture across 3 files. Each file
    /// has 10 matches; line 5 in `b.rs` carries the QUERYHIT marker so
    /// query-aware tests can verify preservation.
    fn grep_fixture() -> String {
        let mut out = String::new();
        for f in ["src/a.rs", "src/b.rs", "src/c.rs"] {
            for i in 1..=10 {
                if f == "src/b.rs" && i == 5 {
                    out.push_str(&format!("{f}:{i}:fn QUERYHIT_target() {{}}\n"));
                } else {
                    out.push_str(&format!("{f}:{i}:fn helper_{i}() {{}}\n"));
                }
            }
        }
        out
    }

    #[test]
    fn name_and_confidence_are_calibrated_constants() {
        let w = SearchCompressorWrapper::default();
        assert_eq!(w.name(), "search_compressor");
        assert!((w.confidence() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn applies_to_only_search_results() {
        let w = SearchCompressorWrapper::default();
        assert_eq!(w.applies_to(), &[ContentType::SearchResults]);
    }

    #[test]
    fn empty_input_returns_skipped_error() {
        let w = SearchCompressorWrapper::default();
        let err = w.apply("", &ctx()).expect_err("empty input is a skip");
        assert!(matches!(err, TransformError::Skipped { .. }));
    }

    #[test]
    fn real_grep_fixture_compresses_and_reports_bytes_saved() {
        let w = SearchCompressorWrapper::default();
        let input = grep_fixture();
        let r = w.apply(&input, &ctx()).expect("fixture compresses");
        assert!(
            r.bytes_saved > 0,
            "30 matches across 3 files should compress; got {} bytes saved",
            r.bytes_saved
        );
        assert!(r.output.len() < input.len());
    }

    #[test]
    fn structure_preserved_is_always_false() {
        let w = SearchCompressorWrapper::default();
        let r = w.apply(&grep_fixture(), &ctx()).unwrap();
        assert!(!r.structure_preserved);
    }

    #[test]
    fn bytes_saved_computed_against_original_input_length() {
        let w = SearchCompressorWrapper::default();
        let input = grep_fixture();
        let r = w.apply(&input, &ctx()).unwrap();
        // bytes_saved = original.len() - output.len(), clamped at zero.
        assert_eq!(r.bytes_saved, input.len() - r.output.len());
    }

    #[test]
    fn cache_key_propagates_to_reversible_via_when_set() {
        // Default config: enable_ccr=true, min_matches_for_ccr=10,
        // min_compression_ratio_for_ccr=0.8. Fixture has 30 matches
        // (well over the 10 threshold), and the per-file cap (5) drops
        // enough to clear the ratio gate.
        let w = SearchCompressorWrapper::default();
        let r = w.apply(&grep_fixture(), &ctx()).unwrap();
        assert!(
            r.reversible_via.is_some(),
            "default config + 30-match fixture should fire CCR"
        );
        // The cache_key is a hex string; sanity-check it.
        let key = r.reversible_via.as_deref().unwrap();
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(!key.is_empty());
    }

    #[test]
    fn user_query_steers_relevance_scoring() {
        // With a query that mentions QUERYHIT, the relevance scorer
        // should keep that line even when neighbors get dropped.
        let w = SearchCompressorWrapper::default();
        let q_ctx = CompressionContext::with_query("find the QUERYHIT_target function");
        let r = w
            .apply(&grep_fixture(), &q_ctx)
            .expect("fixture compresses with query");
        assert!(
            r.output.contains("QUERYHIT_target"),
            "query-relevant line should survive compression; got output: {}",
            r.output
        );
    }

    #[test]
    fn small_input_passes_through_with_no_cache_key() {
        // 2-match input is below CCR thresholds; expect output ≈ input
        // and no reversible_via.
        let w = SearchCompressorWrapper::default();
        let small = "src/a.rs:1:fn one() {}\nsrc/a.rs:2:fn two() {}";
        let r = w.apply(small, &ctx()).expect("small input compresses");
        // Bytes saved may be 0 on this tiny fixture; what matters is
        // CCR didn't fire on a sub-threshold input.
        assert!(r.reversible_via.is_none());
    }

    #[test]
    fn custom_config_is_honored() {
        // Force aggressive caps. With max_total_matches=3 the wrapper
        // must drop a lot.
        let w = SearchCompressorWrapper::new(SearchCompressorConfig {
            max_total_matches: 3,
            ..Default::default()
        });
        let r = w.apply(&grep_fixture(), &ctx()).unwrap();
        assert!(r.bytes_saved > 0);
        // Output should be substantially smaller than the 30-match
        // fixture given the 3-match cap.
        assert!(r.output.len() < grep_fixture().len() / 2);
    }
}
