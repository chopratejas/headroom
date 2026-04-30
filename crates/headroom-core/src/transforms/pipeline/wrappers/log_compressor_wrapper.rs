//! `LogCompressorWrapper` — adapt the existing [`LogCompressor`] to
//! the Phase 3g pipeline's [`LossyTransform`] surface.
//!
//! The underlying compressor already does all the real work (format
//! detection, level classification, stack-trace tracking, dedupe, CCR
//! offload). This wrapper is a thin bridge: it forwards `apply` to
//! `LogCompressor::compress`, projects the rich result onto the
//! pipeline's [`TransformResult`] shape, and registers the transform
//! against the closest content type the detector emits
//! ([`ContentType::BuildOutput`] — there's no separate `Logs`
//! variant today; build/test/lint output is the canonical log source
//! the detector tags).
//!
//! # Why `structure_preserved = false`
//!
//! `LogCompressor` is genuinely lossy: it filters lines by importance
//! score, dedupes warnings, and caps total output by an adaptive K.
//! Dropped lines are gone — the LLM cannot reconstruct them from the
//! compressed view alone. CCR retrieval is the recovery path; that's
//! why `reversible_via` carries the cache key when CCR fires.
//!
//! # Why `confidence = 0.75`
//!
//! Calibrated below `DiffCompressorWrapper`'s 0.85: git-diff hunks
//! preserve clean semantic boundaries (a hunk is the right unit of
//! compression), whereas log line-importance scoring has more inherent
//! uncertainty — the score is a heuristic over level + stack-trace +
//! summary signals, not a guarantee that all relevant lines survive.
//! Still well above `LineImportanceFilter`'s 0.7 because the format-
//! aware passes (pytest/npm/cargo/jest/make detection, summary lines,
//! adaptive sizer) compensate for that line-by-line uncertainty.
//!
//! # Bias = 1.0
//!
//! The neutral default for the adaptive sizer. The pipeline doesn't
//! yet plumb a per-call bias signal; PR4 adds query-aware bias once
//! the orchestrator threads `CompressionContext::query` through to
//! relevance scoring. Until then we pick the neutral baseline so we
//! don't bake a phantom signal into the adaptive K.
//!
//! # No regex in this file
//!
//! All regex is inside `LogCompressor` (and called out in its module
//! doc). The wrapper itself is pure adaptation.

use crate::ccr::InMemoryCcrStore;
use crate::transforms::log_compressor::{LogCompressor, LogCompressorConfig};
use crate::transforms::pipeline::traits::{
    CompressionContext, LossyTransform, TransformError, TransformResult,
};
use crate::transforms::ContentType;

/// Adapts [`LogCompressor`] to the [`LossyTransform`] trait.
///
/// # CCR consistency note
///
/// `LogCompressor::compress` short-circuits CCR emission when no
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
pub struct LogCompressorWrapper {
    inner: LogCompressor,
    store: InMemoryCcrStore,
}

impl LogCompressorWrapper {
    pub const NAME: &'static str = "log_compressor";

    /// Static slice of accepted content types. `BuildOutput` is the
    /// closest match the detector emits — see module docs.
    const APPLIES_TO: &'static [ContentType] = &[ContentType::BuildOutput];

    /// Bias passed to the underlying compressor's adaptive sizer.
    /// Neutral default; see module docs for why we don't thread query
    /// signal yet.
    const NEUTRAL_BIAS: f64 = 1.0;

    pub fn new(config: LogCompressorConfig) -> Self {
        Self {
            inner: LogCompressor::new(config),
            store: InMemoryCcrStore::new(),
        }
    }
}

impl Default for LogCompressorWrapper {
    fn default() -> Self {
        Self::new(LogCompressorConfig::default())
    }
}

impl LossyTransform for LogCompressorWrapper {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        Self::APPLIES_TO
    }

    fn apply(
        &self,
        content: &str,
        _ctx: &CompressionContext,
    ) -> Result<TransformResult, TransformError> {
        if content.is_empty() {
            return Err(TransformError::skipped(Self::NAME, "empty input"));
        }

        // `compress` returns a (LogCompressionResult, LogCompressorStats)
        // tuple — we only need the result for the pipeline surface;
        // the sidecar stats stay inside the inner compressor's own
        // telemetry path.
        let (result, _stats) =
            self.inner
                .compress_with_store(content, Self::NEUTRAL_BIAS, Some(&self.store));

        let bytes_saved = content.len().saturating_sub(result.compressed.len());
        Ok(TransformResult {
            output: result.compressed,
            bytes_saved,
            // Lossy: lines were dropped. Recovery is via CCR retrieval
            // when `cache_key` is set, not from the compressed view.
            structure_preserved: false,
            reversible_via: result.cache_key,
        })
    }

    fn confidence(&self) -> f32 {
        // Calibrated 0.75 — see module docs.
        0.75
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrapper() -> LogCompressorWrapper {
        LogCompressorWrapper::default()
    }

    fn ctx() -> CompressionContext {
        CompressionContext::default()
    }

    /// Build an npm-style log with N lines, mostly noise + a few
    /// errors so the format detector triggers and CCR is exercised
    /// when the wrapper is configured for it.
    fn npm_log(noise_lines: usize, error_lines: usize) -> String {
        let mut s = String::new();
        for i in 0..noise_lines {
            s.push_str(&format!("npm info {}: doing something\n", i));
        }
        for i in 0..error_lines {
            s.push_str(&format!("npm ERR! failure {}: stack overflow at frob\n", i));
        }
        s
    }

    #[test]
    fn name_matches_telemetry_convention() {
        assert_eq!(wrapper().name(), "log_compressor");
    }

    #[test]
    fn confidence_is_calibrated_constant() {
        assert!((wrapper().confidence() - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn applies_to_build_output_only() {
        let w = wrapper();
        assert_eq!(w.applies_to(), &[ContentType::BuildOutput]);
    }

    #[test]
    fn empty_input_returns_skipped_error() {
        let w = wrapper();
        let err = w.apply("", &ctx()).expect_err("empty input is a skip");
        assert!(matches!(err, TransformError::Skipped { .. }));
    }

    #[test]
    fn real_log_fixture_compresses() {
        // 60 noise lines + 5 error lines — well over min_lines_for_ccr (50)
        // so the inner compressor runs the full selection pipeline.
        let w = wrapper();
        let input = npm_log(60, 5);
        assert!(input.lines().count() >= 30);
        let r = w
            .apply(&input, &ctx())
            .expect("well-formed log should compress");
        // Real compression must shrink — not just round-trip.
        assert!(r.bytes_saved > 0, "expected positive savings");
        assert!(r.output.len() < input.len(), "output should be smaller");
    }

    #[test]
    fn cache_key_propagates_to_reversible_via() {
        // Use a permissive ratio threshold so CCR fires deterministically
        // on a moderately-sized log. The default 0.5 ratio is too tight
        // for short logs in tests.
        let w = LogCompressorWrapper::new(LogCompressorConfig {
            max_total_lines: 10,
            min_lines_for_ccr: 5,
            min_compression_ratio_for_ccr: 0.95,
            ..Default::default()
        });
        let input = npm_log(80, 3);
        let r = w.apply(&input, &ctx()).expect("compresses");
        assert!(
            r.reversible_via.is_some(),
            "reversible_via should carry the CCR cache key"
        );
        // Cache key is a non-empty md5 hex prefix.
        assert!(!r.reversible_via.as_ref().unwrap().is_empty());
    }

    #[test]
    fn bytes_saved_computed_against_original_length() {
        let w = wrapper();
        let input = npm_log(60, 5);
        let original_len = input.len();
        let r = w.apply(&input, &ctx()).expect("compresses");
        // bytes_saved == original - output, clamped at 0.
        assert_eq!(r.bytes_saved, original_len.saturating_sub(r.output.len()));
    }

    #[test]
    fn structure_preserved_is_always_false() {
        let w = wrapper();
        let input = npm_log(60, 5);
        let r = w.apply(&input, &ctx()).expect("compresses");
        assert!(!r.structure_preserved);
    }

    #[test]
    fn short_input_below_ccr_threshold_passes_through_without_cache_key() {
        // Below min_lines_for_ccr (50 by default) — inner compressor
        // returns content verbatim, no CCR marker.
        let w = wrapper();
        let input = "npm info one\nnpm info two\nnpm info three";
        let r = w.apply(input, &ctx()).expect("compresses");
        assert!(r.reversible_via.is_none());
        // Verbatim → bytes_saved is 0 (output == input).
        assert_eq!(r.bytes_saved, 0);
    }
}
