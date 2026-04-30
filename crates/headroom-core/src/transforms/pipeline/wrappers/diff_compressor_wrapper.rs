//! `DiffCompressorWrapper` — adapts the existing [`DiffCompressor`] to the
//! Phase 3g [`LossyTransform`] trait shape.
//!
//! # Why this is a `LossyTransform`
//!
//! `DiffCompressor` deliberately drops content: it caps file count, caps
//! per-file hunk count, and trims context lines around `+`/`-` changes.
//! Even though the result is structurally well-formed unified diff, the
//! LLM cannot reconstruct the dropped hunks/files from the compressed
//! view alone — that's exactly the contract of a lossy transform.
//!
//! Hence `structure_preserved = false`. The compressed output is still
//! parseable as a diff, but information has been removed; the structural
//! invariant the orchestrator cares about is round-trip identity, which
//! `DiffCompressor` does not provide.
//!
//! # CCR propagation
//!
//! When `DiffCompressor` decides the savings are large enough (default:
//! more than 20% line reduction), it emits an MD5-derived `cache_key` alongside
//! the compressed bytes. We propagate that key into
//! [`TransformResult::reversible_via`] so downstream CCR retrieval can
//! pull the original bytes back when the LLM requests them. When no
//! cache_key is emitted, `reversible_via` is `None` and the dropped
//! content is gone for good (the underlying compressor already decided
//! it wasn't worth offloading).
//!
//! # Confidence (0.85)
//!
//! Higher than [`LineImportanceFilter`]'s 0.7 because:
//! - Output is deterministic (MD5 cache_key, fixed parser, fixed scorer).
//! - Always-keep additions/deletions semantics mean no `+`/`-` line is
//!   ever silently dropped — only context and middle hunks under cap.
//! - 20+ parity fixtures lock the byte-for-byte behavior against Python.
//!
//! Not 1.0 because hunks beyond `max_hunks_per_file` are dropped
//! (without a per-hunk CCR offload — the cache_key offloads only the
//! whole-diff original, not individual hunks), so the LLM may miss
//! relevant hunks if the cap fires.
//!
//! [`LineImportanceFilter`]: super::super::line_importance_filter::LineImportanceFilter

use std::sync::Arc;

use crate::transforms::diff_compressor::{DiffCompressor, DiffCompressorConfig};
use crate::transforms::pipeline::traits::{
    CompressionContext, LossyTransform, TransformError, TransformResult,
};
use crate::transforms::ContentType;

/// Wraps [`DiffCompressor`] as a [`LossyTransform`].
///
/// Holds an `Arc<DiffCompressor>` so the wrapper can be cheaply cloned
/// and shared across pipeline registrations without re-allocating the
/// underlying config.
#[derive(Debug, Clone)]
pub struct DiffCompressorWrapper {
    inner: Arc<DiffCompressor>,
}

impl DiffCompressorWrapper {
    /// Stable telemetry name. Lowercase snake_case so it composes
    /// cleanly into the strategy-stats JSONB nest landed in Phase 3e.0.
    pub const NAME: &'static str = "diff_compressor";

    /// Static slice — applicability is fixed by the wrapped transform.
    const APPLIES_TO: &'static [ContentType] = &[ContentType::GitDiff];

    /// Build a wrapper around a [`DiffCompressor`] configured with the
    /// given config.
    pub fn new(config: DiffCompressorConfig) -> Self {
        Self {
            inner: Arc::new(DiffCompressor::new(config)),
        }
    }
}

impl Default for DiffCompressorWrapper {
    fn default() -> Self {
        Self::new(DiffCompressorConfig::default())
    }
}

impl LossyTransform for DiffCompressorWrapper {
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

        // The query from CompressionContext is *not* threaded into
        // DiffCompressor's relevance scorer here — the wrapped API
        // accepts a context string, but in PR2 we keep the wrapping
        // minimal and pass an empty context. PR4+ may revisit when
        // ProseFieldCompressor lands and the orchestrator gets a
        // first-class query plumbing decision.
        let (result, _stats) = self.inner.compress_with_stats(content, "");

        let bytes_saved = content.len().saturating_sub(result.compressed.len());

        Ok(TransformResult {
            output: result.compressed,
            bytes_saved,
            // Hunks/files can be dropped — LLM cannot reconstruct from
            // the compressed view. The CCR cache_key (when present)
            // makes the dropped bytes retrievable, but the *output
            // string* itself is not structurally lossless.
            structure_preserved: false,
            // Propagate the underlying compressor's CCR cache_key so
            // the orchestrator / retrieval layer can pull the original
            // bytes back when needed. None when DiffCompressor decided
            // the savings didn't justify a marker.
            reversible_via: result.cache_key,
        })
    }

    fn confidence(&self) -> f32 {
        // See module-level "Confidence (0.85)" docs for the calibration
        // rationale. Fixed constant in PR2; a future PR may upgrade to
        // a stats-derived confidence (e.g. compression_ratio,
        // hunks_dropped) once the strategy-stats nest is consumed by
        // selection logic.
        0.85
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic diff that's long enough to trip
    /// `min_lines_for_ccr` (default 50) so the compressor actually
    /// runs — short diffs pass through unchanged. Mirrors the
    /// `build_synthetic_diff` helper in `diff_compressor.rs` tests.
    fn build_synthetic_diff(n_files: usize) -> String {
        let mut s = String::new();
        for i in 0..n_files {
            s.push_str(&format!(
                "diff --git a/file_{i}.py b/file_{i}.py\n--- a/file_{i}.py\n+++ b/file_{i}.py\n@@ -1,10 +1,12 @@\n",
            ));
            for k in 0..5 {
                s.push_str(&format!(" context_{k}_{i}\n"));
            }
            for k in 0..3 {
                s.push_str(&format!("-removed_{k}_{i}\n"));
            }
            for k in 0..5 {
                s.push_str(&format!("+added_{k}_{i}\n"));
            }
            for k in 0..5 {
                s.push_str(&format!(" tail_{k}_{i}\n"));
            }
        }
        s.push_str("# trailing\n");
        s
    }

    #[test]
    fn name_and_confidence_are_calibrated_constants() {
        let w = DiffCompressorWrapper::default();
        assert_eq!(w.name(), "diff_compressor");
        assert!((w.confidence() - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn applies_to_only_git_diff() {
        let w = DiffCompressorWrapper::default();
        assert_eq!(w.applies_to(), &[ContentType::GitDiff]);
        // Spot-check a handful of other content types are NOT in the
        // applicability slice.
        for other in [
            ContentType::PlainText,
            ContentType::JsonArray,
            ContentType::BuildOutput,
            ContentType::SearchResults,
        ] {
            assert!(
                !w.applies_to().contains(&other),
                "wrapper should not claim {other:?}"
            );
        }
    }

    #[test]
    fn empty_input_returns_skipped_error() {
        let w = DiffCompressorWrapper::default();
        let err = w
            .apply("", &CompressionContext::default())
            .expect_err("empty input is a skip");
        assert!(matches!(err, TransformError::Skipped { .. }));
    }

    #[test]
    fn real_diff_compresses_and_reports_bytes_saved() {
        let w = DiffCompressorWrapper::default();
        let input = build_synthetic_diff(8);
        let result = w
            .apply(&input, &CompressionContext::default())
            .expect("synthetic diff is well-formed");
        assert!(
            result.bytes_saved > 0,
            "8-file synthetic should compress; got bytes_saved={}",
            result.bytes_saved
        );
        // Output length sanity check — should reflect bytes_saved.
        assert_eq!(
            result.bytes_saved,
            input.len().saturating_sub(result.output.len())
        );
        // Compressed output is non-empty.
        assert!(!result.output.is_empty());
    }

    #[test]
    fn cache_key_propagates_to_reversible_via() {
        // The 8-file synthetic compresses 177→129 (ratio ~0.729),
        // beating the default 0.8 threshold → CCR marker emitted →
        // cache_key Some.
        let w = DiffCompressorWrapper::default();
        let input = build_synthetic_diff(8);
        let result = w
            .apply(&input, &CompressionContext::default())
            .expect("valid diff");
        let key = result
            .reversible_via
            .as_ref()
            .expect("expected cache_key to propagate from DiffCompressor");
        // MD5[:24] is a 24-char lowercase hex string.
        assert_eq!(key.len(), 24);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn no_cache_key_when_compressor_skips_ccr() {
        // Short input (< min_lines_for_ccr=50) → compressor passes
        // through unchanged → no cache_key → reversible_via None.
        let w = DiffCompressorWrapper::default();
        let input = "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b";
        let result = w
            .apply(input, &CompressionContext::default())
            .expect("non-empty input is fine");
        assert!(
            result.reversible_via.is_none(),
            "short diff should not emit cache_key, got {:?}",
            result.reversible_via
        );
    }

    #[test]
    fn bytes_saved_computed_against_original_input_length() {
        // Pin the contract: bytes_saved = original.len() - output.len(),
        // saturating to 0. Use a diff where compression actually fires.
        let w = DiffCompressorWrapper::default();
        let input = build_synthetic_diff(8);
        let result = w
            .apply(&input, &CompressionContext::default())
            .expect("valid diff");
        // Re-derive from the same fields and confirm.
        let expected = input.len().saturating_sub(result.output.len());
        assert_eq!(result.bytes_saved, expected);
    }

    #[test]
    fn structure_preserved_is_always_false() {
        // Both compressing and pass-through paths must report
        // structure_preserved=false: the wrapper is a LossyTransform,
        // even when the inner compressor decides to pass through (the
        // contract is about the trait's promise, not the run's outcome).
        let w = DiffCompressorWrapper::default();
        // Path 1: real compression fires.
        let r1 = w
            .apply(&build_synthetic_diff(8), &CompressionContext::default())
            .unwrap();
        assert!(!r1.structure_preserved);

        // Path 2: short input → DiffCompressor pass-through.
        let r2 = w
            .apply(
                "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b",
                &CompressionContext::default(),
            )
            .unwrap();
        assert!(!r2.structure_preserved);
    }

    #[test]
    fn custom_config_is_honored() {
        // Build with a custom config that disables CCR; verify
        // reversible_via stays None even on a diff that would
        // otherwise produce a cache_key.
        let cfg = DiffCompressorConfig {
            enable_ccr: false,
            ..Default::default()
        };
        let w = DiffCompressorWrapper::new(cfg);
        let input = build_synthetic_diff(8);
        let result = w
            .apply(&input, &CompressionContext::default())
            .expect("valid diff");
        assert!(
            result.reversible_via.is_none(),
            "enable_ccr=false should suppress cache_key"
        );
    }
}
