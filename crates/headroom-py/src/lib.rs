//! PyO3 bindings for headroom-core. Exposed to Python as `headroom._core`.
//!
//! # Stage 3b — diff_compressor bridge
//!
//! The `DiffCompressor` family is exported here so the Python
//! `ContentRouter` can route to the Rust implementation in-process via
//! PyO3 instead of running the Python port. Backend selection happens in
//! `headroom.transforms._rust_diff_compressor.RustBackedDiffCompressor`,
//! which mirrors the Python `DiffCompressor` API one-for-one (so callers
//! don't notice the swap).
//!
//! Why in-process: ContentRouter compresses on the proxy's hot path. Any
//! IPC / subprocess / RPC bridge would dominate the cost we're trying to
//! save. PyO3 calls cost ~microseconds; staying in-process is ~free.

use std::collections::BTreeMap;

use headroom_core::transforms::smart_crusher::{
    CrushResult as RustCrushResult, SmartCrusher as RustSmartCrusher,
    SmartCrusherConfig as RustSmartCrusherConfig,
};
use headroom_core::transforms::{
    DiffCompressionResult, DiffCompressor, DiffCompressorConfig, DiffCompressorStats,
};
use pyo3::prelude::*;

/// Identity stub used by the Python smoke test to verify linkage.
#[pyfunction]
fn hello() -> &'static str {
    headroom_core::hello()
}

// ─── DiffCompressorConfig ──────────────────────────────────────────────────

/// Mirror of `headroom.transforms.diff_compressor.DiffCompressorConfig`.
/// Defaults match Python; constructor accepts every field as a kwarg with
/// the same name and type as the Python dataclass for drop-in
/// compatibility.
#[pyclass(name = "DiffCompressorConfig", module = "headroom._core")]
#[derive(Clone)]
struct PyDiffCompressorConfig {
    inner: DiffCompressorConfig,
}

#[pymethods]
impl PyDiffCompressorConfig {
    #[new]
    #[pyo3(signature = (
        max_context_lines = 2,
        max_hunks_per_file = 10,
        max_files = 20,
        always_keep_additions = true,
        always_keep_deletions = true,
        enable_ccr = true,
        min_lines_for_ccr = 50,
        min_compression_ratio_for_ccr = 0.8,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_context_lines: usize,
        max_hunks_per_file: usize,
        max_files: usize,
        always_keep_additions: bool,
        always_keep_deletions: bool,
        enable_ccr: bool,
        min_lines_for_ccr: usize,
        min_compression_ratio_for_ccr: f64,
    ) -> Self {
        Self {
            inner: DiffCompressorConfig {
                max_context_lines,
                max_hunks_per_file,
                max_files,
                always_keep_additions,
                always_keep_deletions,
                enable_ccr,
                min_lines_for_ccr,
                min_compression_ratio_for_ccr,
            },
        }
    }

    // Read-only field accessors mirroring the Python dataclass surface.
    #[getter]
    fn max_context_lines(&self) -> usize {
        self.inner.max_context_lines
    }
    #[getter]
    fn max_hunks_per_file(&self) -> usize {
        self.inner.max_hunks_per_file
    }
    #[getter]
    fn max_files(&self) -> usize {
        self.inner.max_files
    }
    #[getter]
    fn always_keep_additions(&self) -> bool {
        self.inner.always_keep_additions
    }
    #[getter]
    fn always_keep_deletions(&self) -> bool {
        self.inner.always_keep_deletions
    }
    #[getter]
    fn enable_ccr(&self) -> bool {
        self.inner.enable_ccr
    }
    #[getter]
    fn min_lines_for_ccr(&self) -> usize {
        self.inner.min_lines_for_ccr
    }
    #[getter]
    fn min_compression_ratio_for_ccr(&self) -> f64 {
        self.inner.min_compression_ratio_for_ccr
    }

    fn __repr__(&self) -> String {
        format!(
            "DiffCompressorConfig(max_context_lines={}, max_hunks_per_file={}, max_files={}, \
             always_keep_additions={}, always_keep_deletions={}, enable_ccr={}, \
             min_lines_for_ccr={}, min_compression_ratio_for_ccr={})",
            self.inner.max_context_lines,
            self.inner.max_hunks_per_file,
            self.inner.max_files,
            self.inner.always_keep_additions,
            self.inner.always_keep_deletions,
            self.inner.enable_ccr,
            self.inner.min_lines_for_ccr,
            self.inner.min_compression_ratio_for_ccr,
        )
    }
}

// ─── DiffCompressionResult ─────────────────────────────────────────────────

/// Mirror of `headroom.transforms.diff_compressor.DiffCompressionResult`.
/// Read-only on the Python side: ContentRouter consumes fields, doesn't
/// mutate. `compression_ratio` and `tokens_saved_estimate` are exposed as
/// methods (not `@property`) — Python callers reach them via `.method()`.
/// The Python adapter wraps and re-exposes them as properties for full
/// dataclass compatibility.
#[pyclass(name = "DiffCompressionResult", module = "headroom._core")]
struct PyDiffCompressionResult {
    inner: DiffCompressionResult,
}

#[pymethods]
impl PyDiffCompressionResult {
    #[getter]
    fn compressed(&self) -> &str {
        &self.inner.compressed
    }
    #[getter]
    fn original_line_count(&self) -> usize {
        self.inner.original_line_count
    }
    #[getter]
    fn compressed_line_count(&self) -> usize {
        self.inner.compressed_line_count
    }
    #[getter]
    fn files_affected(&self) -> usize {
        self.inner.files_affected
    }
    #[getter]
    fn additions(&self) -> usize {
        self.inner.additions
    }
    #[getter]
    fn deletions(&self) -> usize {
        self.inner.deletions
    }
    #[getter]
    fn hunks_kept(&self) -> usize {
        self.inner.hunks_kept
    }
    #[getter]
    fn hunks_removed(&self) -> usize {
        self.inner.hunks_removed
    }
    #[getter]
    fn cache_key(&self) -> Option<String> {
        self.inner.cache_key.clone()
    }

    /// Mirror of Python `@property compression_ratio`. Returns
    /// `compressed_line_count / original_line_count` (1.0 if input was
    /// empty).
    fn compression_ratio(&self) -> f64 {
        if self.inner.original_line_count == 0 {
            1.0
        } else {
            self.inner.compressed_line_count as f64 / self.inner.original_line_count as f64
        }
    }

    /// Mirror of Python `@property tokens_saved_estimate`. Same `chars *
    /// 40 / 4` heuristic; bytes-equivalent numeric result.
    fn tokens_saved_estimate(&self) -> usize {
        let saved = self
            .inner
            .original_line_count
            .saturating_sub(self.inner.compressed_line_count);
        (saved * 40) / 4
    }

    fn __repr__(&self) -> String {
        format!(
            "DiffCompressionResult(compressed=<{} chars>, original_line_count={}, \
             compressed_line_count={}, files_affected={}, additions={}, deletions={}, \
             hunks_kept={}, hunks_removed={}, cache_key={:?})",
            self.inner.compressed.len(),
            self.inner.original_line_count,
            self.inner.compressed_line_count,
            self.inner.files_affected,
            self.inner.additions,
            self.inner.deletions,
            self.inner.hunks_kept,
            self.inner.hunks_removed,
            self.inner.cache_key,
        )
    }
}

// ─── DiffCompressorStats ───────────────────────────────────────────────────

/// Mirror of Rust `DiffCompressorStats` — sidecar observability not
/// present in the Python dataclass. Returned only from `compress_with_stats`,
/// which the Python adapter exposes as a method on the wrapper. `Vec`s are
/// returned as Python lists; the `BTreeMap` becomes a `dict`.
#[pyclass(name = "DiffCompressorStats", module = "headroom._core")]
struct PyDiffCompressorStats {
    inner: DiffCompressorStats,
}

#[pymethods]
impl PyDiffCompressorStats {
    #[getter]
    fn input_lines(&self) -> usize {
        self.inner.input_lines
    }
    #[getter]
    fn output_lines(&self) -> usize {
        self.inner.output_lines
    }
    #[getter]
    fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio
    }
    #[getter]
    fn files_total(&self) -> usize {
        self.inner.files_total
    }
    #[getter]
    fn files_kept(&self) -> usize {
        self.inner.files_kept
    }
    #[getter]
    fn files_dropped(&self) -> Vec<String> {
        self.inner.files_dropped.clone()
    }
    #[getter]
    fn hunks_total(&self) -> usize {
        self.inner.hunks_total
    }
    #[getter]
    fn hunks_kept(&self) -> usize {
        self.inner.hunks_kept
    }
    #[getter]
    fn hunks_dropped(&self) -> usize {
        self.inner.hunks_dropped
    }
    #[getter]
    fn hunks_dropped_per_file(&self) -> BTreeMap<String, usize> {
        self.inner.hunks_dropped_per_file.clone()
    }
    #[getter]
    fn context_lines_input(&self) -> usize {
        self.inner.context_lines_input
    }
    #[getter]
    fn context_lines_kept(&self) -> usize {
        self.inner.context_lines_kept
    }
    #[getter]
    fn context_lines_trimmed(&self) -> usize {
        self.inner.context_lines_trimmed
    }
    #[getter]
    fn largest_hunk_kept_lines(&self) -> usize {
        self.inner.largest_hunk_kept_lines
    }
    #[getter]
    fn largest_hunk_dropped_lines(&self) -> usize {
        self.inner.largest_hunk_dropped_lines
    }
    #[getter]
    fn parse_warnings(&self) -> Vec<String> {
        self.inner.parse_warnings.clone()
    }
    #[getter]
    fn processing_duration_us(&self) -> u64 {
        self.inner.processing_duration_us
    }
    #[getter]
    fn cache_key_emitted(&self) -> bool {
        self.inner.cache_key_emitted
    }
    #[getter]
    fn ccr_skipped_reason(&self) -> Option<String> {
        self.inner.ccr_skipped_reason.clone()
    }
    #[getter]
    fn file_mode_normalizations(&self) -> Vec<(String, String)> {
        self.inner.file_mode_normalizations.clone()
    }
    #[getter]
    fn binary_files_simplified(&self) -> Vec<String> {
        self.inner.binary_files_simplified.clone()
    }
}

// ─── DiffCompressor ────────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.diff_compressor.DiffCompressor`. The
/// Python adapter wraps this in `RustBackedDiffCompressor` so
/// `ContentRouter` can swap backends transparently.
#[pyclass(name = "DiffCompressor", module = "headroom._core")]
struct PyDiffCompressor {
    inner: DiffCompressor,
}

#[pymethods]
impl PyDiffCompressor {
    /// `__init__(config: DiffCompressorConfig | None = None)` — matches the
    /// Python constructor signature one-for-one.
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<&PyDiffCompressorConfig>) -> Self {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        Self {
            inner: DiffCompressor::new(cfg),
        }
    }

    /// `compress(content: str, context: str = "") -> DiffCompressionResult`.
    /// Argument order and keyword names match the Python implementation.
    #[pyo3(signature = (content, context = ""))]
    fn compress(&self, content: &str, context: &str) -> PyDiffCompressionResult {
        PyDiffCompressionResult {
            inner: self.inner.compress(content, context),
        }
    }

    /// `compress_with_stats(content, context="") -> (result, stats)`.
    /// Sidecar API not present in Python — exposes the Rust observability
    /// struct alongside the parity-equal result. Returned as a 2-tuple to
    /// keep the call site Pythonic.
    #[pyo3(signature = (content, context = ""))]
    fn compress_with_stats(
        &self,
        content: &str,
        context: &str,
    ) -> (PyDiffCompressionResult, PyDiffCompressorStats) {
        let (result, stats) = self.inner.compress_with_stats(content, context);
        (
            PyDiffCompressionResult { inner: result },
            PyDiffCompressorStats { inner: stats },
        )
    }
}

// ─── SmartCrusherConfig ────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.smart_crusher.SmartCrusherConfig`.
/// Defaults match Python's dataclass byte-for-byte. The constructor
/// accepts every field as a kwarg with the same name and type so the
/// Python shim can pass `SmartCrusherConfig(**asdict(py_cfg))`.
#[pyclass(name = "SmartCrusherConfig", module = "headroom._core")]
#[derive(Clone)]
struct PySmartCrusherConfig {
    inner: RustSmartCrusherConfig,
}

#[pymethods]
impl PySmartCrusherConfig {
    #[new]
    #[pyo3(signature = (
        enabled = true,
        min_items_to_analyze = 5,
        min_tokens_to_crush = 200,
        variance_threshold = 2.0,
        uniqueness_threshold = 0.1,
        similarity_threshold = 0.8,
        max_items_after_crush = 15,
        preserve_change_points = true,
        factor_out_constants = false,
        include_summaries = false,
        use_feedback_hints = true,
        toin_confidence_threshold = 0.5,
        dedup_identical_items = true,
        first_fraction = 0.3,
        last_fraction = 0.15,
        relevance_threshold = 0.3,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        enabled: bool,
        min_items_to_analyze: usize,
        min_tokens_to_crush: usize,
        variance_threshold: f64,
        uniqueness_threshold: f64,
        similarity_threshold: f64,
        max_items_after_crush: usize,
        preserve_change_points: bool,
        factor_out_constants: bool,
        include_summaries: bool,
        use_feedback_hints: bool,
        toin_confidence_threshold: f64,
        dedup_identical_items: bool,
        first_fraction: f64,
        last_fraction: f64,
        relevance_threshold: f64,
    ) -> Self {
        Self {
            inner: RustSmartCrusherConfig {
                enabled,
                min_items_to_analyze,
                min_tokens_to_crush,
                variance_threshold,
                uniqueness_threshold,
                similarity_threshold,
                max_items_after_crush,
                preserve_change_points,
                factor_out_constants,
                include_summaries,
                use_feedback_hints,
                toin_confidence_threshold,
                dedup_identical_items,
                first_fraction,
                last_fraction,
                relevance_threshold,
            },
        }
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }
    #[getter]
    fn min_items_to_analyze(&self) -> usize {
        self.inner.min_items_to_analyze
    }
    #[getter]
    fn min_tokens_to_crush(&self) -> usize {
        self.inner.min_tokens_to_crush
    }
    #[getter]
    fn variance_threshold(&self) -> f64 {
        self.inner.variance_threshold
    }
    #[getter]
    fn uniqueness_threshold(&self) -> f64 {
        self.inner.uniqueness_threshold
    }
    #[getter]
    fn similarity_threshold(&self) -> f64 {
        self.inner.similarity_threshold
    }
    #[getter]
    fn max_items_after_crush(&self) -> usize {
        self.inner.max_items_after_crush
    }
    #[getter]
    fn preserve_change_points(&self) -> bool {
        self.inner.preserve_change_points
    }
    #[getter]
    fn factor_out_constants(&self) -> bool {
        self.inner.factor_out_constants
    }
    #[getter]
    fn include_summaries(&self) -> bool {
        self.inner.include_summaries
    }
    #[getter]
    fn use_feedback_hints(&self) -> bool {
        self.inner.use_feedback_hints
    }
    #[getter]
    fn toin_confidence_threshold(&self) -> f64 {
        self.inner.toin_confidence_threshold
    }
    #[getter]
    fn dedup_identical_items(&self) -> bool {
        self.inner.dedup_identical_items
    }
    #[getter]
    fn first_fraction(&self) -> f64 {
        self.inner.first_fraction
    }
    #[getter]
    fn last_fraction(&self) -> f64 {
        self.inner.last_fraction
    }
    #[getter]
    fn relevance_threshold(&self) -> f64 {
        self.inner.relevance_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "SmartCrusherConfig(enabled={}, min_items_to_analyze={}, \
             min_tokens_to_crush={}, max_items_after_crush={}, \
             relevance_threshold={})",
            self.inner.enabled,
            self.inner.min_items_to_analyze,
            self.inner.min_tokens_to_crush,
            self.inner.max_items_after_crush,
            self.inner.relevance_threshold,
        )
    }
}

// ─── CrushResult ───────────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.smart_crusher.CrushResult`. Read-only;
/// the Python shim builds its own dataclass instance from these
/// attributes so callers that destructure with `asdict()` keep working.
#[pyclass(name = "CrushResult", module = "headroom._core")]
struct PyCrushResult {
    inner: RustCrushResult,
}

#[pymethods]
impl PyCrushResult {
    #[getter]
    fn compressed(&self) -> &str {
        &self.inner.compressed
    }
    #[getter]
    fn original(&self) -> &str {
        &self.inner.original
    }
    #[getter]
    fn was_modified(&self) -> bool {
        self.inner.was_modified
    }
    #[getter]
    fn strategy(&self) -> &str {
        &self.inner.strategy
    }

    fn __repr__(&self) -> String {
        format!(
            "CrushResult(compressed=<{} chars>, was_modified={}, strategy={:?})",
            self.inner.compressed.len(),
            self.inner.was_modified,
            self.inner.strategy,
        )
    }
}

// ─── SmartCrusher ──────────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.smart_crusher.SmartCrusher`.
///
/// Constructor accepts only `config` — Python's `relevance_config`,
/// `scorer`, and `ccr_config` parameters are handled in the Python
/// shim (Stage 3c.1 keeps the optional subsystems disabled in Rust;
/// the shim drops those args to preserve call-site compatibility).
#[pyclass(name = "SmartCrusher", module = "headroom._core")]
struct PySmartCrusher {
    inner: RustSmartCrusher,
}

#[pymethods]
impl PySmartCrusher {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<&PySmartCrusherConfig>) -> Self {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        Self {
            inner: RustSmartCrusher::new(cfg),
        }
    }

    /// `crush(content, query="", bias=1.0) -> CrushResult`. Argument
    /// order and keyword names mirror the Python implementation.
    #[pyo3(signature = (content, query = "", bias = 1.0))]
    fn crush(&self, content: &str, query: &str, bias: f64) -> PyCrushResult {
        PyCrushResult {
            inner: self.inner.crush(content, query, bias),
        }
    }

    /// `smart_crush_content(content, query="", bias=1.0) -> (str, bool, str)`.
    /// Mirrors Python's `_smart_crush_content` — used by
    /// `smart_crush_tool_output` convenience function and direct
    /// callers that want the tuple form.
    #[pyo3(signature = (content, query = "", bias = 1.0))]
    fn smart_crush_content(&self, content: &str, query: &str, bias: f64) -> (String, bool, String) {
        self.inner.smart_crush_content(content, query, bias)
    }
}

// ─── Module init ───────────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_class::<PyDiffCompressorConfig>()?;
    m.add_class::<PyDiffCompressionResult>()?;
    m.add_class::<PyDiffCompressorStats>()?;
    m.add_class::<PyDiffCompressor>()?;
    m.add_class::<PySmartCrusherConfig>()?;
    m.add_class::<PyCrushResult>()?;
    m.add_class::<PySmartCrusher>()?;
    Ok(())
}
