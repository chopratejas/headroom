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

use headroom_core::transforms::smart_crusher::compaction::DocumentCompactor;
use headroom_core::transforms::smart_crusher::{
    CrushResult as RustCrushResult, SmartCrusher as RustSmartCrusher,
    SmartCrusherConfig as RustSmartCrusherConfig,
};
use headroom_core::transforms::{
    detect_content_type as rust_detect_content_type,
    is_json_array_of_dicts as rust_is_json_array_of_dicts, CacheStats as RustCacheStats,
    CachedResult as RustCachedResult, CompressionCache as RustCompressionCache,
    CompressionStrategy as RustCompressionStrategy, ContentRouterConfig as RustContentRouterConfig,
    ContentType as RustContentType, DetectionResult as RustDetectionResult, DiffCompressionResult,
    DiffCompressor, DiffCompressorConfig, DiffCompressorStats,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::sync::Arc;
use std::time::Duration;

/// Identity stub used by the Python smoke test to verify linkage.
#[pyfunction]
fn hello() -> &'static str {
    headroom_core::hello()
}

fn type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Build the dict returned by `SmartCrusher.crush_array_json`. Kept
/// outside `#[pymethods]` so we can `unwrap()` `set_item` (it cannot
/// fail when keys are static str literals and values are owned String /
/// Option<String> / Option<&'static str>) without tripping the
/// `clippy::useless_conversion` false positive that fires inside the
/// pyo3 0.22 method-attribute macro.
fn build_crush_array_dict<'py>(
    py: Python<'py>,
    kept_json: String,
    ccr_hash: Option<String>,
    dropped_summary: String,
    strategy_info: String,
    compacted: Option<String>,
    compaction_kind: Option<&'static str>,
) -> Bound<'py, PyDict> {
    let dict = PyDict::new_bound(py);
    dict.set_item("items", kept_json).unwrap();
    dict.set_item("ccr_hash", ccr_hash).unwrap();
    dict.set_item("dropped_summary", dropped_summary).unwrap();
    dict.set_item("strategy_info", strategy_info).unwrap();
    dict.set_item("compacted", compacted).unwrap();
    dict.set_item("compaction_kind", compaction_kind).unwrap();
    dict
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
        lossless_min_savings_ratio = 0.30,
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
        lossless_min_savings_ratio: f64,
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
                lossless_min_savings_ratio,
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

    /// Construct WITHOUT the lossless-first compaction stage. The
    /// public `crush()` API runs the lossy path directly (still with
    /// CCR-Dropped retrieval markers populated when rows are dropped).
    /// Used by the legacy parity fixture harness — those fixtures
    /// were recorded against the pre-PR4 lossy-only behavior.
    #[staticmethod]
    #[pyo3(signature = (config = None))]
    fn without_compaction(config: Option<&PySmartCrusherConfig>) -> Self {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        Self {
            inner: RustSmartCrusher::without_compaction(cfg),
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

    /// Crush a JSON array directly and return the structured result.
    ///
    /// Input is a JSON string holding an array (`[item, item, ...]`).
    /// Returns a dict with:
    /// - `items`: JSON array string of the kept rows after compression
    /// - `ccr_hash`: 12-char hash if rows were dropped, else `None`
    /// - `dropped_summary`: `<<ccr:HASH N_rows_offloaded>>` marker
    ///   text, empty if nothing dropped
    /// - `strategy_info`: debug string describing what ran (e.g.
    ///   `"smart_sample"`, `"lossless:table"`, `"none:adaptive_at_limit"`)
    /// - `compacted`: rendered bytes when the lossless path won, else `None`
    /// - `compaction_kind`: `"table" | "buckets" | "ccr" | None`
    ///
    /// This surfaces `CrushArrayResult` to Python so tests and the proxy
    /// runtime can reach the CCR hash directly (rather than parsing it
    /// out of the prompt marker).
    #[pyo3(signature = (items_json, query = "", bias = 1.0))]
    fn crush_array_json<'py>(
        &self,
        py: Python<'py>,
        items_json: &str,
        query: &str,
        bias: f64,
    ) -> Bound<'py, PyDict> {
        // Errors here surface as Python `RuntimeError` via pyo3's panic
        // catcher — callers are expected to pass valid array-shaped JSON.
        let parsed: serde_json::Value = serde_json::from_str(items_json)
            .unwrap_or_else(|e| panic!("items_json must be JSON: {e}"));
        let items = match parsed {
            serde_json::Value::Array(a) => a,
            other => panic!("items_json must be a JSON array, got {}", type_name(&other)),
        };
        let result = self.inner.crush_array(&items, query, bias);
        let kept_json = serde_json::to_string(&serde_json::Value::Array(result.items))
            .expect("serialize kept items");
        build_crush_array_dict(
            py,
            kept_json,
            result.ccr_hash,
            result.dropped_summary,
            result.strategy_info,
            result.compacted,
            result.compaction_kind,
        )
    }

    /// Run the document-level walker on `doc_json` (JSON string) and
    /// return the compacted document as JSON.
    ///
    /// The walker recursively descends through objects, arrays, and
    /// strings; tabular sub-arrays become rendered CSV+schema strings,
    /// long opaque blobs become `<<ccr:HASH,KIND,SIZE>>` markers (with
    /// originals stashed in this crusher's CCR store, so `ccr_get`
    /// resolves them).
    ///
    /// Distinct from `crush_array_json`: this is the lossless walker
    /// pass without per-array lossy crushing — useful when the caller
    /// wants document-shape compaction (forms, configs, mixed records)
    /// rather than statistical row drop.
    fn compact_document_json(&self, doc_json: &str) -> String {
        let parsed: serde_json::Value =
            serde_json::from_str(doc_json).unwrap_or_else(|e| panic!("doc_json must be JSON: {e}"));
        let mut dc = DocumentCompactor::new();
        if let Some(store) = self.inner.ccr_store() {
            dc = dc.with_ccr_store(store.clone());
        }
        let out = dc.compact(parsed);
        serde_json::to_string(&out).expect("serialize compacted document")
    }

    /// Look up an original payload by CCR hash.
    ///
    /// When the lossy path drops rows, it stashes the **full original**
    /// array into the in-memory CCR store keyed by the 12-char hash
    /// embedded in the prompt's `<<ccr:HASH ...>>` marker. The runtime
    /// (proxy server / MCP retrieval tool) calls this to serve the
    /// dropped rows back to the LLM on demand.
    ///
    /// Returns the canonical-JSON serialization of the original
    /// `[item, item, ...]` array, or `None` if the hash is unknown,
    /// expired, or the crusher was constructed without a CCR store.
    fn ccr_get(&self, hash: &str) -> Option<String> {
        self.inner.ccr_store().and_then(|s| s.get(hash))
    }

    /// Number of entries currently held by the CCR store. `0` if no
    /// store is configured. Informational; use it from tests and
    /// telemetry, not from the retrieval hot path.
    fn ccr_len(&self) -> usize {
        self.inner.ccr_store().map(|s| s.len()).unwrap_or(0)
    }
}

// ─── ContentDetector ───────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.content_detector.DetectionResult`.
///
/// Field names + types match the Python dataclass exactly so the existing
/// Python `ContentRouter` (which `import`s `DetectionResult` directly) can
/// continue to read `.content_type`, `.confidence`, and `.metadata` without
/// modification.
///
/// `content_type` is exposed as the lowercase string tag (e.g.
/// `"json_array"`). The Python wrapper translates it back into the
/// `ContentType` enum so the call-site looks identical.
#[pyclass(name = "DetectionResult", module = "headroom._core")]
#[derive(Clone)]
struct PyDetectionResult {
    inner: RustDetectionResult,
}

#[pymethods]
impl PyDetectionResult {
    #[getter]
    fn content_type(&self) -> &'static str {
        self.inner.content_type.as_str()
    }

    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    /// Per-type metadata bag (e.g. `{"language": "python", "pattern_matches": 5}`
    /// for code, `{"item_count": 3, "is_dict_array": true}` for JSON arrays).
    /// Returned as a fresh `dict` so callers can mutate without affecting
    /// the underlying Rust value.
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.inner.metadata {
            // Convert each JSON value into the closest Python primitive.
            // Detection metadata is always a flat dict of scalars (ints,
            // bools, strings) so we don't need to recurse.
            let py_value: PyObject = match v {
                serde_json::Value::Bool(b) => b.into_py(py),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_u64() {
                        i.into_py(py)
                    } else if let Some(i) = n.as_i64() {
                        i.into_py(py)
                    } else if let Some(f) = n.as_f64() {
                        f.into_py(py)
                    } else {
                        py.None()
                    }
                }
                serde_json::Value::String(s) => PyString::new_bound(py, s).into_py(py),
                serde_json::Value::Null => py.None(),
                // Detection never emits arrays / objects in metadata
                // today; if it ever does, fall through to JSON-string for
                // visibility rather than silently dropping.
                other => PyString::new_bound(py, &other.to_string()).into_py(py),
            };
            dict.set_item(k, py_value)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectionResult(content_type={:?}, confidence={}, metadata=<{} keys>)",
            self.inner.content_type.as_str(),
            self.inner.confidence,
            self.inner.metadata.len()
        )
    }
}

/// Detect the type of `content`. Returns a `DetectionResult` with the
/// same field surface as Python's dataclass.
///
/// Releases the GIL while detecting — pattern matching can be substantial
/// on large bodies (HTML scans, 500-line diff windows), and freeing the
/// GIL lets other Python threads make progress in the meantime.
#[pyfunction]
fn detect_content_type(py: Python<'_>, content: &str) -> PyDetectionResult {
    let owned = content.to_string();
    let result = py.allow_threads(move || rust_detect_content_type(&owned));
    PyDetectionResult { inner: result }
}

/// Quick check: is `content` a JSON array of dictionaries (the format
/// `SmartCrusher` natively handles)?
#[pyfunction]
fn is_json_array_of_dicts(py: Python<'_>, content: &str) -> bool {
    let owned = content.to_string();
    py.allow_threads(move || rust_is_json_array_of_dicts(&owned))
}

// Suppress unused-import warning when ContentType isn't referenced
// directly — `as_str()` is the public surface.
const _: fn() = || {
    let _ = RustContentType::PlainText;
};

// ─── ContentRouter — types + config + cache (PR2) ──────────────────────────

/// Mirror of `headroom.transforms.content_router.ContentRouterConfig`.
///
/// PR2 ports the simple fields that drive dispatch decisions. The
/// complex Python-only fields (`exclude_tools`, `read_lifecycle`,
/// `tool_profiles`) come back in PR6 with the `apply()` proxy port.
/// Constructor accepts every field as a kwarg with the same name and
/// default as the Python dataclass.
#[pyclass(name = "ContentRouterConfig", module = "headroom._core")]
#[derive(Clone)]
struct PyContentRouterConfig {
    inner: RustContentRouterConfig,
}

#[pymethods]
impl PyContentRouterConfig {
    #[new]
    #[pyo3(signature = (
        enable_code_aware = false,
        enable_kompress = true,
        enable_smart_crusher = true,
        enable_search_compressor = true,
        enable_log_compressor = true,
        enable_html_extractor = true,
        enable_image_optimizer = true,
        prefer_code_aware_for_code = false,
        mixed_content_threshold = 2,
        min_section_tokens = 20,
        fallback_strategy = "kompress",
        skip_user_messages = true,
        protect_recent_code = 4,
        protect_analysis_context = true,
        protect_recent_reads_fraction = 0.0,
        min_ratio_relaxed = 0.85,
        min_ratio_aggressive = 0.65,
        ccr_enabled = true,
        ccr_inject_marker = true,
        compress_tagged_content = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        enable_code_aware: bool,
        enable_kompress: bool,
        enable_smart_crusher: bool,
        enable_search_compressor: bool,
        enable_log_compressor: bool,
        enable_html_extractor: bool,
        enable_image_optimizer: bool,
        prefer_code_aware_for_code: bool,
        mixed_content_threshold: usize,
        min_section_tokens: usize,
        fallback_strategy: &str,
        skip_user_messages: bool,
        protect_recent_code: usize,
        protect_analysis_context: bool,
        protect_recent_reads_fraction: f64,
        min_ratio_relaxed: f64,
        min_ratio_aggressive: f64,
        ccr_enabled: bool,
        ccr_inject_marker: bool,
        compress_tagged_content: bool,
    ) -> PyResult<Self> {
        let fallback_strategy = fallback_strategy
            .parse::<RustCompressionStrategy>()
            .map_err(|_| {
                PyValueError::new_err(format!(
                    "unknown fallback_strategy {fallback_strategy:?}; expected one of \
                     code_aware, smart_crusher, search, log, kompress, text, diff, html, \
                     mixed, passthrough"
                ))
            })?;
        Ok(Self {
            inner: RustContentRouterConfig {
                enable_code_aware,
                enable_kompress,
                enable_smart_crusher,
                enable_search_compressor,
                enable_log_compressor,
                enable_html_extractor,
                enable_image_optimizer,
                prefer_code_aware_for_code,
                mixed_content_threshold,
                min_section_tokens,
                fallback_strategy,
                skip_user_messages,
                protect_recent_code,
                protect_analysis_context,
                protect_recent_reads_fraction,
                min_ratio_relaxed,
                min_ratio_aggressive,
                ccr_enabled,
                ccr_inject_marker,
                compress_tagged_content,
            },
        })
    }

    // Read-only field accessors mirroring the Python dataclass.
    #[getter]
    fn enable_code_aware(&self) -> bool {
        self.inner.enable_code_aware
    }
    #[getter]
    fn enable_kompress(&self) -> bool {
        self.inner.enable_kompress
    }
    #[getter]
    fn enable_smart_crusher(&self) -> bool {
        self.inner.enable_smart_crusher
    }
    #[getter]
    fn enable_search_compressor(&self) -> bool {
        self.inner.enable_search_compressor
    }
    #[getter]
    fn enable_log_compressor(&self) -> bool {
        self.inner.enable_log_compressor
    }
    #[getter]
    fn enable_html_extractor(&self) -> bool {
        self.inner.enable_html_extractor
    }
    #[getter]
    fn enable_image_optimizer(&self) -> bool {
        self.inner.enable_image_optimizer
    }
    #[getter]
    fn prefer_code_aware_for_code(&self) -> bool {
        self.inner.prefer_code_aware_for_code
    }
    #[getter]
    fn mixed_content_threshold(&self) -> usize {
        self.inner.mixed_content_threshold
    }
    #[getter]
    fn min_section_tokens(&self) -> usize {
        self.inner.min_section_tokens
    }
    #[getter]
    fn fallback_strategy(&self) -> &'static str {
        self.inner.fallback_strategy.as_str()
    }
    #[getter]
    fn skip_user_messages(&self) -> bool {
        self.inner.skip_user_messages
    }
    #[getter]
    fn protect_recent_code(&self) -> usize {
        self.inner.protect_recent_code
    }
    #[getter]
    fn protect_analysis_context(&self) -> bool {
        self.inner.protect_analysis_context
    }
    #[getter]
    fn protect_recent_reads_fraction(&self) -> f64 {
        self.inner.protect_recent_reads_fraction
    }
    #[getter]
    fn min_ratio_relaxed(&self) -> f64 {
        self.inner.min_ratio_relaxed
    }
    #[getter]
    fn min_ratio_aggressive(&self) -> f64 {
        self.inner.min_ratio_aggressive
    }
    #[getter]
    fn ccr_enabled(&self) -> bool {
        self.inner.ccr_enabled
    }
    #[getter]
    fn ccr_inject_marker(&self) -> bool {
        self.inner.ccr_inject_marker
    }
    #[getter]
    fn compress_tagged_content(&self) -> bool {
        self.inner.compress_tagged_content
    }
}

/// Python-facing snapshot of `CompressionCache.stats()`. Matches the
/// keys returned by Python's `CompressionCache.stats` property so
/// existing telemetry scrapers keep working.
#[pyclass(name = "CacheStats", module = "headroom._core")]
#[derive(Clone)]
struct PyCacheStats {
    inner: RustCacheStats,
}

#[pymethods]
impl PyCacheStats {
    #[getter]
    fn cache_hits(&self) -> u64 {
        self.inner.hits
    }
    #[getter]
    fn cache_skip_hits(&self) -> u64 {
        self.inner.skip_hits
    }
    #[getter]
    fn cache_misses(&self) -> u64 {
        self.inner.misses
    }
    #[getter]
    fn cache_evictions(&self) -> u64 {
        self.inner.evictions
    }
    #[getter]
    fn cache_size(&self) -> u64 {
        self.inner.size
    }
    #[getter]
    fn cache_skip_size(&self) -> u64 {
        self.inner.skip_size
    }
    #[getter]
    fn cache_avg_lookup_ns(&self) -> f64 {
        self.inner.avg_lookup_ns
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        // `unwrap()` is fine: every key is a static `&str` literal and
        // every value is a primitive owned by us, so the `set_item`
        // conversions can't fail. Pulling this out of `#[pymethods]`
        // also dodges the pyo3 0.22 `clippy::useless_conversion`
        // macro false-positive that fires on `?` inside method bodies.
        let d = PyDict::new_bound(py);
        d.set_item("cache_hits", self.inner.hits).unwrap();
        d.set_item("cache_skip_hits", self.inner.skip_hits).unwrap();
        d.set_item("cache_misses", self.inner.misses).unwrap();
        d.set_item("cache_evictions", self.inner.evictions).unwrap();
        d.set_item("cache_size", self.inner.size).unwrap();
        d.set_item("cache_skip_size", self.inner.skip_size).unwrap();
        d.set_item("cache_avg_lookup_ns", self.inner.avg_lookup_ns)
            .unwrap();
        d
    }
}

/// Single cached result tuple — mirrors Python's
/// `(compressed, ratio, strategy)` tuple from `CompressionCache.get`.
/// Exposed as a class with attribute access so PR5 doesn't have to
/// unpack a tuple.
#[pyclass(name = "CachedResult", module = "headroom._core")]
#[derive(Clone)]
struct PyCachedResult {
    inner: RustCachedResult,
}

#[pymethods]
impl PyCachedResult {
    #[getter]
    fn compressed(&self) -> &str {
        &self.inner.compressed
    }
    #[getter]
    fn ratio(&self) -> f64 {
        self.inner.ratio
    }
    #[getter]
    fn strategy(&self) -> &str {
        &self.inner.strategy
    }

    fn as_tuple<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyTuple> {
        pyo3::types::PyTuple::new_bound(
            py,
            [
                self.inner.compressed.clone().into_py(py),
                self.inner.ratio.into_py(py),
                self.inner.strategy.clone().into_py(py),
            ],
        )
    }
}

/// Mirror of `headroom.transforms.content_router.CompressionCache`.
///
/// Backed by an `Arc<RustCompressionCache>` so multiple Python handles
/// to the same logical cache share storage. The proxy hot path will
/// hold one cache per worker pool; tests can build their own.
#[pyclass(name = "CompressionCache", module = "headroom._core")]
struct PyCompressionCache {
    inner: Arc<RustCompressionCache>,
}

#[pymethods]
impl PyCompressionCache {
    #[new]
    #[pyo3(signature = (ttl_seconds = 1800))]
    fn new(ttl_seconds: u64) -> Self {
        Self {
            inner: Arc::new(RustCompressionCache::with_ttl(Duration::from_secs(
                ttl_seconds,
            ))),
        }
    }

    fn get(&self, key: u64) -> Option<PyCachedResult> {
        self.inner.get(key).map(|r| PyCachedResult { inner: r })
    }

    fn is_skipped(&self, key: u64) -> bool {
        self.inner.is_skipped(key)
    }

    fn put(&self, key: u64, compressed: &str, ratio: f64, strategy: &str) {
        self.inner
            .put(key, compressed.to_string(), ratio, strategy.to_string());
    }

    fn mark_skip(&self, key: u64) {
        self.inner.mark_skip(key);
    }

    fn move_to_skip(&self, key: u64) {
        self.inner.move_to_skip(key);
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    fn skip_size(&self) -> usize {
        self.inner.skip_size()
    }

    #[getter]
    fn stats(&self) -> PyCacheStats {
        PyCacheStats {
            inner: self.inner.stats(),
        }
    }

    fn clear(&self) {
        self.inner.clear();
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
    m.add_class::<PyDetectionResult>()?;
    m.add_function(wrap_pyfunction!(detect_content_type, m)?)?;
    m.add_function(wrap_pyfunction!(is_json_array_of_dicts, m)?)?;
    m.add_class::<PyContentRouterConfig>()?;
    m.add_class::<PyCacheStats>()?;
    m.add_class::<PyCachedResult>()?;
    m.add_class::<PyCompressionCache>()?;
    Ok(())
}
