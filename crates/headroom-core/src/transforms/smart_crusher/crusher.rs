//! `SmartCrusher` struct — top-level entry point for compression.
//!
//! Owns the `config`, `anchor_selector`, `scorer`, and `analyzer`
//! singletons that every per-message call needs. Constructed once
//! per process; the struct is `Send + Sync` so it can sit behind an
//! `Arc` in a multi-threaded proxy.
//!
//! This module ports three Python entry points:
//!
//! - `_execute_plan` (line 3617) → `SmartCrusher::execute_plan`
//! - `_crush_array`  (line 2400) → `SmartCrusher::crush_array`
//! - `_crush_mixed_array` (line 2914) → `SmartCrusher::crush_mixed_array`
//!
//! # Stubs that match Python's "everything-disabled" path
//!
//! Python's `_crush_array` calls into TOIN (cross-user pattern
//! learning), feedback (per-tool compression hints), CCR (compress-
//! cache-retrieve store), and telemetry. All four are large separate
//! systems with their own state. For the like-for-like port at Stage
//! 3c.1, we mirror Python's behavior **when those subsystems are
//! disabled**:
//!
//! - **TOIN**: never produces a recommendation, never overrides
//!   `effective_max_items`, never injects preserve_fields/strategy/level.
//! - **Feedback**: never produces hints; default `effective_max_items`.
//! - **CCR**: `enabled=false`; result has `ccr_hash = None`.
//! - **Telemetry**: no-op.
//! - **`_compress_text_within_items`**: pass-through (returns input
//!   unchanged) since text compression has its own port pipeline.
//! - **`summarize_dropped_items`**: empty string.
//!
//! Parity fixtures will be recorded with all four disabled on the
//! Python side, locking byte-equal output. The TOIN/CCR/feedback
//! integration ports happen later (Stage 3c.2 follow-ups).

use serde_json::Value;

use super::analyzer::SmartAnalyzer;
use super::builder::SmartCrusherBuilder;
use super::classifier::{classify_array, ArrayType};
use super::compaction::{Compaction, CompactionStage};
use super::config::SmartCrusherConfig;
use super::crushers::{compute_k_split, crush_number_array, crush_object, crush_string_array};
use super::planning::SmartCrusherPlanner;
use super::traits::{Constraint, CrushEvent, Observer};
use super::types::{CompressionPlan, CompressionStrategy, CrushResult};
use crate::relevance::RelevanceScorer;
use crate::transforms::adaptive_sizer::compute_optimal_k;
use crate::transforms::anchor_selector::AnchorSelector;

/// Return type for `crush_array` — mirrors Python's
/// `(crushed_items, strategy_info, ccr_hash, dropped_summary)` tuple.
///
/// Stage 3c.2 PR2 added `compacted` and `compaction_kind` for the new
/// lossless-first compaction stage. They stay `None` when no
/// [`CompactionStage`] is configured on the crusher (default), so
/// existing parity guarantees are preserved.
pub struct CrushArrayResult {
    pub items: Vec<Value>,
    /// Strategy debug string, e.g. `"smart_sample"`, `"top_n"`,
    /// `"none:adaptive_at_limit"`, `"skip:unique_entities_no_signal"`.
    /// When compaction wins, this is `"compaction:<formatter_name>"`.
    pub strategy_info: String,
    /// CCR retrieval hash if caching is enabled. Stub: always `None`.
    pub ccr_hash: Option<String>,
    /// Categorical summary of dropped items. Stub: always empty.
    pub dropped_summary: String,
    /// Rendered bytes from the compaction stage. `Some(s)` when the
    /// stage was configured AND produced non-`Untouched` output;
    /// `None` otherwise (default OSS path).
    pub compacted: Option<String>,
    /// Top-level [`Compaction`] variant tag — `"table"`, `"buckets"`,
    /// `"ccr"`, or `"untouched"`. Mirrors `compacted` — populated only
    /// when the compaction stage ran.
    pub compaction_kind: Option<&'static str>,
}

/// Top-level SmartCrusher.
///
/// Three pluggable extensions (Stage 3c.2 PR1):
/// - `scorer` — relevance scoring (`HybridScorer` by default).
/// - `constraints` — must-keep predicates (`KeepErrorsConstraint` +
///   `KeepStructuralOutliersConstraint` by default).
/// - `observers` — decision-stream telemetry (`TracingObserver` by
///   default).
///
/// Compose via [`SmartCrusherBuilder`]; or call `SmartCrusher::new()`
/// for the OSS default composition.
pub struct SmartCrusher {
    pub config: SmartCrusherConfig,
    pub anchor_selector: AnchorSelector,
    pub scorer: Box<dyn RelevanceScorer + Send + Sync>,
    pub analyzer: SmartAnalyzer,
    pub constraints: Vec<Box<dyn Constraint>>,
    pub observers: Vec<Box<dyn Observer>>,
    /// Optional lossless-first compaction stage (Stage 3c.2 PR2). When
    /// set, `crush_array` runs compaction up front and short-circuits
    /// the lossy path on success. When `None` (default OSS), parity
    /// with the pre-PR2 lossy-only pipeline is preserved exactly.
    pub compaction: Option<CompactionStage>,
}

impl SmartCrusher {
    /// Construct with the OSS default composition: `HybridScorer` +
    /// `KeepErrorsConstraint` + `KeepStructuralOutliersConstraint` +
    /// `TracingObserver`. Drop-in for callers that don't need
    /// custom extensions — pre-PR1 behavior is preserved exactly.
    pub fn new(config: SmartCrusherConfig) -> Self {
        SmartCrusherBuilder::new(config)
            .with_default_oss_setup()
            .build()
    }

    /// Begin a builder chain for custom composition. The Enterprise
    /// entry point: swap the scorer, add business-rule constraints,
    /// attach an audit observer.
    pub fn builder(config: SmartCrusherConfig) -> SmartCrusherBuilder {
        SmartCrusherBuilder::new(config)
    }

    /// Construct with a custom scorer (legacy convenience). Equivalent
    /// to `SmartCrusher::builder(config).with_scorer(scorer).with_default_oss_setup().build()`
    /// minus the default scorer override; preserved for backward
    /// compatibility with pre-PR1 callers.
    pub fn with_scorer(
        config: SmartCrusherConfig,
        scorer: Box<dyn RelevanceScorer + Send + Sync>,
    ) -> Self {
        SmartCrusherBuilder::new(config)
            .with_scorer(scorer)
            .add_default_oss_constraints()
            .build()
    }

    /// Construct directly from owned parts. Used by
    /// [`SmartCrusherBuilder::build`] — not part of the public stable
    /// API. Prefer the builder.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        config: SmartCrusherConfig,
        anchor_selector: AnchorSelector,
        scorer: Box<dyn RelevanceScorer + Send + Sync>,
        analyzer: SmartAnalyzer,
        constraints: Vec<Box<dyn Constraint>>,
        observers: Vec<Box<dyn Observer>>,
        compaction: Option<CompactionStage>,
    ) -> Self {
        SmartCrusher {
            config,
            anchor_selector,
            scorer,
            analyzer,
            constraints,
            observers,
            compaction,
        }
    }

    fn planner(&self) -> SmartCrusherPlanner<'_> {
        SmartCrusherPlanner::new(
            &self.config,
            &self.anchor_selector,
            &*self.scorer,
            &self.analyzer,
            &self.constraints,
        )
    }

    /// Execute a `CompressionPlan` against `items`, returning the
    /// kept-items list in original-array order. Mirrors Python's
    /// `_execute_plan` (line 3617-3633).
    ///
    /// Schema-preserving: each kept item is cloned unchanged. No
    /// summary objects, generated fields, or wrapper metadata.
    pub fn execute_plan(&self, plan: &CompressionPlan, items: &[Value]) -> Vec<Value> {
        let mut indices = plan.keep_indices.clone();
        indices.sort_unstable();
        indices
            .into_iter()
            .filter(|&idx| idx < items.len())
            .map(|idx| items[idx].clone())
            .collect()
    }

    /// Top-level entry point. Mirrors Python `SmartCrusher.crush`
    /// (line 1581-1603) — used by `ContentRouter` when routing JSON
    /// arrays.
    ///
    /// Parses `content` as JSON, recursively processes it (compressing
    /// arrays at every depth via the appropriate per-type crusher),
    /// then re-serializes with Python-compatible formatting (`, ` and
    /// `: ` separators, ASCII-escaped non-ASCII).
    ///
    /// Returns a `CrushResult` with:
    /// - `compressed`: the re-serialized JSON.
    /// - `original`: the input string (unmodified).
    /// - `was_modified`: whether `compressed` differs from `content`'s
    ///   trimmed form.
    /// - `strategy`: combined strategy info from all crushed arrays
    ///   (or `"passthrough"`).
    pub fn crush(&self, content: &str, query: &str, bias: f64) -> CrushResult {
        let start = std::time::Instant::now();
        let (compressed, was_modified, info) = self.smart_crush_content(content, query, bias);
        let strategy = if info.is_empty() {
            "passthrough".to_string()
        } else {
            info
        };

        // Fire one event per top-level crush. Cheap when no observers
        // are configured (`for o in &[]` is a single null-pointer
        // check); cheap when only `TracingObserver` is configured if
        // the subscriber filters `debug` out (the default in
        // production). Custom observers — audit logs, Loop training
        // stream, metrics — pay whatever they pay.
        if !self.observers.is_empty() {
            let event = CrushEvent {
                strategy: strategy.clone(),
                input_bytes: content.len(),
                output_bytes: compressed.len(),
                elapsed_ns: start.elapsed().as_nanos() as u64,
                was_modified,
            };
            for observer in &self.observers {
                observer.on_event(&event);
            }
        }

        CrushResult {
            compressed,
            original: content.to_string(),
            was_modified,
            strategy,
        }
    }

    /// `SmartCrusher._smart_crush_content` (Python line 2243-2301).
    /// JSON-parse, recursively process, re-serialize. CCR marker
    /// injection is stubbed (CCR is disabled in this stage).
    ///
    /// Returns `(crushed_content, was_modified, info)`.
    pub fn smart_crush_content(
        &self,
        content: &str,
        query_context: &str,
        bias: f64,
    ) -> (String, bool, String) {
        // Parse — non-JSON content passes through unchanged.
        let Ok(parsed) = serde_json::from_str::<Value>(content) else {
            return (content.to_string(), false, String::new());
        };

        let (crushed, info) = self.process_value(&parsed, 0, query_context, bias);

        // Re-serialize with Python `safe_json_dumps` formatting:
        // compact `(",", ":")` separators + `ensure_ascii=False`,
        // preserving object-key insertion order. Matches the Python
        // SmartCrusher output bytes the proxy writes.
        let result = crate::transforms::anchor_selector::python_safe_json_dumps(&crushed);
        let was_modified = result != content.trim();
        (result, was_modified, info)
    }

    /// Maximum recursion depth for nested JSON. Mirrors Python's
    /// `_MAX_PROCESS_DEPTH = 50`. Beyond this, values are returned as-is.
    const MAX_PROCESS_DEPTH: usize = 50;

    /// Recursively process a value, crushing arrays where appropriate.
    /// Mirrors Python `_process_value` (line 2307-2398).
    ///
    /// Returns `(processed_value, info_string)`. CCR markers are
    /// stubbed (Python's tuple has a third element for them — Rust's
    /// version omits since we never produce markers in this stage).
    pub fn process_value(
        &self,
        value: &Value,
        depth: usize,
        query_context: &str,
        bias: f64,
    ) -> (Value, String) {
        if depth >= Self::MAX_PROCESS_DEPTH {
            return (value.clone(), String::new());
        }

        let mut info_parts: Vec<String> = Vec::new();

        match value {
            Value::Array(arr) => {
                let n = arr.len();
                if n >= self.config.min_items_to_analyze {
                    let arr_type = classify_array(arr);
                    match arr_type {
                        ArrayType::DictArray => {
                            let result = self.crush_array(arr, query_context, bias);
                            info_parts.push(format!(
                                "{}({}->{})",
                                result.strategy_info,
                                n,
                                result.items.len()
                            ));
                            return (Value::Array(result.items), info_parts.join(","));
                        }
                        ArrayType::StringArray => {
                            let strs: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                            let (crushed, strategy) = crush_string_array(&strs, &self.config, bias);
                            info_parts.push(format!("{}({}->{})", strategy, n, crushed.len()));
                            let crushed_values: Vec<Value> =
                                crushed.into_iter().map(Value::String).collect();
                            return (Value::Array(crushed_values), info_parts.join(","));
                        }
                        ArrayType::NumberArray => {
                            let (crushed, strategy) = crush_number_array(arr, &self.config, bias);
                            info_parts.push(format!("{}({}->{})", strategy, n, crushed.len()));
                            return (Value::Array(crushed), info_parts.join(","));
                        }
                        ArrayType::MixedArray => {
                            let (crushed, strategy) =
                                self.crush_mixed_array(arr, query_context, bias);
                            info_parts.push(format!("{}({}->{})", strategy, n, crushed.len()));
                            return (Value::Array(crushed), info_parts.join(","));
                        }
                        // NestedArray, BoolArray, Empty → fall through
                        // to recursive descent.
                        _ => {}
                    }
                }

                // Below threshold or not crushable → recurse into items.
                let mut processed: Vec<Value> = Vec::with_capacity(n);
                for item in arr {
                    let (p_item, p_info) = self.process_value(item, depth + 1, query_context, bias);
                    processed.push(p_item);
                    if !p_info.is_empty() {
                        info_parts.push(p_info);
                    }
                }
                (Value::Array(processed), info_parts.join(","))
            }
            Value::Object(map) => {
                // First pass: recurse into values to compress nested arrays.
                let mut processed = serde_json::Map::new();
                for (k, v) in map {
                    let (p_val, p_info) = self.process_value(v, depth + 1, query_context, bias);
                    processed.insert(k.clone(), p_val);
                    if !p_info.is_empty() {
                        info_parts.push(p_info);
                    }
                }

                // Second pass: if the object itself has many keys,
                // compress at the key level.
                if processed.len() >= self.config.min_items_to_analyze {
                    let (crushed_dict, strategy) = crush_object(&processed, &self.config, bias);
                    if strategy != "object:passthrough" {
                        info_parts.push(strategy);
                        return (Value::Object(crushed_dict), info_parts.join(","));
                    }
                }

                (Value::Object(processed), info_parts.join(","))
            }
            // Scalars — passthrough.
            _ => (value.clone(), String::new()),
        }
    }

    /// Compress an array of dict items.
    ///
    /// Direct port of `_crush_array` (Python line 2400-2687) with the
    /// optional subsystems (TOIN / CCR / feedback / telemetry) wired
    /// in their disabled-by-default behavior. See module-level docs
    /// for the rationale.
    ///
    /// # Pipeline
    ///
    /// 1. Compute `item_strings` once (used as input to adaptive
    ///    sizing and downstream relevance scoring).
    /// 2. `compute_optimal_k` → `adaptive_k`.
    /// 3. If `n <= adaptive_k`, return passthrough.
    /// 4. `analyzer.analyze_array(items)` → `analysis`.
    /// 5. If `analysis.recommended_strategy == Skip`, return passthrough
    ///    with a `skip:<reason>` strategy string.
    /// 6. `planner.create_plan(analysis, items, query_context, ...)`.
    /// 7. `execute_plan(plan, items)` → result.
    /// 8. Strategy info = `analysis.recommended_strategy.as_str()`.
    pub fn crush_array(&self, items: &[Value], query_context: &str, bias: f64) -> CrushArrayResult {
        // ── Stage 0 (PR2): lossless-first compaction (opt-in) ──
        //
        // Runs only when a CompactionStage is configured. Sits BEFORE
        // the existing lossy pipeline so the lossless path can
        // short-circuit on success. When the compactor declines
        // (returns Untouched) the rest of crush_array runs unchanged
        // — preserving byte-equal parity with the pre-PR2 path.
        let compaction_result: Option<(String, &'static str)> =
            if let Some(stage) = &self.compaction {
                let (c, rendered) = stage.run(items);
                if c.was_compacted() {
                    Some((rendered, compaction_kind_str(&c)))
                } else {
                    None
                }
            } else {
                None
            };

        let item_strings: Vec<String> = items
            .iter()
            .map(|i| serde_json::to_string(i).unwrap_or_default())
            .collect();
        let item_str_refs: Vec<&str> = item_strings.iter().map(|s| s.as_str()).collect();

        let max_k = if self.config.max_items_after_crush > 0 {
            Some(self.config.max_items_after_crush)
        } else {
            None
        };
        let adaptive_k = compute_optimal_k(&item_str_refs, bias, 3, max_k);

        // Tier-1 boundary: array already small enough.
        if items.len() <= adaptive_k {
            // Python branches into _compress_text_within_items here;
            // stubbed to passthrough (no within-item text compression
            // in this stage).
            return CrushArrayResult {
                items: items.to_vec(),
                strategy_info: compaction_strategy_or(&compaction_result, "none:adaptive_at_limit"),
                ccr_hash: None,
                dropped_summary: String::new(),
                compacted: compaction_result.as_ref().map(|(s, _)| s.clone()),
                compaction_kind: compaction_result.as_ref().map(|(_, k)| *k),
            };
        }

        // TOIN/feedback paths stubbed → effective_max_items = adaptive_k.
        let effective_max_items = adaptive_k;

        let analysis = self.analyzer.analyze_array(items);

        // Crushability gate: not safe to crush → return original.
        if analysis.recommended_strategy == CompressionStrategy::Skip {
            let reason = match &analysis.crushability {
                Some(c) => format!("skip:{}", c.reason),
                None => String::new(),
            };
            return CrushArrayResult {
                items: items.to_vec(),
                strategy_info: compaction_strategy_or(&compaction_result, &reason),
                ccr_hash: None,
                dropped_summary: String::new(),
                compacted: compaction_result.as_ref().map(|(s, _)| s.clone()),
                compaction_kind: compaction_result.as_ref().map(|(_, k)| *k),
            };
        }

        // Plan + execute.
        let plan = self.planner().create_plan(
            &analysis,
            items,
            query_context,
            None, // preserve_fields (TOIN — stubbed)
            Some(effective_max_items),
            Some(&item_strings),
        );
        let result = self.execute_plan(&plan, items);

        // CCR/telemetry/TOIN-record paths stubbed.
        let strategy_info =
            compaction_strategy_or(&compaction_result, analysis.recommended_strategy.as_str());

        CrushArrayResult {
            items: result,
            strategy_info,
            ccr_hash: None,
            dropped_summary: String::new(),
            compacted: compaction_result.as_ref().map(|(s, _)| s.clone()),
            compaction_kind: compaction_result.as_ref().map(|(_, k)| *k),
        }
    }

    /// Compress a mixed-type array by grouping items by type and
    /// compressing each group with the appropriate handler.
    ///
    /// Direct port of `_crush_mixed_array` (Python line 2914-3013).
    ///
    /// Strategy:
    /// 1. Group by type (dict / str / number / list / null / bool / other).
    /// 2. For groups with >= `min_items_to_analyze` items: apply the
    ///    type-specific compressor.
    /// 3. For small groups: keep all items.
    /// 4. Reassemble in original order.
    ///
    /// Returns `(crushed_items, strategy_string)`.
    pub fn crush_mixed_array(
        &self,
        items: &[Value],
        query_context: &str,
        bias: f64,
    ) -> (Vec<Value>, String) {
        let n = items.len();
        if n <= 8 {
            return (items.to_vec(), "mixed:passthrough".to_string());
        }

        // Group by type, tracking original indices.
        let mut groups: GroupBuckets = GroupBuckets::default();
        for (i, item) in items.iter().enumerate() {
            groups.push(group_key(item), i, item.clone());
        }

        let mut keep_indices: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mut strategy_parts: Vec<String> = Vec::new();

        for (type_key, indices, values) in groups.into_iter() {
            // Small groups: keep all items.
            if values.len() < self.config.min_items_to_analyze {
                keep_indices.extend(&indices);
                continue;
            }

            match type_key {
                "dict" => {
                    let CrushArrayResult { items: crushed, .. } =
                        self.crush_array(&values, query_context, bias);
                    // Find which original indices survived by matching
                    // canonical-JSON serialization. Mirrors Python's
                    // `json.dumps(c, sort_keys=True, default=str)`-keyed
                    // set match.
                    let crushed_keys: std::collections::HashSet<String> =
                        crushed.iter().map(canonical_json_for_match).collect();
                    for (i, idx) in indices.iter().enumerate() {
                        if crushed_keys.contains(&canonical_json_for_match(&values[i])) {
                            keep_indices.insert(*idx);
                        }
                    }
                    strategy_parts.push(format!("dict:{}->{}", values.len(), crushed.len()));
                }
                "str" => {
                    let strs: Vec<&str> = values.iter().filter_map(|v| v.as_str()).collect();
                    let (crushed, _) = crush_string_array(&strs, &self.config, bias);
                    let crushed_set: std::collections::HashSet<&str> =
                        crushed.iter().map(|s| s.as_str()).collect();
                    for (i, idx) in indices.iter().enumerate() {
                        if let Some(s) = values[i].as_str() {
                            if crushed_set.contains(s) {
                                keep_indices.insert(*idx);
                            }
                        }
                    }
                    strategy_parts.push(format!("str:{}->{}", values.len(), crushed.len()));
                }
                "number" => {
                    // Python: just adaptive sampling + outlier detection
                    // (no summary prefix). Keeps first/last by index
                    // and items >variance_threshold σ from mean.
                    let item_strings: Vec<String> = values.iter().map(|v| v.to_string()).collect();
                    let item_refs: Vec<&str> = item_strings.iter().map(|s| s.as_str()).collect();
                    let (_kt, kf, kl, _) = compute_k_split(&item_refs, &self.config, bias);

                    let kf = kf.min(values.len());
                    let kl = kl.min(values.len().saturating_sub(kf));
                    let first_idx: Vec<usize> = indices.iter().take(kf).copied().collect();
                    let last_idx: Vec<usize> =
                        indices.iter().rev().take(kl).copied().collect::<Vec<_>>();
                    keep_indices.extend(&first_idx);
                    keep_indices.extend(&last_idx);

                    // Outliers via finite-only stats.
                    let finite: Vec<f64> = values
                        .iter()
                        .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
                        .collect();
                    if finite.len() > 1 {
                        if let Some(mean_v) = super::stats_math::mean(&finite) {
                            if let Some(std_v) = super::stats_math::sample_stdev(&finite) {
                                if std_v > 0.0 {
                                    let threshold = self.config.variance_threshold * std_v;
                                    for (i, val) in values.iter().enumerate() {
                                        if let Some(num) = val.as_f64().filter(|f| f.is_finite()) {
                                            if (num - mean_v).abs() > threshold {
                                                keep_indices.insert(indices[i]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    strategy_parts.push(format!("num:{}", values.len()));
                }
                _ => {
                    // list / bool / none / other → keep all items.
                    keep_indices.extend(&indices);
                }
            }
        }

        // Reassemble in original order.
        let result: Vec<Value> = keep_indices.iter().map(|&i| items[i].clone()).collect();
        let strategy = format!(
            "mixed:adaptive({}->{},{})",
            n,
            result.len(),
            strategy_parts.join(",")
        );
        (result, strategy)
    }
}

// ---------- helpers ----------

/// Group key that mirrors Python's `_crush_mixed_array` switch on
/// `isinstance`. Note the bool-before-number ordering: in Python, bool
/// is a subclass of int, but JSON treats them as distinct types, so we
/// don't have the Python ordering hazard.
fn group_key(item: &Value) -> &'static str {
    match item {
        Value::Object(_) => "dict",
        Value::String(_) => "str",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::Array(_) => "list",
        Value::Null => "none",
    }
}

/// Group buckets keyed by the type-string. Preserves first-occurrence
/// order across keys so dict/str/number/list/none/bool always come out
/// in the same order — matters because `keep_indices` is built
/// incrementally and Python iterates `groups.items()` (insertion order
/// in 3.7+).
#[derive(Default)]
struct GroupBuckets {
    entries: Vec<(&'static str, Vec<usize>, Vec<Value>)>,
    index_of: std::collections::HashMap<&'static str, usize>,
}

impl GroupBuckets {
    fn push(&mut self, key: &'static str, idx: usize, value: Value) {
        match self.index_of.get(key).copied() {
            Some(i) => {
                self.entries[i].1.push(idx);
                self.entries[i].2.push(value);
            }
            None => {
                self.index_of.insert(key, self.entries.len());
                self.entries.push((key, vec![idx], vec![value]));
            }
        }
    }
}

impl IntoIterator for GroupBuckets {
    type Item = (&'static str, Vec<usize>, Vec<Value>);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// Serialize a `Value` for membership comparison. Mirrors Python's
/// `json.dumps(c, sort_keys=True, default=str)` used by
/// `_crush_mixed_array` to match crushed dict items back to their
/// original indices. The `default=str` fallback only matters for
/// non-JSON-serializable Python values; in serde_json land everything
/// is already JSON-native, so plain canonical JSON suffices.
fn canonical_json_for_match(value: &Value) -> String {
    crate::transforms::anchor_selector::python_json_dumps_sort_keys(value)
}

/// Maps a `Compaction` to a stable kind tag exposed via `CrushArrayResult`.
fn compaction_kind_str(c: &Compaction) -> &'static str {
    match c {
        Compaction::Table { .. } => "table",
        Compaction::Buckets { .. } => "buckets",
        Compaction::OpaqueRef { .. } => "ccr",
        Compaction::Untouched(_) => "untouched",
    }
}

/// If compaction won, override the lossy strategy string with
/// `"compaction:<formatter_name>:<kind>"`. Else return `default`.
fn compaction_strategy_or(
    compaction_result: &Option<(String, &'static str)>,
    default: &str,
) -> String {
    match compaction_result {
        Some((_, kind)) => format!("compaction:{kind}"),
        None => default.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn crusher() -> SmartCrusher {
        SmartCrusher::new(SmartCrusherConfig::default())
    }

    // ---------- execute_plan ----------

    #[test]
    fn execute_plan_empty_indices_returns_empty() {
        let c = crusher();
        let plan = CompressionPlan::default();
        let items: Vec<Value> = (0..5).map(|i| json!({"id": i})).collect();
        let result = c.execute_plan(&plan, &items);
        assert!(result.is_empty());
    }

    #[test]
    fn execute_plan_returns_items_in_sorted_index_order() {
        let c = crusher();
        let items: Vec<Value> = (0..10).map(|i| json!({"id": i})).collect();
        let plan = CompressionPlan {
            keep_indices: vec![5, 2, 8, 0],
            ..CompressionPlan::default()
        };
        let result = c.execute_plan(&plan, &items);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0]["id"], 0);
        assert_eq!(result[1]["id"], 2);
        assert_eq!(result[2]["id"], 5);
        assert_eq!(result[3]["id"], 8);
    }

    #[test]
    fn execute_plan_skips_out_of_bounds() {
        let c = crusher();
        let items: Vec<Value> = (0..3).map(|i| json!({"id": i})).collect();
        let plan = CompressionPlan {
            keep_indices: vec![0, 5, 2],
            ..CompressionPlan::default()
        };
        let result = c.execute_plan(&plan, &items);
        assert_eq!(result.len(), 2);
    }

    // ---------- crush_array ----------

    #[test]
    fn crush_array_passthrough_when_below_adaptive_k() {
        let c = crusher();
        let items: Vec<Value> = (0..3).map(|i| json!({"id": i})).collect();
        let result = c.crush_array(&items, "", 1.0);
        assert_eq!(result.items.len(), 3);
        assert_eq!(result.strategy_info, "none:adaptive_at_limit");
        assert!(result.ccr_hash.is_none());
    }

    #[test]
    fn crush_array_skip_path_returns_original_items() {
        // 30 unique dict items with ID-like fields → analyzer should
        // detect "unique_entities_no_signal" and SKIP.
        let c = crusher();
        let items: Vec<Value> = (0..30)
            .map(|i| json!({"id": i, "name": format!("user_{}", i)}))
            .collect();
        let result = c.crush_array(&items, "", 1.0);
        // skip path returns the original items unchanged.
        assert_eq!(result.items.len(), 30);
        assert!(
            result.strategy_info.starts_with("skip:"),
            "expected skip:..., got {}",
            result.strategy_info
        );
    }

    #[test]
    fn crush_array_low_uniqueness_compresses() {
        // 30 items with status=ok across all → low_uniqueness path
        // (crushable, smart_sample strategy).
        let c = crusher();
        let items: Vec<Value> = (0..30).map(|_| json!({"status": "ok"})).collect();
        let result = c.crush_array(&items, "", 1.0);
        assert!(result.items.len() <= 30, "should not exceed original count");
    }

    #[test]
    fn crush_array_keeps_error_items() {
        let c = crusher();
        let mut items: Vec<Value> = (0..30).map(|i| json!({"id": i, "status": "ok"})).collect();
        items.push(json!({"id": 30, "status": "error", "msg": "FATAL"}));
        let result = c.crush_array(&items, "", 1.0);
        // Whatever path is taken, the error item should survive.
        assert!(
            result
                .items
                .iter()
                .any(|item| { item.get("status").and_then(|v| v.as_str()) == Some("error") }),
            "error item must survive crush_array"
        );
    }

    // ---------- crush_mixed_array ----------

    #[test]
    fn crush_mixed_passthrough_at_threshold() {
        let c = crusher();
        let items: Vec<Value> = vec![
            json!(1),
            json!("two"),
            json!({"k": "v"}),
            json!([1, 2]),
            json!(null),
            json!(true),
            json!(3),
            json!("four"),
        ];
        let (result, strat) = c.crush_mixed_array(&items, "", 1.0);
        assert_eq!(result.len(), 8);
        assert_eq!(strat, "mixed:passthrough");
    }

    #[test]
    fn crush_mixed_groups_and_compresses_dicts() {
        let c = crusher();
        // 25 dicts (large group → gets crushed) + 5 strings (small group → all kept).
        let mut items: Vec<Value> = (0..25).map(|i| json!({"id": i, "status": "ok"})).collect();
        for i in 0..5 {
            items.push(json!(format!("string_{}", i)));
        }
        let (result, strat) = c.crush_mixed_array(&items, "", 1.0);
        assert!(strat.starts_with("mixed:adaptive("));
        // The 5 strings (small group) all survive.
        let str_count = result.iter().filter(|v| v.is_string()).count();
        assert_eq!(str_count, 5);
    }

    #[test]
    fn crush_mixed_keeps_lists_and_nulls_unchanged() {
        let c = crusher();
        let mut items: Vec<Value> = vec![json!([1, 2]); 6];
        items.extend(vec![json!(null); 6]);
        items.extend(vec![json!({"k": 1}); 10]);
        let (result, _strat) = c.crush_mixed_array(&items, "", 1.0);
        // Lists and nulls (not dict/str/number) → fall through to "keep all".
        let list_count = result.iter().filter(|v| v.is_array()).count();
        let null_count = result.iter().filter(|v| v.is_null()).count();
        assert_eq!(list_count, 6);
        assert_eq!(null_count, 6);
    }

    #[test]
    fn crusher_construction_default() {
        let c = SmartCrusher::new(SmartCrusherConfig::default());
        assert_eq!(c.config.max_items_after_crush, 15);
    }

    // ---------- top-level crush ----------

    #[test]
    fn crush_non_json_passes_through_unchanged() {
        let c = crusher();
        let result = c.crush("not json at all", "", 1.0);
        assert!(!result.was_modified);
        assert_eq!(result.compressed, "not json at all");
        assert_eq!(result.strategy, "passthrough");
    }

    #[test]
    fn crush_scalar_json_passes_through() {
        let c = crusher();
        let result = c.crush("42", "", 1.0);
        // A scalar is not crushable; should round-trip unchanged.
        assert_eq!(result.compressed, "42");
        assert!(!result.was_modified);
    }

    #[test]
    fn crush_small_array_passes_through() {
        let c = crusher();
        // Compact-form input matches the compact serializer output, so
        // the array is not "modified" even though it round-trips
        // through parse → serialize. (The spaced form `[1, 2, 3]`
        // would mark `was_modified=true` because the compact
        // serializer rewrites it to `[1,2,3]`.)
        let result = c.crush(r#"[1,2,3]"#, "", 1.0);
        // Below min_items_to_analyze=5 → no crushing of the structure.
        assert!(!result.was_modified);
        assert_eq!(result.compressed, "[1,2,3]");
    }

    #[test]
    fn crush_dict_array_crushes_when_low_uniqueness() {
        let c = crusher();
        // 30 dicts all with status=ok → low uniqueness path → crushable.
        let mut input = String::from("[");
        for i in 0..30 {
            if i > 0 {
                input.push(',');
            }
            input.push_str(r#"{"status":"ok"}"#);
        }
        input.push(']');
        let result = c.crush(&input, "", 1.0);
        assert!(
            result.was_modified,
            "30 identical dicts should compress (low_uniqueness_safe_to_sample)"
        );
        assert_ne!(result.strategy, "passthrough");
    }

    #[test]
    fn crush_serializes_with_python_safe_format() {
        let c = crusher();
        // SmartCrusher uses Python's `safe_json_dumps`: compact
        // separators `(",", ":")` + `ensure_ascii=False`, preserving
        // object-key insertion order. A spaced input round-trips to
        // the compact form.
        let input = r#"{"a": 1, "b": 2, "c": 3}"#;
        let result = c.crush(input, "", 1.0);
        assert_eq!(
            result.compressed, r#"{"a":1,"b":2,"c":3}"#,
            "safe_json_dumps emits compact `,` / `:` separators"
        );
    }

    #[test]
    fn crush_recurses_into_nested_arrays() {
        let c = crusher();
        // Top-level dict with a nested array of 30 identical items.
        // The inner array should compress (low_uniqueness path).
        let mut inner = String::from("[");
        for i in 0..30 {
            if i > 0 {
                inner.push(',');
            }
            inner.push_str(r#"{"status":"ok"}"#);
        }
        inner.push(']');
        let input = format!(r#"{{"data": {}}}"#, inner);
        let result = c.crush(&input, "", 1.0);
        assert!(
            result.was_modified,
            "nested compressible array must be crushed even inside a wrapper object"
        );
    }

    #[test]
    fn crusher_with_custom_scorer() {
        use crate::relevance::BM25Scorer;
        let c = SmartCrusher::with_scorer(
            SmartCrusherConfig::default(),
            Box::new(BM25Scorer::default()),
        );
        // Sanity: crushing still works with a swapped scorer.
        let items: Vec<Value> = (0..30).map(|_| json!({"status": "ok"})).collect();
        let result = c.crush_array(&items, "anything", 1.0);
        assert!(result.items.len() <= 30);
    }

    // ---------- Stage 3c.2 PR2: lossless-first compaction wiring ----------

    #[test]
    fn no_compaction_stage_yields_none_compacted_field() {
        // Default SmartCrusher::new() sets compaction = None. Existing
        // parity preserved: compacted/compaction_kind both None.
        let c = crusher();
        let items: Vec<Value> = (0..30).map(|_| json!({"status": "ok"})).collect();
        let result = c.crush_array(&items, "", 1.0);
        assert!(result.compacted.is_none());
        assert!(result.compaction_kind.is_none());
    }

    #[test]
    fn compaction_stage_runs_and_populates_compacted() {
        let c = SmartCrusherBuilder::new(SmartCrusherConfig::default())
            .with_default_oss_setup()
            .with_default_compaction()
            .build();
        let items: Vec<Value> = (0..50)
            .map(|i| json!({"id": i, "name": format!("u_{i}"), "status": "ok"}))
            .collect();
        let result = c.crush_array(&items, "", 1.0);
        let compacted = result.compacted.expect("compacted should be set");
        assert!(compacted.starts_with("[50]{"), "got: {compacted}");
        assert_eq!(result.compaction_kind, Some("table"));
        assert!(result.strategy_info.starts_with("compaction:table"));
    }

    #[test]
    fn compaction_does_not_alter_lossy_items_field() {
        // The compacted bytes are emitted alongside the existing lossy
        // result. items field still holds the lossy-path output so
        // downstream consumers can choose either form.
        let c = SmartCrusherBuilder::new(SmartCrusherConfig::default())
            .with_default_oss_setup()
            .with_default_compaction()
            .build();
        let items: Vec<Value> = (0..30).map(|_| json!({"status": "ok"})).collect();
        let with_compaction = c.crush_array(&items, "", 1.0);

        let baseline = crusher().crush_array(&items, "", 1.0);
        assert_eq!(with_compaction.items.len(), baseline.items.len());
    }

    #[test]
    fn compaction_skips_non_object_array() {
        // Compactor returns Untouched for non-object arrays → no
        // compacted field populated, no kind tag.
        let c = SmartCrusherBuilder::new(SmartCrusherConfig::default())
            .with_default_oss_setup()
            .with_default_compaction()
            .build();
        let items: Vec<Value> = (0..30).map(|i| json!(i)).collect();
        let result = c.crush_array(&items, "", 1.0);
        assert!(result.compacted.is_none());
        assert!(result.compaction_kind.is_none());
    }
}
