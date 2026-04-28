//! `DocumentCompactor` — recursive walker that finds compactable spots
//! anywhere in a JSON document and replaces them in place.
//!
//! # The whole algorithm in one rule
//!
//! ```text
//! match value {
//!     Object(m) => recurse into each field's value
//!     Array(xs) => recurse into each item, then try TabularCompactor on the array
//!     String(s) => parse-as-JSON-and-recurse / CCR-substitute / leave
//!     scalar    => unchanged
//! }
//! ```
//!
//! # Output shape
//!
//! Same JSON shape as input. Compacted spots become **strings** holding
//! the rendered bytes. The wrapping object/array structure is preserved
//! exactly — only bulky leaves get replaced.
//!
//! Example:
//!
//! ```text
//! input:  {"user": "alice", "events": [{...}, {...}, ...]}
//! output: {"user": "alice", "events": "[50]{id:int,action:string}\n1,click\n..."}
//! ```
//!
//! Nested cases cascade naturally — we recurse into the array's items
//! BEFORE running TabularCompactor on the array, so inner sub-tables
//! become strings first and the outer table sees them as cells.

use serde_json::{Map, Value};

use super::classifier::{classify_cell, CellClass};
use super::compactor::{compact, CompactConfig};
use super::formatter::{CsvSchemaFormatter, Formatter};
use super::ir::OpaqueKind;

use sha2::{Digest, Sha256};

/// Walks any JSON value and applies lossless compaction in place.
///
/// Reuses the PR2 primitives:
/// - [`compact`](super::compactor::compact) — array → IR
/// - [`Formatter`] — IR → bytes
/// - [`classify_cell`] + opaque-blob detection
///
/// The walker itself owns no compaction logic; it just decides
/// **where** to apply each primitive in the tree.
///
/// # Budget enforcement
///
/// [`compact_with_budget`] adds an optional byte cap. If the lossless
/// walk produces output larger than the budget, the walker escalates
/// the largest compaction sites to lossy + CCR (drop rows, hash full
/// original, emit retrieval marker) until the document fits. The
/// runtime caches the dropped originals via the returned
/// [`WalkResult::ccr_payloads`].
///
/// [`compact_with_budget`]: DocumentCompactor::compact_with_budget
pub struct DocumentCompactor {
    pub config: CompactConfig,
    pub formatter: Box<dyn Formatter>,
    /// Maximum total kept items per site under budget escalation.
    /// Sites that escalate keep first/last sample of this size and
    /// stash the rest in CCR. Default: 15 (matches `max_items_after_crush`).
    pub lossy_max_items: usize,
}

impl Default for DocumentCompactor {
    fn default() -> Self {
        Self {
            config: CompactConfig::default(),
            formatter: Box::new(CsvSchemaFormatter::new()),
            lossy_max_items: 15,
        }
    }
}

impl DocumentCompactor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_formatter(mut self, formatter: Box<dyn Formatter>) -> Self {
        self.formatter = formatter;
        self
    }

    pub fn with_config(mut self, config: CompactConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_lossy_max_items(mut self, max: usize) -> Self {
        self.lossy_max_items = max;
        self
    }

    /// Walk and compact. Returns a JSON value with the same shape but
    /// with compactable spots replaced by rendered strings.
    ///
    /// **No budget enforcement.** If the lossless rendering is too big
    /// for the model's context, use [`compact_with_budget`] instead.
    ///
    /// [`compact_with_budget`]: DocumentCompactor::compact_with_budget
    pub fn compact(&self, doc: Value) -> Value {
        walk(doc, self)
    }

    /// Walk + compact with a byte budget. Lossless first; if the
    /// rendered document exceeds `budget_bytes`, escalate the biggest
    /// compaction sites to lossy + CCR until the document fits.
    ///
    /// Returns a [`WalkResult`] containing:
    /// - `document` — the JSON tree with placeholders replaced by
    ///   rendered strings (lossless or lossy, decided per site)
    /// - `ccr_payloads` — for each escalated site, the **full original
    ///   items** keyed by hash; the runtime is expected to cache these
    ///   so a retrieval tool can serve them back to the LLM
    /// - `byte_size` — the final document's serialized size
    ///
    /// **No data is lost.** Lossy here means "compressed view inline
    /// plus full payload retrievable via CCR cache." Same semantics
    /// as PR4's `crush_array` lossy path, applied per site.
    pub fn compact_with_budget(&self, doc: Value, budget_bytes: usize) -> WalkResult {
        compact_with_budget(doc, budget_bytes, self)
    }
}

fn walk(v: Value, ctx: &DocumentCompactor) -> Value {
    match v {
        Value::Object(map) => walk_object(map, ctx),
        Value::Array(items) => walk_array(items, ctx),
        Value::String(s) => walk_string(s, ctx),
        scalar => scalar,
    }
}

fn walk_object(map: Map<String, Value>, ctx: &DocumentCompactor) -> Value {
    Value::Object(map.into_iter().map(|(k, v)| (k, walk(v, ctx))).collect())
}

fn walk_array(items: Vec<Value>, ctx: &DocumentCompactor) -> Value {
    // Recurse into items FIRST so inner sub-tables / opaque markers are
    // already in their compacted form when the outer compact runs. This
    // is what makes deep nesting cascade — a stringified-JSON cell
    // becomes a rendered string before the outer table sees it.
    let inner: Vec<Value> = items.into_iter().map(|i| walk(i, ctx)).collect();

    // Then try the array as a whole.
    let c = compact(&inner, &ctx.config);
    if c.was_compacted() {
        Value::String(ctx.formatter.format(&c))
    } else {
        Value::Array(inner)
    }
}

fn walk_string(s: String, ctx: &DocumentCompactor) -> Value {
    // Stringified-JSON: parse, recurse, replace.
    if let Some(parsed) = try_parse_json_container(&s) {
        let recursed = walk(parsed, ctx);
        return match recursed {
            // Sub-table won — already a rendered string.
            Value::String(rendered) => Value::String(rendered),
            // Sub-recursion didn't compact anything; emit compact JSON.
            other => Value::String(serde_json::to_string(&other).unwrap_or(s)),
        };
    }

    // Long opaque blob: substitute with CCR marker.
    if let CellClass::Opaque(kind) = classify_cell(&Value::String(s.clone()), &ctx.config.classify)
    {
        return Value::String(format_ccr_marker(s.as_bytes(), &kind));
    }

    Value::String(s)
}

/// Parse a string as JSON IF it looks like a container (starts with `{`
/// or `[`) AND parses cleanly to Object/Array. Returns None otherwise —
/// we don't recurse on bare scalars even if they parse.
fn try_parse_json_container(s: &str) -> Option<Value> {
    let trimmed = s.trim_start();
    if !matches!(trimmed.chars().next(), Some('{') | Some('[')) {
        return None;
    }
    serde_json::from_str::<Value>(s)
        .ok()
        .filter(|v| matches!(v, Value::Object(_) | Value::Array(_)))
}

fn format_ccr_marker(bytes: &[u8], kind: &OpaqueKind) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let hash: String = h
        .finalize()
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect();
    let kind_str = match kind {
        OpaqueKind::Base64Blob => "base64",
        OpaqueKind::LongString => "string",
        OpaqueKind::HtmlChunk => "html",
        OpaqueKind::Other(s) => s.as_str(),
    };
    format!("<<ccr:{},{},{}>>", hash, kind_str, humanize(bytes.len()))
}

fn humanize(n: usize) -> String {
    if n < 1024 {
        return format!("{n}B");
    }
    let kb = n as f64 / 1024.0;
    if kb < 1024.0 {
        return format!("{kb:.1}KB");
    }
    format!("{:.1}MB", kb / 1024.0)
}

/// Convenience: walk and compact with default config + CSV-schema
/// formatter. Equivalent to `DocumentCompactor::new().compact(doc)`.
pub fn compact_document(doc: Value) -> Value {
    DocumentCompactor::new().compact(doc)
}

// ─── Budget-aware walk (PR3b) ────────────────────────────────────────────────

/// Output of [`DocumentCompactor::compact_with_budget`].
#[derive(Debug, Clone)]
pub struct WalkResult {
    /// The transformed document — placeholders substituted with
    /// chosen rendering (lossless or lossy) per compaction site.
    pub document: Value,
    /// Per-site cache payloads for sites that were escalated to lossy.
    /// The runtime caches each entry's `original_items` keyed by
    /// `hash` so a retrieval tool can serve them back. Empty when
    /// the lossless walk fit under budget.
    pub ccr_payloads: Vec<CcrPayload>,
    /// Final byte size of `document` after substitution.
    pub byte_size: usize,
    /// The budget that was applied.
    pub budget_bytes: usize,
}

/// One escalated site's cache binding.
#[derive(Debug, Clone)]
pub struct CcrPayload {
    /// 12-char SHA-256 hex prefix of the canonical-JSON-serialized
    /// original items array. Same hash function as PR4's `crush_array`
    /// — runtime caching is interoperable.
    pub hash: String,
    /// Full original items the lossy form dropped from the inline
    /// rendering. Caller stashes these for tool-call retrieval.
    pub original_items: Vec<Value>,
    /// Items kept inline by the lossy form (subset of `original_items`).
    pub kept_count: usize,
    /// Items moved to CCR (retrievable via `hash`).
    pub dropped_count: usize,
}

/// Per-site bookkeeping during the budget-aware walk.
struct Site {
    original_items: Vec<Value>,
    lossless: String,
    lossy: String,
    ccr_hash: String,
    kept_count: usize,
    dropped_count: usize,
}

const PLACEHOLDER_PREFIX: &str = "@@__hr_site:";
const PLACEHOLDER_SUFFIX: &str = "@@";

fn placeholder(idx: usize) -> Value {
    Value::String(format!("{PLACEHOLDER_PREFIX}{idx}{PLACEHOLDER_SUFFIX}"))
}

fn try_extract_placeholder(s: &str) -> Option<usize> {
    s.strip_prefix(PLACEHOLDER_PREFIX)
        .and_then(|rest| rest.strip_suffix(PLACEHOLDER_SUFFIX))
        .and_then(|num| num.parse().ok())
}

fn compact_with_budget(doc: Value, budget: usize, ctx: &DocumentCompactor) -> WalkResult {
    // Phase 1: walk + collect sites, leaving placeholders behind.
    let mut sites: Vec<Site> = Vec::new();
    let placeholder_doc = collect_sites(doc, ctx, &mut sites);

    // Phase 2: decide lossless vs lossy per site, greedy by lossless
    // size descending, until under budget.
    let escalated = decide_escalations(&sites, &placeholder_doc, budget);

    // Phase 3: substitute renderings in place.
    let final_doc = substitute(placeholder_doc, &sites, &escalated);

    // Build CCR payloads for escalated sites.
    let ccr_payloads: Vec<CcrPayload> = escalated
        .iter()
        .map(|&idx| {
            let s = &sites[idx];
            CcrPayload {
                hash: s.ccr_hash.clone(),
                original_items: s.original_items.clone(),
                kept_count: s.kept_count,
                dropped_count: s.dropped_count,
            }
        })
        .collect();

    let byte_size = serde_json::to_string(&final_doc)
        .map(|s| s.len())
        .unwrap_or(0);

    WalkResult {
        document: final_doc,
        ccr_payloads,
        byte_size,
        budget_bytes: budget,
    }
}

/// Walks like `walk` but instead of substituting compactions inline
/// it pushes a [`Site`] entry and leaves a placeholder string. String
/// branches (stringified-JSON, opaque blobs) substitute inline since
/// they don't have a meaningful lossy fallback at PR3b scope — only
/// arrays escalate.
fn collect_sites(v: Value, ctx: &DocumentCompactor, sites: &mut Vec<Site>) -> Value {
    match v {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, collect_sites(v, ctx, sites)))
                .collect(),
        ),
        Value::Array(items) => {
            let inner: Vec<Value> = items
                .into_iter()
                .map(|i| collect_sites(i, ctx, sites))
                .collect();
            let c = compact(&inner, &ctx.config);
            if c.was_compacted() {
                let lossless = ctx.formatter.format(&c);
                let (lossy, kept, dropped) = compute_lossy_form(&inner, ctx);
                let ccr_hash = hash_array(&inner);
                let idx = sites.len();
                sites.push(Site {
                    original_items: inner,
                    lossless,
                    lossy,
                    ccr_hash,
                    kept_count: kept,
                    dropped_count: dropped,
                });
                placeholder(idx)
            } else {
                Value::Array(inner)
            }
        }
        // Strings (stringified-JSON / opaque) substitute inline like
        // the lossless walker — they don't participate in escalation.
        Value::String(s) => walk_string(s, ctx),
        scalar => scalar,
    }
}

/// Decide which sites to escalate to lossy. Greedy: sort by lossless
/// rendering size descending; escalate one at a time until the doc
/// fits the budget. Re-serializes after each escalation; for typical
/// payloads (< 10 sites per document) this is negligible.
fn decide_escalations(
    sites: &[Site],
    placeholder_doc: &Value,
    budget: usize,
) -> std::collections::BTreeSet<usize> {
    let mut escalated: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

    // Cheap trial render to check current size.
    let measure = |esc: &std::collections::BTreeSet<usize>| -> usize {
        let v = substitute(placeholder_doc.clone(), sites, esc);
        serde_json::to_string(&v).map(|s| s.len()).unwrap_or(0)
    };

    if measure(&escalated) <= budget {
        return escalated;
    }

    // Indices sorted by lossless size descending.
    let mut indices: Vec<usize> = (0..sites.len()).collect();
    indices.sort_by_key(|&i| std::cmp::Reverse(sites[i].lossless.len()));

    for idx in indices {
        escalated.insert(idx);
        if measure(&escalated) <= budget {
            break;
        }
    }
    escalated
}

fn substitute(v: Value, sites: &[Site], escalated: &std::collections::BTreeSet<usize>) -> Value {
    match v {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, substitute(v, sites, escalated)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|i| substitute(i, sites, escalated))
                .collect(),
        ),
        Value::String(s) => {
            if let Some(idx) = try_extract_placeholder(&s) {
                let site = &sites[idx];
                let rendered = if escalated.contains(&idx) {
                    site.lossy.clone()
                } else {
                    site.lossless.clone()
                };
                Value::String(rendered)
            } else {
                Value::String(s)
            }
        }
        scalar => scalar,
    }
}

/// Build the lossy form: keep first/last sample of items, compact the
/// kept subset, append a CCR retrieval marker. Returns
/// `(rendered, kept_count, dropped_count)`.
fn compute_lossy_form(items: &[Value], ctx: &DocumentCompactor) -> (String, usize, usize) {
    let n = items.len();
    let max_keep = ctx.lossy_max_items;
    let kept: Vec<Value> = if n <= max_keep {
        items.to_vec()
    } else {
        // First half + last half. If max_keep is odd, give the extra to first.
        let last = max_keep / 2;
        let first = max_keep - last;
        let mut v = items[..first].to_vec();
        v.extend_from_slice(&items[n - last..]);
        v
    };
    let kept_count = kept.len();
    let dropped_count = n - kept_count;

    // Compact the kept subset. If even the subset can't be tabulated
    // (very unusual for budget cases), fall back to a JSON dump.
    let c = compact(&kept, &ctx.config);
    let mut rendered = if c.was_compacted() {
        ctx.formatter.format(&c)
    } else {
        serde_json::to_string(&Value::Array(kept)).unwrap_or_default()
    };

    if dropped_count > 0 {
        let h = hash_array(items);
        rendered.push('\n');
        rendered.push_str(&format!("<<ccr:{h} {dropped_count}_rows_offloaded>>"));
    }

    (rendered, kept_count, dropped_count)
}

fn hash_array(items: &[Value]) -> String {
    let canonical = serde_json::to_string(&Value::Array(items.to_vec())).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(canonical.as_bytes());
    h.finalize()
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dc() -> DocumentCompactor {
        DocumentCompactor::new()
    }

    #[test]
    fn top_level_array_of_objects_is_compacted() {
        let doc = json!([
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
            {"id": 3, "name": "carol"},
        ]);
        let out = dc().compact(doc);
        match out {
            Value::String(s) => {
                assert!(s.starts_with("[3]{"), "got: {s}");
                assert!(s.contains("name:string"));
            }
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn nested_array_in_object_field_is_compacted_in_place() {
        let doc = json!({
            "user": "alice",
            "events": [
                {"id": 1, "action": "click"},
                {"id": 2, "action": "hover"},
                {"id": 3, "action": "submit"},
            ],
        });
        let out = dc().compact(doc);
        let obj = out.as_object().expect("object preserved");
        assert_eq!(obj.get("user").and_then(|v| v.as_str()), Some("alice"));
        let events = obj.get("events").and_then(|v| v.as_str()).expect("string");
        assert!(events.starts_with("[3]{"), "got: {events}");
    }

    #[test]
    fn deeply_nested_arrays_compact_at_every_level() {
        let doc = json!({
            "outer": {
                "middle": {
                    "rows": [
                        {"a": 1, "b": "x"},
                        {"a": 2, "b": "y"},
                    ],
                },
            },
        });
        let out = dc().compact(doc);
        let inner = out
            .pointer("/outer/middle/rows")
            .and_then(|v| v.as_str())
            .expect("rows compacted to string");
        assert!(inner.starts_with("[2]{"), "got: {inner}");
    }

    #[test]
    fn stringified_json_in_field_is_parsed_and_compacted() {
        let inner = r#"[{"x":1},{"x":2},{"x":3}]"#;
        let doc = json!({
            "id": "abc",
            "payload": inner,
        });
        let out = dc().compact(doc);
        let payload = out
            .pointer("/payload")
            .and_then(|v| v.as_str())
            .expect("payload compacted");
        assert!(payload.starts_with("[3]{"), "got: {payload}");
    }

    #[test]
    fn long_opaque_string_at_top_level_becomes_ccr_marker() {
        let big = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".repeat(8);
        let out = dc().compact(Value::String(big));
        match out {
            Value::String(s) => assert!(
                s.starts_with("<<ccr:") && s.contains(",base64,"),
                "got: {s}"
            ),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn long_opaque_string_inside_object_field_becomes_ccr_marker() {
        let big = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".repeat(8);
        let doc = json!({"id": 1, "blob": big});
        let out = dc().compact(doc);
        let blob = out.pointer("/blob").and_then(|v| v.as_str()).unwrap();
        assert!(blob.starts_with("<<ccr:"), "got: {blob}");
    }

    #[test]
    fn pure_scalar_object_unchanged() {
        let doc = json!({"a": 1, "b": "short", "c": true, "d": null});
        let out = dc().compact(doc.clone());
        assert_eq!(out, doc);
    }

    #[test]
    fn mixed_doc_only_compactable_parts_change() {
        let doc = json!({
            "user_id": 42,
            "tag": "active",
            "events": [
                {"id": 1, "kind": "x"},
                {"id": 2, "kind": "y"},
            ],
            "config": {"region": "us", "tier": "gold"},
        });
        let out = dc().compact(doc);
        // user_id and tag preserved as scalars.
        assert_eq!(out.pointer("/user_id"), Some(&json!(42)));
        assert_eq!(out.pointer("/tag"), Some(&json!("active")));
        // config preserved as object (not an array, can't tabulate).
        assert!(out
            .pointer("/config")
            .map(|v| v.is_object())
            .unwrap_or(false));
        // events compacted to a string.
        assert!(out
            .pointer("/events")
            .and_then(|v| v.as_str())
            .unwrap()
            .starts_with("[2]{"));
    }

    #[test]
    fn cascading_recursion_outer_table_sees_inner_compacted_string() {
        // Each row has a stringified-JSON `payload`. After the walker
        // recurses into items, each payload is a rendered sub-table
        // string. The outer compact then builds a 3-row × 2-col table
        // where the payload column holds the inner renderings.
        let doc = json!([
            {"id": 1, "payload": r#"[{"x":1},{"x":2},{"x":3}]"#},
            {"id": 2, "payload": r#"[{"x":4},{"x":5}]"#},
        ]);
        let out = dc().compact(doc);
        match out {
            Value::String(s) => {
                assert!(s.starts_with("[2]{"), "outer table: {s}");
                // The inner-rendered sub-tables show up CSV-quoted in
                // the payload column.
                assert!(s.contains("[3]{") || s.contains("\"[3]{"));
            }
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn array_of_scalars_left_alone() {
        // Compactor declines non-object arrays → walker returns the
        // recursed array unchanged.
        let doc = json!([1, 2, 3, "four", 5.0]);
        let out = dc().compact(doc.clone());
        assert_eq!(out, doc);
    }

    #[test]
    fn empty_object_unchanged() {
        let doc = json!({});
        assert_eq!(dc().compact(doc.clone()), doc);
    }

    #[test]
    fn empty_array_unchanged() {
        let doc = json!([]);
        assert_eq!(dc().compact(doc.clone()), doc);
    }

    #[test]
    fn malformed_stringified_json_left_alone() {
        let doc = json!({"payload": "{not valid json"});
        let out = dc().compact(doc.clone());
        assert_eq!(out, doc);
    }

    // ── Budget enforcement (PR3b) ──

    fn build_array(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| {
                json!({
                    "id": i,
                    "name": format!("user_{i}"),
                    "status": if i % 5 == 0 { "warn" } else { "ok" },
                })
            })
            .collect()
    }

    #[test]
    fn budget_satisfied_no_escalation() {
        // Lossless rendering fits the budget → no sites escalated, no
        // CCR payloads.
        let doc = Value::Array(build_array(20));
        let r = dc().compact_with_budget(doc, 100_000);
        assert!(r.ccr_payloads.is_empty());
        assert!(r.byte_size <= 100_000);
        // Document is the lossless rendering.
        match &r.document {
            Value::String(s) => assert!(s.starts_with("[20]{"), "got: {s}"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn budget_triggers_escalation() {
        // Tiny budget → must escalate. The 200-row array can't fit
        // in 100 bytes lossless; lossy form drops most rows to CCR.
        let doc = Value::Array(build_array(200));
        let r = dc().compact_with_budget(doc, 100);
        assert_eq!(r.ccr_payloads.len(), 1, "expected one escalated site");
        let payload = &r.ccr_payloads[0];
        assert_eq!(payload.original_items.len(), 200);
        assert!(payload.dropped_count > 0);
        assert!(payload.kept_count > 0);
        assert!(payload.kept_count + payload.dropped_count == 200);
        // Lossy rendering carries the CCR marker.
        match &r.document {
            Value::String(s) => {
                assert!(s.contains(&format!("<<ccr:{}", payload.hash)), "got: {s}");
                assert!(s.contains("rows_offloaded"));
            }
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn budget_escalates_biggest_site_first() {
        // Two compactable arrays in one doc — small (5 rows) and big
        // (200 rows). With a budget that fits one but not both, the
        // big one escalates while the small one stays lossless.
        let doc = json!({
            "small": build_array(5),
            "big": build_array(200),
        });
        let r = dc().compact_with_budget(doc, 600);
        // Exactly one site escalated.
        assert_eq!(r.ccr_payloads.len(), 1);
        assert_eq!(r.ccr_payloads[0].original_items.len(), 200);
        // Small still lossless (no CCR marker in its rendering).
        let small = r
            .document
            .pointer("/small")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(
            !small.contains("<<ccr:"),
            "small should be lossless: {small}"
        );
        // Big has the marker.
        let big = r.document.pointer("/big").and_then(|v| v.as_str()).unwrap();
        assert!(big.contains("<<ccr:"), "big should be lossy: {big}");
    }

    #[test]
    fn ccr_payload_hash_matches_marker() {
        // The hash in the rendered marker must match the hash in the
        // returned CcrPayload — the runtime keys cache by it.
        let doc = Value::Array(build_array(100));
        let r = dc().compact_with_budget(doc, 200);
        assert_eq!(r.ccr_payloads.len(), 1);
        let h = &r.ccr_payloads[0].hash;
        match &r.document {
            Value::String(s) => assert!(s.contains(&format!("<<ccr:{h}"))),
            _ => panic!(),
        }
    }

    #[test]
    fn no_compactable_sites_returns_doc_as_is() {
        // Pure scalar object → nothing to escalate, document unchanged.
        let doc = json!({"a": 1, "b": "short"});
        let r = dc().compact_with_budget(doc.clone(), 1000);
        assert_eq!(r.document, doc);
        assert!(r.ccr_payloads.is_empty());
    }

    #[test]
    fn placeholder_pattern_does_not_collide_with_real_strings() {
        // Real strings that look vaguely like placeholders must
        // pass through unchanged.
        let doc = json!({
            "comment": "see @@__hr_site:0@@ in the docs",
            "items": build_array(20),
        });
        let r = dc().compact_with_budget(doc, 100_000);
        assert_eq!(
            r.document.pointer("/comment").and_then(|v| v.as_str()),
            Some("see @@__hr_site:0@@ in the docs"),
        );
    }

    #[test]
    fn lossy_max_items_respected() {
        let dc = DocumentCompactor::new().with_lossy_max_items(8);
        let doc = Value::Array(build_array(200));
        let r = dc.compact_with_budget(doc, 100);
        let p = &r.ccr_payloads[0];
        assert_eq!(p.kept_count, 8);
        assert_eq!(p.dropped_count, 192);
    }
}
