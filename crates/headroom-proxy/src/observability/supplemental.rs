//! Transitional "supplemental sources" for `/stats`.
//!
//! During the Python-retirement migration (Phase H) many dashboard blocks are
//! still produced only by the Python proxy — provider quota / subscription /
//! rate-limit panels, plus the prefix-cache, throughput, latency-breakdown,
//! agent-usage, per-project and waste-signal panels that have no Rust recorder
//! yet. So that the Rust proxy can be the **single source of truth** for the
//! dashboard *before* those are ported, an operator can point it at the Python
//! proxy's `/stats` and the Rust `/stats` will fold in those still-Python-only
//! blocks. The allowlist must cover every top-level block the embedded
//! `dashboard.html` reads but the Rust `/stats` does not yet emit — otherwise
//! that panel stays empty even with the fold-in enabled.
//!
//! This module is the (pure) merge step. Fetching is done by the caller and is
//! fail-open: if the Python proxy is unreachable, the Rust `/stats` is returned
//! unchanged. As each block is reimplemented in Rust, drop its key from
//! [`SUPPLEMENTAL_KEYS`] and the merge stops importing it.

use serde_json::Value;

/// Top-level `/stats` keys produced only by the transitional Python proxy. The
/// merge imports exactly these and nothing else, and never overwrites a key the
/// Rust proxy already produced — so a block flips to the Rust-native version the
/// moment Rust starts emitting it, with no code change here.
pub const SUPPLEMENTAL_KEYS: &[&str] = &[
    // Provider quota / subscription / rate-limit panels.
    "copilot_quota",
    "subscription_window",
    "codex_rate_limits",
    "context_tool",
    // Other top-level blocks the embedded dashboard reads that the Rust `/stats`
    // does not yet emit (Python's server.py produces each at the top level).
    // `savings` carries the per-project table and `by_layer` (incl. the CLI-
    // filtering panel — the dashboard reads cli_filtering only nested under
    // `savings`/`tokens`, never as a top-level key, and Python's top-level
    // `cli_filtering` just duplicates `context_tool.stats`, so it is NOT listed
    // here). The rest back the prefix-cache, throughput, latency-breakdown,
    // agent-usage, waste-signal and savings-sparkline panels.
    //
    // When Rust starts emitting one of these natively it MUST emit the WHOLE
    // block: the merge is shallow per-top-level-key (never deep), so a partial
    // Rust `savings` (e.g. only `by_layer`) would suppress Python's `per_project`
    // rather than backfill it. Drop a key here once Rust fully owns it.
    "savings",
    "prefix_cache",
    "throughput",
    "agent_usage",
    "waste_signals",
    "config",
    "ttfb",
    "overhead",
    "pipeline_timing",
    "savings_history",
    "anon_telemetry_shipping",
    "log_full_messages",
];

/// Merge the supplemental (Python-only) blocks from `python` into `rust`.
///
/// Additive and non-destructive: only keys in [`SUPPLEMENTAL_KEYS`] that are
/// *absent* from `rust` are copied in. Returns the augmented value; if either
/// side is not a JSON object, `rust` is returned unchanged.
pub fn merge_supplemental(mut rust: Value, python: &Value) -> Value {
    let (Some(rust_obj), Some(py_obj)) = (rust.as_object_mut(), python.as_object()) else {
        return rust;
    };
    for key in SUPPLEMENTAL_KEYS {
        if rust_obj.contains_key(*key) {
            continue; // never clobber a Rust-native field
        }
        if let Some(value) = py_obj.get(*key) {
            rust_obj.insert((*key).to_string(), value.clone());
        }
    }
    rust
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn imports_only_supplemental_keys() {
        let rust = json!({"requests": {"total": 3}});
        let python = json!({
            "requests": {"total": 999},          // must NOT override Rust
            "copilot_quota": {"latest": {"used": 10}},
            "subscription_window": {"latest": {"pct": 42}},
            "secret_python_only": {"x": 1}        // not in the allowlist → ignored
        });
        let merged = merge_supplemental(rust, &python);
        assert_eq!(merged["requests"]["total"], 3); // unchanged
        assert_eq!(merged["copilot_quota"]["latest"]["used"], 10);
        assert_eq!(merged["subscription_window"]["latest"]["pct"], 42);
        assert!(merged.get("secret_python_only").is_none());
    }

    #[test]
    fn folds_in_the_broader_python_only_panels() {
        // The allowlist must cover every Python-only top-level block the shared
        // dashboard reads (prefix_cache, savings/per_project, throughput, …) or the
        // panel stays empty even with the fold-in enabled.
        let rust = json!({"requests": {"total": 1}});
        let python = json!({
            "prefix_cache": {"hit_rate": 0.8},
            "savings": {"per_project": {"acme": {"tokens_saved": 10}}},
            "throughput": {"rps": 5},
            "agent_usage": {"agents": {}},
            "waste_signals": {"x": 1},
        });
        let merged = merge_supplemental(rust, &python);
        assert_eq!(merged["prefix_cache"]["hit_rate"], 0.8);
        assert_eq!(merged["savings"]["per_project"]["acme"]["tokens_saved"], 10);
        assert_eq!(merged["throughput"]["rps"], 5);
        assert!(merged.get("agent_usage").is_some());
        assert!(merged.get("waste_signals").is_some());
    }

    #[test]
    fn never_overwrites_existing_rust_key() {
        // Once Rust produces `context_tool` itself, the Python copy is ignored.
        let rust = json!({"context_tool": {"source": "rust"}});
        let python = json!({"context_tool": {"source": "python"}});
        let merged = merge_supplemental(rust, &python);
        assert_eq!(merged["context_tool"]["source"], "rust");
    }

    #[test]
    fn missing_supplemental_keys_are_skipped() {
        let rust = json!({"requests": {"total": 1}});
        let python = json!({"copilot_quota": {"latest": {}}}); // only one present
        let merged = merge_supplemental(rust, &python);
        assert!(merged.get("copilot_quota").is_some());
        assert!(merged.get("subscription_window").is_none());
    }

    #[test]
    fn non_object_inputs_return_rust_unchanged() {
        let rust = json!({"a": 1});
        assert_eq!(
            merge_supplemental(rust.clone(), &json!("not an object")),
            rust
        );
        assert_eq!(
            merge_supplemental(json!("rust not object"), &json!({"x": 1})),
            json!("rust not object")
        );
    }
}
