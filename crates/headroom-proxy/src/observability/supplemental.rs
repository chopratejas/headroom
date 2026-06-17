//! Transitional "supplemental sources" for `/stats`.
//!
//! During the Python-retirement migration (Phase H) a few dashboard blocks are
//! still produced only by the Python proxy — provider quota / subscription /
//! rate-limit panels that have no Rust equivalent yet. So that the Rust proxy
//! can be the **single source of truth** for the dashboard *before* those are
//! ported, an operator can point it at the Python proxy's `/stats` and the Rust
//! `/stats` will fold in just those still-Python-only blocks.
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
    "copilot_quota",
    "subscription_window",
    "codex_rate_limits",
    "context_tool",
    "cli_filtering",
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
