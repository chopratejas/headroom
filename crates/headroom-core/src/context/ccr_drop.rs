//! CCR-on-drop — persist dropped messages so the model can retrieve
//! them later via a tool call.
//!
//! This is the OSS-defining behaviour. Without it, dropping a message
//! is destructive (≈ rolling window). With it, the dropped content
//! is parked in the [`CcrStore`] under a content-hash key and a marker
//! is inserted into the surviving message stream pointing at it.
//! When the model later calls `ccr_retrieve(<key>)`, the dropped JSON
//! comes back verbatim.
//!
//! Mirrors Python's `_store_dropped_in_ccr` (intelligent_context.py
//! ~L955) including the marker text format. Differences from Python:
//!
//! - Rust uses the trait-level [`CcrStore`] directly (no Python
//!   `CompressionStore` indirection).
//! - The marker is returned to the caller for insertion; Python
//!   handled insertion outside this helper too.

use std::sync::Arc;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::ccr::CcrStore;

/// Result of stashing dropped messages into the CCR store.
#[derive(Debug, Clone)]
pub struct DropPersist {
    /// CCR cache key the dropped JSON was stored under. The marker
    /// references this so the model knows what to ask for.
    pub key: String,
    /// Human-readable marker string suitable for insertion as a
    /// system or user message in the surviving conversation.
    pub marker: String,
    /// Number of messages persisted.
    pub count: usize,
}

/// Serialize the messages at `dropped_indices` from `original_messages`
/// and stash them in `store` under a content-hash key. Returns metadata
/// describing the persistence so the caller can insert the marker into
/// the live message list.
///
/// Returns `None` when there's nothing to do (no indices or no store)
/// or when serialization fails — failure is non-fatal because the drop
/// itself still happened. The caller logs and continues.
pub fn persist_dropped(
    original_messages: &[Value],
    dropped_indices: &[usize],
    store: &Arc<dyn CcrStore>,
) -> Option<DropPersist> {
    if dropped_indices.is_empty() {
        return None;
    }

    // Collect in original order — re-sort because the cascade may
    // append in non-monotonic order.
    let mut sorted = dropped_indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let dropped: Vec<&Value> = sorted
        .iter()
        .filter_map(|&i| original_messages.get(i))
        .collect();

    if dropped.is_empty() {
        return None;
    }

    let dropped_json = serde_json::to_string_pretty(&dropped).ok()?;

    // Hash + truncate to 12 hex chars — same convention as the rest
    // of the CCR store (see ccr.rs and smart_crusher hashing).
    let mut h = Sha256::new();
    h.update(dropped_json.as_bytes());
    let key = h
        .finalize()
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect::<String>();

    // Role-count summary mirrors Python's marker format so anyone
    // grepping logs sees the same string shape across implementations.
    let marker = build_marker(&dropped);

    // Store the original (full JSON) under the key. The marker stays
    // outside the store — it lives in the live conversation as a
    // pointer; the store holds only the recoverable payload.
    store.put(&key, &dropped_json);

    Some(DropPersist {
        key,
        marker,
        count: dropped.len(),
    })
}

/// Build the human-readable marker string. Format matches Python's
/// `_store_dropped_in_ccr` so log greps work across implementations.
fn build_marker(dropped: &[&Value]) -> String {
    use std::collections::BTreeMap;

    let mut role_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for msg in dropped {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("unknown");
        *role_counts.entry(role).or_insert(0) += 1;
    }

    let parts: Vec<String> = role_counts
        .iter()
        .map(|(role, count)| format!("{count} {role}"))
        .collect();

    format!(
        "[Dropped {} messages: {}. Use ccr_retrieve to access full content.]",
        dropped.len(),
        parts.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;
    use serde_json::json;

    fn store() -> Arc<dyn CcrStore> {
        Arc::new(InMemoryCcrStore::new())
    }

    #[test]
    fn persists_dropped_messages_under_content_hash() {
        let originals = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "drop me"}),
            json!({"role": "assistant", "content": "and me"}),
            json!({"role": "user", "content": "keep me"}),
        ];
        let dropped = vec![1, 2];
        let s = store();
        let result = persist_dropped(&originals, &dropped, &s).expect("should persist");
        assert_eq!(result.count, 2);
        assert_eq!(result.key.len(), 12);
        assert!(result.marker.contains("Dropped 2 messages"));
        assert!(result.marker.contains("ccr_retrieve"));
        // Round-trip: the original JSON is recoverable via the same key.
        let stored = s.get(&result.key).expect("should retrieve");
        let recovered: Vec<Value> = serde_json::from_str(&stored).unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0]["content"], "drop me");
    }

    #[test]
    fn empty_indices_returns_none() {
        let originals = vec![json!({"role": "user", "content": "x"})];
        let s = store();
        assert!(persist_dropped(&originals, &[], &s).is_none());
    }

    #[test]
    fn out_of_range_indices_are_skipped() {
        let originals = vec![json!({"role": "user", "content": "x"})];
        let s = store();
        // Index 99 is out of range — should be silently filtered, not
        // panic. Index 0 valid, so we still get a persist.
        let result = persist_dropped(&originals, &[0, 99], &s).expect("should persist idx 0");
        assert_eq!(result.count, 1);
    }

    #[test]
    fn marker_role_breakdown_is_sorted_alphabetical() {
        // BTreeMap ordering → roles appear alphabetically. Stable
        // marker text across runs.
        let originals = vec![
            json!({"role": "user", "content": "u"}),
            json!({"role": "assistant", "content": "a"}),
            json!({"role": "tool", "content": "t"}),
        ];
        let s = store();
        let result = persist_dropped(&originals, &[0, 1, 2], &s).expect("should persist");
        // "1 assistant" appears before "1 tool" appears before "1 user".
        let m = &result.marker;
        let asst = m.find("assistant").unwrap();
        let tool = m.find("tool").unwrap();
        let user = m.find("user").unwrap();
        assert!(asst < tool && tool < user);
    }

    #[test]
    fn duplicate_indices_are_deduped() {
        let originals = vec![
            json!({"role": "user", "content": "x"}),
            json!({"role": "assistant", "content": "y"}),
        ];
        let s = store();
        let result = persist_dropped(&originals, &[0, 0, 1, 1, 1], &s).expect("should persist");
        // Counts the unique message set, not the input list length.
        assert_eq!(result.count, 2);
    }
}
