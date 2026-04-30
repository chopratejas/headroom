use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use headroom_core::ccr::CcrStore;
use serde_json::{json, Value};

const DEFAULT_CCR_HINT_MAX_ITEMS: u64 = 15;
const DEFAULT_CCR_HINT_MIN_ITEMS: u64 = 3;
const DEFAULT_CCR_HINT_AGGRESSIVENESS: f64 = 0.7;

#[derive(Clone, Default)]
pub struct ProductStore {
    inner: Arc<Mutex<ProductState>>,
}

#[derive(Default)]
struct ProductState {
    ccr_entries: BTreeMap<String, CcrEntry>,
    retrieval_events: Vec<RetrievalEvent>,
    feedback_patterns: BTreeMap<String, FeedbackPattern>,
    toin_patterns: BTreeMap<String, Value>,
}

#[derive(Clone)]
struct CcrEntry {
    original_content: Value,
    original_tokens: u64,
    original_item_count: u64,
    compressed_item_count: u64,
    tool_name: Option<String>,
    retrieval_count: u64,
}

#[derive(Clone)]
struct RetrievalEvent {
    hash: String,
    query: Option<String>,
    items_retrieved: u64,
    total_items: u64,
    tool_name: Option<String>,
    retrieval_type: &'static str,
}

#[derive(Clone, Default)]
struct FeedbackPattern {
    total_compressions: u64,
    total_retrievals: u64,
    full_retrieval_rate: f64,
    search_rate: f64,
    common_queries: Vec<String>,
    queried_fields: Vec<String>,
}

impl ProductStore {
    pub fn retrieve(&self, hash_key: &str) -> Option<Value> {
        let mut inner = self.inner.lock().expect("product store mutex poisoned");
        let entry = inner.ccr_entries.get_mut(hash_key)?;
        entry.retrieval_count += 1;
        let original_item_count = entry.original_item_count;
        let tool_name = entry.tool_name.clone();
        let response = json!({
            "hash": hash_key,
            "original_content": entry.original_content.clone(),
            "original_tokens": entry.original_tokens,
            "original_item_count": entry.original_item_count,
            "compressed_item_count": entry.compressed_item_count,
            "tool_name": entry.tool_name.clone(),
            "retrieval_count": entry.retrieval_count,
        });
        inner.retrieval_events.push(RetrievalEvent {
            hash: hash_key.to_string(),
            query: None,
            items_retrieved: original_item_count,
            total_items: original_item_count,
            tool_name,
            retrieval_type: "full",
        });
        Some(response)
    }

    pub fn search(&self, hash_key: &str, query: &str) -> Value {
        let mut inner = self.inner.lock().expect("product store mutex poisoned");
        let total_items = inner
            .ccr_entries
            .get(hash_key)
            .map(|entry| entry.original_item_count)
            .unwrap_or(0);
        let tool_name = inner
            .ccr_entries
            .get(hash_key)
            .and_then(|entry| entry.tool_name.clone());
        inner.retrieval_events.push(RetrievalEvent {
            hash: hash_key.to_string(),
            query: Some(query.to_string()),
            items_retrieved: 0,
            total_items,
            tool_name,
            retrieval_type: "search",
        });
        json!({
            "hash": hash_key,
            "query": query,
            "results": [],
            "count": 0,
        })
    }

    pub fn ccr_stats(&self) -> Value {
        let inner = self.inner.lock().expect("product store mutex poisoned");
        let total_original_tokens = inner
            .ccr_entries
            .values()
            .map(|entry| entry.original_tokens)
            .sum::<u64>();
        let total_compressed_tokens = inner
            .ccr_entries
            .values()
            .map(|entry| entry.compressed_item_count)
            .sum::<u64>();
        let total_retrievals = inner
            .ccr_entries
            .values()
            .map(|entry| entry.retrieval_count)
            .sum::<u64>();
        json!({
            "store": {
                "entry_count": inner.ccr_entries.len(),
                "max_entries": 0,
                "total_original_tokens": total_original_tokens,
                "total_compressed_tokens": total_compressed_tokens,
                "total_retrievals": total_retrievals,
                "event_count": inner.retrieval_events.len(),
                "backend": {
                    "backend_type": "memory",
                    "entry_count": inner.ccr_entries.len(),
                    "bytes_used": 0
                }
            },
            "recent_retrievals": inner.retrieval_events.iter().rev().take(20).map(|event| {
                json!({
                    "hash": event.hash,
                    "query": event.query,
                    "items_retrieved": event.items_retrieved,
                    "total_items": event.total_items,
                    "tool_name": event.tool_name,
                    "retrieval_type": event.retrieval_type,
                })
            }).collect::<Vec<_>>(),
        })
    }

    pub fn feedback_stats(&self) -> Value {
        let inner = self.inner.lock().expect("product store mutex poisoned");
        let total_compressions = inner
            .feedback_patterns
            .values()
            .map(|pattern| pattern.total_compressions)
            .sum::<u64>();
        let total_retrievals = inner
            .feedback_patterns
            .values()
            .map(|pattern| pattern.total_retrievals)
            .sum::<u64>();
        let tool_patterns = inner
            .feedback_patterns
            .iter()
            .map(|(tool_name, pattern)| {
                (
                    tool_name.clone(),
                    json!({
                        "compressions": pattern.total_compressions,
                        "retrievals": pattern.total_retrievals,
                        "retrieval_rate": if pattern.total_compressions > 0 {
                            pattern.total_retrievals as f64 / pattern.total_compressions as f64
                        } else {
                            0.0
                        },
                        "full_rate": pattern.full_retrieval_rate,
                        "search_rate": pattern.search_rate,
                        "common_queries": pattern.common_queries,
                        "queried_fields": pattern.queried_fields,
                    }),
                )
            })
            .collect::<serde_json::Map<String, Value>>();
        json!({
            "feedback": {
                "total_compressions": total_compressions,
                "total_retrievals": total_retrievals,
                "global_retrieval_rate": if total_compressions > 0 {
                    total_retrievals as f64 / total_compressions as f64
                } else {
                    0.0
                },
                "tools_tracked": inner.feedback_patterns.len(),
                "tool_patterns": tool_patterns,
            },
            "hints_example": {},
        })
    }

    pub fn feedback_for_tool(&self, tool_name: &str) -> Value {
        let inner = self.inner.lock().expect("product store mutex poisoned");
        let pattern = inner.feedback_patterns.get(tool_name);
        json!({
            "tool_name": tool_name,
            "hints": {
                "max_items": DEFAULT_CCR_HINT_MAX_ITEMS,
                "min_items": DEFAULT_CCR_HINT_MIN_ITEMS,
                "suggested_items": Value::Null,
                "aggressiveness": DEFAULT_CCR_HINT_AGGRESSIVENESS,
                "skip_compression": false,
                "preserve_fields": [],
                "reason": format!("No pattern data for {tool_name}, using defaults"),
            },
            "pattern": pattern.map(|pattern| json!({
                "total_compressions": pattern.total_compressions,
                "total_retrievals": pattern.total_retrievals,
                "retrieval_rate": if pattern.total_compressions > 0 {
                    pattern.total_retrievals as f64 / pattern.total_compressions as f64
                } else {
                    0.0
                },
                "full_retrieval_rate": pattern.full_retrieval_rate,
                "search_rate": pattern.search_rate,
                "common_queries": pattern.common_queries,
                "queried_fields": pattern.queried_fields,
            })),
        })
    }

    pub fn toin_stats(&self) -> Value {
        let inner = self.inner.lock().expect("product store mutex poisoned");
        let total_compressions = inner
            .toin_patterns
            .values()
            .filter_map(|pattern| pattern.get("total_compressions").and_then(Value::as_u64))
            .sum::<u64>();
        let total_retrievals = inner
            .toin_patterns
            .values()
            .filter_map(|pattern| pattern.get("total_retrievals").and_then(Value::as_u64))
            .sum::<u64>();
        let patterns_with_recommendations = inner
            .toin_patterns
            .values()
            .filter(|pattern| {
                pattern
                    .get("sample_size")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    >= 10
            })
            .count();
        json!({
            "enabled": true,
            "patterns_tracked": inner.toin_patterns.len(),
            "total_compressions": total_compressions,
            "total_retrievals": total_retrievals,
            "global_retrieval_rate": if total_compressions > 0 {
                total_retrievals as f64 / total_compressions as f64
            } else {
                0.0
            },
            "patterns_with_recommendations": patterns_with_recommendations,
        })
    }

    pub fn toin_patterns(&self, limit: usize) -> Vec<Value> {
        let inner = self.inner.lock().expect("product store mutex poisoned");
        let mut patterns = inner
            .toin_patterns
            .iter()
            .map(|(sig_hash, pattern)| {
                let compressions = pattern
                    .get("total_compressions")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let retrievals = pattern
                    .get("total_retrievals")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let retrieval_rate = if compressions > 0 {
                    retrievals as f64 / compressions as f64
                } else {
                    0.0
                };
                let sample_size = pattern
                    .get("sample_size")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                (
                    sample_size,
                    json!({
                        "hash": &sig_hash[..sig_hash.len().min(12)],
                        "compressions": compressions,
                        "retrievals": retrievals,
                        "retrieval_rate": format!("{:.1}%", retrieval_rate * 100.0),
                        "confidence": pattern.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
                        "skip_recommended": pattern.get("skip_compression_recommended").and_then(Value::as_bool).unwrap_or(false),
                        "optimal_max_items": pattern.get("optimal_max_items").and_then(Value::as_u64).unwrap_or(20),
                    }),
                )
            })
            .collect::<Vec<_>>();
        patterns.sort_by(|left, right| right.0.cmp(&left.0));
        patterns
            .into_iter()
            .take(limit)
            .map(|(_, pattern)| pattern)
            .collect()
    }

    pub fn toin_pattern_detail(&self, hash_prefix: &str) -> Option<Value> {
        let inner = self.inner.lock().expect("product store mutex poisoned");
        inner
            .toin_patterns
            .iter()
            .find(|(signature_hash, _)| signature_hash.starts_with(hash_prefix))
            .map(|(_, pattern)| pattern.clone())
    }
}

impl CcrStore for ProductStore {
    fn put(&self, hash: &str, payload: &str) {
        let mut inner = self.inner.lock().expect("product store mutex poisoned");
        let original_content =
            serde_json::from_str(payload).unwrap_or_else(|_| Value::String(payload.to_string()));
        let original_item_count = match &original_content {
            Value::Array(items) => items.len() as u64,
            Value::String(text) => text.lines().count().max(1) as u64,
            _ => 1,
        };
        let original_tokens = match &original_content {
            Value::String(text) => (text.len() / 4).max(1) as u64,
            Value::Array(items) => items.len() as u64,
            _ => 1,
        };
        inner.ccr_entries.insert(
            hash.to_string(),
            CcrEntry {
                original_content,
                original_tokens,
                original_item_count,
                compressed_item_count: 0,
                tool_name: None,
                retrieval_count: 0,
            },
        );
    }

    fn get(&self, hash: &str) -> Option<String> {
        self.retrieve(hash)
            .and_then(|value| value.get("original_content").cloned())
            .map(|value| match value {
                Value::String(text) => text,
                other => other.to_string(),
            })
    }

    fn len(&self) -> usize {
        let inner = self.inner.lock().expect("product store mutex poisoned");
        inner.ccr_entries.len()
    }
}
