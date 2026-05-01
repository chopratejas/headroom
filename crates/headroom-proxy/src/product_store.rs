use std::collections::{hash_map::DefaultHasher, BTreeMap};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use headroom_core::ccr::CcrStore;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::state_store::{
    backend_details, file_backend, load_json_state, persist_json_state, SharedStateBackend,
};

const DEFAULT_CCR_HINT_MAX_ITEMS: u64 = 15;
const DEFAULT_CCR_HINT_MIN_ITEMS: u64 = 3;
const DEFAULT_CCR_HINT_AGGRESSIVENESS: f64 = 0.7;
const HIGH_RETRIEVAL_THRESHOLD: f64 = 0.5;
const MEDIUM_RETRIEVAL_THRESHOLD: f64 = 0.2;
const PRODUCT_CCR_TABLE_NAME: &str = "product_ccr_entries";
const PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME: &str = "product_retrieval_events";
const PRODUCT_FEEDBACK_TABLE_NAME: &str = "product_feedback_patterns";
const PRODUCT_TOIN_TABLE_NAME: &str = "product_toin_patterns";

#[derive(Clone)]
pub struct ProductStore {
    inner: Arc<Mutex<ProductState>>,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct ProductState {
    ccr_entries: BTreeMap<String, CcrEntry>,
    retrieval_events: Vec<RetrievalEvent>,
    feedback_patterns: BTreeMap<String, FeedbackPattern>,
    toin_patterns: BTreeMap<String, ToinPattern>,
    #[serde(skip)]
    backend: Option<SharedStateBackend>,
}

#[derive(Clone, Serialize, Deserialize)]
struct CcrEntry {
    original_content: Value,
    original_tokens: u64,
    original_item_count: u64,
    compressed_item_count: u64,
    tool_name: Option<String>,
    retrieval_count: u64,
}

#[derive(Clone, Serialize, Deserialize)]
struct RetrievalEvent {
    hash: String,
    query: Option<String>,
    items_retrieved: u64,
    total_items: u64,
    tool_name: Option<String>,
    retrieval_type: String,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct FeedbackPattern {
    total_compressions: u64,
    total_retrievals: u64,
    full_retrievals: u64,
    search_retrievals: u64,
    common_queries: BTreeMap<String, u64>,
    queried_fields: BTreeMap<String, u64>,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct ToinPattern {
    tool_signature_hash: String,
    total_compressions: u64,
    total_items_seen: u64,
    total_items_kept: u64,
    total_retrievals: u64,
    full_retrievals: u64,
    search_retrievals: u64,
    field_retrieval_frequency: BTreeMap<String, u64>,
    common_query_patterns: BTreeMap<String, u64>,
    optimal_max_items: u64,
    skip_compression_recommended: bool,
    preserve_fields: Vec<String>,
    sample_size: u64,
    confidence: f64,
}

impl FeedbackPattern {
    fn retrieval_rate(&self) -> f64 {
        if self.total_compressions == 0 {
            0.0
        } else {
            self.total_retrievals as f64 / self.total_compressions as f64
        }
    }

    fn full_retrieval_rate(&self) -> f64 {
        if self.total_retrievals == 0 {
            0.0
        } else {
            self.full_retrievals as f64 / self.total_retrievals as f64
        }
    }

    fn search_rate(&self) -> f64 {
        if self.total_retrievals == 0 {
            0.0
        } else {
            self.search_retrievals as f64 / self.total_retrievals as f64
        }
    }

    fn top_queries(&self, limit: usize) -> Vec<String> {
        top_keys(&self.common_queries, limit)
    }

    fn top_fields(&self, limit: usize) -> Vec<String> {
        top_keys(&self.queried_fields, limit)
    }
}

impl ToinPattern {
    fn retrieval_rate(&self) -> f64 {
        if self.total_compressions == 0 {
            0.0
        } else {
            self.total_retrievals as f64 / self.total_compressions as f64
        }
    }

    fn full_retrieval_rate(&self) -> f64 {
        if self.total_retrievals == 0 {
            0.0
        } else {
            self.full_retrievals as f64 / self.total_retrievals as f64
        }
    }

    fn update_recommendations(&mut self) {
        let retrieval_rate = self.retrieval_rate();
        self.skip_compression_recommended = false;
        self.optimal_max_items = 20;

        if retrieval_rate > HIGH_RETRIEVAL_THRESHOLD {
            if self.full_retrieval_rate() > 0.8 {
                self.skip_compression_recommended = true;
                self.optimal_max_items =
                    (self.total_items_seen / self.total_compressions.max(1)).max(20);
            } else {
                self.optimal_max_items = 50;
            }
        } else if retrieval_rate > MEDIUM_RETRIEVAL_THRESHOLD {
            self.optimal_max_items = 30;
        }

        self.preserve_fields = top_keys(&self.field_retrieval_frequency, 10);
        self.confidence = ((self.sample_size.min(20) as f64) / 20.0 * 100.0).round() / 100.0;
    }

    fn as_value(&self) -> Value {
        json!({
            "tool_signature_hash": self.tool_signature_hash,
            "total_compressions": self.total_compressions,
            "total_items_seen": self.total_items_seen,
            "total_items_kept": self.total_items_kept,
            "total_retrievals": self.total_retrievals,
            "full_retrievals": self.full_retrievals,
            "search_retrievals": self.search_retrievals,
            "field_retrieval_frequency": self.field_retrieval_frequency,
            "common_query_patterns": top_keys(&self.common_query_patterns, 10),
            "optimal_max_items": self.optimal_max_items,
            "skip_compression_recommended": self.skip_compression_recommended,
            "preserve_fields": self.preserve_fields,
            "sample_size": self.sample_size,
            "confidence": self.confidence,
        })
    }
}

impl ProductStore {
    pub fn new(path: Option<PathBuf>) -> Self {
        let backend = path.map(file_backend);
        Self::new_with_backend(backend)
    }

    pub fn new_with_backend(backend: Option<SharedStateBackend>) -> Self {
        let mut state = load_state(backend.as_ref()).unwrap_or_default();
        state.backend = backend;
        Self {
            inner: Arc::new(Mutex::new(state)),
        }
    }

    pub fn tool_name_for_hash(&self, hash_key: &str) -> Option<String> {
        let inner = self.readable_state();
        inner
            .ccr_entries
            .get(hash_key)
            .and_then(|entry| entry.tool_name.clone())
    }

    pub fn backend_info(&self) -> Value {
        let inner = self.readable_state();
        backend_details(inner.backend.as_ref(), inner.ccr_entries.len())
    }

    pub fn record_compression(
        &self,
        hash_key: Option<&str>,
        tool_name: &str,
        original_count: u64,
        compressed_count: u64,
        query: Option<&str>,
    ) {
        let mut inner = self.inner.lock().expect("product store mutex poisoned");
        refresh_state_from_sqlite(&mut inner);
        let hash_key_owned = hash_key.map(str::to_string);
        if let Some(hash_key) = hash_key {
            if let Some(entry) = inner.ccr_entries.get_mut(hash_key) {
                entry.tool_name = Some(tool_name.to_string());
                entry.compressed_item_count = compressed_count;
            }
        }

        let feedback = inner
            .feedback_patterns
            .entry(tool_name.to_string())
            .or_default();
        feedback.total_compressions += 1;
        if let Some(query) = query.filter(|query| !query.trim().is_empty()) {
            record_ranked_value(
                &mut feedback.common_queries,
                &query.to_ascii_lowercase(),
                100,
            );
            for field in extract_query_fields(query) {
                record_ranked_value(&mut feedback.queried_fields, &field, 50);
            }
        }

        let signature_hash = stable_signature_hash(tool_name);
        let toin = inner
            .toin_patterns
            .entry(signature_hash.clone())
            .or_insert_with(|| ToinPattern {
                tool_signature_hash: signature_hash.clone(),
                ..ToinPattern::default()
            });
        toin.total_compressions += 1;
        toin.total_items_seen += original_count;
        toin.total_items_kept += compressed_count;
        toin.sample_size += 1;
        if let Some(query) = query.filter(|query| !query.trim().is_empty()) {
            record_ranked_value(
                &mut toin.common_query_patterns,
                &anonymize_query_pattern(query),
                20,
            );
        }
        toin.update_recommendations();
        let ccr_entry_snapshot = hash_key_owned
            .as_deref()
            .and_then(|hash| inner.ccr_entries.get(hash).cloned())
            .map(|entry| (hash_key_owned.clone().expect("hash key cloned"), entry));
        let feedback_snapshot = inner.feedback_patterns.get(tool_name).cloned();
        let toin_snapshot = inner
            .toin_patterns
            .get(&signature_hash)
            .cloned()
            .map(|pattern| (signature_hash.clone(), pattern));
        if let Some(path) = sqlite_storage_path(inner.backend.as_ref()) {
            let _ = persist_product_delta_to_sqlite(
                &path,
                ccr_entry_snapshot.as_ref(),
                None,
                feedback_snapshot
                    .as_ref()
                    .map(|pattern| (tool_name, pattern)),
                toin_snapshot
                    .as_ref()
                    .map(|(signature_hash, pattern)| (signature_hash.as_str(), pattern)),
            );
        } else {
            persist_state(&inner);
        }
    }

    pub fn retrieve(&self, hash_key: &str) -> Option<Value> {
        let mut inner = self.inner.lock().expect("product store mutex poisoned");
        refresh_state_from_sqlite(&mut inner);
        let (
            original_content,
            original_tokens,
            original_item_count,
            compressed_item_count,
            tool_name,
            retrieval_count,
        ) = {
            let entry = inner.ccr_entries.get_mut(hash_key)?;
            entry.retrieval_count += 1;
            (
                entry.original_content.clone(),
                entry.original_tokens,
                entry.original_item_count,
                entry.compressed_item_count,
                entry.tool_name.clone(),
                entry.retrieval_count,
            )
        };
        update_patterns_for_retrieval(&mut inner, tool_name.as_deref(), None, "full");
        let response = json!({
            "hash": hash_key,
            "original_content": original_content,
            "original_tokens": original_tokens,
            "original_item_count": original_item_count,
            "compressed_item_count": compressed_item_count,
            "tool_name": tool_name.clone(),
            "retrieval_count": retrieval_count,
        });
        inner.retrieval_events.push(RetrievalEvent {
            hash: hash_key.to_string(),
            query: None,
            items_retrieved: original_item_count,
            total_items: original_item_count,
            tool_name: tool_name.clone(),
            retrieval_type: "full".to_string(),
        });
        let ccr_entry_snapshot = inner
            .ccr_entries
            .get(hash_key)
            .cloned()
            .map(|entry| (hash_key.to_string(), entry));
        let feedback_snapshot = tool_name.as_deref().and_then(|tool| {
            inner
                .feedback_patterns
                .get(tool)
                .cloned()
                .map(|pattern| (tool.to_string(), pattern))
        });
        let toin_signature = tool_name.as_deref().map(stable_signature_hash);
        let toin_snapshot = toin_signature.as_deref().and_then(|signature_hash| {
            inner
                .toin_patterns
                .get(signature_hash)
                .cloned()
                .map(|pattern| (signature_hash.to_string(), pattern))
        });
        let retrieval_event = inner.retrieval_events.last().cloned();
        if let Some(path) = sqlite_storage_path(inner.backend.as_ref()) {
            let _ = persist_product_delta_to_sqlite(
                &path,
                ccr_entry_snapshot.as_ref(),
                retrieval_event.as_ref(),
                feedback_snapshot
                    .as_ref()
                    .map(|(tool_name, pattern)| (tool_name.as_str(), pattern)),
                toin_snapshot
                    .as_ref()
                    .map(|(signature_hash, pattern)| (&signature_hash[..], pattern)),
            );
        } else {
            persist_state(&inner);
        }
        Some(response)
    }

    pub fn search(&self, hash_key: &str, query: &str) -> Value {
        let mut inner = self.inner.lock().expect("product store mutex poisoned");
        refresh_state_from_sqlite(&mut inner);
        let total_items = inner
            .ccr_entries
            .get(hash_key)
            .map(|entry| entry.original_item_count)
            .unwrap_or(0);
        let tool_name = inner
            .ccr_entries
            .get(hash_key)
            .and_then(|entry| entry.tool_name.clone());
        update_patterns_for_retrieval(&mut inner, tool_name.as_deref(), Some(query), "search");
        inner.retrieval_events.push(RetrievalEvent {
            hash: hash_key.to_string(),
            query: Some(query.to_string()),
            items_retrieved: 0,
            total_items,
            tool_name: tool_name.clone(),
            retrieval_type: "search".to_string(),
        });
        let feedback_snapshot = tool_name.as_deref().and_then(|tool| {
            inner
                .feedback_patterns
                .get(tool)
                .cloned()
                .map(|pattern| (tool.to_string(), pattern))
        });
        let toin_signature = tool_name.as_deref().map(stable_signature_hash);
        let toin_snapshot = toin_signature.as_deref().and_then(|signature_hash| {
            inner
                .toin_patterns
                .get(signature_hash)
                .cloned()
                .map(|pattern| (signature_hash.to_string(), pattern))
        });
        let retrieval_event = inner.retrieval_events.last().cloned();
        if let Some(path) = sqlite_storage_path(inner.backend.as_ref()) {
            let _ = persist_product_delta_to_sqlite(
                &path,
                None,
                retrieval_event.as_ref(),
                feedback_snapshot
                    .as_ref()
                    .map(|(tool_name, pattern)| (tool_name.as_str(), pattern)),
                toin_snapshot
                    .as_ref()
                    .map(|(signature_hash, pattern)| (&signature_hash[..], pattern)),
            );
        } else {
            persist_state(&inner);
        }
        json!({
            "hash": hash_key,
            "query": query,
            "results": [],
            "count": 0,
        })
    }

    pub fn ccr_stats(&self) -> Value {
        let inner = self.readable_state();
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
                "backend": backend_details(inner.backend.as_ref(), inner.ccr_entries.len()),
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
        let inner = self.readable_state();
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
                        "retrieval_rate": pattern.retrieval_rate(),
                        "full_rate": pattern.full_retrieval_rate(),
                        "search_rate": pattern.search_rate(),
                        "common_queries": pattern.top_queries(5),
                        "queried_fields": pattern.top_fields(5),
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
        let inner = self.readable_state();
        let pattern = inner.feedback_patterns.get(tool_name);
        json!({
            "tool_name": tool_name,
            "hints": {
                "max_items": pattern.map(compute_hint_max_items).unwrap_or(DEFAULT_CCR_HINT_MAX_ITEMS),
                "min_items": DEFAULT_CCR_HINT_MIN_ITEMS,
                "suggested_items": Value::Null,
                "aggressiveness": pattern.map(compute_hint_aggressiveness).unwrap_or(DEFAULT_CCR_HINT_AGGRESSIVENESS),
                "skip_compression": pattern.map(|pattern| pattern.full_retrieval_rate() > 0.8 && pattern.retrieval_rate() > HIGH_RETRIEVAL_THRESHOLD).unwrap_or(false),
                "preserve_fields": pattern.map(|pattern| pattern.top_fields(5)).unwrap_or_default(),
                "reason": pattern.map(compute_hint_reason).unwrap_or_else(|| format!("No pattern data for {tool_name}, using defaults")),
            },
            "pattern": pattern.map(|pattern| json!({
                "total_compressions": pattern.total_compressions,
                "total_retrievals": pattern.total_retrievals,
                "retrieval_rate": pattern.retrieval_rate(),
                "full_retrieval_rate": pattern.full_retrieval_rate(),
                "search_rate": pattern.search_rate(),
                "common_queries": pattern.top_queries(5),
                "queried_fields": pattern.top_fields(5),
            })),
        })
    }

    pub fn toin_stats(&self) -> Value {
        let inner = self.readable_state();
        let total_compressions = inner
            .toin_patterns
            .values()
            .map(|pattern| pattern.total_compressions)
            .sum::<u64>();
        let total_retrievals = inner
            .toin_patterns
            .values()
            .map(|pattern| pattern.total_retrievals)
            .sum::<u64>();
        let patterns_with_recommendations = inner
            .toin_patterns
            .values()
            .filter(|pattern| pattern.sample_size >= 10)
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
        let inner = self.readable_state();
        let mut patterns = inner
            .toin_patterns
            .iter()
            .map(|(sig_hash, pattern)| {
                let compressions = pattern.total_compressions;
                let retrievals = pattern.total_retrievals;
                let retrieval_rate = if compressions > 0 {
                    retrievals as f64 / compressions as f64
                } else {
                    0.0
                };
                let sample_size = pattern.sample_size;
                (
                    sample_size,
                    json!({
                        "hash": &sig_hash[..sig_hash.len().min(12)],
                        "compressions": compressions,
                        "retrievals": retrievals,
                        "retrieval_rate": format!("{:.1}%", retrieval_rate * 100.0),
                        "confidence": pattern.confidence,
                        "skip_recommended": pattern.skip_compression_recommended,
                        "optimal_max_items": pattern.optimal_max_items,
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
        let inner = self.readable_state();
        inner
            .toin_patterns
            .iter()
            .find(|(signature_hash, _)| signature_hash.starts_with(hash_prefix))
            .map(|(_, pattern)| pattern.as_value())
    }

    fn readable_state(&self) -> ProductState {
        let fallback = {
            self.inner
                .lock()
                .expect("product store mutex poisoned")
                .clone()
        };
        if let Some(path) = sqlite_storage_path(fallback.backend.as_ref()) {
            let mut state = load_state_from_sqlite(&path).unwrap_or_default();
            state.backend = fallback.backend;
            state
        } else {
            fallback
        }
    }
}

impl CcrStore for ProductStore {
    fn put(&self, hash: &str, payload: &str) {
        let mut inner = self.inner.lock().expect("product store mutex poisoned");
        refresh_state_from_sqlite(&mut inner);
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
        let ccr_entry_snapshot = inner
            .ccr_entries
            .get(hash)
            .cloned()
            .map(|entry| (hash.to_string(), entry));
        if let Some(path) = sqlite_storage_path(inner.backend.as_ref()) {
            let _ = persist_product_delta_to_sqlite(
                &path,
                ccr_entry_snapshot.as_ref(),
                None,
                None,
                None,
            );
        } else {
            persist_state(&inner);
        }
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
        let inner = self.readable_state();
        inner.ccr_entries.len()
    }
}

fn load_state(backend: Option<&SharedStateBackend>) -> Option<ProductState> {
    if let Some(path) = sqlite_storage_path(backend) {
        return load_state_from_sqlite(&path);
    }
    load_json_state(backend)
}

fn persist_state(state: &ProductState) {
    if let Some(path) = sqlite_storage_path(state.backend.as_ref()) {
        let _ = persist_state_to_sqlite(state, &path);
        return;
    }
    persist_json_state(state, state.backend.as_ref());
}

fn refresh_state_from_sqlite(state: &mut ProductState) {
    let Some(path) = sqlite_storage_path(state.backend.as_ref()) else {
        return;
    };
    let Some(mut loaded) = load_state_from_sqlite(&path) else {
        return;
    };
    loaded.backend = state.backend.clone();
    *state = loaded;
}

fn sqlite_storage_path(backend: Option<&SharedStateBackend>) -> Option<PathBuf> {
    let backend = backend?;
    if backend.backend_type() == "sqlite" {
        backend.storage_path()
    } else {
        None
    }
}

fn connect_sqlite(path: &PathBuf) -> Option<Connection> {
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return None;
        }
    }
    let connection = Connection::open(path).ok()?;
    for statement in [
        format!(
            "CREATE TABLE IF NOT EXISTS {PRODUCT_CCR_TABLE_NAME} (
                hash TEXT PRIMARY KEY,
                original_content_json TEXT NOT NULL,
                original_tokens INTEGER NOT NULL,
                original_item_count INTEGER NOT NULL,
                compressed_item_count INTEGER NOT NULL,
                tool_name TEXT,
                retrieval_count INTEGER NOT NULL
            )"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS {PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT NOT NULL,
                query TEXT,
                items_retrieved INTEGER NOT NULL,
                total_items INTEGER NOT NULL,
                tool_name TEXT,
                retrieval_type TEXT NOT NULL
            )"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS {PRODUCT_FEEDBACK_TABLE_NAME} (
                tool_name TEXT PRIMARY KEY,
                pattern_json TEXT NOT NULL
            )"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS {PRODUCT_TOIN_TABLE_NAME} (
                signature_hash TEXT PRIMARY KEY,
                pattern_json TEXT NOT NULL
            )"
        ),
    ] {
        if connection.execute(&statement, []).is_err() {
            return None;
        }
    }
    Some(connection)
}

fn load_state_from_sqlite(path: &PathBuf) -> Option<ProductState> {
    let connection = connect_sqlite(path)?;

    let mut ccr_entries = BTreeMap::new();
    let mut ccr_statement = connection
        .prepare(&format!(
            "SELECT hash, original_content_json, original_tokens, original_item_count, compressed_item_count, tool_name, retrieval_count
             FROM {PRODUCT_CCR_TABLE_NAME}"
        ))
        .ok()?;
    let ccr_rows = ccr_statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, u64>(2)?,
                row.get::<_, u64>(3)?,
                row.get::<_, u64>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, u64>(6)?,
            ))
        })
        .ok()?;
    for row in ccr_rows {
        let (
            hash,
            original_content_json,
            original_tokens,
            original_item_count,
            compressed_item_count,
            tool_name,
            retrieval_count,
        ) = row.ok()?;
        ccr_entries.insert(
            hash,
            CcrEntry {
                original_content: serde_json::from_str::<Value>(&original_content_json).ok()?,
                original_tokens,
                original_item_count,
                compressed_item_count,
                tool_name,
                retrieval_count,
            },
        );
    }

    let mut retrieval_events = Vec::new();
    let mut events_statement = connection
        .prepare(&format!(
            "SELECT hash, query, items_retrieved, total_items, tool_name, retrieval_type
             FROM {PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME}
             ORDER BY id ASC"
        ))
        .ok()?;
    let event_rows = events_statement
        .query_map([], |row| {
            Ok(RetrievalEvent {
                hash: row.get(0)?,
                query: row.get(1)?,
                items_retrieved: row.get(2)?,
                total_items: row.get(3)?,
                tool_name: row.get(4)?,
                retrieval_type: row.get(5)?,
            })
        })
        .ok()?;
    for row in event_rows {
        retrieval_events.push(row.ok()?);
    }

    let mut feedback_patterns = BTreeMap::new();
    let mut feedback_statement = connection
        .prepare(&format!(
            "SELECT tool_name, pattern_json FROM {PRODUCT_FEEDBACK_TABLE_NAME}"
        ))
        .ok()?;
    let feedback_rows = feedback_statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .ok()?;
    for row in feedback_rows {
        let (tool_name, pattern_json) = row.ok()?;
        feedback_patterns.insert(
            tool_name,
            serde_json::from_str::<FeedbackPattern>(&pattern_json).ok()?,
        );
    }

    let mut toin_patterns = BTreeMap::new();
    let mut toin_statement = connection
        .prepare(&format!(
            "SELECT signature_hash, pattern_json FROM {PRODUCT_TOIN_TABLE_NAME}"
        ))
        .ok()?;
    let toin_rows = toin_statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .ok()?;
    for row in toin_rows {
        let (signature_hash, pattern_json) = row.ok()?;
        toin_patterns.insert(
            signature_hash,
            serde_json::from_str::<ToinPattern>(&pattern_json).ok()?,
        );
    }

    Some(ProductState {
        ccr_entries,
        retrieval_events,
        feedback_patterns,
        toin_patterns,
        backend: None,
    })
}

fn persist_state_to_sqlite(state: &ProductState, path: &PathBuf) -> bool {
    let mut connection = match connect_sqlite(path) {
        Some(connection) => connection,
        None => return false,
    };
    let transaction = match connection.transaction() {
        Ok(transaction) => transaction,
        Err(_) => return false,
    };

    if transaction
        .execute(&format!("DELETE FROM {PRODUCT_CCR_TABLE_NAME}"), [])
        .is_err()
    {
        return false;
    }
    for (hash, entry) in &state.ccr_entries {
        let Ok(original_content_json) = serde_json::to_string(&entry.original_content) else {
            return false;
        };
        if transaction
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_CCR_TABLE_NAME}
                     (hash, original_content_json, original_tokens, original_item_count, compressed_item_count, tool_name, retrieval_count)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
                ),
                params![
                    hash,
                    original_content_json,
                    entry.original_tokens as i64,
                    entry.original_item_count as i64,
                    entry.compressed_item_count as i64,
                    entry.tool_name,
                    entry.retrieval_count as i64
                ],
            )
            .is_err()
        {
            return false;
        }
    }

    if transaction
        .execute(
            &format!("DELETE FROM {PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME}"),
            [],
        )
        .is_err()
    {
        return false;
    }
    for event in &state.retrieval_events {
        if transaction
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME}
                     (hash, query, items_retrieved, total_items, tool_name, retrieval_type)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
                ),
                params![
                    &event.hash,
                    event.query.as_deref(),
                    event.items_retrieved as i64,
                    event.total_items as i64,
                    event.tool_name.as_deref(),
                    &event.retrieval_type
                ],
            )
            .is_err()
        {
            return false;
        }
    }

    if transaction
        .execute(&format!("DELETE FROM {PRODUCT_FEEDBACK_TABLE_NAME}"), [])
        .is_err()
    {
        return false;
    }
    for (tool_name, pattern) in &state.feedback_patterns {
        let Ok(pattern_json) = serde_json::to_string(pattern) else {
            return false;
        };
        if transaction
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_FEEDBACK_TABLE_NAME} (tool_name, pattern_json)
                     VALUES (?1, ?2)"
                ),
                params![tool_name, pattern_json],
            )
            .is_err()
        {
            return false;
        }
    }

    if transaction
        .execute(&format!("DELETE FROM {PRODUCT_TOIN_TABLE_NAME}"), [])
        .is_err()
    {
        return false;
    }
    for (signature_hash, pattern) in &state.toin_patterns {
        let Ok(pattern_json) = serde_json::to_string(pattern) else {
            return false;
        };
        if transaction
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_TOIN_TABLE_NAME} (signature_hash, pattern_json)
                     VALUES (?1, ?2)"
                ),
                params![signature_hash, pattern_json],
            )
            .is_err()
        {
            return false;
        }
    }

    transaction.commit().is_ok()
}

fn persist_product_delta_to_sqlite(
    path: &PathBuf,
    ccr_entry: Option<&(String, CcrEntry)>,
    retrieval_event: Option<&RetrievalEvent>,
    feedback_pattern: Option<(&str, &FeedbackPattern)>,
    toin_pattern: Option<(&str, &ToinPattern)>,
) -> bool {
    let connection = match connect_sqlite(path) {
        Some(connection) => connection,
        None => return false,
    };
    if let Some((hash, entry)) = ccr_entry {
        let Ok(original_content_json) = serde_json::to_string(&entry.original_content) else {
            return false;
        };
        if connection
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_CCR_TABLE_NAME}
                     (hash, original_content_json, original_tokens, original_item_count, compressed_item_count, tool_name, retrieval_count)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                     ON CONFLICT(hash) DO UPDATE SET
                        original_content_json = excluded.original_content_json,
                        original_tokens = excluded.original_tokens,
                        original_item_count = excluded.original_item_count,
                        compressed_item_count = excluded.compressed_item_count,
                        tool_name = excluded.tool_name,
                        retrieval_count = excluded.retrieval_count"
                ),
                params![
                    hash,
                    original_content_json,
                    entry.original_tokens as i64,
                    entry.original_item_count as i64,
                    entry.compressed_item_count as i64,
                    entry.tool_name.as_deref(),
                    entry.retrieval_count as i64
                ],
            )
            .is_err()
        {
            return false;
        }
    }
    if let Some(event) = retrieval_event {
        if connection
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME}
                     (hash, query, items_retrieved, total_items, tool_name, retrieval_type)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
                ),
                params![
                    &event.hash,
                    event.query.as_deref(),
                    event.items_retrieved as i64,
                    event.total_items as i64,
                    event.tool_name.as_deref(),
                    &event.retrieval_type
                ],
            )
            .is_err()
        {
            return false;
        }
    }
    if let Some((tool_name, pattern)) = feedback_pattern {
        let Ok(pattern_json) = serde_json::to_string(pattern) else {
            return false;
        };
        if connection
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_FEEDBACK_TABLE_NAME} (tool_name, pattern_json)
                     VALUES (?1, ?2)
                     ON CONFLICT(tool_name) DO UPDATE SET
                        pattern_json = excluded.pattern_json"
                ),
                params![tool_name, pattern_json],
            )
            .is_err()
        {
            return false;
        }
    }
    if let Some((signature_hash, pattern)) = toin_pattern {
        let Ok(pattern_json) = serde_json::to_string(pattern) else {
            return false;
        };
        if connection
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_TOIN_TABLE_NAME} (signature_hash, pattern_json)
                     VALUES (?1, ?2)
                     ON CONFLICT(signature_hash) DO UPDATE SET
                        pattern_json = excluded.pattern_json"
                ),
                params![signature_hash, pattern_json],
            )
            .is_err()
        {
            return false;
        }
    }
    true
}

fn stable_signature_hash(tool_name: &str) -> String {
    let mut hasher = DefaultHasher::new();
    tool_name.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn record_ranked_value(map: &mut BTreeMap<String, u64>, value: &str, limit: usize) {
    *map.entry(value.to_string()).or_insert(0) += 1;
    if map.len() > limit {
        let mut items = map
            .iter()
            .map(|(key, count)| (key.clone(), *count))
            .collect::<Vec<_>>();
        items.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        *map = items.into_iter().take(limit).collect();
    }
}

fn top_keys(map: &BTreeMap<String, u64>, limit: usize) -> Vec<String> {
    let mut items = map
        .iter()
        .map(|(key, count)| (key.clone(), *count))
        .collect::<Vec<_>>();
    items.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    items.into_iter().take(limit).map(|(key, _)| key).collect()
}

fn anonymize_query_pattern(query: &str) -> String {
    query
        .split_whitespace()
        .map(|token| {
            if token.contains(':') {
                token.split(':').next().unwrap_or(token)
            } else if token.contains('=') {
                token.split('=').next().unwrap_or(token)
            } else if token.chars().any(|ch| ch.is_ascii_digit()) {
                "<num>"
            } else {
                token
            }
        })
        .take(12)
        .collect::<Vec<_>>()
        .join(" ")
}

fn extract_query_fields(query: &str) -> Vec<String> {
    let common_fields = [
        "id", "name", "status", "error", "message", "type", "code", "result", "value", "data",
        "items", "count",
    ];
    let mut fields = Vec::new();
    for token in query.split_whitespace() {
        for separator in [':', '='] {
            if let Some((field, _)) = token.split_once(separator) {
                let cleaned = field
                    .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-');
                if !cleaned.is_empty() && !fields.iter().any(|existing| existing == cleaned) {
                    fields.push(cleaned.to_string());
                }
            }
        }
    }
    let lower = query.to_ascii_lowercase();
    for field in common_fields {
        if lower.contains(field) && !fields.iter().any(|existing| existing == field) {
            fields.push(field.to_string());
        }
    }
    fields
}

fn update_patterns_for_retrieval(
    inner: &mut ProductState,
    tool_name: Option<&str>,
    query: Option<&str>,
    retrieval_type: &'static str,
) {
    let Some(tool_name) = tool_name else {
        return;
    };
    let feedback = inner
        .feedback_patterns
        .entry(tool_name.to_string())
        .or_default();
    feedback.total_retrievals += 1;
    match retrieval_type {
        "full" => feedback.full_retrievals += 1,
        "search" => feedback.search_retrievals += 1,
        _ => {}
    }
    if let Some(query) = query.filter(|query| !query.trim().is_empty()) {
        record_ranked_value(
            &mut feedback.common_queries,
            &query.to_ascii_lowercase(),
            100,
        );
        for field in extract_query_fields(query) {
            record_ranked_value(&mut feedback.queried_fields, &field, 50);
        }
    }

    let signature_hash = stable_signature_hash(tool_name);
    let toin = inner
        .toin_patterns
        .entry(signature_hash.clone())
        .or_insert_with(|| ToinPattern {
            tool_signature_hash: signature_hash,
            ..ToinPattern::default()
        });
    toin.total_retrievals += 1;
    match retrieval_type {
        "full" => toin.full_retrievals += 1,
        "search" => toin.search_retrievals += 1,
        _ => {}
    }
    if let Some(query) = query.filter(|query| !query.trim().is_empty()) {
        record_ranked_value(
            &mut toin.common_query_patterns,
            &anonymize_query_pattern(query),
            20,
        );
        for field in extract_query_fields(query) {
            record_ranked_value(&mut toin.field_retrieval_frequency, &field, 100);
        }
    }
    toin.update_recommendations();
}

fn compute_hint_max_items(pattern: &FeedbackPattern) -> u64 {
    let retrieval_rate = pattern.retrieval_rate();
    if retrieval_rate > HIGH_RETRIEVAL_THRESHOLD {
        if pattern.full_retrieval_rate() > 0.8 {
            50
        } else {
            50
        }
    } else if retrieval_rate > MEDIUM_RETRIEVAL_THRESHOLD {
        30
    } else {
        DEFAULT_CCR_HINT_MAX_ITEMS
    }
}

fn compute_hint_aggressiveness(pattern: &FeedbackPattern) -> f64 {
    let retrieval_rate = pattern.retrieval_rate();
    if retrieval_rate > HIGH_RETRIEVAL_THRESHOLD {
        0.3
    } else if retrieval_rate > MEDIUM_RETRIEVAL_THRESHOLD {
        0.5
    } else {
        DEFAULT_CCR_HINT_AGGRESSIVENESS
    }
}

fn compute_hint_reason(pattern: &FeedbackPattern) -> String {
    let retrieval_rate = pattern.retrieval_rate();
    if retrieval_rate > HIGH_RETRIEVAL_THRESHOLD {
        if pattern.full_retrieval_rate() > 0.8 {
            format!(
                "Very high full retrieval rate ({:.0}%), recommending skip compression",
                pattern.full_retrieval_rate() * 100.0
            )
        } else {
            format!(
                "High retrieval rate ({:.0}%), recommending less aggressive compression",
                retrieval_rate * 100.0
            )
        }
    } else if retrieval_rate > MEDIUM_RETRIEVAL_THRESHOLD {
        format!(
            "Medium retrieval rate ({:.0}%), recommending moderate compression",
            retrieval_rate * 100.0
        )
    } else {
        format!(
            "Low retrieval rate ({:.0}%), current compression is effective",
            retrieval_rate * 100.0
        )
    }
}

impl Default for ProductStore {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use rusqlite::{params, Connection};

    use super::{
        ProductStore, PRODUCT_CCR_TABLE_NAME, PRODUCT_FEEDBACK_TABLE_NAME,
        PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME, PRODUCT_TOIN_TABLE_NAME,
    };
    use crate::state_store::sqlite_backend;
    use headroom_core::ccr::CcrStore;

    #[test]
    fn product_store_persists_feedback_and_toin_state() {
        let unique = format!(
            "headroom-product-store-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store = ProductStore::new(Some(path.clone()));
        store.put("abc123", "[{\"status\":\"error\"}]");
        store.record_compression(Some("abc123"), "search", 10, 4, Some("status:error"));
        let _ = store.search("abc123", "status:error");
        drop(store);

        let reloaded = ProductStore::new(Some(path.clone()));
        let feedback = reloaded.feedback_for_tool("search");
        assert_eq!(
            feedback["pattern"]["total_compressions"],
            serde_json::json!(1)
        );
        assert_eq!(
            feedback["pattern"]["total_retrievals"],
            serde_json::json!(1)
        );

        let stats = reloaded.toin_stats();
        assert_eq!(stats["patterns_tracked"], serde_json::json!(1));
        assert_eq!(stats["total_retrievals"], serde_json::json!(1));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn product_store_normalizes_sqlite_state_into_tables() {
        let unique = format!(
            "headroom-product-store-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store = ProductStore::new_with_backend(Some(sqlite_backend(path.clone(), "product")));
        store.put("abc123", "[{\"status\":\"error\"}]");
        store.record_compression(Some("abc123"), "search", 10, 4, Some("status:error"));
        let _ = store.search("abc123", "status:error");
        drop(store);

        let connection = Connection::open(&path).unwrap();
        let ccr_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {PRODUCT_CCR_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let retrieval_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let feedback_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {PRODUCT_FEEDBACK_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let toin_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {PRODUCT_TOIN_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(ccr_count, 1);
        assert_eq!(retrieval_count, 1);
        assert_eq!(feedback_count, 1);
        assert_eq!(toin_count, 1);

        let reloaded =
            ProductStore::new_with_backend(Some(sqlite_backend(path.clone(), "product")));
        let feedback = reloaded.feedback_for_tool("search");
        assert_eq!(
            feedback["pattern"]["total_compressions"],
            serde_json::json!(1)
        );
        assert_eq!(
            feedback["pattern"]["total_retrievals"],
            serde_json::json!(1)
        );

        let stats = reloaded.toin_stats();
        assert_eq!(stats["patterns_tracked"], serde_json::json!(1));
        assert_eq!(stats["total_retrievals"], serde_json::json!(1));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn product_store_updates_sqlite_rows_incrementally_for_live_events() {
        let unique = format!(
            "headroom-product-live-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store = ProductStore::new_with_backend(Some(sqlite_backend(path.clone(), "product")));
        store.put("abc123", "[{\"status\":\"error\"}]");
        store.record_compression(Some("abc123"), "search", 10, 4, Some("status:error"));
        let _ = store.retrieve("abc123");
        let _ = store.search("abc123", "status:error");
        drop(store);

        let connection = Connection::open(&path).unwrap();
        let ccr_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {PRODUCT_CCR_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let retrieval_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let feedback_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {PRODUCT_FEEDBACK_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let toin_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {PRODUCT_TOIN_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(ccr_count, 1);
        assert_eq!(retrieval_count, 2);
        assert_eq!(feedback_count, 1);
        assert_eq!(toin_count, 1);

        let reloaded =
            ProductStore::new_with_backend(Some(sqlite_backend(path.clone(), "product")));
        let stats = reloaded.ccr_stats();
        assert_eq!(stats["store"]["entry_count"], serde_json::json!(1));
        assert_eq!(stats["store"]["event_count"], serde_json::json!(2));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn product_store_reads_stats_and_patterns_directly_from_sqlite() {
        let unique = format!(
            "headroom-product-query-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store = ProductStore::new_with_backend(Some(sqlite_backend(path.clone(), "product")));
        store.put("abc123", "[{\"status\":\"error\"}]");
        store.record_compression(Some("abc123"), "search", 10, 4, Some("status:error"));

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_CCR_TABLE_NAME} (
                        hash, original_content_json, original_tokens, original_item_count,
                        compressed_item_count, tool_name, retrieval_count
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                     ON CONFLICT(hash) DO UPDATE SET
                        original_content_json = excluded.original_content_json,
                        original_tokens = excluded.original_tokens,
                        original_item_count = excluded.original_item_count,
                        compressed_item_count = excluded.compressed_item_count,
                        tool_name = excluded.tool_name,
                        retrieval_count = excluded.retrieval_count"
                ),
                params![
                    "external-hash",
                    serde_json::to_string(&serde_json::json!([{"service":"billing"}])).unwrap(),
                    50_i64,
                    5_i64,
                    2_i64,
                    "external-tool",
                    4_i64
                ],
            )
            .unwrap();
        connection
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_RETRIEVAL_EVENTS_TABLE_NAME} (
                        hash, query, items_retrieved, total_items, tool_name, retrieval_type
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
                ),
                params![
                    "external-hash",
                    "service:billing",
                    2_i64,
                    5_i64,
                    "external-tool",
                    "search"
                ],
            )
            .unwrap();
        connection
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_FEEDBACK_TABLE_NAME} (tool_name, pattern_json)
                     VALUES (?1, ?2)
                     ON CONFLICT(tool_name) DO UPDATE SET
                        pattern_json = excluded.pattern_json"
                ),
                params![
                    "external-tool",
                    serde_json::to_string(&serde_json::json!({
                        "total_compressions": 3,
                        "total_retrievals": 2,
                        "full_retrievals": 1,
                        "search_retrievals": 1,
                        "common_queries": {"service:billing": 2},
                        "queried_fields": {"service": 2}
                    }))
                    .unwrap()
                ],
            )
            .unwrap();
        connection
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_TOIN_TABLE_NAME} (signature_hash, pattern_json)
                     VALUES (?1, ?2)
                     ON CONFLICT(signature_hash) DO UPDATE SET
                        pattern_json = excluded.pattern_json"
                ),
                params![
                    "deadbeefcafebabe",
                    serde_json::to_string(&serde_json::json!({
                        "tool_signature_hash": "deadbeefcafebabe",
                        "total_compressions": 3,
                        "total_items_seen": 20,
                        "total_items_kept": 8,
                        "total_retrievals": 2,
                        "full_retrievals": 1,
                        "search_retrievals": 1,
                        "field_retrieval_frequency": {"service": 2},
                        "common_query_patterns": {"service:?": 2},
                        "optimal_max_items": 30,
                        "skip_compression_recommended": false,
                        "preserve_fields": ["service"],
                        "sample_size": 12,
                        "confidence": 0.9
                    }))
                    .unwrap()
                ],
            )
            .unwrap();

        let ccr_stats = store.ccr_stats();
        assert_eq!(ccr_stats["store"]["entry_count"], serde_json::json!(2));
        assert_eq!(ccr_stats["store"]["event_count"], serde_json::json!(1));
        assert_eq!(
            ccr_stats["recent_retrievals"][0]["tool_name"],
            serde_json::json!("external-tool")
        );
        assert_eq!(
            store.tool_name_for_hash("external-hash"),
            Some("external-tool".to_string())
        );
        assert_eq!(store.len(), 2);

        let feedback_stats = store.feedback_stats();
        assert_eq!(
            feedback_stats["feedback"]["tools_tracked"],
            serde_json::json!(2)
        );
        assert!(feedback_stats["feedback"]["tool_patterns"]
            .get("external-tool")
            .is_some());

        let feedback = store.feedback_for_tool("external-tool");
        assert_eq!(
            feedback["pattern"]["total_compressions"],
            serde_json::json!(3)
        );
        assert_eq!(
            feedback["hints"]["preserve_fields"][0],
            serde_json::json!("service")
        );

        let toin_stats = store.toin_stats();
        assert_eq!(toin_stats["patterns_tracked"], serde_json::json!(2));
        assert_eq!(
            toin_stats["patterns_with_recommendations"],
            serde_json::json!(1)
        );

        let patterns = store.toin_patterns(10);
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0]["hash"], serde_json::json!("deadbeefcafe"));

        let detail = store.toin_pattern_detail("deadbeef").unwrap();
        assert_eq!(
            detail["tool_signature_hash"],
            serde_json::json!("deadbeefcafebabe")
        );
        assert_eq!(detail["optimal_max_items"], serde_json::json!(30));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn product_store_reloads_sqlite_state_before_live_retrievals() {
        let unique = format!(
            "headroom-product-reload-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store = ProductStore::new_with_backend(Some(sqlite_backend(path.clone(), "product")));

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {PRODUCT_CCR_TABLE_NAME} (
                        hash TEXT PRIMARY KEY,
                        original_content_json TEXT NOT NULL,
                        original_tokens INTEGER NOT NULL,
                        original_item_count INTEGER NOT NULL,
                        compressed_item_count INTEGER NOT NULL,
                        tool_name TEXT,
                        retrieval_count INTEGER NOT NULL
                    )"
                ),
                [],
            )
            .unwrap();
        connection
            .execute(
                &format!(
                    "INSERT INTO {PRODUCT_CCR_TABLE_NAME} (
                        hash, original_content_json, original_tokens, original_item_count,
                        compressed_item_count, tool_name, retrieval_count
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
                ),
                params![
                    "late-hash",
                    serde_json::to_string(&serde_json::json!([{"service":"billing"}])).unwrap(),
                    50_i64,
                    5_i64,
                    2_i64,
                    "external-tool",
                    0_i64
                ],
            )
            .unwrap();

        let retrieved = store.retrieve("late-hash").unwrap();
        assert_eq!(retrieved["tool_name"], serde_json::json!("external-tool"));
        assert_eq!(retrieved["retrieval_count"], serde_json::json!(1));

        let reloaded =
            ProductStore::new_with_backend(Some(sqlite_backend(path.clone(), "product")));
        let stats = reloaded.ccr_stats();
        assert_eq!(stats["store"]["entry_count"], serde_json::json!(1));
        assert_eq!(stats["store"]["event_count"], serde_json::json!(1));
        assert_eq!(
            reloaded.feedback_for_tool("external-tool")["pattern"]["total_retrievals"],
            serde_json::json!(1)
        );

        let _ = std::fs::remove_file(path);
    }
}
