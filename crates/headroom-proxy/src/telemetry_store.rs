use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::state_store::{
    backend_details, file_backend, load_json_state, persist_json_state, SharedStateBackend,
};

#[derive(Clone)]
pub struct TelemetryStore {
    inner: Arc<Mutex<TelemetryState>>,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct TelemetryState {
    total_compressions: u64,
    total_retrievals: u64,
    total_tokens_saved: i64,
    tool_stats: BTreeMap<String, Value>,
    recommendations: BTreeMap<String, Value>,
    #[serde(skip)]
    backend: Option<SharedStateBackend>,
}

const TELEMETRY_SUMMARY_TABLE_NAME: &str = "telemetry_summary";
const TELEMETRY_TOOL_STATS_TABLE_NAME: &str = "telemetry_tool_stats";
const TELEMETRY_RECOMMENDATIONS_TABLE_NAME: &str = "telemetry_recommendations";

impl TelemetryStore {
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

    pub fn stats(&self) -> Value {
        let inner = self.readable_state();
        json!({
            "enabled": telemetry_enabled(),
            "total_compressions": inner.total_compressions,
            "total_retrievals": inner.total_retrievals,
            "total_tokens_saved": inner.total_tokens_saved,
            "global_retrieval_rate": if inner.total_compressions > 0 {
                inner.total_retrievals as f64 / inner.total_compressions as f64
            } else {
                0.0
            },
            "tool_signatures_tracked": inner.tool_stats.len(),
            "events_in_memory": 0,
            "avg_compression_ratio": average_numeric_field(inner.tool_stats.values(), "avg_compression_ratio"),
            "avg_token_reduction": average_numeric_field(inner.tool_stats.values(), "avg_token_reduction"),
        })
    }

    pub fn backend_info(&self) -> Value {
        let inner = self.readable_state();
        backend_details(inner.backend.as_ref(), inner.tool_stats.len())
    }

    pub fn export_stats(&self) -> Value {
        let inner = self.readable_state();
        let mut export = json!({
            "version": "1.0",
            "export_timestamp": unix_timestamp_seconds(),
            "summary": {
                "total_compressions": inner.total_compressions,
                "total_retrievals": inner.total_retrievals,
                "total_tokens_saved": inner.total_tokens_saved,
                "tool_signatures_tracked": inner.tool_stats.len(),
            },
            "tool_stats": inner.tool_stats,
        });
        if !inner.recommendations.is_empty() {
            export["recommendations"] = json!(inner.recommendations);
        }
        export
    }

    pub fn import_stats(&self, data: &Value) -> Value {
        let mut inner = self.inner.lock().expect("telemetry mutex poisoned");
        let summary = data.get("summary").and_then(Value::as_object);
        inner.total_compressions += summary
            .and_then(|s| s.get("total_compressions"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        inner.total_retrievals += summary
            .and_then(|s| s.get("total_retrievals"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        inner.total_tokens_saved += summary
            .and_then(|s| s.get("total_tokens_saved"))
            .and_then(Value::as_i64)
            .unwrap_or(0);

        if let Some(tool_stats) = data.get("tool_stats").and_then(Value::as_object) {
            for (signature_hash, stats) in tool_stats {
                inner
                    .tool_stats
                    .insert(signature_hash.clone(), stats.clone());
            }
        }

        if let Some(recommendations) = data.get("recommendations").and_then(Value::as_object) {
            for (signature_hash, recommendation) in recommendations {
                inner
                    .recommendations
                    .insert(signature_hash.clone(), recommendation.clone());
            }
        }

        persist_state(&inner);
        drop(inner);
        self.stats()
    }

    pub fn record_compression(
        &self,
        tool_name: &str,
        original_items: u64,
        compressed_items: u64,
        original_tokens: i64,
        compressed_tokens: i64,
    ) {
        let mut inner = self.inner.lock().expect("telemetry mutex poisoned");
        inner.total_compressions += 1;
        inner.total_tokens_saved += (original_tokens - compressed_tokens).max(0);

        let signature_hash = stable_signature_hash(tool_name);
        let stats_snapshot = {
            let stats = inner
                .tool_stats
                .entry(signature_hash.clone())
                .or_insert_with(|| json!({}));
            let sample_size = stats
                .get("sample_size")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                + 1;
            let retrievals = stats.get("retrievals").and_then(Value::as_u64).unwrap_or(0);
            let avg_compression_ratio = update_average(
                stats
                    .get("avg_compression_ratio")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                sample_size,
                if original_items > 0 {
                    compressed_items as f64 / original_items as f64
                } else {
                    1.0
                },
            );
            let avg_token_reduction = update_average(
                stats
                    .get("avg_token_reduction")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                sample_size,
                if original_tokens > 0 {
                    1.0 - (compressed_tokens as f64 / original_tokens as f64)
                } else {
                    0.0
                },
            );
            *stats = json!({
                "tool_name": tool_name,
                "sample_size": sample_size,
                "compressions": stats.get("compressions").and_then(Value::as_u64).unwrap_or(0) + 1,
                "retrievals": retrievals,
                "full_retrievals": stats.get("full_retrievals").and_then(Value::as_u64).unwrap_or(0),
                "search_retrievals": stats.get("search_retrievals").and_then(Value::as_u64).unwrap_or(0),
                "avg_compression_ratio": avg_compression_ratio,
                "avg_token_reduction": avg_token_reduction,
            });
            stats.clone()
        };
        let recommendation = build_recommendation(&stats_snapshot);
        inner
            .recommendations
            .insert(signature_hash.clone(), recommendation.clone());
        if let Some(path) = sqlite_storage_path(inner.backend.as_ref()) {
            let _ = persist_summary_and_tool_to_sqlite(
                &inner,
                &path,
                &signature_hash,
                &stats_snapshot,
                Some(&recommendation),
            );
        } else {
            persist_state(&inner);
        }
    }

    pub fn record_retrieval(&self, tool_name: &str, retrieval_type: &str) {
        let mut inner = self.inner.lock().expect("telemetry mutex poisoned");
        inner.total_retrievals += 1;

        let signature_hash = stable_signature_hash(tool_name);
        let stats_snapshot = {
            let stats = inner
                .tool_stats
                .entry(signature_hash.clone())
                .or_insert_with(|| json!({}));
            let full_retrievals = stats
                .get("full_retrievals")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                + u64::from(retrieval_type == "full");
            let search_retrievals = stats
                .get("search_retrievals")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                + u64::from(retrieval_type == "search");
            *stats = json!({
                "tool_name": stats.get("tool_name").and_then(Value::as_str).unwrap_or(tool_name),
                "sample_size": stats.get("sample_size").and_then(Value::as_u64).unwrap_or(0),
                "compressions": stats.get("compressions").and_then(Value::as_u64).unwrap_or(0),
                "retrievals": stats.get("retrievals").and_then(Value::as_u64).unwrap_or(0) + 1,
                "full_retrievals": full_retrievals,
                "search_retrievals": search_retrievals,
                "avg_compression_ratio": stats.get("avg_compression_ratio").and_then(Value::as_f64).unwrap_or(0.0),
                "avg_token_reduction": stats.get("avg_token_reduction").and_then(Value::as_f64).unwrap_or(0.0),
            });
            stats.clone()
        };
        let recommendation = build_recommendation(&stats_snapshot);
        inner
            .recommendations
            .insert(signature_hash.clone(), recommendation.clone());
        if let Some(path) = sqlite_storage_path(inner.backend.as_ref()) {
            let _ = persist_summary_and_tool_to_sqlite(
                &inner,
                &path,
                &signature_hash,
                &stats_snapshot,
                Some(&recommendation),
            );
        } else {
            persist_state(&inner);
        }
    }

    pub fn tool_list(&self) -> Value {
        let inner = self.readable_state();
        json!({
            "tool_count": inner.tool_stats.len(),
            "tools": inner.tool_stats,
        })
    }

    pub fn tool_detail(&self, signature_hash: &str) -> Option<Value> {
        let inner = self.readable_state();
        let stats = inner.tool_stats.get(signature_hash)?.clone();
        Some(json!({
            "signature_hash": signature_hash,
            "stats": stats,
            "recommendations": inner.recommendations.get(signature_hash).cloned(),
        }))
    }

    fn readable_state(&self) -> TelemetryState {
        let fallback = { self.inner.lock().expect("telemetry mutex poisoned").clone() };
        if let Some(path) = sqlite_storage_path(fallback.backend.as_ref()) {
            let mut state = load_state_from_sqlite(&path).unwrap_or_default();
            state.backend = fallback.backend;
            state
        } else {
            fallback
        }
    }
}

pub fn telemetry_enabled() -> bool {
    telemetry_enabled_from_value(std::env::var("HEADROOM_TELEMETRY").ok().as_deref())
}

fn telemetry_enabled_from_value(value: Option<&str>) -> bool {
    !matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("off" | "false" | "0")
    )
}

fn update_average(current_avg: f64, sample_size: u64, value: f64) -> f64 {
    if sample_size <= 1 {
        value
    } else {
        ((current_avg * (sample_size - 1) as f64) + value) / sample_size as f64
    }
}

fn stable_signature_hash(tool_name: &str) -> String {
    let mut hasher = DefaultHasher::new();
    tool_name.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn build_recommendation(stats: &Value) -> Value {
    let compressions = stats
        .get("compressions")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let retrievals = stats.get("retrievals").and_then(Value::as_u64).unwrap_or(0);
    let retrieval_rate = if compressions > 0 {
        retrievals as f64 / compressions as f64
    } else {
        0.0
    };
    let (recommended_max_items, aggressiveness) = if retrieval_rate > 0.5 {
        (50, 0.3)
    } else if retrieval_rate > 0.2 {
        (30, 0.5)
    } else {
        (15, 0.7)
    };
    json!({
        "confidence": ((compressions.min(20) as f64) / 20.0 * 100.0).round() / 100.0,
        "retrieval_rate": retrieval_rate,
        "recommended_max_items": recommended_max_items,
        "aggressiveness": aggressiveness,
        "skip_compression": retrieval_rate > 0.8
            && stats.get("full_retrievals").and_then(Value::as_u64).unwrap_or(0) >= retrievals.saturating_sub(1),
    })
}

fn average_numeric_field<'a>(values: impl Iterator<Item = &'a Value>, field: &str) -> f64 {
    let mut total = 0.0;
    let mut count = 0.0;
    for value in values {
        if let Some(number) = value.get(field).and_then(Value::as_f64) {
            total += number;
            count += 1.0;
        }
    }
    if count > 0.0 {
        total / count
    } else {
        0.0
    }
}

fn unix_timestamp_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn load_state(backend: Option<&SharedStateBackend>) -> Option<TelemetryState> {
    if let Some(path) = sqlite_storage_path(backend) {
        return load_state_from_sqlite(&path);
    }
    load_json_state(backend)
}

fn persist_state(state: &TelemetryState) {
    if let Some(path) = sqlite_storage_path(state.backend.as_ref()) {
        let _ = persist_state_to_sqlite(state, &path);
        return;
    }
    persist_json_state(state, state.backend.as_ref());
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
    if connection
        .execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {TELEMETRY_SUMMARY_TABLE_NAME} (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    total_compressions INTEGER NOT NULL,
                    total_retrievals INTEGER NOT NULL,
                    total_tokens_saved INTEGER NOT NULL
                )"
            ),
            [],
        )
        .is_err()
    {
        return None;
    }
    if connection
        .execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {TELEMETRY_TOOL_STATS_TABLE_NAME} (
                    signature_hash TEXT PRIMARY KEY,
                    stats_json TEXT NOT NULL
                )"
            ),
            [],
        )
        .is_err()
    {
        return None;
    }
    if connection
        .execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {TELEMETRY_RECOMMENDATIONS_TABLE_NAME} (
                    signature_hash TEXT PRIMARY KEY,
                    recommendation_json TEXT NOT NULL
                )"
            ),
            [],
        )
        .is_err()
    {
        return None;
    }
    Some(connection)
}

fn load_state_from_sqlite(path: &PathBuf) -> Option<TelemetryState> {
    let connection = connect_sqlite(path)?;
    let (total_compressions, total_retrievals, total_tokens_saved) = connection
        .query_row(
            &format!(
                "SELECT total_compressions, total_retrievals, total_tokens_saved
                 FROM {TELEMETRY_SUMMARY_TABLE_NAME}
                 WHERE id = 1"
            ),
            [],
            |row| {
                Ok((
                    row.get::<_, u64>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .ok()
        .unwrap_or((0, 0, 0));

    let mut tool_stats = BTreeMap::new();
    let mut stats_statement = connection
        .prepare(&format!(
            "SELECT signature_hash, stats_json FROM {TELEMETRY_TOOL_STATS_TABLE_NAME}"
        ))
        .ok()?;
    let stats_rows = stats_statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .ok()?;
    for row in stats_rows {
        let (signature_hash, raw_json) = row.ok()?;
        tool_stats.insert(
            signature_hash,
            serde_json::from_str::<Value>(&raw_json).ok()?,
        );
    }

    let mut recommendations = BTreeMap::new();
    let mut recommendations_statement = connection
        .prepare(&format!(
            "SELECT signature_hash, recommendation_json FROM {TELEMETRY_RECOMMENDATIONS_TABLE_NAME}"
        ))
        .ok()?;
    let recommendation_rows = recommendations_statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .ok()?;
    for row in recommendation_rows {
        let (signature_hash, raw_json) = row.ok()?;
        recommendations.insert(
            signature_hash,
            serde_json::from_str::<Value>(&raw_json).ok()?,
        );
    }

    Some(TelemetryState {
        total_compressions,
        total_retrievals,
        total_tokens_saved,
        tool_stats,
        recommendations,
        backend: None,
    })
}

fn persist_state_to_sqlite(state: &TelemetryState, path: &PathBuf) -> bool {
    let mut connection = match connect_sqlite(path) {
        Some(connection) => connection,
        None => return false,
    };
    let transaction = match connection.transaction() {
        Ok(transaction) => transaction,
        Err(_) => return false,
    };
    if transaction
        .execute(
            &format!(
                "INSERT INTO {TELEMETRY_SUMMARY_TABLE_NAME} (
                    id, total_compressions, total_retrievals, total_tokens_saved
                 ) VALUES (1, ?1, ?2, ?3)
                 ON CONFLICT(id) DO UPDATE SET
                    total_compressions = excluded.total_compressions,
                    total_retrievals = excluded.total_retrievals,
                    total_tokens_saved = excluded.total_tokens_saved"
            ),
            params![
                state.total_compressions as i64,
                state.total_retrievals as i64,
                state.total_tokens_saved
            ],
        )
        .is_err()
    {
        return false;
    }
    if transaction
        .execute(
            &format!("DELETE FROM {TELEMETRY_TOOL_STATS_TABLE_NAME}"),
            [],
        )
        .is_err()
    {
        return false;
    }
    for (signature_hash, stats) in &state.tool_stats {
        let Ok(serialized) = serde_json::to_string(stats) else {
            return false;
        };
        if transaction
            .execute(
                &format!(
                    "INSERT INTO {TELEMETRY_TOOL_STATS_TABLE_NAME} (signature_hash, stats_json)
                     VALUES (?1, ?2)"
                ),
                params![signature_hash, serialized],
            )
            .is_err()
        {
            return false;
        }
    }
    if transaction
        .execute(
            &format!("DELETE FROM {TELEMETRY_RECOMMENDATIONS_TABLE_NAME}"),
            [],
        )
        .is_err()
    {
        return false;
    }
    for (signature_hash, recommendation) in &state.recommendations {
        let Ok(serialized) = serde_json::to_string(recommendation) else {
            return false;
        };
        if transaction
            .execute(
                &format!(
                    "INSERT INTO {TELEMETRY_RECOMMENDATIONS_TABLE_NAME} (signature_hash, recommendation_json)
                     VALUES (?1, ?2)"
                ),
                params![signature_hash, serialized],
            )
            .is_err()
        {
            return false;
        }
    }
    transaction.commit().is_ok()
}

fn persist_summary_and_tool_to_sqlite(
    state: &TelemetryState,
    path: &PathBuf,
    signature_hash: &str,
    stats: &Value,
    recommendation: Option<&Value>,
) -> bool {
    let connection = match connect_sqlite(path) {
        Some(connection) => connection,
        None => return false,
    };
    if connection
        .execute(
            &format!(
                "INSERT INTO {TELEMETRY_SUMMARY_TABLE_NAME} (
                    id, total_compressions, total_retrievals, total_tokens_saved
                 ) VALUES (1, ?1, ?2, ?3)
                 ON CONFLICT(id) DO UPDATE SET
                    total_compressions = excluded.total_compressions,
                    total_retrievals = excluded.total_retrievals,
                    total_tokens_saved = excluded.total_tokens_saved"
            ),
            params![
                state.total_compressions as i64,
                state.total_retrievals as i64,
                state.total_tokens_saved
            ],
        )
        .is_err()
    {
        return false;
    }
    let Ok(stats_json) = serde_json::to_string(stats) else {
        return false;
    };
    if connection
        .execute(
            &format!(
                "INSERT INTO {TELEMETRY_TOOL_STATS_TABLE_NAME} (signature_hash, stats_json)
                 VALUES (?1, ?2)
                 ON CONFLICT(signature_hash) DO UPDATE SET
                    stats_json = excluded.stats_json"
            ),
            params![signature_hash, stats_json],
        )
        .is_err()
    {
        return false;
    }
    if let Some(recommendation) = recommendation {
        let Ok(recommendation_json) = serde_json::to_string(recommendation) else {
            return false;
        };
        if connection
            .execute(
                &format!(
                    "INSERT INTO {TELEMETRY_RECOMMENDATIONS_TABLE_NAME} (signature_hash, recommendation_json)
                     VALUES (?1, ?2)
                     ON CONFLICT(signature_hash) DO UPDATE SET
                        recommendation_json = excluded.recommendation_json"
                ),
                params![signature_hash, recommendation_json],
            )
            .is_err()
        {
            return false;
        }
    }
    true
}

impl Default for TelemetryStore {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use rusqlite::Connection;

    use super::*;
    use crate::state_store::sqlite_backend;

    #[test]
    fn import_and_detail_round_trip() {
        let store = TelemetryStore::default();
        let imported = store.import_stats(&json!({
            "summary": {
                "total_compressions": 3,
                "total_retrievals": 1,
                "total_tokens_saved": 42
            },
            "tool_stats": {
                "abc123": {
                    "sample_size": 12,
                    "avg_compression_ratio": 0.25,
                    "avg_token_reduction": 0.75
                }
            },
            "recommendations": {
                "abc123": {
                    "confidence": 0.8
                }
            }
        }));
        assert_eq!(imported["total_compressions"], 3);
        assert_eq!(store.tool_list()["tool_count"], 1);
        assert_eq!(
            store.tool_detail("abc123").unwrap()["recommendations"]["confidence"],
            0.8
        );
    }

    #[test]
    fn telemetry_enabled_respects_opt_out_values() {
        assert!(telemetry_enabled_from_value(None));
        assert!(telemetry_enabled_from_value(Some("on")));
        assert!(!telemetry_enabled_from_value(Some("off")));
        assert!(!telemetry_enabled_from_value(Some("FALSE")));
        assert!(!telemetry_enabled_from_value(Some("0")));
    }

    #[test]
    fn telemetry_store_persists_imported_state() {
        let unique = format!(
            "headroom-telemetry-store-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store = TelemetryStore::new(Some(path.clone()));
        store.import_stats(&json!({
            "summary": {
                "total_compressions": 2,
                "total_retrievals": 1,
                "total_tokens_saved": 12
            },
            "tool_stats": {
                "persisted-tool": {
                    "sample_size": 4,
                    "avg_compression_ratio": 0.5
                }
            }
        }));
        drop(store);

        let reloaded = TelemetryStore::new(Some(path.clone()));
        assert_eq!(reloaded.stats()["total_compressions"], serde_json::json!(2));
        assert_eq!(reloaded.tool_list()["tool_count"], serde_json::json!(1));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn telemetry_store_normalizes_sqlite_summary_and_tool_rows() {
        let unique = format!(
            "headroom-telemetry-store-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store =
            TelemetryStore::new_with_backend(Some(sqlite_backend(path.clone(), "telemetry")));
        store.import_stats(&json!({
            "summary": {
                "total_compressions": 2,
                "total_retrievals": 1,
                "total_tokens_saved": 12
            },
            "tool_stats": {
                "persisted-tool": {
                    "sample_size": 4,
                    "avg_compression_ratio": 0.5
                }
            },
            "recommendations": {
                "persisted-tool": {
                    "confidence": 0.9
                }
            }
        }));
        drop(store);

        let connection = Connection::open(&path).unwrap();
        let summary_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {TELEMETRY_SUMMARY_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let stats_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {TELEMETRY_TOOL_STATS_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let recommendations_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {TELEMETRY_RECOMMENDATIONS_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(summary_count, 1);
        assert_eq!(stats_count, 1);
        assert_eq!(recommendations_count, 1);

        let reloaded =
            TelemetryStore::new_with_backend(Some(sqlite_backend(path.clone(), "telemetry")));
        assert_eq!(reloaded.stats()["total_compressions"], serde_json::json!(2));
        assert_eq!(reloaded.tool_list()["tool_count"], serde_json::json!(1));
        assert_eq!(
            reloaded.tool_detail("persisted-tool").unwrap()["recommendations"]["confidence"],
            0.9
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn telemetry_store_updates_sqlite_rows_incrementally_for_live_events() {
        let unique = format!(
            "headroom-telemetry-live-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store =
            TelemetryStore::new_with_backend(Some(sqlite_backend(path.clone(), "telemetry")));
        store.record_compression("search", 10, 4, 100, 40);
        store.record_retrieval("search", "search");

        let connection = Connection::open(&path).unwrap();
        let summary_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {TELEMETRY_SUMMARY_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let stats_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {TELEMETRY_TOOL_STATS_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let recommendations_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {TELEMETRY_RECOMMENDATIONS_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(summary_count, 1);
        assert_eq!(stats_count, 1);
        assert_eq!(recommendations_count, 1);

        let reloaded =
            TelemetryStore::new_with_backend(Some(sqlite_backend(path.clone(), "telemetry")));
        assert_eq!(reloaded.stats()["total_compressions"], serde_json::json!(1));
        assert_eq!(reloaded.stats()["total_retrievals"], serde_json::json!(1));
        assert_eq!(reloaded.tool_list()["tool_count"], serde_json::json!(1));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn telemetry_store_reads_stats_and_details_directly_from_sqlite() {
        let unique = format!(
            "headroom-telemetry-query-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store =
            TelemetryStore::new_with_backend(Some(sqlite_backend(path.clone(), "telemetry")));
        store.record_compression("search", 10, 4, 100, 40);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                &format!(
                    "INSERT INTO {TELEMETRY_SUMMARY_TABLE_NAME} (
                        id, total_compressions, total_retrievals, total_tokens_saved
                     ) VALUES (1, ?1, ?2, ?3)
                     ON CONFLICT(id) DO UPDATE SET
                        total_compressions = excluded.total_compressions,
                        total_retrievals = excluded.total_retrievals,
                        total_tokens_saved = excluded.total_tokens_saved"
                ),
                params![7_i64, 3_i64, 222_i64],
            )
            .unwrap();
        connection
            .execute(
                &format!(
                    "INSERT INTO {TELEMETRY_TOOL_STATS_TABLE_NAME} (signature_hash, stats_json)
                     VALUES (?1, ?2)
                     ON CONFLICT(signature_hash) DO UPDATE SET
                        stats_json = excluded.stats_json"
                ),
                params![
                    "external-tool",
                    serde_json::to_string(&json!({
                        "tool_name": "external",
                        "sample_size": 2,
                        "compressions": 2,
                        "retrievals": 1,
                        "full_retrievals": 1,
                        "search_retrievals": 0,
                        "avg_compression_ratio": 0.4,
                        "avg_token_reduction": 0.6
                    }))
                    .unwrap()
                ],
            )
            .unwrap();
        connection
            .execute(
                &format!(
                    "INSERT INTO {TELEMETRY_RECOMMENDATIONS_TABLE_NAME} (signature_hash, recommendation_json)
                     VALUES (?1, ?2)
                     ON CONFLICT(signature_hash) DO UPDATE SET
                        recommendation_json = excluded.recommendation_json"
                ),
                params![
                    "external-tool",
                    serde_json::to_string(&json!({
                        "confidence": 0.95,
                        "recommended_max_items": 50
                    }))
                    .unwrap()
                ],
            )
            .unwrap();

        let stats = store.stats();
        assert_eq!(stats["total_compressions"], json!(7));
        assert_eq!(stats["total_retrievals"], json!(3));
        assert_eq!(stats["total_tokens_saved"], json!(222));
        assert_eq!(stats["tool_signatures_tracked"], json!(2));

        let tool_list = store.tool_list();
        assert_eq!(tool_list["tool_count"], json!(2));
        assert!(tool_list["tools"].get("external-tool").is_some());

        let detail = store.tool_detail("external-tool").unwrap();
        assert_eq!(detail["stats"]["tool_name"], json!("external"));
        assert_eq!(detail["recommendations"]["confidence"], json!(0.95));

        let export = store.export_stats();
        assert_eq!(export["summary"]["total_compressions"], json!(7));
        assert_eq!(export["summary"]["tool_signatures_tracked"], json!(2));
        assert!(export["tool_stats"].get("external-tool").is_some());

        let _ = std::fs::remove_file(path);
    }
}
