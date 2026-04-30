use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

#[derive(Clone, Default)]
pub struct TelemetryStore {
    inner: Arc<Mutex<TelemetryState>>,
}

#[derive(Default)]
struct TelemetryState {
    total_compressions: u64,
    total_retrievals: u64,
    total_tokens_saved: i64,
    tool_stats: BTreeMap<String, Value>,
    recommendations: BTreeMap<String, Value>,
}

impl TelemetryStore {
    pub fn stats(&self) -> Value {
        let inner = self.inner.lock().expect("telemetry mutex poisoned");
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

    pub fn export_stats(&self) -> Value {
        let inner = self.inner.lock().expect("telemetry mutex poisoned");
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

        drop(inner);
        self.stats()
    }

    pub fn tool_list(&self) -> Value {
        let inner = self.inner.lock().expect("telemetry mutex poisoned");
        json!({
            "tool_count": inner.tool_stats.len(),
            "tools": inner.tool_stats,
        })
    }

    pub fn tool_detail(&self, signature_hash: &str) -> Option<Value> {
        let inner = self.inner.lock().expect("telemetry mutex poisoned");
        let stats = inner.tool_stats.get(signature_hash)?.clone();
        Some(json!({
            "signature_hash": signature_hash,
            "stats": stats,
            "recommendations": inner.recommendations.get(signature_hash).cloned(),
        }))
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
