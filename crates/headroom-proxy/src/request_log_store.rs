use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::Value;

const MAX_REQUEST_LOG_ENTRIES: usize = 10_000;

#[derive(Clone)]
pub struct RequestLogStore {
    inner: Arc<Mutex<RequestLogState>>,
}

struct RequestLogState {
    entries: VecDeque<Value>,
    path: Option<PathBuf>,
}

impl Default for RequestLogStore {
    fn default() -> Self {
        Self::new(None)
    }
}

impl RequestLogStore {
    pub fn new(path: Option<PathBuf>) -> Self {
        let entries = path
            .as_ref()
            .and_then(load_entries)
            .unwrap_or_default();
        Self {
            inner: Arc::new(Mutex::new(RequestLogState { entries, path })),
        }
    }

    pub fn log(&self, entry: Value) {
        let mut inner = self.inner.lock().expect("request log mutex poisoned");
        if inner.entries.len() == MAX_REQUEST_LOG_ENTRIES {
            inner.entries.pop_front();
        }
        inner.entries.push_back(entry);
        persist_entries(&inner.entries, inner.path.as_ref());
    }

    pub fn recent(&self, limit: usize) -> Vec<Value> {
        self.recent_internal(limit, false)
    }

    pub fn recent_with_messages(&self, limit: usize) -> Vec<Value> {
        self.recent_internal(limit, true)
    }

    pub fn snapshot(&self) -> Vec<Value> {
        let inner = self.inner.lock().expect("request log mutex poisoned");
        inner.entries.iter().cloned().collect()
    }

    pub fn storage_path(&self) -> Option<PathBuf> {
        let inner = self.inner.lock().expect("request log mutex poisoned");
        inner.path.clone()
    }

    fn recent_internal(&self, limit: usize, include_messages: bool) -> Vec<Value> {
        let inner = self.inner.lock().expect("request log mutex poisoned");
        let mut entries = inner
            .entries
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect::<Vec<_>>();
        entries.reverse();

        if include_messages {
            return entries;
        }

        entries
            .into_iter()
            .map(|mut entry| {
                if let Some(object) = entry.as_object_mut() {
                    object.remove("request_messages");
                    object.remove("response_content");
                }
                entry
            })
            .collect()
    }
}

fn load_entries(path: &PathBuf) -> Option<VecDeque<Value>> {
    let raw = fs::read_to_string(path).ok()?;
    let parsed = serde_json::from_str::<Vec<Value>>(&raw).ok()?;
    let mut entries = parsed.into_iter().collect::<VecDeque<_>>();
    while entries.len() > MAX_REQUEST_LOG_ENTRIES {
        entries.pop_front();
    }
    Some(entries)
}

fn persist_entries(entries: &VecDeque<Value>, path: Option<&PathBuf>) {
    let Some(path) = path else {
        return;
    };
    if let Some(parent) = path.parent() {
        if fs::create_dir_all(parent).is_err() {
            return;
        }
    }
    let payload = entries.iter().cloned().collect::<Vec<_>>();
    let Ok(serialized) = serde_json::to_string_pretty(&payload) else {
        return;
    };
    let tmp_path = path.with_extension("tmp");
    if fs::write(&tmp_path, serialized).is_ok() {
        if path.exists() {
            let _ = fs::remove_file(path);
        }
        let _ = fs::rename(&tmp_path, path);
    }
}

#[cfg(test)]
mod tests {
    use super::RequestLogStore;

    #[test]
    fn request_log_store_persists_entries_when_path_is_configured() {
        let unique = format!(
            "headroom-request-log-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store = RequestLogStore::new(Some(path.clone()));
        store.log(serde_json::json!({
            "timestamp": "2026-04-30T12:00:00Z",
            "provider": "headroom",
            "tokens_saved": 25,
        }));
        drop(store);

        let reloaded = RequestLogStore::new(Some(path.clone()));
        let snapshot = reloaded.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0]["provider"], serde_json::json!("headroom"));
        assert_eq!(snapshot[0]["tokens_saved"], serde_json::json!(25));

        let _ = std::fs::remove_file(path);
    }
}
