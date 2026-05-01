use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde_json::Value;

use crate::state_store::{
    backend_details, file_backend, load_json_state, persist_json_state, SharedStateBackend,
};

const MAX_REQUEST_LOG_ENTRIES: usize = 10_000;
const REQUEST_LOG_TABLE_NAME: &str = "request_log_entries";

#[derive(Clone)]
pub struct RequestLogStore {
    inner: Arc<Mutex<RequestLogState>>,
}

struct RequestLogState {
    entries: VecDeque<Value>,
    backend: Option<SharedStateBackend>,
}

impl Default for RequestLogStore {
    fn default() -> Self {
        Self::new(None)
    }
}

impl RequestLogStore {
    pub fn new(path: Option<PathBuf>) -> Self {
        let backend = path.map(file_backend);
        Self::new_with_backend(backend)
    }

    pub fn new_with_backend(backend: Option<SharedStateBackend>) -> Self {
        let entries = load_entries(backend.as_ref()).unwrap_or_default();
        Self {
            inner: Arc::new(Mutex::new(RequestLogState { entries, backend })),
        }
    }

    pub fn log(&self, entry: Value) {
        let mut inner = self.inner.lock().expect("request log mutex poisoned");
        if inner.entries.len() == MAX_REQUEST_LOG_ENTRIES {
            inner.entries.pop_front();
        }
        if persist_entry(&entry, inner.backend.as_ref()) {
            inner.entries.push_back(entry);
            return;
        }
        inner.entries.push_back(entry);
        persist_entries(&inner.entries, inner.backend.as_ref());
    }

    pub fn recent(&self, limit: usize) -> Vec<Value> {
        self.recent_internal(limit, false)
    }

    pub fn recent_with_messages(&self, limit: usize) -> Vec<Value> {
        self.recent_internal(limit, true)
    }

    pub fn snapshot(&self) -> Vec<Value> {
        let inner = self.inner.lock().expect("request log mutex poisoned");
        if let Some(path) = sqlite_storage_path(inner.backend.as_ref()) {
            return load_entries_from_sqlite(&path)
                .unwrap_or_default()
                .into_iter()
                .collect();
        }
        inner.entries.iter().cloned().collect()
    }

    pub fn storage_path(&self) -> Option<PathBuf> {
        let inner = self.inner.lock().expect("request log mutex poisoned");
        inner
            .backend
            .as_ref()
            .and_then(|backend| backend.storage_path())
    }

    pub fn backend_info(&self) -> Value {
        let inner = self.inner.lock().expect("request log mutex poisoned");
        backend_details(inner.backend.as_ref(), inner.entries.len())
    }

    fn recent_internal(&self, limit: usize, include_messages: bool) -> Vec<Value> {
        let inner = self.inner.lock().expect("request log mutex poisoned");
        let mut entries = if let Some(path) = sqlite_storage_path(inner.backend.as_ref()) {
            load_recent_entries_from_sqlite(&path, limit).unwrap_or_default()
        } else {
            inner
                .entries
                .iter()
                .rev()
                .take(limit)
                .cloned()
                .collect::<Vec<_>>()
        };
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

fn load_entries(backend: Option<&SharedStateBackend>) -> Option<VecDeque<Value>> {
    if let Some(path) = sqlite_storage_path(backend) {
        return load_entries_from_sqlite(&path);
    }
    let parsed = load_json_state::<Vec<Value>>(backend)?;
    let mut entries = parsed.into_iter().collect::<VecDeque<_>>();
    while entries.len() > MAX_REQUEST_LOG_ENTRIES {
        entries.pop_front();
    }
    Some(entries)
}

fn persist_entries(entries: &VecDeque<Value>, backend: Option<&SharedStateBackend>) {
    if let Some(path) = sqlite_storage_path(backend) {
        let _ = persist_entries_to_sqlite(entries, &path);
        return;
    }
    let payload = entries.iter().cloned().collect::<Vec<_>>();
    persist_json_state(&payload, backend);
}

fn persist_entry(entry: &Value, backend: Option<&SharedStateBackend>) -> bool {
    let Some(path) = sqlite_storage_path(backend) else {
        return false;
    };
    append_entry_to_sqlite(entry, &path)
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
                "CREATE TABLE IF NOT EXISTS {REQUEST_LOG_TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at INTEGER NOT NULL,
                    entry_json TEXT NOT NULL
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

fn load_entries_from_sqlite(path: &PathBuf) -> Option<VecDeque<Value>> {
    let connection = connect_sqlite(path)?;
    let mut statement = connection
        .prepare(&format!(
            "SELECT entry_json FROM {REQUEST_LOG_TABLE_NAME} ORDER BY id ASC"
        ))
        .ok()?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .ok()?;
    let mut entries = VecDeque::new();
    for row in rows {
        let raw = row.ok()?;
        entries.push_back(serde_json::from_str::<Value>(&raw).ok()?);
    }
    while entries.len() > MAX_REQUEST_LOG_ENTRIES {
        entries.pop_front();
    }
    Some(entries)
}

fn load_recent_entries_from_sqlite(path: &PathBuf, limit: usize) -> Option<Vec<Value>> {
    let connection = connect_sqlite(path)?;
    let mut statement = connection
        .prepare(&format!(
            "SELECT entry_json FROM {REQUEST_LOG_TABLE_NAME} ORDER BY id DESC LIMIT ?1"
        ))
        .ok()?;
    let rows = statement
        .query_map(params![limit as i64], |row| row.get::<_, String>(0))
        .ok()?;
    let mut entries = Vec::new();
    for row in rows {
        let raw = row.ok()?;
        entries.push(serde_json::from_str::<Value>(&raw).ok()?);
    }
    Some(entries)
}

fn append_entry_to_sqlite(entry: &Value, path: &PathBuf) -> bool {
    let connection = match connect_sqlite(path) {
        Some(connection) => connection,
        None => return false,
    };
    let Ok(serialized) = serde_json::to_string(entry) else {
        return false;
    };
    let logged_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    if connection
        .execute(
            &format!(
                "INSERT INTO {REQUEST_LOG_TABLE_NAME} (logged_at, entry_json) VALUES (?1, ?2)"
            ),
            params![logged_at, serialized],
        )
        .is_err()
    {
        return false;
    }
    connection
        .execute(
            &format!(
                "DELETE FROM {REQUEST_LOG_TABLE_NAME}
                 WHERE id NOT IN (
                     SELECT id FROM {REQUEST_LOG_TABLE_NAME}
                     ORDER BY id DESC
                     LIMIT {MAX_REQUEST_LOG_ENTRIES}
                 )"
            ),
            [],
        )
        .is_ok()
}

fn persist_entries_to_sqlite(entries: &VecDeque<Value>, path: &PathBuf) -> bool {
    let mut connection = match connect_sqlite(path) {
        Some(connection) => connection,
        None => return false,
    };
    let transaction = match connection.transaction() {
        Ok(transaction) => transaction,
        Err(_) => return false,
    };
    if transaction
        .execute(&format!("DELETE FROM {REQUEST_LOG_TABLE_NAME}"), [])
        .is_err()
    {
        return false;
    }
    for entry in entries {
        let Ok(serialized) = serde_json::to_string(entry) else {
            return false;
        };
        let logged_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        if transaction
            .execute(
                &format!(
                    "INSERT INTO {REQUEST_LOG_TABLE_NAME} (logged_at, entry_json) VALUES (?1, ?2)"
                ),
                params![logged_at, serialized],
            )
            .is_err()
        {
            return false;
        }
    }
    transaction.commit().is_ok()
}

#[cfg(test)]
mod tests {
    use rusqlite::{params, Connection};

    use super::{RequestLogStore, REQUEST_LOG_TABLE_NAME};
    use crate::state_store::sqlite_backend;

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

    #[test]
    fn request_log_store_normalizes_sqlite_entries_as_rows() {
        let unique = format!(
            "headroom-request-log-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store =
            RequestLogStore::new_with_backend(Some(sqlite_backend(path.clone(), "request_log")));
        store.log(serde_json::json!({
            "timestamp": "2026-04-30T12:00:00Z",
            "provider": "headroom",
            "tokens_saved": 25,
        }));
        store.log(serde_json::json!({
            "timestamp": "2026-04-30T12:00:01Z",
            "provider": "openai",
            "tokens_saved": 11,
        }));
        drop(store);

        let connection = Connection::open(&path).unwrap();
        let row_count: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {REQUEST_LOG_TABLE_NAME}"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(row_count, 2);

        let reloaded =
            RequestLogStore::new_with_backend(Some(sqlite_backend(path.clone(), "request_log")));
        let snapshot = reloaded.snapshot();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[0]["provider"], serde_json::json!("headroom"));
        assert_eq!(snapshot[1]["provider"], serde_json::json!("openai"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn request_log_store_reads_recent_entries_directly_from_sqlite() {
        let unique = format!(
            "headroom-request-log-query-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let store =
            RequestLogStore::new_with_backend(Some(sqlite_backend(path.clone(), "request_log")));
        store.log(serde_json::json!({
            "timestamp": "2026-04-30T12:00:00Z",
            "provider": "headroom",
        }));

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                &format!(
                    "INSERT INTO {REQUEST_LOG_TABLE_NAME} (logged_at, entry_json) VALUES (?1, ?2)"
                ),
                params![
                    1_i64,
                    serde_json::to_string(&serde_json::json!({
                        "timestamp": "2026-04-30T12:00:01Z",
                        "provider": "openai",
                    }))
                    .unwrap()
                ],
            )
            .unwrap();

        let snapshot = store.snapshot();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[1]["provider"], serde_json::json!("openai"));

        let recent = store.recent(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0]["provider"], serde_json::json!("openai"));

        let _ = std::fs::remove_file(path);
    }
}
