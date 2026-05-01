use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

const STATE_TABLE_NAME: &str = "headroom_state";

pub trait StateBackend: Send + Sync {
    fn backend_type(&self) -> &'static str;
    fn storage_path(&self) -> Option<PathBuf>;
    fn read(&self) -> Option<String>;
    fn write(&self, serialized: &str) -> bool;
    fn bytes_used(&self) -> u64;
}

pub type SharedStateBackend = Arc<dyn StateBackend>;

#[derive(Clone)]
pub struct JsonFileStateBackend {
    path: PathBuf,
}

impl JsonFileStateBackend {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl StateBackend for JsonFileStateBackend {
    fn backend_type(&self) -> &'static str {
        "json_file"
    }

    fn storage_path(&self) -> Option<PathBuf> {
        Some(self.path.clone())
    }

    fn read(&self) -> Option<String> {
        fs::read_to_string(&self.path).ok()
    }

    fn write(&self, serialized: &str) -> bool {
        if let Some(parent) = self.path.parent() {
            if fs::create_dir_all(parent).is_err() {
                return false;
            }
        }
        let tmp_path = self.path.with_extension("tmp");
        if fs::write(&tmp_path, serialized).is_err() {
            return false;
        }
        if self.path.exists() && fs::remove_file(&self.path).is_err() {
            let _ = fs::remove_file(&tmp_path);
            return false;
        }
        if fs::rename(&tmp_path, &self.path).is_err() {
            let _ = fs::remove_file(&tmp_path);
            return false;
        }
        true
    }

    fn bytes_used(&self) -> u64 {
        fs::metadata(&self.path)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    }
}

#[derive(Clone)]
pub struct SqliteStateBackend {
    path: PathBuf,
    state_key: String,
}

impl SqliteStateBackend {
    pub fn new(path: PathBuf, state_key: impl Into<String>) -> Self {
        Self {
            path,
            state_key: state_key.into(),
        }
    }

    fn connect(&self) -> Option<Connection> {
        if let Some(parent) = self.path.parent() {
            if fs::create_dir_all(parent).is_err() {
                return None;
            }
        }
        let connection = Connection::open(&self.path).ok()?;
        if connection
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {STATE_TABLE_NAME} (
                        state_key TEXT PRIMARY KEY,
                        state_json TEXT NOT NULL,
                        updated_at INTEGER NOT NULL
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
}

impl StateBackend for SqliteStateBackend {
    fn backend_type(&self) -> &'static str {
        "sqlite"
    }

    fn storage_path(&self) -> Option<PathBuf> {
        Some(self.path.clone())
    }

    fn read(&self) -> Option<String> {
        let connection = self.connect()?;
        connection
            .query_row(
                &format!("SELECT state_json FROM {STATE_TABLE_NAME} WHERE state_key = ?1"),
                params![&self.state_key],
                |row| row.get(0),
            )
            .ok()
    }

    fn write(&self, serialized: &str) -> bool {
        let connection = match self.connect() {
            Some(connection) => connection,
            None => return false,
        };
        connection
            .execute(
                &format!(
                    "INSERT INTO {STATE_TABLE_NAME} (state_key, state_json, updated_at)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(state_key) DO UPDATE SET
                         state_json = excluded.state_json,
                         updated_at = excluded.updated_at"
                ),
                params![
                    &self.state_key,
                    serialized,
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64
                ],
            )
            .is_ok()
    }

    fn bytes_used(&self) -> u64 {
        fs::metadata(&self.path)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    }
}

pub fn file_backend(path: PathBuf) -> SharedStateBackend {
    Arc::new(JsonFileStateBackend::new(path))
}

pub fn sqlite_backend(path: PathBuf, state_key: impl Into<String>) -> SharedStateBackend {
    Arc::new(SqliteStateBackend::new(path, state_key))
}

pub fn uses_sqlite_backend(path: &PathBuf) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("db" | "sqlite" | "sqlite3")
    )
}

pub fn load_json_state<T: DeserializeOwned>(backend: Option<&SharedStateBackend>) -> Option<T> {
    let raw = backend?.read()?;
    serde_json::from_str::<T>(&raw).ok()
}

pub fn persist_json_state<T: Serialize>(value: &T, backend: Option<&SharedStateBackend>) -> bool {
    let Some(backend) = backend else {
        return false;
    };
    let Ok(serialized) = serde_json::to_string_pretty(value) else {
        return false;
    };
    backend.write(&serialized)
}

pub fn backend_details(backend: Option<&SharedStateBackend>, entry_count: usize) -> Value {
    match backend {
        Some(backend) => json!({
            "backend_type": backend.backend_type(),
            "entry_count": entry_count,
            "bytes_used": backend.bytes_used(),
            "path": backend.storage_path().map(|path| path.display().to_string()),
        }),
        None => json!({
            "backend_type": "memory",
            "entry_count": entry_count,
            "bytes_used": 0_u64,
            "path": Value::Null,
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        backend_details, file_backend, persist_json_state, sqlite_backend, uses_sqlite_backend,
    };

    #[test]
    fn file_backend_reports_details_after_persist() {
        let unique = format!(
            "headroom-state-backend-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let backend = file_backend(path.clone());
        assert!(persist_json_state(
            &serde_json::json!({"hello":"world"}),
            Some(&backend)
        ));

        let details = backend_details(Some(&backend), 1);
        assert_eq!(details["backend_type"], serde_json::json!("json_file"));
        assert_eq!(
            details["path"],
            serde_json::json!(path.display().to_string())
        );
        assert!(details["bytes_used"].as_u64().unwrap_or(0) > 0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sqlite_backend_reports_details_after_persist() {
        let unique = format!(
            "headroom-state-backend-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let backend = sqlite_backend(path.clone(), "product");
        assert!(persist_json_state(
            &serde_json::json!({"hello":"world"}),
            Some(&backend)
        ));

        let details = backend_details(Some(&backend), 1);
        assert_eq!(details["backend_type"], serde_json::json!("sqlite"));
        assert_eq!(
            details["path"],
            serde_json::json!(path.display().to_string())
        );
        assert!(details["bytes_used"].as_u64().unwrap_or(0) > 0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sqlite_backend_selection_uses_database_extensions() {
        assert!(uses_sqlite_backend(&PathBuf::from("state.db")));
        assert!(uses_sqlite_backend(&PathBuf::from("state.sqlite")));
        assert!(uses_sqlite_backend(&PathBuf::from("state.sqlite3")));
        assert!(!uses_sqlite_backend(&PathBuf::from("state.json")));
    }
}
