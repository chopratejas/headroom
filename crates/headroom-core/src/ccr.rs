//! CCR (Compress-Cache-Retrieve) storage layer.
//!
//! When a transform compresses data with row-drop or opaque-string
//! substitution, the *original payload* is stashed here keyed by the
//! hash that ends up in the prompt. The runtime later honors retrieval
//! tool calls by looking up the hash in this store and serving back the
//! original. This is the cornerstone of CCR: lossy on the wire, lossless
//! end-to-end.
//!
//! Mirrors the semantics of Python's [`CompressionStore`] (`headroom/
//! cache/compression_store.py`) but stripped down to the contract that
//! actually matters for retrieval — no BM25 search, no retrieval-event
//! feedback, no per-tool metadata. Those live in the runtime layer; this
//! crate only needs put/get.
//!
//! # Concurrency
//!
//! The default [`InMemoryCcrStore`] uses [`DashMap`] (sharded concurrent
//! hash map) so reads and writes targeting different keys never contend.
//! Only the FIFO insertion-order queue (used for capacity-bounded
//! eviction) sits behind a single `Mutex`, and that mutex is held just
//! long enough for an O(1) `push_back` or capacity-sweep.
//!
//! Profile under multi-worker load shows order-of-magnitude lower
//! contention than the previous single-`Mutex<HashMap>` design — see
//! `benches/ccr_store.rs`.
//!
//! # Pluggable backend
//!
//! Production deployments swap in their own [`CcrStore`] backed by Redis,
//! MongoDB, or whatever shared cache fits. The default in-memory store
//! ships ready for single-process use.
//!
//! [`CompressionStore`]: https://github.com/chopratejas/headroom/blob/main/headroom/cache/compression_store.py

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use dashmap::DashMap;

/// Pluggable CCR storage backend. `Send + Sync` so it can sit behind an
/// `Arc` and be shared across threads in the proxy.
pub trait CcrStore: Send + Sync {
    /// Stash `payload` under `hash`. If the hash already exists, the
    /// new payload overwrites — same hash should mean same content, so
    /// re-storing is idempotent.
    fn put(&self, hash: &str, payload: &str);

    /// Look up `hash`. Returns `None` if missing or expired.
    fn get(&self, hash: &str) -> Option<String>;

    /// Number of live entries. Informational; used by tests + telemetry.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Default capacity — matches Python's `CompressionStore` default.
pub const DEFAULT_CAPACITY: usize = 1000;

/// Default TTL — 5 minutes, matching Python.
pub const DEFAULT_TTL: Duration = Duration::from_secs(300);

/// In-memory CCR store backed by [`DashMap`] for sharded concurrent
/// access.
///
/// - **TTL**: 5 minutes by default. Entries past their TTL are dropped
///   on the next `get` (lazy expiry — no background reaper thread).
/// - **Capacity**: 1000 entries by default. When `put` would push us
///   past capacity, the oldest entry (per insertion order) is evicted.
/// - **Concurrency**: gets and puts on distinct keys do not contend.
///   The only serialization point is the insertion-order queue used
///   for capacity eviction; that mutex is held for an O(1) push or a
///   small sweep.
pub struct InMemoryCcrStore {
    map: DashMap<String, Entry>,
    /// FIFO insertion order. Stale entries (already removed from `map`
    /// via TTL expiry) are tolerated — `pop_front` + `map.remove` is a
    /// no-op for missing keys, and capacity-bounded sweeps loop until
    /// they actually evict a real entry.
    order: Mutex<VecDeque<String>>,
    ttl: Duration,
    capacity: usize,
}

#[derive(Clone)]
struct Entry {
    payload: String,
    inserted: Instant,
}

impl InMemoryCcrStore {
    /// Default: 1000 entries, 5-minute TTL.
    pub fn new() -> Self {
        Self::with_capacity_and_ttl(DEFAULT_CAPACITY, DEFAULT_TTL)
    }

    pub fn with_capacity_and_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            map: DashMap::with_capacity(capacity),
            order: Mutex::new(VecDeque::with_capacity(capacity)),
            ttl,
            capacity,
        }
    }

    /// Sweep the order queue, dropping leading entries that no longer
    /// exist in the map (already expired or evicted), then evict
    /// real entries until `map.len() < capacity`. Called only from
    /// `put` on a fresh-key insert path.
    fn evict_until_under_capacity(&self) {
        let mut guard = self.order.lock().expect("ccr order mutex poisoned");
        while self.map.len() >= self.capacity {
            let Some(oldest) = guard.pop_front() else {
                break;
            };
            // `remove` is a no-op if `oldest` was already lazy-expired.
            // Loop continues until we actually shrink the map.
            self.map.remove(&oldest);
        }
    }
}

impl Default for InMemoryCcrStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CcrStore for InMemoryCcrStore {
    fn put(&self, hash: &str, payload: &str) {
        // Idempotent re-store fast-path: same hash → overwrite payload
        // in place, leave the order queue alone. Common when the same
        // tool output flows through multiple times in a session.
        if let Some(mut existing) = self.map.get_mut(hash) {
            existing.payload = payload.to_string();
            existing.inserted = Instant::now();
            return;
        }

        // New entry. Cap-bound first (may sweep a few stale order
        // entries), then insert and append to the FIFO queue.
        if self.map.len() >= self.capacity {
            self.evict_until_under_capacity();
        }
        let entry = Entry {
            payload: payload.to_string(),
            inserted: Instant::now(),
        };
        let prev = self.map.insert(hash.to_string(), entry);
        if prev.is_none() {
            // Truly new key — record in FIFO order. (If `prev.is_some()`
            // it means another thread re-inserted between our get_mut
            // miss and this insert; treat that as a fast-path overwrite
            // and skip the queue append to avoid duplicates.)
            self.order
                .lock()
                .expect("ccr order mutex poisoned")
                .push_back(hash.to_string());
        }
    }

    fn get(&self, hash: &str) -> Option<String> {
        // Read path: shard read-lock, check TTL, clone payload out.
        // No global lock involvement at all — distinct hashes hash to
        // distinct shards and never contend.
        let expired_at = {
            let entry = self.map.get(hash)?;
            if entry.inserted.elapsed() > self.ttl {
                Some(()) // signal expired; drop guard before we remove
            } else {
                return Some(entry.payload.clone());
            }
        };
        if expired_at.is_some() {
            self.map.remove(hash);
            return None;
        }
        None
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get_returns_payload() {
        let store = InMemoryCcrStore::new();
        store.put("abc123", r#"[{"id":1}]"#);
        assert_eq!(store.get("abc123"), Some(r#"[{"id":1}]"#.to_string()));
    }

    #[test]
    fn missing_hash_returns_none() {
        let store = InMemoryCcrStore::new();
        assert_eq!(store.get("never_stored"), None);
    }

    #[test]
    fn put_overwrites_under_same_hash() {
        let store = InMemoryCcrStore::new();
        store.put("h", "first");
        store.put("h", "second");
        assert_eq!(store.get("h"), Some("second".to_string()));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let store = InMemoryCcrStore::with_capacity_and_ttl(2, DEFAULT_TTL);
        store.put("a", "1");
        store.put("b", "2");
        store.put("c", "3");
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("a"), None);
        assert_eq!(store.get("b"), Some("2".to_string()));
        assert_eq!(store.get("c"), Some("3".to_string()));
    }

    #[test]
    fn expired_entries_are_dropped_on_get() {
        let store = InMemoryCcrStore::with_capacity_and_ttl(10, Duration::from_millis(10));
        store.put("a", "1");
        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(store.get("a"), None);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn store_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemoryCcrStore>();
    }

    #[test]
    fn trait_object_is_usable() {
        let store: Box<dyn CcrStore> = Box::new(InMemoryCcrStore::new());
        store.put("h", "v");
        assert_eq!(store.get("h"), Some("v".to_string()));
        assert!(!store.is_empty());
    }

    #[test]
    fn concurrent_puts_and_gets_do_not_corrupt() {
        // Smoke test for the concurrent design — N threads each do
        // P puts and P gets against distinct keys. Every key written
        // must be readable afterwards.
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(InMemoryCcrStore::with_capacity_and_ttl(10_000, DEFAULT_TTL));
        let n_threads = 8;
        let per_thread = 200;

        let mut handles = Vec::new();
        for tid in 0..n_threads {
            let s = store.clone();
            handles.push(thread::spawn(move || {
                for i in 0..per_thread {
                    let key = format!("t{tid}_k{i}");
                    let val = format!("v{tid}_{i}");
                    s.put(&key, &val);
                }
                for i in 0..per_thread {
                    let key = format!("t{tid}_k{i}");
                    let got = s.get(&key);
                    assert_eq!(got, Some(format!("v{tid}_{i}")));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(store.len(), n_threads * per_thread);
    }
}
