//! Two-tier TTL cache used by the `ContentRouter` to skip work on
//! repeat content.
//!
//! Mirrors `headroom.transforms.content_router.CompressionCache`:
//!
//! - **Tier 1 (skip set)**: hashes of content that *won't* compress.
//!   Hits are near-zero-cost (just a key lookup) and let the router
//!   pass-through known-incompressible payloads without re-running any
//!   compressor.
//! - **Tier 2 (result cache)**: cached compressed text + ratio +
//!   strategy for content that *did* compress, so subsequent identical
//!   requests reuse the result.
//!
//! Entries expire after a fixed TTL (default 30 minutes). There's no
//! max-entries cap — TTL is the natural bound, and memory grows with
//! `compressible_content × TTL`, which is bounded by session duration.
//!
//! # Concurrency
//!
//! The proxy serves requests on a thread pool; the router's hot path
//! shares one cache across all workers. We use `DashMap` (sharded
//! concurrent map) so puts/gets don't block each other under load.
//! That's the same pattern PR9 used for the CCR store and gave us
//! ~4× throughput on multi-worker benches.
//!
//! # Stats
//!
//! Counters (`hits`, `misses`, `skip_hits`, `evictions`,
//! `total_lookup_ns`, `lookup_count`) are exposed via [`CompressionCache::stats`].
//! They're informational — telemetry can scrape them periodically.
//! Counts use `AtomicU64` so updates don't take any cache lock.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;

/// Default TTL: 30 minutes. Matches Python's default.
pub const DEFAULT_TTL_SECONDS: u64 = 1800;

/// Tier-2 entry: cached compression result.
#[derive(Debug, Clone)]
struct ResultEntry {
    compressed: String,
    ratio: f64,
    strategy: String,
    inserted: Instant,
}

/// Tier-1 entry: marker that this hash is known non-compressible.
#[derive(Debug, Clone, Copy)]
struct SkipEntry {
    inserted: Instant,
}

/// Two-tier TTL cache. Cheap to clone (it's an `Arc` internally via
/// `DashMap`), so handing the same cache to multiple worker threads is
/// just a refcount bump.
#[derive(Debug)]
pub struct CompressionCache {
    results: DashMap<u64, ResultEntry>,
    skip: DashMap<u64, SkipEntry>,
    ttl: Duration,
    // Counters
    hits: AtomicU64,
    misses: AtomicU64,
    skip_hits: AtomicU64,
    evictions: AtomicU64,
    total_lookup_ns: AtomicU64,
    lookup_count: AtomicU64,
}

/// Snapshot of cache counters at a point in time.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub skip_hits: u64,
    pub evictions: u64,
    pub size: u64,
    pub skip_size: u64,
    /// Average lookup latency in nanoseconds. `0` when no lookups have
    /// happened yet (we don't divide by zero).
    pub avg_lookup_ns: f64,
}

/// Result returned by [`CompressionCache::get`] on a hit.
#[derive(Debug, Clone)]
pub struct CachedResult {
    pub compressed: String,
    pub ratio: f64,
    pub strategy: String,
}

impl CompressionCache {
    /// New cache with the default 30-minute TTL.
    pub fn new() -> Self {
        Self::with_ttl(Duration::from_secs(DEFAULT_TTL_SECONDS))
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            results: DashMap::new(),
            skip: DashMap::new(),
            ttl,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            skip_hits: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            total_lookup_ns: AtomicU64::new(0),
            lookup_count: AtomicU64::new(0),
        }
    }

    /// Look up a Tier-2 cached result. `None` if missing or expired.
    /// Increments `hits` / `misses` / lookup-latency stats.
    ///
    /// Always check [`is_skipped`](Self::is_skipped) first — a hit there
    /// means the content is known not to compress and the router should
    /// short-circuit without ever running this lookup.
    pub fn get(&self, key: u64) -> Option<CachedResult> {
        let t0 = Instant::now();
        let outcome = self.results.get(&key);
        let result = match outcome {
            Some(entry) => {
                if entry.inserted.elapsed() < self.ttl {
                    let r = CachedResult {
                        compressed: entry.compressed.clone(),
                        ratio: entry.ratio,
                        strategy: entry.strategy.clone(),
                    };
                    drop(entry);
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    Some(r)
                } else {
                    drop(entry);
                    self.results.remove(&key);
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        };
        self.total_lookup_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.lookup_count.fetch_add(1, Ordering::Relaxed);
        result
    }

    /// Tier-1 check: is this key known to be incompressible? Increments
    /// `skip_hits` on a hit. Drops expired entries on the way through.
    pub fn is_skipped(&self, key: u64) -> bool {
        if let Some(entry) = self.skip.get(&key) {
            if entry.inserted.elapsed() < self.ttl {
                drop(entry);
                self.skip_hits.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            drop(entry);
            self.skip.remove(&key);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
        false
    }

    /// Store a compressed result in Tier 2. Overwrites any existing
    /// entry for the same key — same content under the same hash is
    /// idempotent.
    pub fn put(&self, key: u64, compressed: String, ratio: f64, strategy: String) {
        self.results.insert(
            key,
            ResultEntry {
                compressed,
                ratio,
                strategy,
                inserted: Instant::now(),
            },
        );
    }

    /// Mark a key as known-non-compressible (Tier 1). Cheap; the
    /// router calls this when a compress attempt didn't improve the
    /// payload.
    pub fn mark_skip(&self, key: u64) {
        self.skip.insert(
            key,
            SkipEntry {
                inserted: Instant::now(),
            },
        );
    }

    /// Move a Tier-2 entry into Tier 1. Used when an external
    /// threshold tightens (e.g. context pressure rose) and the
    /// previously-cached result no longer qualifies as a worthwhile
    /// compression. Removes the result entry.
    pub fn move_to_skip(&self, key: u64) {
        self.results.remove(&key);
        self.skip.insert(
            key,
            SkipEntry {
                inserted: Instant::now(),
            },
        );
    }

    /// Number of live Tier-2 result entries.
    pub fn size(&self) -> usize {
        self.results.len()
    }

    /// Number of live Tier-1 skip entries.
    pub fn skip_size(&self) -> usize {
        self.skip.len()
    }

    /// Snapshot of all counters. Cheap; counters are atomic.
    pub fn stats(&self) -> CacheStats {
        let lookups = self.lookup_count.load(Ordering::Relaxed);
        let total_ns = self.total_lookup_ns.load(Ordering::Relaxed);
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            skip_hits: self.skip_hits.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            size: self.results.len() as u64,
            skip_size: self.skip.len() as u64,
            avg_lookup_ns: if lookups > 0 {
                total_ns as f64 / lookups as f64
            } else {
                0.0
            },
        }
    }

    /// Clear both tiers. Counter values are *not* reset — they remain
    /// useful across cache lifecycles for telemetry. If you need
    /// counter reset, build a fresh cache.
    pub fn clear(&self) {
        self.results.clear();
        self.skip.clear();
    }
}

impl Default for CompressionCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn put_then_get_returns_value() {
        let c = CompressionCache::new();
        c.put(42, "compressed".into(), 0.5, "smart_crusher".into());
        let r = c.get(42).expect("hit");
        assert_eq!(r.compressed, "compressed");
        assert_eq!(r.ratio, 0.5);
        assert_eq!(r.strategy, "smart_crusher");
    }

    #[test]
    fn miss_returns_none_and_increments_misses() {
        let c = CompressionCache::new();
        assert!(c.get(7).is_none());
        let s = c.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 0);
    }

    #[test]
    fn put_overwrites_under_same_key() {
        let c = CompressionCache::new();
        c.put(1, "first".into(), 0.5, "a".into());
        c.put(1, "second".into(), 0.4, "b".into());
        let r = c.get(1).unwrap();
        assert_eq!(r.compressed, "second");
        assert_eq!(c.size(), 1);
    }

    #[test]
    fn ttl_expires_results_on_get() {
        let c = CompressionCache::with_ttl(Duration::from_millis(10));
        c.put(1, "v".into(), 0.5, "s".into());
        std::thread::sleep(Duration::from_millis(25));
        assert!(c.get(1).is_none());
        assert_eq!(c.size(), 0);
        assert!(c.stats().evictions >= 1);
    }

    #[test]
    fn skip_set_blocks_compression() {
        let c = CompressionCache::new();
        c.mark_skip(99);
        assert!(c.is_skipped(99));
        assert_eq!(c.stats().skip_hits, 1);
    }

    #[test]
    fn ttl_expires_skip_set() {
        let c = CompressionCache::with_ttl(Duration::from_millis(10));
        c.mark_skip(1);
        std::thread::sleep(Duration::from_millis(25));
        assert!(!c.is_skipped(1));
        assert_eq!(c.skip_size(), 0);
    }

    #[test]
    fn move_to_skip_clears_result() {
        let c = CompressionCache::new();
        c.put(1, "v".into(), 0.5, "s".into());
        c.move_to_skip(1);
        assert!(c.get(1).is_none());
        assert!(c.is_skipped(1));
    }

    #[test]
    fn clear_drops_both_tiers() {
        let c = CompressionCache::new();
        c.put(1, "v".into(), 0.5, "s".into());
        c.mark_skip(2);
        c.clear();
        assert_eq!(c.size(), 0);
        assert_eq!(c.skip_size(), 0);
    }

    #[test]
    fn stats_avg_lookup_zero_when_no_lookups() {
        let c = CompressionCache::new();
        assert_eq!(c.stats().avg_lookup_ns, 0.0);
    }

    #[test]
    fn cache_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CompressionCache>();
    }

    #[test]
    fn concurrent_puts_and_gets_are_safe() {
        // 8 threads, 1000 ops each, mix of put/get/mark_skip. Stress
        // the DashMap-backed sharding under contention. The point is
        // not "fast" but "doesn't panic / corrupt counters".
        let c = std::sync::Arc::new(CompressionCache::new());
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let c = c.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000u64 {
                    let key = t * 1000 + i;
                    c.put(key, format!("c{key}"), 0.5, "s".into());
                    let _ = c.get(key);
                    if i % 7 == 0 {
                        c.mark_skip(key);
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let stats = c.stats();
        // Sanity: hits + misses == lookup_count (we only call get).
        assert_eq!(stats.hits + stats.misses, 8 * 1000);
    }
}
