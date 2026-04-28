//! ContentRouter — strategy decision + dispatch (in-progress port).
//!
//! Direct port of `headroom/transforms/content_router.py`. The router
//! sits in front of every compressor and decides which one runs based
//! on detected content type. Originally pure-Python; moving in-process
//! to Rust so the proxy hot path doesn't pay the cross-language cost
//! per message.
//!
//! # Port progression
//!
//! | PR  | Scope                                                   |
//! | --- | ------------------------------------------------------- |
//! | PR1 | `ContentDetector` (regex-based detection)               |
//! | PR2 | **Types + config + cache + strategy mapper (this PR)**  |
//! | PR3 | Magika integration                                      |
//! | PR4 | Mixed-content detection + section splitter              |
//! | PR5 | `compress()` + dispatch (Rust → Rust direct, Rust →     |
//! |     | Python via PyO3 callback for not-yet-ported compressors)|
//! | PR6 | `apply()` proxy entrypoint                              |
//! | PR7 | TOIN feedback loop                                      |
//!
//! This module re-exports the public surface from its submodules so
//! callers can `use headroom_core::transforms::content_router::*`.

pub mod cache;
pub mod config;
pub mod strategy_map;
pub mod types;

pub use cache::{CacheStats, CachedResult, CompressionCache, DEFAULT_TTL_SECONDS};
pub use config::ContentRouterConfig;
pub use strategy_map::{content_type_for_strategy, strategy_for_content_type};
pub use types::{CompressionStrategy, ContentSection, RouterCompressionResult, RoutingDecision};
