//! Conversation-level context management for the OSS proxy.
//!
//! `context/` lives at the crate root parallel to `scoring/`,
//! `signals/`, and `transforms/` because it operates on a *different*
//! abstraction than any of them:
//!
//! - `transforms/` and `signals/` work on **content** (a string, a
//!   structured JSON blob, a tool-result body).
//! - `scoring/` scores **messages** for importance.
//! - `context/` orchestrates **conversations** — it decides what to
//!   do when a request's message list is over the model's context
//!   budget. It uses `scoring/` for per-message scores and the CCR
//!   store for drop persistence, but it is itself a higher-level
//!   layer than either.
//!
//! # OSS philosophy
//!
//! OSS ships exactly one strategy: [`DropByScoreStrategy`] —
//! multi-factor scoring + safety rails + CCR-on-drop persistence. This
//! alone outclasses every gateway competitor's rolling-window
//! behaviour. Enterprise plugs additional strategies (compress-first,
//! summarize, memory tiers) in via [`ContextStrategy`] without any
//! orchestrator changes.
//!
//! # Why simpler than the Python original
//!
//! The Python `IntelligentContextManager` shipped three cascading
//! strategies (COMPRESS_FIRST, SUMMARIZE, DROP_BY_SCORE) plus 12+
//! config fields and integration hooks for ContentRouter,
//! ProgressiveSummarizer, TOIN, and the CCR store. Two of those
//! strategies belong to value-adds we're scoping to Enterprise; one
//! (memory tiers) was never implemented at all. This module is the
//! deliberate refactor: keep the OSS-defining behaviour, push the
//! rest behind a trait extension point.

pub mod candidate;
pub mod ccr_drop;
pub mod config;
pub mod manager;
pub mod safety;
pub mod strategy;
pub mod workspace;

pub use config::IcmConfig;
pub use manager::{ApplyCtx, ApplyResult, IntelligentContextManager};
pub use strategy::{ContextStrategy, DropByScoreStrategy};
pub use workspace::{ContextWorkspace, StrategyOutcome};
