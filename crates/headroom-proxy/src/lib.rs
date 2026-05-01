//! headroom-proxy library: transparent reverse proxy in front of the Python
//! Headroom proxy. Used by both `main.rs` and the integration tests.

pub mod config;
pub mod error;
pub mod headers;
pub mod health;
pub mod metrics;
mod product_store;
pub mod proxy;
mod request_log_store;
mod state_store;
mod telemetry_store;
pub mod websocket;

pub use config::Config;
pub use error::ProxyError;
pub use proxy::{build_app, AppState};
