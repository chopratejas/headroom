//! Error types for the proxy.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("upstream request failed: {0}")]
    Upstream(#[from] reqwest::Error),

    #[error("invalid upstream URL: {0}")]
    InvalidUpstream(String),

    #[error("invalid header: {0}")]
    InvalidHeader(String),

    #[error("websocket error: {0}")]
    WebSocket(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, msg) = match &self {
            ProxyError::Upstream(e) if e.is_timeout() => (
                StatusCode::GATEWAY_TIMEOUT,
                format!("upstream timeout: {e}"),
            ),
            ProxyError::Upstream(e) if e.is_connect() => (
                StatusCode::BAD_GATEWAY,
                format!("upstream connect error: {e}"),
            ),
            ProxyError::Upstream(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::InvalidUpstream(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::InvalidHeader(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ProxyError::WebSocket(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };
        tracing::warn!(error = %msg, "proxy error");
        (status, msg).into_response()
    }
}
