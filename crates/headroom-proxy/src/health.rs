//! Health endpoints. These are intercepted by Rust and never forwarded.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::proxy::AppState;

/// Own health: 200 if the proxy process is up.
pub async fn healthz() -> impl IntoResponse {
    Json(json!({ "ok": true, "service": "headroom-proxy" }))
}

/// Dashboard-facing health. The embedded dashboard polls `/health` and reads
/// `{ status, version }` for its status pill (`status === "healthy"`). Served by
/// the proxy itself so the dashboard's status/poll cycle works without an
/// upstream — `/healthz` is the machine probe, `/health` is the dashboard shape.
pub async fn health() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "headroom-proxy",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Upstream health: GETs upstream `/healthz`. Returns 200 when reachable +
/// 2xx, 503 otherwise. The endpoint name is reserved by the proxy and is
/// not forwarded; operators must not name a real upstream route this.
pub async fn healthz_upstream(State(state): State<AppState>) -> Response {
    // Use an absolute path so upstream URLs with non-trailing-slash paths
    // (e.g. http://localhost:8788/api) resolve to /healthz, not replace the
    // last segment. Url::join("healthz") would strip "api" per RFC 3986.
    let mut url = state.config.upstream.clone();
    url.set_path("/healthz");
    url.set_query(None);
    match state.client.get(url).send().await {
        Ok(resp) if resp.status().is_success() => {
            (StatusCode::OK, Json(json!({"ok": true}))).into_response()
        }
        Ok(resp) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"ok": false, "upstream_status": resp.status().as_u16()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"ok": false, "error": e.to_string()})),
        )
            .into_response(),
    }
}
