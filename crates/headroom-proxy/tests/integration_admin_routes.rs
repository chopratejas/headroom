//! Explicit admin/product route scaffolding.

mod common;

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use common::{start_proxy_with_config, start_proxy_with_runtime};
use headroom_proxy::{build_app, AppState, Config};
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, PipelineStage,
};
use http_body_util::BodyExt;
use tower::ServiceExt;
use url::Url;
use wiremock::MockServer;

struct RecordingPlugin {
    events: Arc<Mutex<Vec<PipelineEvent>>>,
}

impl PipelinePlugin for RecordingPlugin {
    fn on_event(&self, event: &mut PipelineEvent) {
        self.events.lock().expect("poisoned").push(event.clone());
    }
}

#[tokio::test]
async fn admin_stats_routes_are_local() {
    let upstream = MockServer::start().await;
    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    let stats = client
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(stats["compression"]["ccr_entries"], 0);
    assert_eq!(stats["requests"]["total"], 0);
    assert_eq!(
        stats["local_state_backends"]["request_log"]["backend_type"],
        "memory"
    );
    assert_eq!(
        stats["local_state_backends"]["product"]["backend_type"],
        "memory"
    );
    assert_eq!(
        stats["local_state_backends"]["telemetry"]["backend_type"],
        "memory"
    );

    let history = client
        .get(format!("{}/stats-history", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(history["history"], serde_json::json!([]));
    assert_eq!(history["exports"]["default_format"], "json");

    let history_csv = client
        .get(format!("{}/stats-history?format=csv", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(history_csv.status(), 200);
    assert!(history_csv
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("text/csv"));

    let debug_memory = client
        .get(format!("{}/debug/memory", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(debug_memory["components"], serde_json::json!({}));
    assert_eq!(debug_memory["is_over_budget"], false);

    let debug_tasks = client
        .get(format!("{}/debug/tasks", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(debug_tasks, serde_json::json!([]));

    let debug_ws_sessions = client
        .get(format!("{}/debug/ws-sessions", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(debug_ws_sessions, serde_json::json!([]));

    let debug_warmup = client
        .get(format!("{}/debug/warmup", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(
        debug_warmup["runtime"]["websocket_sessions"]["active_sessions"],
        0
    );
    assert_eq!(
        debug_warmup["runtime"]["websocket_sessions"]["active_relay_tasks"],
        0
    );

    let dashboard = client
        .get(format!("{}/dashboard", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(dashboard.status(), 200);
    assert!(dashboard
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("text/html"));
    assert!(dashboard
        .text()
        .await
        .unwrap()
        .contains("Headroom Dashboard"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn sqlite_request_log_survives_restart_for_stats_and_feed() {
    let upstream = MockServer::start().await;
    let unique = format!(
        "headroom-admin-request-log-{}.db",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let savings_path = std::env::temp_dir().join(unique);
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.savings_path = Some(savings_path.clone());

    let first = start_proxy_with_config(config.clone()).await;
    let client = reqwest::Client::new();
    let big_content = serde_json::to_string(&serde_json::Value::Array(
        (0..24)
            .map(|index| {
                serde_json::json!({
                    "id": index,
                    "status": if index % 5 == 0 { "warn" } else { "ok" },
                    "service": "billing-api",
                    "duration_ms": 120 + index,
                    "message": format!("request {} completed with verbose payload", index),
                })
            })
            .collect(),
    ))
    .unwrap();

    let compressed = client
        .post(format!("{}/v1/compress", first.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": big_content}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert!(compressed["tokens_saved"].as_u64().unwrap_or(0) > 0);

    first.shutdown().await;

    let second = start_proxy_with_config(config).await;

    let stats = client
        .get(format!("{}/stats", second.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(
        stats["local_state_backends"]["request_log"]["backend_type"],
        "sqlite"
    );
    assert_eq!(stats["recent_requests"].as_array().unwrap().len(), 1);
    assert_eq!(stats["recent_requests"][0]["provider"], "headroom");

    let feed = client
        .get(format!("{}/transformations/feed?limit=1", second.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(feed["transformations"].as_array().unwrap().len(), 1);
    assert_eq!(feed["transformations"][0]["provider"], "headroom");

    second.shutdown().await;
    let _ = std::fs::remove_file(savings_path);
}

#[tokio::test]
async fn native_admin_routes_are_local() {
    let upstream = MockServer::start().await;
    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    let subscription = client
        .get(format!("{}/subscription-window", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(subscription.status(), 503);
    assert_eq!(
        subscription.json::<serde_json::Value>().await.unwrap()["error"],
        "Subscription tracking is not enabled"
    );

    let quota = client
        .get(format!("{}/quota", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(quota, serde_json::json!({}));

    let feed = client
        .get(format!("{}/transformations/feed", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(feed["transformations"], serde_json::json!([]));
    assert_eq!(feed["log_full_messages"], false);

    let cleared = client
        .post(format!("{}/cache/clear", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(cleared["status"], "cache cleared");
    assert_eq!(cleared["previous_size"], 0);
    assert_eq!(cleared["previous_skip_size"], 0);
    assert_eq!(cleared["previous_response_size"], 0);

    proxy.shutdown().await;
}

#[tokio::test]
async fn debug_memory_rejects_non_loopback_client() {
    let upstream = MockServer::start().await;
    let config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    let app = build_app(AppState::new(config).unwrap());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/debug/memory")
                .extension(ConnectInfo(
                    "203.0.113.10:4123".parse::<std::net::SocketAddr>().unwrap(),
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn debug_introspection_routes_reject_non_loopback_client() {
    let upstream = MockServer::start().await;
    let config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    let app = build_app(AppState::new(config).unwrap());
    for path in ["/debug/tasks", "/debug/ws-sessions", "/debug/warmup"] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(path)
                    .extension(ConnectInfo(
                        "203.0.113.10:4123".parse::<std::net::SocketAddr>().unwrap(),
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
    }
}

#[tokio::test]
async fn admin_stats_history_emits_metadata() {
    let upstream = MockServer::start().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/stats-history", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;

    let recorded = events.lock().expect("poisoned");
    let routed_mode = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("route_mode")
                    == Some(&MetadataValue::String("admin_stats_history".to_string()))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("endpoint")
                    == Some(&MetadataValue::String("stats-history".to_string()))
        })
        .expect("input routed metadata event");
    assert_eq!(routed_mode.context.operation, "proxy.admin_stats_history");
    assert_eq!(
        routed_metadata.metadata.get("native_route"),
        Some(&MetadataValue::Bool(true))
    );
}

#[tokio::test]
async fn stats_cached_query_reuses_short_ttl_snapshot() {
    let upstream = MockServer::start().await;
    let state = AppState::new(Config::for_test(Url::parse(&upstream.uri()).unwrap())).unwrap();
    let app = build_app(state.clone());

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/stats?cached=1")
                .extension(ConnectInfo(
                    "127.0.0.1:4123".parse::<std::net::SocketAddr>().unwrap(),
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let first_json: serde_json::Value =
        serde_json::from_slice(&first.into_body().collect().await.unwrap().to_bytes()).unwrap();

    state.metrics.record_request_started();
    state.metrics.record_request_failed(0.01);

    let second = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/stats?cached=1")
                .extension(ConnectInfo(
                    "127.0.0.1:4123".parse::<std::net::SocketAddr>().unwrap(),
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let second_json: serde_json::Value =
        serde_json::from_slice(&second.into_body().collect().await.unwrap().to_bytes()).unwrap();

    let uncached = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .extension(ConnectInfo(
                    "127.0.0.1:4123".parse::<std::net::SocketAddr>().unwrap(),
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let uncached_json: serde_json::Value =
        serde_json::from_slice(&uncached.into_body().collect().await.unwrap().to_bytes()).unwrap();

    assert_eq!(first_json["summary"]["errors"], 0);
    assert_eq!(second_json["summary"]["errors"], 0);
    assert_eq!(uncached_json["summary"]["errors"], 1);
}
