//! Health endpoints: native liveness/readiness/health plus compatibility routes.

mod common;

use common::start_proxy;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn healthz_ok_when_upstream_down() {
    let proxy = start_proxy("http://127.0.0.1:1").await; // unroutable port
    let resp = reqwest::get(format!("{}/healthz", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn livez_reports_proxy_health() {
    let proxy = start_proxy("http://127.0.0.1:1").await;
    let resp = reqwest::get(format!("{}/livez", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["service"], "headroom-proxy");
    assert_eq!(json["status"], "healthy");
    assert_eq!(json["alive"], true);
    proxy.shutdown().await;
}

#[tokio::test]
async fn healthz_upstream_503_when_upstream_down() {
    let proxy = start_proxy("http://127.0.0.1:1").await;
    let resp = reqwest::get(format!("{}/healthz/upstream", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
    proxy.shutdown().await;
}

#[tokio::test]
async fn readyz_uses_upstream_readyz_when_available() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/readyz"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::get(format!("{}/readyz", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["ready"], true);
    assert_eq!(json["checks"]["upstream"]["endpoint"], "/readyz");
    proxy.shutdown().await;
}

#[tokio::test]
async fn readyz_falls_back_to_healthz_for_legacy_upstreams() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/readyz"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/healthz"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::get(format!("{}/readyz", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["ready"], true);
    assert_eq!(json["checks"]["upstream"]["endpoint"], "/healthz");
    proxy.shutdown().await;
}

#[tokio::test]
async fn health_includes_config_and_runtime_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/readyz"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::get(format!("{}/health", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    let normalized_upstream = url::Url::parse(&upstream.uri()).unwrap().to_string();
    assert_eq!(json["status"], "healthy");
    assert_eq!(json["config"]["rewrite_host"], true);
    assert_eq!(json["runtime"]["upstream"]["rewrite_host"], true);
    assert_eq!(json["config"]["openai_api_url"], normalized_upstream);
    assert_eq!(json["config"]["anthropic_api_url"], normalized_upstream);
    assert_eq!(json["config"]["gemini_api_url"], normalized_upstream);
    assert_eq!(json["config"]["databricks_api_url"], normalized_upstream);
    assert_eq!(json["config"]["cloudcode_api_url"], normalized_upstream);
    assert_eq!(
        json["runtime"]["upstream"]["openai_api_url"],
        normalized_upstream
    );
    assert_eq!(
        json["runtime"]["upstream"]["anthropic_api_url"],
        normalized_upstream
    );
    assert_eq!(
        json["runtime"]["upstream"]["gemini_api_url"],
        normalized_upstream
    );
    assert_eq!(
        json["runtime"]["upstream"]["databricks_api_url"],
        normalized_upstream
    );
    assert_eq!(
        json["runtime"]["upstream"]["cloudcode_api_url"],
        normalized_upstream
    );
    assert_eq!(json["config"]["native_openai_chat"], false);
    assert_eq!(json["runtime"]["upstream"]["native_openai_chat"], false);
    assert_eq!(
        json["runtime"]["request_pipeline"]["plugins_enabled"],
        false
    );
    assert_eq!(json["config"]["route_manifest"]["version"], 1);
    assert_eq!(json["runtime"]["route_manifest"]["version"], 1);
    assert!(json["runtime"]["route_manifest"]["families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|family| family["name"] == "admin-local"));
    assert!(json["runtime"]["route_manifest"]["families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|family| family["name"] == "compression"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn healthz_upstream_200_when_upstream_healthy() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/readyz"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::get(format!("{}/healthz/upstream", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}
