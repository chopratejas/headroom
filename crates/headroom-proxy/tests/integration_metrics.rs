//! Prometheus-style metrics endpoint for the Rust proxy.

mod common;

use common::start_proxy;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn metrics_endpoint_reports_proxied_request_counters() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/metrics-demo"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::get(format!("{}/metrics-demo", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics = reqwest::get(format!("{}/metrics", proxy.url()))
        .await
        .unwrap();
    assert_eq!(metrics.status(), 200);
    let content_type = metrics
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.starts_with("text/plain"));

    let body = metrics.text().await.unwrap();
    assert!(body.contains("headroom_requests_total 1"));
    assert!(body.contains("headroom_requests_in_flight 0"));
    assert!(body.contains("headroom_request_duration_seconds_count 1"));
    assert!(body.contains("headroom_upstream_responses_total{status_class=\"2xx\"} 1"));

    proxy.shutdown().await;
}

#[tokio::test]
async fn metrics_endpoint_is_intercepted_by_rust() {
    let upstream = MockServer::start().await;
    let proxy = start_proxy(&upstream.uri()).await;

    let metrics = reqwest::get(format!("{}/metrics", proxy.url()))
        .await
        .unwrap();
    assert_eq!(metrics.status(), 200);
    let body = metrics.text().await.unwrap();
    assert!(body.contains("# HELP headroom_requests_total"));
    assert!(body.contains("headroom_requests_total 0"));

    proxy.shutdown().await;
}
