//! `/stats` endpoint: the Rust-native savings JSON is served by default and
//! exposes the backend-agnostic dashboard contract.

mod common;

use common::{start_proxy, start_proxy_with};
use wiremock::matchers::{method, path as path_matcher};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn stats_endpoint_serves_savings_json_by_default() {
    let upstream = MockServer::start().await;
    let proxy = start_proxy(&upstream.uri()).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .expect("GET /stats");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("stats json");

    // Core contract the dashboard consumes is present and well-formed on a
    // fresh proxy (no traffic yet → zeros, not missing keys).
    assert_eq!(body["requests"]["total"], 0);
    assert_eq!(body["tokens"]["saved"], 0);
    assert_eq!(body["tokens"]["savings_percent"], 0.0);
    assert!(body["requests"]["by_provider"].is_object());
    assert!(body["cost"]["per_model"].is_object());
    assert!(body["persistent_savings"]["lifetime"].is_object());
    assert!(body["display_session"].is_object());

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_records_llm_request_attributed_by_provider() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;

    // Recording is gated on the compression master switch; enable it.
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let req_body = serde_json::json!({
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(req_body.to_string())
        .send()
        .await
        .expect("POST /v1/messages");
    assert_eq!(resp.status(), 200);

    let stats: serde_json::Value = reqwest::Client::new()
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .expect("GET /stats")
        .json()
        .await
        .expect("stats json");

    // The request was recorded and attributed to the Anthropic backend.
    assert_eq!(stats["requests"]["total"], 1);
    assert_eq!(stats["requests"]["by_provider"]["anthropic"], 1);

    proxy.shutdown().await;
}

#[tokio::test]
async fn dashboard_endpoint_serves_embedded_html() {
    let upstream = MockServer::start().await;
    let proxy = start_proxy(&upstream.uri()).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/dashboard", proxy.url()))
        .send()
        .await
        .expect("GET /dashboard");
    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(content_type.contains("text/html"), "got {content_type}");

    let body = resp.text().await.expect("dashboard html");
    // The embedded page is the real dashboard and polls the stats endpoint.
    assert!(body.contains("/stats"), "dashboard should reference /stats");

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_folds_in_supplemental_python_blocks() {
    let upstream = MockServer::start().await; // LLM upstream (unused here)
    let python = MockServer::start().await; // transitional Python proxy

    Mock::given(method("GET"))
        .and(path_matcher("/stats"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "copilot_quota": {"latest": {"used": 7}},
            "requests": {"total": 999}  // must NOT override the Rust-native value
        })))
        .mount(&python)
        .await;

    let url = format!("{}/stats", python.uri());
    let proxy = start_proxy_with(&upstream.uri(), move |c| {
        c.upstream_stats_url = Some(url.clone())
    })
    .await;

    let stats: serde_json::Value = reqwest::Client::new()
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .expect("GET /stats")
        .json()
        .await
        .expect("stats json");

    // Python-only block surfaced; Rust-native `requests.total` untouched.
    assert_eq!(stats["copilot_quota"]["latest"]["used"], 7);
    assert_eq!(stats["requests"]["total"], 0);

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_fail_open_when_python_upstream_unreachable() {
    let upstream = MockServer::start().await;
    // Port 1 is never listening → fetch fails; /stats must still succeed.
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.upstream_stats_url = Some("http://127.0.0.1:1/stats".to_string());
    })
    .await;

    let resp = reqwest::Client::new()
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .expect("GET /stats");
    assert_eq!(resp.status(), 200);
    let stats: serde_json::Value = resp.json().await.expect("stats json");
    assert!(stats.get("copilot_quota").is_none());
    assert_eq!(stats["requests"]["total"], 0);

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_persist_across_restart() {
    let dir = std::env::temp_dir().join(format!("hr-persist-{}", uuid::Uuid::new_v4()));
    let path = dir.join("savings.json");

    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;

    let req_body = serde_json::json!({
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    });

    // First instance: persists to `path`, records one request, shuts down.
    {
        let p = path.clone();
        let proxy = start_proxy_with(&upstream.uri(), move |c| {
            c.compression = true;
            c.savings_path = Some(p);
        })
        .await;
        let resp = reqwest::Client::new()
            .post(format!("{}/v1/messages", proxy.url()))
            .header("content-type", "application/json")
            .body(req_body.to_string())
            .send()
            .await
            .expect("POST");
        assert_eq!(resp.status(), 200);
        proxy.shutdown().await;
    }
    assert!(path.exists(), "savings file should have been written");

    // Second instance: same path, no traffic — `/stats` reflects the persisted
    // lifetime from the first instance.
    {
        let p = path.clone();
        let proxy = start_proxy_with(&upstream.uri(), move |c| c.savings_path = Some(p)).await;
        let stats: serde_json::Value = reqwest::Client::new()
            .get(format!("{}/stats", proxy.url()))
            .send()
            .await
            .expect("GET /stats")
            .json()
            .await
            .expect("stats json");
        assert_eq!(stats["requests"]["total"], 1);
        assert_eq!(stats["persistent_savings"]["lifetime"]["requests"], 1);
        proxy.shutdown().await;
    }

    std::fs::remove_dir_all(&dir).ok();
}
