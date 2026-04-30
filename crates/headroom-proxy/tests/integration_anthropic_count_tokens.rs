//! Feature-flagged native `/v1/messages/count_tokens` route scaffolding.

mod common;

use std::sync::{Arc, Mutex};

use common::{start_proxy_with_config, start_proxy_with_config_and_runtime};
use headroom_proxy::Config;
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, PipelineStage,
};
use url::Url;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct RecordingPlugin {
    events: Arc<Mutex<Vec<PipelineEvent>>>,
}

impl PipelinePlugin for RecordingPlugin {
    fn on_event(&self, event: &mut PipelineEvent) {
        self.events.lock().expect("poisoned").push(event.clone());
    }
}

#[tokio::test]
async fn native_anthropic_count_tokens_route_forwards_when_enabled() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/count_tokens"))
        .and(body_json(serde_json::json!({
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "input_tokens": 12
                })),
        )
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_count_tokens = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages/count_tokens", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["input_tokens"], 12);
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_count_tokens_use_anthropic_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let direct_anthropic = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/count_tokens"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "input_tokens": 34
                })),
        )
        .mount(&direct_anthropic)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.native_anthropic_count_tokens = true;
    config.anthropic_api_url = Url::parse(&direct_anthropic.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages/count_tokens", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["input_tokens"], 34);
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_count_tokens_route_emits_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/count_tokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "input_tokens": 12
        })))
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_count_tokens = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages/count_tokens", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "system": "be terse",
            "messages": [{"role": "user", "content": "hello"}]
        }))
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
                    == Some(&MetadataValue::String(
                        "native_anthropic_count_tokens".to_string(),
                    ))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("system_present") == Some(&MetadataValue::Bool(true))
        })
        .expect("input routed metadata event");
    assert_eq!(
        routed_mode.context.operation,
        "proxy.anthropic_count_tokens"
    );
    assert_eq!(routed_mode.context.model, "claude-haiku-4-5");
    assert_eq!(
        routed_metadata.metadata.get("messages_count"),
        Some(&MetadataValue::U64(1))
    );
}

#[tokio::test]
async fn native_anthropic_count_tokens_route_rejects_invalid_requests() {
    let upstream = MockServer::start().await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_count_tokens = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages/count_tokens", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body = resp.text().await.unwrap();
    assert!(body.contains("missing array field `messages`"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_count_tokens_route_forwards_passthrough_system_shape() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/count_tokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "input_tokens": 12
        })))
        .mount(&upstream)
        .await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_count_tokens = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages/count_tokens", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "system": {"text": "be terse"},
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.json::<serde_json::Value>().await.unwrap();
    assert_eq!(body["input_tokens"], 12);
    proxy.shutdown().await;
}
