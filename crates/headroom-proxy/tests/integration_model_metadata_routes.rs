//! Provider-runtime-aware model metadata route scaffolding.

mod common;

use std::sync::{Arc, Mutex};

use common::{start_proxy_with_config, start_proxy_with_runtime};
use headroom_proxy::Config;
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, PipelineStage,
};
use url::Url;
use wiremock::matchers::{header, method, path};
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
async fn model_metadata_routes_forward_to_upstream() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "data": [] })),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models/gpt-4o-mini"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "id": "gpt-4o-mini" })),
        )
        .mount(&upstream)
        .await;

    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    let list = client
        .get(format!("{}/v1/models", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(list.status(), 200);

    let get = client
        .get(format!("{}/v1/models/gpt-4o-mini", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(get.status(), 200);

    proxy.shutdown().await;
}

#[tokio::test]
async fn model_metadata_routes_use_selected_provider_target_urls() {
    let legacy_upstream = MockServer::start().await;
    let direct_openai = MockServer::start().await;
    let direct_anthropic = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "provider": "openai" })),
        )
        .mount(&direct_openai)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models/claude-3-5-sonnet"))
        .and(header("x-api-key", "anthropic-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "provider": "anthropic" })),
        )
        .mount(&direct_anthropic)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.openai_api_url = Url::parse(&direct_openai.uri()).unwrap();
    config.anthropic_api_url = Url::parse(&direct_anthropic.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;
    let client = reqwest::Client::new();

    let openai = client
        .get(format!("{}/v1/models", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(openai.status(), 200);
    let openai_json: serde_json::Value = openai.json().await.unwrap();
    assert_eq!(openai_json["provider"], "openai");

    let anthropic = client
        .get(format!("{}/v1/models/claude-3-5-sonnet", proxy.url()))
        .header("x-api-key", "anthropic-key")
        .send()
        .await
        .unwrap();
    assert_eq!(anthropic.status(), 200);
    let anthropic_json: serde_json::Value = anthropic.json().await.unwrap();
    assert_eq!(anthropic_json["provider"], "anthropic");

    proxy.shutdown().await;
}

#[tokio::test]
async fn model_metadata_routes_preserve_anthropic_auth_headers() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("x-api-key", "anthropic-key"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&upstream)
        .await;

    let proxy = common::start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .get(format!("{}/v1/models", proxy.url()))
        .header("x-api-key", "anthropic-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn model_metadata_route_emits_provider_aware_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models/claude-3-5-sonnet"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "id": "claude-3-5-sonnet" })),
        )
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/v1/models/claude-3-5-sonnet", proxy.url()))
        .header("anthropic-version", "2023-06-01")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;

    let recorded = events.lock().expect("poisoned");
    let event_dump = recorded
        .iter()
        .map(|event| format!("{:?} {:?} {:?}", event.stage, event.context, event.metadata))
        .collect::<Vec<_>>()
        .join("\n");
    let routed_mode = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("route_mode")
                    == Some(&MetadataValue::String("model_metadata_get".to_string()))
        })
        .unwrap_or_else(|| panic!("input routed route_mode event\n{event_dump}"));
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("model_metadata_provider")
                    == Some(&MetadataValue::String("anthropic".to_string()))
        })
        .unwrap_or_else(|| panic!("input routed provider metadata event\n{event_dump}"));
    assert_eq!(routed_mode.context.operation, "proxy.model_metadata_get");
    assert_eq!(routed_mode.context.provider, "anthropic");
    assert_eq!(routed_mode.context.model, "claude-3-5-sonnet");
    assert_eq!(
        routed_metadata.metadata.get("model_id"),
        Some(&MetadataValue::String("claude-3-5-sonnet".to_string()))
    );
}
