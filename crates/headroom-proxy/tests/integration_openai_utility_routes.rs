//! Explicit OpenAI utility route scaffolding.

mod common;

use std::sync::{Arc, Mutex};

use common::{start_proxy_with_config, start_proxy_with_runtime};
use headroom_proxy::Config;
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, PipelineStage,
};
use url::Url;
use wiremock::matchers::{method, path};
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
async fn openai_utility_routes_forward_to_upstream() {
    let upstream = MockServer::start().await;
    for route in [
        "/v1/embeddings",
        "/v1/moderations",
        "/v1/images/generations",
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
    ] {
        Mock::given(method("POST"))
            .and(path(route))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({ "route": route })),
            )
            .mount(&upstream)
            .await;
    }

    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    for route in [
        "/v1/embeddings",
        "/v1/moderations",
        "/v1/images/generations",
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
    ] {
        let resp = client
            .post(format!("{}{}", proxy.url(), route))
            .json(&serde_json::json!({"input":"hello"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{route}");
        let json: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(json["route"], route);
    }

    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_utility_routes_use_openai_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let direct_openai = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "route": "direct" })),
        )
        .mount(&direct_openai)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.openai_api_url = Url::parse(&direct_openai.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/embeddings", proxy.url()))
        .json(&serde_json::json!({
            "model": "text-embedding-3-small",
            "input": "hello"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["route"], "direct");

    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_utility_route_emits_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "data": [] })),
        )
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/embeddings", proxy.url()))
        .json(&serde_json::json!({
            "model": "text-embedding-3-small",
            "input": "hello"
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
                    == Some(&MetadataValue::String("openai_embeddings".to_string()))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("endpoint")
                    == Some(&MetadataValue::String("embeddings".to_string()))
        })
        .expect("input routed metadata event");
    assert_eq!(routed_mode.context.operation, "proxy.openai_embeddings");
    assert_eq!(
        routed_metadata.metadata.get("native_route"),
        Some(&MetadataValue::Bool(true))
    );
}
