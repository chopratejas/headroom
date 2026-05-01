//! Explicit Gemini passthrough route scaffolding.

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
async fn gemini_passthrough_routes_forward_to_upstream() {
    let upstream = MockServer::start().await;
    for (m, p, body) in [
        ("GET", "/v1beta/models", "models"),
        ("GET", "/v1beta/models/demo", "model"),
        ("POST", "/v1beta/models/demo:embedContent", "embed"),
        (
            "POST",
            "/v1beta/models/demo:batchEmbedContents",
            "batch-embed",
        ),
        (
            "POST",
            "/v1beta/models/demo:batchGenerateContent",
            "batch-generate",
        ),
        ("GET", "/v1beta/batches/b1", "batch-get"),
        ("POST", "/v1beta/batches/b1:cancel", "batch-cancel"),
        ("DELETE", "/v1beta/batches/b1", "batch-delete"),
        ("POST", "/v1beta/cachedContents", "cache-create"),
        ("GET", "/v1beta/cachedContents", "cache-list"),
        ("GET", "/v1beta/cachedContents/cache-1", "cache-get"),
        ("DELETE", "/v1beta/cachedContents/cache-1", "cache-delete"),
    ] {
        let template = ResponseTemplate::new(200).set_body_string(body);
        match m {
            "GET" => {
                Mock::given(method("GET"))
                    .and(path(p))
                    .respond_with(template)
                    .mount(&upstream)
                    .await;
            }
            "POST" => {
                Mock::given(method("POST"))
                    .and(path(p))
                    .respond_with(template)
                    .mount(&upstream)
                    .await;
            }
            "DELETE" => {
                Mock::given(method("DELETE"))
                    .and(path(p))
                    .respond_with(template)
                    .mount(&upstream)
                    .await;
            }
            _ => unreachable!(),
        }
    }

    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    assert_eq!(
        client
            .get(format!("{}/v1beta/models", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "models"
    );
    assert_eq!(
        client
            .get(format!("{}/v1beta/models/demo", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "model"
    );
    assert_eq!(
        client
            .post(format!("{}/v1beta/models/demo:embedContent", proxy.url()))
            .json(&serde_json::json!({"content":{}}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "embed"
    );
    assert_eq!(
        client
            .post(format!(
                "{}/v1beta/models/demo:batchEmbedContents",
                proxy.url()
            ))
            .json(&serde_json::json!({"requests":[]}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "batch-embed"
    );
    assert_eq!(
        client
            .post(format!(
                "{}/v1beta/models/demo:batchGenerateContent",
                proxy.url()
            ))
            .json(&serde_json::json!({"requests":[]}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "batch-generate"
    );
    assert_eq!(
        client
            .get(format!("{}/v1beta/batches/b1", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "batch-get"
    );
    assert_eq!(
        client
            .post(format!("{}/v1beta/batches/b1:cancel", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "batch-cancel"
    );
    assert_eq!(
        client
            .delete(format!("{}/v1beta/batches/b1", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "batch-delete"
    );
    assert_eq!(
        client
            .post(format!("{}/v1beta/cachedContents", proxy.url()))
            .json(&serde_json::json!({"ttl":"60s"}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "cache-create"
    );
    assert_eq!(
        client
            .get(format!("{}/v1beta/cachedContents", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "cache-list"
    );
    assert_eq!(
        client
            .get(format!("{}/v1beta/cachedContents/cache-1", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "cache-get"
    );
    assert_eq!(
        client
            .delete(format!("{}/v1beta/cachedContents/cache-1", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "cache-delete"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn gemini_passthrough_routes_use_gemini_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let direct_gemini = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1beta/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("models-direct"))
        .mount(&direct_gemini)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/demo:batchGenerateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_string("batch-generate-direct"))
        .mount(&direct_gemini)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.gemini_api_url = Url::parse(&direct_gemini.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;
    let client = reqwest::Client::new();

    assert_eq!(
        client
            .get(format!("{}/v1beta/models", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "models-direct"
    );
    assert_eq!(
        client
            .post(format!(
                "{}/v1beta/models/demo:batchGenerateContent",
                proxy.url()
            ))
            .json(&serde_json::json!({"requests":[]}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "batch-generate-direct"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn gemini_batch_generate_content_emits_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/demo:batchGenerateContent"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "name": "operations/demo" })),
        )
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/demo:batchGenerateContent",
            proxy.url()
        ))
        .json(&serde_json::json!({"requests":[]}))
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
                        "gemini_batch_generate_content".to_string(),
                    ))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("endpoint")
                    == Some(&MetadataValue::String("batchGenerateContent".to_string()))
        })
        .expect("input routed metadata event");
    assert_eq!(
        routed_mode.context.operation,
        "proxy.gemini_batch_generate_content"
    );
    assert_eq!(routed_mode.context.provider, "gemini");
    assert_eq!(routed_mode.context.model, "demo");
    assert_eq!(
        routed_metadata.metadata.get("native_route"),
        Some(&MetadataValue::Bool(true))
    );
}
