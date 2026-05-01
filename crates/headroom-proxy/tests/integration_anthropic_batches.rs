//! Explicit Anthropic batch route scaffolding.

mod common;

use std::sync::{Arc, Mutex};

use common::{
    start_proxy_with_config, start_proxy_with_config_and_runtime, start_proxy_with_runtime,
};
use headroom_proxy::Config;
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, PipelineStage,
};
use url::Url;
use wiremock::matchers::{method, path, path_regex, query_param};
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
async fn anthropic_batch_routes_forward_to_upstream() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .respond_with(ResponseTemplate::new(200).set_body_string("create"))
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches"))
        .and(query_param("limit", "5"))
        .respond_with(ResponseTemplate::new(200).set_body_string("list"))
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path_regex(r"^/v1/messages/batches/[^/]+$"))
        .respond_with(ResponseTemplate::new(200).set_body_string("get"))
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path_regex(r"^/v1/messages/batches/[^/]+/results$"))
        .respond_with(ResponseTemplate::new(200).set_body_string("results"))
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path_regex(r"^/v1/messages/batches/[^/]+/cancel$"))
        .respond_with(ResponseTemplate::new(200).set_body_string("cancel"))
        .mount(&upstream)
        .await;

    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    assert_eq!(
        client
            .post(format!("{}/v1/messages/batches", proxy.url()))
            .json(&serde_json::json!({"requests":[{"custom_id":"r1","params":{"model":"claude-3-5-sonnet-20241022","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}}]}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "create"
    );
    assert_eq!(
        client
            .get(format!("{}/v1/messages/batches?limit=5", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "list"
    );
    assert_eq!(
        client
            .get(format!("{}/v1/messages/batches/batch_123", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "get"
    );
    assert_eq!(
        client
            .get(format!(
                "{}/v1/messages/batches/batch_123/results",
                proxy.url()
            ))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "results"
    );
    assert_eq!(
        client
            .post(format!(
                "{}/v1/messages/batches/batch_123/cancel",
                proxy.url()
            ))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "cancel"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn anthropic_batch_routes_use_anthropic_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let direct_anthropic = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .respond_with(ResponseTemplate::new(200).set_body_string("create-direct"))
        .mount(&direct_anthropic)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches"))
        .and(query_param("limit", "5"))
        .respond_with(ResponseTemplate::new(200).set_body_string("list-direct"))
        .mount(&direct_anthropic)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.anthropic_api_url = Url::parse(&direct_anthropic.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;
    let client = reqwest::Client::new();

    assert_eq!(
        client
            .post(format!("{}/v1/messages/batches", proxy.url()))
            .json(&serde_json::json!({"requests":[{"custom_id":"r1","params":{"model":"claude-3-5-sonnet-20241022","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}}]}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "create-direct"
    );
    assert_eq!(
        client
            .get(format!("{}/v1/messages/batches?limit=5", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "list-direct"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn anthropic_batch_create_emits_compressed_input_stage_for_large_requests() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "id": "batch_123" })),
        )
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.anthropic_api_url = Url::parse(&upstream.uri()).unwrap();
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let long_text = (0..40)
        .map(|index| {
            format!(
                "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
            )
        })
        .collect::<String>();
    let requests = (0..3)
        .map(|index| {
            serde_json::json!({
                "custom_id": format!("req-{index}"),
                "params": {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 128,
                    "messages": (0..4)
                        .map(|message_index| serde_json::json!({
                            "role": "user",
                            "content": format!("{long_text}\nmessage {index}-{message_index}"),
                        }))
                        .collect::<Vec<_>>(),
                }
            })
        })
        .collect::<Vec<_>>();

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages/batches", proxy.url()))
        .json(&serde_json::json!({ "requests": requests }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;

    let recorded = events.lock().expect("poisoned");
    let compressed = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputCompressed
                && event.metadata.get("compression_status")
                    == Some(&MetadataValue::String("compressed".to_string()))
        })
        .expect("compressed input stage event");
    assert_eq!(compressed.context.operation, "proxy.anthropic_batch_create");
}

#[tokio::test]
async fn anthropic_batch_results_route_emits_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/batch_123/results"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "id": "batch_123" })),
        )
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let resp = reqwest::Client::new()
        .get(format!(
            "{}/v1/messages/batches/batch_123/results",
            proxy.url()
        ))
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
                        "anthropic_batch_results".to_string(),
                    ))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("batch_id")
                    == Some(&MetadataValue::String("batch_123".to_string()))
        })
        .expect("input routed metadata event");
    assert_eq!(
        routed_mode.context.operation,
        "proxy.anthropic_batch_results"
    );
    assert_eq!(
        routed_metadata.metadata.get("endpoint"),
        Some(&MetadataValue::String("messages/batches".to_string()))
    );
    assert_eq!(
        routed_metadata.metadata.get("result_stream"),
        Some(&MetadataValue::Bool(true))
    );
}
