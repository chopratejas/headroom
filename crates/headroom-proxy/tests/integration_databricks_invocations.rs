//! Explicit Databricks invocation route scaffolding.

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
async fn databricks_invocations_forward_to_upstream() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/serving-endpoints/demo/invocations"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&upstream)
        .await;

    let proxy = common::start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!(
            "{}/serving-endpoints/demo/invocations",
            proxy.url()
        ))
        .json(&serde_json::json!({"messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "ok");
    proxy.shutdown().await;
}

#[tokio::test]
async fn databricks_invocations_use_databricks_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let databricks_target = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/serving-endpoints/demo/invocations"))
        .respond_with(ResponseTemplate::new(200).set_body_string("direct"))
        .mount(&databricks_target)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.databricks_api_url = Url::parse(&databricks_target.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/serving-endpoints/demo/invocations",
            proxy.url()
        ))
        .json(&serde_json::json!({"messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "direct");
    proxy.shutdown().await;
}

#[tokio::test]
async fn databricks_invocations_emit_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/serving-endpoints/demo/invocations"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "id": "demo" })),
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
            "{}/serving-endpoints/demo/invocations",
            proxy.url()
        ))
        .json(&serde_json::json!({"messages":[{"role":"user","content":"hi"}]}))
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
                    == Some(&MetadataValue::String("databricks_invocations".to_string()))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("endpoint")
                    == Some(&MetadataValue::String(
                        "serving-endpoints/invocations".to_string(),
                    ))
        })
        .expect("input routed metadata event");
    assert_eq!(
        routed_mode.context.operation,
        "proxy.databricks_invocations"
    );
    assert_eq!(routed_mode.context.provider, "databricks");
    assert_eq!(routed_mode.context.model, "demo");
    assert_eq!(
        routed_metadata.metadata.get("native_route"),
        Some(&MetadataValue::Bool(true))
    );
}
