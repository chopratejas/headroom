//! Canonical lifecycle emission from the Rust proxy.

mod common;

use std::sync::{Arc, Mutex};

use common::start_proxy_with_runtime;
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, CANONICAL_PIPELINE_STAGES,
};
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
async fn proxy_emits_canonical_lifecycle_for_http_requests() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/pipeline"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));

    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;
    let resp = reqwest::get(format!("{}/pipeline", proxy.url()))
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;

    let recorded = events.lock().expect("poisoned");
    let stages: Vec<_> = recorded.iter().map(|event| event.stage).collect();
    assert_eq!(stages, CANONICAL_PIPELINE_STAGES);

    assert_eq!(recorded[0].context.component, "headroom-proxy");
    assert_eq!(recorded[0].context.operation, "proxy.forward_http");
    assert_eq!(
        recorded[5].metadata.get("route_mode"),
        Some(&MetadataValue::String("passthrough".to_string()))
    );
    assert_eq!(
        recorded[6].metadata.get("compression_status"),
        Some(&MetadataValue::String("bypassed".to_string()))
    );
    assert_eq!(
        recorded[9].metadata.get("upstream_status"),
        Some(&MetadataValue::String("200".to_string()))
    );
    assert_eq!(
        recorded[10].metadata.get("response_status"),
        Some(&MetadataValue::String("200".to_string()))
    );
}
