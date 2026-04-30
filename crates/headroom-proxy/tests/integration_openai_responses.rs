//! Explicit OpenAI `/v1/responses` alias scaffolding.

mod common;

use std::sync::{Arc, Mutex};

use base64::engine::general_purpose::URL_SAFE;
use base64::Engine as _;
use common::{start_proxy, start_proxy_with_config, start_proxy_with_runtime};
use futures_util::{SinkExt, StreamExt};
use headroom_proxy::Config;
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, PipelineStage,
};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::handshake::server::{
    Request as WsRequest, Response as WsResponse,
};
use tokio_tungstenite::tungstenite::Message;
use url::Url;
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct RecordingPlugin {
    events: Arc<Mutex<Vec<PipelineEvent>>>,
}

impl PipelinePlugin for RecordingPlugin {
    fn on_event(&self, event: &mut PipelineEvent) {
        self.events.lock().expect("poisoned").push(event.clone());
    }
}

fn jwt_with_chatgpt_account_id(account_id: &str) -> String {
    let header = URL_SAFE
        .encode(br#"{"alg":"none","typ":"JWT"}"#)
        .trim_end_matches('=')
        .to_string();
    let payload = URL_SAFE
        .encode(
            format!(r#"{{"https://api.openai.com/auth":{{"chatgpt_account_id":"{account_id}"}}}}"#)
                .as_bytes(),
        )
        .trim_end_matches('=')
        .to_string();
    format!("{header}.{payload}.")
}

async fn start_ws_echo_upstream() -> (
    std::net::SocketAddr,
    Arc<Mutex<Vec<String>>>,
    tokio::sync::oneshot::Sender<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen_paths = Arc::new(Mutex::new(Vec::new()));
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
    tokio::spawn({
        let seen_paths = seen_paths.clone();
        async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { continue };
                        let seen_paths = seen_paths.clone();
                        tokio::spawn(async move {
                            let callback = move |request: &WsRequest, response: WsResponse| {
                                seen_paths
                                    .lock()
                                    .expect("poisoned")
                                    .push(request.uri().path().to_string());
                                Ok(response)
                            };
                            let Ok(ws) = tokio_tungstenite::accept_hdr_async(stream, callback).await else {
                                return;
                            };
                            let (mut sink, mut source) = ws.split();
                            while let Some(Ok(message)) = source.next().await {
                                if sink.send(message).await.is_err() {
                                    break;
                                }
                            }
                        });
                    }
                }
            }
        }
    });
    (addr, seen_paths, stop_tx)
}

async fn start_ws_echo_upstream_with_openai_beta() -> (
    std::net::SocketAddr,
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<Option<String>>>>,
    tokio::sync::oneshot::Sender<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen_paths = Arc::new(Mutex::new(Vec::new()));
    let seen_beta_headers = Arc::new(Mutex::new(Vec::new()));
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
    tokio::spawn({
        let seen_paths = seen_paths.clone();
        let seen_beta_headers = seen_beta_headers.clone();
        async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { continue };
                        let seen_paths = seen_paths.clone();
                        let seen_beta_headers = seen_beta_headers.clone();
                        tokio::spawn(async move {
                            let callback = move |request: &WsRequest, response: WsResponse| {
                                seen_paths
                                    .lock()
                                    .expect("poisoned")
                                    .push(request.uri().path().to_string());
                                seen_beta_headers
                                    .lock()
                                    .expect("poisoned")
                                    .push(
                                        request
                                            .headers()
                                            .get("openai-beta")
                                            .and_then(|value| value.to_str().ok())
                                            .map(str::to_string),
                                    );
                                Ok(response)
                            };
                            let Ok(ws) = tokio_tungstenite::accept_hdr_async(stream, callback).await else {
                                return;
                            };
                            let (mut sink, mut source) = ws.split();
                            while let Some(Ok(message)) = source.next().await {
                                if sink.send(message).await.is_err() {
                                    break;
                                }
                            }
                        });
                    }
                }
            }
        }
    });
    (addr, seen_paths, seen_beta_headers, stop_tx)
}

#[tokio::test]
async fn openai_responses_alias_forwards_to_canonical_path() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_json(serde_json::json!({
            "model": "gpt-5.4",
            "input": "hello"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "resp_1",
                    "object": "response"
                })),
        )
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/backend-api/codex/responses", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-5.4",
            "input": "hello"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["id"], "resp_1");
    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_responses_alias_uses_openai_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let direct_openai = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_json(serde_json::json!({
            "model": "gpt-5.4",
            "input": "hello"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "resp_direct",
                    "object": "response"
                })),
        )
        .mount(&direct_openai)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.openai_api_url = Url::parse(&direct_openai.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/backend-api/codex/responses", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-5.4",
            "input": "hello"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["id"], "resp_direct");
    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_responses_subpath_alias_preserves_query_and_derives_chatgpt_account_id() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses/compact"))
        .and(query_param("trace", "jwt"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "resp_2",
                    "object": "response"
                })),
        )
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1/codex/responses/compact?trace=jwt",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "model": "gpt-5.4",
            "input": "hello"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_responses_alias_emits_route_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses/compact"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "resp_3",
                    "object": "response"
                })),
        )
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/backend-api/responses/compact", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-5.4",
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
                    == Some(&MetadataValue::String("openai_responses".to_string()))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("canonical_path")
                    == Some(&MetadataValue::String("/v1/responses/compact".to_string()))
        })
        .expect("input routed metadata event");
    assert_eq!(routed_mode.context.operation, "proxy.openai_responses");
    assert_eq!(
        routed_metadata.metadata.get("codex_alias"),
        Some(&MetadataValue::Bool(true))
    );
    assert_eq!(
        routed_metadata.metadata.get("chatgpt_auth"),
        Some(&MetadataValue::Bool(false))
    );
}

#[tokio::test]
async fn openai_responses_websocket_alias_forwards_to_canonical_path() {
    let (upstream_addr, seen_paths, stop_tx) = start_ws_echo_upstream().await;
    let proxy = start_proxy(
        &Url::parse(&format!("http://{upstream_addr}"))
            .unwrap()
            .to_string(),
    )
    .await;

    let (mut ws, _) =
        tokio_tungstenite::connect_async(format!("{}/backend-api/codex/responses", proxy.ws_url()))
            .await
            .unwrap();
    ws.send(Message::Text("hello".into())).await.unwrap();
    let echoed = ws.next().await.unwrap().unwrap();
    match echoed {
        Message::Text(text) => assert_eq!(text.as_str(), "hello"),
        other => panic!("expected text, got {other:?}"),
    }
    ws.close(None).await.unwrap();

    proxy.shutdown().await;
    let _ = stop_tx.send(());

    let paths = seen_paths.lock().expect("poisoned");
    assert_eq!(paths.as_slice(), ["/v1/responses"]);
}

#[tokio::test]
async fn openai_responses_websocket_alias_uses_openai_target_url() {
    let (upstream_addr, seen_paths, stop_tx) = start_ws_echo_upstream().await;
    let mut config = Config::for_test(Url::parse("http://127.0.0.1:1").unwrap());
    config.openai_api_url = Url::parse(&format!("http://{upstream_addr}")).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let (mut ws, _) =
        tokio_tungstenite::connect_async(format!("{}/backend-api/codex/responses", proxy.ws_url()))
            .await
            .unwrap();
    ws.send(Message::Text("hello".into())).await.unwrap();
    let echoed = ws.next().await.unwrap().unwrap();
    match echoed {
        Message::Text(text) => assert_eq!(text.as_str(), "hello"),
        other => panic!("expected text, got {other:?}"),
    }
    ws.close(None).await.unwrap();

    proxy.shutdown().await;
    let _ = stop_tx.send(());

    let paths = seen_paths.lock().expect("poisoned");
    assert_eq!(paths.as_slice(), ["/v1/responses"]);
}

#[tokio::test]
async fn openai_responses_websocket_alias_injects_openai_beta_header_when_missing() {
    let (upstream_addr, seen_paths, seen_beta_headers, stop_tx) =
        start_ws_echo_upstream_with_openai_beta().await;
    let proxy = start_proxy(
        &Url::parse(&format!("http://{upstream_addr}"))
            .unwrap()
            .to_string(),
    )
    .await;

    let (mut ws, _) =
        tokio_tungstenite::connect_async(format!("{}/backend-api/codex/responses", proxy.ws_url()))
            .await
            .unwrap();
    ws.send(Message::Text("hello".into())).await.unwrap();
    let echoed = ws.next().await.unwrap().unwrap();
    match echoed {
        Message::Text(text) => assert_eq!(text.as_str(), "hello"),
        other => panic!("expected text, got {other:?}"),
    }
    ws.close(None).await.unwrap();

    proxy.shutdown().await;
    let _ = stop_tx.send(());

    let paths = seen_paths.lock().expect("poisoned");
    assert_eq!(paths.as_slice(), ["/v1/responses"]);
    let beta_headers = seen_beta_headers.lock().expect("poisoned");
    assert_eq!(
        beta_headers.as_slice(),
        [Some("responses_websockets=2026-02-06".to_string())]
    );
}

#[tokio::test]
async fn openai_responses_websocket_alias_preserves_existing_openai_beta_header() {
    let (upstream_addr, _seen_paths, seen_beta_headers, stop_tx) =
        start_ws_echo_upstream_with_openai_beta().await;
    let proxy = start_proxy(
        &Url::parse(&format!("http://{upstream_addr}"))
            .unwrap()
            .to_string(),
    )
    .await;

    let mut request = format!("{}/backend-api/codex/responses", proxy.ws_url())
        .into_client_request()
        .unwrap();
    request.headers_mut().insert(
        "OpenAI-Beta",
        "responses_websockets=custom-version".parse().unwrap(),
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(request).await.unwrap();
    ws.send(Message::Text("hello".into())).await.unwrap();
    let echoed = ws.next().await.unwrap().unwrap();
    match echoed {
        Message::Text(text) => assert_eq!(text.as_str(), "hello"),
        other => panic!("expected text, got {other:?}"),
    }
    ws.close(None).await.unwrap();

    proxy.shutdown().await;
    let _ = stop_tx.send(());

    let beta_headers = seen_beta_headers.lock().expect("poisoned");
    assert_eq!(
        beta_headers.as_slice(),
        [Some("responses_websockets=custom-version".to_string())]
    );
}
