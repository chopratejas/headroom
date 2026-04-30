//! Shared test harness: spin up a Rust proxy bound to an ephemeral port
//! pointed at an arbitrary upstream URL.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use headroom_proxy::{build_app, AppState, Config};
use headroom_runtime::PipelineDispatcher;
use http_body_util::StreamBody;
use hyper::body::Frame;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::sync::oneshot;
use tokio_stream::wrappers::ReceiverStream;
use url::Url;

#[allow(dead_code)]
pub struct ProxyHandle {
    pub addr: SocketAddr,
    pub shutdown: Option<oneshot::Sender<()>>,
    pub task: tokio::task::JoinHandle<()>,
}

#[allow(dead_code)]
impl ProxyHandle {
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
    pub fn ws_url(&self) -> String {
        format!("ws://{}", self.addr)
    }
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
    }
}

#[allow(dead_code)]
pub async fn start_proxy(upstream: &str) -> ProxyHandle {
    start_proxy_with_runtime(upstream, Arc::new(PipelineDispatcher::new())).await
}

#[allow(dead_code)]
pub async fn start_proxy_with_config(config: Config) -> ProxyHandle {
    start_proxy_with_config_and_runtime(config, Arc::new(PipelineDispatcher::new())).await
}

#[allow(dead_code)]
pub async fn start_proxy_with_runtime(
    upstream: &str,
    runtime: Arc<PipelineDispatcher>,
) -> ProxyHandle {
    let upstream_url: Url = upstream.parse().expect("valid upstream url");
    let config = Config::for_test(upstream_url);
    start_proxy_with_config_and_runtime(config, runtime).await
}

#[allow(dead_code)]
pub async fn start_proxy_with_config_and_runtime(
    config: Config,
    runtime: Arc<PipelineDispatcher>,
) -> ProxyHandle {
    let state = AppState::new_with_runtime(config.clone(), runtime).expect("app state");
    let app = build_app(state).into_make_service_with_connect_info::<SocketAddr>();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral");
    let addr = listener.local_addr().expect("local addr");
    let (tx, rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    // Tiny delay to let the listener start accepting on slow CI.
    tokio::time::sleep(Duration::from_millis(20)).await;
    ProxyHandle {
        addr,
        shutdown: Some(tx),
        task,
    }
}

/// Hold a reference to the config so dead_code doesn't strip its use.
#[allow(dead_code)]
pub fn _config_ref() -> Arc<Config> {
    Arc::new(Config::for_test(Url::parse("http://127.0.0.1:1").unwrap()))
}

#[allow(dead_code)]
pub struct StreamingUpstreamHandle {
    pub addr: SocketAddr,
    pub requests: Arc<AtomicUsize>,
    pub task: tokio::task::JoinHandle<()>,
}

#[allow(dead_code)]
pub async fn start_streaming_upstream(
    expected_path: &'static str,
    content_type: &'static str,
    chunks: Vec<&'static str>,
    delay: Duration,
) -> StreamingUpstreamHandle {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral");
    let addr = listener.local_addr().expect("local addr");
    let requests = Arc::new(AtomicUsize::new(0));
    let task = tokio::spawn({
        let requests = requests.clone();
        async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let requests = requests.clone();
                let chunks = chunks.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(
                            io,
                            service_fn(move |req: Request<hyper::body::Incoming>| {
                                let requests = requests.clone();
                                let chunks = chunks.clone();
                                async move {
                                    assert_eq!(req.uri().path(), expected_path);
                                    requests.fetch_add(1, Ordering::Relaxed);

                                    let (tx, rx) = tokio::sync::mpsc::channel::<
                                        Result<Frame<Bytes>, std::io::Error>,
                                    >(
                                        chunks.len().max(1)
                                    );
                                    tokio::spawn(async move {
                                        for chunk in chunks {
                                            if tx
                                                .send(Ok(Frame::data(Bytes::from(chunk))))
                                                .await
                                                .is_err()
                                            {
                                                return;
                                            }
                                            tokio::time::sleep(delay).await;
                                        }
                                    });

                                    let body = StreamBody::new(ReceiverStream::new(rx));
                                    Ok::<_, std::convert::Infallible>(
                                        Response::builder()
                                            .status(200)
                                            .header("content-type", content_type)
                                            .body(body)
                                            .expect("stream response"),
                                    )
                                }
                            }),
                        )
                        .await;
                });
            }
        }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    StreamingUpstreamHandle {
        addr,
        requests,
        task,
    }
}
