//! Shared test harness: spin up a Rust proxy bound to an ephemeral port
//! pointed at an arbitrary upstream URL.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use headroom_proxy::vertex::TokenSource;
use headroom_proxy::{build_app, AppState, Config};
use tokio::sync::oneshot;
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
    start_proxy_with(upstream, |_| {}).await
}

/// Start a proxy with a customized `Config`. The closure receives a
/// mutable reference to the default `Config::for_test` and may toggle
/// flags like `compression` before the proxy is built.
#[allow(dead_code)]
pub async fn start_proxy_with<F>(upstream: &str, customize: F) -> ProxyHandle
where
    F: FnOnce(&mut Config),
{
    start_proxy_with_state(upstream, customize, |_state| {}).await
}

/// Start a proxy with both a `Config` customization and an
/// `AppState` customization. PR-D4 uses this to inject a
/// `StaticTokenSource` for Vertex tests (so they never hit real GCP).
#[allow(dead_code)]
pub async fn start_proxy_with_state<FCfg, FState>(
    upstream: &str,
    customize_config: FCfg,
    customize_state: FState,
) -> ProxyHandle
where
    FCfg: FnOnce(&mut Config),
    FState: FnOnce(&mut AppState),
{
    let upstream_url: Url = upstream.parse().expect("valid upstream url");
    let mut config = Config::for_test(upstream_url);
    customize_config(&mut config);
    let mut state = AppState::new(config.clone()).expect("app state");
    customize_state(&mut state);
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

/// Convenience: replace the default `vertex_token_source` with a
/// `StaticTokenSource` returning the supplied bearer string. Used by
/// the PR-D4 Vertex integration tests so they never hit real GCP.
#[allow(dead_code)]
pub fn install_static_token_source(state: &mut AppState, bearer: &str) {
    state.vertex_token_source = Arc::new(headroom_proxy::vertex::StaticTokenSource::new(
        bearer.to_string(),
    )) as Arc<dyn TokenSource>;
}

/// Hold a reference to the config so dead_code doesn't strip its use.
#[allow(dead_code)]
pub fn _config_ref() -> Arc<Config> {
    Arc::new(Config::for_test(Url::parse("http://127.0.0.1:1").unwrap()))
}
