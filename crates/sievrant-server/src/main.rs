pub mod api;

use std::{net::SocketAddr, sync::Arc, time::Duration};

use anyhow::Context;
use sievrant::{Services, SievrantConfig};
use sievrant_observability::{init_observability, ObservabilityConfig};
use tonic::transport::Server as TonicServer;
use tracing::{info, warn};

use api::{
    auth::ApiKeys,
    grpc::{proto::sievrant_service_server::SievrantServiceServer, SievrantGrpcService},
    rest::build_router,
    state::AppState,
};

async fn shutdown_signal(mut rx: tokio::sync::watch::Receiver<bool>) {
    let _ = rx.wait_for(|stop| *stop).await;
}

async fn wait_for_termination() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // A local .env is convenient in development; in containers use real env vars.
    let _ = dotenvy::dotenv();

    let config = SievrantConfig::load().context("failed to load configuration")?;
    config.validate().context("invalid configuration")?;

    init_observability(&ObservabilityConfig {
        enable_json_logs: config.observability.log_format.eq_ignore_ascii_case("json"),
        otlp_endpoint: config
            .observability
            .otlp_enabled
            .then(|| config.observability.otlp_endpoint.clone()),
    })
    .map_err(|e| anyhow::anyhow!("failed to initialise logging: {e}"))?;

    let keys = ApiKeys::from_config(&config.server).map_err(|e| anyhow::anyhow!(e))?;
    if keys.is_disabled() {
        warn!("SIEVRANT_AUTH_DISABLED=true: every request is accepted as tenant 'default'. Never use this in production.");
    } else {
        info!(
            tenants = keys.tenant_count(),
            "API key authentication enabled"
        );
    }

    std::fs::create_dir_all(&config.storage.data_dir)
        .with_context(|| format!("cannot create data dir '{}'", config.storage.data_dir))?;

    let srv = config.server.clone();
    let services =
        Arc::new(Services::from_config(config).context("failed to initialise providers")?);
    info!(providers = ?services.summary, data_dir = %services.config.storage.data_dir, "providers ready");

    let state = AppState::new(services, keys);
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);

    let rest_addr: SocketAddr =
        format!("{}:{}", srv.host, srv.port)
            .parse()
            .with_context(|| {
                format!(
                    "invalid SIEVRANT_HOST/SIEVRANT_PORT '{}:{}'",
                    srv.host, srv.port
                )
            })?;
    let listener = tokio::net::TcpListener::bind(rest_addr)
        .await
        .with_context(|| format!("cannot bind REST port {rest_addr}"))?;
    info!("REST listening on http://{rest_addr} (docs at /swagger-ui)");
    let router = build_router(state.clone());
    let rest_rx = stop_rx.clone();
    let rest = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal(rest_rx))
            .await
    });

    let grpc = if srv.grpc_enabled {
        let grpc_addr: SocketAddr = format!("{}:{}", srv.host, srv.grpc_port)
            .parse()
            .with_context(|| format!("invalid SIEVRANT_GRPC_PORT '{}'", srv.grpc_port))?;
        info!("gRPC listening on {grpc_addr}");
        let svc = SievrantServiceServer::new(SievrantGrpcService::new(state.clone()));
        let rx = stop_rx.clone();
        Some(tokio::spawn(
            TonicServer::builder()
                .timeout(Duration::from_secs(srv.request_timeout_secs.max(1)))
                .concurrency_limit_per_connection(64)
                .add_service(svc)
                .serve_with_shutdown(grpc_addr, shutdown_signal(rx)),
        ))
    } else {
        None
    };

    wait_for_termination().await;
    info!("shutdown signal received; draining in-flight requests");
    let _ = stop_tx.send(true);

    rest.await
        .context("REST task panicked")?
        .context("REST server error")?;
    if let Some(g) = grpc {
        g.await
            .context("gRPC task panicked")?
            .context("gRPC server error")?;
    }
    info!("shutdown complete");
    Ok(())
}
