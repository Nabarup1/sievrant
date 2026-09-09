pub mod api;

use std::collections::HashMap;
use tracing::info;
use sievrant_observability::{init_observability, ObservabilityConfig};
use tonic::transport::Server as TonicServer;

use api::rest::{build_router, AppState};
use api::grpc::{SievrantGrpcService, proto::sievrant_service_server::SievrantServiceServer};

#[tokio::main]
async fn main() {
    init_observability(&ObservabilityConfig::default()).unwrap();

    let mut keys = HashMap::new();
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"sk-test");
    let hash = format!("{:x}", hasher.finalize());
    keys.insert(hash, "tenant-test".to_string());
    let state = AppState::new(keys);

    let rest_addr = "0.0.0.0:8080";
    let grpc_addr = "0.0.0.0:50051".parse().unwrap();

    let rest_state = state.clone();
    
    // Spawn REST Server
    let rest_task = tokio::spawn(async move {
        info!("sievrant-server REST listening on {rest_addr}");
        let listener = tokio::net::TcpListener::bind(rest_addr).await.unwrap();
        axum::serve(listener, build_router(rest_state)).await.unwrap();
    });

    // Spawn gRPC Server
    let grpc_task = tokio::spawn(async move {
        info!("sievrant-server gRPC listening on {grpc_addr}");
        let grpc_service = SievrantGrpcService::new(state.corpora.clone(), state.api_keys.clone());
        TonicServer::builder()
            .add_service(SievrantServiceServer::new(grpc_service))
            .serve(grpc_addr)
            .await
            .unwrap();
    });

    let _ = tokio::join!(rest_task, grpc_task);
}
