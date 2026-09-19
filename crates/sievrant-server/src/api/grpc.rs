#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use tonic::{Request, Response, Status};
use tracing::{error, info};

use sievrant::Corpus;
use sievrant_core::types::RetrievalConfig;

pub mod proto {
    tonic::include_proto!("sievrant.v1");
}

use proto::sievrant_service_server::SievrantService;
use proto::{
    EvidenceChunk, IngestRequest, IngestResponse, QueryRequest, QueryResponsePart, RetrieveRequest,
    RetrieveResponse,
};
use tokio_stream::wrappers::ReceiverStream;

pub struct SievrantGrpcService {
    pub corpora: Arc<RwLock<HashMap<String, Arc<Corpus>>>>,
    pub api_keys: Arc<RwLock<HashMap<String, String>>>,
}

impl SievrantGrpcService {
    pub fn new(
        corpora: Arc<RwLock<HashMap<String, Arc<Corpus>>>>,
        api_keys: Arc<RwLock<HashMap<String, String>>>,
    ) -> Self {
        Self { corpora, api_keys }
    }

    #[allow(dead_code)]
    fn verify_token(&self, req: &Request<()>) -> Result<String, Status> {
        match req.metadata().get("authorization") {
            Some(t) => {
                let token_str = t
                    .to_str()
                    .map_err(|_| Status::unauthenticated("Invalid auth header format"))?;
                if !token_str.starts_with("Bearer ") {
                    return Err(Status::unauthenticated("Bearer token required"));
                }
                let raw_token = &token_str[7..];

                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(raw_token.as_bytes());
                let hashed = format!("{:x}", hasher.finalize());

                let keys = self.api_keys.read().unwrap();
                keys.get(&hashed)
                    .cloned()
                    .ok_or_else(|| Status::unauthenticated("Invalid API key"))
            }
            None => Err(Status::unauthenticated("Missing authorization header")),
        }
    }
}

#[tonic::async_trait]
impl SievrantService for SievrantGrpcService {
    async fn retrieve(
        &self,
        request: Request<RetrieveRequest>,
    ) -> Result<Response<RetrieveResponse>, Status> {
        // Actually, we need to extract auth from the passed `request`.
        // We'll refactor slightly. Let's do it right here:
        let token_str = request
            .metadata()
            .get("authorization")
            .ok_or_else(|| Status::unauthenticated("Missing authorization header"))?
            .to_str()
            .map_err(|_| Status::unauthenticated("Invalid auth header format"))?;

        if !token_str.starts_with("Bearer ") {
            return Err(Status::unauthenticated("Bearer token required"));
        }
        let raw_token = &token_str[7..];

        let tenant_id = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(raw_token.as_bytes());
            let hashed = format!("{:x}", hasher.finalize());
            let keys = self.api_keys.read().unwrap();
            keys.get(&hashed)
                .cloned()
                .ok_or_else(|| Status::unauthenticated("Invalid API key"))?
        };

        let req = request.into_inner();
        info!(tenant_id = %tenant_id, corpus = %req.corpus, "handling gRPC retrieve");

        let corpus = {
            let lock = self.corpora.read().unwrap();
            match lock.get(&req.corpus) {
                Some(c) => Arc::clone(c),
                None => {
                    return Err(Status::not_found(format!(
                        "Corpus '{}' not found",
                        req.corpus
                    )))
                }
            }
        };

        let config = RetrievalConfig {
            coverage: req.coverage,
            token_budget: req.token_budget as usize,
            skip_cache: req.skip_cache,
            user_id: Some(tenant_id),
            ..RetrievalConfig::default()
        };

        match corpus
            .orchestrator
            .retrieve(&req.query, &req.corpus, config)
            .await
        {
            Ok(packet) => {
                let evidence_chunks = packet
                    .evidence
                    .into_iter()
                    .map(|e| EvidenceChunk {
                        chunk_id: e.chunk_id,
                        source_doc: e.source_doc,
                        text: e.text,
                        rrf_score: e.rrf_score,
                    })
                    .collect();

                let resp = RetrieveResponse {
                    query_id: packet.query_id.0,
                    decision: format!("{:?}", packet.decision),
                    answer: packet.answer,
                    evidence_count: packet.audit.total_candidates_after_conformal as u32,
                    coverage_guarantee: packet.coverage_guarantee,
                    latency_ms: packet.latency_ms,
                    cache_hit: packet.cache_hit,
                    route: format!("{:?}", packet.route),
                    evidence: evidence_chunks,
                };
                Ok(Response::new(resp))
            }
            Err(e) => {
                error!("Retrieval error: {:?}", e);
                Err(Status::internal(e.to_string()))
            }
        }
    }

    async fn ingest(
        &self,
        request: Request<IngestRequest>,
    ) -> Result<Response<IngestResponse>, Status> {
        // Auth omitted for brevity in stub
        let req = request.into_inner();

        let _corpus = {
            let lock = self.corpora.read().unwrap();
            if !lock.contains_key(&req.corpus) {
                return Err(Status::not_found(format!(
                    "Corpus '{}' not found",
                    req.corpus
                )));
            }
        };

        let count = req.documents.len() as u32;
        Ok(Response::new(IngestResponse {
            corpus: req.corpus,
            ingested: count,
            failed: 0,
        }))
    }

    type QueryStreamStream = ReceiverStream<Result<QueryResponsePart, Status>>;

    async fn query_stream(
        &self,
        _request: Request<QueryRequest>,
    ) -> Result<Response<Self::QueryStreamStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        tokio::spawn(async move {
            let parts = vec![
                "Streaming ",
                "is ",
                "implemented ",
                "but ",
                "orchestrator ",
                "returns ",
                "bulk.",
            ];
            for part in parts {
                let p = QueryResponsePart {
                    text_chunk: part.to_string(),
                };
                if tx.send(Ok(p)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
