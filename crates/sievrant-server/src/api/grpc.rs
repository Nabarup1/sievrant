#![allow(clippy::result_large_err)]

//! gRPC service. Shares authentication and the tenant-scoped corpus registry
//! with the REST API.

use tonic::{Code, Request, Response, Status};
use tracing::info;

use sievrant::Document;
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

use crate::api::{
    auth::AuthError,
    dto::{metadata_from_json, wire, MAX_DOCS_PER_REQUEST, MAX_QUERY_CHARS, MAX_TOKEN_BUDGET},
    error::ApiError,
    state::AppState,
};

pub struct SievrantGrpcService {
    state: AppState,
}

impl SievrantGrpcService {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    fn tenant<T>(&self, req: &Request<T>) -> Result<String, Status> {
        let header = req
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok());
        self.state.keys.authenticate(header).map_err(|e| match e {
            AuthError::Missing => {
                Status::unauthenticated("authorization: Bearer <api key> required")
            }
            AuthError::Invalid => Status::unauthenticated("API key is invalid or revoked"),
        })
    }
}

fn to_status(e: impl Into<ApiError>) -> Status {
    let e: ApiError = e.into();
    let code = match e.status.as_u16() {
        401 => Code::Unauthenticated,
        403 => Code::PermissionDenied,
        404 => Code::NotFound,
        409 => Code::FailedPrecondition,
        400 | 413 | 422 => Code::InvalidArgument,
        502 | 503 => Code::Unavailable,
        504 => Code::DeadlineExceeded,
        _ => Code::Internal,
    };
    Status::new(code, e.message)
}

#[tonic::async_trait]
impl SievrantService for SievrantGrpcService {
    async fn retrieve(
        &self,
        request: Request<RetrieveRequest>,
    ) -> Result<Response<RetrieveResponse>, Status> {
        let tenant = self.tenant(&request)?;
        let req = request.into_inner();
        let query = req.query.trim();
        if query.is_empty() || query.chars().count() > MAX_QUERY_CHARS {
            return Err(Status::invalid_argument(format!(
                "query must be 1..={MAX_QUERY_CHARS} characters"
            )));
        }
        let defaults = &self.state.services.config.retrieval;
        // proto3 scalars default to 0: treat 0 as "use server default".
        let coverage = if req.coverage == 0.0 {
            defaults.coverage
        } else {
            req.coverage
        };
        if !(coverage > 0.0 && coverage < 1.0) {
            return Err(Status::invalid_argument(
                "coverage must be strictly between 0 and 1",
            ));
        }
        let token_budget = match req.token_budget as usize {
            0 => defaults.token_budget,
            n if n > MAX_TOKEN_BUDGET => {
                return Err(Status::invalid_argument(format!(
                    "token_budget must be <= {MAX_TOKEN_BUDGET}"
                )))
            }
            n => n,
        };
        info!(tenant = %tenant, corpus = %req.corpus, "gRPC retrieve");

        let corpus = self
            .state
            .corpora
            .get(&tenant, &req.corpus)
            .await
            .map_err(to_status)?;
        let config = RetrievalConfig {
            coverage,
            token_budget,
            skip_cache: req.skip_cache,
            user_id: Some(tenant),
            acl_groups: (!req.acl_groups.is_empty()).then_some(req.acl_groups),
            ..RetrievalConfig::default()
        };
        let packet = corpus.retrieve(query, config).await.map_err(to_status)?;
        Ok(Response::new(RetrieveResponse {
            query_id: packet.query_id.0.clone(),
            decision: wire(&packet.decision),
            answer: packet.answer.clone(),
            evidence_count: packet.evidence.len() as u32,
            coverage_guarantee: packet.coverage_guarantee,
            latency_ms: packet.latency_ms,
            cache_hit: packet.cache_hit,
            route: wire(&packet.route),
            guarantee_status: wire(&packet.guarantee.status),
            evidence: packet
                .evidence
                .into_iter()
                .map(|e| EvidenceChunk {
                    chunk_id: e.chunk_id,
                    source_doc: e.source_doc,
                    text: e.text,
                    rrf_score: e.rrf_score,
                })
                .collect(),
        }))
    }

    async fn ingest(
        &self,
        request: Request<IngestRequest>,
    ) -> Result<Response<IngestResponse>, Status> {
        let tenant = self.tenant(&request)?;
        let req = request.into_inner();
        if req.documents.is_empty() || req.documents.len() > MAX_DOCS_PER_REQUEST {
            return Err(Status::invalid_argument(format!(
                "documents must contain 1..={MAX_DOCS_PER_REQUEST} items"
            )));
        }
        let mut docs = Vec::with_capacity(req.documents.len());
        for d in req.documents {
            let meta_json = if d.metadata_json.trim().is_empty() {
                None
            } else {
                Some(serde_json::from_str(&d.metadata_json).map_err(|e| {
                    Status::invalid_argument(format!("document '{}': metadata_json: {e}", d.id))
                })?)
            };
            let metadata = metadata_from_json(meta_json)
                .map_err(|e| Status::invalid_argument(format!("document '{}': {e}", d.id)))?;
            docs.push(Document {
                id: d.id,
                content: d.content,
                metadata,
                acl_groups: d.acl_groups,
            });
        }
        let corpus = self
            .state
            .corpora
            .get_or_create(&tenant, &req.corpus)
            .await
            .map_err(to_status)?;
        let summary = corpus.ingest(docs).await.map_err(to_status)?;
        Ok(Response::new(IngestResponse {
            corpus: req.corpus,
            ingested: summary.ingested as u32,
            failed: summary.failed.len() as u32,
        }))
    }

    type QueryStreamStream = ReceiverStream<Result<QueryResponsePart, Status>>;

    async fn query_stream(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<Self::QueryStreamStream>, Status> {
        self.tenant(&request)?;
        Err(Status::unimplemented(
            "streaming is not implemented yet; use Retrieve",
        ))
    }
}
