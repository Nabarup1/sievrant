use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Extension, Path, Query, State},
    http::{header, HeaderValue, Method, StatusCode},
    middleware::from_fn_with_state,
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use tower_http::{
    cors::{AllowOrigin, Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use sievrant::{CalibrationExample, CalibrationSource, Corpus, Document, RetrievalConfig};

use crate::api::{
    dto::*,
    error::{ApiError, ApiJson},
    middleware::{auth_middleware, limits_middleware, TenantContext},
    state::AppState,
};

type ApiResult<T = Response> = Result<T, ApiError>;

#[derive(OpenApi)]
#[openapi(
    info(title = "Sievrant API", description = "Retrieval with a calibrated coverage guarantee."),
    paths(
        query_handler, query_full_handler, search_handler, ingest_handler, feedback_handler,
        list_corpora_handler, get_corpus_handler, list_documents_handler, delete_document_handler,
        calibration_status_handler, gaps_handler, calibrate_handler, bootstrap_handler,
        recalibrate_feedback_handler, graph_ingest_handler, tree_summary_handler,
        info_handler, health_handler
    ),
    components(schemas(
        QueryPayload, QueryResponse, IngestPayload, IngestDocument, IngestResponse,
        IngestFailureDto, FeedbackPayload, CalibrationPayload, CalibrationExampleDto,
        BootstrapPayload, CorpusRef, GraphIngestPayload, TreeSummaryPayload, AcceptedResponse
    )),
    tags((name = "sievrant", description = "Sievrant Retrieval Engine API"))
)]
pub struct ApiDoc;

// ---------------------------------------------------------------- helpers

fn retrieval_config(
    state: &AppState,
    tenant: &str,
    p: &QueryPayload,
) -> ApiResult<RetrievalConfig> {
    let q = p.query.trim();
    if q.is_empty() {
        return Err(ApiError::validation("query must not be empty"));
    }
    if q.chars().count() > MAX_QUERY_CHARS {
        return Err(ApiError::validation(format!(
            "query exceeds {MAX_QUERY_CHARS} characters"
        )));
    }
    let d = &state.services.config.retrieval;
    let coverage = p.coverage.unwrap_or(d.coverage);
    if !(coverage > 0.0 && coverage < 1.0) {
        return Err(ApiError::validation(
            "coverage must be strictly between 0 and 1",
        ));
    }
    let token_budget = p.token_budget.unwrap_or(d.token_budget);
    if token_budget == 0 || token_budget > MAX_TOKEN_BUDGET {
        return Err(ApiError::validation(format!(
            "token_budget must be 1..={MAX_TOKEN_BUDGET}"
        )));
    }
    if p.acl_groups.len() > 256 || p.acl_groups.iter().any(|g| g.trim().is_empty()) {
        return Err(ApiError::validation(
            "acl_groups: at most 256 non-empty names",
        ));
    }
    Ok(RetrievalConfig {
        coverage,
        token_budget,
        skip_cache: p.skip_cache,
        user_id: Some(tenant.to_string()),
        acl_groups: (!p.acl_groups.is_empty()).then(|| p.acl_groups.clone()),
        generate_answer: p.generate_answer.unwrap_or(true),
        ..RetrievalConfig::default()
    })
}

async fn corpus(state: &AppState, tenant: &TenantContext, name: &str) -> ApiResult<Arc<Corpus>> {
    Ok(state.corpora.get(&tenant.tenant_id, name).await?)
}

// ---------------------------------------------------------------- retrieval

/// Retrieve and (optionally) answer. Summary response.
#[utoipa::path(post, path = "/v1/query", request_body = QueryPayload,
    responses((status = 200, body = QueryResponse), (status = 401), (status = 404), (status = 422)),
    security(("bearer_auth" = [])))]
pub async fn query_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<QueryPayload>,
) -> ApiResult {
    let cfg = retrieval_config(&state, &tenant.tenant_id, &p)?;
    let c = corpus(&state, &tenant, &p.corpus).await?;
    let packet = c.retrieve(p.query.trim(), cfg).await?;
    Ok(Json(QueryResponse::from(&packet)).into_response())
}

/// Same as /v1/query but returns the full Evidence Packet (evidence, audit, guarantee, receipt).
#[utoipa::path(post, path = "/v1/query/full", request_body = QueryPayload,
    responses((status = 200, description = "EvidencePacket"), (status = 401), (status = 404), (status = 422)),
    security(("bearer_auth" = [])))]
pub async fn query_full_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<QueryPayload>,
) -> ApiResult {
    let cfg = retrieval_config(&state, &tenant.tenant_id, &p)?;
    let c = corpus(&state, &tenant, &p.corpus).await?;
    let packet = c.retrieve(p.query.trim(), cfg).await?;
    Ok(Json(packet).into_response())
}

/// Ranked candidates without gating or answer generation.
#[utoipa::path(post, path = "/v1/search", request_body = QueryPayload,
    responses((status = 200, description = "Ranked hits"), (status = 401), (status = 404)),
    security(("bearer_auth" = [])))]
pub async fn search_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<QueryPayload>,
) -> ApiResult {
    let cfg = retrieval_config(&state, &tenant.tenant_id, &p)?;
    let c = corpus(&state, &tenant, &p.corpus).await?;
    let hits = c.search(p.query.trim(), cfg).await?;
    Ok(Json(serde_json::json!({ "corpus": p.corpus, "hits": hits })).into_response())
}

// ---------------------------------------------------------------- ingestion

/// Ingest documents. Creates the corpus if needed. Changed documents replace older versions.
#[utoipa::path(post, path = "/v1/ingest", request_body = IngestPayload,
    responses((status = 202, body = IngestResponse), (status = 401), (status = 422)),
    security(("bearer_auth" = [])))]
pub async fn ingest_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<IngestPayload>,
) -> ApiResult {
    if p.documents.is_empty() || p.documents.len() > MAX_DOCS_PER_REQUEST {
        return Err(ApiError::validation(format!(
            "documents must contain 1..={MAX_DOCS_PER_REQUEST} items"
        )));
    }
    let mut docs = Vec::with_capacity(p.documents.len());
    for d in p.documents {
        if d.id.trim().is_empty() {
            return Err(ApiError::validation("every document needs a non-empty id"));
        }
        let metadata = metadata_from_json(d.metadata)
            .map_err(|e| ApiError::validation(format!("document '{}': {e}", d.id)))?;
        docs.push(Document {
            id: d.id,
            content: d.content,
            metadata,
            acl_groups: d.acl_groups,
        });
    }
    let c = state
        .corpora
        .get_or_create(&tenant.tenant_id, &p.corpus)
        .await?;
    let s = c.ingest(docs).await?;
    info!(tenant = %tenant.tenant_id, corpus = %p.corpus, ingested = s.ingested, failed = s.failed.len(), "ingest");
    Ok((
        StatusCode::ACCEPTED,
        Json(IngestResponse {
            corpus: p.corpus,
            ingested: s.ingested,
            unchanged: s.unchanged,
            failed: s.failed.len(),
            chunks: s.chunks,
            failures: s
                .failed
                .into_iter()
                .map(|f| IngestFailureDto {
                    doc_id: f.doc_id,
                    reason: f.reason,
                })
                .collect(),
        }),
    )
        .into_response())
}

/// Delete a document and its chunks from every index.
#[utoipa::path(delete, path = "/v1/corpora/{corpus}/documents/{doc_id}",
    responses((status = 200), (status = 404)), security(("bearer_auth" = [])))]
pub async fn delete_document_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path((name, doc_id)): Path<(String, String)>,
) -> ApiResult {
    let c = corpus(&state, &tenant, &name).await?;
    if !c.delete_document(&doc_id).await? {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "document_not_found",
            format!("document '{doc_id}' not found"),
        ));
    }
    Ok(
        Json(serde_json::json!({ "deleted": true, "corpus": name, "doc_id": doc_id }))
            .into_response(),
    )
}

// ---------------------------------------------------------------- corpora

/// Corpora owned by the calling tenant.
#[utoipa::path(get, path = "/v1/corpora", responses((status = 200)), security(("bearer_auth" = [])))]
pub async fn list_corpora_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
) -> ApiResult {
    let mut out = Vec::new();
    for name in state.corpora.list(&tenant.tenant_id) {
        match state.corpora.get(&tenant.tenant_id, &name).await {
            Ok(c) => out.push(serde_json::to_value(c.stats()).unwrap_or_default()),
            Err(e) => {
                tracing::warn!(corpus = %name, error = %e, "could not open corpus for listing")
            }
        }
    }
    Ok(Json(serde_json::json!({ "corpora": out })).into_response())
}

#[utoipa::path(get, path = "/v1/corpora/{corpus}", responses((status = 200), (status = 404)),
    security(("bearer_auth" = [])))]
pub async fn get_corpus_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(name): Path<String>,
) -> ApiResult {
    let c = corpus(&state, &tenant, &name).await?;
    Ok(Json(c.stats()).into_response())
}

#[utoipa::path(get, path = "/v1/corpora/{corpus}/documents", responses((status = 200), (status = 404)),
    security(("bearer_auth" = [])))]
pub async fn list_documents_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(name): Path<String>,
    Query(page): Query<PageParams>,
) -> ApiResult {
    let c = corpus(&state, &tenant, &name).await?;
    let offset = page.offset.unwrap_or(0);
    let limit = page.limit.unwrap_or(100).clamp(1, 1000);
    Ok(Json(serde_json::json!({
        "corpus": name, "offset": offset, "limit": limit,
        "documents": c.documents(offset, limit),
    }))
    .into_response())
}

/// Most frequent queries the corpus abstained on.
#[utoipa::path(get, path = "/v1/corpora/{corpus}/gaps", responses((status = 200), (status = 404)),
    security(("bearer_auth" = [])))]
pub async fn gaps_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(name): Path<String>,
    Query(page): Query<PageParams>,
) -> ApiResult {
    let c = corpus(&state, &tenant, &name).await?;
    let limit = page.limit.unwrap_or(20).clamp(1, 500);
    Ok(Json(serde_json::json!({ "corpus": name, "gaps": c.gaps(limit) })).into_response())
}

// ---------------------------------------------------------------- calibration

/// Calibration state, thresholds, held-out validation and observed coverage.
#[utoipa::path(get, path = "/v1/corpora/{corpus}/calibration", responses((status = 200), (status = 404)),
    security(("bearer_auth" = [])))]
pub async fn calibration_status_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(name): Path<String>,
    Query(q): Query<CoverageParams>,
) -> ApiResult {
    let coverage = q
        .coverage
        .unwrap_or(state.services.config.retrieval.coverage);
    if !(coverage > 0.0 && coverage < 1.0) {
        return Err(ApiError::validation(
            "coverage must be strictly between 0 and 1",
        ));
    }
    let c = corpus(&state, &tenant, &name).await?;
    Ok(Json(c.calibration_status(coverage)).into_response())
}

/// Calibrate from labelled (query, relevant documents) examples.
#[utoipa::path(post, path = "/v1/calibration", request_body = CalibrationPayload,
    responses((status = 200, description = "CalibrationReport"), (status = 404), (status = 422)),
    security(("bearer_auth" = [])))]
pub async fn calibrate_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<CalibrationPayload>,
) -> ApiResult {
    if p.examples.len() > MAX_CALIBRATION_EXAMPLES {
        return Err(ApiError::validation(format!(
            "at most {MAX_CALIBRATION_EXAMPLES} examples per request"
        )));
    }
    let source = match p.source.as_deref().map(str::to_ascii_lowercase).as_deref() {
        None | Some("labeled") | Some("labelled") => CalibrationSource::Labeled,
        Some("synthetic") => CalibrationSource::Synthetic,
        Some(other) => {
            return Err(ApiError::validation(format!(
                "source must be 'labeled' or 'synthetic', got '{other}'"
            )))
        }
    };
    let examples = p
        .examples
        .into_iter()
        .map(|e| CalibrationExample {
            query: e.query,
            relevant_doc_ids: e.relevant_doc_ids,
            relevant_chunk_ids: e.relevant_chunk_ids,
        })
        .collect();
    let c = corpus(&state, &tenant, &p.corpus).await?;
    Ok(Json(c.calibrate(examples, source).await?).into_response())
}

/// Zero-label calibration from LLM-generated questions. Requires an LLM provider.
#[utoipa::path(post, path = "/v1/calibration/bootstrap", request_body = BootstrapPayload,
    responses((status = 200, description = "CalibrationReport"), (status = 404), (status = 422)),
    security(("bearer_auth" = [])))]
pub async fn bootstrap_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<BootstrapPayload>,
) -> ApiResult {
    let n = p.n.unwrap_or(100);
    if !(20..=MAX_BOOTSTRAP_EXAMPLES).contains(&n) {
        return Err(ApiError::validation(format!(
            "n must be 20..={MAX_BOOTSTRAP_EXAMPLES}"
        )));
    }
    let c = corpus(&state, &tenant, &p.corpus).await?;
    Ok(Json(c.bootstrap_calibration(n, p.seed.unwrap_or(0x5eed)).await?).into_response())
}

/// Recalibrate from feedback labels collected via /v1/feedback.
#[utoipa::path(post, path = "/v1/calibration/feedback", request_body = CorpusRef,
    responses((status = 200, description = "CalibrationReport"), (status = 404), (status = 422)),
    security(("bearer_auth" = [])))]
pub async fn recalibrate_feedback_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<CorpusRef>,
) -> ApiResult {
    let c = corpus(&state, &tenant, &p.corpus).await?;
    Ok(Json(c.recalibrate_from_feedback().await?).into_response())
}

/// Report which documents/chunks were actually relevant to a served query.
#[utoipa::path(post, path = "/v1/feedback", request_body = FeedbackPayload,
    responses((status = 200), (status = 404)), security(("bearer_auth" = [])))]
pub async fn feedback_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<FeedbackPayload>,
) -> ApiResult {
    let c = corpus(&state, &tenant, &p.corpus).await?;
    match c.feedback(&p.query_id, &p.relevant_doc_ids, &p.relevant_chunk_ids)? {
        Some(outcome) => Ok(Json(outcome).into_response()),
        None => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "query_not_found",
            format!("query_id '{}' is unknown or too old", p.query_id),
        )),
    }
}

// ---------------------------------------------------------------- experimental

fn experimental_disabled() -> ApiError {
    ApiError::new(
        StatusCode::CONFLICT,
        "experimental_disabled",
        "graph/tree indexes are off; set SIEVRANT_EXPERIMENTAL_INDEXES=true",
    )
}

#[utoipa::path(post, path = "/v1/graph/ingest", request_body = GraphIngestPayload,
    responses((status = 202, body = AcceptedResponse), (status = 404), (status = 409)),
    security(("bearer_auth" = [])))]
pub async fn graph_ingest_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<GraphIngestPayload>,
) -> ApiResult {
    let c = corpus(&state, &tenant, &p.corpus).await?;
    let graph = c
        .orchestrator
        .indexes
        .graph
        .clone()
        .ok_or_else(experimental_disabled)?;
    for req in &p.requests {
        let relations: Vec<(String, String, String)> = req
            .relations
            .iter()
            .map(|r| (r.source.clone(), r.target.clone(), r.relation.clone()))
            .collect();
        graph
            .ingest_graph_entities(&req.chunk_id, &req.entities, &relations)
            .await?;
    }
    Ok((
        StatusCode::ACCEPTED,
        Json(AcceptedResponse {
            corpus: p.corpus,
            ingested: p.requests.len(),
        }),
    )
        .into_response())
}

#[utoipa::path(post, path = "/v1/tree/summary", request_body = TreeSummaryPayload,
    responses((status = 202, body = AcceptedResponse), (status = 404), (status = 409)),
    security(("bearer_auth" = [])))]
pub async fn tree_summary_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    ApiJson(p): ApiJson<TreeSummaryPayload>,
) -> ApiResult {
    let c = corpus(&state, &tenant, &p.corpus).await?;
    let tree = c
        .orchestrator
        .indexes
        .tree
        .clone()
        .ok_or_else(experimental_disabled)?;
    for req in &p.requests {
        tree.ingest_tree_summary(&req.node_id, req.summary.clone())
            .await?;
    }
    Ok((
        StatusCode::ACCEPTED,
        Json(AcceptedResponse {
            corpus: p.corpus,
            ingested: p.requests.len(),
        }),
    )
        .into_response())
}

// ---------------------------------------------------------------- meta

/// Active providers (no secrets) and the calling tenant.
#[utoipa::path(get, path = "/v1/info", responses((status = 200)), security(("bearer_auth" = [])))]
pub async fn info_handler(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
) -> ApiResult {
    Ok(Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "tenant": tenant.tenant_id,
        "providers": state.services.summary,
    }))
    .into_response())
}

/// Liveness probe. No authentication.
#[utoipa::path(get, path = "/v1/health", responses((status = 200)))]
pub async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

fn cors_layer(origins: Option<&str>) -> Option<CorsLayer> {
    let origins: Vec<&str> = origins?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if origins.is_empty() {
        return None;
    }
    let allow = if origins.contains(&"*") {
        AllowOrigin::from(Any)
    } else {
        AllowOrigin::list(origins.iter().filter_map(|o| HeaderValue::from_str(o).ok()))
    };
    Some(
        CorsLayer::new()
            .allow_origin(allow)
            .allow_methods([Method::GET, Method::POST, Method::DELETE])
            .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE]),
    )
}

pub fn build_router(state: AppState) -> Router {
    let srv = &state.services.config.server;
    let protected = Router::new()
        .route("/v1/query", post(query_handler))
        .route("/v1/query/full", post(query_full_handler))
        .route("/v1/search", post(search_handler))
        .route("/v1/ingest", post(ingest_handler))
        .route("/v1/feedback", post(feedback_handler))
        .route("/v1/corpora", get(list_corpora_handler))
        .route("/v1/corpora/:corpus", get(get_corpus_handler))
        .route("/v1/corpora/:corpus/documents", get(list_documents_handler))
        .route(
            "/v1/corpora/:corpus/documents/:doc_id",
            delete(delete_document_handler),
        )
        .route(
            "/v1/corpora/:corpus/calibration",
            get(calibration_status_handler),
        )
        .route("/v1/corpora/:corpus/gaps", get(gaps_handler))
        .route("/v1/calibration", post(calibrate_handler))
        .route("/v1/calibration/bootstrap", post(bootstrap_handler))
        .route(
            "/v1/calibration/feedback",
            post(recalibrate_feedback_handler),
        )
        .route("/v1/graph/ingest", post(graph_ingest_handler))
        .route("/v1/tree/summary", post(tree_summary_handler))
        .route("/v1/info", get(info_handler))
        .route_layer(from_fn_with_state(state.clone(), auth_middleware))
        .layer(from_fn_with_state(state.clone(), limits_middleware));

    let mut app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .merge(protected)
        .route("/v1/health", get(health_handler))
        .layer(DefaultBodyLimit::max(srv.max_body_bytes))
        .layer(TraceLayer::new_for_http());
    if let Some(cors) = cors_layer(srv.cors_origins.as_deref()) {
        app = app.layer(cors);
    }
    app.with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::auth::ApiKeys;
    use axum::{body::Body, http::Request};
    use sievrant::{Services, SievrantConfig};
    use tower::ServiceExt;

    const KEY_A: &str = "test-key-tenant-a-0001";
    const KEY_B: &str = "test-key-tenant-b-0002";

    /// Fully offline configuration: hashing embedder, embedding reranker, no LLM.
    fn offline_config(dir: &std::path::Path) -> SievrantConfig {
        let mut cfg = SievrantConfig::default();
        cfg.storage.data_dir = dir.to_string_lossy().into_owned();
        cfg.llm.provider = "none".into();
        cfg.fast_llm.provider = "none".into();
        cfg.embedding.provider = "local".into();
        cfg.reranker.provider = "embedding".into();
        cfg.vector.backend = "memory".into();
        cfg.audit.enabled = false;
        cfg.retrieval.query_expansion = false;
        cfg
    }

    fn app(dir: &std::path::Path) -> Router {
        let services = Arc::new(Services::from_config(offline_config(dir)).unwrap());
        let keys = ApiKeys::from_pairs(&[("tenant_a", KEY_A), ("tenant_b", KEY_B)]);
        build_router(AppState::new(services, keys))
    }

    async fn call(
        app: &Router,
        method: &str,
        uri: &str,
        key: Option<&str>,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let mut req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json");
        if let Some(k) = key {
            req = req.header("authorization", format!("Bearer {k}"));
        }
        let body = if body.is_null() {
            Body::empty()
        } else {
            Body::from(body.to_string())
        };
        let resp = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    #[tokio::test]
    async fn health_is_public_and_query_requires_auth() {
        let dir = tempfile::tempdir().unwrap();
        let app = app(dir.path());
        assert_eq!(
            call(&app, "GET", "/v1/health", None, serde_json::Value::Null)
                .await
                .0,
            StatusCode::OK
        );
        let q = serde_json::json!({"query": "x", "corpus": "c"});
        let (s, body) = call(&app, "POST", "/v1/query", None, q.clone()).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"], "unauthorized");
        assert_eq!(
            call(&app, "POST", "/v1/query", Some("wrong-key-000000000"), q)
                .await
                .0,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn validation_and_bad_json_return_json_errors() {
        let dir = tempfile::tempdir().unwrap();
        let app = app(dir.path());
        let (s, b) = call(
            &app,
            "POST",
            "/v1/query",
            Some(KEY_A),
            serde_json::json!({"query": "x", "corpus": "c", "coverage": 1.5}),
        )
        .await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(b["error"], "validation_error");
        let (s, _) = call(
            &app,
            "POST",
            "/v1/ingest",
            Some(KEY_A),
            serde_json::json!({"corpus": "../escape", "documents": [{"id": "d", "content": "x"}]}),
        )
        .await;
        assert_eq!(
            s,
            StatusCode::UNPROCESSABLE_ENTITY,
            "path-like corpus names must be rejected"
        );
        let (s, b) = call(
            &app,
            "POST",
            "/v1/query",
            Some(KEY_A),
            serde_json::json!({"corpus": "c"}),
        )
        .await;
        assert!(s.is_client_error());
        assert!(b["error"].is_string());
    }

    #[tokio::test]
    async fn missing_corpus_is_404() {
        let dir = tempfile::tempdir().unwrap();
        let app = app(dir.path());
        let (s, b) = call(
            &app,
            "POST",
            "/v1/query",
            Some(KEY_A),
            serde_json::json!({"query": "x", "corpus": "nope"}),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_eq!(b["error"], "corpus_not_found");
    }

    fn corpus_docs() -> Vec<(String, String)> {
        let topics = [
            "volcano lava magma eruption",
            "glacier ice arctic freeze",
            "violin orchestra symphony bow",
            "bitcoin blockchain ledger mining",
            "tomato garden compost seedling",
            "marathon runner stamina sprint",
            "telescope galaxy nebula astronomy",
            "espresso barista crema roast",
            "submarine sonar torpedo depth",
            "pyramid pharaoh egypt tomb",
            "penguin antarctica colony waddle",
            "chess gambit checkmate bishop",
            "hurricane cyclone storm surge",
            "sourdough yeast bread oven",
            "kangaroo marsupial outback pouch",
            "saxophone jazz improvisation reed",
            "earthquake tectonic fault seismic",
            "origami paper crane fold",
            "photosynthesis chlorophyll leaf sunlight",
            "lighthouse beacon coast keeper",
            "falcon raptor talon dive",
            "quantum entanglement qubit superposition",
            "cactus desert succulent spine",
            "mosaic tile ceramic grout",
            "beekeeper honey hive pollen",
        ];
        topics
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    format!("doc{i:02}"),
                    format!("This document is about {t}. It discusses {t} in detail."),
                )
            })
            .collect()
    }

    /// End to end over HTTP: ingest, tenant isolation, uncalibrated answer,
    /// calibrate, then a calibrated answer with a coverage guarantee.
    #[tokio::test]
    async fn ingest_calibrate_query_end_to_end_with_tenant_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let app = app(dir.path());
        let docs: Vec<serde_json::Value> = corpus_docs()
            .iter()
            .map(|(id, text)| serde_json::json!({"id": id, "content": text, "metadata": {"title": id}}))
            .collect();
        let (s, b) = call(
            &app,
            "POST",
            "/v1/ingest",
            Some(KEY_A),
            serde_json::json!({"corpus": "kb", "documents": docs}),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{b}");
        assert_eq!(b["ingested"], 25);

        // Tenant B cannot see tenant A's corpus.
        let q = serde_json::json!({"query": "volcano magma eruption", "corpus": "kb"});
        assert_eq!(
            call(&app, "POST", "/v1/query", Some(KEY_B), q.clone())
                .await
                .0,
            StatusCode::NOT_FOUND
        );
        let (_, list_b) = call(
            &app,
            "GET",
            "/v1/corpora",
            Some(KEY_B),
            serde_json::Value::Null,
        )
        .await;
        assert_eq!(list_b["corpora"].as_array().unwrap().len(), 0);

        // Before calibration: no guarantee is claimed.
        let (s, b) = call(&app, "POST", "/v1/query", Some(KEY_A), q.clone()).await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert_eq!(b["guarantee_status"], "uncalibrated");
        assert!(b["coverage_guarantee"].is_null());

        // Calibrate with one labelled query per document.
        let examples: Vec<serde_json::Value> = corpus_docs()
            .iter()
            .map(|(id, text)| {
                let words: Vec<&str> = text.split_whitespace().skip(4).take(3).collect();
                serde_json::json!({"query": words.join(" "), "relevant_doc_ids": [id]})
            })
            .collect();
        let (s, report) = call(
            &app,
            "POST",
            "/v1/calibration",
            Some(KEY_A),
            serde_json::json!({"corpus": "kb", "examples": examples}),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{report}");
        assert_eq!(report["answerable"], 25);

        let (s, status) = call(
            &app,
            "GET",
            "/v1/corpora/kb/calibration?coverage=0.9",
            Some(KEY_A),
            serde_json::Value::Null,
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(status["status"], "calibrated", "{status}");

        let (s, full) = call(
            &app,
            "POST",
            "/v1/query/full",
            Some(KEY_A),
            serde_json::json!({"query": "volcano magma eruption", "corpus": "kb", "coverage": 0.9}),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{full}");
        assert_eq!(full["guarantee"]["status"], "calibrated", "{full}");
        assert_eq!(full["coverage_guarantee"], 0.9);

        // Feedback on the served query is accepted; unknown ids are 404.
        let qid = full["query_id"].as_str().unwrap();
        let (s, fb) = call(
            &app,
            "POST",
            "/v1/feedback",
            Some(KEY_A),
            serde_json::json!({"corpus": "kb", "query_id": qid, "relevant_doc_ids": ["doc00"]}),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{fb}");
        assert_eq!(
            call(
                &app,
                "POST",
                "/v1/feedback",
                Some(KEY_A),
                serde_json::json!({"corpus": "kb", "query_id": "missing"})
            )
            .await
            .0,
            StatusCode::NOT_FOUND
        );

        // Delete.
        assert_eq!(
            call(
                &app,
                "DELETE",
                "/v1/corpora/kb/documents/doc00",
                Some(KEY_A),
                serde_json::Value::Null
            )
            .await
            .0,
            StatusCode::OK
        );
        assert_eq!(
            call(
                &app,
                "DELETE",
                "/v1/corpora/kb/documents/doc00",
                Some(KEY_A),
                serde_json::Value::Null
            )
            .await
            .0,
            StatusCode::NOT_FOUND
        );
    }
}
