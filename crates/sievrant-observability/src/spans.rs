use tracing::{span, Level, Span};
use sievrant_core::types::{Decision, Route};

/// Defines the OpenTelemetry span tree and semantic attributes for the Sievrant pipeline.
pub struct RetrievalSpans;

impl RetrievalSpans {
    /// Root span: `sievrant.retrieve`
    pub fn root(query_id: &str, corpus: &str, user_id: Option<&str>) -> Span {
        span!(
            Level::INFO,
            "sievrant.retrieve",
            sievrant.query_id = query_id,
            sievrant.corpus = corpus,
            sievrant.user_id = user_id.unwrap_or("anonymous"),
        )
    }

    pub fn cache_check() -> Span {
        span!(Level::DEBUG, "sievrant.cache.check")
    }

    pub fn query_route() -> Span {
        span!(Level::DEBUG, "sievrant.query.route")
    }

    pub fn query_process(route: &Route) -> Span {
        span!(
            Level::DEBUG,
            "sievrant.query.process",
            sievrant.route = ?route
        )
    }

    pub fn query_expand() -> Span {
        span!(Level::DEBUG, "sievrant.query.expand")
    }

    pub fn query_embed() -> Span {
        span!(Level::DEBUG, "sievrant.query.embed")
    }

    pub fn acl_resolve() -> Span {
        span!(Level::DEBUG, "sievrant.acl.resolve")
    }

    pub fn index_search() -> Span {
        span!(Level::DEBUG, "sievrant.index.search")
    }

    pub fn index_vector() -> Span {
        span!(Level::DEBUG, "sievrant.index.vector")
    }

    pub fn index_bm25() -> Span {
        span!(Level::DEBUG, "sievrant.index.bm25")
    }

    pub fn index_tree() -> Span {
        span!(Level::DEBUG, "sievrant.index.tree")
    }

    pub fn index_graph() -> Span {
        span!(Level::DEBUG, "sievrant.index.graph")
    }

    pub fn math_rrf() -> Span {
        span!(Level::DEBUG, "sievrant.math.rrf")
    }

    pub fn rerank(model_version: &str) -> Span {
        span!(
            Level::DEBUG,
            "sievrant.rerank",
            sievrant.model_version = model_version
        )
    }

    pub fn math_submodular() -> Span {
        span!(Level::DEBUG, "sievrant.math.submodular")
    }

    pub fn math_conformal(decision: &Decision) -> Span {
        span!(
            Level::DEBUG,
            "sievrant.math.conformal",
            sievrant.decision = ?decision
        )
    }

    pub fn llm_generate(model_version: &str) -> Span {
        span!(
            Level::DEBUG,
            "sievrant.llm.generate",
            sievrant.model_version = model_version
        )
    }

    pub fn cache_store() -> Span {
        span!(Level::DEBUG, "sievrant.cache.store")
    }

    pub fn audit_log() -> Span {
        span!(Level::DEBUG, "sievrant.audit.log")
    }
}
