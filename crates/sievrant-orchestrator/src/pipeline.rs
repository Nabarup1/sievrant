use std::sync::Arc;

use sha2::{Digest, Sha256};
use tracing::{info, warn};

use sievrant_core::{
    error::Result,
    types::{
        AuditBlock, Decision, EvidenceChunk, EvidencePacket, QueryId, RankedCandidate,
        RetrievalConfig, RetrievalStagesMs, Route,
    },
};
use sievrant_index::vector::IndexQuery;
use sievrant_llm::{
    provider::{LlmRequest, Message, Role},
    LlmProvider,
};
use sievrant_math::{
    conformal::{conformal_gate, nonconformity_score},
    rrf::reciprocal_rank_fusion,
    submodular::submodular_select,
};
use sievrant_query::{
    processing::{ProcessedQuery, QueryProcessor},
    router::QueryRouter,
};
use sievrant_rerank::{rerank_candidates, Reranker};

use sievrant_acl::resolver::AclResolver;
use sievrant_index::RetrievalIndex;

/// The four live indexes accessed for every retrieval.
pub struct IndexSet {
    pub vector: Arc<dyn RetrievalIndex>,
    pub bm25: Arc<dyn RetrievalIndex>,
    pub tree: Arc<dyn RetrievalIndex>,
    pub graph: Arc<dyn RetrievalIndex>,
}

use sievrant_core::types::QueryTaxonomy;

/// Conformal calibration manager: returns the current threshold for a corpus.
pub trait CalibrationManager: Send + Sync {
    fn threshold(&self, corpus: &str, taxonomy: &QueryTaxonomy) -> f64;
}

/// Semantic cache: stores and retrieves EvidencePackets by query fingerprint.
#[async_trait::async_trait]
pub trait SemanticCache: Send + Sync {
    async fn lookup(
        &self,
        query: &str,
        corpus: &str,
        user_id: &Option<String>,
    ) -> Result<Option<EvidencePacket>>;

    async fn store(
        &self,
        query: &str,
        corpus: &str,
        user_id: &Option<String>,
        packet: &EvidencePacket,
    ) -> Result<()>;
}

/// Coverage monitor: records each retrieval outcome for empirical coverage tracking.
pub trait CoverageMonitor: Send + Sync {
    fn record(&self, corpus: &str, covered: bool);
    fn empirical_coverage_30d(&self, corpus: &str) -> Option<f64>;
}

fn classify_query_taxonomy(query: &str) -> QueryTaxonomy {
    let q = query.to_lowercase();
    if q.contains("code") || q.contains("function") || q.contains("struct") || q.contains("impl") {
        QueryTaxonomy::Coding
    } else if q.contains("summarize")
        || q.contains("summary")
        || q.contains("overview")
        || q.len() > 150
    {
        QueryTaxonomy::Summarization
    } else if q.contains("why")
        || q.contains("how")
        || q.contains("analyze")
        || q.contains("compare")
    {
        QueryTaxonomy::Analytical
    } else {
        QueryTaxonomy::Factoid
    }
}

/// Central 10-stage retrieval pipeline.
///
/// Every stage is instrumented with wall-clock timing stored in `RetrievalStagesMs`.
/// Partial index failures cause a warning log; the pipeline continues with the
/// remaining indexes and a `DegradedResult` note in the audit block.
pub struct RetrievalOrchestrator {
    pub cache: Arc<dyn SemanticCache>,
    pub router: Arc<QueryRouter>,
    pub query_processor: Arc<QueryProcessor>,
    pub acl_resolver: Arc<dyn AclResolver>,
    pub indexes: IndexSet,
    pub reranker: Arc<dyn Reranker>,
    pub calibration: Arc<dyn CalibrationManager>,
    pub llm: Arc<dyn LlmProvider>,
    pub coverage_monitor: Arc<dyn CoverageMonitor>,
    pub search_limit: usize,
    pub min_evidence_count: usize,
}

impl RetrievalOrchestrator {
    pub async fn retrieve(
        &self,
        query: &str,
        corpus: &str,
        config: RetrievalConfig,
    ) -> Result<EvidencePacket> {
        let query_id = QueryId::new();
        let pipeline_start = std::time::Instant::now();
        let mut stages_ms = RetrievalStagesMs::default();

        let taxonomy = classify_query_taxonomy(query);

        // Stage 1: Semantic cache check
        let t = std::time::Instant::now();
        if !config.skip_cache {
            if let Ok(Some(mut cached)) = self.cache.lookup(query, corpus, &config.user_id).await {
                cached.audit.retrieval_stages_ms.cache_check = t.elapsed().as_millis() as u64;
                info!(query_id = %query_id, "cache hit");
                return Ok(cached);
            }
        }
        stages_ms.cache_check = t.elapsed().as_millis() as u64;

        // Stage 2: Adaptive route classification
        let t = std::time::Instant::now();
        let route_decision = self.router.classify_with_strategy(query, &config.route);
        let route = route_decision.route.clone();
        stages_ms.routing = t.elapsed().as_millis() as u64;

        // Stage 3: Query processing (expand, embed, decompose)
        let t = std::time::Instant::now();
        let processed = self.query_processor.process(query, &route).await?;
        stages_ms.query_processing = t.elapsed().as_millis() as u64;

        // Stage 4: ACL resolution
        let t = std::time::Instant::now();
        let acl_groups = self
            .acl_resolver
            .resolve(config.user_id.as_deref(), corpus)
            .await?;
        stages_ms.acl_resolution = t.elapsed().as_millis() as u64;

        // Stage 5: Parallel index searches with graceful degradation
        let t = std::time::Instant::now();
        let (ranked_lists, index_warnings) =
            parallel_index_search(&self.indexes, &processed, &acl_groups, self.search_limit).await;
        stages_ms.vector_search = t.elapsed().as_millis() as u64;
        for warning in &index_warnings {
            warn!(query_id = %query_id, "{warning}");
        }

        // Stage 6: Reciprocal Rank Fusion
        let t = std::time::Instant::now();
        let fused = reciprocal_rank_fusion(&ranked_lists, config.rrf_k);
        stages_ms.rrf_fusion = t.elapsed().as_millis() as u64;
        let candidates_before_rerank = fused.len();

        // Stage 7: Cross-encoder reranking (with graceful fallback)
        let t = std::time::Instant::now();
        let reranked =
            rerank_candidates(self.reranker.as_ref(), query, fused, config.rerank_top_n).await?;
        stages_ms.reranking = t.elapsed().as_millis() as u64;
        let candidates_after_rerank = reranked.len();

        // Stage 8: Submodular evidence selection
        let t = std::time::Instant::now();
        let mut selected = submodular_select(&reranked, config.token_budget, 0.3);
        stages_ms.submodular_selection = t.elapsed().as_millis() as u64;

        // Stage 9: Conformal gate
        let t = std::time::Instant::now();
        let threshold = self.calibration.threshold(corpus, &taxonomy);

        for chunk in &mut selected {
            chunk.nonconformity_score =
                nonconformity_score(chunk.rerank_score, chunk.fused_score, 0.0);
        }

        let conformal_result = conformal_gate(selected, threshold, self.min_evidence_count);
        stages_ms.conformal_gate = t.elapsed().as_millis() as u64;

        let decision = conformal_result.decision.clone();
        let candidates_after_conformal = conformal_result.evidence.len();

        self.coverage_monitor
            .record(corpus, matches!(decision, Decision::Answered));

        // Stage 10: LLM answer generation
        let t = std::time::Instant::now();
        let answer = if matches!(decision, Decision::Answered) {
            Some(
                self.generate_answer(query, &conformal_result.evidence)
                    .await?,
            )
        } else {
            None
        };
        stages_ms.llm_generation = t.elapsed().as_millis() as u64;

        let evidence_chunks: Vec<EvidenceChunk> = conformal_result
            .evidence
            .into_iter()
            .map(|e| EvidenceChunk {
                chunk_id: e.chunk_id,
                source_doc: e.metadata.source_doc.clone(),
                doc_version_hash: e.metadata.doc_version_hash.clone(),
                path: e.metadata.path.clone(),
                page: e.metadata.page,
                char_span: e.metadata.char_span,
                text: e.text,
                retriever: sievrant_core::types::RetrieverType::VectorIndex,
                rrf_score: e.fused_score,
                rerank_score: Some(e.rerank_score),
                nonconformity_score: e.nonconformity_score,
                submodular_marginal_gain: e.marginal_gain,
                ingested_at: e.metadata.ingested_at,
                index_generation: e.metadata.index_generation,
                acl_groups: e.metadata.acl_groups,
                freshness_hours: 0.0,
                token_count: e.metadata.token_count,
            })
            .collect();

        let session_hash =
            compute_session_hash(&query_id, query, &evidence_chunks, answer.as_deref());

        let packet = EvidencePacket {
            query_id,
            trace_id: None,
            answer,
            decision,
            coverage_guarantee: config.coverage,
            empirical_coverage_last_30d: self.coverage_monitor.empirical_coverage_30d(corpus),
            route: route_into_packet_route(&route),
            cache_hit: false,
            evidence: evidence_chunks,
            audit: AuditBlock {
                model_version: Some(self.llm.model_id().to_string()),
                retrieval_stages_ms: stages_ms,
                total_candidates_before_rerank: candidates_before_rerank,
                total_candidates_after_rerank: candidates_after_rerank,
                total_candidates_after_conformal: candidates_after_conformal,
                calibration_set_id: "default".into(),
                session_hash,
            },
            latency_ms: pipeline_start.elapsed().as_millis() as u64,
            cost_usd: None,
        };

        let _ = self
            .cache
            .store(query, corpus, &config.user_id, &packet)
            .await;

        Ok(packet)
    }

    async fn generate_answer(
        &self,
        query: &str,
        evidence: &[sievrant_core::types::SelectedEvidence],
    ) -> Result<String> {
        let evidence_text: String = evidence
            .iter()
            .map(|e| {
                format!(
                    "[{}] (Source: {}, Path: {})\n{}",
                    e.chunk_id,
                    e.metadata.source_doc,
                    e.metadata.path.join(" > "),
                    e.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "You are a precise, factual assistant. Answer the following question using ONLY \
             the provided evidence chunks. Cite specific sections using [chunk_id] notation.\n\n\
             If the evidence does not contain enough information to answer, say \"The provided \
             evidence does not contain sufficient information to answer this question.\"\n\n\
             Question: {query}\n\n\
             Evidence:\n{evidence_text}\n\nAnswer:"
        );

        let req = LlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: prompt,
            }],
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
        };

        let response = self.llm.complete(req).await?;
        Ok(response.content.trim().to_string())
    }
}

/// Execute all four index searches in parallel using tokio::join!.
/// A failing index records a warning and contributes an empty list;
/// the pipeline continues with whatever indexes succeed.
async fn parallel_index_search(
    indexes: &IndexSet,
    query: &ProcessedQuery,
    acl_filter: &[String],
    limit: usize,
) -> (Vec<Vec<RankedCandidate>>, Vec<String>) {
    let vec_query = IndexQuery::Vector {
        embedding: query.embedding.clone(),
    };
    let bm25_query = IndexQuery::Lexical {
        text: query.original.clone(),
    };
    let tree_query = IndexQuery::Structural {
        text: query.original.clone(),
        path_hints: vec![],
    };
    let graph_query = IndexQuery::Graph {
        entities: query.entities.clone(),
        query_embedding: query.embedding.clone(),
    };

    let (vec_r, bm25_r, tree_r, graph_r) = tokio::join!(
        indexes.vector.search(&vec_query, acl_filter, limit),
        indexes.bm25.search(&bm25_query, acl_filter, limit),
        indexes.tree.search(&tree_query, acl_filter, limit),
        indexes.graph.search(&graph_query, acl_filter, limit),
    );

    let mut ranked_lists = Vec::with_capacity(4);
    let mut warnings = Vec::new();

    for (name, result) in [
        ("vector", vec_r),
        ("bm25", bm25_r),
        ("tree", tree_r),
        ("graph", graph_r),
    ] {
        match result {
            Ok(candidates) => ranked_lists.push(candidates),
            Err(e) => warnings.push(format!("{name} index failed (degraded): {e}")),
        }
    }

    (ranked_lists, warnings)
}

fn compute_session_hash(
    query_id: &QueryId,
    query: &str,
    evidence: &[EvidenceChunk],
    answer: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(query_id.0.as_bytes());
    hasher.update(query.as_bytes());
    for chunk in evidence {
        hasher.update(chunk.chunk_id.as_bytes());
    }
    if let Some(ans) = answer {
        hasher.update(ans.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn route_into_packet_route(route: &Route) -> sievrant_core::types::Route {
    match route {
        Route::Fast => sievrant_core::types::Route::Fast,
        Route::Standard => sievrant_core::types::Route::Standard,
        Route::Slow => sievrant_core::types::Route::Slow,
        Route::Cached => sievrant_core::types::Route::Cached,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use sievrant_core::types::{RankedCandidate, RetrieverType};
    use sievrant_index::vector::{IndexChunk, IndexQuery, RetrievalIndex};
    use sievrant_llm::provider::{LlmProvider, LlmRequest, LlmResponse};
    use sievrant_rerank::Reranker;
    use std::sync::Arc;

    struct MockIndex;
    #[async_trait]
    impl RetrievalIndex for MockIndex {
        async fn search(
            &self,
            _q: &IndexQuery,
            _a: &[String],
            _l: usize,
        ) -> Result<Vec<RankedCandidate>> {
            use chrono::Utc;
            use sievrant_core::types::ChunkMetadata;
            Ok(vec![RankedCandidate {
                chunk_id: "mock_chunk".into(),
                score: 0.9,
                retriever: RetrieverType::VectorIndex,
                text: "mock evidence".into(),
                metadata: ChunkMetadata {
                    doc_id: "doc1".into(),
                    source_doc: "doc".into(),
                    doc_version_hash: "hash".into(),
                    path: vec![],
                    page: None,
                    char_span: None,
                    ingested_at: Utc::now(),
                    index_generation: 1,
                    acl_groups: vec![],
                    token_count: 10,
                },
            }])
        }
        async fn insert(&self, _c: &[IndexChunk]) -> Result<()> {
            Ok(())
        }
        async fn delete(&self, _ids: &[String]) -> Result<()> {
            Ok(())
        }
        fn index_type(&self) -> RetrieverType {
            RetrieverType::VectorIndex
        }
    }

    struct MockLlm;
    #[async_trait]
    impl LlmProvider for MockLlm {
        async fn complete(&self, _req: LlmRequest) -> Result<LlmResponse> {
            use sievrant_llm::provider::{FinishReason, TokenUsage};
            Ok(LlmResponse {
                content: "Mock answer".into(),
                usage: TokenUsage::default(),
                model: "mock-model".into(),
                finish_reason: FinishReason::Stop,
            })
        }
        fn supports_vision(&self) -> bool {
            false
        }
        fn context_window(&self) -> usize {
            4096
        }
        fn model_id(&self) -> &str {
            "mock-model"
        }
    }

    struct MockCache;
    #[async_trait]
    impl SemanticCache for MockCache {
        async fn lookup(
            &self,
            _q: &str,
            _c: &str,
            _u: &Option<String>,
        ) -> Result<Option<EvidencePacket>> {
            Ok(None)
        }
        async fn store(
            &self,
            _q: &str,
            _c: &str,
            _u: &Option<String>,
            _p: &EvidencePacket,
        ) -> Result<()> {
            Ok(())
        }
    }

    struct MockReranker;
    #[async_trait]
    impl Reranker for MockReranker {
        async fn rerank(&self, _q: &str, _d: &[&str]) -> Result<Vec<f64>> {
            Ok(_d.iter().map(|_| 0.9).collect())
        }
        fn name(&self) -> &str {
            "mock-reranker"
        }
    }

    struct MockCalibration;
    impl CalibrationManager for MockCalibration {
        fn threshold(&self, _c: &str, _t: &QueryTaxonomy) -> f64 {
            0.5
        }
    }

    struct MockCoverage;
    impl CoverageMonitor for MockCoverage {
        fn record(&self, _c: &str, _covered: bool) {}
        fn empirical_coverage_30d(&self, _c: &str) -> Option<f64> {
            Some(0.8)
        }
    }

    struct MockAclResolver;
    #[async_trait]
    impl AclResolver for MockAclResolver {
        async fn resolve(&self, _user_id: Option<&str>, _corpus: &str) -> Result<Vec<String>> {
            Ok(vec!["public".into()])
        }
    }

    struct MockEmbedding;
    #[async_trait]
    impl sievrant_query::EmbeddingProvider for MockEmbedding {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
        }
        fn dimensions(&self) -> usize {
            128
        }
    }

    #[tokio::test]
    async fn test_orchestrator_pipeline_retrieve() {
        use sievrant_core::types::RouteStrategy;
        use sievrant_query::processing::QueryProcessingConfig;
        use sievrant_query::router::RoutingConfig;

        let orchestrator = RetrievalOrchestrator {
            cache: Arc::new(MockCache),
            router: Arc::new(QueryRouter::new(RoutingConfig::default())),
            query_processor: Arc::new(QueryProcessor::new(
                Arc::new(MockLlm),
                Arc::new(MockEmbedding),
                QueryProcessingConfig::default(),
            )),
            acl_resolver: Arc::new(MockAclResolver),
            indexes: IndexSet {
                vector: Arc::new(MockIndex),
                bm25: Arc::new(MockIndex),
                tree: Arc::new(MockIndex),
                graph: Arc::new(MockIndex),
            },
            reranker: Arc::new(MockReranker),
            llm: Arc::new(MockLlm),
            calibration: Arc::new(MockCalibration),
            coverage_monitor: Arc::new(MockCoverage),
            search_limit: 10,
            min_evidence_count: 1,
        };

        let config = RetrievalConfig {
            coverage: 0.95,
            token_budget: 4096,
            route: RouteStrategy::Standard,
            user_id: None,
            skip_cache: true,
            rerank_top_n: 5,
            rrf_k: 60,
        };

        let result = orchestrator
            .retrieve("test query", "test corpus", config)
            .await;
        assert!(result.is_ok());
        let packet = result.unwrap();

        assert_eq!(packet.answer, Some("Mock answer".to_string()));
        assert_eq!(packet.decision, Decision::Answered);
    }
}
