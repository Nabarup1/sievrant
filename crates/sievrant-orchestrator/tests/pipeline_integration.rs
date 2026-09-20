use async_trait::async_trait;
use std::sync::Arc;

use sievrant_core::{
    error::Result,
    types::{ChunkMetadata, Decision, EvidencePacket, RankedCandidate, RetrievalConfig, RetrieverType, RouteStrategy},
};
use sievrant_index::vector::{IndexChunk, IndexQuery};
use sievrant_index::RetrievalIndex;
use sievrant_llm::provider::{LlmProvider, LlmRequest, LlmResponse, TokenUsage, FinishReason};
use sievrant_orchestrator::pipeline::{
    CalibrationManager, CoverageMonitor, IndexSet, RetrievalOrchestrator, SemanticCache,
};
use sievrant_query::{
    processing::{QueryProcessingConfig, QueryProcessor},
    router::{QueryRouter, RoutingConfig},
    EmbeddingProvider,
};
use sievrant_rerank::Reranker;
use sievrant_acl::resolver::AclResolver;
use sievrant_core::types::QueryTaxonomy;

struct MockIndex;
#[async_trait]
impl RetrievalIndex for MockIndex {
    async fn search(&self, _q: &IndexQuery, _a: &[String], _l: usize) -> Result<Vec<RankedCandidate>> {
        use chrono::Utc;
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
    async fn insert(&self, _c: &[IndexChunk]) -> Result<()> { Ok(()) }
    async fn delete(&self, _ids: &[String]) -> Result<()> { Ok(()) }
    fn index_type(&self) -> RetrieverType { RetrieverType::VectorIndex }
}

struct MockLlm;
#[async_trait]
impl LlmProvider for MockLlm {
    async fn complete(&self, _req: LlmRequest) -> Result<LlmResponse> {
        Ok(LlmResponse {
            content: "Mock answer".into(),
            usage: TokenUsage::default(),
            model: "mock-model".into(),
            finish_reason: FinishReason::Stop,
        })
    }
    fn supports_vision(&self) -> bool { false }
    fn context_window(&self) -> usize { 4096 }
    fn model_id(&self) -> &str { "mock-model" }
}

struct MockCache;
#[async_trait]
impl SemanticCache for MockCache {
    async fn lookup(&self, _q: &str, _c: &str, _u: &Option<String>) -> Result<Option<EvidencePacket>> { Ok(None) }
    async fn store(&self, _q: &str, _c: &str, _u: &Option<String>, _p: &EvidencePacket) -> Result<()> { Ok(()) }
}

struct MockReranker;
#[async_trait]
impl Reranker for MockReranker {
    async fn rerank(&self, _q: &str, _d: &[&str]) -> Result<Vec<f64>> { Ok(_d.iter().map(|_| 0.9).collect()) }
    fn name(&self) -> &str { "mock-reranker" }
}

struct MockCalibration { threshold: f64 }
impl CalibrationManager for MockCalibration {
    fn threshold(&self, _c: &str, _t: &QueryTaxonomy) -> f64 { self.threshold }
}

struct MockCoverage;
impl CoverageMonitor for MockCoverage {
    fn record(&self, _c: &str, _covered: bool) {}
    fn empirical_coverage_30d(&self, _c: &str) -> Option<f64> { Some(0.8) }
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
impl EmbeddingProvider for MockEmbedding {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
    }
    fn dimensions(&self) -> usize { 128 }
}

fn create_orchestrator(threshold: f64) -> RetrievalOrchestrator {
    RetrievalOrchestrator {
        cache: Arc::new(MockCache),
        router: Arc::new(QueryRouter::new(RoutingConfig::default())),
        query_processor: Arc::new(QueryProcessor::new(
            Arc::new(MockLlm),
            Arc::new(MockEmbedding),
            QueryProcessingConfig::default()
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
        calibration: Arc::new(MockCalibration { threshold }),
        coverage_monitor: Arc::new(MockCoverage),
        search_limit: 10,
        min_evidence_count: 1,
    }
}

#[tokio::test]
async fn test_pipeline_integration_answered() {
    let orchestrator = create_orchestrator(0.5); // high enough threshold to pass
    let config = RetrievalConfig {
        coverage: 0.95,
        token_budget: 4096,
        route: RouteStrategy::Standard,
        user_id: None,
        skip_cache: true,
        rerank_top_n: 5,
        rrf_k: 60,
    };

    let result = orchestrator.retrieve("test query", "test corpus", config).await.unwrap();
    assert_eq!(result.decision, Decision::Answered);
    assert_eq!(result.answer, Some("Mock answer".into()));
}

#[tokio::test]
async fn test_pipeline_integration_abstained() {
    let orchestrator = create_orchestrator(0.0); // 0 threshold -> all rejected
    let config = RetrievalConfig {
        coverage: 0.95,
        token_budget: 4096,
        route: RouteStrategy::Standard,
        user_id: None,
        skip_cache: true,
        rerank_top_n: 5,
        rrf_k: 60,
    };

    let result = orchestrator.retrieve("test query", "test corpus", config).await.unwrap();
    assert_eq!(result.decision, Decision::Abstained);
    assert!(result.answer.is_none());
    assert!(result.evidence.is_empty());
}
