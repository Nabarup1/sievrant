use sievrant_core::types::{Decision, RankedCandidate, RetrievalConfig, RetrieverType};
use sievrant_index::vector::{IndexChunk, IndexQuery, RetrievalIndex};
use sievrant_llm::provider::{FinishReason, LlmProvider, LlmRequest, LlmResponse, TokenUsage};
use sievrant_orchestrator::pipeline::{CalibrationManager, CoverageMonitor, IndexSet, RetrievalOrchestrator, SemanticCache};
use sievrant_query::processing::QueryProcessor;
use sievrant_query::router::QueryRouter;
use sievrant_rerank::Reranker;
use std::sync::Arc;

struct MockIndex;
#[async_trait::async_trait]
impl RetrievalIndex for MockIndex {
    async fn search(
        &self,
        _q: &IndexQuery,
        _a: &[String],
        _l: usize,
    ) -> sievrant_core::error::Result<Vec<RankedCandidate>> {
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
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: vec![],
                token_count: 10,
            },
        }])
    }
    async fn insert(&self, _c: &[IndexChunk]) -> sievrant_core::error::Result<()> {
        Ok(())
    }
    async fn delete(&self, _ids: &[String]) -> sievrant_core::error::Result<()> {
        Ok(())
    }
    fn index_type(&self) -> RetrieverType {
        RetrieverType::VectorIndex
    }
}

struct MockLlm;
#[async_trait::async_trait]
impl LlmProvider for MockLlm {
    async fn complete(&self, _req: LlmRequest) -> sievrant_core::error::Result<LlmResponse> {
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
#[async_trait::async_trait]
impl SemanticCache for MockCache {
    async fn lookup(
        &self,
        _q: &str,
        _c: &str,
        _u: &Option<String>,
    ) -> sievrant_core::error::Result<Option<sievrant_core::types::EvidencePacket>> {
        Ok(None)
    }
    async fn store(
        &self,
        _q: &str,
        _c: &str,
        _u: &Option<String>,
        _p: &sievrant_core::types::EvidencePacket,
    ) -> sievrant_core::error::Result<()> {
        Ok(())
    }
}

struct MockReranker;
#[async_trait::async_trait]
impl Reranker for MockReranker {
    async fn rerank(&self, _q: &str, _d: &[&str]) -> sievrant_core::error::Result<Vec<f64>> {
        Ok(_d.iter().map(|_| 0.9).collect())
    }
    fn name(&self) -> &str {
        "mock-reranker"
    }
}

struct MockCalibration(f64);
impl CalibrationManager for MockCalibration {
    fn threshold(&self, _c: &str, _t: &sievrant_core::types::QueryTaxonomy) -> f64 {
        self.0
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
#[async_trait::async_trait]
impl sievrant_acl::resolver::AclResolver for MockAclResolver {
    async fn resolve(
        &self,
        _user_id: Option<&str>,
        _corpus: &str,
    ) -> sievrant_core::error::Result<Vec<String>> {
        Ok(vec!["public".into()])
    }
}

struct MockEmbedding;
#[async_trait::async_trait]
impl sievrant_query::EmbeddingProvider for MockEmbedding {
    async fn embed(&self, texts: &[String]) -> sievrant_core::error::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
    }
    fn dimensions(&self) -> usize {
        128
    }
}

async fn build_orchestrator(threshold: f64) -> RetrievalOrchestrator {
    RetrievalOrchestrator {
        cache: Arc::new(MockCache),
        router: Arc::new(QueryRouter::new(sievrant_query::router::RoutingConfig::default())),
        query_processor: Arc::new(QueryProcessor::new(
            Arc::new(MockLlm),
            Arc::new(MockEmbedding),
            sievrant_query::processing::QueryProcessingConfig::default(),
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
        calibration: Arc::new(MockCalibration(threshold)),
        coverage_monitor: Arc::new(MockCoverage),
        search_limit: 10,
        min_evidence_count: 1, // Require 1 chunk to Answer
    }
}

#[tokio::test]
async fn test_pipeline_decision_answered() {
    let orchestrator = build_orchestrator(0.99).await; // High threshold, everything passes
    let config = RetrievalConfig {
        coverage: 0.95,
        token_budget: 4096,
        route: sievrant_core::types::RouteStrategy::Standard,
        user_id: None,
        skip_cache: true,
        rerank_top_n: 5,
        rrf_k: 60,
    };

    let packet = orchestrator
        .retrieve("test query", "test corpus", config)
        .await
        .unwrap();

    assert!(matches!(packet.decision, Decision::Answered));
    assert!(packet.answer.is_some());
    assert!(!packet.evidence.is_empty());
}

#[tokio::test]
async fn test_pipeline_decision_abstained() {
    // Threshold 0.0 means no chunk with nonconformity > 0.0 will pass.
    // The mock scores (0.9, 0.9) will yield a nonconformity score of around 0.19.
    let orchestrator = build_orchestrator(0.0).await; 
    let config = RetrievalConfig {
        coverage: 0.95,
        token_budget: 4096,
        route: sievrant_core::types::RouteStrategy::Standard,
        user_id: None,
        skip_cache: true,
        rerank_top_n: 5,
        rrf_k: 60,
    };

    let packet = orchestrator
        .retrieve("test query", "test corpus", config)
        .await
        .unwrap();

    assert!(matches!(packet.decision, Decision::Abstained));
    assert!(packet.answer.is_none());
    assert!(packet.evidence.is_empty()); // all chunks rejected by gate
}
