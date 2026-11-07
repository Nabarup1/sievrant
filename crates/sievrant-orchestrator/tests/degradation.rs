//! A failing component must never produce a response that claims a guarantee.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use async_trait::async_trait;
use sievrant_acl::resolver::AclResolver;
use sievrant_calibration::{CalibrationRecord, CorpusCalibration};
use sievrant_core::error::{IndexError, RerankError, Result};
use sievrant_core::types::{
    CalibrationSource, ChunkMetadata, Decision, EvidencePacket, GuaranteeStatus, RankedCandidate,
    RetrievalConfig, RetrieverType,
};
use sievrant_index::vector::{IndexChunk, IndexQuery};
use sievrant_index::RetrievalIndex;
use sievrant_orchestrator::pipeline::{
    IndexSet, PipelineSettings, RetrievalOrchestrator, SemanticCache,
};
use sievrant_query::processing::{QueryProcessingConfig, QueryProcessor};
use sievrant_query::router::{QueryRouter, RoutingConfig};
use sievrant_rerank::Reranker;

struct OkIndex;
#[async_trait]
impl RetrievalIndex for OkIndex {
    async fn search(
        &self,
        _q: &IndexQuery,
        _a: &[String],
        _l: usize,
    ) -> Result<Vec<RankedCandidate>> {
        Ok(vec![RankedCandidate {
            chunk_id: "d1:0:0".into(),
            score: 1.0,
            retriever: RetrieverType::VectorIndex,
            text: "relevant".into(),
            metadata: ChunkMetadata {
                doc_id: "d1".into(),
                source_doc: "d1".into(),
                doc_version_hash: "h".into(),
                path: vec![],
                page: None,
                char_span: None,
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: vec!["public".into()],
                token_count: 5,
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

struct BrokenIndex;
#[async_trait]
impl RetrievalIndex for BrokenIndex {
    async fn search(
        &self,
        _q: &IndexQuery,
        _a: &[String],
        _l: usize,
    ) -> Result<Vec<RankedCandidate>> {
        Err(IndexError::Bm25("disk on fire".into()).into())
    }
    async fn insert(&self, _c: &[IndexChunk]) -> Result<()> {
        Ok(())
    }
    async fn delete(&self, _ids: &[String]) -> Result<()> {
        Ok(())
    }
    fn index_type(&self) -> RetrieverType {
        RetrieverType::Bm25Index
    }
}

struct FixedReranker(bool);
#[async_trait]
impl Reranker for FixedReranker {
    async fn rerank(&self, _q: &str, docs: &[&str]) -> Result<Vec<f64>> {
        if self.0 {
            Ok(docs.iter().map(|_| 0.9).collect())
        } else {
            Err(RerankError::Api("upstream 503".into()).into())
        }
    }
    fn name(&self) -> &str {
        "fixed"
    }
}

struct NoCache;
#[async_trait]
impl SemanticCache for NoCache {
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

struct PublicAcl;
#[async_trait]
impl AclResolver for PublicAcl {
    async fn resolve(&self, _u: Option<&str>, _c: &str) -> Result<Vec<String>> {
        Ok(vec!["public".into()])
    }
}

struct ZeroEmbedding;
#[async_trait]
impl sievrant_query::EmbeddingProvider for ZeroEmbedding {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 4]).collect())
    }
    fn dimensions(&self) -> usize {
        4
    }
}

fn build(dir: &tempfile::TempDir, reranker_ok: bool) -> RetrievalOrchestrator {
    let calibration = CorpusCalibration::open(dir.path()).unwrap();
    calibration
        .set_record(CalibrationRecord {
            set_id: "cal".into(),
            source: CalibrationSource::Labeled,
            created_at: chrono::Utc::now(),
            fingerprint: "fp".into(),
            corpus_chunk_count: 10,
            scores: (0..40).map(|i| Some(0.05 + i as f64 * 0.003)).collect(),
            unanswerable_count: 0,
            unanswerable_abstained_at_95: 0,
            holdout: vec![],
        })
        .unwrap();
    RetrievalOrchestrator {
        cache: Arc::new(NoCache),
        router: Arc::new(QueryRouter::new(RoutingConfig::default())),
        query_processor: Arc::new(QueryProcessor::new(
            None,
            Arc::new(ZeroEmbedding),
            QueryProcessingConfig::default(),
        )),
        acl_resolver: Arc::new(PublicAcl),
        indexes: IndexSet {
            vector: Arc::new(OkIndex),
            bm25: Arc::new(BrokenIndex),
            tree: None,
            graph: None,
        },
        reranker: Arc::new(FixedReranker(reranker_ok)),
        calibration: Arc::new(calibration),
        llm: None,
        signer: None,
        settings: PipelineSettings {
            search_limit: 10,
            rerank_top_n: 10,
            rrf_k: 60,
            max_evidence_chunks: 10,
            max_answer_tokens: 256,
            growth_factor: 1.5,
            drift_tolerance: 0.02,
            reranker_name: "fixed".into(),
            embedding_model: "zero".into(),
            fingerprint: "fp".into(),
        },
        chunk_count: Arc::new(AtomicU64::new(10)),
        tenant: "t".into(),
    }
}

fn cfg() -> RetrievalConfig {
    RetrievalConfig {
        skip_cache: true,
        ..Default::default()
    }
}

#[tokio::test]
async fn failing_index_is_reported_and_pipeline_continues() {
    let dir = tempfile::TempDir::new().unwrap();
    let p = build(&dir, true).retrieve("q", "c", cfg()).await.unwrap();
    assert_eq!(p.decision, Decision::EvidenceOnly);
    assert_eq!(p.evidence.len(), 1);
    assert!(p
        .audit
        .warnings
        .iter()
        .any(|w| w.contains("bm25 index failed")));
    assert_eq!(p.guarantee.status, GuaranteeStatus::Calibrated);
}

#[tokio::test]
async fn reranker_failure_withholds_the_guarantee() {
    let dir = tempfile::TempDir::new().unwrap();
    let p = build(&dir, false).retrieve("q", "c", cfg()).await.unwrap();
    assert_eq!(p.guarantee.status, GuaranteeStatus::Degraded);
    assert_eq!(p.coverage_guarantee, None);
    assert!(p.audit.warnings.iter().any(|w| w.contains("reranker")));
}
