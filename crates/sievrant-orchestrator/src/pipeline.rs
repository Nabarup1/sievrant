use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tracing::{info, warn};

use sievrant_audit::{ReceiptPayload, ReceiptSigner};
use sievrant_calibration::{CorpusCalibration, LoggedCandidate, QueryRecord};
use sievrant_core::{
    error::Result,
    types::{
        AuditBlock, Decision, EvidenceChunk, EvidencePacket, GuaranteeInfo, GuaranteeStatus,
        QueryId, RerankedCandidate, RetrievalConfig, RetrievalStagesMs, RetrieverType, Route,
        EVIDENCE_PACKET_SCHEMA_VERSION,
    },
};
use sievrant_llm::LlmProvider;
use sievrant_math::{
    conformal::nonconformity, rrf::reciprocal_rank_fusion, submodular::submodular_select,
};
use sievrant_query::{
    processing::{ProcessedQuery, QueryProcessor},
    router::QueryRouter,
};
use sievrant_rerank::{rerank_candidates, Reranker};

use sievrant_acl::resolver::AclResolver;
use sievrant_index::RetrievalIndex;

use crate::stages::*;

/// Indexes queried for every retrieval. Tree and graph are experimental and
/// only present when enabled.
pub struct IndexSet {
    pub vector: Arc<dyn RetrievalIndex>,
    pub bm25: Arc<dyn RetrievalIndex>,
    pub tree: Option<Arc<dyn RetrievalIndex>>,
    pub graph: Option<Arc<dyn RetrievalIndex>>,
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

/// Settings that shape candidate scores. Calibration is tied to their fingerprint.
#[derive(Debug, Clone)]
pub struct PipelineSettings {
    pub search_limit: usize,
    pub rerank_top_n: usize,
    pub rrf_k: usize,
    pub max_evidence_chunks: usize,
    pub max_answer_tokens: usize,
    pub growth_factor: f64,
    pub drift_tolerance: f64,
    pub reranker_name: String,
    pub embedding_model: String,
    /// Stable hash of everything above that affects scores.
    pub fingerprint: String,
}

/// Candidates after reranking, before the gate.
pub struct ScoredCandidates {
    pub candidates: Vec<RerankedCandidate>,
    pub retrievers: HashMap<String, RetrieverType>,
    pub degraded_rerank: bool,
    pub warnings: Vec<String>,
    pub route: Route,
    pub stages_ms: RetrievalStagesMs,
    pub candidates_before_rerank: usize,
}

/// Retrieval pipeline for one corpus.
pub struct RetrievalOrchestrator {
    pub cache: Arc<dyn SemanticCache>,
    pub router: Arc<QueryRouter>,
    pub query_processor: Arc<QueryProcessor>,
    pub acl_resolver: Arc<dyn AclResolver>,
    pub indexes: IndexSet,
    pub reranker: Arc<dyn Reranker>,
    pub calibration: Arc<CorpusCalibration>,
    pub llm: Option<Arc<dyn LlmProvider>>,
    pub signer: Option<Arc<ReceiptSigner>>,
    pub settings: PipelineSettings,
    /// Live chunk count, maintained by the corpus on ingest/delete.
    pub chunk_count: Arc<AtomicU64>,
    /// Tenant that owns this corpus; recorded in receipts.
    pub tenant: String,
}

fn ms(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

impl RetrievalOrchestrator {
    /// ACL groups for a request. The caller's API key is trusted to assert the
    /// end user's groups; with none asserted only `public` documents are visible.
    async fn acl_groups(&self, config: &RetrievalConfig, corpus: &str) -> Result<Vec<String>> {
        match &config.acl_groups {
            Some(groups) if !groups.is_empty() => Ok(groups.clone()),
            _ => {
                self.acl_resolver
                    .resolve(config.user_id.as_deref(), corpus)
                    .await
            }
        }
    }

    /// Run routing, retrieval, fusion and reranking. `acl` of `None` searches
    /// every document (internal use: calibration and evaluation only).
    pub async fn score(
        &self,
        query: &str,
        config: &RetrievalConfig,
        acl: Option<&[String]>,
    ) -> Result<ScoredCandidates> {
        let mut stages_ms = RetrievalStagesMs::default();
        let mut warnings = Vec::new();

        let t = Instant::now();
        let route_decision = self.router.classify_with_strategy(query, &config.route);
        stages_ms.routing = ms(t);

        let t = Instant::now();
        let processed: ProcessedQuery = self
            .query_processor
            .process(query, &route_decision.route)
            .await?;
        stages_ms.query_processing = ms(t);
        let route = if self.query_processor.expands() {
            route_decision.route
        } else {
            Route::Fast
        };

        let t = Instant::now();
        let limit = self.settings.search_limit;
        let (ranked_lists, index_warnings) =
            parallel_index_search(&self.indexes, &processed, acl.unwrap_or(&[]), limit).await;
        stages_ms.vector_search = ms(t);
        for w in index_warnings {
            warn!("{w}");
            warnings.push(w);
        }

        let t = Instant::now();
        // rrf_k comes from settings so scores always match the calibrated pipeline.
        let fused = reciprocal_rank_fusion(&ranked_lists, self.settings.rrf_k);
        stages_ms.rrf_fusion = ms(t);
        let candidates_before_rerank = fused.len();
        let retrievers: HashMap<String, RetrieverType> = fused
            .iter()
            .filter_map(|c| {
                c.source_scores
                    .first()
                    .map(|(r, _)| (c.chunk_id.clone(), r.clone()))
            })
            .collect();

        let t = Instant::now();
        let reranked = rerank_candidates(
            self.reranker.as_ref(),
            query,
            fused,
            self.settings.rerank_top_n,
        )
        .await?;
        stages_ms.reranking = ms(t);
        if reranked.degraded {
            warnings.push(format!(
                "reranker '{}' failed; fusion scores used",
                self.reranker.name()
            ));
        }

        Ok(ScoredCandidates {
            candidates: reranked.candidates,
            retrievers,
            degraded_rerank: reranked.degraded,
            warnings,
            route,
            stages_ms,
            candidates_before_rerank,
        })
    }

    pub async fn retrieve(
        &self,
        query: &str,
        corpus: &str,
        config: RetrievalConfig,
    ) -> Result<EvidencePacket> {
        let query_id = QueryId::new();
        let pipeline_start = Instant::now();

        let t = Instant::now();
        if !config.skip_cache {
            if let Ok(Some(mut cached)) = self.cache.lookup(query, corpus, &config.user_id).await {
                cached.audit.retrieval_stages_ms.cache_check = ms(t);
                cached.cache_hit = true;
                info!(query_id = %query_id, "cache hit");
                return Ok(cached);
            }
        }
        let cache_ms = ms(t);

        let t = Instant::now();
        let acl = self.acl_groups(&config, corpus).await?;
        let acl_ms = ms(t);

        let scored = self.score(query, &config, Some(&acl)).await?;
        let mut stages_ms = scored.stages_ms;
        stages_ms.cache_check = cache_ms;
        stages_ms.acl_resolution = acl_ms;

        // Conformal gate.
        let t = Instant::now();
        let chunk_count = self.chunk_count.load(Ordering::Relaxed);
        let mut gate = self.calibration.decide(
            config.coverage,
            &self.settings.fingerprint,
            chunk_count,
            self.settings.growth_factor,
        );
        if scored.degraded_rerank && gate.status == GuaranteeStatus::Calibrated {
            gate.status = GuaranteeStatus::Degraded;
            gate.tau = None;
            gate.reason = Some(
                "reranker failed for this request; scores are not comparable to calibration".into(),
            );
        }

        let mut candidates = scored.candidates;
        let nonconf: Vec<f64> = candidates
            .iter()
            .map(|c| nonconformity(c.rerank_score))
            .collect();
        let logged: Vec<LoggedCandidate> = candidates
            .iter()
            .zip(&nonconf)
            .map(|(c, s)| LoggedCandidate {
                chunk_id: c.chunk_id.clone(),
                doc_id: c.metadata.doc_id.clone(),
                nonconformity: *s,
            })
            .collect();

        let conformal_set: Vec<RerankedCandidate> = match gate.tau {
            Some(tau) => candidates
                .drain(..)
                .zip(&nonconf)
                .filter(|(_, s)| **s <= tau)
                .map(|(c, _)| c)
                .collect(),
            None => std::mem::take(&mut candidates),
        };
        let set_size = conformal_set.len();
        stages_ms.conformal_gate = ms(t);

        // Token-budgeted, redundancy-aware selection within the set.
        let t = Instant::now();
        let mut selected = submodular_select(&conformal_set, config.token_budget, 0.3);
        selected.truncate(self.settings.max_evidence_chunks);
        for s in &mut selected {
            s.nonconformity_score = nonconformity(s.rerank_score);
        }
        stages_ms.submodular_selection = ms(t);
        let budget_truncated = gate.tau.is_some() && selected.len() < set_size;

        let mut decision = if selected.is_empty() {
            Decision::Abstained
        } else {
            Decision::Answered
        };

        // Answer generation.
        let t = Instant::now();
        let mut answer = None;
        if decision == Decision::Answered {
            match (&self.llm, config.generate_answer) {
                (Some(llm), true) => {
                    answer = Some(
                        generate_answer(
                            llm.as_ref(),
                            query,
                            &selected,
                            self.settings.max_answer_tokens,
                        )
                        .await?,
                    );
                }
                _ => decision = Decision::EvidenceOnly,
            }
        }
        stages_ms.llm_generation = ms(t);

        let observed = self
            .calibration
            .observed(config.coverage, self.settings.drift_tolerance);
        let guarantee =
            build_guarantee(&gate, config.coverage, set_size, budget_truncated, observed);
        let coverage_guarantee =
            (guarantee.status == GuaranteeStatus::Calibrated).then_some(config.coverage);

        let evidence_chunks: Vec<EvidenceChunk> = map_evidence_chunks(selected, &scored.retrievers);
        let session_hash =
            compute_session_hash(&query_id, query, &evidence_chunks, answer.as_deref());

        let mut packet = EvidencePacket {
            schema_version: EVIDENCE_PACKET_SCHEMA_VERSION.into(),
            query_id: query_id.clone(),
            trace_id: None,
            answer,
            decision: decision.clone(),
            coverage_guarantee,
            empirical_coverage_last_30d: guarantee.observed.as_ref().map(|o| o.rate),
            guarantee,
            route: route_into_packet_route(&scored.route),
            cache_hit: false,
            evidence: evidence_chunks,
            audit: AuditBlock {
                model_version: self.llm.as_ref().map(|l| l.model_id().to_string()),
                retrieval_stages_ms: stages_ms,
                total_candidates_before_rerank: scored.candidates_before_rerank,
                total_candidates_after_rerank: logged.len(),
                total_candidates_after_conformal: set_size,
                calibration_set_id: gate.set_id.clone().unwrap_or_else(|| "none".into()),
                session_hash,
                reranker: Some(self.settings.reranker_name.clone()),
                embedding_model: Some(self.settings.embedding_model.clone()),
                pipeline_fingerprint: Some(self.settings.fingerprint.clone()),
                warnings: scored.warnings,
            },
            latency_ms: 0,
            cost_usd: None,
            receipt: None,
        };

        if let Err(e) = self.calibration.log_query(QueryRecord {
            query_id: query_id.0.clone(),
            at: chrono::Utc::now(),
            query: query.to_string(),
            decision,
            requested_coverage: config.coverage,
            tau: gate.tau,
            status: packet.guarantee.status,
            candidates: logged,
        }) {
            warn!(error = %e, "failed to write query log");
        }

        packet.latency_ms = pipeline_start.elapsed().as_millis() as u64;
        if let Some(signer) = &self.signer {
            let payload = ReceiptPayload::from_packet(&self.tenant, corpus, query, &packet);
            packet.receipt = Some(signer.sign(&payload)?);
        }

        let _ = self
            .cache
            .store(query, corpus, &config.user_id, &packet)
            .await;

        Ok(packet)
    }
}

fn build_guarantee(
    gate: &sievrant_calibration::GateDecision,
    coverage: f64,
    set_size: usize,
    budget_truncated: bool,
    observed: Option<sievrant_core::types::ObservedCoverage>,
) -> GuaranteeInfo {
    let statement = match gate.status {
        GuaranteeStatus::Calibrated => {
            let source = match gate.source {
                Some(sievrant_core::types::CalibrationSource::Synthetic) => "synthetic",
                Some(sievrant_core::types::CalibrationSource::Feedback) => "feedback-labelled",
                _ => "labelled",
            };
            let mut s = format!(
                "For queries like the {n} {source} calibration queries, the gated evidence set \
                 ({set_size} passages) contains at least one relevant passage with probability \
                 at least {coverage}. This bounds missed evidence, not answer correctness.",
                n = gate.size.unwrap_or(0),
            );
            if budget_truncated {
                s.push_str(
                    " The token budget returned a subset of the set; raise token_budget to \
                     receive all of it.",
                );
            }
            s
        }
        _ => format!(
            "No coverage guarantee: {}.",
            gate.reason.as_deref().unwrap_or("not calibrated")
        ),
    };
    GuaranteeInfo {
        status: gate.status,
        statement,
        requested_coverage: coverage,
        calibration_set_id: gate.set_id.clone(),
        calibration_source: gate.source,
        calibration_size: gate.size,
        threshold: gate.tau,
        conformal_set_size: set_size,
        budget_truncated,
        max_achievable_coverage: gate.max_achievable_coverage,
        observed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use sievrant_calibration::CalibrationRecord;
    use sievrant_core::types::{CalibrationSource, ChunkMetadata, RankedCandidate};
    use sievrant_index::vector::{IndexChunk, IndexQuery};
    use sievrant_llm::provider::{FinishReason, LlmRequest, LlmResponse, TokenUsage};
    use sievrant_query::processing::QueryProcessingConfig;
    use sievrant_query::router::RoutingConfig;
    use tempfile::TempDir;

    /// Returns two chunks; the reranker scores "relevant" text high.
    struct TwoChunkIndex;
    #[async_trait]
    impl RetrievalIndex for TwoChunkIndex {
        async fn search(
            &self,
            _q: &IndexQuery,
            acl: &[String],
            _l: usize,
        ) -> Result<Vec<RankedCandidate>> {
            let make = |id: &str, text: &str, groups: &[&str]| RankedCandidate {
                chunk_id: id.into(),
                score: 0.5,
                retriever: RetrieverType::Bm25Index,
                text: text.into(),
                metadata: ChunkMetadata {
                    doc_id: id.split(':').next().unwrap().into(),
                    source_doc: "doc".into(),
                    doc_version_hash: "h".into(),
                    path: vec![],
                    page: None,
                    char_span: None,
                    ingested_at: chrono::Utc::now(),
                    index_generation: 1,
                    acl_groups: groups.iter().map(|g| g.to_string()).collect(),
                    token_count: 10,
                },
            };
            let all = vec![
                make("d1:0:0", "relevant passage", &["public"]),
                make("d2:0:0", "noise passage", &["public"]),
                make("d3:0:0", "relevant secret", &["legal"]),
            ];
            Ok(all
                .into_iter()
                .filter(|c| acl.is_empty() || c.metadata.acl_groups.iter().any(|g| acl.contains(g)))
                .collect())
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

    struct KeywordReranker;
    #[async_trait]
    impl Reranker for KeywordReranker {
        async fn rerank(&self, _q: &str, docs: &[&str]) -> Result<Vec<f64>> {
            Ok(docs
                .iter()
                .map(|d| if d.contains("relevant") { 0.9 } else { 0.1 })
                .collect())
        }
        fn name(&self) -> &str {
            "keyword"
        }
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
        async fn resolve(&self, _user_id: Option<&str>, _corpus: &str) -> Result<Vec<String>> {
            Ok(vec!["public".into()])
        }
    }

    struct ZeroEmbedding;
    #[async_trait]
    impl sievrant_query::EmbeddingProvider for ZeroEmbedding {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
        }
        fn dimensions(&self) -> usize {
            8
        }
    }

    fn orchestrator(dir: &TempDir, llm: bool) -> RetrievalOrchestrator {
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
                vector: Arc::new(TwoChunkIndex),
                bm25: Arc::new(TwoChunkIndex),
                tree: None,
                graph: None,
            },
            reranker: Arc::new(KeywordReranker),
            calibration: Arc::new(CorpusCalibration::open(dir.path()).unwrap()),
            llm: llm.then(|| Arc::new(MockLlm) as Arc<dyn LlmProvider>),
            signer: None,
            settings: PipelineSettings {
                search_limit: 10,
                rerank_top_n: 10,
                rrf_k: 60,
                max_evidence_chunks: 10,
                max_answer_tokens: 512,
                growth_factor: 1.5,
                drift_tolerance: 0.02,
                reranker_name: "keyword".into(),
                embedding_model: "zero".into(),
                fingerprint: "fp".into(),
            },
            chunk_count: Arc::new(AtomicU64::new(100)),
            tenant: "t".into(),
        }
    }

    fn calibrate(o: &RetrievalOrchestrator, scores: Vec<Option<f64>>) {
        o.calibration
            .set_record(CalibrationRecord {
                set_id: "cal".into(),
                source: CalibrationSource::Labeled,
                created_at: chrono::Utc::now(),
                fingerprint: "fp".into(),
                corpus_chunk_count: 100,
                scores,
                unanswerable_count: 0,
                unanswerable_abstained_at_95: 0,
                holdout: vec![],
            })
            .unwrap();
    }

    fn cfg() -> RetrievalConfig {
        RetrievalConfig {
            skip_cache: true,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn uncalibrated_corpus_answers_without_claiming_a_guarantee() {
        let dir = TempDir::new().unwrap();
        let o = orchestrator(&dir, true);
        let p = o.retrieve("q", "c", cfg()).await.unwrap();
        assert_eq!(p.decision, Decision::Answered);
        assert_eq!(p.answer.as_deref(), Some("Mock answer"));
        assert_eq!(p.coverage_guarantee, None);
        assert_eq!(p.guarantee.status, GuaranteeStatus::Uncalibrated);
        assert!(p.guarantee.statement.starts_with("No coverage guarantee"));
    }

    #[tokio::test]
    async fn calibrated_gate_keeps_relevant_and_drops_noise() {
        let dir = TempDir::new().unwrap();
        let o = orchestrator(&dir, true);
        // Relevant candidates typically score nonconformity ~0.1.
        calibrate(&o, (0..50).map(|i| Some(0.05 + i as f64 * 0.002)).collect());
        let p = o.retrieve("q", "c", cfg()).await.unwrap();
        assert_eq!(p.guarantee.status, GuaranteeStatus::Calibrated);
        assert_eq!(p.coverage_guarantee, Some(0.95));
        let ids: Vec<&str> = p.evidence.iter().map(|e| e.chunk_id.as_str()).collect();
        assert_eq!(ids, vec!["d1:0:0"]);
        assert_eq!(p.evidence[0].retriever, RetrieverType::Bm25Index);
        assert!(p
            .guarantee
            .statement
            .contains("50 labelled calibration queries"));
    }

    #[tokio::test]
    async fn gate_abstains_when_nothing_passes() {
        let dir = TempDir::new().unwrap();
        let o = orchestrator(&dir, true);
        // Threshold far below any candidate's nonconformity (0.1 / 0.9).
        calibrate(&o, (0..50).map(|i| Some(i as f64 * 0.0001)).collect());
        let p = o.retrieve("q", "c", cfg()).await.unwrap();
        assert_eq!(p.decision, Decision::Abstained);
        assert!(p.answer.is_none());
        assert!(p.evidence.is_empty());
        assert_eq!(o.calibration.gaps(5).len(), 1);
    }

    #[tokio::test]
    async fn no_llm_or_evidence_only_request_skips_generation() {
        let dir = TempDir::new().unwrap();
        let p = orchestrator(&dir, false)
            .retrieve("q", "c", cfg())
            .await
            .unwrap();
        assert_eq!(p.decision, Decision::EvidenceOnly);
        assert!(p.answer.is_none());

        let dir = TempDir::new().unwrap();
        let mut c = cfg();
        c.generate_answer = false;
        let p = orchestrator(&dir, true)
            .retrieve("q", "c", c)
            .await
            .unwrap();
        assert_eq!(p.decision, Decision::EvidenceOnly);
    }

    #[tokio::test]
    async fn acl_groups_restrict_evidence() {
        let dir = TempDir::new().unwrap();
        let o = orchestrator(&dir, false);
        let p = o.retrieve("q", "c", cfg()).await.unwrap();
        assert!(p.evidence.iter().all(|e| e.chunk_id != "d3:0:0"));

        let mut c = cfg();
        c.acl_groups = Some(vec!["legal".into()]);
        let p = o.retrieve("q", "c", c).await.unwrap();
        assert!(p.evidence.iter().all(|e| e.chunk_id == "d3:0:0"));
    }

    #[tokio::test]
    async fn feedback_on_served_query_updates_observed_coverage() {
        let dir = TempDir::new().unwrap();
        let o = orchestrator(&dir, false);
        calibrate(&o, (0..50).map(|i| Some(0.05 + i as f64 * 0.002)).collect());
        let p = o.retrieve("q", "c", cfg()).await.unwrap();
        let out = o
            .calibration
            .submit_feedback(&p.query_id.0, &["d1".into()], &[])
            .unwrap()
            .unwrap();
        assert_eq!(out.covered, Some(true));
        let p2 = o.retrieve("q", "c", cfg()).await.unwrap();
        assert_eq!(p2.guarantee.observed.unwrap().observations, 1);
    }

    #[tokio::test]
    async fn signed_packets_verify() {
        let dir = TempDir::new().unwrap();
        let mut o = orchestrator(&dir, true);
        let signer = ReceiptSigner::from_base64_seed(&ReceiptSigner::generate_seed()).unwrap();
        let pk = signer.public_key_base64();
        o.signer = Some(Arc::new(signer));
        let p = o.retrieve("what is it?", "c", cfg()).await.unwrap();
        let check = sievrant_audit::verify_packet(&p, &pk, Some("what is it?")).unwrap();
        assert!(check.ok(), "{:?}", check.problems);
    }
}
