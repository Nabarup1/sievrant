use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a single retrieval request. Generated at pipeline entry.
/// Using a newtype to prevent accidental confusion with other string IDs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryId(pub String);

impl QueryId {
    pub fn new() -> Self {
        QueryId(Uuid::new_v4().to_string())
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for QueryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Outcome of a single retrieval. Drives downstream agent behaviour and UI branching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Decision {
    /// The conformal gate passed and an answer was generated.
    Answered,
    /// The conformal gate rejected all evidence; no answer produced.
    Abstained,
    /// Evidence was retrieved but LLM generation was skipped (evidence-only call).
    EvidenceOnly,
}

/// Which pipeline path a query took. Drives cost attribution and latency SLO tracking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Route {
    /// Single-pass hybrid search, no query expansion.
    Fast,
    /// Full pipeline with expansion but no multi-hop decomposition.
    Standard,
    /// Multi-hop decomposition with iterative sub-query retrieval.
    Slow,
    /// Served from semantic cache; no retrieval executed.
    Cached,
}

/// Which index contributed a specific evidence chunk.
/// Stored per-chunk so downstream agents can apply their own trust policies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrieverType {
    VectorIndex,
    Bm25Index,
    TreeIndex,
    GraphIndex,
}

/// Per-pipeline-stage latency breakdown in milliseconds.
/// Stored in every Evidence Packet for operational debugging and SLO verification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalStagesMs {
    pub cache_check: u64,
    pub routing: u64,
    pub query_processing: u64,
    pub acl_resolution: u64,
    pub vector_search: u64,
    pub bm25_search: u64,
    pub tree_search: u64,
    pub graph_search: u64,
    pub rrf_fusion: u64,
    pub reranking: u64,
    pub submodular_selection: u64,
    pub conformal_gate: u64,
    pub llm_generation: u64,
}

/// Compliance and debugging block attached to every Evidence Packet.
/// Satisfies EU AI Act Articles 12 and 13 record-keeping requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditBlock {
    /// LLM model identifier used for answer generation, e.g. "claude-sonnet-4-20250514".
    pub model_version: Option<String>,

    /// Per-stage wall-clock latency breakdown.
    pub retrieval_stages_ms: RetrievalStagesMs,

    /// Candidate count entering the reranker (post-RRF).
    pub total_candidates_before_rerank: usize,

    /// Candidate count after reranking, before submodular selection.
    pub total_candidates_after_rerank: usize,

    /// Evidence count that passed the conformal gate.
    pub total_candidates_after_conformal: usize,

    /// Identifies the calibration set used to derive the conformal threshold.
    /// "default" when no per-tenant calibration data exists.
    pub calibration_set_id: String,

    /// SHA-256 of (query_id || query_text || evidence_ids || answer || model || timestamp).
    /// Tamper-evident hash for audit log integrity verification.
    pub session_hash: String,
}

impl Default for AuditBlock {
    fn default() -> Self {
        Self {
            model_version: None,
            retrieval_stages_ms: RetrievalStagesMs::default(),
            total_candidates_before_rerank: 0,
            total_candidates_after_rerank: 0,
            total_candidates_after_conformal: 0,
            calibration_set_id: "default".into(),
            session_hash: String::new(),
        }
    }
}

/// A single piece of retrieved evidence with full provenance, scoring, and access metadata.
/// This is the atomic unit of the Evidence Packet; every field has a named consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceChunk {
    /// Globally unique chunk identifier: "doc_id:section_idx:chunk_idx".
    pub chunk_id: String,

    /// Source document filename or URI.
    pub source_doc: String,

    /// SHA-256 of source document content at ingestion time.
    /// Agent uses this to detect whether the source changed since retrieval.
    pub doc_version_hash: String,

    /// Hierarchical path within the document.
    /// Example: ["Article 4", "4.2 Term", "paragraph 3"].
    /// Used by citation UI to render precise micro-citations.
    pub path: Vec<String>,

    /// Page number in the source document, if the format supports pagination.
    pub page: Option<u32>,

    /// Character span [start, end] in the source document for inline highlighting.
    pub char_span: Option<[usize; 2]>,

    /// The actual text content of this chunk.
    pub text: String,

    /// Which index was responsible for surfacing this chunk.
    pub retriever: RetrieverType,

    /// Score from Reciprocal Rank Fusion, always positive.
    pub rrf_score: f64,

    /// Score from the cross-encoder reranker. None if reranking was skipped due to degradation.
    pub rerank_score: Option<f64>,

    /// Conformal non-conformity score. Lower is more trustworthy. Compared against the
    /// calibrated threshold to decide ANSWERED vs ABSTAINED.
    pub nonconformity_score: f64,

    /// Marginal information gain assigned by submodular selection. Used for coverage analytics.
    pub submodular_marginal_gain: f64,

    /// Timestamp when this chunk was written into the corpus during ingestion.
    pub ingested_at: DateTime<Utc>,

    /// LSM merge generation at the time this chunk was indexed. Consumers can detect
    /// whether cached results are from a stale or current index generation.
    pub index_generation: u64,

    /// ACL groups this chunk belongs to. Populated at ingestion; verified at query time.
    pub acl_groups: Vec<String>,

    /// Hours elapsed since this chunk was last ingested or updated.
    pub freshness_hours: f64,

    /// Approximate token count used for budget accounting in submodular selection.
    pub token_count: usize,
}

/// The primary output of every Sievrant retrieval call.
/// This struct is both the API response and the compliance artifact (EU AI Act Art. 12/13).
/// See PRD.md Section 4 for the full field justification table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePacket {
    /// Unique ID for this retrieval request. Ties together logs, traces, and audit records.
    pub query_id: QueryId,

    /// OpenTelemetry trace ID. Set when OTel is enabled; used to correlate with Jaeger/Tempo.
    pub trace_id: Option<String>,

    /// Generated answer text. None when decision is ABSTAINED or EVIDENCE_ONLY.
    pub answer: Option<String>,

    /// Retrieval outcome. Agent uses this to branch: render answer, show "I don't know", or
    /// present raw evidence for human review.
    pub decision: Decision,

    /// Nominal conformal coverage target configured for this call (e.g. 0.95).
    pub coverage_guarantee: f64,

    /// Empirical coverage rate observed over the last 30 days for this corpus and calibration set.
    /// None until the monitoring window has accumulated enough observations.
    /// ML engineers use this to verify the guarantee is actually holding.
    pub empirical_coverage_last_30d: Option<f64>,

    /// Which pipeline path this query took.
    pub route: Route,

    /// True when this packet was served from the semantic cache without running the pipeline.
    pub cache_hit: bool,

    /// Evidence chunks that passed submodular selection and the conformal gate.
    pub evidence: Vec<EvidenceChunk>,

    /// Full compliance and debugging audit block.
    pub audit: AuditBlock,

    /// Total wall-clock latency from query receipt to packet serialization, in milliseconds.
    pub latency_ms: u64,

    /// Estimated USD cost for this retrieval (embedding, reranker API, LLM tokens).
    /// None when cost tracking is disabled or a local model is used.
    pub cost_usd: Option<f64>,
}

/// Metadata about a document corpus stored on disk and returned by corpus_stats().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusMetadata {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub document_count: u64,
    pub chunk_count: u64,
    pub index_generation: u64,
    pub calibration_set_id: Option<String>,
}

/// A document submitted for ingestion into a corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: DocumentMetadata,
    /// ACL groups controlling who can retrieve chunks from this document.
    pub acl_groups: Vec<String>,
}

/// Optional descriptive metadata attached to a document at ingestion time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentMetadata {
    pub filename: Option<String>,
    pub title: Option<String>,
    pub content_type: Option<String>,
    pub source_url: Option<String>,
    pub author: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
    /// Arbitrary key-value pairs for application-specific metadata.
    pub custom: HashMap<String, serde_json::Value>,
}

/// User-facing routing strategy. Controls which pipeline path a query takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteStrategy {
    /// Delegate path selection to the adaptive router classifier.
    Auto,
    Fast,
    Standard,
    Slow,
}

/// Per-call configuration for the retrieve() pipeline.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Conformal coverage target. Must be in (0.0, 1.0). Default 0.95.
    pub coverage: f64,
    /// Maximum total tokens across all selected evidence chunks.
    pub token_budget: usize,
    pub route: RouteStrategy,
    /// User identity for ACL resolution. None applies "public" group only.
    pub user_id: Option<String>,
    /// Skip the semantic cache for this call.
    pub skip_cache: bool,
    /// Number of RRF-fused candidates to pass to the cross-encoder reranker.
    pub rerank_top_n: usize,
    /// RRF constant k. 60 is the published default from Cormack et al. 2009.
    pub rrf_k: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            coverage: 0.95,
            token_budget: 4000,
            route: RouteStrategy::Auto,
            user_id: None,
            skip_cache: false,
            rerank_top_n: 100,
            rrf_k: 60,
        }
    }
}

/// Metadata stored alongside each chunk in all four indexes.
/// Propagated from Document at ingestion time; never mutated after that.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub doc_id: String,
    pub source_doc: String,
    pub doc_version_hash: String,
    pub path: Vec<String>,
    pub page: Option<u32>,
    pub char_span: Option<[usize; 2]>,
    pub ingested_at: DateTime<Utc>,
    pub index_generation: u64,
    pub acl_groups: Vec<String>,
    pub token_count: usize,
}

/// A raw result from a single retrieval index before fusion.
#[derive(Debug, Clone)]
pub struct RankedCandidate {
    pub chunk_id: String,
    pub score: f64,
    pub retriever: RetrieverType,
    pub text: String,
    pub metadata: ChunkMetadata,
}

/// Result after RRF fusion across all four indexes.
/// `source_scores` preserves per-index scores for downstream audit use.
#[derive(Debug, Clone)]
pub struct FusedCandidate {
    pub chunk_id: String,
    pub fused_score: f64,
    pub source_scores: Vec<(RetrieverType, f64)>,
    pub text: String,
    pub metadata: ChunkMetadata,
}

/// Result after the cross-encoder reranker assigns a more accurate relevance score.
#[derive(Debug, Clone)]
pub struct RerankedCandidate {
    pub chunk_id: String,
    pub fused_score: f64,
    pub rerank_score: f64,
    pub text: String,
    pub metadata: ChunkMetadata,
}

/// Result after submodular selection. `nonconformity_score` is populated by the conformal gate.
#[derive(Debug, Clone)]
pub struct SelectedEvidence {
    pub chunk_id: String,
    pub fused_score: f64,
    pub rerank_score: f64,
    pub marginal_gain: f64,
    pub nonconformity_score: f64,
    pub text: String,
    pub metadata: ChunkMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn build_full_packet() -> EvidencePacket {
        EvidencePacket {
            query_id: QueryId("test-id".into()),
            trace_id: Some("trace-abc".into()),
            answer: Some("The lease term is 60 months.".into()),
            decision: Decision::Answered,
            coverage_guarantee: 0.95,
            empirical_coverage_last_30d: Some(0.947),
            route: Route::Standard,
            cache_hit: false,
            evidence: vec![EvidenceChunk {
                chunk_id: "doc_1:sec_4:p3".into(),
                source_doc: "lease.pdf".into(),
                doc_version_hash: "sha256:abc123".into(),
                path: vec!["Section 4".into(), "Term".into()],
                page: Some(14),
                char_span: Some([1882, 2043]),
                text: "The lease term is 60 months.".into(),
                retriever: RetrieverType::VectorIndex,
                rrf_score: 0.0164,
                rerank_score: Some(0.92),
                nonconformity_score: 0.03,
                submodular_marginal_gain: 0.41,
                ingested_at: Utc::now(),
                index_generation: 1,
                acl_groups: vec!["legal".into()],
                freshness_hours: 12.4,
                token_count: 128,
            }],
            audit: AuditBlock {
                model_version: Some("gpt-4o".into()),
                retrieval_stages_ms: RetrievalStagesMs {
                    vector_search: 45,
                    reranking: 67,
                    ..Default::default()
                },
                total_candidates_before_rerank: 100,
                total_candidates_after_rerank: 25,
                total_candidates_after_conformal: 8,
                calibration_set_id: "cal_v1".into(),
                session_hash: "sha256:deadbeef".into(),
            },
            latency_ms: 214,
            cost_usd: Some(0.0021),
        }
    }

    #[test]
    fn evidence_packet_round_trips_through_json() {
        let packet = build_full_packet();
        let json = serde_json::to_string(&packet).expect("serialization failed");
        let restored: EvidencePacket = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(packet.query_id, restored.query_id);
        assert_eq!(packet.decision, restored.decision);
        assert_eq!(packet.evidence.len(), restored.evidence.len());
        assert_eq!(
            packet.evidence[0].nonconformity_score,
            restored.evidence[0].nonconformity_score
        );
    }

    #[test]
    fn query_id_new_produces_unique_ids() {
        let ids: HashSet<String> = (0..1000).map(|_| QueryId::new().0).collect();
        assert_eq!(ids.len(), 1000, "duplicate QueryId generated");
    }

    #[test]
    fn decision_serializes_to_screaming_snake_case() {
        assert_eq!(
            serde_json::to_string(&Decision::EvidenceOnly).unwrap(),
            "\"EVIDENCE_ONLY\""
        );
        assert_eq!(
            serde_json::to_string(&Decision::Answered).unwrap(),
            "\"ANSWERED\""
        );
    }

    #[test]
    fn retrieval_config_default_values_are_sane() {
        let cfg = RetrievalConfig::default();
        assert!((cfg.coverage - 0.95).abs() < f64::EPSILON);
        assert_eq!(cfg.token_budget, 4000);
        assert_eq!(cfg.rrf_k, 60);
        assert_eq!(cfg.rerank_top_n, 100);
        assert!(!cfg.skip_cache);
    }
}
