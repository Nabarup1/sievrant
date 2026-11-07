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

/// The taxonomy category of a query for Mondrian Conformal Prediction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QueryTaxonomy {
    Factoid,
    Summarization,
    Analytical,
    Coding,
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

    /// Reranker backend that produced the scores the gate used.
    #[serde(default)]
    pub reranker: Option<String>,

    /// Embedding model used for the vector index.
    #[serde(default)]
    pub embedding_model: Option<String>,

    /// Hash of every setting that affects candidate scores. Calibration is
    /// only valid for the fingerprint it was computed with.
    #[serde(default)]
    pub pipeline_fingerprint: Option<String>,

    /// Non-fatal problems during this request (failed index, reranker fallback).
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl Default for AuditBlock {
    fn default() -> Self {
        Self {
            model_version: None,
            retrieval_stages_ms: RetrievalStagesMs::default(),
            total_candidates_before_rerank: 0,
            total_candidates_after_rerank: 0,
            total_candidates_after_conformal: 0,
            calibration_set_id: "none".into(),
            session_hash: String::new(),
            reranker: None,
            embedding_model: None,
            pipeline_fingerprint: None,
            warnings: Vec::new(),
        }
    }
}

/// Whether the statistical coverage guarantee applies to a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuaranteeStatus {
    /// Calibrated for this pipeline; the guarantee applies.
    Calibrated,
    /// No calibration set exists for this corpus.
    Uncalibrated,
    /// Calibration exists but has too few examples for the requested coverage.
    InsufficientData,
    /// The retriever misses relevant evidence too often to reach the target.
    Unachievable,
    /// The pipeline or corpus changed since calibration; recalibrate.
    Stale,
    /// A pipeline component failed for this request (e.g. reranker fallback).
    Degraded,
}

/// Where the calibration labels came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationSource {
    /// Human-labelled (query, relevant documents) pairs.
    Labeled,
    /// Questions generated from the corpus by an LLM. Assumes real traffic
    /// resembles the generated questions.
    Synthetic,
    /// Labels collected through the feedback API.
    Feedback,
}

/// Observed coverage from user feedback, with a 95% Wilson interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedCoverage {
    pub rate: f64,
    pub observations: usize,
    pub lower_95: f64,
    pub upper_95: f64,
    /// True when the interval lies entirely below the target minus tolerance.
    pub drift_alert: bool,
}

/// Machine- and human-readable statement of what this response guarantees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuaranteeInfo {
    pub status: GuaranteeStatus,
    /// Plain-language description of what holds (or why nothing does).
    pub statement: String,
    pub requested_coverage: f64,
    pub calibration_set_id: Option<String>,
    pub calibration_source: Option<CalibrationSource>,
    pub calibration_size: Option<usize>,
    /// Nonconformity threshold applied by the gate.
    pub threshold: Option<f64>,
    /// Candidates that passed the gate, before token-budget selection.
    pub conformal_set_size: usize,
    /// True when the token budget dropped members of the conformal set. The
    /// guarantee covers the full set; returned evidence is a subset.
    pub budget_truncated: bool,
    /// Highest coverage this calibration set supports, when the target is unachievable.
    pub max_achievable_coverage: Option<f64>,
    pub observed: Option<ObservedCoverage>,
}

impl GuaranteeInfo {
    pub fn none(status: GuaranteeStatus, requested_coverage: f64, statement: &str) -> Self {
        Self {
            status,
            statement: statement.to_string(),
            requested_coverage,
            calibration_set_id: None,
            calibration_source: None,
            calibration_size: None,
            threshold: None,
            conformal_set_size: 0,
            budget_truncated: false,
            max_achievable_coverage: None,
            observed: None,
        }
    }
}

/// Signed retrieval receipt. `payload` is the exact JSON string that was
/// signed, so any language can verify it without canonicalisation rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    /// Always "ed25519".
    pub alg: String,
    /// First 16 hex chars of SHA-256(public key).
    pub key_id: String,
    pub payload: String,
    /// Base64 signature over `payload` bytes.
    pub signature: String,
}

/// Version of the Evidence Packet wire format.
pub const EVIDENCE_PACKET_SCHEMA_VERSION: &str = "1.0";

fn default_schema_version() -> String {
    EVIDENCE_PACKET_SCHEMA_VERSION.to_string()
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
    /// Wire-format version. Additive changes keep "1.x"; breaking changes bump the major.
    #[serde(default = "default_schema_version")]
    pub schema_version: String,

    /// Unique ID for this retrieval request. Ties together logs, traces, and audit records.
    pub query_id: QueryId,

    /// OpenTelemetry trace ID. Set when OTel is enabled; used to correlate with Jaeger/Tempo.
    pub trace_id: Option<String>,

    /// Generated answer text. None when decision is ABSTAINED or EVIDENCE_ONLY.
    pub answer: Option<String>,

    /// Retrieval outcome. Agent uses this to branch: render answer, show "I don't know", or
    /// present raw evidence for human review.
    pub decision: Decision,

    /// Coverage level guaranteed for this response, e.g. 0.95. `None` whenever
    /// the guarantee does not apply; see `guarantee.status` for why.
    pub coverage_guarantee: Option<f64>,

    /// Details of the guarantee: status, threshold, calibration provenance.
    pub guarantee: GuaranteeInfo,

    /// Observed coverage over the last 30 days from feedback, if any.
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

    /// Signed receipt binding query, evidence, answer and guarantee. Present
    /// when a signing key is configured.
    #[serde(default)]
    pub receipt: Option<Receipt>,
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

/// A request to explicitly ingest extracted entities and relations into the Graph Index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphIngestRequest {
    pub chunk_id: String,
    pub entities: Vec<String>,
    pub relations: Vec<RelationDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationDto {
    pub source: String,
    pub target: String,
    pub relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSummaryRequest {
    pub node_id: String,
    pub summary: String,
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
    /// ACL groups the caller is allowed to read. `None` or empty means `["public"]`.
    pub acl_groups: Option<Vec<String>>,
    /// Generate an LLM answer. `false` returns evidence only.
    pub generate_answer: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            coverage: 0.95,
            token_budget: 4000,
            route: RouteStrategy::Auto,
            user_id: None,
            skip_cache: false,
            rerank_top_n: 50,
            rrf_k: 60,
            acl_groups: None,
            generate_answer: true,
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
            schema_version: EVIDENCE_PACKET_SCHEMA_VERSION.into(),
            query_id: QueryId("test-id".into()),
            trace_id: Some("trace-abc".into()),
            answer: Some("The lease term is 60 months.".into()),
            decision: Decision::Answered,
            coverage_guarantee: Some(0.95),
            guarantee: GuaranteeInfo::none(GuaranteeStatus::Calibrated, 0.95, "test"),
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
                ..Default::default()
            },
            latency_ms: 214,
            cost_usd: Some(0.0021),
            receipt: None,
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
    fn packets_without_new_fields_still_deserialize() {
        let mut v = serde_json::to_value(build_full_packet()).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("schema_version");
        obj.remove("receipt");
        let p: EvidencePacket = serde_json::from_value(v).unwrap();
        assert_eq!(p.schema_version, EVIDENCE_PACKET_SCHEMA_VERSION);
        assert!(p.receipt.is_none());
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
        assert_eq!(cfg.rerank_top_n, 50);
        assert!(cfg.generate_answer);
        assert!(!cfg.skip_cache);
    }
}
