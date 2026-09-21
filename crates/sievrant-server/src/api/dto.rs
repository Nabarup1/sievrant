use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use sievrant_core::types::EvidencePacket;

#[derive(Debug, Deserialize, ToSchema)]
pub struct QueryPayload {
    pub query: String,
    pub corpus: String,
    #[serde(default = "default_coverage")]
    pub coverage: f64,
    #[serde(default = "default_budget")]
    pub token_budget: usize,
    #[serde(default)]
    pub acl_groups: Vec<String>,
    #[serde(default)]
    pub skip_cache: bool,
}

fn default_coverage() -> f64 {
    0.95
}
fn default_budget() -> usize {
    4000
}

#[derive(Debug, Serialize, ToSchema)]
pub struct QueryResponse {
    pub query_id: String,
    pub decision: String,
    pub answer: Option<String>,
    pub evidence_count: usize,
    pub coverage_guarantee: f64,
    pub latency_ms: u64,
    pub cache_hit: bool,
    pub route: String,
}

impl From<&EvidencePacket> for QueryResponse {
    fn from(p: &EvidencePacket) -> Self {
        Self {
            query_id: p.query_id.0.clone(),
            decision: format!("{:?}", p.decision),
            answer: p.answer.clone(),
            evidence_count: p.evidence.len(),
            coverage_guarantee: p.coverage_guarantee,
            latency_ms: p.latency_ms,
            cache_hit: p.cache_hit,
            route: format!("{:?}", p.route),
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct IngestPayload {
    pub corpus: String,
    pub documents: Vec<IngestDocument>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct IngestDocument {
    pub id: String,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct IngestResponse {
    pub corpus: String,
    pub ingested: usize,
    pub failed: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct GraphIngestPayload {
    pub corpus: String,
    pub requests: Vec<sievrant_core::types::GraphIngestRequest>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct GraphIngestResponse {
    pub corpus: String,
    pub ingested: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct TreeSummaryPayload {
    pub corpus: String,
    pub requests: Vec<sievrant_core::types::TreeSummaryRequest>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TreeSummaryResponse {
    pub corpus: String,
    pub ingested: usize,
}
