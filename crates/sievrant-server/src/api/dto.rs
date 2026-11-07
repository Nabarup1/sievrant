use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sievrant_core::types::{DocumentMetadata, EvidencePacket};
use utoipa::ToSchema;

pub const MAX_QUERY_CHARS: usize = 8_000;
pub const MAX_TOKEN_BUDGET: usize = 200_000;
pub const MAX_DOCS_PER_REQUEST: usize = 1_000;
pub const MAX_CALIBRATION_EXAMPLES: usize = 10_000;
pub const MAX_BOOTSTRAP_EXAMPLES: usize = 2_000;

/// Serialize an enum to its wire string (e.g. `Decision::Answered` -> "ANSWERED").
pub fn wire<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct QueryPayload {
    pub query: String,
    pub corpus: String,
    /// Coverage target in (0, 1). Defaults to the server's `SIEVRANT_DEFAULT_COVERAGE`.
    pub coverage: Option<f64>,
    /// Token budget for selected evidence. Defaults to `SIEVRANT_TOKEN_BUDGET`.
    pub token_budget: Option<usize>,
    /// ACL groups the end user may read. Empty means `public` only.
    #[serde(default)]
    pub acl_groups: Vec<String>,
    #[serde(default)]
    pub skip_cache: bool,
    /// Set false to return evidence without an LLM answer.
    pub generate_answer: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct QueryResponse {
    pub query_id: String,
    /// ANSWERED, ABSTAINED or EVIDENCE_ONLY.
    pub decision: String,
    pub answer: Option<String>,
    pub evidence_count: usize,
    /// Guaranteed coverage, or null when no guarantee applies (see `guarantee_status`).
    pub coverage_guarantee: Option<f64>,
    /// calibrated, uncalibrated, insufficient_data, unachievable, stale or degraded.
    pub guarantee_status: String,
    pub guarantee_statement: String,
    pub latency_ms: u64,
    pub cache_hit: bool,
    pub route: String,
}

impl From<&EvidencePacket> for QueryResponse {
    fn from(p: &EvidencePacket) -> Self {
        Self {
            query_id: p.query_id.0.clone(),
            decision: wire(&p.decision),
            answer: p.answer.clone(),
            evidence_count: p.evidence.len(),
            coverage_guarantee: p.coverage_guarantee,
            guarantee_status: wire(&p.guarantee.status),
            guarantee_statement: p.guarantee.statement.clone(),
            latency_ms: p.latency_ms,
            cache_hit: p.cache_hit,
            route: wire(&p.route),
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
    /// Known keys: filename, title, content_type, source_url, author,
    /// created_at (RFC 3339). Anything else is stored as custom metadata.
    #[schema(value_type = Option<Object>)]
    pub metadata: Option<serde_json::Value>,
    /// Access groups. Empty means `public`.
    #[serde(default)]
    pub acl_groups: Vec<String>,
}

/// Map free-form JSON metadata onto [`DocumentMetadata`].
pub fn metadata_from_json(value: Option<serde_json::Value>) -> Result<DocumentMetadata, String> {
    let mut meta = DocumentMetadata::default();
    let obj = match value {
        None | Some(serde_json::Value::Null) => return Ok(meta),
        Some(serde_json::Value::Object(o)) => o,
        Some(_) => return Err("metadata must be a JSON object".into()),
    };
    let mut custom = HashMap::new();
    for (k, v) in obj {
        let s = v.as_str().map(str::to_owned);
        match (k.as_str(), s) {
            ("filename", Some(s)) => meta.filename = Some(s),
            ("title", Some(s)) => meta.title = Some(s),
            ("content_type", Some(s)) => meta.content_type = Some(s),
            ("source_url", Some(s)) => meta.source_url = Some(s),
            ("author", Some(s)) => meta.author = Some(s),
            ("created_at", Some(s)) => {
                meta.created_at = Some(
                    chrono::DateTime::parse_from_rfc3339(&s)
                        .map_err(|e| format!("metadata.created_at: {e}"))?
                        .with_timezone(&chrono::Utc),
                )
            }
            _ => {
                custom.insert(k, v);
            }
        }
    }
    meta.custom = custom;
    Ok(meta)
}

#[derive(Debug, Serialize, ToSchema)]
pub struct IngestFailureDto {
    pub doc_id: String,
    pub reason: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct IngestResponse {
    pub corpus: String,
    /// Documents added or replaced.
    pub ingested: usize,
    /// Documents skipped because their content was unchanged.
    pub unchanged: usize,
    pub failed: usize,
    pub chunks: usize,
    pub failures: Vec<IngestFailureDto>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct FeedbackPayload {
    pub corpus: String,
    /// `query_id` from a previous query response.
    pub query_id: String,
    #[serde(default)]
    pub relevant_doc_ids: Vec<String>,
    #[serde(default)]
    pub relevant_chunk_ids: Vec<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CalibrationExampleDto {
    pub query: String,
    /// Leave both lists empty for a query the corpus cannot answer.
    #[serde(default)]
    pub relevant_doc_ids: Vec<String>,
    #[serde(default)]
    pub relevant_chunk_ids: Vec<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CalibrationPayload {
    pub corpus: String,
    pub examples: Vec<CalibrationExampleDto>,
    /// "labeled" (default) or "synthetic".
    pub source: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct BootstrapPayload {
    pub corpus: String,
    /// Number of LLM-generated questions (default 100).
    pub n: Option<usize>,
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CorpusRef {
    pub corpus: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct GraphIngestPayload {
    pub corpus: String,
    pub requests: Vec<sievrant_core::types::GraphIngestRequest>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct TreeSummaryPayload {
    pub corpus: String,
    pub requests: Vec<sievrant_core::types::TreeSummaryRequest>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AcceptedResponse {
    pub corpus: String,
    pub ingested: usize,
}

#[derive(Debug, Deserialize)]
pub struct PageParams {
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct CoverageParams {
    pub coverage: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_maps_known_keys_and_keeps_custom() {
        let m = metadata_from_json(Some(serde_json::json!({
            "title": "Lease", "author": "Legal", "created_at": "2025-01-01T00:00:00Z",
            "dept": "finance", "pages": 12
        })))
        .unwrap();
        assert_eq!(m.title.as_deref(), Some("Lease"));
        assert_eq!(m.author.as_deref(), Some("Legal"));
        assert!(m.created_at.is_some());
        assert_eq!(m.custom.len(), 2);
        assert!(metadata_from_json(Some(serde_json::json!("x"))).is_err());
        assert!(metadata_from_json(Some(serde_json::json!({"created_at": "nope"}))).is_err());
    }
}
