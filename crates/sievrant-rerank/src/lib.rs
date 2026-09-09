pub mod colbert;
pub mod cross_encoder;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sievrant_core::{
    error::{RerankError, Result},
    types::{FusedCandidate, RerankedCandidate},
};

/// Core reranking contract. Every implementation must be thread-safe (Send + Sync)
/// so it can be shared across tokio tasks via Arc.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Score a batch of (query, document) pairs in order.
    /// Returns one score per document in the same order as the input slice.
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>>;

    /// Human-readable name for logging and audit records.
    fn name(&self) -> &str;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RerankerType {
    Onnx,
    Api,
    ColBert,
    /// Skip reranking; fused RRF scores are used directly.
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    pub reranker_type: RerankerType,
    pub model_path: Option<String>,
    pub api_url: Option<String>,
    pub api_key: Option<String>,
    pub model_name: Option<String>,
    /// Candidates passed to the reranker from RRF output.
    pub top_n: usize,
    pub max_sequence_length: usize,
    /// Batch size for ONNX inference.
    pub batch_size: usize,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            reranker_type: RerankerType::None,
            model_path: None,
            api_url: None,
            api_key: None,
            model_name: None,
            top_n: 100,
            max_sequence_length: 512,
            batch_size: 32,
        }
    }
}

/// API-based reranker. Calls any HTTP endpoint that accepts a Cohere-compatible
/// rerank request body and returns scored results. Compatible with:
///   - Cohere /v1/rerank
///   - Jina AI reranker API
///   - Local vllm/reranker servers exposing the same schema
pub struct ApiReranker {
    client: reqwest::Client,
    api_url: String,
    api_key: String,
    model: String,
}

impl ApiReranker {
    pub fn new(api_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_url: api_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl Reranker for ApiReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        #[derive(Serialize)]
        struct RerankRequest<'a> {
            model: &'a str,
            query: &'a str,
            documents: &'a [&'a str],
            top_n: usize,
            return_documents: bool,
        }

        #[derive(Deserialize)]
        struct RerankResponse {
            results: Vec<RerankResult>,
        }

        #[derive(Deserialize)]
        struct RerankResult {
            index: usize,
            relevance_score: f64,
        }

        let n = documents.len();
        let request_body = RerankRequest {
            model: &self.model,
            query,
            documents,
            top_n: n,
            return_documents: false,
        };

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| RerankError::Api(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(RerankError::Api(format!("HTTP {status}: {body}")).into());
        }

        let parsed: RerankResponse = response
            .json()
            .await
            .map_err(|e| RerankError::Api(e.to_string()))?;

        let mut scores = vec![0.0f64; n];
        for result in parsed.results {
            if result.index < n {
                scores[result.index] = result.relevance_score;
            }
        }

        Ok(scores)
    }

    fn name(&self) -> &str {
        "api-reranker"
    }
}

/// Take the top N fused candidates, rescore each (query, chunk) pair via the
/// reranker, and return results sorted by rerank score descending.
///
/// On reranker failure the function falls back to returning candidates sorted
/// by their original fused RRF score so the pipeline degrades gracefully.
pub async fn rerank_candidates(
    reranker: &dyn Reranker,
    query: &str,
    candidates: Vec<FusedCandidate>,
    top_n: usize,
) -> Result<Vec<RerankedCandidate>> {
    let top_candidates: Vec<FusedCandidate> = candidates.into_iter().take(top_n).collect();
    let texts: Vec<&str> = top_candidates.iter().map(|c| c.text.as_str()).collect();

    let scores = match reranker.rerank(query, &texts).await {
        Ok(s) => s,
        Err(_) => {
            // Graceful degradation: propagate fused scores as the rerank score
            // so the rest of the pipeline can continue unmodified.
            return Ok(top_candidates
                .into_iter()
                .map(|c| RerankedCandidate {
                    chunk_id: c.chunk_id,
                    fused_score: c.fused_score,
                    rerank_score: c.fused_score,
                    text: c.text,
                    metadata: c.metadata,
                })
                .collect());
        }
    };

    let mut reranked: Vec<RerankedCandidate> = top_candidates
        .into_iter()
        .zip(scores)
        .map(|(candidate, rerank_score)| RerankedCandidate {
            chunk_id: candidate.chunk_id,
            fused_score: candidate.fused_score,
            rerank_score,
            text: candidate.text,
            metadata: candidate.metadata,
        })
        .collect();

    reranked.sort_by(|a, b| {
        b.rerank_score
            .partial_cmp(&a.rerank_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(reranked)
}

/// A mock reranker used in tests and benchmarks that returns a deterministic
/// score based on how many query terms appear in the document text.
/// Does not require any model files to be present.
pub struct TermOverlapReranker;

#[async_trait]
impl Reranker for TermOverlapReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let scores = documents
            .iter()
            .map(|doc| {
                let text = doc.to_lowercase();
                query_terms
                    .iter()
                    .filter(|t| text.contains(t.to_lowercase().as_str()))
                    .count() as f64
            })
            .collect();
        Ok(scores)
    }

    fn name(&self) -> &str {
        "term-overlap-mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::types::ChunkMetadata;

    fn make_fused(id: &str, score: f64, text: &str) -> FusedCandidate {
        FusedCandidate {
            chunk_id: id.to_string(),
            fused_score: score,
            source_scores: vec![],
            text: text.to_string(),
            metadata: ChunkMetadata {
                doc_id: id.into(),
                source_doc: "test.pdf".into(),
                doc_version_hash: "sha256:test".into(),
                path: vec![],
                page: None,
                char_span: None,
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: vec![],
                token_count: 10,
            },
        }
    }

    #[tokio::test]
    async fn reranking_changes_order_when_text_matches_query() {
        let reranker = TermOverlapReranker;
        let candidates = vec![
            make_fused("c1", 0.9, "unrelated content about machinery"),
            make_fused("c2", 0.5, "lease term is sixty months duration"),
            make_fused("c3", 0.4, "payment schedule and amounts"),
        ];

        let result = rerank_candidates(&reranker, "lease term duration", candidates, 10)
            .await
            .unwrap();

        // c2 matches "lease", "term", "duration" so should rank first after reranking
        assert_eq!(result[0].chunk_id, "c2", "reranking should promote c2");
    }

    #[tokio::test]
    async fn reranking_preserves_top_n_limit() {
        let reranker = TermOverlapReranker;
        let candidates: Vec<FusedCandidate> = (0..20)
            .map(|i| make_fused(&format!("c{i}"), i as f64, "some text"))
            .collect();

        let result = rerank_candidates(&reranker, "query", candidates, 5)
            .await
            .unwrap();

        assert_eq!(result.len(), 5);
    }

    #[tokio::test]
    async fn reranking_falls_back_on_error() {
        struct AlwaysFailsReranker;

        #[async_trait]
        impl Reranker for AlwaysFailsReranker {
            async fn rerank(&self, _query: &str, _documents: &[&str]) -> Result<Vec<f64>> {
                Err(RerankError::Unavailable("mock failure".into()).into())
            }

            fn name(&self) -> &str {
                "always-fails"
            }
        }

        let candidates = vec![
            make_fused("c1", 0.8, "text a"),
            make_fused("c2", 0.6, "text b"),
        ];

        let result = rerank_candidates(&AlwaysFailsReranker, "query", candidates, 10)
            .await
            .unwrap();

        // Fallback: fused scores used as rerank scores, order preserved
        assert_eq!(result.len(), 2);
        assert!((result[0].rerank_score - 0.8).abs() < 1e-10);
    }
}
