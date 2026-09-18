use async_trait::async_trait;
use ort::session::{builder::GraphOptimizationLevel, Session};
use sievrant_core::error::{RerankError, Result};

use crate::Reranker;

/// ColBERT late-interaction reranker.
///
/// Instead of a full cross-attention pass over the concatenated pair,
/// ColBERT encodes query and document independently into per-token dense
/// vectors, then scores via MaxSim: for each query token, find the maximum
/// cosine similarity to any document token, then sum across all query tokens.
///
/// Trade-off vs cross-encoder:
///   - Document embeddings are precomputable at index time (lower query latency)
///   - Slightly lower accuracy than full cross-attention (~1-3 NDCG@10 points)
///
/// This implementation runs both query and document encoding through the same
/// ONNX session, so it supports any model exported with a shared encoder.
pub struct ColBertReranker {
    session: std::sync::Mutex<Session>,
    max_query_tokens: usize,
    max_doc_tokens: usize,
}

impl ColBertReranker {
    pub fn load(model_path: &str, max_query_tokens: usize, max_doc_tokens: usize) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| RerankError::ModelLoad(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RerankError::ModelLoad(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| RerankError::ModelLoad(e.to_string()))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            max_query_tokens,
            max_doc_tokens,
        })
    }

    /// MaxSim scoring between two token embedding matrices.
    ///
    /// For each query token vector q_i, find max over all document token vectors
    /// d_j of the dot product q_i . d_j. Sum these maxima across all query tokens.
    ///
    /// Assumes embeddings are already L2 normalised (cosine similarity = dot product).
    pub fn max_sim_score(query_embeddings: &[Vec<f32>], doc_embeddings: &[Vec<f32>]) -> f64 {
        sievrant_math::simd::fast_max_sim_score(query_embeddings, doc_embeddings)
    }

    fn encode_tokens(&self, text: &str, max_len: usize) -> Result<Vec<Vec<f32>>> {
        let tokens: Vec<i64> = std::iter::once(101i64)
            .chain(
                text.split_whitespace()
                    .take(max_len.saturating_sub(2))
                    .map(|w| w.bytes().map(|b| b as i64).sum::<i64>() % 30000 + 1000),
            )
            .chain(std::iter::once(102i64))
            .collect();

        let seq_len = tokens.len();
        let attention_mask: Vec<i64> = vec![1; seq_len];
        let token_type_ids: Vec<i64> = vec![0; seq_len];
        let _shape = [1i64, seq_len as i64];

        let input_ids_tensor = ort::value::Tensor::from_array(([1usize, seq_len], tokens))
            .map_err(|e| RerankError::Inference(e.to_string()))?;
        let attn_tensor = ort::value::Tensor::from_array(([1usize, seq_len], attention_mask))
            .map_err(|e| RerankError::Inference(e.to_string()))?;
        let type_tensor = ort::value::Tensor::from_array(([1usize, seq_len], token_type_ids))
            .map_err(|e| RerankError::Inference(e.to_string()))?;

        let mut session_guard = self.session.lock().unwrap();
        let outputs = session_guard
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attn_tensor,
                "token_type_ids" => type_tensor,
            ])
            .map_err(|e| RerankError::Inference(e.to_string()))?;

        let tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| RerankError::Inference(e.to_string()))?;

        let raw = tensor.1;
        let hidden_dim = raw.len() / seq_len;

        let embeddings = (0..seq_len)
            .map(|i| raw[i * hidden_dim..(i + 1) * hidden_dim].to_vec())
            .collect();

        Ok(embeddings)
    }
}

#[async_trait]
impl Reranker for ColBertReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        let query_emb = self.encode_tokens(query, self.max_query_tokens)?;

        documents
            .iter()
            .map(|doc| {
                let doc_emb = self.encode_tokens(doc, self.max_doc_tokens)?;
                Ok(Self::max_sim_score(&query_emb, &doc_emb))
            })
            .collect()
    }

    fn name(&self) -> &str {
        "colbert"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_sim_empty_inputs_return_zero() {
        assert_eq!(ColBertReranker::max_sim_score(&[], &[vec![1.0, 0.0]]), 0.0);
        assert_eq!(ColBertReranker::max_sim_score(&[vec![1.0, 0.0]], &[]), 0.0);
    }

    #[test]
    fn max_sim_identical_vectors_return_high_score() {
        let q = vec![vec![1.0f32, 0.0]];
        let d = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let score = ColBertReranker::max_sim_score(&q, &d);
        assert!((score - 1.0).abs() < 1e-6, "expected ~1.0, got {score}");
    }

    #[test]
    fn max_sim_orthogonal_vectors_contribute_zero() {
        let q = vec![vec![1.0f32, 0.0]];
        let d = vec![vec![0.0f32, 1.0]];
        let score = ColBertReranker::max_sim_score(&q, &d);
        assert!(score.abs() < 1e-6, "expected ~0.0, got {score}");
    }
}
