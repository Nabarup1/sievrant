use async_trait::async_trait;
use ort::session::{Session, builder::GraphOptimizationLevel};
use sievrant_core::error::{RerankError, Result};

use crate::Reranker;

/// Whitespace tokenizer that maps tokens to integer IDs by byte value.
/// This is intentionally minimal; in production swap for a real WordPiece/SentencePiece
/// tokenizer loaded alongside the ONNX model weights.
struct SimpleTokenizer {
    max_length: usize,
}

impl SimpleTokenizer {
    fn new(max_length: usize) -> Self {
        Self { max_length }
    }

    /// Encode a (query, document) pair into input_ids, attention_mask, token_type_ids.
    /// Format: [CLS] query tokens [SEP] document tokens [SEP]
    /// All tensors are padded or truncated to max_length.
    fn encode_pair(&self, query: &str, document: &str) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        const CLS: i64 = 101;
        const SEP: i64 = 102;
        const PAD: i64 = 0;

        let query_tokens: Vec<i64> = query.split_whitespace()
            .map(|w| w.bytes().map(|b| b as i64).sum::<i64>() % 30000 + 1000)
            .collect();

        let doc_tokens: Vec<i64> = document.split_whitespace()
            .map(|w| w.bytes().map(|b| b as i64).sum::<i64>() % 30000 + 1000)
            .collect();

        // Reserve slots for CLS + SEP + SEP
        let budget = self.max_length.saturating_sub(3);
        let query_len = query_tokens.len().min(budget / 2);
        let doc_len = doc_tokens.len().min(budget - query_len);

        let mut input_ids = Vec::with_capacity(self.max_length);
        let mut token_type_ids = Vec::with_capacity(self.max_length);

        input_ids.push(CLS);
        token_type_ids.push(0i64);

        for &t in &query_tokens[..query_len] {
            input_ids.push(t);
            token_type_ids.push(0);
        }

        input_ids.push(SEP);
        token_type_ids.push(0);

        for &t in &doc_tokens[..doc_len] {
            input_ids.push(t);
            token_type_ids.push(1);
        }

        input_ids.push(SEP);
        token_type_ids.push(1);

        let seq_len = input_ids.len();
        let mut attention_mask: Vec<i64> = vec![1; seq_len];

        while input_ids.len() < self.max_length {
            input_ids.push(PAD);
            attention_mask.push(0);
            token_type_ids.push(0);
        }

        (input_ids, attention_mask, token_type_ids)
    }
}

/// Cross encoder that runs locally via ONNX Runtime.
/// Expected model: a two-class BERT-style classifier (e.g. bge-reranker-v2-m3 exported to ONNX).
/// The relevance score is derived from the logit of the positive class.
pub struct OnnxCrossEncoder {
    session: std::sync::Mutex<Session>,
    tokenizer: SimpleTokenizer,
    batch_size: usize,
}

impl OnnxCrossEncoder {
    pub fn load(model_path: &str, max_length: usize, batch_size: usize) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| RerankError::ModelLoad(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RerankError::ModelLoad(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| RerankError::ModelLoad(e.to_string()))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer: SimpleTokenizer::new(max_length),
            batch_size,
        })
    }

    fn run_batch(
        &self,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<f64>> {
        let mut all_scores = Vec::with_capacity(documents.len());

        for doc_batch in documents.chunks(self.batch_size) {
            let batch_len = doc_batch.len();
            let max_len = self.tokenizer.max_length;

            let mut input_ids_flat = Vec::with_capacity(batch_len * max_len);
            let mut attention_mask_flat = Vec::with_capacity(batch_len * max_len);
            let mut token_type_flat = Vec::with_capacity(batch_len * max_len);

            for doc in doc_batch {
                let (ids, mask, types) = self.tokenizer.encode_pair(query, doc);
                input_ids_flat.extend(ids);
                attention_mask_flat.extend(mask);
                token_type_flat.extend(types);
            }

            let _shape = [batch_len as i64, max_len as i64];

            let input_ids_tensor = ort::value::Tensor::from_array(([batch_len, max_len], input_ids_flat))
                .map_err(|e| RerankError::Inference(e.to_string()))?;
            let attn_tensor = ort::value::Tensor::from_array(([batch_len, max_len], attention_mask_flat))
                .map_err(|e| RerankError::Inference(e.to_string()))?;
            let type_tensor = ort::value::Tensor::from_array(([batch_len, max_len], token_type_flat))
                .map_err(|e| RerankError::Inference(e.to_string()))?;

            let mut session_guard = self.session.lock().unwrap();
            let outputs = session_guard.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attn_tensor,
                "token_type_ids" => type_tensor,
            ]).map_err(|e| RerankError::Inference(e.to_string()))?;

            let logits = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| RerankError::Inference(e.to_string()))?;

            let logits_slice = logits.1;

            // Assume shape [batch, 2]; positive logit is index 1.
            // Single-output models (shape [batch, 1]) fall back to index 0.
            let logit_stride = if logits_slice.len() / batch_len == 2 { 2 } else { 1 };
            let pos_offset = if logit_stride == 2 { 1 } else { 0 };

            for i in 0..batch_len {
                let score = logits_slice[i * logit_stride + pos_offset] as f64;
                all_scores.push(score);
            }
        }

        Ok(all_scores)
    }
}

#[async_trait]
impl Reranker for OnnxCrossEncoder {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        self.run_batch(query, documents)
    }

    fn name(&self) -> &str {
        "onnx-cross-encoder"
    }
}
