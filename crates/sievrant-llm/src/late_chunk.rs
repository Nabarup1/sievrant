use async_trait::async_trait;

use sievrant_core::error::{LlmError, Result, SievrantError};

/// Contract for embedding providers that support Late Chunking.
///
/// Late Chunking passes the entire document through the transformer once,
/// retaining per-token hidden states. It then mean-pools the token states
/// that fall within each chunk's character span to produce contextually
/// rich chunk embeddings that carry global document context.
///
/// Standard providers (OpenAI text-embedding-3) do not expose per-token
/// embeddings. This trait is implemented by local ONNX models and the Jina API.
#[async_trait]
pub trait LateChunkingProvider: Send + Sync {
    /// Embed a document using late chunking.
    ///
    /// `document_text` is the full original document string.
    /// `chunk_spans` are [start_byte, end_byte) half-open character ranges.
    /// Returns one embedding vector per span, in the same order as the input.
    async fn embed_late_chunking(
        &self,
        document_text: &str,
        chunk_spans: &[[usize; 2]],
    ) -> Result<Vec<Vec<f32>>>;

    /// Dimensionality of the output vectors.
    fn dimensions(&self) -> usize;
}

/// Token alignment helper: maps character byte offsets to token indices.
///
/// Given the byte offsets of each token (as produced by the model's tokenizer),
/// returns the range of token indices [start_token, end_token) that fall
/// within the character span `[span_start, span_end)`.
///
/// This is the critical alignment step in Late Chunking. An off-by-one
/// error here would pool the wrong tokens and corrupt the embedding.
pub fn map_span_to_tokens(
    token_byte_offsets: &[(usize, usize)],
    span_start: usize,
    span_end: usize,
) -> Option<(usize, usize)> {
    let mut start_tok = None;
    let mut end_tok = None;

    for (i, &(tok_start, tok_end)) in token_byte_offsets.iter().enumerate() {
        // A token is "inside" the span if it overlaps: tok_end > span_start AND tok_start < span_end
        if tok_end > span_start && tok_start < span_end {
            if start_tok.is_none() {
                start_tok = Some(i);
            }
            end_tok = Some(i + 1); // exclusive upper bound
        }
    }

    match (start_tok, end_tok) {
        (Some(s), Some(e)) => Some((s, e)),
        _ => None,
    }
}

/// Mean-pool a slice of token hidden-state vectors into a single embedding.
///
/// All token vectors must have the same dimensionality. Returns `None` if the
/// slice is empty to prevent division-by-zero panics.
pub fn mean_pool(token_vectors: &[&[f32]]) -> Option<Vec<f32>> {
    if token_vectors.is_empty() {
        return None;
    }

    let dim = token_vectors[0].len();
    let mut pooled = vec![0.0f32; dim];

    for vec in token_vectors {
        debug_assert_eq!(
            vec.len(),
            dim,
            "all token vectors must have the same dimensionality"
        );
        for (acc, v) in pooled.iter_mut().zip(vec.iter()) {
            *acc += v;
        }
    }

    let n = token_vectors.len() as f32;
    for val in pooled.iter_mut() {
        *val /= n;
    }

    Some(pooled)
}

/// Pool embeddings from a flat list of token vectors using pre-computed
/// token index spans. Returns one vector per span.
///
/// This is the core operation of the late chunking pipeline:
/// all token hidden states arrive as a flat matrix, and each chunk gets
/// the mean of its corresponding slice.
pub fn pool_chunks(
    all_token_vectors: &[Vec<f32>],
    token_spans: &[(usize, usize)],
) -> Result<Vec<Vec<f32>>> {
    let mut result = Vec::with_capacity(token_spans.len());

    for &(start, end) in token_spans {
        if end > all_token_vectors.len() {
            return Err(SievrantError::Llm(LlmError::ParseError {
                provider: "late-chunking".into(),
                reason: format!(
                    "token span [{start}, {end}) exceeds token count {}",
                    all_token_vectors.len()
                ),
            }));
        }
        if start >= end {
            return Err(SievrantError::Llm(LlmError::ParseError {
                provider: "late-chunking".into(),
                reason: format!("token span [{start}, {end}) is empty or inverted"),
            }));
        }

        let slice: Vec<&[f32]> = all_token_vectors[start..end]
            .iter()
            .map(|v| v.as_slice())
            .collect();

        let pooled = mean_pool(&slice).ok_or_else(|| {
            SievrantError::Llm(LlmError::ParseError {
                provider: "late-chunking".into(),
                reason: "empty token slice for pooling".into(),
            })
        })?;

        result.push(pooled);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_averages_correctly() {
        let v1 = vec![1.0f32, 2.0, 3.0];
        let v2 = vec![3.0f32, 4.0, 5.0];
        let pooled = mean_pool(&[&v1, &v2]).unwrap();
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 3.0).abs() < 1e-6);
        assert!((pooled[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_empty_returns_none() {
        assert!(mean_pool(&[]).is_none());
    }

    #[test]
    fn map_span_to_tokens_basic_alignment() {
        // Token 0: bytes [0, 4), Token 1: bytes [5, 9), Token 2: bytes [10, 14)
        let offsets = vec![(0, 4), (5, 9), (10, 14)];
        let (s, e) = map_span_to_tokens(&offsets, 0, 9).unwrap();
        assert_eq!(s, 0);
        assert_eq!(e, 2);
    }

    #[test]
    fn map_span_to_tokens_no_overlap_returns_none() {
        let offsets = vec![(0, 4), (5, 9)];
        assert!(map_span_to_tokens(&offsets, 20, 30).is_none());
    }

    #[test]
    fn pool_chunks_rejects_out_of_bounds_span() {
        let tokens = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let err = pool_chunks(&tokens, &[(0, 10)]);
        assert!(err.is_err());
    }

    #[test]
    fn pool_chunks_correct_output_count() {
        let tokens: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32, i as f32]).collect();
        let spans = vec![(0, 2), (2, 4), (4, 6)];
        let result = pool_chunks(&tokens, &spans).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn pool_chunks_rejects_empty_span() {
        let tokens = vec![vec![1.0f32], vec![2.0f32]];
        let err = pool_chunks(&tokens, &[(1, 1)]);
        assert!(err.is_err());
        let err_inverted = pool_chunks(&tokens, &[(2, 1)]);
        assert!(err_inverted.is_err());
    }
}
