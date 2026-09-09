use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A complete record of token usage for a single retrieval request.
///
/// The `tokens_saved` field is the core SaaS marketing metric:
/// how many tokens Conformal Prediction avoided versus naive top-K RAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub tenant_id: String,
    pub query_id: String,
    /// Tokens consumed by the retrieval stage (embedding, reranking prompts).
    pub input_tokens_retrieval: usize,
    /// Tokens consumed by the LLM generation call.
    pub input_tokens_generation: usize,
    pub output_tokens: usize,
    /// Estimate of what a naive top-K=20 RAG would have consumed.
    /// Always >= actual tokens when Conformal Prediction is working correctly.
    pub naive_rag_token_estimate: usize,
    /// naive_rag_token_estimate - (input_tokens_retrieval + input_tokens_generation)
    pub tokens_saved: usize,
    pub model_id: String,
    pub timestamp: DateTime<Utc>,
    /// Cost in USD cents at the model's published price.
    pub cost_usd_cents: f64,
}

impl UsageRecord {
    /// Build a UsageRecord and compute the derived `tokens_saved` field.
    ///
    /// `naive_k` is the assumed top-K for the naive baseline (default: 20).
    /// `avg_chunk_tokens` is the average chunk size used to estimate naive cost.
    pub fn new(
        tenant_id: impl Into<String>,
        query_id: impl Into<String>,
        model_id: impl Into<String>,
        input_tokens_retrieval: usize,
        input_tokens_generation: usize,
        output_tokens: usize,
        actual_chunks_retrieved: usize,
        avg_chunk_tokens: usize,
        cost_per_1k_tokens_usd: f64,
    ) -> Self {
        const NAIVE_K: usize = 20;
        let naive_rag_token_estimate = NAIVE_K * avg_chunk_tokens + input_tokens_generation;
        let actual_total = input_tokens_retrieval + input_tokens_generation + output_tokens;
        let tokens_saved = naive_rag_token_estimate.saturating_sub(actual_total);
        let cost_usd_cents = (actual_total as f64 / 1000.0) * cost_per_1k_tokens_usd * 100.0;

        Self {
            tenant_id: tenant_id.into(),
            query_id: query_id.into(),
            model_id: model_id.into(),
            input_tokens_retrieval,
            input_tokens_generation,
            output_tokens,
            naive_rag_token_estimate,
            tokens_saved,
            timestamp: Utc::now(),
            cost_usd_cents,
        }
    }

    pub fn total_input_tokens(&self) -> usize {
        self.input_tokens_retrieval + self.input_tokens_generation
    }

    pub fn total_tokens(&self) -> usize {
        self.input_tokens_retrieval + self.input_tokens_generation + self.output_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn naive_estimate_always_gte_actual_when_conformal_saves_tokens() {
        let rec = UsageRecord::new(
            "tenant-1", "q-1", "gpt-4o",
            50,   // retrieval tokens (3 chunks * ~17 tokens of context)
            200,  // generation input
            100,  // output
            3,    // actual chunks retrieved by conformal gate
            200,  // avg chunk tokens
            0.03, // cost per 1k tokens
        );
        assert!(rec.naive_rag_token_estimate >= rec.input_tokens_retrieval + rec.input_tokens_generation);
        assert!(rec.tokens_saved > 0, "should have saved tokens vs naive top-20");
    }

    proptest! {
        #[test]
        fn tokens_saved_never_underflows(
            retrieval in 0usize..10_000,
            generation in 0usize..10_000,
            output in 0usize..5_000,
        ) {
            let rec = UsageRecord::new(
                "t", "q", "m", retrieval, generation, output, 3, 200, 0.03
            );
            // saturating_sub must prevent underflow
            assert!(rec.tokens_saved <= rec.naive_rag_token_estimate);
        }
    }
}
