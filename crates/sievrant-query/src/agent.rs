use std::collections::HashSet;

use async_trait::async_trait;
use serde::Serialize;
use tracing::{debug, info, warn};

use sievrant_core::error::Result;
use sievrant_core::types::EvidenceChunk;

/// Accumulated state for a single multi-hop retrieval session.
#[derive(Debug, Clone, Serialize)]
pub struct RetrievalState {
    pub original_query: String,
    pub sub_queries: Vec<String>,
    pub accumulated_evidence: Vec<EvidenceChunk>,
    pub iteration_count: usize,

    pub max_iterations: usize,
    /// Whether the LLM declared the accumulated evidence sufficient.
    pub satisfied: bool,
}

impl RetrievalState {
    pub fn new(query: impl Into<String>, max_iterations: usize) -> Self {
        Self {
            original_query: query.into(),
            sub_queries: Vec::new(),
            accumulated_evidence: Vec::new(),
            iteration_count: 0,
            max_iterations,
            satisfied: false,
        }
    }

    /// Add unique chunks from a single sub-query pass, deduplicating by chunk_id.
    pub fn absorb_evidence(&mut self, new_chunks: Vec<EvidenceChunk>) {
        let seen: HashSet<String> = self
            .accumulated_evidence
            .iter()
            .map(|c| c.chunk_id.clone())
            .collect();

        for chunk in new_chunks {
            if !seen.contains(&chunk.chunk_id) {
                self.accumulated_evidence.push(chunk);
            }
        }
    }
}

/// Contract for a multi-hop agentic retriever.
///
/// Implementations drive a loop that decomposes complex queries into
/// simpler sub-queries, executes them in parallel, and accumulates evidence
/// until the LLM declares the original question answerable.
#[async_trait]
pub trait AgenticRetriever: Send + Sync {
    /// Execute the full multi-hop loop and return the final state.
    async fn execute_multi_hop(
        &self,
        query: &str,
        max_hops: usize,
    ) -> Result<RetrievalState>;
}

/// Default agentic retriever.
///
/// On each iteration it calls `sub_query::decompose_query` to ask the LLM for
/// the next set of sub-queries, then fans out via `tokio::task::JoinSet`
/// to run all sub-queries concurrently. Iteration stops when the LLM signals
/// sufficiency or `max_hops` is exhausted.
pub struct DefaultAgenticRetriever<R, L>
where
    R: Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<EvidenceChunk>>> + Send>>
        + Send
        + Sync
        + 'static,
    L: Fn(String, String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<String>>> + Send>>
        + Send
        + Sync
        + 'static,
{
    /// Executes a single sub-query and returns the retrieved chunks.
    pub retrieval_fn: R,
    /// Asks the LLM to decompose remaining uncertainty into sub-queries.
    /// Returns an empty Vec when the LLM considers the evidence sufficient.
    pub decompose_fn: L,
}

#[async_trait]
impl<R, L> AgenticRetriever for DefaultAgenticRetriever<R, L>
where
    R: Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<EvidenceChunk>>> + Send>>
        + Send
        + Sync
        + 'static,
    L: Fn(String, String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<String>>> + Send>>
        + Send
        + Sync
        + 'static,
{
    async fn execute_multi_hop(
        &self,
        query: &str,
        max_hops: usize,
    ) -> Result<RetrievalState> {
        let mut state = RetrievalState::new(query, max_hops);

        // First pass: treat the original query as the initial sub-query.
        let initial_chunks = (self.retrieval_fn)(query.to_string()).await?;
        state.absorb_evidence(initial_chunks);
        state.iteration_count += 1;

        while state.iteration_count < max_hops && !state.satisfied {
            let evidence_summary = state
                .accumulated_evidence
                .iter()
                .map(|c| c.chunk_id.as_str())
                .collect::<Vec<_>>()
                .join(", ");

            let sub_queries = (self.decompose_fn)(
                state.original_query.clone(),
                evidence_summary,
            )
            .await?;

            if sub_queries.is_empty() {
                info!(iteration = state.iteration_count, "LLM signalled evidence sufficient");
                state.satisfied = true;
                break;
            }

            state.sub_queries.extend(sub_queries.clone());
            debug!(count = sub_queries.len(), "running parallel sub-queries");

            // Run all sub-queries for this hop concurrently.
            let mut join_set = tokio::task::JoinSet::new();
            for sq in sub_queries {
                let chunk = (self.retrieval_fn)(sq);
                join_set.spawn(chunk);
            }

            while let Some(result) = join_set.join_next().await {
                match result {
                    Ok(Ok(chunks)) => state.absorb_evidence(chunks),
                    Ok(Err(e)) => {
                        warn!(error = %e, "sub-query failed; continuing with partial evidence")
                    }
                    Err(join_err) => {
                        warn!(error = %join_err, "sub-query task panicked; skipping")
                    }
                }
            }

            state.iteration_count += 1;
        }

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(id: &str) -> EvidenceChunk {
        EvidenceChunk {
            chunk_id: id.to_string(),
            source_doc: "test.txt".into(),
            doc_version_hash: "abc".into(),
            path: vec![],
            page: None,
            char_span: None,
            text: "text".into(),
            retriever: sievrant_core::types::RetrieverType::VectorIndex,
            rrf_score: 0.1,
            rerank_score: None,
            nonconformity_score: 0.1,
            submodular_marginal_gain: 0.1,
            ingested_at: chrono::Utc::now(),
            index_generation: 0,
            acl_groups: vec![],
            freshness_hours: 0.0,
            token_count: 10,
        }
    }

    #[tokio::test]
    async fn retrieves_on_first_pass() {
        let retriever = DefaultAgenticRetriever {
            retrieval_fn: |_q| Box::pin(async { Ok(vec![make_chunk("c1"), make_chunk("c2")]) }),
            // LLM immediately signals satisfied
            decompose_fn: |_q, _e| Box::pin(async { Ok(vec![]) }),
        };

        let state = retriever.execute_multi_hop("what is X?", 5).await.unwrap();
        assert_eq!(state.accumulated_evidence.len(), 2);
        assert!(state.satisfied);
    }

    #[tokio::test]
    async fn deduplicates_across_hops() {
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let retriever = DefaultAgenticRetriever {
            retrieval_fn: |_q| Box::pin(async {
                Ok(vec![make_chunk("c1")]) // always returns same chunk
            }),
            decompose_fn: move |_q, _e| {
                let n = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Box::pin(async move {
                    if n == 0 {
                        Ok(vec!["sub-query-1".into()])
                    } else {
                        Ok(vec![]) // signal done on second decompose
                    }
                })
            },
        };

        let state = retriever.execute_multi_hop("multi hop Q", 10).await.unwrap();
        // Deduplication must prevent duplicates even when multiple hops return the same chunk.
        assert_eq!(state.accumulated_evidence.len(), 1, "duplicate chunks must be deduplicated");
    }
}
