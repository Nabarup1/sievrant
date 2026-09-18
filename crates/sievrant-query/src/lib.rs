pub mod agent;
pub mod cache;
pub mod processing;
pub mod router;

use async_trait::async_trait;
use sievrant_core::error::Result;

/// Embedding provider contract used by QueryProcessor and VectorIndex.
/// Defined here to avoid a circular dependency between sievrant-query and sievrant-index.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of texts and return one vector per input.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}
