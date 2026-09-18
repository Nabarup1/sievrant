use async_trait::async_trait;
use serde_json::json;
use tracing::debug;

use sievrant_core::error::{LlmError, Result, SievrantError};

use crate::client::ResilientHttpClient;

/// Contract for producing dense vector embeddings from text.
///
/// Implementations exist for OpenAI, local ONNX models, Jina, and Cohere.
/// All implementations handle internal batching, rate limits, and retries.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a single text string. Implementations should reuse the batch
    /// infrastructure under the hood to avoid duplicating retry logic.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed many strings at once. The provider handles chunking into
    /// acceptable API payload sizes internally. Results are returned in
    /// the same order as the input slice.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// The dimensionality of the produced embedding vectors.
    fn dimensions(&self) -> usize;

    /// Human-readable name used in observability logs.
    fn provider_name(&self) -> &str;
}

/// OpenAI text-embedding-3-small / text-embedding-3-large via REST API.
///
/// Batches up to 2048 inputs per request (OpenAI's documented limit).
/// Retries 429 and 5xx responses with exponential backoff via the shared client.
pub struct OpenAiEmbeddingProvider {
    client: ResilientHttpClient,
    api_key: String,
    model: String,
    base_url: String,
    /// Cached dimension count for the configured model.
    dims: usize,
}

impl OpenAiEmbeddingProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::custom("https://api.openai.com/v1", api_key, model)
    }

    pub fn custom(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let base_url = base_url.into();
        let model = model.into();
        let api_key = api_key.into();
        let dims = match model.as_str() {
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            "nomic-embed-text" => 768,
            _ => 1536, // text-embedding-3-small default
        };
        Self {
            client: ResilientHttpClient::new(5, 300),
            api_key,
            model,
            base_url,
            dims,
        }
    }

    /// Issue one batch request for up to 2048 texts and parse the result.
    async fn request_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let body = json!({
            "input": texts,
            "model": self.model,
            "encoding_format": "float"
        });

        let endpoint = format!("{}/embeddings", self.base_url);
        let resp = self
            .client
            .post_json(&endpoint, &body, &self.api_key)
            .await?;

        let data = resp["data"].as_array().ok_or_else(|| {
            SievrantError::Llm(LlmError::ParseError {
                provider: "openai-embedding".into(),
                reason: "missing 'data' field in embeddings response".into(),
            })
        })?;

        data.iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .ok_or_else(|| {
                        SievrantError::Llm(LlmError::ParseError {
                            provider: "openai-embedding".into(),
                            reason: "missing 'embedding' field in data item".into(),
                        })
                    })
                    .and_then(|arr| {
                        arr.iter()
                            .map(|v| {
                                v.as_f64().map(|f| f as f32).ok_or_else(|| {
                                    SievrantError::Llm(LlmError::ParseError {
                                        provider: "openai-embedding".into(),
                                        reason: "non-numeric embedding value".into(),
                                    })
                                })
                            })
                            .collect::<Result<Vec<f32>>>()
                    })
            })
            .collect()
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text.to_string()]).await?;
        batch.into_iter().next().ok_or_else(|| {
            SievrantError::Llm(LlmError::ParseError {
                provider: "openai-embedding".into(),
                reason: "empty embedding response".into(),
            })
        })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        const BATCH_SIZE: usize = 2048;
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(BATCH_SIZE) {
            debug!(
                batch_size = chunk.len(),
                model = %self.model,
                "embedding batch"
            );
            let mut batch_result = self.request_batch(&chunk.to_vec()).await?;
            results.append(&mut batch_result);
        }

        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn provider_name(&self) -> &str {
        "openai-embedding"
    }
}

/// A provider that tries a list of `EmbeddingProvider`s in order.
/// If the primary fails for any reason, it falls back to the next one.
/// This is the main resilience mechanism for air-gapped or rate-limited setups.
pub struct FallbackEmbeddingProvider {
    providers: Vec<Box<dyn EmbeddingProvider>>,
}

impl FallbackEmbeddingProvider {
    pub fn new(providers: Vec<Box<dyn EmbeddingProvider>>) -> Self {
        assert!(
            !providers.is_empty(),
            "FallbackEmbeddingProvider needs at least one provider"
        );
        Self { providers }
    }
}

#[async_trait]
impl EmbeddingProvider for FallbackEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut last_err = None;
        for p in &self.providers {
            match p.embed(text).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    tracing::warn!(
                        provider = p.provider_name(),
                        "embedding provider failed, trying next"
                    );
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap())
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut last_err = None;
        for p in &self.providers {
            match p.embed_batch(texts).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    tracing::warn!(
                        provider = p.provider_name(),
                        "embedding batch provider failed, trying next"
                    );
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap())
    }

    fn dimensions(&self) -> usize {
        // Fallback providers should all share the same dimensionality.
        self.providers[0].dimensions()
    }

    fn provider_name(&self) -> &str {
        "fallback-embedding"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_provider_dimensions() {
        let p = OpenAiEmbeddingProvider::new("key", "text-embedding-3-large");
        assert_eq!(p.dimensions(), 3072);
    }

    #[test]
    fn openai_provider_small_default_dimensions() {
        let p = OpenAiEmbeddingProvider::new("key", "text-embedding-3-small");
        assert_eq!(p.dimensions(), 1536);
    }
}
