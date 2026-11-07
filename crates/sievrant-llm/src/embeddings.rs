use std::sync::atomic::{AtomicUsize, Ordering};

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

    /// Model identifier; part of the pipeline fingerprint.
    fn model_id(&self) -> String {
        self.provider_name().to_string()
    }
}

/// Client for any OpenAI-compatible `/embeddings` endpoint: OpenAI, Gemini
/// (OpenAI-compatible surface), Mistral, Voyage, Jina, Together, Fireworks,
/// Ollama, LM Studio, vLLM, TEI and similar servers.
///
/// Retries 429 and 5xx responses with exponential backoff via the shared client.
/// The vector dimension is taken from configuration when set, otherwise it is
/// learned from the first response.
pub struct OpenAiEmbeddingProvider {
    client: ResilientHttpClient,
    provider: String,
    api_key: String,
    model: String,
    base_url: String,
    batch_size: usize,
    float_encoding: bool,
    /// Configured or learned dimension count; 0 means not yet known.
    dims: AtomicUsize,
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
            "text-embedding-3-small" | "text-embedding-ada-002" => 1536,
            "nomic-embed-text" => 768,
            _ => 0,
        };
        Self {
            client: ResilientHttpClient::new(5, 300),
            provider: "openai-embedding".into(),
            api_key,
            model,
            base_url: base_url.trim_end_matches('/').to_string(),
            batch_size: 128,
            float_encoding: true,
            dims: AtomicUsize::new(dims),
        }
    }

    pub fn named(mut self, provider: &str) -> Self {
        self.provider = provider.to_string();
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub fn with_dimensions(self, dims: Option<usize>) -> Self {
        if let Some(d) = dims.filter(|d| *d > 0) {
            self.dims.store(d, Ordering::Relaxed);
        }
        self
    }

    /// Only OpenAI documents `encoding_format: "float"`; some compatible
    /// servers reject unknown fields, so it is opt-in.
    pub fn with_float_encoding(mut self, enabled: bool) -> Self {
        self.float_encoding = enabled;
        self
    }

    /// Issue one batch request for up to 2048 texts and parse the result.
    async fn request_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut body = json!({
            "input": texts,
            "model": self.model,
        });
        if self.float_encoding {
            body["encoding_format"] = json!("float");
        }

        let endpoint = format!("{}/embeddings", self.base_url);
        let resp = self
            .client
            .post_json(&endpoint, &body, &self.api_key)
            .await?;

        let data = resp["data"].as_array().ok_or_else(|| {
            SievrantError::Llm(LlmError::ParseError {
                provider: self.provider.clone(),
                reason: "missing 'data' field in embeddings response".into(),
            })
        })?;

        data.iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .ok_or_else(|| {
                        SievrantError::Llm(LlmError::ParseError {
                            provider: self.provider.clone(),
                            reason: "missing 'embedding' field in data item".into(),
                        })
                    })
                    .and_then(|arr| {
                        arr.iter()
                            .map(|v| {
                                v.as_f64().map(|f| f as f32).ok_or_else(|| {
                                    SievrantError::Llm(LlmError::ParseError {
                                        provider: self.provider.clone(),
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
                provider: self.provider.clone(),
                reason: "empty embedding response".into(),
            })
        })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            debug!(
                batch_size = chunk.len(),
                model = %self.model,
                "embedding batch"
            );
            let mut batch_result = self.request_batch(chunk).await?;
            if batch_result.len() != chunk.len() {
                return Err(SievrantError::Llm(LlmError::ParseError {
                    provider: self.provider.clone(),
                    reason: format!(
                        "expected {} embeddings, got {}",
                        chunk.len(),
                        batch_result.len()
                    ),
                }));
            }
            results.append(&mut batch_result);
        }

        if let Some(first) = results.first() {
            self.dims.store(first.len(), Ordering::Relaxed);
        }
        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dims.load(Ordering::Relaxed)
    }

    fn provider_name(&self) -> &str {
        &self.provider
    }

    fn model_id(&self) -> String {
        format!("{}:{}", self.provider, self.model)
    }
}

/// Deterministic lexical embedder based on feature hashing.
///
/// Word unigrams and bigrams are hashed into a fixed number of buckets with
/// sublinear term frequency and L2 normalisation. It needs no network, key or
/// model download, which makes it useful for offline development, CI and
/// reproducible benchmarks. It captures word overlap only, so production
/// deployments should use a real embedding model.
pub struct HashingEmbeddingProvider {
    dims: usize,
}

impl HashingEmbeddingProvider {
    pub fn new(dims: usize) -> Self {
        Self { dims: dims.max(16) }
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }

    fn embed_one(&self, text: &str) -> Vec<f32> {
        let lower = text.to_lowercase();
        let tokens: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 1)
            .collect();

        let mut counts: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
        for t in &tokens {
            *counts.entry(Self::fnv1a(t.as_bytes())).or_insert(0.0) += 1.0;
        }
        for pair in tokens.windows(2) {
            let bigram = format!("{} {}", pair[0], pair[1]);
            *counts.entry(Self::fnv1a(bigram.as_bytes())).or_insert(0.0) += 0.5;
        }

        let mut v = vec![0.0f32; self.dims];
        for (h, tf) in counts {
            let idx = (h % self.dims as u64) as usize;
            let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            v[idx] += sign * (1.0 + tf.ln());
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }
}

#[async_trait]
impl EmbeddingProvider for HashingEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed_one(text))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn provider_name(&self) -> &str {
        "local-hash"
    }

    fn model_id(&self) -> String {
        format!("local-hash:{}", self.dims)
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

    #[tokio::test]
    async fn hashing_embedder_is_deterministic_and_normalised() {
        let e = HashingEmbeddingProvider::new(256);
        let a = e.embed("The lease term is 60 months").await.unwrap();
        let b = e.embed("The lease term is 60 months").await.unwrap();
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn hashing_embedder_ranks_overlapping_text_higher() {
        let e = HashingEmbeddingProvider::new(1024);
        let q = e.embed("lease term months").await.unwrap();
        let rel = e.embed("the lease term is sixty months").await.unwrap();
        let other = e.embed("quarterly revenue grew in europe").await.unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(dot(&q, &rel) > dot(&q, &other));
    }

    #[test]
    fn unknown_model_dimension_is_learned_later() {
        let p = OpenAiEmbeddingProvider::custom("http://x", "k", "some-model");
        assert_eq!(p.dimensions(), 0);
        let p = p.with_dimensions(Some(384));
        assert_eq!(p.dimensions(), 384);
    }

    #[test]
    fn openai_provider_small_default_dimensions() {
        let p = OpenAiEmbeddingProvider::new("key", "text-embedding-3-small");
        assert_eq!(p.dimensions(), 1536);
    }
}
