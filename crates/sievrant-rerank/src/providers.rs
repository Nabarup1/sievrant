//! Production reranker backends.
//!
//! | provider    | backend                                                  |
//! |-------------|----------------------------------------------------------|
//! | `cohere`    | Cohere `/v2/rerank`                                      |
//! | `jina`      | Jina `/v1/rerank`                                        |
//! | `voyage`    | Voyage `/v1/rerank`                                      |
//! | `custom`    | any Cohere-compatible endpoint (TEI, Infinity, vLLM)     |
//! | `embedding` | cosine similarity from the configured embedding model    |
//! | `llm`       | graded relevance from the configured (fast) LLM          |
//! | `none`      | no reranking; fusion scores are used                     |

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use sievrant_core::error::{RerankError, Result};
use sievrant_llm::{EmbeddingProvider, LlmProvider, LlmRequest, Message, Role};

use crate::Reranker;

/// Response shape shared by Cohere, Jina and Voyage (`results` or `data`).
#[derive(Deserialize)]
struct RerankResponse {
    #[serde(alias = "data")]
    results: Vec<RerankResult>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f64,
}

/// HTTP reranker for Cohere-style APIs.
pub struct HttpReranker {
    client: reqwest::Client,
    name: String,
    url: String,
    api_key: String,
    model: String,
    /// Voyage names the result count `top_k`; the others use `top_n`.
    top_field: &'static str,
    max_retries: u32,
}

impl HttpReranker {
    pub fn new(name: &str, url: &str, api_key: &str, model: &str) -> Self {
        let top_field = if name == "voyage" { "top_k" } else { "top_n" };
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client"),
            name: name.to_string(),
            url: url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            top_field,
            max_retries: 3,
        }
    }

    pub fn cohere(api_key: &str, model: &str) -> Self {
        Self::new("cohere", "https://api.cohere.com/v2/rerank", api_key, model)
    }

    pub fn jina(api_key: &str, model: &str) -> Self {
        Self::new("jina", "https://api.jina.ai/v1/rerank", api_key, model)
    }

    pub fn voyage(api_key: &str, model: &str) -> Self {
        Self::new(
            "voyage",
            "https://api.voyageai.com/v1/rerank",
            api_key,
            model,
        )
    }
}

#[async_trait]
impl Reranker for HttpReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }
        let n = documents.len();
        let mut body = json!({
            "model": self.model,
            "query": query,
            "documents": documents,
        });
        body[self.top_field] = json!(n);

        let mut attempt = 0u32;
        let response = loop {
            let mut rb = self.client.post(&self.url).json(&body);
            if !self.api_key.is_empty() {
                rb = rb.bearer_auth(&self.api_key);
            }
            match rb.send().await {
                Ok(resp) if resp.status().is_success() => break resp,
                Ok(resp) => {
                    let status = resp.status();
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if !retryable || attempt >= self.max_retries {
                        let text = resp.text().await.unwrap_or_default();
                        return Err(RerankError::Api(format!(
                            "{}: HTTP {status}: {}",
                            self.name,
                            text.chars().take(500).collect::<String>()
                        ))
                        .into());
                    }
                }
                Err(e) => {
                    if attempt >= self.max_retries || !(e.is_timeout() || e.is_connect()) {
                        return Err(RerankError::Api(format!(
                            "{}: {}",
                            self.name,
                            e.without_url()
                        ))
                        .into());
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(400u64 << attempt)).await;
            attempt += 1;
        };

        let parsed: RerankResponse = response
            .json()
            .await
            .map_err(|e| RerankError::Api(format!("{}: {e}", self.name)))?;

        let mut scores = vec![f64::NAN; n];
        for r in parsed.results {
            if r.index < n {
                scores[r.index] = r.relevance_score;
            }
        }
        if scores.iter().any(|s| s.is_nan()) {
            return Err(RerankError::Api(format!(
                "{}: response did not score every document",
                self.name
            ))
            .into());
        }
        Ok(scores)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Scores candidates by cosine similarity between query and passage
/// embeddings, mapped to [0, 1]. Gives BM25-only hits a semantic score on the
/// same scale as vector hits. Passage embeddings are cached by text.
pub struct EmbeddingReranker {
    embedder: Arc<dyn EmbeddingProvider>,
    cache: Mutex<HashMap<u64, Arc<Vec<f32>>>>,
    max_cache: usize,
}

impl EmbeddingReranker {
    pub fn new(embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            embedder,
            cache: Mutex::new(HashMap::new()),
            max_cache: 50_000,
        }
    }

    fn key(text: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in text.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na.sqrt() * nb.sqrt())
        }
    }
}

#[async_trait]
impl Reranker for EmbeddingReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        let q = self.embedder.embed(query).await?;

        let mut vectors: Vec<Option<Arc<Vec<f32>>>> = {
            let cache = self.cache.lock().unwrap();
            documents
                .iter()
                .map(|d| cache.get(&Self::key(d)).cloned())
                .collect()
        };
        let missing: Vec<usize> = (0..documents.len())
            .filter(|i| vectors[*i].is_none())
            .collect();
        if !missing.is_empty() {
            let texts: Vec<String> = missing.iter().map(|i| documents[*i].to_string()).collect();
            let fresh = self.embedder.embed_batch(&texts).await?;
            let mut cache = self.cache.lock().unwrap();
            if cache.len() + fresh.len() > self.max_cache {
                cache.clear();
            }
            for (i, v) in missing.into_iter().zip(fresh) {
                let v = Arc::new(v);
                cache.insert(Self::key(documents[i]), v.clone());
                vectors[i] = Some(v);
            }
        }

        Ok(vectors
            .into_iter()
            .map(|v| (1.0 + Self::cosine(&q, v.as_deref().unwrap_or(&vec![]))) / 2.0)
            .collect())
    }

    fn name(&self) -> &str {
        "embedding-cosine"
    }
}

/// Asks an LLM to grade each passage 0-10 for relevance, in batches.
/// Slower and costlier than a dedicated reranker, but works with any chat
/// provider. Scores are grade / 10.
pub struct LlmReranker {
    llm: Arc<dyn LlmProvider>,
    batch_size: usize,
    max_passage_chars: usize,
}

impl LlmReranker {
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            llm,
            batch_size: 20,
            max_passage_chars: 1200,
        }
    }

    fn parse_grades(text: &str, expected: usize) -> Option<Vec<f64>> {
        let start = text.find('[')?;
        let end = text[start..].find(']')? + start;
        let grades: Vec<f64> = text[start + 1..end]
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        (grades.len() == expected).then(|| {
            grades
                .into_iter()
                .map(|g| (g.clamp(0.0, 10.0)) / 10.0)
                .collect()
        })
    }
}

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        let mut scores = Vec::with_capacity(documents.len());
        for batch in documents.chunks(self.batch_size) {
            let passages = batch
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let clipped: String = d.chars().take(self.max_passage_chars).collect();
                    format!("[{i}] {clipped}")
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            let prompt = format!(
                "Grade how well each passage answers the query, from 0 (irrelevant) \
                 to 10 (fully answers it). Reply with only a JSON array of {n} integers \
                 in passage order.\n\nQuery: {query}\n\nPassages:\n{passages}",
                n = batch.len()
            );
            let resp = self
                .llm
                .complete(LlmRequest {
                    messages: vec![Message {
                        role: Role::User,
                        content: prompt,
                    }],
                    max_tokens: 2048,
                    temperature: 0.0,
                    system_prompt: None,
                })
                .await?;
            let grades = Self::parse_grades(&resp.content, batch.len())
                .ok_or_else(|| RerankError::Api("llm reranker: could not parse grades".into()))?;
            scores.extend(grades);
        }
        Ok(scores)
    }

    fn name(&self) -> &str {
        "llm-graded"
    }
}

/// Passes fusion scores through unchanged (signalled by returning an error
/// that `rerank_candidates` treats as "no reranker configured").
pub struct NoReranker;

#[async_trait]
impl Reranker for NoReranker {
    async fn rerank(&self, _query: &str, _documents: &[&str]) -> Result<Vec<f64>> {
        Err(RerankError::Unavailable("reranking disabled".into()).into())
    }

    fn name(&self) -> &str {
        "none"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_llm::HashingEmbeddingProvider;

    #[test]
    fn rerank_response_accepts_results_or_data() {
        let cohere: RerankResponse =
            serde_json::from_str(r#"{"results":[{"index":1,"relevance_score":0.9}]}"#).unwrap();
        let voyage: RerankResponse =
            serde_json::from_str(r#"{"data":[{"index":0,"relevance_score":0.2}]}"#).unwrap();
        assert_eq!(cohere.results[0].index, 1);
        assert_eq!(voyage.results[0].relevance_score, 0.2);
    }

    #[test]
    fn voyage_uses_top_k() {
        assert_eq!(HttpReranker::voyage("k", "m").top_field, "top_k");
        assert_eq!(HttpReranker::cohere("k", "m").top_field, "top_n");
    }

    #[test]
    fn llm_grades_parse_and_normalise() {
        assert_eq!(
            LlmReranker::parse_grades("Here: [10, 0, 5]", 3),
            Some(vec![1.0, 0.0, 0.5])
        );
        assert_eq!(LlmReranker::parse_grades("[1, 2]", 3), None);
        assert_eq!(LlmReranker::parse_grades("no array", 1), None);
    }

    #[tokio::test]
    async fn embedding_reranker_prefers_overlapping_passage() {
        let r = EmbeddingReranker::new(Arc::new(HashingEmbeddingProvider::new(1024)));
        let s = r
            .rerank(
                "lease term months",
                &["the lease term is sixty months", "revenue grew in europe"],
            )
            .await
            .unwrap();
        assert!(s[0] > s[1]);
        assert!(s.iter().all(|x| (0.0..=1.0).contains(x)));
        // Second call is served from cache and is identical.
        let again = r
            .rerank(
                "lease term months",
                &["the lease term is sixty months", "revenue grew in europe"],
            )
            .await
            .unwrap();
        assert_eq!(s, again);
    }
}
