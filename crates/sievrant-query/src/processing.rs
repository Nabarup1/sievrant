use std::sync::Arc;

use sievrant_core::{
    error::{QueryError, Result},
    types::Route,
};
use sievrant_llm::{
    provider::{LlmRequest, Message, Role},
    LlmProvider,
};

use crate::EmbeddingProvider;

/// Configuration for query processing behaviour.
#[derive(Debug, Clone)]
pub struct QueryProcessingConfig {
    /// Maximum tokens for expansion/decomposition LLM calls.
    pub max_expansion_tokens: usize,
    /// Maximum number of sub-queries on the Slow path.
    pub max_sub_queries: usize,
    /// Enables HyDE (Hypothetical Document Embeddings). Increases latency.
    pub hyde_enabled: bool,
}

impl Default for QueryProcessingConfig {
    fn default() -> Self {
        Self {
            max_expansion_tokens: 256,
            max_sub_queries: 5,
            hyde_enabled: false,
        }
    }
}

/// Enriched query ready for multi-index retrieval.
pub struct ProcessedQuery {
    pub original: String,
    /// LLM-expanded version of the query (Standard/Slow paths only).
    pub expanded: Option<String>,
    /// Sub-queries from multi-hop decomposition (Slow path only).
    pub sub_queries: Option<Vec<String>>,
    /// Dense embedding of the (possibly HyDE-enhanced) query.
    pub embedding: Vec<f32>,
    /// Extracted named entities for the graph index PPR seed.
    pub entities: Vec<String>,
}

pub struct QueryProcessor {
    pub llm: Arc<dyn LlmProvider>,
    pub embedding_provider: Arc<dyn EmbeddingProvider>,
    pub config: QueryProcessingConfig,
}

impl QueryProcessor {
    pub fn new(
        llm: Arc<dyn LlmProvider>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        config: QueryProcessingConfig,
    ) -> Self {
        Self {
            llm,
            embedding_provider,
            config,
        }
    }

    pub async fn process(&self, query: &str, route: &Route) -> Result<ProcessedQuery> {
        if query.trim().is_empty() {
            return Err(QueryError::EmptyQuery.into());
        }

        let entities = extract_entities(query);

        match route {
            Route::Fast | Route::Cached => {
                let embeddings = self.embedding_provider.embed(&[query.to_string()]).await?;
                Ok(ProcessedQuery {
                    original: query.to_string(),
                    expanded: None,
                    sub_queries: None,
                    embedding: embeddings.into_iter().next().unwrap_or_default(),
                    entities,
                })
            }

            Route::Standard => {
                let expanded = self.expand_query(query).await?;
                let embed_text = if self.config.hyde_enabled {
                    self.generate_hyde(query).await.ok()
                } else {
                    None
                };

                let embedding = if let Some(hyde_vec) = embed_text {
                    hyde_vec
                } else {
                    let embeddings = self.embedding_provider.embed(std::slice::from_ref(&expanded)).await?;
                    embeddings.into_iter().next().unwrap_or_default()
                };

                Ok(ProcessedQuery {
                    original: query.to_string(),
                    expanded: Some(expanded),
                    sub_queries: None,
                    embedding,
                    entities,
                })
            }

            Route::Slow => {
                let expanded = self.expand_query(query).await?;
                let sub_queries = self.decompose_query(query).await?;

                let embeddings = self.embedding_provider.embed(std::slice::from_ref(&expanded)).await?;
                let embedding = embeddings.into_iter().next().unwrap_or_default();

                Ok(ProcessedQuery {
                    original: query.to_string(),
                    expanded: Some(expanded),
                    sub_queries: Some(sub_queries),
                    embedding,
                    entities,
                })
            }
        }
    }

    async fn expand_query(&self, query: &str) -> Result<String> {
        let prompt = format!(
            "Given the following query, generate 2-3 alternative phrasings and relevant synonyms.\n\
             Return only the expanded query text, combining the original with alternatives.\n\
             Do not add any explanation.\n\n\
             Query: {query}"
        );

        let req = LlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: prompt,
            }],
            max_tokens: self.config.max_expansion_tokens,
            temperature: 0.0,
            system_prompt: None,
        };

        let response = self
            .llm
            .complete(req)
            .await
            .map_err(|e| QueryError::Expansion(e.to_string()))?;

        Ok(response.content.trim().to_string())
    }

    /// HyDE: generate a hypothetical document and embed it.
    /// The resulting embedding is typically a better retrieval signal than
    /// embedding the short query directly.
    async fn generate_hyde(&self, query: &str) -> Result<Vec<f32>> {
        let prompt = format!(
            "Write a short paragraph that would directly answer the following question.\n\
             Do not say you are answering a question. Just write the answer.\n\n\
             Question: {query}"
        );

        let req = LlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: prompt,
            }],
            max_tokens: 256,
            temperature: 0.3,
            system_prompt: None,
        };

        let response = self
            .llm
            .complete(req)
            .await
            .map_err(|e| QueryError::Embedding(e.to_string()))?;

        let embeddings = self.embedding_provider.embed(&[response.content]).await?;

        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    async fn decompose_query(&self, query: &str) -> Result<Vec<String>> {
        let prompt = format!(
            "Break the following complex question into independent sub-questions\n\
             that can each be answered by searching a document corpus.\n\
             Return each sub-question on a separate line.\n\
             Do not add numbering or explanation.\n\n\
             Question: {query}"
        );

        let req = LlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: prompt,
            }],
            max_tokens: self.config.max_expansion_tokens,
            temperature: 0.0,
            system_prompt: None,
        };

        let response = self
            .llm
            .complete(req)
            .await
            .map_err(|e| QueryError::Decomposition(e.to_string()))?;

        let sub_queries: Vec<String> = response
            .content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .take(self.config.max_sub_queries)
            .collect();

        Ok(sub_queries)
    }
}

/// Lightweight entity extractor reused from the graph index.
/// Extracts Section/Article/Clause references, quoted terms, and capitalized phrases.
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let words: Vec<&str> = text.split_whitespace().collect();
    for window in words.windows(2) {
        if [
            "Section", "Article", "Clause", "Exhibit", "Schedule", "Appendix",
        ]
        .contains(&window[0])
        {
            let entity = format!("{} {}", window[0], window[1]);
            if seen.insert(entity.clone()) {
                entities.push(entity);
            }
        }
    }

    let mut remaining = text;
    while let Some(start) = remaining.find('"') {
        let rest = &remaining[start + 1..];
        if let Some(end) = rest.find('"') {
            let term = rest[..end].trim().to_string();
            if !term.is_empty() && seen.insert(term.clone()) {
                entities.push(term);
            }
            remaining = &rest[end + 1..];
        } else {
            break;
        }
    }

    entities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_entities_finds_section_refs() {
        let entities = extract_entities("Pursuant to Section 4.2 and Article 7 of the agreement");
        assert!(entities.contains(&"Section 4.2".into()));
        assert!(entities.contains(&"Article 7".into()));
    }

    #[test]
    fn extract_entities_finds_quoted_terms() {
        let entities = extract_entities(r#"What is the "Effective Date" and "Lease Term"?"#);
        assert!(entities.contains(&"Effective Date".into()));
        assert!(entities.contains(&"Lease Term".into()));
    }
}
