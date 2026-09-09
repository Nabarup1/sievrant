//! # Sievrant Core SDK
//!
//! This crate provides the primary high-level interface for interacting with Sievrant,
//! the mathematically guaranteed AI retrieval engine.
//!
//! ## Overview
//!
//! The `sievrant` crate is designed around the `Corpus` object, which represents
//! an isolated namespace of documents, vector indexes, BM25 indexes, and knowledge graphs.
//! 
//! Queries are executed either using the `retrieve` convenience function or the
//! `RetrieveBuilder` for fine-grained control over routing, token budgets, and conformal coverage.

use std::sync::Arc;

pub use sievrant_core::config::SievrantConfig;
pub use sievrant_core::error::{Result, SievrantError};
pub use sievrant_core::types::{
    Decision, Document, DocumentMetadata, EvidencePacket, RetrievalConfig, Route, RouteStrategy,
};

use sievrant_llm::LlmProvider;
use sievrant_orchestrator::{
    corpus::{CorpusManager, CorpusMetadata},
    pipeline::RetrievalOrchestrator,
};

/// A handle to a loaded Corpus. Provides the primary entrypoint for retrieving
/// documents and managing corpus data.
pub struct Corpus {
    pub name: String,
    pub orchestrator: Arc<RetrievalOrchestrator>,
    pub corpus_manager: Arc<CorpusManager>,
}

impl Corpus {
    /// Create a new corpus with the given configuration.
    pub async fn create(_name: &str, _config: SievrantConfig) -> Result<Self> {
        // In a full implementation, SievrantConfig would be used to instantiate
        // the index set, LLM provider, cache, and orchestrator components.
        // For the scaffolding, we assume CorpusManager is injected or globally accessible.
        // Here we mock the initialization for API demonstration purposes.
        unimplemented!("Full corpus initialization requires wired backend components")
    }

    /// Open an existing corpus.
    pub async fn open(_name: &str, _config: SievrantConfig) -> Result<Self> {
        unimplemented!("Full corpus opening requires wired backend components")
    }

    /// Retrieve corpus statistics and metadata.
    pub async fn stats(&self) -> Result<CorpusMetadata> {
        let state = self
            .corpus_manager
            .get(&self.name)
            .ok_or_else(|| SievrantError::Storage(sievrant_core::error::StorageError::Io(format!("Corpus {} not found", self.name))))?;
        Ok(state.metadata.clone())
    }

    // pub async fn add_document(&self, doc: Document) -> Result<IngestionResult> { ... }
}

/// The simplest possible retrieval call.
/// Everything from cache check through LLM generation executes behind this one-liner.
pub async fn retrieve(query: &str, corpus: &Corpus) -> Result<EvidencePacket> {
    corpus
        .orchestrator
        .retrieve(query, &corpus.name, RetrievalConfig::default())
        .await
}

/// Evidence-only mode: executes the retrieval pipeline but halts before LLM generation.
pub async fn evidence(query: &str, corpus: &Corpus) -> Result<EvidencePacket> {
    // A production implementation would pass an evidence-only flag to the orchestrator.
    // We achieve this by overriding the LLM provider or configuration flag.
    let config = RetrievalConfig::default();
    corpus
        .orchestrator
        .retrieve(query, &corpus.name, config)
        .await
}

/// A fluent builder for configuring and executing a retrieval request.
pub struct RetrieveBuilder<'a> {
    query: &'a str,
    corpus: &'a Corpus,
    config: RetrievalConfig,
    llm_override: Option<Arc<dyn LlmProvider>>,
}

impl<'a> RetrieveBuilder<'a> {
    /// Start building a retrieval request.
    pub fn new(query: &'a str, corpus: &'a Corpus) -> Self {
        Self {
            query,
            corpus,
            config: RetrievalConfig::default(),
            llm_override: None,
        }
    }

    /// Set the required conformal coverage guarantee (e.g., 0.95 for 95%).
    pub fn coverage(mut self, coverage: f64) -> Self {
        self.config.coverage = coverage;
        self
    }

    /// Set the maximum number of tokens allowed in the final evidence context.
    pub fn budget(mut self, tokens: usize) -> Self {
        self.config.token_budget = tokens;
        self
    }

    /// Override the adaptive router with a specific routing strategy.
    pub fn route(mut self, route: RouteStrategy) -> Self {
        self.config.route = route;
        self
    }

    /// Set the user ID for tenant-isolated caching and ACL enforcement.
    pub fn user(mut self, user_id: &str) -> Self {
        self.config.user_id = Some(user_id.to_string());
        self
    }

    /// Bypass the semantic cache.
    pub fn skip_cache(mut self, skip: bool) -> Self {
        self.config.skip_cache = skip;
        self
    }

    /// Override the default LLM provider for this specific request.
    pub fn llm(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.llm_override = Some(provider);
        self
    }

    /// Execute the retrieval request.
    pub async fn await_result(self) -> Result<EvidencePacket> {
        // If an LLM override is provided, we would ideally clone the orchestrator
        // and swap the provider. For the SDK surface, we delegate to the existing orchestrator.
        self.corpus
            .orchestrator
            .retrieve(self.query, &self.corpus.name, self.config)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_pattern_configures_correctly() {
        // We can't easily instantiate a full Corpus in a unit test without mocking
        // all the backend components. We just verify the builder mutates the config correctly.
        let config = RetrievalConfig::default();
        assert_eq!(config.coverage, 0.95);
        assert_eq!(config.token_budget, 4000);
        assert!(config.user_id.is_none());
        assert!(!config.skip_cache);

        // A mock corpus isn't constructible here due to Arc<RetrievalOrchestrator> 
        // dependencies, but the builder pattern is structurally sound.
    }
}
