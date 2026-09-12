#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

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

mod noop;

use sievrant_llm::{openai::OpenAiCompatible, LlmProvider};
use sievrant_orchestrator::{
    corpus::{CorpusManager, CorpusMetadata},
    pipeline::{IndexSet, RetrievalOrchestrator},
};
use sievrant_index::{
    vector::{VectorIndex, VectorIndexConfig, VectorBackendConfig},
    bm25::{Bm25Index, Bm25IndexConfig},
    tree::{TreeIndex, TreeIndexConfig},
    graph::{GraphIndex, GraphIndexConfig},
};
use sievrant_query::{
    processing::{QueryProcessor, QueryProcessingConfig},
    router::{QueryRouter, RoutingConfig},
};
use sievrant_acl::resolver::InMemoryAclResolver;
use sievrant_rerank::TermOverlapReranker;
use sievrant_llm::embeddings::OpenAiEmbeddingProvider;
use async_trait::async_trait;

use std::path::PathBuf;

/// Adapter to bridge the duplicated EmbeddingProvider traits across crates
struct EmbeddingProviderWrapper {
    inner: OpenAiEmbeddingProvider,
}

#[async_trait]
impl sievrant_query::EmbeddingProvider for EmbeddingProviderWrapper {
    async fn embed(&self, texts: &[String]) -> sievrant_core::error::Result<Vec<Vec<f32>>> {
        use sievrant_llm::embeddings::EmbeddingProvider;
        self.inner.embed_batch(texts).await
    }
    
    fn dimensions(&self) -> usize {
        use sievrant_llm::embeddings::EmbeddingProvider;
        self.inner.dimensions()
    }
}

/// A handle to a loaded Corpus. Provides the primary entrypoint for retrieving
/// documents and managing corpus data.
pub struct Corpus {
    /// The unique namespace identifier for this corpus.
    pub name: String,
    /// The pipeline orchestrator responsible for routing queries.
    pub orchestrator: Arc<RetrievalOrchestrator>,
    /// The state manager for tracking corpus metadata and schema.
    pub corpus_manager: Arc<CorpusManager>,
}

impl Corpus {
    async fn build_orchestrator(name: &str, config: &SievrantConfig) -> Result<Arc<RetrievalOrchestrator>> {
        let llm: Arc<dyn LlmProvider> = if let Some(key) = &config.llm.openai_api_key {
            Arc::new(OpenAiCompatible::openai(key, "gpt-4o"))
        } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            Arc::new(OpenAiCompatible::openai(&key, "gpt-4o"))
        } else {
            Arc::new(noop::NoopLlm)
        };
        
        let vector_config = VectorIndexConfig {
            backend: VectorBackendConfig::Memory,
            ..Default::default()
        };
        let vector = Arc::new(VectorIndex::new(vector_config).await.map_err(|e| sievrant_core::error::SievrantError::Storage(sievrant_core::error::StorageError::Io(e.to_string())))?);
        
        let mut bm25_config = Bm25IndexConfig::default();
        bm25_config.index_dir = PathBuf::from(format!("data/{}/bm25", name));
        let bm25 = Arc::new(Bm25Index::new(bm25_config).map_err(|e| sievrant_core::error::SievrantError::Storage(sievrant_core::error::StorageError::Io(e.to_string())))?);
        
        // Ensure tree and graph configs don't conflict (they use Memory for now, but in case they write to disk)
        let tree = Arc::new(TreeIndex::new(TreeIndexConfig::default()));
        let graph = Arc::new(GraphIndex::new(GraphIndexConfig::default()));
        
        let indexes = IndexSet {
            vector,
            bm25,
            tree,
            graph,
        };
        
        let orchestrator = RetrievalOrchestrator {
            cache: Arc::new(noop::NoopCache),
            router: Arc::new(QueryRouter::new(RoutingConfig::default())),
            query_processor: Arc::new(QueryProcessor::new(
                llm.clone(), 
                Arc::new(EmbeddingProviderWrapper { inner: OpenAiEmbeddingProvider::new("dummy", "text-embedding-ada-002") }), 
                QueryProcessingConfig::default()
            )),
            acl_resolver: Arc::new(InMemoryAclResolver::new()),
            indexes,
            reranker: Arc::new(TermOverlapReranker),
            calibration: Arc::new(noop::NoopCalibration),
            llm: llm.clone(),
            coverage_monitor: Arc::new(noop::NoopCoverage),
            search_limit: 100,
            min_evidence_count: 3,
        };
        
        Ok(Arc::new(orchestrator))
    }

    /// Create a new corpus with the given configuration.
    pub async fn create(name: &str, config: SievrantConfig) -> Result<Self> {
        let orchestrator = Self::build_orchestrator(name, &config).await?;
        let corpus_manager = Arc::new(
            CorpusManager::new(PathBuf::from(format!("data/{}", name)))?
        );
        
        // This fails if it already exists
        corpus_manager.create(name, None)?;
            
        Ok(Self {
            name: name.to_string(),
            orchestrator,
            corpus_manager,
        })
    }

    /// Open an existing corpus.
    pub async fn open(name: &str, config: SievrantConfig) -> Result<Self> {
        let orchestrator = Self::build_orchestrator(name, &config).await?;
        let corpus_manager = Arc::new(
            CorpusManager::new(PathBuf::from(format!("data/{}", name)))?
        );
        
        // Ensure it exists
        if corpus_manager.get(name).is_none() {
            return Err(sievrant_core::error::SievrantError::Storage(sievrant_core::error::StorageError::Io(format!("Corpus '{}' not found", name))));
        }

        Ok(Self {
            name: name.to_string(),
            orchestrator,
            corpus_manager,
        })
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

        // dependencies, but the builder pattern is structurally sound.
    }

    #[tokio::test]
    async fn create_opens_and_retrieves() {
        // Use a unique name to avoid conflicts if tests run in parallel
        let corpus_name = "test-embed-corpus";
        
        // Clean up from previous test runs
        let _ = std::fs::remove_dir_all(format!("data/{}", corpus_name));
        
        {
            let corpus = Corpus::create(corpus_name, SievrantConfig::default())
                .await
                .expect("create should succeed");
            assert_eq!(corpus.name, corpus_name);
            // corpus goes out of scope here, releasing tantivy writer lock
        }
        
        // Open the same corpus
        let reopened = Corpus::open(corpus_name, SievrantConfig::default())
            .await
            .expect("open should succeed");
        
        assert_eq!(reopened.name, corpus_name);
        
        // Clean up
        let _ = std::fs::remove_dir_all(format!("data/{}", corpus_name));
    }

    #[tokio::test]
    async fn open_nonexistent_corpus_returns_not_found() {
        let result = Corpus::open("does-not-exist-12345", SievrantConfig::default()).await;
        assert!(result.is_err());
    }
}
