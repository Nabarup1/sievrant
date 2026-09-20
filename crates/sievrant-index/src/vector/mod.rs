pub mod memory;
pub mod qdrant;

use async_trait::async_trait;

use sievrant_core::{
    error::Result,
    types::{ChunkMetadata, RankedCandidate, RetrieverType},
};

#[derive(Debug, Clone)]
pub enum IndexQuery {
    Vector {
        embedding: Vec<f32>,
    },
    Lexical {
        text: String,
    },
    Structural {
        text: String,
        path_hints: Vec<String>,
    },
    Graph {
        entities: Vec<String>,
        query_embedding: Vec<f32>,
    },
}

#[derive(Debug, Clone)]
pub struct IndexChunk {
    pub chunk_id: String,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: ChunkMetadata,
}

#[async_trait]
pub trait RetrievalIndex: Send + Sync {
    async fn search(
        &self,
        query: &IndexQuery,
        acl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<RankedCandidate>>;

    async fn insert(&self, chunks: &[IndexChunk]) -> Result<()>;
    async fn delete(&self, chunk_ids: &[String]) -> Result<()>;
    fn index_type(&self) -> RetrieverType;

    async fn ingest_graph_entities(
        &self,
        _chunk_id: &str,
        _entities: &[String],
        _relations: &[(String, String, String)],
    ) -> Result<()> {
        Ok(())
    }

    async fn ingest_tree_summary(&self, _node_id: &str, _summary: String) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
    fn model_id(&self) -> &str;
}

#[derive(Debug, Clone)]
pub enum VectorBackendConfig {
    Memory,
    Qdrant {
        url: String,
        api_key: Option<String>,
        collection_name: String,
    },
    // LanceDb { uri: String },
}

impl Default for VectorBackendConfig {
    fn default() -> Self {
        Self::Qdrant {
            url: "http://localhost:6333".into(),
            api_key: None,
            collection_name: "sievrant".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    pub backend: VectorBackendConfig,
    pub embedding_dimensions: usize,
    pub hnsw_m: u64,
    pub hnsw_ef: u64,
    pub search_ef: u64,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            backend: VectorBackendConfig::default(),
            embedding_dimensions: 3072,
            hnsw_m: 16,
            hnsw_ef: 100,
            search_ef: 128,
        }
    }
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert_vectors(&self, chunks: &[IndexChunk]) -> Result<()>;

    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        acl_filter: &[String],
    ) -> Result<Vec<RankedCandidate>>;

    async fn delete_documents(&self, chunk_ids: &[String]) -> Result<()>;
}

pub struct VectorIndex {
    store: Box<dyn VectorStore>,
}

impl VectorIndex {
    pub async fn new(config: VectorIndexConfig) -> Result<Self> {
        let store: Box<dyn VectorStore> = match config.backend {
            VectorBackendConfig::Memory => {
                Box::new(memory::MemoryStore::new(config.embedding_dimensions))
            }
            VectorBackendConfig::Qdrant {
                url,
                api_key,
                collection_name,
            } => {
                let qdrant = qdrant::QdrantStore::new(
                    url,
                    api_key,
                    collection_name,
                    config.embedding_dimensions,
                )
                .await?;
                Box::new(qdrant)
            }
        };

        Ok(Self { store })
    }
}

#[async_trait]
impl RetrievalIndex for VectorIndex {
    async fn search(
        &self,
        query: &IndexQuery,
        acl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<RankedCandidate>> {
        let embedding = match query {
            IndexQuery::Vector { embedding } => embedding,
            IndexQuery::Graph {
                query_embedding, ..
            } => query_embedding,
            _ => return Ok(vec![]),
        };

        self.store.search(embedding, limit, acl_filter).await
    }

    async fn insert(&self, chunks: &[IndexChunk]) -> Result<()> {
        self.store.upsert_vectors(chunks).await
    }

    async fn delete(&self, chunk_ids: &[String]) -> Result<()> {
        self.store.delete_documents(chunk_ids).await
    }

    fn index_type(&self) -> RetrieverType {
        RetrieverType::VectorIndex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index_config_defaults() {
        let config = VectorIndexConfig::default();
        assert_eq!(config.embedding_dimensions, 3072);
        assert_eq!(config.hnsw_m, 16);
        assert_eq!(config.hnsw_ef, 100);
        assert_eq!(config.search_ef, 128);

        match config.backend {
            VectorBackendConfig::Qdrant { collection_name, .. } => {
                assert_eq!(collection_name, "sievrant");
            }
            _ => panic!("Expected default backend to be Qdrant"),
        }
    }
}
