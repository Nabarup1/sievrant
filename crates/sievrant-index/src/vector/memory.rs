use std::sync::Arc;
use tokio::sync::RwLock;

use async_trait::async_trait;

use sievrant_core::{
    error::Result,
    types::{RankedCandidate, RetrieverType},
};
use super::{VectorStore, IndexChunk};

#[derive(Clone)]
struct MemoryPoint {
    chunk: IndexChunk,
}

pub struct MemoryStore {
    _dimensions: usize,
    points: Arc<RwLock<Vec<MemoryPoint>>>,
}

impl MemoryStore {
    pub fn new(dimensions: usize) -> Self {
        Self {
            _dimensions: dimensions,
            points: Arc::new(RwLock::new(Vec::new())),
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        }
    }
}

#[async_trait]
impl VectorStore for MemoryStore {
    async fn upsert_vectors(&self, chunks: &[IndexChunk]) -> Result<()> {
        let mut write_guard = self.points.write().await;
        
        for chunk in chunks {
            // Naive upsert (delete existing then insert)
            write_guard.retain(|p| p.chunk.chunk_id != chunk.chunk_id);
            if chunk.embedding.is_some() {
                write_guard.push(MemoryPoint {
                    chunk: chunk.clone(),
                });
            }
        }
        
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        acl_filter: &[String],
    ) -> Result<Vec<RankedCandidate>> {
        let read_guard = self.points.read().await;
        
        let mut results: Vec<(f32, &IndexChunk)> = read_guard.iter().filter_map(|point| {
            if !acl_filter.is_empty() {
                let has_access = acl_filter.iter().any(|allowed_group| {
                    point.chunk.metadata.acl_groups.contains(allowed_group)
                });
                if !has_access {
                    return None;
                }
            }
            
            let embedding = point.chunk.embedding.as_ref()?;
            let sim = Self::cosine_similarity(query_embedding, embedding);
            Some((sim, &point.chunk))
        }).collect();
        
        results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let candidates = results.into_iter().take(top_k).map(|(sim, chunk)| {
            RankedCandidate {
                chunk_id: chunk.chunk_id.clone(),
                score: sim as f64,
                retriever: RetrieverType::VectorIndex,
                text: chunk.text.clone(),
                metadata: chunk.metadata.clone(),
            }
        }).collect();
        
        Ok(candidates)
    }

    async fn delete_documents(&self, chunk_ids: &[String]) -> Result<()> {
        let mut write_guard = self.points.write().await;
        write_guard.retain(|p| !chunk_ids.contains(&p.chunk.chunk_id));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::types::ChunkMetadata;

    fn test_chunk(id: &str, groups: Vec<&str>, embedding: Vec<f32>) -> IndexChunk {
        IndexChunk {
            chunk_id: id.to_string(),
            text: "test".to_string(),
            embedding: Some(embedding),
            metadata: ChunkMetadata {
                doc_id: "doc1".to_string(),
                source_doc: "doc1.txt".to_string(),
                doc_version_hash: "hash".to_string(),
                path: vec![],
                page: None,
                char_span: None,
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: groups.into_iter().map(|s| s.to_string()).collect(),
                token_count: 10,
            },
        }
    }

    #[tokio::test]
    async fn memory_store_enforces_acl() {
        let store = MemoryStore::new(2);
        
        let chunks = vec![
            test_chunk("c1", vec!["engineering"], vec![1.0, 0.0]),
            test_chunk("c2", vec!["hr"], vec![1.0, 0.0]),
        ];
        
        store.upsert_vectors(&chunks).await.unwrap();

        // Engineering can only see c1
        let results = store.search(&[1.0, 0.0], 10, &["engineering".to_string()]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "c1");

        // HR can only see c2
        let results = store.search(&[1.0, 0.0], 10, &["hr".to_string()]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "c2");

        // Executive can see neither
        let results = store.search(&[1.0, 0.0], 10, &["executive".to_string()]).await.unwrap();
        assert_eq!(results.len(), 0);

        // No filter means see all (internal system query)
        let results = store.search(&[1.0, 0.0], 10, &[]).await.unwrap();
        assert_eq!(results.len(), 2);
    }
}
