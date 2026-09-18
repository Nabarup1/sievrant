use std::{path::PathBuf, sync::RwLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use sievrant_core::{
    error::Result,
    types::{ChunkMetadata, RankedCandidate, RetrieverType},
};

use crate::vector::{IndexChunk, IndexQuery, RetrievalIndex};

#[derive(Debug, Clone)]
pub struct TreeIndexConfig {
    /// Beam search width: how many children are expanded at each level.
    pub beam_width: usize,
    pub max_depth: usize,
    pub data_dir: PathBuf,
}

impl Default for TreeIndexConfig {
    fn default() -> Self {
        Self {
            beam_width: 5,
            max_depth: 10,
            data_dir: PathBuf::from("data/tree"),
        }
    }
}

/// A node in the hierarchical document tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    pub id: String,
    pub label: String,
    pub level: u32,
    pub summary: Option<String>,
    pub chunk_id: Option<String>,
    pub children: Vec<TreeNode>,
    pub metadata: Option<ChunkMetadata>,
}

impl TreeNode {
    fn new_internal(id: String, label: String, level: u32) -> Self {
        Self {
            id,
            label,
            level,
            summary: None,
            chunk_id: None,
            children: vec![],
            metadata: None,
        }
    }

    fn new_leaf(
        id: String,
        label: String,
        level: u32,
        chunk_id: String,
        metadata: ChunkMetadata,
    ) -> Self {
        Self {
            id,
            label,
            level,
            summary: None,
            chunk_id: Some(chunk_id),
            children: vec![],
            metadata: Some(metadata),
        }
    }

    fn score(&self, query_terms: &[&str]) -> f64 {
        let text =
            format!("{} {}", self.label, self.summary.as_deref().unwrap_or("")).to_lowercase();

        let matches = query_terms
            .iter()
            .filter(|t| text.contains(t.to_lowercase().as_str()))
            .count();

        let depth_penalty = 1.0 / (1.0 + self.level as f64 * 0.1);
        (matches as f64) * depth_penalty
    }

    fn acl_allowed(&self, allowed_groups: &[String]) -> bool {
        match &self.metadata {
            None => true,
            Some(meta) => {
                if meta.acl_groups.is_empty() {
                    return true;
                }
                meta.acl_groups.iter().any(|g| allowed_groups.contains(g))
            }
        }
    }
}

pub struct TreeIndex {
    root: RwLock<TreeNode>,
    config: TreeIndexConfig,
}

impl TreeIndex {
    pub fn new(config: TreeIndexConfig) -> Self {
        let root = TreeNode::new_internal("root".into(), "root".into(), 0);
        Self {
            root: RwLock::new(root),
            config,
        }
    }

    pub fn build_from_chunks(chunks: &[(String, ChunkMetadata, Vec<String>)]) -> Self {
        let config = TreeIndexConfig::default();
        let index = Self::new(config);
        let mut root = index.root.write().unwrap();

        for (chunk_id, metadata, path) in chunks {
            Self::insert_at(&mut root, chunk_id, metadata, path, 0);
        }

        drop(root);
        index
    }

    fn insert_at(
        node: &mut TreeNode,
        chunk_id: &str,
        metadata: &ChunkMetadata,
        path: &[String],
        depth: usize,
    ) {
        if path.is_empty() {
            let leaf = TreeNode::new_leaf(
                format!("{}:leaf", chunk_id),
                chunk_id.to_string(),
                node.level + 1,
                chunk_id.to_string(),
                metadata.clone(),
            );
            node.children.push(leaf);
            return;
        }

        let label = &path[0];
        let child = node.children.iter_mut().find(|c| &c.label == label);

        if let Some(existing) = child {
            Self::insert_at(existing, chunk_id, metadata, &path[1..], depth + 1);
        } else {
            let child_id = format!("{}:{}", node.id, label);
            let mut new_child = TreeNode::new_internal(child_id, label.clone(), node.level + 1);
            Self::insert_at(&mut new_child, chunk_id, metadata, &path[1..], depth + 1);
            node.children.push(new_child);
        }
    }

    fn beam_search<'a>(
        node: &'a TreeNode,
        query_terms: &[&str],
        acl_filter: &[String],
        beam_width: usize,
        max_depth: usize,
        current_depth: usize,
        results: &mut Vec<(f64, &'a TreeNode)>,
    ) {
        // Collect and sort all raw references first, then recurse.
        // We need 'node' to live long enough. We collect child refs with scores.
        if current_depth >= max_depth {
            return;
        }

        let mut scored_children: Vec<(&TreeNode, f64)> = node
            .children
            .iter()
            .map(|child| {
                let score = child.score(query_terms);
                (child, score)
            })
            .collect();

        scored_children.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (child, score) in scored_children.into_iter().take(beam_width) {
            if child.chunk_id.is_some() {
                if child.acl_allowed(acl_filter) {
                    results.push((score, child));
                }
            } else {
                Self::beam_search(
                    child,
                    query_terms,
                    acl_filter,
                    beam_width,
                    max_depth,
                    current_depth + 1,
                    results,
                );
            }
        }
    }
}

#[async_trait]
impl RetrievalIndex for TreeIndex {
    async fn search(
        &self,
        query: &IndexQuery,
        acl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<RankedCandidate>> {
        let text = match query {
            IndexQuery::Structural { text, .. } => text.clone(),
            IndexQuery::Lexical { text } => text.clone(),
            _ => return Ok(vec![]),
        };

        let query_terms: Vec<&str> = text.split_whitespace().collect();
        let root = self.root.read().unwrap();

        let mut raw_results: Vec<(f64, &TreeNode)> = Vec::new();
        Self::beam_search(
            &root,
            &query_terms,
            acl_filter,
            self.config.beam_width,
            self.config.max_depth,
            0,
            &mut raw_results,
        );

        raw_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let candidates = raw_results
            .into_iter()
            .take(limit)
            .filter_map(|(score, node)| {
                let chunk_id = node.chunk_id.as_ref()?.clone();
                let metadata = node.metadata.as_ref()?.clone();
                Some(RankedCandidate {
                    chunk_id,
                    score,
                    retriever: RetrieverType::TreeIndex,
                    text: node.label.clone(),
                    metadata,
                })
            })
            .collect();

        Ok(candidates)
    }

    async fn insert(&self, chunks: &[IndexChunk]) -> Result<()> {
        let mut root = self.root.write().unwrap();
        for chunk in chunks {
            Self::insert_at(
                &mut root,
                &chunk.chunk_id,
                &chunk.metadata,
                &chunk.metadata.path,
                0,
            );
        }
        Ok(())
    }

    async fn delete(&self, chunk_ids: &[String]) -> Result<()> {
        let mut root = self.root.write().unwrap();
        let chunk_set: std::collections::HashSet<&str> =
            chunk_ids.iter().map(|s| s.as_str()).collect();
        Self::prune_deleted(&mut root, &chunk_set);
        Ok(())
    }

    fn index_type(&self) -> RetrieverType {
        RetrieverType::TreeIndex
    }

    async fn ingest_tree_summary(&self, node_id: &str, summary: String) -> Result<()> {
        self.inject_summary(node_id, summary);
        Ok(())
    }
}

impl TreeIndex {
    fn prune_deleted(node: &mut TreeNode, deleted_ids: &std::collections::HashSet<&str>) {
        node.children.retain(|child| {
            if let Some(cid) = &child.chunk_id {
                !deleted_ids.contains(cid.as_str())
            } else {
                true
            }
        });
        for child in &mut node.children {
            Self::prune_deleted(child, deleted_ids);
        }
    }

    pub fn inject_summary(&self, node_id: &str, summary: String) -> bool {
        let mut root = self.root.write().unwrap();
        Self::inject_summary_recursive(&mut root, node_id, summary)
    }

    fn inject_summary_recursive(node: &mut TreeNode, target_id: &str, summary: String) -> bool {
        if node.id == target_id {
            node.summary = Some(summary);
            return true;
        }
        for child in &mut node.children {
            if Self::inject_summary_recursive(child, target_id, summary.clone()) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::types::ChunkMetadata;

    fn make_metadata(doc_id: &str, path: Vec<String>, acl: Vec<String>) -> ChunkMetadata {
        ChunkMetadata {
            doc_id: doc_id.into(),
            source_doc: "test.pdf".into(),
            doc_version_hash: "sha256:abc".into(),
            path,
            page: None,
            char_span: None,
            ingested_at: chrono::Utc::now(),
            index_generation: 1,
            acl_groups: acl,
            token_count: 50,
        }
    }

    #[tokio::test]
    async fn tree_builds_and_finds_leaves_by_path() {
        let config = TreeIndexConfig::default();
        let idx = TreeIndex::new(config);
        let meta = make_metadata(
            "doc1",
            vec!["Article 4".into(), "Term".into()],
            vec!["public".into()],
        );

        let chunk = IndexChunk {
            chunk_id: "doc1:0".into(),
            text: "lease term is 60 months".into(),
            embedding: None,
            metadata: meta,
        };
        idx.insert(&[chunk]).await.unwrap();

        let results = idx
            .search(
                &IndexQuery::Structural {
                    text: "Article 4 Term".into(),
                    path_hints: vec!["Article 4".into()],
                },
                &["public".into()],
                10,
            )
            .await
            .unwrap();

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn acl_filter_excludes_restricted_leaves() {
        let config = TreeIndexConfig::default();
        let idx = TreeIndex::new(config);
        let meta = make_metadata("doc1", vec!["Section 1".into()], vec!["restricted".into()]);
        let chunk = IndexChunk {
            chunk_id: "doc1:0".into(),
            text: "confidential data".into(),
            embedding: None,
            metadata: meta,
        };
        idx.insert(&[chunk]).await.unwrap();

        let results = idx
            .search(
                &IndexQuery::Structural {
                    text: "Section 1".into(),
                    path_hints: vec![],
                },
                &["public".into()],
                10,
            )
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_build_from_chunks_and_beam_search_acl() {
        // Create 2 chunks with different paths and ACLs
        let meta1 = ChunkMetadata {
            doc_id: "doc1".into(),
            source_doc: "doc1_src".into(),
            doc_version_hash: "hash".into(),
            path: vec![],
            page: None,
            char_span: None,
            ingested_at: chrono::Utc::now(),
            index_generation: 1,
            acl_groups: vec!["groupA".into()],
            token_count: 10,
        };

        let meta2 = ChunkMetadata {
            doc_id: "doc2".into(),
            source_doc: "doc2_src".into(),
            doc_version_hash: "hash".into(),
            path: vec![],
            page: None,
            char_span: None,
            ingested_at: chrono::Utc::now(),
            index_generation: 1,
            acl_groups: vec!["groupB".into()],
            token_count: 10,
        };

        let chunks = vec![
            ("chunk1".to_string(), meta1.clone(), vec!["root".into(), "branchA".into()]),
            ("chunk2".to_string(), meta2.clone(), vec!["root".into(), "branchB".into()]),
        ];

        let index = TreeIndex::build_from_chunks(&chunks);

        // Run beam search as a user with "groupA" access only
        let mut results = vec![];
        let root = index.root.read().unwrap();
        
        TreeIndex::beam_search(
            &root,
            &["brancha", "branchb"], // query terms
            &["groupA".into()],      // acl filter
            5,                       // beam_width
            10,                      // max_depth
            0,
            &mut results,
        );

        // Only chunk1 should be retrieved, chunk2 is filtered out due to ACL
        assert_eq!(results.len(), 1);
        let (_, node) = results[0];
        assert_eq!(node.chunk_id.as_deref(), Some("chunk1"));
    }
}
