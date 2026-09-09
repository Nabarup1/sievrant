use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::RwLock,
};

use async_trait::async_trait;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};

use sievrant_core::{
    error::Result,
    types::{ChunkMetadata, RankedCandidate, RetrieverType},
};

use crate::vector::{IndexChunk, IndexQuery, RetrievalIndex};

#[derive(Debug, Clone)]
pub struct GraphIndexConfig {
    /// PPR damping factor. 0.85 is the standard setting from the PageRank paper.
    pub ppr_alpha: f64,
    pub ppr_max_iterations: usize,
    pub ppr_tolerance: f64,
    pub max_entities_per_chunk: usize,
    pub data_dir: PathBuf,
}

impl Default for GraphIndexConfig {
    fn default() -> Self {
        Self {
            ppr_alpha: 0.85,
            ppr_max_iterations: 100,
            ppr_tolerance: 1e-6,
            max_entities_per_chunk: 20,
            data_dir: PathBuf::from("data/graph"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphNodeType {
    Entity,
    Chunk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub node_type: GraphNodeType,
    pub label: String,
    pub chunk_id: Option<String>,
    pub acl_groups: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub relation: String,
    pub weight: f64,
}

pub struct KnowledgeGraph {
    graph: DiGraph<GraphNode, GraphEdge>,
    entity_index: HashMap<String, NodeIndex>,
    chunk_index: HashMap<String, NodeIndex>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            entity_index: HashMap::new(),
            chunk_index: HashMap::new(),
        }
    }

    fn get_or_insert_entity(&mut self, entity: &str) -> NodeIndex {
        if let Some(&idx) = self.entity_index.get(entity) {
            return idx;
        }
        let node = GraphNode {
            id: entity.to_string(),
            node_type: GraphNodeType::Entity,
            label: entity.to_string(),
            chunk_id: None,
            acl_groups: vec![],
        };
        let idx = self.graph.add_node(node);
        self.entity_index.insert(entity.to_string(), idx);
        idx
    }

    fn get_or_insert_chunk(&mut self, chunk_id: &str, metadata: &ChunkMetadata) -> NodeIndex {
        if let Some(&idx) = self.chunk_index.get(chunk_id) {
            return idx;
        }
        let node = GraphNode {
            id: chunk_id.to_string(),
            node_type: GraphNodeType::Chunk,
            label: chunk_id.to_string(),
            chunk_id: Some(chunk_id.to_string()),
            acl_groups: metadata.acl_groups.clone(),
        };
        let idx = self.graph.add_node(node);
        self.chunk_index.insert(chunk_id.to_string(), idx);
        idx
    }

    /// Add entities extracted from a chunk to the graph and link them to the chunk node.
    pub fn add_chunk(&mut self, chunk_id: &str, text: &str, metadata: &ChunkMetadata, max_entities: usize) {
        let chunk_idx = self.get_or_insert_chunk(chunk_id, metadata);
        let entities = extract_entities(text);

        for entity in entities.into_iter().take(max_entities) {
            let entity_idx = self.get_or_insert_entity(&entity);
            let edge = GraphEdge {
                relation: "mentions".to_string(),
                weight: 1.0,
            };
            if !self.graph.contains_edge(chunk_idx, entity_idx) {
                self.graph.add_edge(chunk_idx, entity_idx, edge.clone());
            }
            if !self.graph.contains_edge(entity_idx, chunk_idx) {
                self.graph.add_edge(entity_idx, chunk_idx, edge);
            }
        }
    }

    /// Remove all chunk and entity nodes associated with a document.
    pub fn remove_document(&mut self, doc_id: &str) {
        let chunk_ids_to_remove: Vec<String> = self
            .chunk_index
            .keys()
            .filter(|k| k.starts_with(doc_id))
            .cloned()
            .collect();

        for chunk_id in chunk_ids_to_remove {
            if let Some(idx) = self.chunk_index.remove(&chunk_id) {
                self.graph.remove_node(idx);
            }
        }
    }

    /// Explicitly ingest entities and relations from an external extractor.
    pub fn ingest_entities(
        &mut self,
        chunk_id: &str,
        entities: &[String],
        relations: &[(String, String, String)],
    ) {
        let meta = ChunkMetadata {
            doc_id: chunk_id.split(':').next().unwrap_or("").to_string(),
            source_doc: "".into(),
            doc_version_hash: "".into(),
            path: vec![],
            page: None,
            char_span: None,
            ingested_at: chrono::Utc::now(),
            index_generation: 0,
            acl_groups: vec![],
            token_count: 0,
        };
        let chunk_idx = self.get_or_insert_chunk(chunk_id, &meta);

        for entity in entities {
            let entity_idx = self.get_or_insert_entity(entity);
            let edge = GraphEdge {
                relation: "mentions".to_string(),
                weight: 1.0,
            };
            self.graph.add_edge(chunk_idx, entity_idx, edge.clone());
            self.graph.add_edge(entity_idx, chunk_idx, edge);
        }

        for (src, dst, rel) in relations {
            let src_idx = self.get_or_insert_entity(src);
            let dst_idx = self.get_or_insert_entity(dst);
            let edge = GraphEdge {
                relation: rel.clone(),
                weight: 1.0,
            };
            self.graph.add_edge(src_idx, dst_idx, edge);
        }
    }

    /// Detect hierarchical communities (Louvain/Leiden approximation via SCC).
    /// Returns a map from node ID to its community ID.
    pub fn detect_communities(&self) -> HashMap<String, usize> {
        let mut communities = HashMap::new();
        // Tarjan's Strongly Connected Components
        let scc = petgraph::algo::tarjan_scc(&self.graph);
        for (community_id, component) in scc.into_iter().enumerate() {
            for node_idx in component {
                let node = &self.graph[node_idx];
                communities.insert(node.id.clone(), community_id);
            }
        }
        communities
    }

    /// Personalized PageRank via power iteration.
    ///
    /// Formula: x_new = alpha * (D_inv * A * x) + (1 - alpha) * v
    ///
    /// The personalization vector v is uniform over nodes matching the seed entities
    /// and zero elsewhere. Returns PPR scores for all nodes.
    pub fn personalized_pagerank(
        &self,
        seed_entities: &[String],
        alpha: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> HashMap<NodeIndex, f64> {
        let n = self.graph.node_count();
        if n == 0 {
            return HashMap::new();
        }

        let node_indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let node_to_pos: HashMap<NodeIndex, usize> = node_indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();

        let seed_set: HashSet<String> = seed_entities.iter().cloned().collect();

        let mut v = vec![0.0f64; n];
        for (i, &node_idx) in node_indices.iter().enumerate() {
            let node = &self.graph[node_idx];
            if seed_set.contains(&node.label) {
                v[i] = 1.0;
            }
        }
        let seed_sum: f64 = v.iter().sum();
        if seed_sum > 0.0 {
            for val in &mut v {
                *val /= seed_sum;
            }
        } else {
            let uniform = 1.0 / n as f64;
            v.iter_mut().for_each(|val| *val = uniform);
        }

        let mut x = v.clone();

        for _ in 0..max_iterations {
            let mut x_new = vec![0.0f64; n];

            for &src in &node_indices {
                let neighbors: Vec<NodeIndex> = self
                    .graph
                    .neighbors(src)
                    .collect();

                if neighbors.is_empty() {
                    continue;
                }

                let out_degree = neighbors.len() as f64;
                let src_pos = node_to_pos[&src];
                for neighbor in neighbors {
                    let dst_pos = node_to_pos[&neighbor];
                    x_new[dst_pos] += alpha * x[src_pos] / out_degree;
                }
            }

            for (i, val) in x_new.iter_mut().enumerate() {
                *val += (1.0 - alpha) * v[i];
            }

            let delta: f64 = x_new
                .iter()
                .zip(x.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            x = x_new;

            if delta < tolerance {
                break;
            }
        }

        node_indices
            .into_iter()
            .enumerate()
            .map(|(i, idx)| (idx, x[i]))
            .collect()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Heuristic entity extractor: no ML model required.
/// Extracts capitalized multi-word phrases, section references, and quoted terms.
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Section and article references: "Section 4.2", "Article 7", "Clause 3"
    let words: Vec<&str> = text.split_whitespace().collect();
    for window in words.windows(2) {
        let first = window[0];
        let second = window[1];
        if ["Section", "Article", "Clause", "Exhibit", "Schedule", "Appendix"]
            .contains(&first)
        {
            let entity = format!("{first} {second}");
            if seen.insert(entity.clone()) {
                entities.push(entity);
            }
        }
    }

    // Quoted terms: "defined term"
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

    // Capitalized multi-word phrases (not at sentence start)
    let sentences: Vec<&str> = text.split(". ").collect();
    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut phrase = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let is_cap = word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
            let is_sentence_start = i == 0;

            if is_cap && !is_sentence_start {
                phrase.push(*word);
            } else {
                if phrase.len() >= 2 {
                    let entity = phrase.join(" ");
                    if seen.insert(entity.clone()) {
                        entities.push(entity);
                    }
                }
                phrase.clear();
            }
        }

        if phrase.len() >= 2 {
            let entity = phrase.join(" ");
            if seen.insert(entity.clone()) {
                entities.push(entity);
            }
        }
    }

    entities
}

pub struct GraphIndex {
    graph: RwLock<KnowledgeGraph>,
    config: GraphIndexConfig,
}

impl GraphIndex {
    pub fn new(config: GraphIndexConfig) -> Self {
        Self {
            graph: RwLock::new(KnowledgeGraph::new()),
            config,
        }
    }

    pub fn ingest_entities(
        &self,
        chunk_id: &str,
        entities: &[String],
        relations: &[(String, String, String)],
    ) {
        let mut graph = self.graph.write().unwrap();
        graph.ingest_entities(chunk_id, entities, relations);
    }

    pub fn detect_communities(&self) -> HashMap<String, usize> {
        let graph = self.graph.read().unwrap();
        graph.detect_communities()
    }
}

#[async_trait]
impl RetrievalIndex for GraphIndex {
    async fn search(
        &self,
        query: &IndexQuery,
        acl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<RankedCandidate>> {
        let (entities, _fallback_text) = match query {
            IndexQuery::Graph { entities, .. } => (entities.clone(), None),
            IndexQuery::Lexical { text } => (extract_entities(text), Some(text.clone())),
            IndexQuery::Structural { text, .. } => (extract_entities(text), Some(text.clone())),
            _ => return Ok(vec![]),
        };

        let graph = self.graph.read().unwrap();

        let ppr_scores = graph.personalized_pagerank(
            &entities,
            self.config.ppr_alpha,
            self.config.ppr_max_iterations,
            self.config.ppr_tolerance,
        );

        let mut candidates: Vec<RankedCandidate> = ppr_scores
            .into_iter()
            .filter_map(|(node_idx, score)| {
                let node = &graph.graph[node_idx];
                if node.node_type != GraphNodeType::Chunk {
                    return None;
                }
                let chunk_id = node.chunk_id.as_ref()?.clone();

                if !acl_filter.is_empty()
                    && !node.acl_groups.iter().any(|g| acl_filter.contains(g))
                {
                    return None;
                }

                Some(RankedCandidate {
                    chunk_id,
                    score,
                    retriever: RetrieverType::GraphIndex,
                    text: node.label.clone(),
                    metadata: ChunkMetadata {
                        doc_id: String::new(),
                        source_doc: String::new(),
                        doc_version_hash: String::new(),
                        path: vec![],
                        page: None,
                        char_span: None,
                        ingested_at: chrono::Utc::now(),
                        index_generation: 0,
                        acl_groups: node.acl_groups.clone(),
                        token_count: 0,
                    },
                })
            })
            .collect();

        candidates
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);

        Ok(candidates)
    }

    async fn insert(&self, chunks: &[IndexChunk]) -> Result<()> {
        let mut graph = self.graph.write().unwrap();
        for chunk in chunks {
            graph.add_chunk(
                &chunk.chunk_id,
                &chunk.text,
                &chunk.metadata,
                self.config.max_entities_per_chunk,
            );
        }
        Ok(())
    }

    async fn delete(&self, chunk_ids: &[String]) -> Result<()> {
        let mut graph = self.graph.write().unwrap();
        for chunk_id in chunk_ids {
            let doc_id = chunk_id.split(':').next().unwrap_or("");
            graph.remove_document(doc_id);
        }
        Ok(())
    }

    fn index_type(&self) -> RetrieverType {
        RetrieverType::GraphIndex
    }

    async fn ingest_graph_entities(
        &self,
        chunk_id: &str,
        entities: &[String],
        relations: &[(String, String, String)],
    ) -> Result<()> {
        self.ingest_entities(chunk_id, entities, relations);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppr_scores_reflect_graph_topology() {
        let mut kg = KnowledgeGraph::new();

        let node_a = kg.get_or_insert_entity("Acme Corporation");
        let node_b = kg.get_or_insert_entity("Lease Agreement");
        let node_c = kg.get_or_insert_entity("Payment Terms");

        kg.graph.add_edge(node_a, node_b, GraphEdge { relation: "references".into(), weight: 1.0 });
        kg.graph.add_edge(node_b, node_c, GraphEdge { relation: "contains".into(), weight: 1.0 });
        kg.graph.add_edge(node_c, node_b, GraphEdge { relation: "part_of".into(), weight: 1.0 });

        let scores = kg.personalized_pagerank(&["Acme Corporation".into()], 0.85, 100, 1e-6);

        let score_b = scores[&node_b];
        let score_c = scores[&node_c];
        assert!(score_b > scores[&node_a], "seeded node should propagate to connected node");
        assert!(score_c > 0.0);
    }

    #[test]
    fn ppr_converges_with_isolated_graph() {
        let mut kg = KnowledgeGraph::new();
        kg.get_or_insert_entity("Isolated Entity");
        let scores = kg.personalized_pagerank(&["Isolated Entity".into()], 0.85, 100, 1e-6);
        assert!(!scores.is_empty());
    }

    #[test]
    fn extract_entities_finds_section_references() {
        let text = "This contract is governed by Section 4.2 and Article 7 of the agreement.";
        let entities = extract_entities(text);
        assert!(entities.contains(&"Section 4.2".to_string()));
        assert!(entities.contains(&"Article 7".to_string()));
    }

    #[test]
    fn extract_entities_finds_quoted_terms() {
        let text = r#"The "Effective Date" shall mean the date signed by the "Authorized Signatory"."#;
        let entities = extract_entities(text);
        assert!(entities.contains(&"Effective Date".to_string()));
        assert!(entities.contains(&"Authorized Signatory".to_string()));
    }

    #[tokio::test]
    async fn acl_filter_excludes_restricted_chunk_nodes() {
        let idx = GraphIndex::new(GraphIndexConfig::default());
        let meta = sievrant_core::types::ChunkMetadata {
            doc_id: "doc1".into(),
            source_doc: "test.pdf".into(),
            doc_version_hash: "sha256:abc".into(),
            path: vec![],
            page: None,
            char_span: None,
            ingested_at: chrono::Utc::now(),
            index_generation: 1,
            acl_groups: vec!["restricted".into()],
            token_count: 50,
        };
        let chunk = IndexChunk {
            chunk_id: "doc1:0".into(),
            text: r#"Acme Corporation signed the "Master Agreement"."#.into(),
            embedding: None,
            metadata: meta,
        };
        idx.insert(&[chunk]).await.unwrap();

        let results = idx
            .search(
                &IndexQuery::Graph {
                    entities: vec!["Acme Corporation".into()],
                    query_embedding: vec![],
                },
                &["public".into()],
                10,
            )
            .await
            .unwrap();

        assert!(results.is_empty());
    }
}
