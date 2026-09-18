use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};

use sievrant_core::types::ChunkMetadata;

use super::extract_entities;

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
    pub(crate) graph: DiGraph<GraphNode, GraphEdge>,
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
