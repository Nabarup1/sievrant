pub mod ppr;

use std::{collections::HashSet, sync::RwLock};

use async_trait::async_trait;

use sievrant_core::{
    error::Result,
    types::{ChunkMetadata, RankedCandidate, RetrieverType},
};

use crate::vector::{IndexChunk, IndexQuery, RetrievalIndex};

pub use ppr::{GraphEdge, GraphIndexConfig, GraphNode, GraphNodeType, KnowledgeGraph};

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
        if [
            "Section", "Article", "Clause", "Exhibit", "Schedule", "Appendix",
        ]
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
            let is_cap = word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);
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

    pub fn detect_communities(&self) -> std::collections::HashMap<String, usize> {
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

                if !acl_filter.is_empty() && !node.acl_groups.iter().any(|g| acl_filter.contains(g))
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

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

        kg.graph.add_edge(
            node_a,
            node_b,
            GraphEdge {
                relation: "references".into(),
                weight: 1.0,
            },
        );
        kg.graph.add_edge(
            node_b,
            node_c,
            GraphEdge {
                relation: "contains".into(),
                weight: 1.0,
            },
        );
        kg.graph.add_edge(
            node_c,
            node_b,
            GraphEdge {
                relation: "part_of".into(),
                weight: 1.0,
            },
        );

        let scores = kg.personalized_pagerank(&["Acme Corporation".into()], 0.85, 100, 1e-6);

        let score_b = scores[&node_b];
        let score_c = scores[&node_c];
        assert!(
            score_b > scores[&node_a],
            "seeded node should propagate to connected node"
        );
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
        let text =
            r#"The "Effective Date" shall mean the date signed by the "Authorized Signatory"."#;
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
