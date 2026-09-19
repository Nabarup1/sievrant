use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::types::{ChunkMetadata, Document, DocumentMetadata};

/// How to split a document into chunks. StructureAware is the recommended default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Respects document hierarchy. Never splits mid-section when the section fits in the limit.
    StructureAware {
        max_chunk_tokens: usize,
        overlap_tokens: usize,
    },
    /// Splits at semantic topic boundaries. More expensive; intended for unstructured prose.
    Semantic {
        max_chunk_tokens: usize,
        similarity_threshold: f64,
    },
    /// Fixed-size splitting. Included for compatibility; destroys semantic structure.
    FixedSize {
        chunk_size_tokens: usize,
        overlap_tokens: usize,
    },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        ChunkingStrategy::StructureAware {
            max_chunk_tokens: 512,
            overlap_tokens: 50,
        }
    }
}

/// Trait for parsing raw document bytes or text into a structured representation.
#[async_trait]
pub trait DocumentParser: Send + Sync {
    async fn parse(&self, content: &str, content_type: &str) -> Result<ParsedDocument>;
}

/// A document parsed into its hierarchical section structure.
pub struct ParsedDocument {
    pub sections: Vec<Section>,
    pub title: Option<String>,
    pub metadata: DocumentMetadata,
}

/// A structural node in a parsed document.
pub struct Section {
    pub heading: Option<String>,
    /// Heading level: 1 for H1, 2 for H2, and so on.
    pub level: u32,
    pub content: String,
    pub children: Vec<Section>,
    pub page: Option<u32>,
    pub char_span: [usize; 2],
}

/// A single chunk ready for indexing.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    /// Hierarchical path from root to this chunk, e.g. ["Article 4", "4.2 Term"].
    pub path: Vec<String>,
    pub page: Option<u32>,
    pub char_span: Option<[usize; 2]>,
    pub token_count: usize,
}

/// Approximate token count using the 4-chars-per-token heuristic.
/// Exact counts come from the embedding model tokenizer.
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Compute SHA-256 of document content for version tracking.
pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

/// Splits a ParsedDocument into chunks according to the chosen strategy.
pub struct Chunker;

impl Chunker {
    pub fn chunk(doc: &ParsedDocument, strategy: &ChunkingStrategy, doc_id: &str) -> Vec<Chunk> {
        match strategy {
            ChunkingStrategy::StructureAware {
                max_chunk_tokens,
                overlap_tokens,
            } => Self::structure_aware(doc, *max_chunk_tokens, *overlap_tokens, doc_id),
            ChunkingStrategy::FixedSize {
                chunk_size_tokens,
                overlap_tokens,
            } => Self::fixed_size(doc, *chunk_size_tokens, *overlap_tokens, doc_id),
            ChunkingStrategy::Semantic {
                max_chunk_tokens, ..
            } => {
                // Semantic splitting falls back to structure-aware for v1.
                // Topic-shift detection via embedding similarity is added in a later iteration.
                Self::structure_aware(doc, *max_chunk_tokens, 50, doc_id)
            }
        }
    }

    fn structure_aware(
        doc: &ParsedDocument,
        max_tokens: usize,
        overlap_tokens: usize,
        doc_id: &str,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut chunk_idx = 0usize;

        for section in &doc.sections {
            Self::process_section(
                section,
                &[],
                max_tokens,
                overlap_tokens,
                doc_id,
                &mut chunk_idx,
                &mut chunks,
            );
        }

        chunks
    }

    fn process_section(
        section: &Section,
        parent_path: &[String],
        max_tokens: usize,
        overlap_tokens: usize,
        doc_id: &str,
        chunk_idx: &mut usize,
        output: &mut Vec<Chunk>,
    ) {
        let mut path = parent_path.to_vec();
        if let Some(h) = &section.heading {
            path.push(h.clone());
        }

        let section_tokens = estimate_tokens(&section.content);

        if section_tokens <= max_tokens {
            // Whole section fits: emit as a single chunk.
            if !section.content.trim().is_empty() {
                output.push(Chunk {
                    id: format!("{doc_id}:{}", *chunk_idx),
                    text: section.content.clone(),
                    path: path.clone(),
                    page: section.page,
                    char_span: Some(section.char_span),
                    token_count: section_tokens,
                });
                *chunk_idx += 1;
            }
        } else {
            // Section is too large: split at paragraph boundaries.
            let paragraphs: Vec<&str> = section.content.split("\n\n").collect();
            let mut buffer = String::new();
            let mut buffer_tokens = 0;

            for para in paragraphs {
                let para_tokens = estimate_tokens(para);

                if buffer_tokens + para_tokens > max_tokens && !buffer.is_empty() {
                    output.push(Chunk {
                        id: format!("{doc_id}:{}", *chunk_idx),
                        text: buffer.trim().to_string(),
                        path: path.clone(),
                        page: section.page,
                        char_span: None,
                        token_count: buffer_tokens,
                    });
                    *chunk_idx += 1;

                    // Carry over the overlap from the previous chunk.
                    let words: Vec<&str> = buffer.split_whitespace().collect();
                    let overlap_words = words
                        .iter()
                        .rev()
                        .take(overlap_tokens)
                        .rev()
                        .cloned()
                        .collect::<Vec<_>>();
                    buffer = overlap_words.join(" ");
                    buffer_tokens = estimate_tokens(&buffer);
                }

                if !buffer.is_empty() {
                    buffer.push_str("\n\n");
                }
                buffer.push_str(para);
                buffer_tokens += para_tokens;
            }

            if !buffer.trim().is_empty() {
                output.push(Chunk {
                    id: format!("{doc_id}:{}", *chunk_idx),
                    text: buffer.trim().to_string(),
                    path: path.clone(),
                    page: section.page,
                    char_span: None,
                    token_count: buffer_tokens,
                });
                *chunk_idx += 1;
            }
        }

        for child in &section.children {
            Self::process_section(
                child,
                &path,
                max_tokens,
                overlap_tokens,
                doc_id,
                chunk_idx,
                output,
            );
        }
    }

    fn fixed_size(
        doc: &ParsedDocument,
        chunk_size: usize,
        overlap_tokens: usize,
        doc_id: &str,
    ) -> Vec<Chunk> {
        let full_text: String = doc
            .sections
            .iter()
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let words: Vec<&str> = full_text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut chunk_idx = 0usize;

        while start < words.len() {
            let end = (start + chunk_size * 4).min(words.len());
            let text = words[start..end].join(" ");
            let token_count = estimate_tokens(&text);

            chunks.push(Chunk {
                id: format!("{doc_id}:{chunk_idx}"),
                text,
                path: vec![],
                page: None,
                char_span: None,
                token_count,
            });

            chunk_idx += 1;
            start = end.saturating_sub(overlap_tokens * 4);
            if start >= end {
                break;
            }
        }

        chunks
    }
}

/// Result returned from the ingestion pipeline after processing one document.
#[derive(Debug, Clone)]
pub struct IngestionResult {
    pub doc_id: String,
    pub doc_version_hash: String,
    pub chunk_count: usize,
    pub total_tokens: usize,
    pub ingested_at: chrono::DateTime<Utc>,
}

/// Enriches a chunk with full ChunkMetadata for storage in all indexes.
pub fn build_chunk_metadata(
    chunk: &Chunk,
    doc: &Document,
    doc_version_hash: &str,
    index_generation: u64,
) -> ChunkMetadata {
    ChunkMetadata {
        doc_id: doc.id.clone(),
        source_doc: doc
            .metadata
            .filename
            .clone()
            .unwrap_or_else(|| doc.id.clone()),
        doc_version_hash: doc_version_hash.to_string(),
        path: chunk.path.clone(),
        page: chunk.page,
        char_span: chunk.char_span,
        ingested_at: Utc::now(),
        index_generation,
        acl_groups: doc.acl_groups.clone(),
        token_count: chunk.token_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Document, DocumentMetadata};

    fn make_section(heading: &str, content: &str, children: Vec<Section>) -> Section {
        Section {
            heading: Some(heading.into()),
            level: 1,
            content: content.into(),
            children,
            page: None,
            char_span: [0, content.len()],
        }
    }

    fn make_doc(sections: Vec<Section>) -> ParsedDocument {
        ParsedDocument {
            sections,
            title: Some("Test Document".into()),
            metadata: DocumentMetadata::default(),
        }
    }

    #[test]
    fn structure_aware_chunker_respects_token_limit() {
        let long_content = "word ".repeat(300) + "\n\n" + &"word ".repeat(300);
        let doc = make_doc(vec![make_section("Section 1", &long_content, vec![])]);
        let strategy = ChunkingStrategy::StructureAware {
            max_chunk_tokens: 512,
            overlap_tokens: 50,
        };
        let chunks = Chunker::chunk(&doc, &strategy, "doc1");
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                chunk.token_count <= 512,
                "chunk token count {} exceeds reasonable limit",
                chunk.token_count
            );
        }
    }

    #[test]
    fn structure_aware_chunker_preserves_path() {
        let doc = make_doc(vec![make_section(
            "Article 4",
            "Short section content.",
            vec![make_section("4.1 Term", "Sub section content.", vec![])],
        )]);
        let strategy = ChunkingStrategy::default();
        let chunks = Chunker::chunk(&doc, &strategy, "doc1");
        let has_nested_path = chunks
            .iter()
            .any(|c| c.path.contains(&"4.1 Term".to_string()));
        assert!(
            has_nested_path,
            "nested section path not preserved in chunks"
        );
    }

    #[test]
    fn doc_version_hash_changes_with_content() {
        let h1 = content_hash("original content");
        let h2 = content_hash("modified content");
        assert_ne!(h1, h2);
        assert!(h1.starts_with("sha256:"));
    }

    #[test]
    fn acl_groups_propagate_from_document_to_chunk_metadata() {
        let doc = Document {
            id: "doc1".into(),
            content: "test content".into(),
            metadata: DocumentMetadata::default(),
            acl_groups: vec!["legal".into(), "finance".into()],
        };
        let chunk = Chunk {
            id: "doc1:0".into(),
            text: "test".into(),
            path: vec![],
            page: None,
            char_span: None,
            token_count: 4,
        };
        let meta = build_chunk_metadata(&chunk, &doc, "sha256:abc", 1);
        assert_eq!(meta.acl_groups, vec!["legal", "finance"]);
    }

    #[test]
    fn estimate_tokens_is_approximately_correct() {
        let text = "a".repeat(400);
        assert_eq!(estimate_tokens(&text), 100);
    }
}
