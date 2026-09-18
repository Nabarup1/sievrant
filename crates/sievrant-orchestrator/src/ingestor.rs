use std::sync::Arc;

use sievrant_core::{
    config::SievrantConfig,
    error::Result,
    ingestion::{
        build_chunk_metadata, content_hash, Chunk, Chunker, ChunkingStrategy, IngestionResult,
        ParsedDocument, Section,
    },
    types::Document,
};

/// Drives the first four ingestion stages: parse, chunk, enrich metadata, tag ACLs.
/// Stages 5 (WAL write) and 6 (index distribution) are handled by the CorpusManager
/// implemented in prompt 20, which calls into the WAL and index crates respectively.
pub struct IngestionPipeline {
    _config: Arc<SievrantConfig>,
}

impl IngestionPipeline {
    pub fn new(config: Arc<SievrantConfig>) -> Self {
        Self { _config: config }
    }

    /// Ingest a single document. Returns an IngestionResult with chunk count and hash.
    pub async fn ingest(
        &self,
        doc: Document,
        _corpus: &str,
    ) -> Result<(IngestionResult, Vec<EnrichedChunk>)> {
        let doc_version_hash = content_hash(&doc.content);

        let parsed = self.parse_document(&doc);

        let strategy = ChunkingStrategy::default();
        let chunks = Chunker::chunk(&parsed, &strategy, &doc.id);

        let total_tokens: usize = chunks.iter().map(|c| c.token_count).sum();
        let chunk_count = chunks.len();

        let enriched: Vec<EnrichedChunk> = chunks
            .iter()
            .map(|chunk| {
                let metadata = build_chunk_metadata(chunk, &doc, &doc_version_hash, 0);
                EnrichedChunk {
                    chunk: chunk.clone(),
                    metadata,
                    acl_groups: doc.acl_groups.clone(),
                }
            })
            .collect();

        let result = IngestionResult {
            doc_id: doc.id.clone(),
            doc_version_hash,
            chunk_count,
            total_tokens,
            ingested_at: chrono::Utc::now(),
        };

        Ok((result, enriched))
    }

    /// Ingest a batch of documents sequentially. Each document is independent.
    pub async fn ingest_batch(
        &self,
        docs: Vec<Document>,
        corpus: &str,
    ) -> Result<Vec<(IngestionResult, Vec<EnrichedChunk>)>> {
        let mut results = Vec::with_capacity(docs.len());
        for doc in docs {
            results.push(self.ingest(doc, corpus).await?);
        }
        Ok(results)
    }

    /// Parse document content into a section-based structure.
    /// For v1 this is a plain text parser that detects Markdown headings.
    /// Future implementations can add PDF, DOCX, and HTML parsers behind the DocumentParser trait.
    fn parse_document(&self, doc: &Document) -> ParsedDocument {
        let mut sections: Vec<Section> = Vec::new();
        let mut current_heading: Option<String> = None;
        let mut current_content = String::new();
        let mut current_level = 1u32;
        let mut char_offset = 0usize;
        let mut section_start = 0usize;

        for line in doc.content.lines() {
            let line_len = line.len() + 1;

            if let Some(stripped) = line.strip_prefix("### ") {
                Self::flush_section(
                    &mut sections,
                    &mut current_heading,
                    &mut current_content,
                    current_level,
                    section_start,
                    char_offset,
                );
                current_heading = Some(stripped.trim().to_string());
                current_level = 3;
                section_start = char_offset;
            } else if let Some(stripped) = line.strip_prefix("## ") {
                Self::flush_section(
                    &mut sections,
                    &mut current_heading,
                    &mut current_content,
                    current_level,
                    section_start,
                    char_offset,
                );
                current_heading = Some(stripped.trim().to_string());
                current_level = 2;
                section_start = char_offset;
            } else if let Some(stripped) = line.strip_prefix("# ") {
                Self::flush_section(
                    &mut sections,
                    &mut current_heading,
                    &mut current_content,
                    current_level,
                    section_start,
                    char_offset,
                );
                current_heading = Some(stripped.trim().to_string());
                current_level = 1;
                section_start = char_offset;
            } else {
                if !current_content.is_empty() {
                    current_content.push('\n');
                }
                current_content.push_str(line);
            }

            char_offset += line_len;
        }

        Self::flush_section(
            &mut sections,
            &mut current_heading,
            &mut current_content,
            current_level,
            section_start,
            char_offset,
        );

        ParsedDocument {
            sections,
            title: doc.metadata.title.clone(),
            metadata: doc.metadata.clone(),
        }
    }

    fn flush_section(
        sections: &mut Vec<Section>,
        heading: &mut Option<String>,
        content: &mut String,
        level: u32,
        start: usize,
        end: usize,
    ) {
        if !content.trim().is_empty() || heading.is_some() {
            sections.push(Section {
                heading: heading.take(),
                level,
                content: content.trim().to_string(),
                children: vec![],
                page: None,
                char_span: [start, end],
            });
        }
        *content = String::new();
    }
}

/// A chunk with its fully computed metadata and ACL groups, ready for the WAL and indexes.
pub struct EnrichedChunk {
    pub chunk: Chunk,
    pub metadata: sievrant_core::types::ChunkMetadata,
    pub acl_groups: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::types::{Document, DocumentMetadata};
    use std::sync::Arc;

    fn make_pipeline() -> IngestionPipeline {
        IngestionPipeline::new(Arc::new(SievrantConfig::default()))
    }

    fn make_doc(content: &str, acl_groups: Vec<String>) -> Document {
        Document {
            id: "doc1".into(),
            content: content.into(),
            metadata: DocumentMetadata::default(),
            acl_groups,
        }
    }

    #[tokio::test]
    async fn ingestion_produces_chunks_with_correct_metadata() {
        let pipeline = make_pipeline();
        let doc = make_doc(
            "# Section 1\n\nSome content here.\n\n## Section 2\n\nMore content.",
            vec!["public".into()],
        );
        let (result, enriched) = pipeline.ingest(doc, "test_corpus").await.unwrap();
        assert!(result.chunk_count > 0);
        assert!(result.doc_version_hash.starts_with("sha256:"));
        for ec in &enriched {
            assert!(!ec.chunk.text.is_empty());
            assert_eq!(ec.acl_groups, vec!["public"]);
        }
    }

    #[tokio::test]
    async fn content_hash_changes_when_document_changes() {
        let pipeline = make_pipeline();
        let doc1 = make_doc("original", vec![]);
        let doc2 = make_doc("modified", vec![]);
        let (r1, _) = pipeline.ingest(doc1, "test").await.unwrap();
        let (r2, _) = pipeline.ingest(doc2, "test").await.unwrap();
        assert_ne!(r1.doc_version_hash, r2.doc_version_hash);
    }
}
