use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use tantivy::{
    collector::TopDocs,
    directory::MmapDirectory,
    query::{BooleanQuery, Occur, QueryParser, TermQuery},
    schema::{
        Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, FAST, STORED,
        STRING,
    },
    Index, IndexWriter, ReloadPolicy, TantivyDocument, Term,
};

use sievrant_core::{
    error::{IndexError, Result},
    types::{ChunkMetadata, RankedCandidate, RetrieverType},
};

use crate::vector::{IndexChunk, IndexQuery, RetrievalIndex};

#[derive(Debug, Clone)]
pub struct Bm25IndexConfig {
    pub index_dir: PathBuf,
    /// Tantivy writer heap size in bytes.
    pub heap_size_bytes: usize,
    pub commit_interval_seconds: u64,
    pub num_threads: usize,
}

impl Default for Bm25IndexConfig {
    fn default() -> Self {
        Self {
            index_dir: PathBuf::from("data/bm25"),
            heap_size_bytes: 50 * 1024 * 1024,
            commit_interval_seconds: 30,
            num_threads: 1,
        }
    }
}

/// Field names used in the Tantivy schema.
struct BM25Fields {
    chunk_id: Field,
    text: Field,
    doc_id: Field,
    source_doc: Field,
    doc_version_hash: Field,
    acl_group: Field,
    token_count: Field,
    index_generation: Field,
}

fn build_schema() -> (Schema, BM25Fields) {
    let mut builder = Schema::builder();

    let text_options = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("en_stem")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );

    let fields = BM25Fields {
        chunk_id: builder.add_text_field("chunk_id", STRING | STORED),
        text: builder.add_text_field("text", text_options | STORED),
        doc_id: builder.add_text_field("doc_id", STRING | STORED),
        source_doc: builder.add_text_field("source_doc", STRING | STORED),
        doc_version_hash: builder.add_text_field("doc_version_hash", STRING | STORED),
        acl_group: builder.add_text_field("acl_group", STRING),
        token_count: builder.add_u64_field("token_count", STORED | FAST),
        index_generation: builder.add_u64_field("index_generation", STORED | FAST),
    };

    (builder.build(), fields)
}

pub struct Bm25Index {
    index: Index,
    writer: Arc<RwLock<IndexWriter>>,
    fields: BM25Fields,
    _schema: Schema,
}

impl Bm25Index {
    pub fn new(config: Bm25IndexConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.index_dir).map_err(|e| IndexError::Bm25(e.to_string()))?;

        let (schema, fields) = build_schema();

        let dir =
            MmapDirectory::open(&config.index_dir).map_err(|e| IndexError::Bm25(e.to_string()))?;

        let index = Index::open_or_create(dir, schema.clone())
            .map_err(|e| IndexError::Bm25(e.to_string()))?;

        let writer = index
            .writer_with_num_threads(config.num_threads, config.heap_size_bytes)
            .map_err(|e| IndexError::Bm25(e.to_string()))?;

        Ok(Self {
            index,
            writer: Arc::new(RwLock::new(writer)),
            fields,
            _schema: schema,
        })
    }

    pub fn commit(&self) -> Result<()> {
        self.writer
            .write()
            .unwrap()
            .commit()
            .map_err(|e| IndexError::Bm25(e.to_string()))?;
        Ok(())
    }

    /// Build an ACL query clause: any chunk whose acl_group field matches any of the allowed groups.
    fn acl_query(&self, allowed_groups: &[String]) -> Box<dyn tantivy::query::Query> {
        let sub_queries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = allowed_groups
            .iter()
            .map(|group| {
                let term = Term::from_field_text(self.fields.acl_group, group);
                let q: Box<dyn tantivy::query::Query> =
                    Box::new(TermQuery::new(term, IndexRecordOption::Basic));
                (Occur::Should, q)
            })
            .collect();

        Box::new(BooleanQuery::new(sub_queries))
    }
}

#[async_trait]
impl RetrievalIndex for Bm25Index {
    async fn search(
        &self,
        query: &IndexQuery,
        acl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<RankedCandidate>> {
        let text = match query {
            IndexQuery::Lexical { text } => text.clone(),
            IndexQuery::Structural { text, .. } => text.clone(),
            _ => return Ok(vec![]),
        };

        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| IndexError::Bm25(e.to_string()))?;

        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.fields.text]);

        let sanitized_text: String = text
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c.is_whitespace() {
                    c
                } else {
                    ' '
                }
            })
            .collect();

        let text_query = query_parser
            .parse_query(&sanitized_text)
            .unwrap_or_else(|_| Box::new(tantivy::query::BooleanQuery::new(vec![])));

        let combined: Box<dyn tantivy::query::Query> = if acl_filter.is_empty() {
            text_query
        } else {
            let acl_q = self.acl_query(acl_filter);
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, text_query),
                (Occur::Must, acl_q),
            ]))
        };

        let top_docs = searcher
            .search(&combined, &TopDocs::with_limit(limit))
            .map_err(|e| IndexError::Bm25(e.to_string()))?;

        let candidates = top_docs
            .into_iter()
            .filter_map(|(score, doc_addr)| {
                let doc: TantivyDocument = searcher.doc(doc_addr).ok()?;
                let get_str = |field: Field| -> String {
                    doc.get_first(field)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                };

                let chunk_id = get_str(self.fields.chunk_id);
                let text = get_str(self.fields.text);
                let doc_id = get_str(self.fields.doc_id);
                let source_doc = get_str(self.fields.source_doc);
                let doc_version_hash = get_str(self.fields.doc_version_hash);
                let token_count = doc
                    .get_first(self.fields.token_count)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let index_generation = doc
                    .get_first(self.fields.index_generation)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                Some(RankedCandidate {
                    chunk_id,
                    score: score as f64,
                    retriever: RetrieverType::Bm25Index,
                    text,
                    metadata: ChunkMetadata {
                        doc_id,
                        source_doc,
                        doc_version_hash,
                        path: vec![],
                        page: None,
                        char_span: None,
                        ingested_at: chrono::Utc::now(),
                        index_generation,
                        acl_groups: vec![],
                        token_count,
                    },
                })
            })
            .collect();

        Ok(candidates)
    }

    async fn insert(&self, chunks: &[IndexChunk]) -> Result<()> {
        let mut writer = self.writer.write().unwrap();

        for chunk in chunks {
            let mut doc = TantivyDocument::default();
            doc.add_text(self.fields.chunk_id, &chunk.chunk_id);
            doc.add_text(self.fields.text, &chunk.text);
            doc.add_text(self.fields.doc_id, &chunk.metadata.doc_id);
            doc.add_text(self.fields.source_doc, &chunk.metadata.source_doc);
            doc.add_text(
                self.fields.doc_version_hash,
                &chunk.metadata.doc_version_hash,
            );
            doc.add_u64(self.fields.token_count, chunk.metadata.token_count as u64);
            doc.add_u64(
                self.fields.index_generation,
                chunk.metadata.index_generation,
            );

            for group in &chunk.metadata.acl_groups {
                doc.add_text(self.fields.acl_group, group);
            }

            writer
                .add_document(doc)
                .map_err(|e| IndexError::Bm25(e.to_string()))?;
        }

        // Commit so the batch is durable and visible to new searchers.
        writer
            .commit()
            .map_err(|e| IndexError::Bm25(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, chunk_ids: &[String]) -> Result<()> {
        let mut writer = self.writer.write().unwrap();
        for chunk_id in chunk_ids {
            let term = Term::from_field_text(self.fields.chunk_id, chunk_id);
            writer.delete_term(term);
        }
        writer
            .commit()
            .map_err(|e| IndexError::Bm25(e.to_string()))?;
        Ok(())
    }

    fn index_type(&self) -> RetrieverType {
        RetrieverType::Bm25Index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::types::ChunkMetadata;
    use tempfile::TempDir;

    fn make_config(dir: &TempDir) -> Bm25IndexConfig {
        Bm25IndexConfig {
            index_dir: dir.path().to_path_buf(),
            ..Default::default()
        }
    }

    fn make_chunk(id: &str, text: &str, acl_groups: Vec<String>) -> IndexChunk {
        IndexChunk {
            chunk_id: id.into(),
            text: text.into(),
            embedding: None,
            metadata: ChunkMetadata {
                doc_id: "doc1".into(),
                source_doc: "test.pdf".into(),
                doc_version_hash: "sha256:abc".into(),
                path: vec![],
                page: None,
                char_span: None,
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups,
                token_count: 10,
            },
        }
    }

    #[tokio::test]
    async fn indexes_and_retrieves_by_keyword() {
        let dir = TempDir::new().unwrap();
        let idx = Bm25Index::new(make_config(&dir)).unwrap();
        let chunk = make_chunk(
            "c1",
            "indemnification clause limits liability",
            vec!["public".into()],
        );
        idx.insert(&[chunk]).await.unwrap();
        idx.commit().unwrap();

        let results = idx
            .search(
                &IndexQuery::Lexical {
                    text: "indemnification".into(),
                },
                &["public".into()],
                10,
            )
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "c1");
    }

    #[tokio::test]
    async fn acl_filter_excludes_unauthorized_chunks() {
        let dir = TempDir::new().unwrap();
        let idx = Bm25Index::new(make_config(&dir)).unwrap();
        let restricted = make_chunk(
            "c_restricted",
            "confidential payment terms",
            vec!["legal".into()],
        );
        let public = make_chunk(
            "c_public",
            "payment schedule overview",
            vec!["public".into()],
        );
        idx.insert(&[restricted, public]).await.unwrap();
        idx.commit().unwrap();

        let results = idx
            .search(
                &IndexQuery::Lexical {
                    text: "payment".into(),
                },
                &["public".into()],
                10,
            )
            .await
            .unwrap();

        assert!(results.iter().all(|r| r.chunk_id != "c_restricted"));
        assert!(results.iter().any(|r| r.chunk_id == "c_public"));
    }

    #[tokio::test]
    async fn deletion_removes_chunk_from_results() {
        let dir = TempDir::new().unwrap();
        let idx = Bm25Index::new(make_config(&dir)).unwrap();
        let chunk = make_chunk("c_del", "lease termination clause", vec!["public".into()]);
        idx.insert(&[chunk]).await.unwrap();
        idx.commit().unwrap();

        idx.delete(&["c_del".into()]).await.unwrap();
        idx.commit().unwrap();

        let results = idx
            .search(
                &IndexQuery::Lexical {
                    text: "lease termination".into(),
                },
                &["public".into()],
                10,
            )
            .await
            .unwrap();

        assert!(results.iter().all(|r| r.chunk_id != "c_del"));
    }
}
