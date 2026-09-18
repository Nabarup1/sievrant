use std::collections::HashMap;

use async_trait::async_trait;
use qdrant_client::{
    qdrant::{
        vectors_config::Config, with_payload_selector::SelectorOptions, Condition,
        CreateCollection, Distance, FieldCondition, Filter, Match, PointId, PointStruct,
        SearchPoints, VectorParams, VectorsConfig, WithPayloadSelector,
    },
    Qdrant,
};

use super::{IndexChunk, VectorStore};
use sievrant_core::{
    error::{IndexError, Result},
    types::{ChunkMetadata, RankedCandidate, RetrieverType},
};

pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
}

impl QdrantStore {
    pub async fn new(
        url: String,
        _api_key: Option<String>,
        collection_name: String,
        dim: usize,
    ) -> Result<Self> {
        let client = Qdrant::from_url(&url)
            .build()
            .map_err(|e| IndexError::Vector(e.to_string()))?;

        let exists = client
            .collection_exists(&collection_name)
            .await
            .map_err(|e| IndexError::Vector(e.to_string()))?;

        if !exists {
            client
                .create_collection(CreateCollection {
                    collection_name: collection_name.clone(),
                    vectors_config: Some(VectorsConfig {
                        config: Some(Config::Params(VectorParams {
                            size: dim as u64,
                            distance: Distance::Cosine.into(),
                            ..Default::default()
                        })),
                    }),
                    ..Default::default()
                })
                .await
                .map_err(|e| IndexError::Vector(e.to_string()))?;
        }

        Ok(Self {
            client,
            collection_name,
        })
    }

    fn build_acl_filter(allowed_groups: &[String]) -> Option<Filter> {
        if allowed_groups.is_empty() {
            return None;
        }

        let conditions: Vec<Condition> = allowed_groups
            .iter()
            .map(|group| Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    FieldCondition {
                        key: "acl_groups".into(),
                        r#match: Some(Match {
                            match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                group.clone(),
                            )),
                        }),
                        ..Default::default()
                    },
                )),
            })
            .collect();

        Some(Filter {
            should: conditions,
            ..Default::default()
        })
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn upsert_vectors(&self, chunks: &[IndexChunk]) -> Result<()> {
        let points: Vec<PointStruct> = chunks
            .iter()
            .filter_map(|chunk| {
                let embedding = chunk.embedding.as_ref()?;
                let mut payload: HashMap<String, qdrant_client::qdrant::Value> = HashMap::new();

                let insert_str = |m: &mut HashMap<String, qdrant_client::qdrant::Value>,
                                  key: &str,
                                  val: &str| {
                    m.insert(
                        key.to_string(),
                        qdrant_client::qdrant::Value {
                            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                                val.to_string(),
                            )),
                        },
                    );
                };

                insert_str(&mut payload, "chunk_id", &chunk.chunk_id);
                insert_str(&mut payload, "text", &chunk.text);
                insert_str(&mut payload, "doc_id", &chunk.metadata.doc_id);
                insert_str(&mut payload, "source_doc", &chunk.metadata.source_doc);
                insert_str(
                    &mut payload,
                    "doc_version_hash",
                    &chunk.metadata.doc_version_hash,
                );

                let acl_list: qdrant_client::qdrant::Value = qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
                        qdrant_client::qdrant::ListValue {
                            values: chunk
                                .metadata
                                .acl_groups
                                .iter()
                                .map(|g| qdrant_client::qdrant::Value {
                                    kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                                        g.clone(),
                                    )),
                                })
                                .collect(),
                        },
                    )),
                };
                payload.insert("acl_groups".to_string(), acl_list);

                payload.insert(
                    "token_count".to_string(),
                    qdrant_client::qdrant::Value {
                        kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(
                            chunk.metadata.token_count as i64,
                        )),
                    },
                );

                payload.insert(
                    "index_generation".to_string(),
                    qdrant_client::qdrant::Value {
                        kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(
                            chunk.metadata.index_generation as i64,
                        )),
                    },
                );

                Some(PointStruct {
                    id: Some(PointId {
                        point_id_options: Some(
                            qdrant_client::qdrant::point_id::PointIdOptions::Uuid(
                                chunk.chunk_id.clone(),
                            ),
                        ),
                    }),
                    vectors: Some(embedding.clone().into()),
                    payload,
                })
            })
            .collect();

        if !points.is_empty() {
            let req =
                qdrant_client::qdrant::UpsertPointsBuilder::new(&self.collection_name, points);
            self.client
                .upsert_points(req)
                .await
                .map_err(|e| IndexError::Vector(e.to_string()))?;
        }

        Ok(())
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        acl_filter: &[String],
    ) -> Result<Vec<RankedCandidate>> {
        let filter = Self::build_acl_filter(acl_filter);

        let results = self
            .client
            .search_points(SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: query_embedding.to_vec(),
                filter,
                limit: top_k as u64,
                with_payload: Some(WithPayloadSelector {
                    selector_options: Some(SelectorOptions::Enable(true)),
                }),
                params: None,
                score_threshold: None,
                offset: None,
                vector_name: None,
                with_vectors: None,
                read_consistency: None,
                shard_key_selector: None,
                sparse_indices: None,
                timeout: None,
            })
            .await
            .map_err(|e| IndexError::Vector(e.to_string()))?;

        let candidates = results
            .result
            .into_iter()
            .filter_map(|point| {
                let payload = &point.payload;
                let chunk_id = payload.get("chunk_id")?.as_str()?.to_string();
                let text = payload.get("text")?.as_str()?.to_string();
                let doc_id = payload
                    .get("doc_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let source_doc = payload
                    .get("source_doc")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let acl_groups: Vec<String> = payload
                    .get("acl_groups")
                    .and_then(|v| v.as_list())
                    .map(|list| {
                        list.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                let metadata = ChunkMetadata {
                    doc_id,
                    source_doc,
                    doc_version_hash: payload
                        .get("doc_version_hash")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default(),
                    path: vec![],
                    page: None,
                    char_span: None,
                    ingested_at: chrono::Utc::now(),
                    index_generation: payload
                        .get("index_generation")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0) as u64,
                    acl_groups,
                    token_count: payload
                        .get("token_count")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0) as usize,
                };

                Some(RankedCandidate {
                    chunk_id,
                    score: point.score as f64,
                    retriever: RetrieverType::VectorIndex,
                    text,
                    metadata,
                })
            })
            .collect();

        Ok(candidates)
    }

    async fn delete_documents(&self, chunk_ids: &[String]) -> Result<()> {
        let ids: Vec<PointId> = chunk_ids
            .iter()
            .map(|id| PointId {
                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(
                    id.clone(),
                )),
            })
            .collect();

        let req = qdrant_client::qdrant::DeletePointsBuilder::new(&self.collection_name).points(
            qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Points(
                qdrant_client::qdrant::PointsIdsList { ids },
            ),
        );

        self.client
            .delete_points(req)
            .await
            .map_err(|e| IndexError::Vector(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acl_filter_with_no_groups_returns_none() {
        assert!(QdrantStore::build_acl_filter(&[]).is_none());
    }

    #[test]
    fn acl_filter_with_groups_builds_should_filter() {
        let filter = QdrantStore::build_acl_filter(&["legal".into(), "finance".into()]);
        assert!(filter.is_some());
        let f = filter.unwrap();
        assert_eq!(f.should.len(), 2);
    }
}
