pub mod mock_data {
    use crate::types::EvidenceChunk;

    pub fn build_mock_chunk(id: &str) -> EvidenceChunk {
        EvidenceChunk {
            chunk_id: id.to_string(),
            source_doc: "doc1".to_string(),
            doc_version_hash: "hash".to_string(),
            path: vec![],
            page: None,
            char_span: None,
            text: "Mock text".to_string(),
            retriever: crate::types::RetrieverType::VectorIndex,
            rrf_score: 0.9,
            rerank_score: Some(0.9),
            nonconformity_score: 0.1,
            submodular_marginal_gain: 0.1,
            ingested_at: chrono::Utc::now(),
            index_generation: 1,
            acl_groups: vec![],
            freshness_hours: 0.0,
            token_count: 10,
        }
    }
}
