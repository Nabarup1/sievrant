use std::collections::HashMap;

use sha2::{Digest, Sha256};
use sievrant_core::error::Result;
use sievrant_core::types::{
    EvidenceChunk, QueryId, RankedCandidate, RetrieverType, Route, SelectedEvidence,
};
use sievrant_index::vector::IndexQuery;
use sievrant_llm::{
    provider::{LlmRequest, Message, Role},
    LlmProvider,
};
use sievrant_query::processing::ProcessedQuery;

use crate::pipeline::IndexSet;

/// Execute the index searches in parallel. Tree and graph indexes are only
/// queried when configured. A failing index records a warning and contributes
/// an empty list; the pipeline continues with whatever indexes succeed.
pub async fn parallel_index_search(
    indexes: &IndexSet,
    query: &ProcessedQuery,
    acl_filter: &[String],
    limit: usize,
) -> (Vec<Vec<RankedCandidate>>, Vec<String>) {
    let vec_query = IndexQuery::Vector {
        embedding: query.embedding.clone(),
    };
    let bm25_query = IndexQuery::Lexical {
        text: query.original.clone(),
    };
    let tree_query = IndexQuery::Structural {
        text: query.original.clone(),
        path_hints: vec![],
    };
    let graph_query = IndexQuery::Graph {
        entities: query.entities.clone(),
        query_embedding: query.embedding.clone(),
    };

    let optional = |idx: &Option<std::sync::Arc<dyn sievrant_index::RetrievalIndex>>,
                    q: IndexQuery| {
        let idx = idx.clone();
        async move {
            match idx {
                Some(i) => Some(i.search(&q, acl_filter, limit).await),
                None => None,
            }
        }
    };

    let (vec_r, bm25_r, tree_r, graph_r) = tokio::join!(
        indexes.vector.search(&vec_query, acl_filter, limit),
        indexes.bm25.search(&bm25_query, acl_filter, limit),
        optional(&indexes.tree, tree_query),
        optional(&indexes.graph, graph_query),
    );

    let mut ranked_lists = Vec::with_capacity(4);
    let mut warnings = Vec::new();

    for (name, result) in [
        ("vector", Some(vec_r)),
        ("bm25", Some(bm25_r)),
        ("tree", tree_r),
        ("graph", graph_r),
    ] {
        let Some(result) = result else { continue };
        match result {
            Ok(candidates) => ranked_lists.push(candidates),
            Err(e) => warnings.push(format!("{name} index failed (degraded): {e}")),
        }
    }

    (ranked_lists, warnings)
}

pub fn compute_session_hash(
    query_id: &QueryId,
    query: &str,
    evidence: &[EvidenceChunk],
    answer: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(query_id.0.as_bytes());
    hasher.update(query.as_bytes());
    for chunk in evidence {
        hasher.update(chunk.chunk_id.as_bytes());
    }
    if let Some(ans) = answer {
        hasher.update(ans.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

pub fn route_into_packet_route(route: &Route) -> sievrant_core::types::Route {
    match route {
        Route::Fast => sievrant_core::types::Route::Fast,
        Route::Standard => sievrant_core::types::Route::Standard,
        Route::Slow => sievrant_core::types::Route::Slow,
        Route::Cached => sievrant_core::types::Route::Cached,
    }
}

pub async fn generate_answer(
    llm: &dyn LlmProvider,
    query: &str,
    evidence: &[SelectedEvidence],
    max_tokens: usize,
) -> Result<String> {
    let evidence_text: String = evidence
        .iter()
        .map(|e| {
            format!(
                "[{}] (Source: {}, Path: {})\n{}",
                e.chunk_id,
                e.metadata.source_doc,
                e.metadata.path.join(" > "),
                e.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let system = "You answer questions strictly from the evidence passages you are given. \
        Cite every claim with the passage id in square brackets, e.g. [doc:0:1]. \
        If the passages do not contain the answer, say that the provided evidence is \
        insufficient instead of guessing. Keep answers concise.";
    let prompt = format!("Question: {query}\n\nEvidence:\n{evidence_text}");

    let req = LlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: prompt,
        }],
        max_tokens,
        temperature: 0.0,
        system_prompt: Some(system.to_string()),
    };

    let response = llm.complete(req).await?;
    Ok(response.content.trim().to_string())
}

pub fn map_evidence_chunks(
    evidence: Vec<SelectedEvidence>,
    retrievers: &HashMap<String, RetrieverType>,
) -> Vec<EvidenceChunk> {
    let now = chrono::Utc::now();
    evidence
        .into_iter()
        .map(|e| EvidenceChunk {
            retriever: retrievers
                .get(&e.chunk_id)
                .cloned()
                .unwrap_or(RetrieverType::VectorIndex),
            freshness_hours: (now - e.metadata.ingested_at).num_seconds().max(0) as f64 / 3600.0,
            chunk_id: e.chunk_id,
            source_doc: e.metadata.source_doc,
            doc_version_hash: e.metadata.doc_version_hash,
            path: e.metadata.path,
            page: e.metadata.page,
            char_span: e.metadata.char_span,
            text: e.text,
            rrf_score: e.fused_score,
            rerank_score: Some(e.rerank_score),
            nonconformity_score: e.nonconformity_score,
            submodular_marginal_gain: e.marginal_gain,
            ingested_at: e.metadata.ingested_at,
            index_generation: e.metadata.index_generation,
            acl_groups: e.metadata.acl_groups,
            token_count: e.metadata.token_count,
        })
        .collect()
}
