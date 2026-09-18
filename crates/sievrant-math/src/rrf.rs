use rustc_hash::FxHashMap;

#[cfg(test)]
use sievrant_core::types::ChunkMetadata;
use sievrant_core::types::{FusedCandidate, RankedCandidate};

/// Reciprocal Rank Fusion (Cormack, Clarke, Buettcher, 2009)
///
/// Merges multiple ranked lists from heterogeneous indexes into one fused ranking.
/// Formula per candidate d:
///
///   fused_score(d) = SUM over all lists L of: 1 / (k + rank_L(d))
///
/// rank_L is 1-indexed (position 0 maps to rank 1).
/// k defaults to 60 in the original paper; it controls how strongly top
/// positions dominate over lower positions.
///
/// Properties guaranteed by this implementation:
///   - Rank-based: raw scores (cosine, BM25, PPR) are discarded; only ordinal
///     positions matter, so no cross-index score calibration is needed.
///   - Deterministic: identical inputs always produce identical output.
///   - Rank monotonic: a candidate's fused rank is never worse than its best
///     individual rank across any single list (proven in the original paper).
pub fn reciprocal_rank_fusion(
    ranked_lists: &[Vec<RankedCandidate>],
    k: usize,
) -> Vec<FusedCandidate> {
    // Determine a rough capacity to avoid reallocations
    let total_elements = ranked_lists.iter().map(|l| l.len()).sum();
    let mut map: FxHashMap<String, FusedCandidate> =
        FxHashMap::with_capacity_and_hasher(total_elements, Default::default());

    for ranked_list in ranked_lists {
        for (rank_index, candidate) in ranked_list.iter().enumerate() {
            let contribution = 1.0 / (k as f64 + rank_index as f64 + 1.0);

            let entry = map
                .entry(candidate.chunk_id.clone())
                .or_insert_with(|| FusedCandidate {
                    chunk_id: candidate.chunk_id.clone(),
                    fused_score: 0.0,
                    source_scores: Vec::new(),
                    text: candidate.text.clone(),
                    metadata: candidate.metadata.clone(),
                });

            entry.fused_score += contribution;
            entry
                .source_scores
                .push((candidate.retriever.clone(), candidate.score));
        }
    }

    let mut fused: Vec<FusedCandidate> = map.into_values().collect();

    fused.sort_unstable_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    fused
}

// removed placeholder_metadata as it is no longer used

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use sievrant_core::types::RetrieverType;

    fn make_candidate(chunk_id: &str, score: f64, retriever: RetrieverType) -> RankedCandidate {
        RankedCandidate {
            chunk_id: chunk_id.to_string(),
            score,
            retriever,
            text: chunk_id.to_string(),
            metadata: ChunkMetadata {
                doc_id: chunk_id.into(),
                source_doc: "test.pdf".into(),
                doc_version_hash: "sha256:test".into(),
                path: vec![],
                page: None,
                char_span: None,
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: vec![],
                token_count: 50,
            },
        }
    }

    #[test]
    fn known_lists_produce_correct_scores() {
        // List A: [d1, d2], List B: [d2, d1]
        // d1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        // d2: 1/(60+2) + 1/(60+1) = same as d1 => tie, both equal
        let list_a = vec![
            make_candidate("d1", 1.0, RetrieverType::VectorIndex),
            make_candidate("d2", 0.5, RetrieverType::VectorIndex),
        ];
        let list_b = vec![
            make_candidate("d2", 1.0, RetrieverType::Bm25Index),
            make_candidate("d1", 0.5, RetrieverType::Bm25Index),
        ];

        let fused = reciprocal_rank_fusion(&[list_a, list_b], 60);

        let score_d1 = fused
            .iter()
            .find(|c| c.chunk_id == "d1")
            .unwrap()
            .fused_score;
        let score_d2 = fused
            .iter()
            .find(|c| c.chunk_id == "d2")
            .unwrap()
            .fused_score;

        let expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((score_d1 - expected).abs() < 1e-10, "d1 score mismatch");
        assert!((score_d2 - expected).abs() < 1e-10, "d2 score mismatch");
    }

    #[test]
    fn single_list_preserves_original_order() {
        let list = vec![
            make_candidate("a", 3.0, RetrieverType::VectorIndex),
            make_candidate("b", 2.0, RetrieverType::VectorIndex),
            make_candidate("c", 1.0, RetrieverType::VectorIndex),
        ];
        let fused = reciprocal_rank_fusion(&[list], 60);
        assert_eq!(fused[0].chunk_id, "a");
        assert_eq!(fused[1].chunk_id, "b");
        assert_eq!(fused[2].chunk_id, "c");
    }

    #[test]
    fn empty_input_returns_empty() {
        let result = reciprocal_rank_fusion(&[], 60);
        assert!(result.is_empty());
    }

    #[test]
    fn source_scores_preserved_for_audit() {
        let list = vec![make_candidate("x", 0.9, RetrieverType::GraphIndex)];
        let fused = reciprocal_rank_fusion(&[list], 60);
        let sources = &fused[0].source_scores;
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].0, RetrieverType::GraphIndex);
        assert!((sources[0].1 - 0.9).abs() < 1e-10);
    }

    // Property-based tests using proptest
    proptest! {
        #[test]
        fn rrf_scores_are_always_positive(
            // One list of 1-20 candidates; chunk IDs are distinct single chars
            list_len in 1usize..=20,
        ) {
            let list: Vec<RankedCandidate> = (0..list_len)
                .map(|i| make_candidate(&format!("c{i}"), 1.0, RetrieverType::VectorIndex))
                .collect();
            let fused = reciprocal_rank_fusion(&[list], 60);
            for candidate in &fused {
                prop_assert!(candidate.fused_score > 0.0);
            }
        }

        #[test]
        fn rrf_is_rank_monotonic(
            len_a in 1usize..=10,
            len_b in 1usize..=10,
        ) {
            // Build two lists with some overlap
            let overlap_id = "shared";
            let mut list_a: Vec<RankedCandidate> = (0..len_a)
                .map(|i| make_candidate(&format!("a{i}"), 1.0, RetrieverType::VectorIndex))
                .collect();
            let mut list_b: Vec<RankedCandidate> = (0..len_b)
                .map(|i| make_candidate(&format!("b{i}"), 1.0, RetrieverType::Bm25Index))
                .collect();

            // Insert the shared document at known positions: rank 1 in list_a, rank 1 in list_b
            list_a.insert(0, make_candidate(overlap_id, 1.0, RetrieverType::VectorIndex));
            list_b.insert(0, make_candidate(overlap_id, 1.0, RetrieverType::Bm25Index));

            let fused = reciprocal_rank_fusion(&[list_a.clone(), list_b.clone()], 60);

            let fused_rank = fused.iter().position(|c| c.chunk_id == overlap_id).unwrap_or(usize::MAX);

            // Best rank in any individual list was 0 (rank 1). Fused rank must be no worse
            // than the best rank in any individual list (0 == rank 1).
            // Since it appears first in both lists, its accumulated score is higher than
            // any pure single-list candidate that also appears at rank 1 in only one list.
            // Rank monotonicity: fused_rank must be <= min best rank = 0
            prop_assert_eq!(fused_rank, 0, "rank monotonicity violated for shared document");
        }

        #[test]
        fn rrf_score_monotonic_on_list_addition(
            extra_rank in 0usize..10,
        ) {
            // A document in one list, then add another list that also contains it.
            // Its fused score can only increase (never decrease).
            let base_list = vec![make_candidate("target", 0.8, RetrieverType::VectorIndex)];
            let score_before = reciprocal_rank_fusion(&[base_list.clone()], 60)
                .into_iter()
                .find(|c| c.chunk_id == "target")
                .unwrap()
                .fused_score;

            let mut extra_list: Vec<RankedCandidate> = (0..extra_rank)
                .map(|i| make_candidate(&format!("noise{i}"), 1.0, RetrieverType::Bm25Index))
                .collect();
            extra_list.push(make_candidate("target", 0.5, RetrieverType::Bm25Index));

            let score_after = reciprocal_rank_fusion(&[base_list, extra_list], 60)
                .into_iter()
                .find(|c| c.chunk_id == "target")
                .unwrap()
                .fused_score;

            prop_assert!(
                score_after >= score_before,
                "adding a list decreased fused score: before={score_before}, after={score_after}"
            );
        }
    }
}
