use proptest::prelude::*;
use sievrant_core::types::{ChunkMetadata, RankedCandidate, RetrieverType, RerankedCandidate, QueryTaxonomy};
use sievrant_math::{
    conformal::{compute_threshold, CalibrationSet, CalibrationEntry, CalibrationScore},
    rrf::reciprocal_rank_fusion,
    submodular::submodular_select,
};

fn make_calibration_set(id: &str, scores: &[f64]) -> CalibrationSet {
    let entry = CalibrationEntry {
        query: "test query".into(),
        taxonomy: QueryTaxonomy::Factoid,
        relevant_chunk_ids: vec![],
        scores: scores
            .iter()
            .enumerate()
            .map(|(i, &s)| CalibrationScore {
                chunk_id: format!("c{i}"),
                nonconformity_score: s,
                is_relevant: true,
            })
            .collect(),
    };
    CalibrationSet {
        id: id.to_string(),
        entries: vec![entry],
        created_at: chrono::Utc::now(),
    }
}

fn gen_candidates() -> impl Strategy<Value = Vec<RankedCandidate>> {
    prop::collection::vec(
        (
            "[a-z0-9]{4,8}",
            prop::num::f64::NORMAL | prop::num::f64::POSITIVE,
        )
            .prop_map(|(id, score)| RankedCandidate {
                chunk_id: id,
                text: "test".into(),
                score: score.abs().min(10.0), // Keep reasonable bounds
                metadata: ChunkMetadata {
                    doc_id: "doc1".into(),
                    source_doc: "doc1".into(),
                    doc_version_hash: "hash".into(),
                    path: vec![],
                    page: None,
                    char_span: None,
                    ingested_at: chrono::Utc::now(),
                    index_generation: 1,
                    acl_groups: vec![],
                    token_count: 10,
                },
                retriever: RetrieverType::VectorIndex,
            }),
        0..100,
    )
}

fn gen_evidence() -> impl Strategy<Value = Vec<RerankedCandidate>> {
    prop::collection::vec(
        (
            "[a-z0-9]{4,8}",
            prop::num::f64::NORMAL | prop::num::f64::POSITIVE,
        )
            .prop_map(|(id, score)| RerankedCandidate {
                chunk_id: id,
                text: "test".into(),
                metadata: ChunkMetadata {
                    doc_id: "doc1".into(),
                    source_doc: "doc1".into(),
                    doc_version_hash: "hash".into(),
                    path: vec![],
                    page: None,
                    char_span: None,
                    ingested_at: chrono::Utc::now(),
                    index_generation: 1,
                    acl_groups: vec![],
                    token_count: 10,
                },
                fused_score: score.abs().min(1.0),
                rerank_score: score.abs().min(1.0),
            }),
        0..50,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn rrf_scores_are_always_positive(
        l1 in gen_candidates(),
        l2 in gen_candidates()
    ) {
        let fused = reciprocal_rank_fusion(&[l1, l2], 60);
        for item in fused {
            prop_assert!(item.fused_score > 0.0, "Score was not positive: {}", item.fused_score);
        }
    }

    #[test]
    fn rrf_single_list_preserves_order(
        mut list in gen_candidates()
    ) {
        // Sort descending to simulate a ranked list
        list.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        let mut seen = std::collections::HashSet::new();
        list.retain(|c| seen.insert(c.chunk_id.clone()));
        
        let fused = reciprocal_rank_fusion(&[list.clone()], 60);
        
        // Output should have exactly the same ordering of chunk_ids
        let orig_ids: Vec<_> = list.iter().map(|c| &c.chunk_id).collect();
        let fused_ids: Vec<_> = fused.iter().map(|c| &c.chunk_id).collect();
        
        prop_assert_eq!(orig_ids, fused_ids, "RRF on a single list failed to preserve order");
    }

    #[test]
    fn submodular_never_exceeds_budget(
        evidence in gen_evidence(),
        budget in 10usize..500usize
    ) {
        let selected = submodular_select(&evidence, budget, 0.3);
        let total_tokens: usize = selected.iter().map(|e| e.metadata.token_count).sum();
        prop_assert!(total_tokens <= budget, "Exceeded budget: {} > {}", total_tokens, budget);
    }
    
    #[test]
    fn conformal_threshold_is_monotonic(
        scores in prop::collection::vec(0.0f64..2.0f64, 10..100)
    ) {
        // For two coverage targets where A < B, Threshold(A) <= Threshold(B)
        let cal = make_calibration_set("prop", &scores);
        let t_90 = compute_threshold(&cal, &QueryTaxonomy::Factoid, 0.90).unwrap();
        let t_95 = compute_threshold(&cal, &QueryTaxonomy::Factoid, 0.95).unwrap();
        let t_99 = compute_threshold(&cal, &QueryTaxonomy::Factoid, 0.99).unwrap();
        
        prop_assert!(t_90 <= t_95, "Threshold not monotonic: t90={} > t95={}", t_90, t_95);
        prop_assert!(t_95 <= t_99, "Threshold not monotonic: t95={} > t99={}", t_95, t_99);
    }
}

#[test]
fn correctness_rrf_known_values() {
    let c1 = RankedCandidate { chunk_id: "A".into(), score: 0.9, text: "".into(), metadata: dummy_meta(), retriever: RetrieverType::VectorIndex };
    let c2 = RankedCandidate { chunk_id: "B".into(), score: 0.8, text: "".into(), metadata: dummy_meta(), retriever: RetrieverType::VectorIndex };
    let c3 = RankedCandidate { chunk_id: "C".into(), score: 0.7, text: "".into(), metadata: dummy_meta(), retriever: RetrieverType::VectorIndex };
    
    // List 1: A, B, C (ranks: 1, 2, 3)
    let l1 = vec![c1.clone(), c2.clone(), c3.clone()];
    // List 2: B, A (ranks: 1, 2)
    let l2 = vec![c2.clone(), c1.clone()];
    
    let fused = reciprocal_rank_fusion(&[l1, l2], 60);
    
    // A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    // B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.032522
    // C: 1/(60+3)            = 1/63        = 0.015873
    
    // A and B should have identical scores and tie for first
    assert_eq!(fused.len(), 3);
    assert!(fused[0].chunk_id == "A" || fused[0].chunk_id == "B");
    assert!(fused[1].chunk_id == "A" || fused[1].chunk_id == "B");
    assert_eq!(fused[2].chunk_id, "C");
    
    assert!((fused[0].fused_score - 0.032522).abs() < 1e-4);
}

#[test]
fn correctness_conformal_coverage_audit() {
    // Generate synthetic calibration data (nonconformity scores)
    // and verify that the exact formula achieves the nominal coverage.
    let n = 1000;
    // Uniformly distributed scores between 0 and 1
    let cal_scores: Vec<f64> = (1..=n).map(|i| i as f64 / n as f64).collect();
    
    let cal = make_calibration_set("test", &cal_scores);
    let target_coverage = 0.95;
    let threshold = compute_threshold(&cal, &QueryTaxonomy::Factoid, target_coverage).unwrap();
    
    // Evaluate coverage on the same set (in practice this would be a holdout set, 
    // but exchangeability guarantees hold in expectation)
    let covered_count = cal_scores.iter().filter(|&&s| s <= threshold).count();
    let empirical = covered_count as f64 / n as f64;
    
    // It should be exactly 95% (950 / 1000) based on the ceiling formula
    assert!(empirical >= target_coverage, "Empirical {} < Target {}", empirical, target_coverage);
    assert!(empirical - target_coverage < 0.02, "Empirical {} too far above Target {}", empirical, target_coverage);
}

fn dummy_meta() -> ChunkMetadata {
    ChunkMetadata {
        doc_id: "doc1".into(),
        source_doc: "doc1".into(),
        doc_version_hash: "hash".into(),
        path: vec![],
        page: None,
        char_span: None,
        ingested_at: chrono::Utc::now(),
        index_generation: 1,
        acl_groups: vec![],
        token_count: 10,
    }
}
