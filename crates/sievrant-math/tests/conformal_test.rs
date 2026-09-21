use sievrant_core::types::{ChunkMetadata, Decision, QueryTaxonomy, SelectedEvidence};
use sievrant_math::conformal::{
    compute_threshold, conformal_gate, nonconformity_score, CalibrationEntry, CalibrationScore,
    CalibrationSet,
};

#[test]
fn test_conformal_gate_edge_cases() {
    let mut evidence = vec![];
    let scores = [0.2, 0.3, 0.4, 0.5, 0.6];
    for (i, &score) in scores.iter().enumerate() {
        evidence.push(SelectedEvidence {
            chunk_id: format!("chunk-{}", i),
            fused_score: 0.9,
            rerank_score: 0.8,
            marginal_gain: 0.1,
            nonconformity_score: score,
            text: format!("evidence text {}", i),
            metadata: ChunkMetadata {
                doc_id: "doc1".to_string(),
                source_doc: "doc1.pdf".to_string(),
                doc_version_hash: "hash".to_string(),
                path: vec![],
                page: None,
                char_span: None,
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: vec![],
                token_count: 100,
            },
        });
    }

    // Test coverage boundary where exactly min_evidence_count passes
    // threshold = 0.4. Scores: 0.2, 0.3, 0.4 will pass. (3 chunks)
    let res = conformal_gate(evidence.clone(), 0.4, 3);
    assert!(matches!(res.decision, Decision::Answered));
    assert_eq!(res.evidence.len(), 3);
    assert_eq!(res.rejected.len(), 2);

    // Test where it abstains due to min_evidence_count
    // threshold = 0.3. Scores: 0.2, 0.3 will pass. (2 chunks)
    let res_abstain = conformal_gate(evidence.clone(), 0.3, 3);
    assert!(matches!(res_abstain.decision, Decision::Abstained));
    assert_eq!(res_abstain.evidence.len(), 2);
    assert_eq!(res_abstain.rejected.len(), 3);
}

#[test]
fn test_compute_threshold_n_zero() {
    let empty_cal = CalibrationSet {
        id: "empty".to_string(),
        entries: vec![],
        created_at: chrono::Utc::now(),
    };

    let result = compute_threshold(&empty_cal, &QueryTaxonomy::Factoid, 0.95);
    assert!(result.is_err()); // InsufficientData error
}

#[test]
fn test_compute_threshold_high_coverage() {
    let mut scores = vec![];
    for i in 1..=100 {
        scores.push(CalibrationScore {
            chunk_id: format!("c{}", i),
            nonconformity_score: i as f64 * 0.01,
            is_relevant: true,
        });
    }

    let cal = CalibrationSet {
        id: "test".to_string(),
        entries: vec![CalibrationEntry {
            query: "test".to_string(),
            taxonomy: QueryTaxonomy::Analytical,
            relevant_chunk_ids: vec![],
            scores,
        }],
        created_at: chrono::Utc::now(),
    };

    // For n=100, target_count = ceil(101 * 0.95) = 96
    // So the threshold should cover 96 items. Score 96 is 0.96.
    let threshold = compute_threshold(&cal, &QueryTaxonomy::Analytical, 0.95).unwrap();
    assert!((0.95..=0.97).contains(&threshold));
}

#[test]
fn test_nonconformity_score() {
    let score = nonconformity_score(0.8, 0.7, 0.1);
    // 0.6 * 0.8 + 0.3 * 0.7 + 0.1 * 0.1 = 0.48 + 0.21 + 0.01 = 0.7
    // 1.0 - 0.7 = 0.3
    assert!((score - 0.3).abs() < 1e-6);
}
