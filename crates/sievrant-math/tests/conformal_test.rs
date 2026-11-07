use sievrant_math::conformal::{conformal_set, conformal_threshold, query_score, Threshold};

#[test]
fn empty_calibration_set_is_insufficient() {
    assert!(matches!(
        conformal_threshold(&[], 0.95),
        Threshold::InsufficientData { provided: 0, .. }
    ));
}

#[test]
fn high_coverage_with_enough_data_calibrates() {
    let scores: Vec<f64> = (0..200).map(|i| i as f64 / 200.0).collect();
    let tau = conformal_threshold(&scores, 0.95).tau().unwrap();
    let covered = scores.iter().filter(|s| **s <= tau).count();
    assert!(covered as f64 / 200.0 >= 0.95);
}

#[test]
fn end_to_end_set_contains_best_relevant_candidate() {
    // Calibration: best relevant candidate scores for 50 queries.
    let cal: Vec<f64> = (0..50)
        .map(|i| query_score([0.05 + i as f64 * 0.01, 0.9]))
        .collect();
    let tau = conformal_threshold(&cal, 0.9).tau().unwrap();

    // Serving: a query whose relevant candidate scores like a typical one.
    let candidates = [0.95, 0.2, 0.6, 0.1];
    let set = conformal_set(&candidates, tau);
    assert!(set.contains(&3));
    assert!(!set.contains(&0));
}
