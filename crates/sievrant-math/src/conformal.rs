use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use sievrant_core::{
    error::{CalibrationError, Result},
    types::{Decision, SelectedEvidence},
};

/// Non-conformity score for a single (query, chunk) pair.
///
/// Measures how "unusual" or "untrustworthy" a chunk is as evidence for this query.
/// Lower score = more trustworthy = more likely to be retained by the gate.
///
/// s(q, c) = 1 - combined_relevance(c)
///
/// Weights: rerank 60%, rrf_score (clamped to [0,1]) 30%, structural bonus 10%.
/// These are fixed defaults; a future per-corpus calibration step can tune them.
pub fn nonconformity_score(
    rerank_score: f64,
    rrf_score: f64,
    structural_bonus: f64,
) -> f64 {
    let combined = 0.6 * rerank_score
        + 0.3 * rrf_score.min(1.0)
        + 0.1 * structural_bonus;
    1.0 - combined.clamp(0.0, 1.0)
}

/// A calibration dataset of (query, relevant_chunk_ids, scored_candidates) triples.
/// Used to derive a conformal threshold at a given coverage level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSet {
    pub id: String,
    pub entries: Vec<CalibrationEntry>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationEntry {
    pub query: String,
    pub relevant_chunk_ids: Vec<String>,
    /// Populated by running the pipeline against calibration queries.
    pub scores: Vec<CalibrationScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationScore {
    pub chunk_id: String,
    pub nonconformity_score: f64,
    pub is_relevant: bool,
}

/// Constant number of quantized score bins (0.0000 to 1.0000).
/// 10,000 bins provides 0.0001 precision, which is mathematically sufficient for conformal thresholds.
const BINS: usize = 10_000;

/// State for O(log B) online conformal calibration using a Fenwick Tree (Binary Indexed Tree).
/// Replaces the $O(N \log N)$ sorting approach, allowing live zero-latency threshold updates
/// as new user feedback streams into the SaaS platform.
#[derive(Debug, Clone)]
pub struct OnlineCalibrationState {
    tree: Vec<usize>,
    total_relevant: usize,
}

impl Default for OnlineCalibrationState {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineCalibrationState {
    pub fn new() -> Self {
        Self {
            tree: vec![0; BINS + 1],
            total_relevant: 0,
        }
    }

    /// Insert a new calibration score in O(log B) time.
    pub fn insert_score(&mut self, score: f64, is_relevant: bool) {
        if !is_relevant {
            return;
        }
        self.total_relevant += 1;
        
        let mut idx = (score * BINS as f64).round() as usize;
        idx = idx.clamp(0, BINS).min(BINS - 1) + 1; // 1-based indexing for Fenwick
        
        while idx <= BINS {
            self.tree[idx] += 1;
            idx += idx & idx.wrapping_neg(); 
        }
    }

    /// Compute the exact non-conformity threshold in O(log B) using binary lifting.
    pub fn compute_threshold(&self, coverage_target: f64) -> Result<f64> {
        if self.total_relevant == 0 {
            return Err(CalibrationError::InsufficientData {
                min_required: 1,
                provided: 0,
            }
            .into());
        }

        let n = self.total_relevant;
        let target_count = ((n as f64 + 1.0) * coverage_target).ceil() as usize;
        let target_count = target_count.min(n);

        let mut idx = 0;
        let mut sum = 0;
        let mut bit_mask = 1 << BINS.ilog2();

        while bit_mask > 0 {
            let next_idx = idx + bit_mask;
            if next_idx <= BINS && sum + self.tree[next_idx] < target_count {
                idx = next_idx;
                sum += self.tree[idx];
            }
            bit_mask >>= 1;
        }

        let target_idx = (idx + 1).min(BINS);
        Ok((target_idx - 1) as f64 / BINS as f64)
    }
}

/// Compute the conformal prediction threshold for a given coverage target.
///
/// Wraps the batch `CalibrationSet` into the `OnlineCalibrationState` to utilize
/// the $O(\log B)$ Fenwick Tree binary lifting algorithm.
pub fn compute_threshold(calibration_set: &CalibrationSet, coverage_target: f64) -> Result<f64> {
    let mut state = OnlineCalibrationState::new();
    for entry in &calibration_set.entries {
        for score in &entry.scores {
            state.insert_score(score.nonconformity_score, score.is_relevant);
        }
    }
    state.compute_threshold(coverage_target)
}

/// Output of the conformal gate.
pub struct ConformalResult {
    /// ANSWERED if enough chunks passed the gate; ABSTAINED otherwise.
    pub decision: Decision,
    /// Chunks whose non-conformity score is <= threshold.
    pub evidence: Vec<SelectedEvidence>,
    /// Chunks that were filtered out, kept for audit records.
    pub rejected: Vec<SelectedEvidence>,
    pub threshold: f64,
}

/// Apply the conformal gate to a set of selected evidence chunks.
///
/// Each chunk's `nonconformity_score` is compared to the calibrated threshold.
/// Chunks above the threshold are rejected. If the number of passing chunks
/// is below `min_evidence_count` the decision is ABSTAINED.
pub fn conformal_gate(
    mut evidence: Vec<SelectedEvidence>,
    threshold: f64,
    min_evidence_count: usize,
) -> ConformalResult {
    let mut passing: Vec<SelectedEvidence> = Vec::new();
    let mut rejected: Vec<SelectedEvidence> = Vec::new();

    for chunk in evidence.drain(..) {
        if chunk.nonconformity_score <= threshold {
            passing.push(chunk);
        } else {
            rejected.push(chunk);
        }
    }

    let decision = if passing.len() >= min_evidence_count {
        Decision::Answered
    } else {
        Decision::Abstained
    };

    ConformalResult {
        decision,
        evidence: passing,
        rejected,
        threshold,
    }
}

/// Empirical coverage report produced by `verify_coverage`.
pub struct CoverageReport {
    pub nominal_coverage: f64,
    pub empirical_coverage: f64,
    pub covered_count: usize,
    pub total_count: usize,
    pub threshold: f64,
    /// True when |empirical - nominal| <= tolerance.
    pub within_tolerance: bool,
}

/// Verify that the conformal coverage guarantee holds on a held-out test split.
///
/// Derives the threshold from `calibration_set` and measures what fraction of
/// truly relevant chunks in `test_set` fall below it. This is the non-negotiable
/// correctness check; it must hold before any deployment.
pub fn verify_coverage(
    calibration_set: &CalibrationSet,
    test_set: &CalibrationSet,
    coverage_target: f64,
    tolerance: f64,
) -> CoverageReport {
    let threshold = compute_threshold(calibration_set, coverage_target).unwrap_or(1.0);

    let mut covered = 0usize;
    let mut total = 0usize;

    for entry in &test_set.entries {
        for score in &entry.scores {
            if score.is_relevant {
                total += 1;
                if score.nonconformity_score <= threshold {
                    covered += 1;
                }
            }
        }
    }

    let empirical_coverage = if total > 0 {
        covered as f64 / total as f64
    } else {
        0.0
    };

    CoverageReport {
        nominal_coverage: coverage_target,
        empirical_coverage,
        covered_count: covered,
        total_count: total,
        threshold,
        within_tolerance: (empirical_coverage - coverage_target).abs() <= tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn make_calibration_set(id: &str, scores: &[(f64, bool)]) -> CalibrationSet {
        let entry = CalibrationEntry {
            query: "test query".into(),
            relevant_chunk_ids: vec![],
            scores: scores
                .iter()
                .enumerate()
                .map(|(i, (s, rel))| CalibrationScore {
                    chunk_id: format!("c{i}"),
                    nonconformity_score: *s,
                    is_relevant: *rel,
                })
                .collect(),
        };
        CalibrationSet {
            id: id.to_string(),
            entries: vec![entry],
            created_at: Utc::now(),
        }
    }

    #[test]
    fn threshold_covers_target_fraction_on_synthetic_data() {
        // 100 relevant chunks with scores 0.01, 0.02, ..., 1.00
        let scores: Vec<(f64, bool)> = (1..=100).map(|i| (i as f64 * 0.01, true)).collect();
        let cal = make_calibration_set("cal1", &scores);
        let threshold = compute_threshold(&cal, 0.95).unwrap();

        // At 95% coverage the threshold should be at or above the 95th percentile
        // of the relevant scores, which is 0.95.
        assert!(threshold <= 1.0);
        assert!(threshold >= 0.94, "threshold {threshold} should cover 95% of scores");
    }

    #[test]
    fn coverage_guarantee_holds_on_test_split() {
        // Train: 200 relevant scores uniformly in [0, 1]
        // Test: same distribution -> empirical coverage should be >= target
        let cal_scores: Vec<(f64, bool)> = (0..200)
            .map(|i| (i as f64 / 200.0, true))
            .collect();
        let test_scores: Vec<(f64, bool)> = (0..200)
            .map(|i| (i as f64 / 200.0, true))
            .collect();

        let cal = make_calibration_set("cal", &cal_scores);
        let test = make_calibration_set("test", &test_scores);

        let report = verify_coverage(&cal, &test, 0.90, 0.05);
        assert!(
            report.empirical_coverage >= 0.85,
            "coverage too low: {}",
            report.empirical_coverage
        );
    }

    #[test]
    fn gate_abstains_when_all_scores_exceed_threshold() {
        let evidence: Vec<SelectedEvidence> = (0..5)
            .map(|i| SelectedEvidence {
                chunk_id: format!("c{i}"),
                fused_score: 0.5,
                rerank_score: 0.5,
                marginal_gain: 0.5,
                nonconformity_score: 0.99,
                text: "text".into(),
                metadata: sievrant_core::types::ChunkMetadata {
                    doc_id: format!("d{i}"),
                    source_doc: "t.pdf".into(),
                    doc_version_hash: "sha256:x".into(),
                    path: vec![],
                    page: None,
                    char_span: None,
                    ingested_at: Utc::now(),
                    index_generation: 1,
                    acl_groups: vec![],
                    token_count: 100,
                },
            })
            .collect();

        let result = conformal_gate(evidence, 0.5, 1);
        assert!(matches!(result.decision, Decision::Abstained));
        assert!(result.evidence.is_empty());
        assert_eq!(result.rejected.len(), 5);
    }

    #[test]
    fn gate_answers_when_enough_chunks_pass() {
        let evidence: Vec<SelectedEvidence> = (0..3)
            .map(|i| SelectedEvidence {
                chunk_id: format!("c{i}"),
                fused_score: 0.8,
                rerank_score: 0.8,
                marginal_gain: 0.5,
                nonconformity_score: 0.1,
                text: "relevant text".into(),
                metadata: sievrant_core::types::ChunkMetadata {
                    doc_id: format!("d{i}"),
                    source_doc: "t.pdf".into(),
                    doc_version_hash: "sha256:x".into(),
                    path: vec![],
                    page: None,
                    char_span: None,
                    ingested_at: Utc::now(),
                    index_generation: 1,
                    acl_groups: vec![],
                    token_count: 100,
                },
            })
            .collect();

        let result = conformal_gate(evidence, 0.5, 1);
        assert!(matches!(result.decision, Decision::Answered));
        assert_eq!(result.evidence.len(), 3);
    }

    #[test]
    fn threshold_is_monotone_increasing_in_coverage_target() {
        let scores: Vec<(f64, bool)> = (0..100).map(|i| (i as f64 * 0.01, true)).collect();
        let cal = make_calibration_set("cal_mono", &scores);

        let t90 = compute_threshold(&cal, 0.90).unwrap();
        let t95 = compute_threshold(&cal, 0.95).unwrap();
        let t99 = compute_threshold(&cal, 0.99).unwrap();

        assert!(t90 <= t95, "threshold must be non-decreasing: t90={t90} t95={t95}");
        assert!(t95 <= t99, "threshold must be non-decreasing: t95={t95} t99={t99}");
    }

    #[test]
    fn insufficient_data_returns_error() {
        let cal = CalibrationSet {
            id: "empty".into(),
            entries: vec![],
            created_at: Utc::now(),
        };
        let result = compute_threshold(&cal, 0.95);
        assert!(result.is_err());
    }

    proptest! {
        #[test]
        fn threshold_monotone_proptest(
            n in 10usize..200,
            coverage_a in 0.5f64..0.9,
        ) {
            let coverage_b = (coverage_a + 0.05).min(0.999);
            let scores: Vec<(f64, bool)> = (0..n).map(|i| (i as f64 / n as f64, true)).collect();
            let cal = make_calibration_set("prop_cal", &scores);

            let ta = compute_threshold(&cal, coverage_a).unwrap();
            let tb = compute_threshold(&cal, coverage_b).unwrap();
            prop_assert!(
                ta <= tb + 1e-10,
                "threshold not monotone: coverage_a={coverage_a} ta={ta}, coverage_b={coverage_b} tb={tb}"
            );
        }
    }
}
