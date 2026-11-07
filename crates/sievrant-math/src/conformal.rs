//! Split conformal prediction for retrieval.
//!
//! # What is guaranteed
//!
//! For each labelled calibration query `i` the pipeline produces a list of
//! reranked candidates. The query's score is
//!
//! ```text
//! s_i = min over relevant candidates c of  nonconformity(c)
//! ```
//!
//! i.e. the nonconformity of the *best* relevant candidate, or `+inf` when
//! the retriever surfaced no relevant candidate at all.
//!
//! Given `n` such scores and a coverage target `1 - alpha`, the threshold
//! `tau` is the `ceil((n + 1)(1 - alpha))`-th smallest score. At query time
//! the conformal evidence set is every candidate with
//! `nonconformity(c) <= tau`.
//!
//! If a new answerable query is exchangeable with the calibration queries
//! (drawn from the same distribution, served by the same pipeline), then
//!
//! ```text
//! P(conformal set contains at least one relevant candidate) >= 1 - alpha
//! ```
//!
//! A consequence is that the system abstains (returns an empty set) on at
//! most an `alpha` fraction of answerable queries.
//!
//! # What is NOT guaranteed
//!
//! - Answer correctness or absence of hallucination. The guarantee is about
//!   retrieved evidence, not about what an LLM writes from it.
//! - Abstention on unanswerable queries. That rate is measured, not bounded.
//! - Anything when the exchangeability assumption fails (traffic drift,
//!   pipeline changes). Live coverage monitoring exists to detect this.
//!
//! When more than an `alpha` fraction of calibration queries have
//! `s_i = +inf`, the retriever itself misses too often for the target to be
//! reachable. [`conformal_threshold`] reports this instead of returning a
//! threshold that would silently fail to deliver the promised coverage.

use serde::{Deserialize, Serialize};

/// Nonconformity of a single candidate. Lower means more trustworthy.
///
/// Any fixed, deterministic function of the reranker output yields a valid
/// guarantee; what matters is that calibration and serving use the same one.
pub fn nonconformity(rerank_score: f64) -> f64 {
    1.0 - rerank_score
}

/// Query-level score for one calibration example: the nonconformity of the
/// best relevant candidate, or `f64::INFINITY` if none was retrieved.
pub fn query_score<I>(relevant_candidate_nonconformities: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    relevant_candidate_nonconformities
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

/// Outcome of computing a conformal threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Threshold {
    /// A finite threshold that delivers the requested coverage.
    Calibrated { tau: f64 },
    /// Too few calibration examples for this coverage level. Split conformal
    /// needs `n >= coverage / (1 - coverage)` (19 for 95%, 99 for 99%).
    InsufficientData { required: usize, provided: usize },
    /// The retriever misses relevant evidence too often. `max_coverage` is
    /// the highest target this calibration set can support.
    Unachievable { max_coverage: f64 },
}

impl Threshold {
    /// The threshold value, if calibrated.
    pub fn tau(&self) -> Option<f64> {
        match self {
            Threshold::Calibrated { tau } => Some(*tau),
            _ => None,
        }
    }
}

/// Minimum calibration size for a given coverage target.
pub fn min_examples_for(coverage: f64) -> usize {
    // Smallest n with ceil((n + 1) * coverage) <= n.
    let mut n = 1usize;
    while (((n + 1) as f64) * coverage).ceil() as usize > n {
        n += 1;
    }
    n
}

/// Compute the split-conformal threshold from query-level scores.
///
/// `scores` need not be sorted. `coverage` must lie in (0, 1).
pub fn conformal_threshold(scores: &[f64], coverage: f64) -> Threshold {
    assert!(
        coverage > 0.0 && coverage < 1.0,
        "coverage must be in (0, 1), got {coverage}"
    );
    let n = scores.len();
    let required = min_examples_for(coverage);
    if n < required {
        return Threshold::InsufficientData {
            required,
            provided: n,
        };
    }

    let mut sorted: Vec<f64> = scores.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let k = (((n + 1) as f64) * coverage).ceil() as usize;
    let tau = sorted[k - 1];

    if tau.is_finite() {
        Threshold::Calibrated { tau }
    } else {
        let finite = sorted.iter().filter(|s| s.is_finite()).count();
        // Largest coverage c with ceil((n + 1) c) <= finite.
        let max_coverage = finite as f64 / (n + 1) as f64;
        Threshold::Unachievable { max_coverage }
    }
}

/// Indices of candidates that belong to the conformal set.
pub fn conformal_set(candidate_nonconformities: &[f64], tau: f64) -> Vec<usize> {
    candidate_nonconformities
        .iter()
        .enumerate()
        .filter(|(_, s)| **s <= tau)
        .map(|(i, _)| i)
        .collect()
}

/// Empirical coverage of a threshold on held-out query scores.
/// Returns `None` for an empty slice.
pub fn empirical_coverage(held_out_scores: &[f64], tau: f64) -> Option<f64> {
    if held_out_scores.is_empty() {
        return None;
    }
    let covered = held_out_scores.iter().filter(|s| **s <= tau).count();
    Some(covered as f64 / held_out_scores.len() as f64)
}

/// Wilson score interval for a binomial proportion at ~95% confidence.
/// Used to decide whether observed coverage is *significantly* below target
/// instead of alerting on noise.
pub fn wilson_interval(successes: usize, trials: usize) -> Option<(f64, f64)> {
    if trials == 0 {
        return None;
    }
    let z = 1.959_963_984_540_054_f64;
    let n = trials as f64;
    let p = successes as f64 / n;
    let denom = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denom;
    let half = z * ((p * (1.0 - p) / n) + z * z / (4.0 * n * n)).sqrt() / denom;
    Some(((centre - half).max(0.0), (centre + half).min(1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so the statistical tests are reproducible
    /// without pulling in a rand dependency.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn nonconformity_is_decreasing_in_score() {
        assert!(nonconformity(0.9) < nonconformity(0.1));
    }

    #[test]
    fn query_score_takes_best_relevant_and_inf_when_missing() {
        assert_eq!(query_score([0.4, 0.1, 0.7]), 0.1);
        assert!(query_score(std::iter::empty()).is_infinite());
    }

    #[test]
    fn min_examples_matches_closed_form() {
        assert_eq!(min_examples_for(0.9), 9);
        assert_eq!(min_examples_for(0.95), 19);
        assert_eq!(min_examples_for(0.99), 99);
    }

    #[test]
    fn threshold_uses_finite_sample_quantile() {
        // 100 scores 0.01..=1.00; k = ceil(101 * 0.9) = 91 -> 0.91
        let scores: Vec<f64> = (1..=100).map(|i| i as f64 / 100.0).collect();
        let t = conformal_threshold(&scores, 0.9);
        assert_eq!(t, Threshold::Calibrated { tau: 0.91 });
    }

    #[test]
    fn too_few_examples_is_reported() {
        let t = conformal_threshold(&[0.1, 0.2, 0.3], 0.95);
        assert_eq!(
            t,
            Threshold::InsufficientData {
                required: 19,
                provided: 3
            }
        );
    }

    #[test]
    fn retrieval_misses_make_target_unachievable() {
        // 20% of queries have no relevant candidate at all.
        let mut scores: Vec<f64> = (0..80).map(|i| i as f64 / 100.0).collect();
        scores.extend(std::iter::repeat_n(f64::INFINITY, 20));
        match conformal_threshold(&scores, 0.95) {
            Threshold::Unachievable { max_coverage } => {
                assert!((max_coverage - 80.0 / 101.0).abs() < 1e-12);
            }
            other => panic!("expected Unachievable, got {other:?}"),
        }
        // A lower target that the retriever can meet still calibrates.
        assert!(conformal_threshold(&scores, 0.7).tau().is_some());
    }

    #[test]
    fn threshold_is_monotone_in_coverage() {
        let mut rng = Rng(7);
        let scores: Vec<f64> = (0..500).map(|_| rng.next_f64()).collect();
        let mut last = f64::NEG_INFINITY;
        for c in [0.5, 0.8, 0.9, 0.95, 0.99] {
            let tau = conformal_threshold(&scores, c).tau().unwrap();
            assert!(tau >= last);
            last = tau;
        }
    }

    #[test]
    fn conformal_set_selects_scores_at_or_below_tau() {
        assert_eq!(conformal_set(&[0.1, 0.5, 0.3, 0.9], 0.3), vec![0, 2]);
    }

    /// The core statistical property: averaged over many random
    /// calibration/test splits, coverage lands in
    /// [1 - alpha, 1 - alpha + 1/(n+1)] up to Monte Carlo error.
    #[test]
    fn marginal_coverage_holds_on_exchangeable_data() {
        let mut rng = Rng(0x5eed);
        let n_cal = 200;
        let n_test = 200;
        let trials = 400;
        for &coverage in &[0.8, 0.9, 0.95] {
            let mut total = 0.0;
            for _ in 0..trials {
                // Skewed score distribution with 1% retrieval misses (low enough
                // that every trial stays achievable at 95%).
                let mut draw = || {
                    let u = rng.next_f64();
                    if u < 0.01 {
                        f64::INFINITY
                    } else {
                        rng.next_f64().powf(3.0)
                    }
                };
                let cal: Vec<f64> = (0..n_cal).map(|_| draw()).collect();
                let test: Vec<f64> = (0..n_test).map(|_| draw()).collect();
                let tau = conformal_threshold(&cal, coverage).tau().unwrap();
                total += empirical_coverage(&test, tau).unwrap();
            }
            let mean = total / trials as f64;
            let upper = coverage + 1.0 / (n_cal as f64 + 1.0);
            assert!(
                mean >= coverage - 0.005,
                "coverage {mean:.4} below target {coverage}"
            );
            assert!(
                mean <= upper + 0.005,
                "coverage {mean:.4} above upper bound {upper:.4}"
            );
        }
    }

    #[test]
    fn wilson_interval_brackets_proportion() {
        let (lo, hi) = wilson_interval(90, 100).unwrap();
        assert!(lo < 0.9 && hi > 0.9);
        assert!(wilson_interval(0, 0).is_none());
    }
}
