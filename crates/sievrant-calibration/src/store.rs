//! Per-corpus calibration state, persisted under the corpus data directory:
//!
//! - `calibration.json`: query-level conformal scores plus provenance,
//! - `querylog.jsonl`: what each served query saw (for feedback and gap reports),
//! - `feedback.jsonl`: coverage observations from the feedback API.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use sievrant_core::error::{CalibrationError, Result, SievrantError};
use sievrant_core::types::{CalibrationSource, Decision, GuaranteeStatus, ObservedCoverage};
use sievrant_math::conformal::{
    conformal_threshold, empirical_coverage, wilson_interval, Threshold,
};

const MAX_LOGGED_QUERIES: usize = 20_000;
const MAX_QUERYLOG_BYTES: u64 = 100 * 1024 * 1024;
const COVERAGE_WINDOW_DAYS: i64 = 30;

fn io_err(e: impl std::fmt::Display) -> SievrantError {
    SievrantError::Calibration(CalibrationError::Io(e.to_string()))
}

fn parse_err(e: impl std::fmt::Display) -> SievrantError {
    SievrantError::Calibration(CalibrationError::Parse(e.to_string()))
}

/// Split-half validation of a calibration set at one coverage level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoldoutResult {
    pub coverage: f64,
    /// Mean held-out coverage across random 50/50 splits.
    pub mean_coverage: f64,
    /// Worst split.
    pub min_coverage: f64,
    /// Splits where the target was reachable on the calibration half.
    pub splits: usize,
}

/// A stored calibration set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationRecord {
    pub set_id: String,
    pub source: CalibrationSource,
    pub created_at: DateTime<Utc>,
    /// Pipeline fingerprint the scores were computed with.
    pub fingerprint: String,
    /// Corpus size (chunks) at calibration time, for staleness checks.
    pub corpus_chunk_count: u64,
    /// Query-level nonconformity scores. `None` = retriever missed every
    /// relevant document for that query (score +inf).
    pub scores: Vec<Option<f64>>,
    /// Examples labelled unanswerable (no relevant documents). Excluded from
    /// the guarantee; used to measure abstention on out-of-scope queries.
    pub unanswerable_count: usize,
    /// Of the unanswerable examples, how many the 95% gate would reject.
    pub unanswerable_abstained_at_95: usize,
    pub holdout: Vec<HoldoutResult>,
}

impl CalibrationRecord {
    pub fn finite_scores(&self) -> Vec<f64> {
        self.scores
            .iter()
            .map(|s| s.unwrap_or(f64::INFINITY))
            .collect()
    }

    /// Fraction of calibration queries where retrieval found any relevant document.
    pub fn retrieval_recall(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.iter().filter(|s| s.is_some()).count() as f64 / self.scores.len() as f64
    }
}

/// Run `splits` random 50/50 splits and report held-out coverage.
pub fn split_half_validation(
    scores: &[f64],
    coverage: f64,
    splits: usize,
    seed: u64,
) -> HoldoutResult {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut total = 0.0;
    let mut min = f64::INFINITY;
    let mut used = 0usize;
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    for _ in 0..splits {
        for i in (1..idx.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        let half = idx.len() / 2;
        let cal: Vec<f64> = idx[..half].iter().map(|i| scores[*i]).collect();
        let test: Vec<f64> = idx[half..].iter().map(|i| scores[*i]).collect();
        if let Threshold::Calibrated { tau } = conformal_threshold(&cal, coverage) {
            if let Some(c) = empirical_coverage(&test, tau) {
                total += c;
                min = min.min(c);
                used += 1;
            }
        }
    }
    HoldoutResult {
        coverage,
        mean_coverage: if used > 0 { total / used as f64 } else { 0.0 },
        min_coverage: if used > 0 { min } else { 0.0 },
        splits: used,
    }
}

/// What the gate should do for one request.
#[derive(Debug, Clone)]
pub struct GateDecision {
    pub status: GuaranteeStatus,
    pub tau: Option<f64>,
    pub max_achievable_coverage: Option<f64>,
    pub set_id: Option<String>,
    pub source: Option<CalibrationSource>,
    pub size: Option<usize>,
    /// Why the guarantee does not apply, when it doesn't.
    pub reason: Option<String>,
}

/// One candidate as the gate saw it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggedCandidate {
    pub chunk_id: String,
    pub doc_id: String,
    pub nonconformity: f64,
}

/// What a served query saw. Used to evaluate feedback and build gap reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRecord {
    pub query_id: String,
    pub at: DateTime<Utc>,
    pub query: String,
    pub decision: Decision,
    pub requested_coverage: f64,
    pub tau: Option<f64>,
    pub status: GuaranteeStatus,
    pub candidates: Vec<LoggedCandidate>,
}

/// A labelled observation from the feedback API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackObservation {
    pub query_id: String,
    pub at: DateTime<Utc>,
    pub requested_coverage: f64,
    /// Whether the conformal set contained a relevant candidate.
    pub covered: bool,
    /// Query-level score; `None` if no relevant candidate was retrieved.
    pub score: Option<f64>,
}

/// Result of submitting feedback.
#[derive(Debug, Clone, Serialize)]
pub struct FeedbackOutcome {
    pub query_id: String,
    /// `None` when the query was served without a calibrated threshold.
    pub covered: Option<bool>,
    pub relevant_retrieved: bool,
}

/// A query users asked that the corpus could not answer.
#[derive(Debug, Clone, Serialize)]
pub struct GapEntry {
    pub query: String,
    pub count: usize,
    pub last_seen: DateTime<Utc>,
    /// Best candidate nonconformity seen; how close the corpus came.
    pub best_nonconformity: Option<f64>,
}

/// Calibration, monitoring and query history for one corpus.
pub struct CorpusCalibration {
    dir: PathBuf,
    record: RwLock<Option<CalibrationRecord>>,
    queries: RwLock<(VecDeque<String>, HashMap<String, QueryRecord>)>,
    feedback: RwLock<Vec<FeedbackObservation>>,
}

impl CorpusCalibration {
    /// Open (or create) calibration state in `dir`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(io_err)?;

        let record = match fs::read_to_string(dir.join("calibration.json")) {
            Ok(text) => Some(serde_json::from_str(&text).map_err(parse_err)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(io_err(e)),
        };

        let mut order = VecDeque::new();
        let mut map = HashMap::new();
        for rec in read_jsonl::<QueryRecord>(&dir.join("querylog.jsonl"))? {
            order.push_back(rec.query_id.clone());
            map.insert(rec.query_id.clone(), rec);
            if order.len() > MAX_LOGGED_QUERIES {
                if let Some(old) = order.pop_front() {
                    map.remove(&old);
                }
            }
        }
        let feedback = read_jsonl::<FeedbackObservation>(&dir.join("feedback.jsonl"))?;

        Ok(Self {
            dir,
            record: RwLock::new(record),
            queries: RwLock::new((order, map)),
            feedback: RwLock::new(feedback),
        })
    }

    pub fn record(&self) -> Option<CalibrationRecord> {
        self.record.read().unwrap().clone()
    }

    /// Replace the calibration set and persist it atomically.
    pub fn set_record(&self, record: CalibrationRecord) -> Result<()> {
        let tmp = self.dir.join("calibration.json.tmp");
        let data = serde_json::to_string_pretty(&record).map_err(parse_err)?;
        fs::write(&tmp, data).map_err(io_err)?;
        fs::rename(&tmp, self.dir.join("calibration.json")).map_err(io_err)?;
        *self.record.write().unwrap() = Some(record);
        Ok(())
    }

    /// Remove the calibration set.
    pub fn clear_record(&self) -> Result<()> {
        match fs::remove_file(self.dir.join("calibration.json")) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_err(e)),
        }
        *self.record.write().unwrap() = None;
        Ok(())
    }

    /// Decide whether and how to gate a request.
    pub fn decide(
        &self,
        coverage: f64,
        fingerprint: &str,
        corpus_chunk_count: u64,
        growth_factor: f64,
    ) -> GateDecision {
        let guard = self.record.read().unwrap();
        let Some(rec) = guard.as_ref() else {
            return GateDecision {
                status: GuaranteeStatus::Uncalibrated,
                tau: None,
                max_achievable_coverage: None,
                set_id: None,
                source: None,
                size: None,
                reason: Some(
                    "corpus has no calibration set; POST /v1/calibration or /v1/calibration/bootstrap"
                        .into(),
                ),
            };
        };
        let mut d = GateDecision {
            status: GuaranteeStatus::Calibrated,
            tau: None,
            max_achievable_coverage: None,
            set_id: Some(rec.set_id.clone()),
            source: Some(rec.source),
            size: Some(rec.scores.len()),
            reason: None,
        };
        if rec.fingerprint != fingerprint {
            d.status = GuaranteeStatus::Stale;
            d.reason = Some("retrieval pipeline changed since calibration; recalibrate".into());
            return d;
        }
        let base = rec.corpus_chunk_count.max(1) as f64;
        let ratio = corpus_chunk_count as f64 / base;
        if ratio > growth_factor || ratio < 1.0 / growth_factor {
            d.status = GuaranteeStatus::Stale;
            d.reason = Some(format!(
                "corpus size changed from {} to {} chunks since calibration; recalibrate",
                rec.corpus_chunk_count, corpus_chunk_count
            ));
            return d;
        }
        match conformal_threshold(&rec.finite_scores(), coverage) {
            Threshold::Calibrated { tau } => d.tau = Some(tau),
            Threshold::InsufficientData { required, provided } => {
                d.status = GuaranteeStatus::InsufficientData;
                d.reason = Some(format!(
                    "{coverage} coverage needs at least {required} calibration queries, have {provided}"
                ));
            }
            Threshold::Unachievable { max_coverage } => {
                d.status = GuaranteeStatus::Unachievable;
                d.max_achievable_coverage = Some(max_coverage);
                d.reason = Some(format!(
                    "retrieval found relevant evidence for only {:.1}% of calibration queries; \
                     the highest supportable target is {:.3}",
                    rec.retrieval_recall() * 100.0,
                    max_coverage
                ));
            }
        }
        d
    }

    /// Record a served query.
    pub fn log_query(&self, rec: QueryRecord) -> Result<()> {
        let path = self.dir.join("querylog.jsonl");
        if fs::metadata(&path)
            .map(|m| m.len() > MAX_QUERYLOG_BYTES)
            .unwrap_or(false)
        {
            let _ = fs::rename(&path, self.dir.join("querylog.1.jsonl"));
        }
        append_jsonl(&path, &rec)?;
        let mut q = self.queries.write().unwrap();
        q.0.push_back(rec.query_id.clone());
        q.1.insert(rec.query_id.clone(), rec);
        if q.0.len() > MAX_LOGGED_QUERIES {
            if let Some(old) = q.0.pop_front() {
                q.1.remove(&old);
            }
        }
        Ok(())
    }

    pub fn query(&self, query_id: &str) -> Option<QueryRecord> {
        self.queries.read().unwrap().1.get(query_id).cloned()
    }

    /// Record which documents/chunks were actually relevant to a served query.
    pub fn submit_feedback(
        &self,
        query_id: &str,
        relevant_doc_ids: &[String],
        relevant_chunk_ids: &[String],
    ) -> Result<Option<FeedbackOutcome>> {
        let Some(rec) = self.query(query_id) else {
            return Ok(None);
        };
        let relevant: Vec<f64> = rec
            .candidates
            .iter()
            .filter(|c| {
                relevant_doc_ids.iter().any(|d| d == &c.doc_id)
                    || relevant_chunk_ids.iter().any(|id| id == &c.chunk_id)
            })
            .map(|c| c.nonconformity)
            .collect();
        let score = sievrant_math::conformal::query_score(relevant.iter().copied());
        let score = score.is_finite().then_some(score);

        let covered = rec.tau.map(|tau| score.is_some_and(|s| s <= tau));
        if let Some(covered) = covered {
            if rec.status == GuaranteeStatus::Calibrated {
                let obs = FeedbackObservation {
                    query_id: query_id.to_string(),
                    at: Utc::now(),
                    requested_coverage: rec.requested_coverage,
                    covered,
                    score,
                };
                append_jsonl(&self.dir.join("feedback.jsonl"), &obs)?;
                self.feedback.write().unwrap().push(obs);
            }
        }
        Ok(Some(FeedbackOutcome {
            query_id: query_id.to_string(),
            covered,
            relevant_retrieved: score.is_some(),
        }))
    }

    /// Query-level scores from feedback, usable as a `Feedback` calibration set.
    pub fn feedback_scores(&self) -> Vec<Option<f64>> {
        self.feedback
            .read()
            .unwrap()
            .iter()
            .map(|o| o.score)
            .collect()
    }

    /// Observed coverage over the last 30 days at (approximately) this target.
    pub fn observed(&self, coverage: f64, tolerance: f64) -> Option<ObservedCoverage> {
        let cutoff = Utc::now() - Duration::days(COVERAGE_WINDOW_DAYS);
        let fb = self.feedback.read().unwrap();
        let relevant: Vec<&FeedbackObservation> = fb
            .iter()
            .filter(|o| o.at >= cutoff && (o.requested_coverage - coverage).abs() < 1e-9)
            .collect();
        let n = relevant.len();
        let covered = relevant.iter().filter(|o| o.covered).count();
        let (lower, upper) = wilson_interval(covered, n)?;
        Some(ObservedCoverage {
            rate: covered as f64 / n as f64,
            observations: n,
            lower_95: lower,
            upper_95: upper,
            drift_alert: n >= 30 && upper < coverage - tolerance,
        })
    }

    /// Most frequent abstained queries, newest first on ties.
    pub fn gaps(&self, limit: usize) -> Vec<GapEntry> {
        let q = self.queries.read().unwrap();
        let mut by_text: HashMap<String, GapEntry> = HashMap::new();
        for rec in q.1.values() {
            if rec.decision != Decision::Abstained {
                continue;
            }
            let key = rec
                .query
                .to_lowercase()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let best = rec
                .candidates
                .iter()
                .map(|c| c.nonconformity)
                .fold(None, |acc: Option<f64>, s| {
                    Some(acc.map_or(s, |a| a.min(s)))
                });
            let entry = by_text.entry(key.clone()).or_insert(GapEntry {
                query: key,
                count: 0,
                last_seen: rec.at,
                best_nonconformity: best,
            });
            entry.count += 1;
            entry.last_seen = entry.last_seen.max(rec.at);
            entry.best_nonconformity = match (entry.best_nonconformity, best) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
        let mut out: Vec<GapEntry> = by_text.into_values().collect();
        out.sort_by(|a, b| b.count.cmp(&a.count).then(b.last_seen.cmp(&a.last_seen)));
        out.truncate(limit);
        out
    }

    /// Mean conformal set size and abstention rate over logged calibrated queries.
    pub fn serving_stats(&self) -> (usize, Option<f64>, Option<f64>) {
        let q = self.queries.read().unwrap();
        let gated: Vec<&QueryRecord> =
            q.1.values()
                .filter(|r| r.status == GuaranteeStatus::Calibrated && r.tau.is_some())
                .collect();
        if gated.is_empty() {
            return (q.1.len(), None, None);
        }
        let sizes: f64 = gated
            .iter()
            .map(|r| {
                let tau = r.tau.unwrap();
                r.candidates
                    .iter()
                    .filter(|c| c.nonconformity <= tau)
                    .count() as f64
            })
            .sum();
        let abstained = gated
            .iter()
            .filter(|r| r.decision == Decision::Abstained)
            .count() as f64;
        let n = gated.len() as f64;
        (q.1.len(), Some(sizes / n), Some(abstained / n))
    }
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(io_err(e)),
    };
    let mut out = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.map_err(io_err)?;
        if line.trim().is_empty() {
            continue;
        }
        // A torn final line after a crash is skipped rather than fatal.
        if let Ok(v) = serde_json::from_str(&line) {
            out.push(v);
        }
    }
    Ok(out)
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut line = serde_json::to_string(value).map_err(parse_err)?;
    line.push('\n');
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(io_err)?;
    f.write_all(line.as_bytes()).map_err(io_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn record(scores: Vec<Option<f64>>) -> CalibrationRecord {
        CalibrationRecord {
            set_id: "cal-1".into(),
            source: CalibrationSource::Labeled,
            created_at: Utc::now(),
            fingerprint: "fp".into(),
            corpus_chunk_count: 100,
            scores,
            unanswerable_count: 0,
            unanswerable_abstained_at_95: 0,
            holdout: vec![],
        }
    }

    fn uniform(n: usize) -> Vec<Option<f64>> {
        (0..n).map(|i| Some(i as f64 / n as f64)).collect()
    }

    fn query(id: &str, tau: Option<f64>, decision: Decision) -> QueryRecord {
        QueryRecord {
            query_id: id.into(),
            at: Utc::now(),
            query: "What is the Lease term?".into(),
            decision,
            requested_coverage: 0.9,
            tau,
            status: if tau.is_some() {
                GuaranteeStatus::Calibrated
            } else {
                GuaranteeStatus::Uncalibrated
            },
            candidates: vec![
                LoggedCandidate {
                    chunk_id: "d1:0".into(),
                    doc_id: "d1".into(),
                    nonconformity: 0.2,
                },
                LoggedCandidate {
                    chunk_id: "d2:0".into(),
                    doc_id: "d2".into(),
                    nonconformity: 0.7,
                },
            ],
        }
    }

    #[test]
    fn uncalibrated_corpus_reports_status() {
        let dir = TempDir::new().unwrap();
        let c = CorpusCalibration::open(dir.path()).unwrap();
        let d = c.decide(0.95, "fp", 100, 1.5);
        assert_eq!(d.status, GuaranteeStatus::Uncalibrated);
        assert!(d.tau.is_none());
    }

    #[test]
    fn calibrated_record_persists_and_yields_threshold() {
        let dir = TempDir::new().unwrap();
        {
            let c = CorpusCalibration::open(dir.path()).unwrap();
            c.set_record(record(uniform(100))).unwrap();
        }
        let c = CorpusCalibration::open(dir.path()).unwrap();
        let d = c.decide(0.9, "fp", 110, 1.5);
        assert_eq!(d.status, GuaranteeStatus::Calibrated);
        assert!(d.tau.unwrap() > 0.85);
        assert_eq!(d.size, Some(100));
    }

    #[test]
    fn pipeline_or_corpus_change_marks_stale() {
        let dir = TempDir::new().unwrap();
        let c = CorpusCalibration::open(dir.path()).unwrap();
        c.set_record(record(uniform(100))).unwrap();
        assert_eq!(
            c.decide(0.9, "other", 100, 1.5).status,
            GuaranteeStatus::Stale
        );
        assert_eq!(c.decide(0.9, "fp", 400, 1.5).status, GuaranteeStatus::Stale);
    }

    #[test]
    fn insufficient_and_unachievable_are_distinguished() {
        let dir = TempDir::new().unwrap();
        let c = CorpusCalibration::open(dir.path()).unwrap();
        c.set_record(record(uniform(10))).unwrap();
        assert_eq!(
            c.decide(0.95, "fp", 100, 1.5).status,
            GuaranteeStatus::InsufficientData
        );
        let mut scores = uniform(70);
        scores.extend(std::iter::repeat_n(None, 30));
        c.set_record(record(scores)).unwrap();
        let d = c.decide(0.95, "fp", 100, 1.5);
        assert_eq!(d.status, GuaranteeStatus::Unachievable);
        assert!(d.max_achievable_coverage.unwrap() < 0.71);
    }

    #[test]
    fn feedback_measures_coverage_against_served_threshold() {
        let dir = TempDir::new().unwrap();
        let c = CorpusCalibration::open(dir.path()).unwrap();
        c.log_query(query("q1", Some(0.5), Decision::Answered))
            .unwrap();
        c.log_query(query("q2", Some(0.5), Decision::Answered))
            .unwrap();

        let hit = c
            .submit_feedback("q1", &["d1".into()], &[])
            .unwrap()
            .unwrap();
        assert_eq!(hit.covered, Some(true));
        let miss = c
            .submit_feedback("q2", &["d2".into()], &[])
            .unwrap()
            .unwrap();
        assert_eq!(miss.covered, Some(false));
        assert!(c.submit_feedback("nope", &[], &[]).unwrap().is_none());

        let obs = c.observed(0.9, 0.02).unwrap();
        assert_eq!(obs.observations, 2);
        assert!((obs.rate - 0.5).abs() < 1e-12);
        assert!(
            !obs.drift_alert,
            "two observations must not trigger an alert"
        );
        assert_eq!(c.feedback_scores(), vec![Some(0.2), Some(0.7)]);

        // Survives restart.
        let reopened = CorpusCalibration::open(dir.path()).unwrap();
        assert!(reopened.query("q1").is_some());
        assert_eq!(reopened.observed(0.9, 0.02).unwrap().observations, 2);
    }

    #[test]
    fn uncalibrated_queries_do_not_count_toward_coverage() {
        let dir = TempDir::new().unwrap();
        let c = CorpusCalibration::open(dir.path()).unwrap();
        c.log_query(query("q1", None, Decision::Answered)).unwrap();
        let out = c
            .submit_feedback("q1", &["d1".into()], &[])
            .unwrap()
            .unwrap();
        assert_eq!(out.covered, None);
        assert!(out.relevant_retrieved);
        assert!(c.observed(0.9, 0.02).is_none());
    }

    #[test]
    fn gaps_group_abstained_queries() {
        let dir = TempDir::new().unwrap();
        let c = CorpusCalibration::open(dir.path()).unwrap();
        c.log_query(query("a", Some(0.1), Decision::Abstained))
            .unwrap();
        c.log_query(query("b", Some(0.1), Decision::Abstained))
            .unwrap();
        c.log_query(query("c", Some(0.9), Decision::Answered))
            .unwrap();
        let gaps = c.gaps(10);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].count, 2);
        assert_eq!(gaps[0].query, "what is the lease term?");
        assert_eq!(gaps[0].best_nonconformity, Some(0.2));
    }

    #[test]
    fn split_half_validation_is_near_target_on_uniform_scores() {
        let scores: Vec<f64> = (0..400).map(|i| (i * 37 % 400) as f64 / 400.0).collect();
        let r = split_half_validation(&scores, 0.9, 200, 42);
        assert_eq!(r.splits, 200);
        assert!(r.mean_coverage > 0.88 && r.mean_coverage < 0.93, "{r:?}");
    }

    #[test]
    fn torn_log_line_is_skipped() {
        let dir = TempDir::new().unwrap();
        {
            let c = CorpusCalibration::open(dir.path()).unwrap();
            c.log_query(query("q1", Some(0.5), Decision::Answered))
                .unwrap();
        }
        let mut f = OpenOptions::new()
            .append(true)
            .open(dir.path().join("querylog.jsonl"))
            .unwrap();
        f.write_all(b"{\"query_id\": \"trunc").unwrap();
        let c = CorpusCalibration::open(dir.path()).unwrap();
        assert!(c.query("q1").is_some());
    }
}
