//! Calibration workflows for a corpus.
//!
//! - [`Corpus::calibrate`]: labelled `(query, relevant documents)` pairs.
//! - [`Corpus::bootstrap_calibration`]: zero-label calibration. An LLM
//!   writes one realistic question per sampled passage, and that passage's
//!   document is the label. It removes the need for a labelling project
//!   before the first guarantee, at the cost of assuming real traffic
//!   resembles the generated questions; packets say so, and feedback-based
//!   recalibration replaces it once real labels exist.
//! - [`Corpus::recalibrate_from_feedback`]: use labels collected in production.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

use sievrant_calibration::{split_half_validation, CalibrationRecord, HoldoutResult};
use sievrant_core::error::{CalibrationError, Result, SievrantError};
use sievrant_core::types::{CalibrationSource, GuaranteeStatus, RetrievalConfig};
use sievrant_llm::{LlmRequest, Message, Role};
use sievrant_math::conformal::{conformal_threshold, nonconformity, query_score, Threshold};

use crate::Corpus;

const COVERAGE_LEVELS: [f64; 4] = [0.8, 0.9, 0.95, 0.99];
const MIN_ANSWERABLE: usize = 20;
const CONCURRENCY: usize = 8;

fn cal_err(msg: impl Into<String>) -> SievrantError {
    SievrantError::Calibration(CalibrationError::Parse(msg.into()))
}

/// One labelled calibration query. Leave both id lists empty for a query the
/// corpus cannot answer; those measure abstention and are excluded from the
/// guarantee.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationExample {
    pub query: String,
    #[serde(default)]
    pub relevant_doc_ids: Vec<String>,
    #[serde(default)]
    pub relevant_chunk_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThresholdReport {
    pub coverage: f64,
    #[serde(flatten)]
    pub threshold: Threshold,
}

/// Summary of a calibration run.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationReport {
    pub set_id: String,
    pub source: CalibrationSource,
    pub examples: usize,
    pub answerable: usize,
    pub unanswerable: usize,
    /// Fraction of answerable queries where retrieval surfaced any relevant
    /// passage among the scored candidates. Caps achievable coverage.
    pub retrieval_recall: f64,
    pub thresholds: Vec<ThresholdReport>,
    /// Split-half validation: held-out coverage should sit at or above target.
    pub holdout: Vec<HoldoutResult>,
    /// Of unanswerable queries, the fraction the 95% gate abstains on.
    pub unanswerable_abstention_at_95: Option<f64>,
    pub warnings: Vec<String>,
}

/// Current calibration state of a corpus.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationStatus {
    pub status: GuaranteeStatus,
    pub reason: Option<String>,
    pub set_id: Option<String>,
    pub source: Option<CalibrationSource>,
    pub size: Option<usize>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub retrieval_recall: Option<f64>,
    pub thresholds: Vec<ThresholdReport>,
    pub holdout: Vec<HoldoutResult>,
    pub pipeline_fingerprint: String,
    pub observed: Option<sievrant_core::types::ObservedCoverage>,
    pub logged_queries: usize,
    pub mean_conformal_set_size: Option<f64>,
    pub abstention_rate: Option<f64>,
    pub feedback_labels: usize,
}

impl Corpus {
    /// Score each example with the live pipeline and return
    /// `(query score, best candidate nonconformity)`.
    async fn score_examples(
        &self,
        examples: Vec<CalibrationExample>,
    ) -> Result<Vec<(CalibrationExample, Option<f64>, Option<f64>)>> {
        let orchestrator = self.orchestrator.clone();
        let config = RetrievalConfig {
            rrf_k: self.services.config.retrieval.rrf_k,
            skip_cache: true,
            ..RetrievalConfig::default()
        };
        let mut out = Vec::with_capacity(examples.len());
        let mut set = JoinSet::new();
        let mut iter = examples.into_iter().enumerate();
        let mut results: Vec<Option<(CalibrationExample, Option<f64>, Option<f64>)>> = Vec::new();

        loop {
            while set.len() < CONCURRENCY {
                let Some((i, ex)) = iter.next() else { break };
                results.push(None);
                let orch = orchestrator.clone();
                let cfg = config.clone();
                set.spawn(async move {
                    let scored = orch.score(&ex.query, &cfg, None).await?;
                    if scored.degraded_rerank {
                        return Err(cal_err(format!(
                            "reranker failed while scoring calibration query {:?}",
                            ex.query
                        )));
                    }
                    let relevant = scored.candidates.iter().filter(|c| {
                        ex.relevant_doc_ids.contains(&c.metadata.doc_id)
                            || ex.relevant_chunk_ids.contains(&c.chunk_id)
                    });
                    let s = query_score(relevant.map(|c| nonconformity(c.rerank_score)));
                    let best = scored
                        .candidates
                        .iter()
                        .map(|c| nonconformity(c.rerank_score))
                        .fold(None, |a: Option<f64>, s| Some(a.map_or(s, |a| a.min(s))));
                    Ok::<_, SievrantError>((i, ex, s.is_finite().then_some(s), best))
                });
            }
            match set.join_next().await {
                None => break,
                Some(joined) => {
                    let (i, ex, s, best) =
                        joined.map_err(|e| cal_err(format!("calibration task failed: {e}")))??;
                    results[i] = Some((ex, s, best));
                }
            }
        }
        for r in results.into_iter().flatten() {
            out.push(r);
        }
        Ok(out)
    }

    /// Calibrate from labelled examples.
    pub async fn calibrate(
        &self,
        examples: Vec<CalibrationExample>,
        source: CalibrationSource,
    ) -> Result<CalibrationReport> {
        let _guard = self.ingest_lock.lock().await;
        let mut warnings = Vec::new();

        let unknown: usize = examples
            .iter()
            .flat_map(|e| e.relevant_doc_ids.iter())
            .filter(|d| !self.store.has_doc(d))
            .count();
        if unknown > 0 {
            warnings.push(format!(
                "{unknown} relevant_doc_ids do not exist in this corpus; check for typos or missing ingestion"
            ));
        }

        let total = examples.len();
        let scored = self.score_examples(examples).await?;
        let (answerable, unanswerable): (Vec<_>, Vec<_>) =
            scored.into_iter().partition(|(ex, _, _)| {
                !ex.relevant_doc_ids.is_empty() || !ex.relevant_chunk_ids.is_empty()
            });

        if answerable.len() < MIN_ANSWERABLE {
            return Err(SievrantError::Calibration(
                CalibrationError::InsufficientData {
                    min_required: MIN_ANSWERABLE,
                    provided: answerable.len(),
                },
            ));
        }
        let min_recommended = self.services.config.calibration.min_calibration_examples;
        if answerable.len() < min_recommended {
            warnings.push(format!(
                "{} answerable examples; at least {min_recommended} are recommended for stable thresholds",
                answerable.len()
            ));
        }

        let scores: Vec<Option<f64>> = answerable.iter().map(|(_, s, _)| *s).collect();
        let finite: Vec<f64> = scores.iter().map(|s| s.unwrap_or(f64::INFINITY)).collect();
        let recall = scores.iter().filter(|s| s.is_some()).count() as f64 / scores.len() as f64;
        if recall < 0.95 {
            warnings.push(format!(
                "retrieval found a relevant passage for only {:.1}% of answerable queries; \
                 coverage above that is unachievable. Raise SIEVRANT_RERANK_TOP_N / \
                 SIEVRANT_SEARCH_LIMIT or improve embeddings.",
                recall * 100.0
            ));
        }

        let thresholds: Vec<ThresholdReport> = COVERAGE_LEVELS
            .iter()
            .map(|c| ThresholdReport {
                coverage: *c,
                threshold: conformal_threshold(&finite, *c),
            })
            .collect();
        let holdout: Vec<HoldoutResult> = [0.8, 0.9, 0.95]
            .iter()
            .filter(|c| finite.len() / 2 >= sievrant_math::conformal::min_examples_for(**c))
            .map(|c| split_half_validation(&finite, *c, 200, 0x5eed))
            .collect();

        let tau95 = conformal_threshold(&finite, 0.95).tau();
        let unanswerable_abstained = tau95.map(|tau| {
            unanswerable
                .iter()
                .filter(|(_, _, best)| best.is_none_or(|b| b > tau))
                .count()
        });

        let set_id = format!(
            "{}-{}",
            match source {
                CalibrationSource::Labeled => "cal",
                CalibrationSource::Synthetic => "syn",
                CalibrationSource::Feedback => "fb",
            },
            chrono::Utc::now().format("%Y%m%dT%H%M%S")
        );

        self.orchestrator
            .calibration
            .set_record(CalibrationRecord {
                set_id: set_id.clone(),
                source,
                created_at: chrono::Utc::now(),
                fingerprint: self.services.fingerprint(),
                corpus_chunk_count: self.store.chunk_count() as u64,
                scores,
                unanswerable_count: unanswerable.len(),
                unanswerable_abstained_at_95: unanswerable_abstained.unwrap_or(0),
                holdout: holdout.clone(),
            })?;

        Ok(CalibrationReport {
            set_id,
            source,
            examples: total,
            answerable: answerable.len(),
            unanswerable: unanswerable.len(),
            retrieval_recall: recall,
            thresholds,
            holdout,
            unanswerable_abstention_at_95: match (unanswerable_abstained, unanswerable.len()) {
                (Some(a), n) if n > 0 => Some(a as f64 / n as f64),
                _ => None,
            },
            warnings,
        })
    }

    /// Zero-label calibration from LLM-generated questions.
    pub async fn bootstrap_calibration(&self, n: usize, seed: u64) -> Result<CalibrationReport> {
        let llm = self.services.fast_llm.clone().ok_or_else(|| {
            cal_err("bootstrap calibration needs an LLM; set SIEVRANT_LLM_PROVIDER or SIEVRANT_FAST_LLM_PROVIDER")
        })?;
        let n = n.clamp(MIN_ANSWERABLE, 2000);
        let passages = self.store.sample(n, 25, seed);
        if passages.len() < MIN_ANSWERABLE {
            return Err(cal_err(format!(
                "corpus has only {} passages with enough text; ingest more documents first",
                passages.len()
            )));
        }

        let mut set = JoinSet::new();
        let mut iter = passages.into_iter();
        let mut examples = Vec::new();
        let mut failures = 0usize;
        loop {
            while set.len() < CONCURRENCY {
                let Some(p) = iter.next() else { break };
                let llm: Arc<dyn sievrant_llm::LlmProvider> = llm.clone();
                set.spawn(async move {
                    let clipped: String = p.text.chars().take(2500).collect();
                    let prompt = format!(
                        "Below is a passage from a document collection. Write one question a \
                         real user of this collection might ask that this passage answers. \
                         Use your own words rather than copying phrases, do not mention \"the \
                         passage\" or \"the document\", and reply with only the question.\n\n\
                         Passage:\n{clipped}"
                    );
                    let resp = llm
                        .complete(LlmRequest {
                            messages: vec![Message {
                                role: Role::User,
                                content: prompt,
                            }],
                            max_tokens: 1024,
                            temperature: 0.7,
                            system_prompt: None,
                        })
                        .await;
                    (p.metadata.doc_id, resp)
                });
            }
            match set.join_next().await {
                None => break,
                Some(Ok((doc_id, Ok(resp)))) => {
                    let q = resp.content.trim().trim_matches('"').trim().to_string();
                    if q.len() >= 8 && q.len() <= 500 {
                        examples.push(CalibrationExample {
                            query: q,
                            relevant_doc_ids: vec![doc_id],
                            relevant_chunk_ids: vec![],
                        });
                    } else {
                        failures += 1;
                    }
                }
                Some(_) => failures += 1,
            }
        }

        let mut report = self
            .calibrate(examples, CalibrationSource::Synthetic)
            .await?;
        if failures > 0 {
            report
                .warnings
                .push(format!("{failures} passages produced no usable question"));
        }
        report.warnings.push(
            "synthetic calibration assumes real queries resemble LLM-written questions; \
             collect feedback and recalibrate from it when possible"
                .into(),
        );
        Ok(report)
    }

    /// Replace the calibration set with scores from production feedback.
    pub async fn recalibrate_from_feedback(&self) -> Result<CalibrationReport> {
        let scores = self.orchestrator.calibration.feedback_scores();
        if scores.len() < MIN_ANSWERABLE {
            return Err(SievrantError::Calibration(
                CalibrationError::InsufficientData {
                    min_required: MIN_ANSWERABLE,
                    provided: scores.len(),
                },
            ));
        }
        let finite: Vec<f64> = scores.iter().map(|s| s.unwrap_or(f64::INFINITY)).collect();
        let recall = scores.iter().filter(|s| s.is_some()).count() as f64 / scores.len() as f64;
        let holdout: Vec<HoldoutResult> = [0.8, 0.9, 0.95]
            .iter()
            .filter(|c| finite.len() / 2 >= sievrant_math::conformal::min_examples_for(**c))
            .map(|c| split_half_validation(&finite, *c, 200, 0x5eed))
            .collect();
        let set_id = format!("fb-{}", chrono::Utc::now().format("%Y%m%dT%H%M%S"));
        self.orchestrator
            .calibration
            .set_record(CalibrationRecord {
                set_id: set_id.clone(),
                source: CalibrationSource::Feedback,
                created_at: chrono::Utc::now(),
                fingerprint: self.services.fingerprint(),
                corpus_chunk_count: self.store.chunk_count() as u64,
                scores: scores.clone(),
                unanswerable_count: 0,
                unanswerable_abstained_at_95: 0,
                holdout: holdout.clone(),
            })?;
        Ok(CalibrationReport {
            set_id,
            source: CalibrationSource::Feedback,
            examples: scores.len(),
            answerable: scores.len(),
            unanswerable: 0,
            retrieval_recall: recall,
            thresholds: COVERAGE_LEVELS
                .iter()
                .map(|c| ThresholdReport {
                    coverage: *c,
                    threshold: conformal_threshold(&finite, *c),
                })
                .collect(),
            holdout,
            unanswerable_abstention_at_95: None,
            warnings: vec![
                "feedback labels come from queries users chose to label; check they are representative"
                    .into(),
            ],
        })
    }

    /// Calibration state at a given coverage target.
    pub fn calibration_status(&self, coverage: f64) -> CalibrationStatus {
        let cal = &self.orchestrator.calibration;
        let fingerprint = self.services.fingerprint();
        let decision = cal.decide(
            coverage,
            &fingerprint,
            self.store.chunk_count() as u64,
            self.services
                .config
                .calibration
                .recalibration_corpus_growth_factor,
        );
        let record = cal.record();
        let (logged, mean_set, abstention) = cal.serving_stats();
        CalibrationStatus {
            status: decision.status,
            reason: decision.reason,
            set_id: decision.set_id,
            source: decision.source,
            size: decision.size,
            created_at: record.as_ref().map(|r| r.created_at),
            retrieval_recall: record.as_ref().map(|r| r.retrieval_recall()),
            thresholds: record
                .as_ref()
                .map(|r| {
                    let f = r.finite_scores();
                    COVERAGE_LEVELS
                        .iter()
                        .map(|c| ThresholdReport {
                            coverage: *c,
                            threshold: conformal_threshold(&f, *c),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            holdout: record.map(|r| r.holdout).unwrap_or_default(),
            pipeline_fingerprint: fingerprint,
            observed: cal.observed(coverage, self.services.config.calibration.drift_tolerance),
            logged_queries: logged,
            mean_conformal_set_size: mean_set,
            abstention_rate: abstention,
            feedback_labels: cal.feedback_scores().len(),
        }
    }
}
