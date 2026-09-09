use std::path::PathBuf;
use dashmap::DashMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use sievrant_core::error::{CalibrationError, Result, SievrantError};
use sievrant_core::types::QueryTaxonomy;
use sievrant_math::conformal::{CalibrationEntry, CalibrationScore, CalibrationSet as MathCalSet, compute_threshold};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationObservation {
    pub query: String,
    pub relevant_chunk_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSet {
    pub set_id: String,
    pub corpus: String,
    pub observations: Vec<CalibrationObservation>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub set_id: String,
    pub observation_count: usize,
    pub threshold_90: f64,
    pub threshold_95: f64,
    pub threshold_99: f64,
    pub holdout_coverage_95: f64,
    pub is_valid: bool,
}

/// Manages per-tenant calibration sets, threshold computations, and onboarding flows.
pub struct CalibrationManager {
    calibration_dir: PathBuf,
    _default_set: CalibrationSet,
    /// Tenant-specific calibration sets
    tenant_sets: DashMap<String, CalibrationSet>,
    /// Cache of (set_id, coverage) -> threshold
    thresholds: DashMap<(String, OrderedFloat<f64>), f64>,
}

impl CalibrationManager {
    pub fn new(calibration_dir: impl Into<PathBuf>) -> Result<Self> {
        let calibration_dir: PathBuf = calibration_dir.into();
        std::fs::create_dir_all(&calibration_dir)
            .map_err(|e| SievrantError::Calibration(CalibrationError::Io(e.to_string())))?;

        // In a real implementation, the default set would be loaded from bundled resources
        let default_set = CalibrationSet {
            set_id: "default".into(),
            corpus: "generic".into(),
            observations: vec![],
            created_at: chrono::Utc::now(),
        };

        Ok(Self {
            calibration_dir,
            _default_set: default_set,
            tenant_sets: DashMap::new(),
            thresholds: DashMap::new(),
        })
    }

    /// Retrieve the precomputed threshold for a corpus at a specific coverage target.
    /// Falls back to the default generic threshold if no tenant calibration data exists.
    pub fn threshold(&self, corpus: &str, coverage: f64) -> f64 {
        let set_id = if self.tenant_sets.contains_key(corpus) {
            corpus
        } else {
            "default"
        };
        
        let key = (set_id.to_string(), OrderedFloat(coverage));
        if let Some(t) = self.thresholds.get(&key) {
            *t
        } else {
            // Default fallback if somehow not precomputed (e.g., during testing)
            1.5 // A very conservative default nonconformity score
        }
    }

    /// Onboard a new calibration set for a tenant corpus.
    /// Requires at least 100 labeled observations to guarantee statistical power.
    pub fn onboard_calibration(
        &self,
        corpus: &str,
        observations: Vec<CalibrationObservation>,
    ) -> Result<CalibrationReport> {
        // 1. Check minimum data
        if observations.len() < 100 {
            return Err(SievrantError::Calibration(CalibrationError::InsufficientData {
                min_required: 100,
                provided: observations.len(),
            }));
        }

        // 2. Validation split (80/20)
        let split_idx = (observations.len() as f64 * 0.8) as usize;
        let (train_obs, val_obs) = observations.split_at(split_idx);

        // 3. Compute nonconformity scores for train (mocked for scaffolding)
        // In reality, this dispatches queries through the retrieval pipeline
        let mut train_scores: Vec<f64> = train_obs
            .iter()
            .enumerate()
            .map(|(i, _)| (i as f64 / train_obs.len() as f64).abs())
            .collect();
        train_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Build a CalibrationSet from the raw sorted scores so compute_threshold
        // receives the type it expects (annotated as relevant so it uses all of them).
        let cal_set = MathCalSet {
            id: "onboarding".into(),
            entries: vec![CalibrationEntry {
                query: "calibration".into(),
                taxonomy: QueryTaxonomy::Factoid,
                relevant_chunk_ids: vec![],
                scores: train_scores.iter().map(|&s| CalibrationScore {
                    chunk_id: String::new(),
                    nonconformity_score: s,
                    is_relevant: true,
                }).collect(),
            }],
            created_at: chrono::Utc::now(),
        };

        let t90 = compute_threshold(&cal_set, &QueryTaxonomy::Factoid, 0.90)?;
        let t95 = compute_threshold(&cal_set, &QueryTaxonomy::Factoid, 0.95)?;
        let t99 = compute_threshold(&cal_set, &QueryTaxonomy::Factoid, 0.99)?;

        // 5. Verify on holdout set (mocked: assume scores are uniformly distributed)
        let val_scores: Vec<f64> = val_obs
            .iter()
            .enumerate()
            .map(|(i, _)| (i as f64 / val_obs.len() as f64).abs())
            .collect();
        
        let covered_count = val_scores.iter().filter(|&&s| s <= t95).count();
        let holdout_coverage = covered_count as f64 / val_scores.len() as f64;
        
        let is_valid = holdout_coverage >= 0.93; // Accept slight variance on holdout

        // 6. Store
        let set = CalibrationSet {
            set_id: corpus.to_string(),
            corpus: corpus.to_string(),
            observations,
            created_at: chrono::Utc::now(),
        };

        self.tenant_sets.insert(corpus.to_string(), set.clone());
        self.thresholds.insert((corpus.to_string(), OrderedFloat(0.90)), t90);
        self.thresholds.insert((corpus.to_string(), OrderedFloat(0.95)), t95);
        self.thresholds.insert((corpus.to_string(), OrderedFloat(0.99)), t99);
        
        self.persist_set(&set)?;

        // 7. Report
        Ok(CalibrationReport {
            set_id: corpus.to_string(),
            observation_count: set.observations.len(),
            threshold_90: t90,
            threshold_95: t95,
            threshold_99: t99,
            holdout_coverage_95: holdout_coverage,
            is_valid,
        })
    }
    
    fn persist_set(&self, set: &CalibrationSet) -> Result<()> {
        let path = self.calibration_dir.join(format!("{}.json", set.set_id));
        let data = serde_json::to_string_pretty(set)
            .map_err(|e| SievrantError::Calibration(CalibrationError::Parse(e.to_string())))?;
        std::fs::write(&path, data)
            .map_err(|e| SievrantError::Calibration(CalibrationError::Io(e.to_string())))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn validates_minimum_data_requirement() {
        let dir = TempDir::new().unwrap();
        let manager = CalibrationManager::new(dir.path()).unwrap();
        
        // Only 5 observations
        let obs = vec![CalibrationObservation { query: "q".into(), relevant_chunk_ids: vec![] }; 5];
        
        let result = manager.onboard_calibration("test_corp", obs);
        assert!(result.is_err());
        if let Err(SievrantError::Calibration(CalibrationError::InsufficientData { min_required, .. })) = result {
            assert_eq!(min_required, 100);
        } else {
            panic!("Expected Calibration error");
        }
    }

    #[test]
    fn onboarding_flow_computes_thresholds() {
        let dir = TempDir::new().unwrap();
        let manager = CalibrationManager::new(dir.path()).unwrap();
        
        let obs = vec![CalibrationObservation { query: "q".into(), relevant_chunk_ids: vec!["c1".into()] }; 150];
        
        let report = manager.onboard_calibration("tenant_a", obs).unwrap();
        assert!(report.is_valid);
        assert!(report.threshold_95 >= report.threshold_90);
        assert!(report.threshold_99 >= report.threshold_95);
        
        // Verify we can retrieve it
        let t = manager.threshold("tenant_a", 0.95);
        assert_eq!(t, report.threshold_95);
    }

    #[test]
    fn fallback_to_default_set() {
        let dir = TempDir::new().unwrap();
        let manager = CalibrationManager::new(dir.path()).unwrap();
        
        // tenant_b has not been onboarded, should fall back to 1.5 default
        let t = manager.threshold("tenant_b", 0.95);
        assert_eq!(t, 1.5);
    }
}
