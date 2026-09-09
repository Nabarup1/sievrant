use std::{collections::VecDeque, sync::RwLock};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sievrant_core::error::Result;

/// An observation of whether a relevant chunk was actually retrieved
/// under the conformal gate during live operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageObservation {
    pub timestamp: DateTime<Utc>,
    pub corpus: String,
    pub calibration_set_id: String,
    pub nominal_coverage: f64,
    /// Did the conformal gate include the relevant chunk?
    pub was_covered: bool,
}

/// A raised alert when empirical coverage drifts significantly below
/// the nominal target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAlert {
    pub corpus: String,
    pub nominal_coverage: f64,
    pub empirical_coverage: f64,
    /// empirical - nominal (usually negative when alerting)
    pub drift: f64,
    pub tolerance: f64,
    pub recommendation: String,
    pub triggered_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    pub drift_tolerance: f64,
    pub min_observations_for_alert: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            drift_tolerance: 0.05, // 5% absolute drift
            min_observations_for_alert: 50,
        }
    }
}

/// Live coverage monitoring that tracks whether the conformal guarantee
/// is actually holding in production.
pub struct CoverageMonitor {
    window: RwLock<VecDeque<CoverageObservation>>,
    window_days: u64,
    config: CalibrationConfig,
}

impl CoverageMonitor {
    pub fn new(window_days: u64, config: CalibrationConfig) -> Self {
        Self {
            window: RwLock::new(VecDeque::new()),
            window_days,
            config,
        }
    }

    /// Record a new coverage observation. Evicts old observations
    /// outside the rolling window.
    pub fn record(&self, obs: CoverageObservation) {
        let mut window = self.window.write().unwrap();
        window.push_back(obs);
        
        // Evict observations older than the window
        let cutoff = Utc::now() - Duration::days(self.window_days as i64);
        while let Some(front) = window.front() {
            if front.timestamp < cutoff {
                window.pop_front();
            } else {
                break;
            }
        }
    }

    /// Compute the empirical coverage rate over the current window for a specific corpus.
    pub fn get_empirical_coverage(&self, corpus: &str) -> Option<f64> {
        let window = self.window.read().unwrap();
        
        let mut count = 0;
        let mut covered = 0;

        for obs in window.iter() {
            if obs.corpus == corpus {
                count += 1;
                if obs.was_covered {
                    covered += 1;
                }
            }
        }

        if count == 0 {
            None
        } else {
            Some(covered as f64 / count as f64)
        }
    }

    /// Compare empirical vs nominal coverage. Fire an alert if drift exceeds tolerance.
    pub fn check_drift(&self, corpus: &str, nominal_coverage: f64) -> Option<DriftAlert> {
        let window = self.window.read().unwrap();
        
        let mut count = 0;
        let mut covered = 0;

        for obs in window.iter() {
            if obs.corpus == corpus {
                count += 1;
                if obs.was_covered {
                    covered += 1;
                }
            }
        }

        if count < self.config.min_observations_for_alert {
            return None; // Not enough statistical power yet
        }

        let empirical = covered as f64 / count as f64;
        let drift = empirical - nominal_coverage;

        // Alert if empirical coverage is lower than nominal by more than tolerance
        if drift < -self.config.drift_tolerance {
            Some(DriftAlert {
                corpus: corpus.to_string(),
                nominal_coverage,
                empirical_coverage: empirical,
                drift,
                tolerance: self.config.drift_tolerance,
                recommendation: "Recalibrate with fresh data: empirical coverage has dropped below the nominal target.".into(),
                triggered_at: Utc::now(),
            })
        } else {
            None
        }
    }

    /// Get a string report summarizing current coverage health for a corpus.
    pub fn get_report(&self, corpus: &str) -> Result<String> {
        let window = self.window.read().unwrap();
        let mut count = 0;
        let mut covered = 0;
        let mut sum_nominal = 0.0;

        for obs in window.iter() {
            if obs.corpus == corpus {
                count += 1;
                if obs.was_covered {
                    covered += 1;
                }
                sum_nominal += obs.nominal_coverage;
            }
        }

        if count == 0 {
            return Ok(format!("No coverage data available for corpus: {}", corpus));
        }

        let empirical = covered as f64 / count as f64;
        let avg_nominal = sum_nominal / count as f64;
        let diff = empirical - avg_nominal;

        let status = if diff < -self.config.drift_tolerance {
            "CRITICAL: Drift exceeds tolerance. Recalibration required."
        } else if diff < 0.0 {
            "WARNING: Slight under-coverage detected."
        } else {
            "HEALTHY: Empirical coverage exceeds or matches nominal target."
        };

        Ok(format!(
            "Coverage Report for '{}':\n\
             Total Observations (last {} days): {}\n\
             Average Nominal Target: {:.1}%\n\
             Empirical Coverage: {:.1}%\n\
             Drift: {:.1}%\n\
             Status: {}",
            corpus,
            self.window_days,
            count,
            avg_nominal * 100.0,
            empirical * 100.0,
            diff * 100.0,
            status
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_obs(corpus: &str, was_covered: bool) -> CoverageObservation {
        CoverageObservation {
            timestamp: Utc::now(),
            corpus: corpus.to_string(),
            calibration_set_id: "default".into(),
            nominal_coverage: 0.95,
            was_covered,
        }
    }

    #[test]
    fn computes_empirical_coverage_correctly() {
        let monitor = CoverageMonitor::new(30, CalibrationConfig::default());
        
        monitor.record(dummy_obs("test", true));
        monitor.record(dummy_obs("test", true));
        monitor.record(dummy_obs("test", false));
        monitor.record(dummy_obs("other", false)); // Should be ignored

        let empirical = monitor.get_empirical_coverage("test").unwrap();
        assert!((empirical - (2.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn drift_detection_fires_when_below_tolerance() {
        let config = CalibrationConfig {
            drift_tolerance: 0.05,
            min_observations_for_alert: 5,
        };
        let monitor = CoverageMonitor::new(30, config);

        // 80% empirical vs 95% nominal -> 15% drift > 5% tolerance -> alert
        for _ in 0..4 { monitor.record(dummy_obs("test", true)); }
        for _ in 0..1 { monitor.record(dummy_obs("test", false)); }

        let alert = monitor.check_drift("test", 0.95);
        assert!(alert.is_some());
        let a = alert.unwrap();
        assert_eq!(a.empirical_coverage, 0.80);
        assert!((a.drift - (-0.15)).abs() < 1e-6);
    }

    #[test]
    fn rolling_window_evicts_old_observations() {
        let monitor = CoverageMonitor::new(1, CalibrationConfig::default());
        
        let old_obs = CoverageObservation {
            timestamp: Utc::now() - Duration::days(2),
            corpus: "test".into(),
            calibration_set_id: "default".into(),
            nominal_coverage: 0.95,
            was_covered: true,
        };
        
        monitor.record(old_obs); // Should be immediately evicted
        assert!(monitor.get_empirical_coverage("test").is_none());

        monitor.record(dummy_obs("test", true));
        assert_eq!(monitor.get_empirical_coverage("test").unwrap(), 1.0);
    }
}
