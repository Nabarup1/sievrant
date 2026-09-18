use std::collections::VecDeque;
use std::sync::RwLock;

use crate::metering::UsageRecord;

/// Per-tenant token savings aggregator.
///
/// Provides the "Sievrant saved you $X this month" metric by accumulating
/// a rolling window of UsageRecords and computing cumulative savings.
pub struct SavingsMargin {
    /// Rolling buffer of recent records (capped to avoid unbounded memory).
    records: RwLock<VecDeque<UsageRecord>>,
    max_records: usize,
}

impl SavingsMargin {
    pub fn new(max_records: usize) -> Self {
        Self {
            records: RwLock::new(VecDeque::with_capacity(max_records)),
            max_records,
        }
    }

    /// Add a new usage record to the rolling window.
    pub fn record(&self, usage: UsageRecord) {
        let mut buf = self.records.write().unwrap();
        if buf.len() == self.max_records {
            buf.pop_front();
        }
        buf.push_back(usage);
    }

    /// Total tokens saved across all records in the window.
    pub fn total_tokens_saved(&self) -> usize {
        self.records
            .read()
            .unwrap()
            .iter()
            .map(|r| r.tokens_saved)
            .sum()
    }

    /// Estimated dollar savings based on a cost-per-1k-tokens rate.
    /// This is the headline number shown on the billing dashboard.
    pub fn estimated_usd_saved(&self, cost_per_1k_usd: f64) -> f64 {
        let tokens = self.total_tokens_saved();
        (tokens as f64 / 1000.0) * cost_per_1k_usd
    }

    /// Number of queries where Conformal Prediction produced a non-zero saving.
    pub fn effective_queries(&self) -> usize {
        self.records
            .read()
            .unwrap()
            .iter()
            .filter(|r| r.tokens_saved > 0)
            .count()
    }

    /// Total queries in the rolling window.
    pub fn total_queries(&self) -> usize {
        self.records.read().unwrap().len()
    }
}

/// Summary snapshot returned by the dashboard API.
#[derive(Debug, Clone)]
pub struct MarginReport {
    pub total_queries: usize,
    pub effective_queries: usize,
    pub total_tokens_saved: usize,
    pub estimated_usd_saved: f64,
}

impl SavingsMargin {
    pub fn report(&self, cost_per_1k_usd: f64) -> MarginReport {
        MarginReport {
            total_queries: self.total_queries(),
            effective_queries: self.effective_queries(),
            total_tokens_saved: self.total_tokens_saved(),
            estimated_usd_saved: self.estimated_usd_saved(cost_per_1k_usd),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(tokens_saved: usize) -> UsageRecord {
        let r = UsageRecord::new("t", "q", "m", 100, 200, 80, 3, 200, 0.03);
        UsageRecord { tokens_saved, ..r }
    }

    #[test]
    fn accumulates_savings() {
        let margin = SavingsMargin::new(1000);
        margin.record(make_record(500));
        margin.record(make_record(300));
        assert_eq!(margin.total_tokens_saved(), 800);
    }

    #[test]
    fn rolling_window_evicts_oldest() {
        let margin = SavingsMargin::new(2);
        margin.record(make_record(100));
        margin.record(make_record(200));
        margin.record(make_record(999)); // evicts 100
        assert_eq!(margin.total_tokens_saved(), 1199);
    }

    #[test]
    fn usd_estimate_is_proportional() {
        let margin = SavingsMargin::new(100);
        margin.record(make_record(1000));
        // 1000 tokens at $0.03/1k = $0.03
        let usd = margin.estimated_usd_saved(0.03);
        assert!((usd - 0.03).abs() < 1e-10);
    }
}
