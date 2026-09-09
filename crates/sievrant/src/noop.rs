use async_trait::async_trait;
use sievrant_core::{
    error::Result,
    types::{EvidencePacket, QueryTaxonomy},
};
use sievrant_orchestrator::pipeline::{CalibrationManager, CoverageMonitor, SemanticCache};

pub struct NoopCache;
#[async_trait]
impl SemanticCache for NoopCache {
    async fn lookup(
        &self,
        _query: &str,
        _corpus: &str,
        _user_id: &Option<String>,
    ) -> Result<Option<EvidencePacket>> {
        Ok(None)
    }

    async fn store(
        &self,
        _query: &str,
        _corpus: &str,
        _user_id: &Option<String>,
        _packet: &EvidencePacket,
    ) -> Result<()> {
        Ok(())
    }
}

pub struct NoopCalibration;
impl CalibrationManager for NoopCalibration {
    fn threshold(&self, _corpus: &str, _taxonomy: &QueryTaxonomy) -> f64 {
        0.5 // Default threshold
    }
}

pub struct NoopCoverage;
impl CoverageMonitor for NoopCoverage {
    fn record(&self, _corpus: &str, _covered: bool) {}
    fn empirical_coverage_30d(&self, _corpus: &str) -> Option<f64> {
        None
    }
}
