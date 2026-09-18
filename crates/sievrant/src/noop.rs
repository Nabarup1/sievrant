use async_trait::async_trait;
use sievrant_core::{
    error::Result,
    types::{EvidencePacket, QueryTaxonomy},
};
use sievrant_llm::{FinishReason, LlmProvider, LlmRequest, LlmResponse, TokenUsage};
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

pub struct NoopLlm;
#[async_trait]
impl LlmProvider for NoopLlm {
    async fn complete(&self, _req: LlmRequest) -> sievrant_core::error::Result<LlmResponse> {
        Ok(LlmResponse {
            content: "Mock generated answer (LLM disabled)".to_string(),
            usage: TokenUsage::default(),
            model: self.model_id().to_string(),
            finish_reason: FinishReason::Stop,
        })
    }

    fn supports_vision(&self) -> bool {
        false
    }

    fn model_id(&self) -> &str {
        "noop-llm"
    }

    fn context_window(&self) -> usize {
        128000
    }
}
