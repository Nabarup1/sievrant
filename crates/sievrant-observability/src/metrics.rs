use opentelemetry::{
    metrics::{Counter, Histogram, MeterProvider, ObservableGauge},
    KeyValue,
};

/// Registry of all OpenTelemetry metrics for the Sievrant pipeline.
pub struct RetrievalMetrics {
    pub retrieval_latency: Histogram<f64>,
    pub request_totals: Counter<u64>,
    pub cache_hits: Counter<u64>,
    pub empirical_coverage: ObservableGauge<f64>,
    pub token_budget_utilization: Histogram<f64>,
}

impl RetrievalMetrics {
    pub fn new(meter_provider: &impl MeterProvider) -> Self {
        let meter = meter_provider.meter("sievrant");

        let retrieval_latency = meter
            .f64_histogram("sievrant_retrieval_latency_ms")
            .with_description("End-to-end latency of a retrieval request")
            .with_unit("ms")
            .build();

        let request_totals = meter
            .u64_counter("sievrant_request_totals")
            .with_description("Total number of retrieval requests")
            .build();

        let cache_hits = meter
            .u64_counter("sievrant_cache_hits")
            .with_description("Total number of semantic cache hits")
            .build();

        let empirical_coverage = meter
            .f64_observable_gauge("sievrant_empirical_coverage")
            .with_description("Rolling empirical coverage rate")
            .build();

        let token_budget_utilization = meter
            .f64_histogram("sievrant_token_budget_utilization")
            .with_description("Percentage of token budget consumed by selected evidence")
            .build();

        Self {
            retrieval_latency,
            request_totals,
            cache_hits,
            empirical_coverage,
            token_budget_utilization,
        }
    }

    /// Record a completed request
    pub fn record_request(&self, corpus: &str, latency_ms: f64, cache_hit: bool, budget_util: f64) {
        let labels = [KeyValue::new("corpus", corpus.to_string())];

        self.request_totals.add(1, &labels);
        self.retrieval_latency.record(latency_ms, &labels);
        self.token_budget_utilization.record(budget_util, &labels);

        if cache_hit {
            self.cache_hits.add(1, &labels);
        }
    }
}
