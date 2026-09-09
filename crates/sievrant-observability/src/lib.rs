pub mod billing;
pub mod margins;
pub mod metering;
pub mod metrics;
pub mod spans;

use sievrant_core::error::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

pub struct ObservabilityConfig {
    pub enable_json_logs: bool,
    pub otlp_endpoint: Option<String>,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_json_logs: false,
            otlp_endpoint: None,
        }
    }
}

/// Initialize the observability stack: tracing subscriber and optional OTLP exporters.
pub fn init_observability(config: &ObservabilityConfig) -> Result<()> {
    let fmt_layer = if config.enable_json_logs {
        tracing_subscriber::fmt::layer()
            .json()
            .with_current_span(true)
            .boxed()
    } else {
        tracing_subscriber::fmt::layer().boxed()
    };

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    // In a full production implementation, we would also configure:
    // 1. opentelemetry_otlp::new_pipeline() for tracing and metrics export
    // 2. Add an OpenTelemetry tracing layer to the subscriber
    
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();

    Ok(())
}
